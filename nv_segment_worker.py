"""In-memory NV-Segment-CT (VISTA3D) inference for ScanXm.

Only the commercially usable NV-Segment-CT model is supported. Model files are
loaded from an adjacent ``NV-Segment-CT`` directory and are never bundled with
this repository.
"""

from __future__ import annotations

import gc
import os
import sys
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

try:
    from transformers.utils import logging as _hf_logging

    _hf_logging.set_verbosity_error()
except Exception:
    pass


def _model_metadata(name: str, *, point: bool = False) -> Dict:
    return {
        "name": name,
        "input_mode": "volume",
        "supports": {"box": False, "point": point, "mask": False},
        "propagation": {"slice_to_stack": False},
        "datatype": {"dtype": "float32", "channels": 1, "layout": "CHW"},
    }


NV_SEGMENT_MODELS = [
    _model_metadata("CT_Full"),
    _model_metadata("CT_Interactive", point=True),
]
NV_INTERACTIVE_NAME = "CT_Interactive"
_FULL_MODEL_NAMES = {"CT_Full"}
_MODEL_NAMES = {item["name"] for item in NV_SEGMENT_MODELS}

_HERE = Path(__file__).resolve().parent
MODEL_DIR = (_HERE / "NV-Segment-CT").resolve()

ROI_SIZE = (128, 128, 128)
OVERLAP = 0.3
MIN_WEIGHT_MATCH_FRACTION = 0.85
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# All model operations are serialized. In particular, session cleanup waits for
# in-flight inference before another ScanXm instance may load a model.
_RUNTIME_LOCK = threading.RLock()
_INTERACTIVE = {
    "pipeline": None,
    "pipeline_module": None,
    "image": None,
    "model_name": None,
    "dimensions": None,
}


def is_nv_model(name: str) -> bool:
    return name in _MODEL_NAMES


def is_nv_full_model(name: str) -> bool:
    return name in _FULL_MODEL_NAMES


def is_nv_interactive_model(name: str) -> bool:
    return name == NV_INTERACTIVE_NAME


def model_installation_status() -> Tuple[bool, str]:
    required = [
        MODEL_DIR / "hugging_face_pipeline.py",
        MODEL_DIR / "vista3d_pipeline.py",
        MODEL_DIR / "vista3d_pretrained_model",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return False, "Missing NV-Segment-CT files: " + ", ".join(missing)
    return True, str(MODEL_DIR)


def _choose_weight_file(pretrained_dir: Path) -> Path:
    candidates = [
        pretrained_dir / "model.pt",
        pretrained_dir / "pytorch_model.bin",
        pretrained_dir / "model.safetensors",
        pretrained_dir / "model_monai1.3.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No usable weight file found in {pretrained_dir}")


def _load_state_dict(weight_file: Path):
    if weight_file.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(weight_file), device="cpu")

    try:
        checkpoint = torch.load(str(weight_file), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(weight_file), map_location="cpu")
    except Exception:
        checkpoint = torch.load(str(weight_file), map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "network"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _strip_weight_prefixes(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "model.", "network."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def _force_load_network_weights(pipeline, pretrained_dir: Path, device, log) -> None:
    weight_file = _choose_weight_file(pretrained_dir)
    log(f"[VISTA3D] Loading weights from {weight_file}")
    source_state = _load_state_dict(weight_file)
    if not isinstance(source_state, dict):
        raise RuntimeError(f"Unexpected checkpoint type: {type(source_state)}")
    if not hasattr(pipeline, "model") or not hasattr(pipeline.model, "network"):
        raise RuntimeError("VISTA3D pipeline does not expose model.network")

    network = pipeline.model.network
    target_state = network.state_dict()
    compatible = {}
    for key, value in source_state.items():
        clean_key = _strip_weight_prefixes(key)
        if clean_key in target_state and hasattr(value, "shape"):
            if tuple(value.shape) == tuple(target_state[clean_key].shape):
                compatible[clean_key] = value

    fraction = len(compatible) / max(1, len(target_state))
    log(
        f"[VISTA3D] Matched {len(compatible)} / {len(target_state)} "
        f"weights ({fraction:.1%})"
    )
    if fraction < MIN_WEIGHT_MATCH_FRACTION:
        raise RuntimeError(
            f"Only {fraction:.1%} of model weights matched; refusing unsafe inference"
        )

    network.load_state_dict(compatible, strict=False)
    pipeline.model.to(device)
    pipeline.model.eval()


def _purge_model_modules() -> None:
    model_dir = os.path.normcase(str(MODEL_DIR))
    for name, module in list(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        normalized = os.path.normcase(os.path.abspath(filename))
        if normalized.startswith(model_dir + os.sep):
            del sys.modules[name]
    while str(MODEL_DIR) in sys.path:
        sys.path.remove(str(MODEL_DIR))


def _build_pipeline(log=print):
    installed, detail = model_installation_status()
    if not installed:
        raise FileNotFoundError(
            detail + ". Run: python download_model.py --accept-license"
        )

    pretrained_dir = MODEL_DIR / "vista3d_pretrained_model"
    _purge_model_modules()
    sys.path.insert(0, str(MODEL_DIR))

    from hugging_face_pipeline import HuggingFacePipelineHelper
    import vista3d_pipeline as pipeline_module

    device = torch.device(DEVICE)
    log(f"[VISTA3D] Initializing {MODEL_DIR} on {device}")
    helper = HuggingFacePipelineHelper("vista3d")
    pipeline = helper.init_pipeline(str(pretrained_dir), device=device)
    _force_load_network_weights(pipeline, pretrained_dir, device, log)

    # ScanXm passes an in-memory volume; disable all model file input/output.
    pipeline._preprocess_params["load_image"] = False
    pipeline.preprocessing_transforms = pipeline._init_preprocessing_transforms(
        load_image=False
    )
    pipeline._postprocess_params["save_output"] = False
    pipeline.postprocessing_transforms = pipeline._init_postprocessing_transforms(
        save_output=False
    )
    if hasattr(pipeline, "_forward_params"):
        pipeline._forward_params.clear()
    if hasattr(pipeline, "inferer"):
        if hasattr(pipeline.inferer, "roi_size"):
            pipeline.inferer.roi_size = ROI_SIZE
        if hasattr(pipeline.inferer, "overlap"):
            pipeline.inferer.overlap = OVERLAP
    return pipeline, pipeline_module


def _volume_bytes_to_xyz(data: bytes, width: int, height: int, depth: int):
    expected = width * height * depth * np.dtype(np.float32).itemsize
    if len(data) != expected:
        raise ValueError(f"Volume size is {len(data)} bytes; expected {expected}")
    volume = np.frombuffer(data, dtype=np.float32).reshape((depth, height, width))
    return np.ascontiguousarray(volume.transpose(2, 1, 0)[::-1, ::-1, :])


def _make_affine(spacing, origin):
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0], affine[1, 1], affine[2, 2] = spacing
    affine[0, 3], affine[1, 3], affine[2, 3] = origin
    return affine


def _build_metatensor(volume_xyz: np.ndarray, affine: np.ndarray):
    from monai.data import MetaTensor

    array = volume_xyz[np.newaxis, ...]
    affine_tensor = torch.as_tensor(affine, dtype=torch.float64)
    image = MetaTensor(torch.from_numpy(np.ascontiguousarray(array)))
    image.affine = affine_tensor
    image.meta["original_affine"] = affine_tensor
    image.meta["original_channel_dim"] = 0
    image.meta["spatial_shape"] = np.asarray(volume_xyz.shape, dtype=np.int64)
    image.meta["space"] = "RAS"
    return image


def _parse_points(points_text: str, labels_text: str, width, height, depth):
    if not points_text:
        return None, None

    point_tokens = [token for token in points_text.split(";") if token.strip()]
    label_tokens = [token for token in labels_text.split(";") if token.strip()]
    points: List[List[int]] = []
    labels: List[int] = []

    for index, token in enumerate(point_tokens):
        values = [value.strip() for value in token.strip("() ").split(",")]
        if len(values) != 3:
            raise ValueError(f"Point '{token}' must have x,y,z coordinates")
        x, y, z = (int(round(float(value))) for value in values)
        if not (0 <= x < width and 0 <= y < height and 0 <= z < depth):
            raise ValueError(f"Point {(x, y, z)} lies outside the uploaded volume")
        # ScanXm's raw volume is flipped on X/Y when converted to model space.
        points.append([width - 1 - x, height - 1 - y, z])
        label = int(float(label_tokens[index])) if index < len(label_tokens) else 1
        if label not in (-1, 0, 1, 2, 3):
            raise ValueError(f"Unsupported point label {label}")
        labels.append(label)
    return (points, labels) if points else (None, None)


def _extract_prediction(result):
    item = result[0] if isinstance(result, (list, tuple)) else result
    if isinstance(item, dict):
        if "pred" not in item:
            raise RuntimeError(f"VISTA3D output has no pred field: {list(item)}")
        return item["pred"]
    return item


def _prediction_to_bytes(prediction, width, height, depth) -> bytes:
    if torch.is_tensor(prediction):
        while prediction.ndim > 3:
            prediction = prediction.squeeze(0)
        array = prediction.detach().cpu().numpy()
    else:
        array = np.asarray(prediction)
        while array.ndim > 3:
            array = array[0]

    array = np.rint(np.nan_to_num(array, nan=0.0)).astype(np.int32)
    array[array == 255] = 0
    array[array < 0] = 0
    if array.shape != (width, height, depth):
        raise RuntimeError(
            f"VISTA3D output shape is {array.shape}; expected {(width, height, depth)}"
        )
    dhw = array[::-1, ::-1, :].transpose(2, 1, 0)
    dhw = np.clip(dhw, 0, np.iinfo(np.uint16).max).astype(np.uint16, copy=False)
    return np.ascontiguousarray(dhw).tobytes(order="C")


def _headers(width, height, depth, model_name, extra=None):
    headers = {
        "X-DType": "uint16",
        "X-Endian": "little",
        "X-Order": "DHW",
        "X-Width": str(width),
        "X-Height": str(height),
        "X-Depth": str(depth),
        "X-Model": model_name,
    }
    if extra:
        headers.update(extra)
    return headers


def _run_pipeline(
    pipeline,
    pipeline_module,
    image,
    *,
    points=None,
    point_labels=None,
    label_prompt=None,
    point_window=False,
):
    if hasattr(pipeline, "inferer") and hasattr(pipeline.inferer, "use_point_window"):
        pipeline.inferer.use_point_window = bool(point_window)

    inputs = {"image": image}
    kwargs = {"amp": torch.cuda.is_available(), "save_output": False}
    if points is not None:
        inputs["points"] = points
        inputs["point_labels"] = point_labels
        if label_prompt is not None:
            inputs["label_prompt"] = label_prompt
        kwargs["hyper_kwargs"] = {"user_prompt": 1}
    else:
        kwargs["hyper_kwargs"] = {
            "user_prompt": 1,
            "everything_labels": pipeline_module.VISTA3DPipeline.EVERYTHING_LABEL,
        }
    return _extract_prediction(pipeline(inputs, **kwargs))


def _stop_interactive_locked() -> None:
    pipeline = _INTERACTIVE.get("pipeline")
    try:
        if pipeline is not None and hasattr(pipeline, "model"):
            pipeline.model.to("cpu")
    except Exception:
        pass
    _INTERACTIVE.update(
        {
            "pipeline": None,
            "pipeline_module": None,
            "image": None,
            "model_name": None,
            "dimensions": None,
        }
    )


def run_nv_segment_full(
    voxels_f32_dhw: bytes,
    width: int,
    height: int,
    depth: int,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    model_name: str = "CT_Full",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[bytes, Dict[str, str]]:
    if not is_nv_full_model(model_name):
        raise ValueError(f"Unsupported full-volume VISTA3D model: {model_name}")
    log = progress_callback or (lambda *_: None)

    with _RUNTIME_LOCK:
        _stop_interactive_locked()
        _purge_model_modules()
        pipeline = None
        volume = None
        image = None
        prediction = None
        try:
            log("Converting volume")
            volume = _volume_bytes_to_xyz(voxels_f32_dhw, width, height, depth)
            image = _build_metatensor(volume, _make_affine(spacing, origin))
            log("Initializing VISTA3D")
            pipeline, pipeline_module = _build_pipeline(log=log)
            log("Running VISTA3D inference")
            prediction = _run_pipeline(pipeline, pipeline_module, image)
            log("Postprocessing")
            payload = _prediction_to_bytes(prediction, width, height, depth)
            return payload, _headers(width, height, depth, model_name)
        finally:
            try:
                if pipeline is not None and hasattr(pipeline, "model"):
                    pipeline.model.to("cpu")
            except Exception:
                pass
            pipeline = None
            prediction = None
            image = None
            volume = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _purge_model_modules()


def init_nv_interactive(
    voxels_f32_dhw: bytes,
    width: int,
    height: int,
    depth: int,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    model_name: str = NV_INTERACTIVE_NAME,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict:
    if not is_nv_interactive_model(model_name):
        raise ValueError(f"Unsupported interactive VISTA3D model: {model_name}")
    log = progress_callback or print

    with _RUNTIME_LOCK:
        _stop_interactive_locked()
        _purge_model_modules()
        volume = _volume_bytes_to_xyz(voxels_f32_dhw, width, height, depth)
        image = _build_metatensor(volume, _make_affine(spacing, origin))
        pipeline, pipeline_module = _build_pipeline(log=log)
        _INTERACTIVE.update(
            {
                "pipeline": pipeline,
                "pipeline_module": pipeline_module,
                "image": image,
                "model_name": model_name,
                "dimensions": (width, height, depth),
            }
        )
        log(f"[VISTA3D] Interactive session ready ({width}x{height}x{depth})")

    return {
        "region": False,
        "point": True,
        "mask": False,
        "apply_4d": False,
        "reset": True,
        "interactive_3d": True,
    }


def infer_nv_interactive(
    points_text: str,
    labels_text: str,
    label_prompt: Optional[int] = None,
) -> Tuple[bytes, Dict[str, str]]:
    with _RUNTIME_LOCK:
        pipeline = _INTERACTIVE["pipeline"]
        pipeline_module = _INTERACTIVE["pipeline_module"]
        image = _INTERACTIVE["image"]
        model_name = _INTERACTIVE["model_name"]
        dimensions = _INTERACTIVE["dimensions"]
        if pipeline is None or image is None or dimensions is None:
            raise RuntimeError("VISTA3D interactive session is not initialized")

        width, height, depth = dimensions
        points, labels = _parse_points(
            points_text, labels_text, width, height, depth
        )
        if points is None:
            empty = np.zeros((depth, height, width), dtype=np.uint16)
            return empty.tobytes(), _headers(
                width, height, depth, model_name, {"X-Points": "0"}
            )

        prompt = [int(label_prompt)] if label_prompt is not None else None
        prediction = _run_pipeline(
            pipeline,
            pipeline_module,
            image,
            points=points,
            point_labels=labels,
            label_prompt=prompt,
            point_window=True,
        )
        payload = _prediction_to_bytes(prediction, width, height, depth)
        extra = {"X-Points": str(len(points))}
        if prompt is not None:
            extra["X-LabelPrompt"] = str(prompt[0])
        return payload, _headers(width, height, depth, model_name, extra)


def reset_nv_interactive() -> str:
    with _RUNTIME_LOCK:
        if _INTERACTIVE["pipeline"] is None:
            return "interactive session not initialized"
        # Points are supplied afresh on every request; no point history is held.
        return "interactive point state reset"


def stop_nv_all() -> None:
    with _RUNTIME_LOCK:
        _stop_interactive_locked()
        _purge_model_modules()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
