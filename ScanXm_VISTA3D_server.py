"""Local ScanXm server for NVIDIA NV-Segment-CT (VISTA3D) only."""

from __future__ import annotations

import argparse
import atexit
import collections
import datetime as _datetime
import os
import re
import secrets
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, Tuple

from flask import Flask, Response, g, jsonify, request

import nv_segment_worker
from nv_segment_worker import (
    NV_SEGMENT_MODELS,
    is_nv_full_model,
    is_nv_interactive_model,
    run_nv_segment_full,
)


app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

# A fresh key is created for every process start and displayed only in the
# terminal. ScanXm includes it as the first component of every request path.
SERVER_KEY = secrets.token_urlsafe(32)

UPLOAD_DIR = Path(tempfile.gettempdir()) / "scanxm_vista3d_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
_UPLOAD_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")

_SESSION_LOCK = threading.RLock()
_CURRENT_SESSION_ID = None

_JOB_LOCK = threading.RLock()
_JOB = {
    "generation": 0,
    "status": "idle",  # idle | running | done | error
    "message": "",
    "payload": None,
    "headers": {},
}
_CANCEL_EVENT = threading.Event()

_LOG_LOCK = threading.RLock()
_LOG_BUFFER = collections.deque(maxlen=500)
_CLEANUP_LOCK = threading.RLock()
_SHUTTING_DOWN = False


def log_event(message: str) -> None:
    timestamp = _datetime.datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {message}"
    with _LOG_LOCK:
        _LOG_BUFFER.append(entry)
    print(entry, flush=True)


def _safe_upload_path(upload_id: str) -> Path:
    if not upload_id or not _UPLOAD_ID_PATTERN.fullmatch(upload_id):
        raise ValueError("Invalid upload_id")
    return UPLOAD_DIR / upload_id


def _clear_uploads() -> None:
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        for path in UPLOAD_DIR.iterdir():
            if path.is_file():
                try:
                    path.unlink()
                except OSError:
                    pass
    except OSError:
        pass


def _reset_job_state(reason: str = "") -> None:
    with _JOB_LOCK:
        _JOB["generation"] += 1
        _JOB.update(
            {
                "status": "idle",
                "message": reason,
                "payload": None,
                "headers": {},
            }
        )


def _force_wipe_state(reason: str) -> None:
    """Cancel publishable work and synchronously release model/VRAM state.

    CUDA kernels cannot safely be killed from another Python thread. The worker
    serializes model operations, so this call waits for any current kernel to
    finish before cleanup and before a new ScanXm session proceeds.
    """

    with _CLEANUP_LOCK:
        log_event(f"Cleaning server state: {reason}")
        _CANCEL_EVENT.set()
        _reset_job_state(reason)
        try:
            nv_segment_worker.stop_nv_all()
        except Exception as error:
            log_event(f"Cleanup warning: {error}")
        _clear_uploads()
        _CANCEL_EVENT.clear()
        log_event("VISTA3D state and upload buffers cleared")


def _shutdown_cleanup() -> None:
    global _SHUTTING_DOWN
    if _SHUTTING_DOWN:
        return
    _SHUTTING_DOWN = True
    try:
        _force_wipe_state("server shutdown")
    except Exception:
        pass


atexit.register(_shutdown_cleanup)


@app.before_request
def _authenticate_and_select_session():
    global _CURRENT_SESSION_ID

    g.request_started = time.monotonic()
    first_component = request.path.lstrip("/").split("/", 1)[0]
    if not first_component or not secrets.compare_digest(first_component, SERVER_KEY):
        return Response("Unauthorized", status=401, mimetype="text/plain")

    client_session = (request.headers.get("X-Session-ID") or "").strip()
    if not client_session:
        return None
    if len(client_session) > 256:
        return jsonify({"status": "error", "message": "X-Session-ID is too long"}), 400

    with _SESSION_LOCK:
        if _CURRENT_SESSION_ID is None:
            _CURRENT_SESSION_ID = client_session
            log_event("First ScanXm client session connected")
        elif client_session != _CURRENT_SESSION_ID:
            log_event("New ScanXm client session detected")
            _force_wipe_state("new ScanXm client session")
            _CURRENT_SESSION_ID = client_session
    return None


@app.after_request
def _finish_request(response):
    try:
        response.headers.setdefault("Cache-Control", "no-store")
        content_type = response.headers.get("Content-Type", "")
        if content_type.startswith("application/json") and "charset=" not in content_type:
            response.headers["Content-Type"] = "application/json; charset=utf-8"

        path = request.path
        prefix = f"/{SERVER_KEY}"
        if path.startswith(prefix):
            path = path[len(prefix) :] or "/"
        if path not in ("/info/", "/getresult", "/model-status", "/logs"):
            elapsed_ms = int((time.monotonic() - g.request_started) * 1000)
            log_event(f"{request.method} {path} -> {response.status_code} ({elapsed_ms} ms)")
    except Exception:
        pass
    return response


def _leaf(model: Dict) -> Dict:
    result = {"kind": "leaf", "label": model["name"]}
    result.update(model)
    return result


@app.route("/<key>/info/", methods=["GET"])
def info(key):
    installed, detail = nv_segment_worker.model_installation_status()
    return (
        jsonify(
            {
                "groups": [
                    {"group": "VISTA-3D", "items": [_leaf(m) for m in NV_SEGMENT_MODELS]}
                ],
                "services": {
                    "llm": {"enabled": False},
                    "stt": {"enabled": False},
                    "tts": {"enabled": False},
                },
                "model_installation": {"ready": installed, "detail": detail},
            }
        ),
        200,
    )


@app.route("/<key>/upload_chunk", methods=["POST"])
def upload_chunk(key):
    try:
        content_type = (request.content_type or "").lower()
        if "multipart/form-data" in content_type:
            upload_id = request.form.get("upload_id", "")
            chunk = request.files.get("chunk")
            if chunk is None:
                return jsonify({"status": "error", "message": "Missing chunk"}), 400
            data = chunk.read()
        else:
            upload_id = request.args.get("upload_id", "")
            data = request.get_data(cache=False)
        if not data:
            return jsonify({"status": "error", "message": "Empty chunk"}), 400

        path = _safe_upload_path(upload_id)
        with path.open("ab") as stream:
            stream.write(data)
        return jsonify({"status": "success"}), 200
    except ValueError as error:
        return jsonify({"status": "error", "message": str(error)}), 400


def _read_volume_request() -> Tuple[bytes, int, int, int, tuple, tuple]:
    upload_id = (request.form.get("upload_id") or "").strip()
    if upload_id:
        path = _safe_upload_path(upload_id)
        if not path.is_file():
            raise ValueError("Uploaded volume was not found")
        data = path.read_bytes()
        try:
            path.unlink()
        except OSError:
            pass
    else:
        volume_file = request.files.get("voxels")
        if volume_file is None:
            raise ValueError("Missing voxels or upload_id")
        data = volume_file.read()

    try:
        width = int(request.form.get("width", "0"))
        height = int(request.form.get("height", "0"))
        depth = int(request.form.get("depth", "0"))
    except ValueError as error:
        raise ValueError("width, height and depth must be integers") from error
    if width <= 0 or height <= 0 or depth <= 0:
        raise ValueError("width, height and depth must be positive")
    if (request.form.get("dtype", "float32") or "").lower() != "float32":
        raise ValueError("Only float32 volume uploads are supported")

    try:
        spacing = tuple(float(v) for v in request.form.get("spacing", "1,1,1").split(","))
        origin = tuple(float(v) for v in request.form.get("origin", "0,0,0").split(","))
    except ValueError as error:
        raise ValueError("spacing and origin must contain three numbers") from error
    if len(spacing) != 3 or len(origin) != 3:
        raise ValueError("spacing and origin must contain three numbers")

    expected = width * height * depth * 4
    if len(data) != expected:
        raise ValueError(f"Uploaded volume is {len(data)} bytes; expected {expected}")
    return data, width, height, depth, spacing, origin


def _start_full_job(data, width, height, depth, spacing, origin, model):
    with _JOB_LOCK:
        if _JOB["status"] == "running":
            return False
        _JOB["generation"] += 1
        generation = _JOB["generation"]
        _JOB.update(
            {"status": "running", "message": "Starting", "payload": None, "headers": {}}
        )
    _CANCEL_EVENT.clear()

    def update_status(message: str) -> None:
        if _CANCEL_EVENT.is_set():
            raise InterruptedError("VISTA3D job cancelled")
        with _JOB_LOCK:
            if generation != _JOB["generation"]:
                raise InterruptedError("VISTA3D job belongs to an old session")
            _JOB["message"] = message
        log_event(f"[Worker] {message}")

    def worker() -> None:
        try:
            payload, headers = run_nv_segment_full(
                data,
                width,
                height,
                depth,
                spacing,
                origin,
                model,
                progress_callback=update_status,
            )
            with _JOB_LOCK:
                if generation != _JOB["generation"] or _CANCEL_EVENT.is_set():
                    return
                _JOB.update(
                    {
                        "status": "done",
                        "message": "done",
                        "payload": payload,
                        "headers": dict(headers or {}),
                    }
                )
            log_event("VISTA3D full-volume job completed")
        except InterruptedError as error:
            log_event(str(error))
        except Exception as error:
            traceback.print_exc()
            with _JOB_LOCK:
                if generation == _JOB["generation"]:
                    _JOB.update(
                        {"status": "error", "message": str(error), "payload": None, "headers": {}}
                    )
            log_event(f"VISTA3D job failed: {error}")

    threading.Thread(target=worker, name="vista3d-full", daemon=True).start()
    return True


@app.route("/<key>/initmodel", methods=["POST"])
def initmodel(key):
    if request.is_json:
        model = ((request.get_json(silent=True) or {}).get("model") or "").strip()
    else:
        model = (request.values.get("model") or "").strip()
    if not nv_segment_worker.is_nv_model(model):
        return jsonify({"status": "error", "message": f"Unknown model: '{model}'"}), 400

    with _JOB_LOCK:
        if _JOB["status"] == "running":
            return jsonify({"status": "error", "message": "Server is busy"}), 409

    try:
        data, width, height, depth, spacing, origin = _read_volume_request()
    except ValueError as error:
        return jsonify({"status": "error", "message": str(error)}), 400

    if is_nv_interactive_model(model):
        try:
            capabilities = nv_segment_worker.init_nv_interactive(
                data,
                width,
                height,
                depth,
                spacing,
                origin,
                model,
                progress_callback=log_event,
            )
            return (
                jsonify(
                    {
                        "status": "success",
                        "message": "VISTA3D interactive session initialized",
                        "capabilities": capabilities,
                    }
                ),
                200,
            )
        except Exception as error:
            traceback.print_exc()
            return jsonify({"status": "error", "message": str(error)}), 500

    if not is_nv_full_model(model):
        return jsonify({"status": "error", "message": "Unsupported model mode"}), 400
    if not _start_full_job(data, width, height, depth, spacing, origin, model):
        return jsonify({"status": "error", "message": "Server is busy"}), 409
    return jsonify({"status": "accepted", "message": "started"}), 202


@app.route("/<key>/getresult", methods=["GET"])
def getresult(key):
    with _JOB_LOCK:
        status = _JOB["status"]
        message = _JOB["message"]
        payload = _JOB["payload"]
        headers = dict(_JOB["headers"])
    if status == "idle":
        return Response(status=204)
    if status == "running":
        response = Response(status=204)
        response.headers["X-Progress-Phase"] = message
        return response
    if status == "error":
        return jsonify({"status": "error", "message": message}), 500
    if payload is None:
        return jsonify({"status": "error", "message": "Result is missing"}), 500

    response = Response(payload, mimetype="application/octet-stream")
    for name, value in headers.items():
        if name.lower() not in ("content-length", "transfer-encoding"):
            response.headers[name] = value
    return response


@app.route("/<key>/model-status", methods=["GET"])
def model_status(key):
    with _JOB_LOCK:
        return jsonify(
            {
                "status": _JOB["status"],
                "model": "CT_Full" if _JOB["status"] != "idle" else None,
                "message": _JOB["message"],
                "error": _JOB["message"] if _JOB["status"] == "error" else None,
                "capabilities": None,
            }
        )


@app.route("/<key>/nv_infer", methods=["POST"])
def nv_infer(key):
    try:
        label = request.form.get("label")
        label_prompt = int(label) if label not in (None, "") else None
        payload, headers = nv_segment_worker.infer_nv_interactive(
            request.form.get("input_points", ""),
            request.form.get("input_labels", ""),
            label_prompt,
        )
        response = Response(payload, mimetype="application/octet-stream")
        for name, value in headers.items():
            response.headers[name] = value
        return response
    except Exception as error:
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(error)}), 500


@app.route("/<key>/nv/reset", methods=["POST"])
def nv_reset(key):
    try:
        return jsonify(
            {"status": "success", "message": nv_segment_worker.reset_nv_interactive()}
        )
    except Exception as error:
        return jsonify({"status": "error", "message": str(error)}), 500


@app.route("/<key>/reset", methods=["POST"])
def reset(key):
    return nv_reset(key)


@app.route("/<key>/stop", methods=["POST"])
def stop(key):
    _force_wipe_state("stop requested by ScanXm")
    return jsonify({"status": "success", "message": "VISTA3D state and VRAM cleared"})


@app.route("/<key>/logs", methods=["GET"])
def logs(key):
    with _LOG_LOCK:
        entries = list(_LOG_BUFFER)
    since = request.args.get("since", type=int)
    if since is not None and 0 <= since < len(entries):
        selected = entries[since:]
        start = since
    else:
        selected = entries
        start = 0
    return jsonify(
        {"lines": selected, "next_index": start + len(selected), "total": len(entries)}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    installed, detail = nv_segment_worker.model_installation_status()
    print("\nScanXm VISTA3D local server")
    print("========================================")
    print(f"Model files: {'ready' if installed else 'NOT FOUND'}")
    print(f"Model path:  {detail}")
    print(f"Server key:  {SERVER_KEY}")
    print(f"ScanXm URL:  http://{args.host}:{args.port}/{SERVER_KEY}")
    print("========================================")
    print("Keep this terminal open. Press Ctrl+C to stop.\n", flush=True)

    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        _shutdown_cleanup()


if __name__ == "__main__":
    main()
