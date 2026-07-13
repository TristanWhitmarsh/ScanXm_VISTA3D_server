# ScanXm VISTA3D Server

A small local Python server connecting [ScanXm](https://scanxm.com/) to
NVIDIA's `NV-Segment-CT` (VISTA3D) model for automatic and interactive 3D CT
segmentation.

It includes:

- `CT_Full` performs automatic multi-structure CT segmentation.
- `CT_Interactive` performs point-guided 3D CT segmentation.
- `NV-Segment-CTMR` is not supported because its model weights have
  non-commercial restrictions.
- Uploaded CT data is processed in memory. Chunk files are temporary and are
  deleted after loading, on client changes, and when ScanXm requests cleanup.

## Important notices

This software is not a medical device and is not intended to provide a medical
diagnosis. Validate model output for your use case and comply with applicable
clinical, privacy, data-protection and regulatory requirements.

The model weights are not part of this repository. NVIDIA distributes them
under the NVIDIA Open Model License Agreement. Read [Licensing](#licensing)
before downloading the model.

## Requirements

- Windows 10/11 or a modern 64-bit Linux distribution
- Miniconda or Anaconda
- Python 3.10
- Sufficient system RAM for 3D medical-image inference
- An NVIDIA GPU with sufficient VRAM is recommended, but not required
- An NVIDIA driver compatible with CUDA 12.4 when using GPU acceleration
- Internet access during installation and model download

## Installation

Clone the repository and enter it:

```text
git clone https://github.com/TristanWhitmarsh/ScanXm_VISTA3D_server.git
cd ScanXm_VISTA3D_server
```

Create a dedicated Conda environment:

```text
conda create -y -n scanxm python=3.10.19
conda activate scanxm
python -m pip install --upgrade pip setuptools wheel
```

The shared environment is called `scanxm` so other ScanXm AI server
repositories can be installed into the same working environment.

Choose one PyTorch installation.

For an NVIDIA GPU, install the CUDA 12.4 build:

```text
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
```

For CPU-only operation:

```text
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
```

Install the remaining pinned dependencies:

```text
python -m pip install -r requirements.txt
```

Confirm which compute backend PyTorch will use:

```text
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

`CUDA: True` means VISTA3D will use the NVIDIA GPU. `CUDA: False` is valid and
the server will use the CPU, although 3D model initialization and inference can
be substantially slower. For GPU operation, installing the full CUDA Toolkit
is normally unnecessary because the PyTorch wheel supplies its CUDA runtime;
an appropriate NVIDIA driver is still required.

## Download NV-Segment-CT

First read the
[NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
If you accept it, run:

```text
python download_model.py --accept-license
```

This downloads the official
[`nvidia/NV-Segment-CT`](https://huggingface.co/nvidia/NV-Segment-CT)
repository into `NV-Segment-CT/`. That directory is ignored by Git because it
contains large, separately licensed files.

The expected layout is:

```text
ScanXm_VISTA3D_server/
|-- ScanXm_VISTA3D_server.py
|-- nv_segment_worker.py
|-- download_model.py
|-- requirements.txt
`-- NV-Segment-CT/
    |-- hugging_face_pipeline.py
    |-- vista3d_pipeline.py
    `-- vista3d_pretrained_model/
```

`NV-Segment-CT` must remain directly inside the repository folder, beside
`ScanXm_VISTA3D_server.py`. The download command above places it there
automatically.

## Start the server

Activate the environment and run:

```text
conda activate scanxm
python ScanXm_VISTA3D_server.py
```

Every start generates a new cryptographically random server key. The terminal
prints output similar to:

```text
ScanXm VISTA3D local server
========================================
Model files: ready
Model path:  C:\...\ScanXm_VISTA3D_server\NV-Segment-CT
Server key:  EXAMPLE_RANDOM_KEY
ScanXm URL:  http://127.0.0.1:8000/EXAMPLE_RANDOM_KEY
========================================
```

Copy the complete `ScanXm URL` into ScanXm. Keep the terminal open while using
the model. A new key is generated after every server restart, so the URL in
ScanXm must then be updated.

The default address is deliberately local-only (`127.0.0.1`). An advanced user
can listen on another interface:

```text
python ScanXm_VISTA3D_server.py --host 0.0.0.0 --port 8000
```

Doing so exposes the server to the network. The URL key is a lightweight access
control, not a substitute for TLS, a firewall or a properly authenticated
reverse proxy. Do not expose this development server directly to the internet.

## Session cleanup

ScanXm sends an `X-Session-ID` with requests. When this server sees a different
session ID, it:

1. Cancels publication of results belonging to the previous client.
2. Waits for an in-progress model inference operation to finish safely.
3. Releases the VISTA3D model and interactive volume from memory/VRAM.
4. Deletes incomplete temporary uploads.
5. Allows the new ScanXm session to continue.

The `/stop` endpoint performs the same model and upload cleanup when requested
by ScanXm.

## Licensing

The original integration code in this repository is licensed under the
[Apache License 2.0](LICENSE).

The separately downloaded NVIDIA repository contains code under Apache License
2.0 and model weights under the
[NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for details.

This independent project is not affiliated with or endorsed by NVIDIA.
