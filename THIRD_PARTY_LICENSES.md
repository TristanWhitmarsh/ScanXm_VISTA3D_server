# Third-party software and model licence

This repository contains integration code licensed under the Apache License 2.0.
It does **not** contain NVIDIA model weights.

## NVIDIA NV-Segment-CT

The server is designed to load NVIDIA's `NV-Segment-CT` (VISTA3D) model. The
model is downloaded separately from its official repository:

- Model repository: <https://huggingface.co/nvidia/NV-Segment-CT>
- Model licence: [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)

The downloaded model repository includes source files under Apache License 2.0
and weights under the NVIDIA Open Model License Agreement. Those terms apply
independently of this repository's Apache 2.0 licence. Users are responsible for
reviewing and complying with the current NVIDIA terms before downloading or
using the model.

`NV-Segment-CTMR` is intentionally unsupported and is not downloaded by this
project because its weights are released under non-commercial terms.

This project is independent and is not endorsed by or affiliated with NVIDIA.
NVIDIA, VISTA3D and NV-Segment may be trademarks of their respective owners.

