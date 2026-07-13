"""Download NV-Segment-CT from its official NVIDIA Hugging Face repository."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_REPOSITORY = "nvidia/NV-Segment-CT"
MODEL_LICENSE_URL = (
    "https://www.nvidia.com/en-us/agreements/enterprise-software/"
    "nvidia-open-model-license/"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="Confirm that you have read and accept NVIDIA's model licence",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parent / "NV-Segment-CT",
    )
    args = parser.parse_args()

    if not args.accept_license:
        parser.error(
            "Read the NVIDIA Open Model License first, then rerun with "
            f"--accept-license\n{MODEL_LICENSE_URL}"
        )

    destination = args.destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_REPOSITORY} to {destination}")
    snapshot_download(
        repo_id=MODEL_REPOSITORY,
        local_dir=str(destination),
        local_dir_use_symlinks=False,
    )
    print("NV-Segment-CT download complete.")


if __name__ == "__main__":
    main()

