"""Save minimal runtime and source provenance without user environment secrets."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = [
        "torch",
        "torchvision",
        "gsplat",
        "transformers",
        "modelscope",
        "numpy",
        "scipy",
        "opencv-python-headless",
    ]
    files = [ROOT / "pyproject.toml", ROOT / "uv.lock", *sorted((ROOT / "src").rglob("*.py"))]
    result = {
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in names},
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    }
    output = ROOT / "artifacts/environment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(output)


if __name__ == "__main__":
    main()
