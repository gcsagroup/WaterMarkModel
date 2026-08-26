from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import load_obb_checkpoint
else:
    from .model import load_obb_checkpoint


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "config.yaml"
SOURCE_WEIGHT = ROOT / "weight" / "best.pt"
OUTPUT_WEIGHT = ROOT / "weight" / "best_standalone.pt"
OUTPUT_METADATA = ROOT / "weight" / "best_standalone.json"


class InferenceOnly(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.model(image)
        return output[0] if isinstance(output, (tuple, list)) else output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_checkpoint(
    source: str | Path = SOURCE_WEIGHT,
    destination: str | Path = OUTPUT_WEIGHT,
    config_path: str | Path = CONFIG_PATH,
) -> Path:
    """Convert an Ultralytics training checkpoint into an inference-only TorchScript .pt."""

    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    input_size = config["model"]["input_size"]
    height, width = (input_size, input_size) if isinstance(input_size, int) else input_size
    if height % 32 or width % 32:
        raise ValueError(f"TorchScript input size must be divisible by 32, got {(height, width)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_obb_checkpoint(source, device).float().eval()
    wrapper = InferenceOnly(model).to(device).eval()
    example = torch.zeros((1, 3, int(height), int(width)), device=device, dtype=torch.float32)
    with torch.inference_mode():
        expected = wrapper(example)
        scripted = torch.jit.trace(wrapper, example, strict=False, check_trace=False)
        actual = scripted(example)
    if not torch.isfinite(actual).all():
        raise FloatingPointError("Converted TorchScript output contains NaN/Inf")
    if not torch.allclose(expected, actual, rtol=1e-5, atol=1e-5):
        difference = float((expected - actual).abs().max())
        raise RuntimeError(f"TorchScript verification failed; max absolute difference={difference}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Store parameters in FP16 to keep the deployment artifact compact. The
    # pipeline converts them back to FP32 by default before inference because
    # some CUDA/cuDNN combinations are unstable for this model in FP16.
    torch.jit.save(scripted.half(), str(destination))
    reloaded = torch.jit.load(str(destination), map_location=device).float().eval()
    with torch.inference_mode():
        reloaded_output = reloaded(example)
    if not torch.allclose(expected, reloaded_output, rtol=1e-5, atol=1e-5):
        difference = float((expected - reloaded_output).abs().max())
        raise RuntimeError(f"Reloaded TorchScript verification failed; max difference={difference}")
    metadata = {
        "format": "torchscript",
        "task": "obb",
        "source_weight": str(source),
        "source_sha256": _sha256(source),
        "output_weight": str(destination),
        "output_sha256": _sha256(destination),
        "input_size": [int(height), int(width)],
        "storage_dtype": "float16",
        "verified_runtime_dtype": "float32",
        "class_names": config["model"]["class_names"],
        "max_abs_verification_error": float((expected - reloaded_output).abs().max()),
    }
    OUTPUT_METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return destination


if __name__ == "__main__":
    convert_checkpoint()
