"""Standalone MobileNetV3-Large binary watermark classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


MODEL_VARIANT = "mobilenet_v3_large"


class WatermarkClassifier(nn.Module):
    """Return one watermark-presence logit for each 320x320 tile."""

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        try:
            from torchvision.models import mobilenet_v3_large
        except ImportError as exc:
            raise ImportError("需要安装torchvision以构建MobileNetV3-Large") from exc
        network = mobilenet_v3_large(weights=None)
        in_features = network.classifier[-1].in_features
        network.classifier[-2] = nn.Dropout(p=dropout, inplace=True)
        network.classifier[-1] = nn.Linear(in_features, 1)
        self.network = network

    def forward(self, images: Tensor) -> Tensor:
        return self.network(images).flatten()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_classifier(
    weights_path: str | Path,
    device: torch.device,
    dropout: float = 0.2,
) -> tuple[WatermarkClassifier, dict[str, Any]]:
    path = Path(weights_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"权重文件不存在: {path}")
    checkpoint = _torch_load(path)
    metadata: dict[str, Any] = {}
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        metadata = {key: value for key, value in checkpoint.items() if key != "model_state_dict"}
        variant = checkpoint.get("model_variant", MODEL_VARIANT)
        if variant != MODEL_VARIANT:
            raise ValueError(f"权重模型为{variant}，当前pipeline要求{MODEL_VARIANT}")
    elif isinstance(checkpoint, dict) and checkpoint and all(
        isinstance(value, Tensor) for value in checkpoint.values()
    ):
        state = checkpoint
    else:
        raise ValueError(f"无法识别的PyTorch权重格式: {path}")
    if not isinstance(state, dict):
        raise ValueError("checkpoint中的model_state_dict无效")
    model = WatermarkClassifier(dropout=dropout)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, metadata
