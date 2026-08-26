from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class LetterboxInfo:
    original_width: int
    original_height: int
    input_width: int
    input_height: int
    scale: float
    pad_left: int
    pad_top: int


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unsupported or corrupt image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray, jpeg_quality: int = 95) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".jpg"
    parameters = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if suffix in {".jpg", ".jpeg"} else []
    success, encoded = cv2.imencode(suffix, image, parameters)
    if not success:
        raise OSError(f"Failed to encode output image: {path}")
    encoded.tofile(path)


def letterbox(
    image: np.ndarray,
    size: tuple[int, int],
    padding_color: tuple[int, int, int],
    scale_up: bool,
) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize without distortion and center-pad to the fixed training input size."""

    input_height, input_width = size
    original_height, original_width = image.shape[:2]
    scale = min(input_width / original_width, input_height / original_height)
    if not scale_up:
        scale = min(scale, 1.0)
    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))
    if (resized_width, resized_height) != (original_width, original_height):
        interpolation = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
        image = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)

    pad_width = input_width - resized_width
    pad_height = input_height - resized_height
    if pad_width < 0 or pad_height < 0:
        raise ValueError(f"Invalid letterbox size {size} for resized image {image.shape[:2]}")
    left = round(pad_width / 2 - 0.1)
    right = round(pad_width / 2 + 0.1)
    top = round(pad_height / 2 - 0.1)
    bottom = round(pad_height / 2 + 0.1)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    info = LetterboxInfo(
        original_width=original_width,
        original_height=original_height,
        input_width=input_width,
        input_height=input_height,
        scale=scale,
        pad_left=left,
        pad_top=top,
    )
    return image, info


def image_to_tensor(image: np.ndarray, device: torch.device, fp16: bool) -> torch.Tensor:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0)
    tensor = tensor.to(device=device, non_blocking=True)
    tensor = tensor.half() if fp16 else tensor.float()
    return tensor / 255.0


def discover_images(directory: str | Path, recursive: bool) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Input folder not found: {directory}")
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
