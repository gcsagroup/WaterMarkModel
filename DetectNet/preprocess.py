"""Training-compatible native-resolution sliding-window preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import Tensor


DEFAULT_IMAGE_SUFFIXES = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)


def sliding_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= 0 or tile_size <= 0 or stride <= 0 or stride > tile_size:
        raise ValueError("图像尺寸、tile_size和stride配置无效")
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if positions[-1] != final:
        positions.append(final)
    return positions


def sliding_coordinates(
    width: int, height: int, tile_size: int, stride: int
) -> list[tuple[int, int]]:
    return [
        (x, y)
        for y in sliding_positions(height, tile_size, stride)
        for x in sliding_positions(width, tile_size, stride)
    ]


def read_rgb_image(path: str | Path, apply_exif_orientation: bool = True) -> Image.Image:
    image_path = Path(path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"图像不存在: {image_path}")
    with Image.open(image_path) as file:
        image = file.copy()
    if apply_exif_orientation:
        image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def normalize_tile(
    tile: Image.Image,
    tile_size: int,
    padding_color: tuple[int, int, int],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> Tensor:
    if any(value <= 0 for value in std):
        raise ValueError("归一化std必须大于0")
    canvas = Image.new("RGB", (tile_size, tile_size), padding_color)
    canvas.paste(tile.convert("RGB"), (0, 0))
    array = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    mean_tensor = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def iter_image_tiles(
    image: Image.Image,
    tile_size: int,
    stride: int,
    padding_color: tuple[int, int, int],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> Iterator[tuple[int, int, Tensor]]:
    image = image.convert("RGB")
    for x, y in sliding_coordinates(image.width, image.height, tile_size, stride):
        crop = image.crop(
            (x, y, min(x + tile_size, image.width), min(y + tile_size, image.height))
        )
        yield x, y, normalize_tile(crop, tile_size, padding_color, mean, std)


def discover_images(
    input_path: str | Path,
    recursive: bool,
    suffixes: Sequence[str] = DEFAULT_IMAGE_SUFFIXES,
) -> list[Path]:
    path = Path(input_path).expanduser().resolve()
    supported = {suffix.lower() for suffix in suffixes}
    if path.is_file():
        if path.suffix.lower() not in supported:
            raise ValueError(f"不支持的图像后缀: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"输入路径不存在: {path}")
    candidates = path.rglob("*") if recursive else path.iterdir()
    images = sorted(
        item for item in candidates if item.is_file() and item.suffix.lower() in supported
    )
    if not images:
        raise ValueError(f"输入路径下没有支持的图像: {path}")
    return images
