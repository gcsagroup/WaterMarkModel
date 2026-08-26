"""Configuration-driven PyTorch inference pipeline for watermark screening."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import torch
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import MODEL_VARIANT, load_classifier
    from preprocess import discover_images, iter_image_tiles, read_rgb_image
    from settings import (
        DEFAULT_CONFIG,
        as_float_triplet,
        as_int_triplet,
        load_config,
        required,
        resolve_config_path,
    )
else:
    from .model import MODEL_VARIANT, load_classifier
    from .preprocess import discover_images, iter_image_tiles, read_rgb_image
    from .settings import (
        DEFAULT_CONFIG,
        as_float_triplet,
        as_int_triplet,
        load_config,
        required,
        resolve_config_path,
    )


@dataclass(frozen=True)
class DetectionResult:
    image: str
    has_watermark: bool
    probability: float
    threshold: float
    original_size: tuple[int, int]
    tile_count: int
    max_tile_coordinate: tuple[int, int]
    elapsed_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求CUDA，但当前环境不可用")
    return device


class DetectionPipeline:
    """Load configuration/weights once and expose file, directory and test APIs."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG) -> None:
        self.config, self.config_path = load_config(config_path)
        model_config = self.config["model"]
        preprocess_config = self.config["preprocess"]
        inference_config = self.config["inference"]

        architecture = str(required(model_config, "architecture", "model")).lower()
        if architecture != MODEL_VARIANT:
            raise ValueError(
                f"model.architecture={architecture!r}，当前pipeline要求{MODEL_VARIANT!r}"
            )

        self.weights_path = resolve_config_path(
            required(model_config, "weights", "model"), self.config_path
        )
        self.device = resolve_device(str(required(model_config, "device", "model")))
        self.dropout = float(required(model_config, "dropout", "model"))
        self.model, self.checkpoint_metadata = load_classifier(
            self.weights_path, self.device, self.dropout
        )

        self.tile_size = int(required(preprocess_config, "tile_size", "preprocess"))
        self.stride = int(required(preprocess_config, "stride", "preprocess"))
        if self.tile_size <= 0 or self.stride <= 0 or self.stride > self.tile_size:
            raise ValueError("preprocess.tile_size/stride配置无效")
        self.padding_color = as_int_triplet(
            required(preprocess_config, "padding_color", "preprocess"),
            "preprocess.padding_color",
        )
        self.mean = as_float_triplet(
            required(preprocess_config, "imagenet_mean", "preprocess"),
            "preprocess.imagenet_mean",
        )
        self.std = as_float_triplet(
            required(preprocess_config, "imagenet_std", "preprocess"),
            "preprocess.imagenet_std",
        )
        self.apply_exif_orientation = bool(
            required(preprocess_config, "apply_exif_orientation", "preprocess")
        )

        self.threshold = float(required(inference_config, "threshold", "inference"))
        self.tile_batch_size = int(
            required(inference_config, "tile_batch_size", "inference")
        )
        self.amp_enabled = (
            bool(required(inference_config, "amp", "inference"))
            and self.device.type == "cuda"
        )
        self.recursive = bool(required(inference_config, "recursive", "inference"))
        self.continue_on_error = bool(
            required(inference_config, "continue_on_error", "inference")
        )
        self.extensions = tuple(
            str(value).lower()
            for value in required(inference_config, "image_extensions", "inference")
        )
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("inference.threshold必须位于0到1")
        if self.tile_batch_size <= 0:
            raise ValueError("inference.tile_batch_size必须为正数")

    @torch.inference_mode()
    def infer_file(self, image_path: str | Path) -> DetectionResult:
        path = Path(image_path).expanduser().resolve()
        started = time.perf_counter()
        image = read_rgb_image(path, self.apply_exif_orientation)
        original_size = (image.width, image.height)
        pending_tensors: list[torch.Tensor] = []
        pending_coordinates: list[tuple[int, int]] = []
        maximum_probability = -1.0
        maximum_coordinate = (0, 0)
        tile_count = 0

        def process_pending() -> None:
            nonlocal maximum_probability, maximum_coordinate
            if not pending_tensors:
                return
            batch = torch.stack(pending_tensors).to(self.device, non_blocking=True)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                probabilities = self.model(batch).float().sigmoid()
            if not bool(torch.isfinite(probabilities).all()):
                raise FloatingPointError(f"模型对图像{path}产生NaN/Inf")
            local_value, local_index = probabilities.max(dim=0)
            value = float(local_value.item())
            if value > maximum_probability:
                maximum_probability = value
                maximum_coordinate = pending_coordinates[int(local_index.item())]
            pending_tensors.clear()
            pending_coordinates.clear()

        for x, y, tile in iter_image_tiles(
            image,
            self.tile_size,
            self.stride,
            self.padding_color,
            self.mean,
            self.std,
        ):
            pending_tensors.append(tile)
            pending_coordinates.append((x, y))
            tile_count += 1
            if len(pending_tensors) >= self.tile_batch_size:
                process_pending()
        process_pending()
        if maximum_probability < 0.0:
            raise RuntimeError(f"图像未产生任何滑窗: {path}")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return DetectionResult(
            image=str(path),
            has_watermark=maximum_probability >= self.threshold,
            probability=maximum_probability,
            threshold=self.threshold,
            original_size=original_size,
            tile_count=tile_count,
            max_tile_coordinate=maximum_coordinate,
            elapsed_ms=elapsed_ms,
        )

    def infer_directory(self, directory: str | Path) -> list[DetectionResult]:
        paths = discover_images(directory, self.recursive, self.extensions)
        return self._infer_paths(paths)

    def infer_path(self, input_path: str | Path) -> list[DetectionResult]:
        paths = discover_images(input_path, self.recursive, self.extensions)
        return self._infer_paths(paths)

    def _infer_paths(self, paths: Sequence[Path]) -> list[DetectionResult]:
        results: list[DetectionResult] = []
        progress = tqdm(paths, desc="水印初筛", dynamic_ncols=True, disable=len(paths) <= 1)
        for path in progress:
            try:
                result = self.infer_file(path)
            except Exception as exc:
                if not self.continue_on_error:
                    raise
                result = DetectionResult(
                    image=str(path),
                    has_watermark=False,
                    probability=0.0,
                    threshold=self.threshold,
                    original_size=(0, 0),
                    tile_count=0,
                    max_tile_coordinate=(0, 0),
                    elapsed_ms=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            results.append(result)
            progress.set_postfix(
                probability=f"{result.probability:.4f}",
                watermark=result.has_watermark,
                refresh=True,
            )
        return results

    def save_results(
        self,
        results: Sequence[DetectionResult],
        output_path: str | Path,
        input_path: str | Path,
    ) -> Path:
        output = Path(output_path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        successful = [result for result in results if result.error is None]
        document = {
            "task": "visible_watermark_screening",
            "model_variant": MODEL_VARIANT,
            "weights": str(self.weights_path),
            "input": str(Path(input_path).expanduser().resolve()),
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "threshold": self.threshold,
            "aggregation": "max_tile_probability",
            "total": len(results),
            "watermark_candidates": sum(result.has_watermark for result in successful),
            "non_watermark": sum(not result.has_watermark for result in successful),
            "errors": sum(result.error is not None for result in results),
            "results": [result.to_dict() for result in results],
        }
        output.write_text(
            json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return output

    def run(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
    ) -> list[DetectionResult]:
        results = self.infer_path(input_path)
        if output_path is not None:
            saved = self.save_results(results, output_path, input_path)
            print(f"结果已保存: {saved}")
        return results

    def test(self) -> list[DetectionResult]:
        test_config = self.config["test"]
        input_path = resolve_config_path(
            required(test_config, "input", "test"), self.config_path
        )
        output_path = resolve_config_path(
            required(test_config, "result", "test"), self.config_path
        )
        return self.run(input_path, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可见水印初筛PyTorch pipeline")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML配置文件")
    parser.add_argument("--input", default=None, help="单张图片或图片文件夹；省略则运行test")
    parser.add_argument("--output", default=None, help="可选JSON输出路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = DetectionPipeline(args.config)
    results = pipeline.test() if args.input is None else pipeline.run(args.input, args.output)
    for result in results:
        status = "疑似水印" if result.has_watermark else "暂未检出水印"
        error = f", error={result.error}" if result.error else ""
        print(f"{result.image}: {status}, probability={result.probability:.6f}{error}")


if __name__ == "__main__":
    main()
