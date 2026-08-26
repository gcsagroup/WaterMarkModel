from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import load_obb_checkpoint
    from postprocess import non_max_suppression_obb, xywhr_to_corners
    from preprocess import discover_images, image_to_tensor, letterbox, read_image, write_image
else:
    from .model import load_obb_checkpoint
    from .postprocess import non_max_suppression_obb, xywhr_to_corners
    from .preprocess import discover_images, image_to_tensor, letterbox, read_image, write_image


DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "config.yaml"


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing config value: {context}.{key}")
    return mapping[key]


def _resolve_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return (config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()


def _select_device(value: str) -> torch.device:
    normalized = str(value).strip().lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested in config, but PyTorch cannot access CUDA: {value}")
    return device


def _pair(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        result = (value, value)
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        result = (int(value[0]), int(value[1]))
    else:
        raise ValueError(f"{name} must be an integer or [height, width]")
    if result[0] <= 0 or result[1] <= 0 or result[0] % 32 or result[1] % 32:
        raise ValueError(f"{name} dimensions must be positive multiples of 32, got {result}")
    return result


class OBBInferencePipeline:
    """Config-driven, standalone OBB inference pipeline."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        loaded = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Config root must be a mapping: {self.config_path}")
        self.config = loaded

        model_config = _required(loaded, "model", "config")
        inference_config = _required(loaded, "inference", "config")
        preprocess_config = _required(loaded, "preprocess", "config")
        visualization_config = _required(loaded, "visualization", "config")
        test_config = _required(loaded, "test", "config")
        for name, section in (
            ("model", model_config),
            ("inference", inference_config),
            ("preprocess", preprocess_config),
            ("visualization", visualization_config),
            ("test", test_config),
        ):
            if not isinstance(section, dict):
                raise ValueError(f"config.{name} must be a mapping")

        self.weights_path = _resolve_path(_required(model_config, "weights", "model"), self.config_path)
        self.device = _select_device(_required(model_config, "device", "model"))
        self.input_size = _pair(_required(model_config, "input_size", "model"), "model.input_size")
        self.fp16 = bool(_required(model_config, "fp16", "model")) and self.device.type == "cuda"
        self.fp16_fallback = bool(_required(model_config, "fp16_fallback", "model"))
        self.fp16_disable_cudnn = bool(_required(model_config, "fp16_disable_cudnn", "model"))
        self._fp16_warning_emitted = False
        raw_names = _required(model_config, "class_names", "model")
        if isinstance(raw_names, list):
            self.class_names = {index: str(name) for index, name in enumerate(raw_names)}
        elif isinstance(raw_names, dict):
            self.class_names = {int(index): str(name) for index, name in raw_names.items()}
        else:
            raise ValueError("model.class_names must be a list or ID-to-name mapping")

        self.confidence_threshold = float(_required(inference_config, "confidence_threshold", "inference"))
        self.iou_threshold = float(_required(inference_config, "iou_threshold", "inference"))
        self.max_detections = int(_required(inference_config, "max_detections", "inference"))
        self.max_nms = int(_required(inference_config, "max_nms", "inference"))
        if not 0 <= self.confidence_threshold <= 1 or not 0 <= self.iou_threshold <= 1:
            raise ValueError("confidence_threshold and iou_threshold must be in [0, 1]")
        if self.max_detections <= 0 or self.max_nms < self.max_detections:
            raise ValueError("max_nms must be >= max_detections > 0")

        color = _required(preprocess_config, "padding_color", "preprocess")
        if not isinstance(color, list) or len(color) != 3:
            raise ValueError("preprocess.padding_color must contain three BGR values")
        self.padding_color = tuple(int(channel) for channel in color)
        self.scale_up = bool(_required(preprocess_config, "scale_up", "preprocess"))

        self.line_width = int(_required(visualization_config, "line_width", "visualization"))
        self.font_scale = float(_required(visualization_config, "font_scale", "visualization"))
        self.text_thickness = int(_required(visualization_config, "text_thickness", "visualization"))
        self.box_color = tuple(int(channel) for channel in _required(visualization_config, "color_bgr", "visualization"))
        self.jpeg_quality = int(_required(visualization_config, "jpeg_quality", "visualization"))

        self.test_input = _resolve_path(_required(test_config, "input", "test"), self.config_path)
        self.test_output = _resolve_path(_required(test_config, "output", "test"), self.config_path)
        self.test_recursive = bool(_required(test_config, "recursive", "test"))

        self.model = load_obb_checkpoint(self.weights_path, self.device)
        self.model.half() if self.fp16 else self.model.float()
        checkpoint_names = getattr(self.model, "names", None)
        if checkpoint_names and len(checkpoint_names) != len(self.class_names):
            raise ValueError(
                f"Config has {len(self.class_names)} classes but checkpoint has {len(checkpoint_names)}"
            )

    def _infer(self, image: np.ndarray) -> tuple[torch.Tensor, dict[str, Any]]:
        processed, letterbox_info = letterbox(
            image, self.input_size, self.padding_color, self.scale_up
        )
        tensor = image_to_tensor(processed, self.device, self.fp16)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            use_cudnn = not (self.fp16 and self.fp16_disable_cudnn)
            with torch.backends.cudnn.flags(enabled=use_cudnn):
                raw = self.model(tensor)
            prediction = raw[0] if isinstance(raw, (tuple, list)) else raw
            if not torch.isfinite(prediction).all():
                if not self.fp16 or not self.fp16_fallback:
                    raise FloatingPointError(
                        "Model output contains NaN/Inf. Disable model.fp16 or enable model.fp16_fallback."
                    )
                if not self._fp16_warning_emitted:
                    print("Warning: FP16 produced NaN/Inf; switching this pipeline instance to FP32.")
                    self._fp16_warning_emitted = True
                self.fp16 = False
                self.model.float()
                tensor = tensor.float()
                raw = self.model(tensor)
                prediction = raw[0] if isinstance(raw, (tuple, list)) else raw
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError("Model output still contains NaN/Inf after FP32 fallback")
            detections = non_max_suppression_obb(
                prediction,
                self.confidence_threshold,
                self.iou_threshold,
                self.max_detections,
                self.max_nms,
                len(self.class_names),
            )[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000

        detections = detections.float().cpu()
        if len(detections):
            detections[:, 0] = (detections[:, 0] - letterbox_info.pad_left) / letterbox_info.scale
            detections[:, 1] = (detections[:, 1] - letterbox_info.pad_top) / letterbox_info.scale
            detections[:, 2:4] /= letterbox_info.scale
        metadata = {
            "input_width": letterbox_info.input_width,
            "input_height": letterbox_info.input_height,
            "scale": letterbox_info.scale,
            "pad_left": letterbox_info.pad_left,
            "pad_top": letterbox_info.pad_top,
            "inference_ms": elapsed_ms,
        }
        return detections, metadata

    def _serialize(
        self, source: Path, image: np.ndarray, detections: torch.Tensor, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        height, width = image.shape[:2]
        items: list[dict[str, Any]] = []
        for index, detection in enumerate(detections):
            xywhr = torch.cat((detection[:4], detection[6:7]))
            corners = xywhr_to_corners(xywhr).numpy()
            normalized = corners / np.array([width, height], dtype=np.float32)
            normalized = np.clip(normalized, 0.0, 1.0)
            class_id = int(detection[5].item())
            item = {
                "id": index,
                "class_id": class_id,
                "class_name": self.class_names.get(class_id, str(class_id)),
                "confidence": round(float(detection[4].item()), 8),
                "obb": {
                    "corners_pixel": [[round(float(x), 4), round(float(y), 4)] for x, y in corners],
                    "corners_normalized": [
                        [round(float(x), 8), round(float(y), 8)] for x, y in normalized
                    ],
                    "center_pixel": [round(float(detection[0]), 4), round(float(detection[1]), 4)],
                    "width_pixel": round(float(detection[2]), 4),
                    "height_pixel": round(float(detection[3]), 4),
                    "angle_radians": round(float(detection[6]), 8),
                    "angle_degrees": round(math.degrees(float(detection[6])), 4),
                },
            }
            item["obb_label"] = [
                class_id,
                *[coordinate for point in item["obb"]["corners_normalized"] for coordinate in point],
            ]
            items.append(item)
        return {
            "image": {
                "file_name": source.name,
                "source_path": str(source.resolve()),
                "width": width,
                "height": height,
            },
            "annotation_type": "obb_prediction",
            "class_names": {str(index): name for index, name in self.class_names.items()},
            "corner_order": "clockwise",
            "model": {
                "weights": str(self.weights_path),
                "device": str(self.device),
                "fp16": self.fp16,
                "fp16_fallback": self.fp16_fallback,
                "fp16_disable_cudnn": self.fp16_disable_cudnn,
                "confidence_threshold": self.confidence_threshold,
                "iou_threshold": self.iou_threshold,
            },
            "preprocess": {key: round(value, 6) if isinstance(value, float) else value for key, value in metadata.items()},
            "detection_count": len(items),
            "watermarks": items,
        }

    def _draw(self, image: np.ndarray, result: dict[str, Any]) -> np.ndarray:
        annotated = image.copy()
        height, width = annotated.shape[:2]
        for item in result["watermarks"]:
            raw = np.asarray(item["obb"]["corners_pixel"], dtype=np.float32)
            raw[:, 0] = np.clip(raw[:, 0], 0, max(width - 1, 0))
            raw[:, 1] = np.clip(raw[:, 1], 0, max(height - 1, 0))
            points = np.rint(raw).astype(np.int32)
            cv2.polylines(annotated, [points], True, self.box_color, self.line_width, cv2.LINE_AA)
            label = f'{item["class_name"]} {item["confidence"]:.2f}'
            origin = tuple(points[np.argmin(points[:, 1])].tolist())
            text_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.text_thickness
            )
            x = min(max(origin[0], 0), max(width - text_size[0] - 2, 0))
            y = max(origin[1] - 4, text_size[1] + baseline + 2)
            cv2.rectangle(
                annotated,
                (x, y - text_size[1] - baseline - 2),
                (x + text_size[0] + 2, y + 1),
                self.box_color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x + 1, y - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (0, 0, 0),
                self.text_thickness,
                cv2.LINE_AA,
            )
        return annotated

    def predict_file(
        self, input_path: str | Path, output_dir: str | Path | None = None
    ) -> dict[str, Any]:
        """Infer one image and optionally save its JSON and annotated image."""

        source = Path(input_path).expanduser().resolve()
        image = read_image(source)
        detections, metadata = self._infer(image)
        result = self._serialize(source, image, detections, metadata)
        if output_dir is not None:
            destination = Path(output_dir).expanduser().resolve()
            destination.mkdir(parents=True, exist_ok=True)
            json_path = destination / f"{source.stem}.json"
            image_path = destination / f"{source.stem}_annotated{source.suffix.lower()}"
            json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            write_image(image_path, self._draw(image, result), self.jpeg_quality)
            result["outputs"] = {"json": str(json_path), "annotated_image": str(image_path)}
        return result

    def predict_folder(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        recursive: bool = False,
    ) -> list[dict[str, Any]]:
        """Infer all supported images in a folder and preserve relative subfolders."""

        source_root = Path(input_dir).expanduser().resolve()
        destination_root = Path(output_dir).expanduser().resolve()
        images = discover_images(source_root, recursive)
        if not images:
            raise ValueError(f"No supported images found in: {source_root}")
        results = []
        for index, image_path in enumerate(images, start=1):
            relative_parent = image_path.relative_to(source_root).parent
            result = self.predict_file(image_path, destination_root / relative_parent)
            results.append(result)
            print(
                f"[{index}/{len(images)}] {image_path.name}: "
                f"{result['detection_count']} detections, "
                f"{result['preprocess']['inference_ms']:.2f} ms"
            )
        return results

    def run_test(self) -> list[dict[str, Any]]:
        """Run the config-defined test input folder and write test/output artifacts."""

        return self.predict_folder(self.test_input, self.test_output, self.test_recursive)


def main() -> int:
    pipeline = OBBInferencePipeline(DEFAULT_CONFIG)
    results = pipeline.run_test()
    print(f"Completed {len(results)} images. Output: {pipeline.test_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
