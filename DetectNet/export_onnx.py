"""Export the configured PyTorch classifier to FP32/FP16/FP8/INT8 ONNX."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

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


class ExportableClassifier(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checked(model: Any, path: Path) -> None:
    import onnx

    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    onnx.checker.check_model(onnx.load(str(path)))


def validate_runtime(
    model: nn.Module,
    onnx_path: Path,
    sample: torch.Tensor,
    rtol: float,
    atol: float,
    strict: bool,
) -> dict[str, Any]:
    try:
        import onnxruntime as ort

        with torch.inference_mode():
            expected = model(sample).cpu().numpy()
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        actual = session.run(None, {input_name: sample.cpu().numpy()})[0]
        if expected.shape != actual.shape:
            raise RuntimeError(f"输出shape不一致: PyTorch={expected.shape}, ONNX={actual.shape}")
        difference = np.abs(expected - actual)
        result = {
            "passed": bool(np.allclose(expected, actual, rtol=rtol, atol=atol)),
            "max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()),
            "rtol": rtol,
            "atol": atol,
        }
        if strict and not result["passed"]:
            raise RuntimeError(f"ONNX数值验证失败: {result}")
        return result
    except Exception as exc:
        if strict:
            raise
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}"}


def convert_fp16(source: Any) -> Any:
    from onnxconverter_common import float16

    return float16.convert_float_to_float16(
        deepcopy(source),
        keep_io_types=True,
        disable_shape_infer=False,
        min_positive_val=5.960464477539063e-08,
        max_finite_val=65504.0,
    )


def convert_fp8_weight_only(source: Any, target_opset: int) -> tuple[Any, int, int]:
    """Store 2D+ weights as per-output-channel E4M3FN and restore them in-graph."""

    import ml_dtypes
    import onnx
    from onnx import TensorProto, helper, numpy_helper, version_converter

    if target_opset < 19:
        raise ValueError("FP8权重要求opset_version >= 19")
    current_opset = next(
        (entry.version for entry in source.opset_import if entry.domain in {"", "ai.onnx"}),
        0,
    )
    model = (
        version_converter.convert_version(deepcopy(source), target_opset)
        if current_opset != target_opset
        else deepcopy(source)
    )
    retained = []
    quantized = []
    restore_nodes = []
    for initializer in model.graph.initializer:
        if initializer.data_type != TensorProto.FLOAT or len(initializer.dims) < 2:
            retained.append(initializer)
            continue
        values = np.asarray(numpy_helper.to_array(initializer), dtype=np.float32)
        reduction_axes = tuple(range(1, values.ndim))
        maximum = np.max(np.abs(values), axis=reduction_axes, keepdims=True)
        scale = np.where(maximum > 0, maximum / 448.0, 1.0).astype(np.float32)
        encoded = np.asarray(
            np.clip(values / scale, -448.0, 448.0), dtype=ml_dtypes.float8_e4m3fn
        )
        encoded_name = f"{initializer.name}.e4m3fn"
        cast_name = f"{initializer.name}.fp32"
        scale_name = f"{initializer.name}.scale"
        quantized.extend(
            [
                numpy_helper.from_array(encoded, name=encoded_name),
                numpy_helper.from_array(scale, name=scale_name),
            ]
        )
        restore_nodes.extend(
            [
                helper.make_node(
                    "Cast",
                    [encoded_name],
                    [cast_name],
                    name=f"{initializer.name}.cast_fp8_to_fp32",
                    to=TensorProto.FLOAT,
                ),
                helper.make_node(
                    "Mul",
                    [cast_name, scale_name],
                    [initializer.name],
                    name=f"{initializer.name}.restore_scale",
                ),
            ]
        )
    original_nodes = list(model.graph.node)
    del model.graph.initializer[:]
    model.graph.initializer.extend([*retained, *quantized])
    del model.graph.node[:]
    model.graph.node.extend([*restore_nodes, *original_nodes])
    onnx.checker.check_model(model)
    return model, len(restore_nodes) // 2, len(retained)


class TileCalibrationReader:
    def __init__(self, samples: Sequence[np.ndarray]) -> None:
        self.samples = list(samples)
        self.position = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.position >= len(self.samples):
            return None
        sample = self.samples[self.position]
        self.position += 1
        return {"images": sample}

    def rewind(self) -> None:
        self.position = 0


def load_calibration_tiles(
    config: dict[str, Any], config_path: Path
) -> list[np.ndarray]:
    preprocess_config = config["preprocess"]
    inference_config = config["inference"]
    export_config = config["export"]
    architecture = str(required(model_config, "architecture", "model")).lower()
    if architecture != MODEL_VARIANT:
        raise ValueError(
            f"model.architecture={architecture!r}，当前导出器要求{MODEL_VARIANT!r}"
        )
    directory = resolve_config_path(
        required(export_config, "calibration_input", "export"), config_path
    )
    recursive = bool(required(export_config, "calibration_recursive", "export"))
    limit = int(required(export_config, "calibration_max_tiles", "export"))
    if limit <= 0:
        raise ValueError("export.calibration_max_tiles必须为正数")
    paths = discover_images(
        directory,
        recursive,
        required(inference_config, "image_extensions", "inference"),
    )
    tile_size = int(required(preprocess_config, "tile_size", "preprocess"))
    stride = int(required(preprocess_config, "stride", "preprocess"))
    padding_color = as_int_triplet(
        required(preprocess_config, "padding_color", "preprocess"),
        "preprocess.padding_color",
    )
    mean = as_float_triplet(
        required(preprocess_config, "imagenet_mean", "preprocess"),
        "preprocess.imagenet_mean",
    )
    std = as_float_triplet(
        required(preprocess_config, "imagenet_std", "preprocess"),
        "preprocess.imagenet_std",
    )
    apply_exif = bool(
        required(preprocess_config, "apply_exif_orientation", "preprocess")
    )
    samples: list[np.ndarray] = []
    for path in paths:
        image = read_rgb_image(path, apply_exif)
        for _, _, tile in iter_image_tiles(
            image, tile_size, stride, padding_color, mean, std
        ):
            samples.append(tile.unsqueeze(0).numpy().astype(np.float32, copy=False))
            if len(samples) >= limit:
                return samples
    if not samples:
        raise ValueError(f"校准目录没有产生滑窗: {directory}")
    return samples


def quantize_int8_static(
    source_path: Path,
    target_path: Path,
    samples: Sequence[np.ndarray],
    method: str,
    per_channel: bool,
) -> None:
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    methods = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
        "distribution": CalibrationMethod.Distribution,
    }
    key = method.strip().lower()
    if key not in methods:
        raise ValueError(f"不支持的校准方法: {method}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(source_path),
        str(target_path),
        TileCalibrationReader(samples),
        quant_format=QuantFormat.QDQ,
        per_channel=per_channel,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=methods[key],
        op_types_to_quantize=["Conv", "Gemm", "MatMul"],
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    )


def export_all(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Path]:
    import onnx

    config, resolved_config = load_config(config_path)
    model_config = config["model"]
    preprocess_config = config["preprocess"]
    inference_config = config["inference"]
    export_config = config["export"]
    weights = resolve_config_path(required(model_config, "weights", "model"), resolved_config)
    output_dir = resolve_config_path(
        required(export_config, "output_dir", "export"), resolved_config
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        precision: output_dir / str(required(export_config, f"{precision}_name", "export"))
        for precision in ("fp32", "fp16", "fp8", "int8")
    }
    opset = int(required(export_config, "opset_version", "export"))
    if opset < 19:
        raise ValueError("为支持FP8，export.opset_version必须至少为19")
    tile_size = int(required(preprocess_config, "tile_size", "preprocess"))
    dropout = float(required(model_config, "dropout", "model"))
    threshold = float(required(inference_config, "threshold", "inference"))
    validate = bool(required(export_config, "validate", "export"))
    strict_quantized = bool(
        required(export_config, "strict_quantized_validation", "export")
    )
    fp8_mode = str(required(export_config, "fp8_mode", "export")).strip().lower()
    if fp8_mode != "weight_only_e4m3fn":
        raise ValueError(
            "export.fp8_mode当前仅支持'weight_only_e4m3fn'，"
            f"实际为{fp8_mode!r}"
        )

    classifier, checkpoint_metadata = load_classifier(weights, torch.device("cpu"), dropout)
    model = ExportableClassifier(classifier.float().eval()).eval()
    calibration_samples = load_calibration_tiles(config, resolved_config)
    sample = torch.from_numpy(calibration_samples[0])
    with torch.inference_mode():
        reference = model(sample)
    if not bool(torch.isfinite(reference).all()):
        raise FloatingPointError("PyTorch源模型输出NaN/Inf")

    dynamic_axes = (
        {"images": {0: "batch"}, "logits": {0: "batch"}}
        if bool(required(export_config, "dynamic_batch", "export"))
        else None
    )
    torch.onnx.export(
        model,
        (sample,),
        str(paths["fp32"]),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    fp32_model = onnx.load(str(paths["fp32"]))
    onnx.checker.check_model(fp32_model)

    fp16_model = convert_fp16(fp32_model)
    save_checked(fp16_model, paths["fp16"])
    fp8_model, fp8_quantized, fp8_retained = convert_fp8_weight_only(fp32_model, opset)
    save_checked(fp8_model, paths["fp8"])
    quantize_int8_static(
        paths["fp32"],
        paths["int8"],
        calibration_samples,
        str(required(export_config, "calibration_method", "export")),
        bool(required(export_config, "int8_per_channel", "export")),
    )
    onnx.checker.check_model(onnx.load(str(paths["int8"])))

    tolerances = {
        precision: (
            float(required(export_config, f"{precision}_rtol", "export")),
            float(required(export_config, f"{precision}_atol", "export")),
        )
        for precision in ("fp32", "fp16", "fp8", "int8")
    }
    validations = {
        precision: (
            validate_runtime(
                model,
                path,
                sample,
                *tolerances[precision],
                strict=(precision == "fp32" or strict_quantized),
            )
            if validate
            else None
        )
        for precision, path in paths.items()
    }
    checkpoint_epoch = checkpoint_metadata.get("epoch")
    selected_threshold = checkpoint_metadata.get("selected_threshold")
    common_metadata = {
        "task": "visible_watermark_tile_classification",
        "model_variant": MODEL_VARIANT,
        "source_weight": str(weights),
        "source_sha256": sha256(weights),
        "source_epoch": checkpoint_epoch,
        "config_threshold": threshold,
        "checkpoint_selected_threshold": selected_threshold,
        "input": {
            "name": "images",
            "shape": ["batch", 3, tile_size, tile_size],
            "dtype": "float32",
            "normalization": "ImageNet mean/std; performed outside ONNX",
        },
        "output": {
            "name": "logits",
            "shape": ["batch"],
            "postprocess": "sigmoid per tile, max across image tiles, compare with threshold",
        },
        "opset_version": opset,
    }
    for precision, path in paths.items():
        metadata = {
            **common_metadata,
            "precision": precision,
            "output_model": str(path),
            "output_sha256": sha256(path),
            "validation": validations[precision],
        }
        if precision == "fp8":
            metadata.update(
                {
                    "storage": "E4M3FN per-output-channel weight-only",
                    "native_fp8_compute": False,
                    "compute_precision": "FP32 after in-graph Cast and scale",
                    "quantized_initializers": fp8_quantized,
                    "retained_initializers": fp8_retained,
                }
            )
        if precision == "int8":
            metadata.update(
                {
                    "format": "static QDQ",
                    "native_int8_compute": True,
                    "calibration_method": str(
                        required(export_config, "calibration_method", "export")
                    ),
                    "calibration_tiles": len(calibration_samples),
                }
            )
        path.with_suffix(".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (output_dir / "export_summary.json").write_text(
        json.dumps(
            {
                "source_weight": str(weights),
                "tile_size": tile_size,
                "opset_version": opset,
                "artifacts": {
                    precision: {
                        "path": str(path),
                        "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
                        "validation": validations[precision],
                    }
                    for precision, path in paths.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                precision: {
                    "path": str(path),
                    "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
                    "validation": validations[precision],
                }
                for precision, path in paths.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="导出Detect MobileNetV3-Large ONNX")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    export_all(args.config)


if __name__ == "__main__":
    main()
