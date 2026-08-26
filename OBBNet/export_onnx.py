from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import load_obb_checkpoint
    from preprocess import discover_images, letterbox, read_image
else:
    from .model import load_obb_checkpoint
    from .preprocess import discover_images, letterbox, read_image


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


class ExportableModel(nn.Module):
    """Expose only the decoded OBB prediction tensor from the inference model."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(images)
        return output[0] if isinstance(output, (tuple, list)) else output


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing config value: {context}.{key}")
    return mapping[key]


def _resolve_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value).expanduser()
    return (config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()


def _input_size(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        height, width = value, value
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        height, width = int(value[0]), int(value[1])
    else:
        raise ValueError("model.input_size must be an integer or [height, width]")
    if height <= 0 or width <= 0 or height % 32 or width % 32:
        raise ValueError(f"Input dimensions must be positive multiples of 32, got {(height, width)}")
    return height, width


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_runtime(
    model: nn.Module,
    onnx_path: Path,
    sample: torch.Tensor,
    rtol: float,
    atol: float,
    strict: bool = True,
) -> dict[str, float | bool] | None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("ONNX Runtime is not installed; numerical validation was skipped.")
        return None

    with torch.inference_mode():
        expected = model(sample).cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    actual = session.run(None, {input_name: sample.cpu().numpy()})[0]
    if expected.shape != actual.shape:
        raise RuntimeError(f"Output shape mismatch: PyTorch={expected.shape}, ONNX={actual.shape}")
    differences = np.abs(expected - actual)
    max_error = float(np.max(differences))
    mean_error = float(np.mean(differences))
    passed = bool(np.allclose(expected, actual, rtol=rtol, atol=atol))
    if strict and not passed:
        raise RuntimeError(
            f"ONNX numerical validation failed: max_abs_error={max_error}, "
            f"rtol={rtol}, atol={atol}"
        )
    return {"passed": passed, "max_abs_error": max_error, "mean_abs_error": mean_error}


def _save_checked(model: Any, path: Path) -> None:
    import onnx

    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    # Check the serialized representation as well, not only the in-memory graph.
    onnx.checker.check_model(onnx.load(str(path)))


def _convert_fp16(source: Any) -> Any:
    try:
        from onnxconverter_common import float16
    except ImportError as exc:
        raise RuntimeError(
            "onnxconverter-common is required for FP16 conversion. Install requirements.txt first."
        ) from exc
    return float16.convert_float_to_float16(
        deepcopy(source),
        keep_io_types=True,
        disable_shape_infer=False,
        min_positive_val=5.960464477539063e-08,
        max_finite_val=65504.0,
    )


def _convert_fp8_weights(source: Any, target_opset: int) -> tuple[Any, int, int]:
    """Store convolution weights as per-output-channel E4M3FN values."""

    import ml_dtypes
    import onnx
    from onnx import TensorProto, helper, numpy_helper, version_converter

    current_opset = next(
        (entry.version for entry in source.opset_import if entry.domain in {"", "ai.onnx"}), 0
    )
    if target_opset < 19:
        raise ValueError("FP8 export requires export.fp8_opset_version >= 19")
    model = (
        version_converter.convert_version(deepcopy(source), target_opset)
        if current_opset != target_opset
        else deepcopy(source)
    )

    retained = []
    quantized = []
    dequantize_nodes = []
    for initializer in model.graph.initializer:
        if initializer.data_type != TensorProto.FLOAT or len(initializer.dims) < 2:
            retained.append(initializer)
            continue
        values = np.asarray(numpy_helper.to_array(initializer), dtype=np.float32)
        reduction_axes = tuple(range(1, values.ndim))
        max_abs = np.max(np.abs(values), axis=reduction_axes, keepdims=True)
        scale = np.where(max_abs > 0, max_abs / 448.0, 1.0).astype(np.float32)
        encoded = np.asarray(np.clip(values / scale, -448.0, 448.0), dtype=ml_dtypes.float8_e4m3fn)
        encoded_name = f"{initializer.name}.e4m3fn"
        cast_name = f"{initializer.name}.cast_fp32"
        scale_name = f"{initializer.name}.scale"
        quantized.append(numpy_helper.from_array(encoded, name=encoded_name))
        quantized.append(numpy_helper.from_array(np.asarray(scale, dtype=np.float32), name=scale_name))
        dequantize_nodes.append(
            helper.make_node(
                "Cast",
                [encoded_name],
                [cast_name],
                name=f"{initializer.name}.cast",
                to=TensorProto.FLOAT,
            )
        )
        dequantize_nodes.append(
            helper.make_node(
                "Mul",
                [cast_name, scale_name],
                [initializer.name],
                name=f"{initializer.name}.rescale",
            )
        )

    del model.graph.initializer[:]
    model.graph.initializer.extend([*retained, *quantized])
    original_nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend([*dequantize_nodes, *original_nodes])
    onnx.checker.check_model(model)
    return model, len(dequantize_nodes) // 2, len(retained)


def _convert_int8_weights(source: Any) -> tuple[Any, int, int]:
    """Store convolution weights as symmetric per-output-channel INT8 values."""

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = deepcopy(source)
    retained = []
    quantized = []
    dequantize_nodes = []
    for initializer in model.graph.initializer:
        if initializer.data_type != TensorProto.FLOAT or len(initializer.dims) < 2:
            retained.append(initializer)
            continue
        values = np.asarray(numpy_helper.to_array(initializer), dtype=np.float32)
        reduction_axes = tuple(range(1, values.ndim))
        max_abs = np.max(np.abs(values), axis=reduction_axes)
        scales = np.where(max_abs > 0, max_abs / 127.0, 1.0).astype(np.float32)
        broadcast_shape = (values.shape[0],) + (1,) * (values.ndim - 1)
        encoded = np.clip(
            np.rint(values / scales.reshape(broadcast_shape)), -127, 127
        ).astype(np.int8)
        encoded_name = f"{initializer.name}.int8"
        scale_name = f"{initializer.name}.scale"
        zero_name = f"{initializer.name}.zero"
        quantized.extend(
            [
                numpy_helper.from_array(encoded, name=encoded_name),
                numpy_helper.from_array(scales, name=scale_name),
                numpy_helper.from_array(np.zeros(values.shape[0], dtype=np.int8), name=zero_name),
            ]
        )
        dequantize_nodes.append(
            helper.make_node(
                "DequantizeLinear",
                [encoded_name, scale_name, zero_name],
                [initializer.name],
                name=f"{initializer.name}.dequantize",
                axis=0,
            )
        )

    del model.graph.initializer[:]
    model.graph.initializer.extend([*retained, *quantized])
    original_nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend([*dequantize_nodes, *original_nodes])
    onnx.checker.check_model(model)
    return model, len(dequantize_nodes), len(retained)


class _ArrayCalibrationReader:
    def __init__(self, samples: list[np.ndarray]) -> None:
        self.samples = samples
        self.position = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.position >= len(self.samples):
            return None
        sample = self.samples[self.position]
        self.position += 1
        return {"images": sample}

    def rewind(self) -> None:
        self.position = 0


def _load_calibration_samples(
    directory: Path,
    size: tuple[int, int],
    padding_color: tuple[int, int, int],
    scale_up: bool,
    recursive: bool,
    limit: int,
) -> list[np.ndarray]:
    paths = discover_images(directory, recursive)
    if limit > 0:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"No calibration images found under: {directory}")
    samples = []
    for path in paths:
        image, _ = letterbox(read_image(path), size, padding_color, scale_up)
        rgb = image[:, :, ::-1]
        sample = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]).astype(np.float32) / 255.0
        samples.append(sample)
    return samples


def _convert_int8_static(
    source_path: Path,
    target_path: Path,
    samples: list[np.ndarray],
    calibration_method: str,
) -> None:
    import onnx
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
    }
    method_key = calibration_method.strip().lower()
    if method_key not in methods:
        raise ValueError(
            "export.int8_calibration_method must be one of: minmax, entropy, percentile"
        )
    source = onnx.load(str(source_path))
    excluded_nodes = [node.name for node in source.graph.node if "/model/model.23/" in node.name]
    target_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(source_path),
        str(target_path),
        _ArrayCalibrationReader(samples),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=methods[method_key],
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        nodes_to_exclude=excluded_nodes,
        extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
    )


def export_onnx(config_path: str | Path = DEFAULT_CONFIG) -> Path:
    """Convert the configured PyTorch checkpoint to a checked ONNX model."""

    config_path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    model_config = _required(config, "model", "config")
    export_config = _required(config, "export", "config")
    if not isinstance(model_config, dict) or not isinstance(export_config, dict):
        raise ValueError("config.model and config.export must be mappings")

    weights = _resolve_path(_required(model_config, "weights", "model"), config_path)
    output = _resolve_path(_required(export_config, "onnx_output", "export"), config_path)
    fp16_output = _resolve_path(_required(export_config, "fp16_output", "export"), config_path)
    fp8_output = _resolve_path(_required(export_config, "fp8_output", "export"), config_path)
    int8_output = _resolve_path(_required(export_config, "int8_output", "export"), config_path)
    calibration_input = _resolve_path(
        _required(export_config, "int8_calibration_input", "export"), config_path
    )
    height, width = _input_size(_required(model_config, "input_size", "model"))
    opset = int(_required(export_config, "opset_version", "export"))
    fp8_opset = int(_required(export_config, "fp8_opset_version", "export"))
    validate = bool(_required(export_config, "validate", "export"))
    rtol = float(_required(export_config, "validation_rtol", "export"))
    atol = float(_required(export_config, "validation_atol", "export"))
    fp16_rtol = float(_required(export_config, "fp16_validation_rtol", "export"))
    fp16_atol = float(_required(export_config, "fp16_validation_atol", "export"))
    fp8_rtol = float(_required(export_config, "fp8_validation_rtol", "export"))
    fp8_atol = float(_required(export_config, "fp8_validation_atol", "export"))
    int8_rtol = float(_required(export_config, "int8_validation_rtol", "export"))
    int8_atol = float(_required(export_config, "int8_validation_atol", "export"))
    calibration_recursive = bool(
        _required(export_config, "int8_calibration_recursive", "export")
    )
    calibration_limit = int(_required(export_config, "int8_calibration_max_images", "export"))
    calibration_method = str(_required(export_config, "int8_calibration_method", "export"))
    int8_mode = str(_required(export_config, "int8_mode", "export")).strip().lower()
    if int8_mode not in {"weight_only", "static_qdq"}:
        raise ValueError("export.int8_mode must be weight_only or static_qdq")
    strict_quantized = bool(_required(export_config, "strict_quantized_validation", "export"))
    if not 12 <= opset <= 21:
        raise ValueError(f"export.opset_version must be between 12 and 21, got {opset}")

    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "The onnx package is required. Install Interference/OBB/requirements.txt first."
        ) from exc

    device = torch.device("cpu")
    source_model = load_obb_checkpoint(weights, device).float().eval()
    model = ExportableModel(source_model).eval()
    sample = torch.rand((1, 3, height, width), dtype=torch.float32)
    with torch.inference_mode():
        reference = model(sample)
    if not torch.isfinite(reference).all():
        raise FloatingPointError("Source model produced NaN/Inf in FP32 before conversion")

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (sample,),
        str(output),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["predictions"],
        dynamo=False,
    )
    checked = onnx.load(str(output))
    onnx.checker.check_model(checked)

    fp32_validation = _validate_runtime(model, output, sample, rtol, atol) if validate else None
    base_metadata = {
        "format": "onnx",
        "task": "obb",
        "source_weight": str(weights),
        "source_sha256": _sha256(weights),
        "input": {"name": "images", "shape": [1, 3, height, width], "dtype": "float32"},
        "output": {
            "name": "predictions",
            "shape": list(reference.shape),
            "layout": "[batch, 4 box values + class scores + angle values, anchors]",
        },
        "postprocess_included": False,
    }

    metadata = {
        **base_metadata,
        "precision": "fp32",
        "output_model": str(output),
        "output_sha256": _sha256(output),
        "opset_version": opset,
        "validation": fp32_validation,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fp16_model = _convert_fp16(checked)
    _save_checked(fp16_model, fp16_output)
    fp16_validation = (
        _validate_runtime(
            model, fp16_output, sample, fp16_rtol, fp16_atol, strict=strict_quantized
        )
        if validate
        else None
    )
    fp16_metadata = {
        **base_metadata,
        "precision": "fp16",
        "compute_precision": "fp16 with fp32 input/output",
        "output_model": str(fp16_output),
        "output_sha256": _sha256(fp16_output),
        "opset_version": opset,
        "validation": fp16_validation,
    }
    fp16_output.with_suffix(".json").write_text(
        json.dumps(fp16_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fp8_model, quantized_initializers, retained_initializers = _convert_fp8_weights(
        checked, fp8_opset
    )
    _save_checked(fp8_model, fp8_output)
    fp8_validation = (
        _validate_runtime(model, fp8_output, sample, fp8_rtol, fp8_atol, strict=strict_quantized)
        if validate
        else None
    )
    fp8_metadata = {
        **base_metadata,
        "precision": "fp8_weight_only_e4m3fn",
        "compute_precision": "fp32",
        "native_fp8_compute": False,
        "quantized_initializers": quantized_initializers,
        "retained_initializers": retained_initializers,
        "scale_granularity": "per_output_channel",
        "output_model": str(fp8_output),
        "output_sha256": _sha256(fp8_output),
        "opset_version": fp8_opset,
        "validation": fp8_validation,
    }
    fp8_output.with_suffix(".json").write_text(
        json.dumps(fp8_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    calibration_metadata = None
    int8_sample = sample
    if int8_mode == "static_qdq":
        preprocess_config = _required(config, "preprocess", "config")
        padding_color = tuple(
            int(value) for value in _required(preprocess_config, "padding_color", "preprocess")
        )
        if len(padding_color) != 3:
            raise ValueError("preprocess.padding_color must contain three values")
        scale_up = bool(_required(preprocess_config, "scale_up", "preprocess"))
        calibration_samples = _load_calibration_samples(
            calibration_input,
            (height, width),
            padding_color,
            scale_up,
            calibration_recursive,
            calibration_limit,
        )
        _convert_int8_static(output, int8_output, calibration_samples, calibration_method)
        int8_sample = torch.from_numpy(calibration_samples[0])
        calibration_metadata = {
            "input": str(calibration_input),
            "image_count": len(calibration_samples),
            "method": calibration_method.lower(),
        }
        int8_quantized_initializers = None
        int8_retained_initializers = None
    else:
        int8_model, int8_quantized_initializers, int8_retained_initializers = (
            _convert_int8_weights(checked)
        )
        _save_checked(int8_model, int8_output)
    onnx.checker.check_model(onnx.load(str(int8_output)))
    int8_validation = (
        _validate_runtime(
            model, int8_output, int8_sample, int8_rtol, int8_atol, strict=strict_quantized
        )
        if validate
        else None
    )
    int8_metadata = {
        **base_metadata,
        "precision": "int8_static_qdq" if int8_mode == "static_qdq" else "int8_weight_only",
        "compute_precision": "mixed int8/fp32" if int8_mode == "static_qdq" else "fp32",
        "native_int8_compute": int8_mode == "static_qdq",
        "activation_precision": "int8_symmetric" if int8_mode == "static_qdq" else "fp32",
        "weight_precision": "int8_symmetric_per_channel",
        "fp32_scope": "detection_head" if int8_mode == "static_qdq" else "all_compute",
        "quantized_initializers": int8_quantized_initializers,
        "retained_initializers": int8_retained_initializers,
        "calibration": calibration_metadata,
        "output_model": str(int8_output),
        "output_sha256": _sha256(int8_output),
        "opset_version": opset,
        "validation": int8_validation,
    }
    int8_output.with_suffix(".json").write_text(
        json.dumps(int8_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "fp32": metadata,
        "fp16": fp16_metadata,
        "fp8": fp8_metadata,
        "int8": int8_metadata,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return output


if __name__ == "__main__":
    export_onnx()
