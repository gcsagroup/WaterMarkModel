[简体中文](README.zh-Simplified.md) | [繁體中文](README.zh-Traditional.md) | [English](README.en.md)

# Visible Watermark OBB Detector

> This directory provides standalone PyTorch inference, testing, weight conversion, and ONNX export for a visible-watermark Oriented Bounding Box (OBB) detector.</br>
> The model has one class, `watermark`. Screen images with `Detect` first, then send candidate images to this `OBB` model for detection and annotation to reduce compute usage. Treat OBB output as the final model result.

## 1. Project Status

- Default input size: `960 × 960`; batch size is fixed at `1`.
- Default inference weight: `weight/best.pt`.
- Accepts a single image or a directory.
- Supports JPG, JPEG, PNG, BMP, WebP, and TIFF.
- Produces one OBB prediction JSON and one annotated image per input image.
- Inference depends only on PyTorch, OpenCV, NumPy, and PyYAML; the training framework is not needed for compatible weights.
- FP32, FP16, FP8 weight-storage, and INT8 weight-storage ONNX files are included.

## 2. Directory Structure

```text
OBB/
├── README.zh-Simplified.md    # Handover documentation (Simplified Chinese)
├── README.zh-Traditional.md   # Handover documentation (Traditional Chinese)
├── README.en.md               # Handover documentation (English)
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
├── .gitattributes            # Binary-file attributes
├── __init__.py               # Python package entry point
├── config/
│   └── config.yaml           # Inference, test, and export configuration
├── weight/                   # PyTorch, TorchScript, and ONNX weights and reports
├── trainResult/              # Training metrics, loss curves, and batch visualizations
├── test/
│   ├── input/                # Test images
│   └── output/               # Generated inference results; not committed
├── model.py                  # Standalone model architecture and weight loading
├── preprocess.py             # Decode, aspect-ratio resize, padding, and tensor conversion
├── postprocess.py            # Confidence filtering, oriented-box NMS, and coordinates
├── pipeline.py               # Image, directory, and configured test APIs
├── test_pipeline.py          # Unit tests and batch test entry point
├── convert_checkpoint.py     # Convert original weights to standalone TorchScript
└── export_onnx.py            # FP32/FP16/FP8/INT8 ONNX export
```

For compatibility with existing callers, entry paths such as `pipeline.py` and `config/config.yaml` have not changed. Existing directory names such as `trainResult` and `Interference` are also retained.

## 3. Installation

Python 3.10 or later is recommended. From this directory:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

ONNX dependencies are optional when running only `.pt` inference. Installing the full `requirements.txt` enables all features.

## 4. Model Input and Output

### 4.1 Input Contract

| Item | Description |
|---|---|
| Input name | `images` (ONNX) |
| Input shape | `[1, 3, 960, 960]`: batch, channels, height, width |
| Data type | FP32; PyTorch CUDA inference can optionally try FP16 |
| Channel order | RGB |
| Value range | `[0, 1]` |
| Class | `0: watermark` |

Input size is controlled by `model.input_size` in [`config/config.yaml`](config/config.yaml). Width and height must be positive multiples of 32. Current ONNX exports use fixed input dimensions and batch size.

### 4.2 Raw Model Output

The single-class model produces:

```text
[1, 6, 18900]
```

The six values in the second dimension are:

```text
center_x, center_y, width, height, watermark_score, angle_radians
```

- `18900` is the total candidate-point count across three feature-map scales.
- Center and size use the preprocessed 960 × 960 coordinate system.
- The angle branch produces an angle in approximately `[-π/4, 3π/4]`.
- The raw tensor has not undergone thresholding, oriented-box NMS, or source-coordinate restoration and is not a final annotation result.

### 4.3 Final JSON Output

```json
{
  "image": {
    "file_name": "example.jpg",
    "width": 1920,
    "height": 1080
  },
  "annotation_type": "obb_prediction",
  "corner_order": "clockwise",
  "detection_count": 1,
  "watermarks": [
    {
      "id": 0,
      "class_id": 0,
      "class_name": "watermark",
      "confidence": 0.95,
      "obb": {
        "corners_pixel": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "corners_normalized": [[0, 0], [0, 0], [0, 0], [0, 0]],
        "center_pixel": [0, 0],
        "width_pixel": 0,
        "height_pixel": 0,
        "angle_radians": 0,
        "angle_degrees": 0
      },
      "obb_label": [0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
  ]
}
```

`obb_label` contains `class_id + 4 normalized corners`, for nine values total. JSON also records the weight path, runtime device, thresholds, scale, padding, and per-image model inference time for diagnostics.

## 5. Input Preprocessing

Preprocessing is implemented in [`preprocess.py`](preprocess.py):

1. Decode with OpenCV; the in-memory channel order is initially BGR.
2. Preserve aspect ratio and resize to the largest dimensions that fit completely within `960 × 960`, without stretching or cropping.
3. Center-pad vertically or horizontally with BGR `[114, 114, 114]`.
4. Use `INTER_AREA` when downscaling and `INTER_LINEAR` when upscaling.
5. Convert BGR to RGB, HWC to CHW, and add the batch dimension.
6. Convert to FP32/FP16 and divide pixels by 255 to obtain `[0, 1]`.

```text
scale = min(960 / original_width, 960 / original_height)
```

When `preprocess.scale_up: true`, small images may be enlarged. With `false`, they are only center-padded. This matches the aspect-ratio resize and gray-padding strategy used during training.

## 6. Output Postprocessing

Postprocessing is implemented in [`postprocess.py`](postprocess.py) and [`pipeline.py`](pipeline.py):

1. Read class confidence and remove candidates with `confidence ≤ 0.25`.
2. If too many candidates remain, retain the highest-scoring `3000`.
3. Apply probabilistic-IoU oriented-box NMS with default IoU threshold `0.45`.
4. Retain at most `300` detections per image.
5. Convert center, width, height, and angle into four oriented-box corners.
6. Subtract Letterbox padding and divide by scale to map coordinates back to the source image.
7. Generate pixel corners, normalized corners, center, size, radians, and degrees.
8. Draw the oriented box, class name, and confidence on a copy of the source image.

Change the confidence threshold, NMS threshold, and maximum detection count under `inference` in [`config/config.yaml`](config/config.yaml). If real-world missed detections are high, lower the confidence threshold for recall evaluation, but determine the production value on an independent real-world validation set.

## 7. Weight Files

Files under `weight/`:

| File | Precision/format | Purpose and notes |
|---|---|---|
| `best.pt` | PyTorch checkpoint, ~40.50 MiB | Default weight with original model object; load only trusted files |
| `best_standalone.pt` | TorchScript, ~41.18 MiB | Standalone inference weight generated by `convert_checkpoint.py` |
| `best.onnx` | ONNX FP32, ~79.97 MiB | Baseline deployment model with fixed `[1,3,960,960]` input |
| `best_fp16.onnx` | ONNX FP16, ~40.08 MiB | FP16 graph with FP32 I/O; runtime must support FP16 operators |
| `best_fp8.onnx` | ONNX FP8 E4M3FN weights, ~20.46 MiB | Per-output-channel storage quantization restored to FP32 at runtime |
| `best_int8.onnx` | ONNX INT8 weights, ~20.44 MiB | Per-output-channel storage quantization restored to FP32 at runtime |
| `best*.json` | JSON | Source SHA-256, tensor contract, precision mode, and numerical validation |

Notes:

- `best.json` is the export report for `best.onnx`, not a label file for `best.pt`.
- [`config/config.yaml`](config/config.yaml) points to `best.pt` by default. To use TorchScript, set `model.weights` to `../weight/best_standalone.pt`.
- The current pipeline loads `.pt`; ONNX files target ONNX Runtime, mobile converters, or hardware compilers and cannot directly replace `model.weights`.
- FP8/INT8 defaults use weight-storage quantization, not end-to-end low-precision computation. Setting `export.int8_mode` to `static_qdq` uses calibration images for mixed static quantization; reevaluate mAP and missed detections.
- These weights may exceed GitHub's single-file web upload limit. Use Git LFS for `*.pt` and `*.onnx`, or publish them in a Release with checksums in the README.

Generate standalone weights:

```bash
python convert_checkpoint.py
```

Export all ONNX variants:

```bash
python export_onnx.py
```

## 8. Training Results

Training ran for 60 epochs and took approximately `18144.1 s` (about 5 hours 2 minutes). Per-epoch training/validation losses, learning rates, and detection metrics are stored in [`trainResult/results.csv`](trainResult/results.csv), which is the authoritative raw record.

### 8.1 Metric Summary

| Metric | Best value | Epoch |
|---|---:|---:|
| Precision | 0.98469 | 30 |
| Recall | 0.95558 | 60 |
| mAP50 | 0.98048 | 40 |
| mAP50-95 | 0.89379 | 60 |

Epoch 60 metrics:

| train/box | train/cls | train/dfl | train/angle | val/box | val/cls | val/dfl | val/angle | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.41093 | 0.31891 | 1.01352 | 0.01248 | 0.41423 | 0.32093 | 1.05277 | 0.01842 | 0.97692 | 0.95558 | 0.98012 | 0.89379 |

### 8.2 Confusion Matrix

The matrix records 981 correct watermark detections, 53 background false positives, and 38 missed watermarks. It comes from the training pipeline's validation set and does not replace real-world business evaluation.

![Confusion matrix](trainResult/confusion_matrix.png)

### 8.3 Loss Trends

Training and validation box, cls, dfl, and angle losses generally converge. Validation angle loss stabilizes around 0.018 late in training while training loss continues to fall; real-world domain shift must still be evaluated.

![Loss trends](trainResult/training_loss_trends.png)

### 8.4 Combined Metrics

This chart shows losses, Precision, Recall, mAP, and learning rate over time.

![Combined training metrics](trainResult/results.png)

### 8.5 Training-label Visualization

Use these images to inspect Letterbox behavior, oriented-box direction, class IDs, and multi-object annotations.

| Train batch 0 | Train batch 1 |
|---|---|
| ![Train batch 0](trainResult/train_batch0.jpg) | ![Train batch 1](trainResult/train_batch1.jpg) |

### 8.6 Validation Labels and Predictions

`labels` are manual/generated labels and `pred` contains model predictions. Review them in pairs, focusing on translucent, small, edge, rotated, and adjacent watermarks.

| Validation labels, batch 0 | Predictions, batch 0 |
|---|---|
| ![Validation labels 0](trainResult/val_batch0_labels.jpg) | ![Validation predictions 0](trainResult/val_batch0_pred.jpg) |

| Validation labels, batch 1 | Predictions, batch 1 |
|---|---|
| ![Validation labels 1](trainResult/val_batch1_labels.jpg) | ![Validation predictions 1](trainResult/val_batch1_pred.jpg) |

## 9. Test Set and Usage

### 9.1 Test Directories

- `test/input/`: 40 synthetic, natural, and real-source images for smoke tests and qualitative review.
- `test/output/`: generated JSON and annotated images; excluded from Git by default.
- Test images are for engineering validation only. Confirm copyright, privacy, and data authorization before publishing them.

Each input produces:

```text
test/output/<original-name>.json
test/output/<original-name>_annotated.<original-extension>
```

### 9.2 Run Configured Test Images

```bash
python pipeline.py
```

The program reads `config.test.input` and writes to `config.test.output`. `config.test.recursive` controls recursive scanning.

You can also call the test function directly:

```python
from test_pipeline import run_configured_input_test

results = run_configured_input_test()
print(f"Processed {len(results)} images")
```

### 9.3 Single-image Inference

```python
from pipeline import OBBInferencePipeline

detector = OBBInferencePipeline("config/config.yaml")
result = detector.predict_file(
    "test/input/example.jpg",
    output_dir="test/output",
)
print(result["watermarks"])
```

### 9.4 Directory Inference

```python
from pipeline import OBBInferencePipeline

detector = OBBInferencePipeline("config/config.yaml")
results = detector.predict_folder(
    input_dir="path/to/images",
    output_dir="path/to/output",
    recursive=False,
)
```

### 9.5 Automated Tests

```bash
python -m unittest test_pipeline.py
```

Tests cover weight loading, standalone TorchScript validation, Letterbox behavior for large/small images, oriented-box corner conversion, and end-to-end inference over configured test images. The last test overwrites matching files in `test/output/`; runtime depends on hardware and image count.

## 10. Configuration

Key parameters in [`config/config.yaml`](config/config.yaml):

| Section | Purpose |
|---|---|
| `model` | Weight path, device, input size, FP16 switch, and class names |
| `inference` | Confidence threshold, oriented-box NMS threshold, and maximum detections |
| `preprocess` | Padding color and whether small images may be enlarged |
| `visualization` | Line width, font, colors, and JPEG quality |
| `test` | Test input/output and recursive scanning |
| `export` | ONNX output, opset, quantization, calibration, and numerical validation |

Relative paths are resolved from the directory containing `config/config.yaml`, not the caller's working directory.

## 11. Known Issues

1. The model performs well on synthetic watermark data, reaching approximately 90% mAP50-95 on the known distribution, but limited real-world data yields only mAP50-95 ≥ 70% on real internet images.
2. It may fail to annotate irregularly positioned, very small, overlapping, very low-opacity, or very high-opacity watermarks (the latter can be classified as subtitles). Missed detections remain possible, so end users need a manual annotation option.

## 12. Important Notes

1. Training/validation metrics mainly describe the current data distribution. Before release, measure Precision, Recall, mAP, and image-level missed-detection rate on independent real data excluded from tuning.
2. Watermark detection usually prioritizes missed-detection control. Threshold changes must also account for false positives and downstream processing cost.
3. Validate FP16, FP8, and INT8 operator support, latency, memory, and accuracy on the target NPU/GPU runtime. Smaller files do not necessarily run faster.
4. `.pt` uses pickle-based loading; load only trusted weights. Prefer ONNX/Core ML/TFLite/vendor formats validated by the target toolchain for mobile deployment.
5. The model was retrained from YOLO11m-OBB. Its training environment used Ultralytics and related components, while this project reimplements inference in PyTorch and removes the Ultralytics runtime dependency.
