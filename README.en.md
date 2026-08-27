[简体中文](README.md) | [繁體中文](README.zh-Traditional.md) | [English](README.en.md)

# Visible Watermark Detection

This repository contains two connected visible-watermark models:

1. **DetectNet**: a recall-oriented image-level screening model based on `MobileNetV3-Large`. It determines whether an image may contain a visible watermark.
2. **OBBNet**: a fine-detection model based on `YOLO11 OBB` oriented bounding boxes. It confirms candidates and returns watermark count, location, orientation, and confidence.

Recommended workflow:

```text
Input image
   │
   ▼
DetectNet screening
   ├── Not a candidate ──► Finish
   │
   └── Watermark candidate
          │
          ▼
       OBBNet detection
          │
          ├── JSON: count, confidence, and oriented-box coordinates
          └── Annotated image: oriented boxes drawn on the original
```

> DetectNet does not provide the final watermark decision. To reduce missed detections, tune its threshold for recall and use OBBNet output as the final model result.

## 1. Repository Layout

```text
water_mark_detect/
├── README.zh-Simplified.md           # Unified documentation (Simplified Chinese)
├── README.zh-Traditional.md          # Unified documentation (Traditional Chinese)
├── README.en.md                      # Unified documentation (English)
├── DetectNet/                        # Stage 1: image-level watermark screening
│   ├── config/config.yaml            # Inference, preprocessing, test, and export settings
│   ├── test/                         # Test images and summary results
│   ├── trainResult/                  # Training records and curves
│   ├── weights/                      # PyTorch and ONNX weights
│   ├── model.py                      # Model definition and PT weight loading
│   ├── preprocess.py                 # Tiling, padding, and normalization
│   ├── pipeline.py                   # Image, directory, and test inference APIs
│   ├── test_pipeline.py              # Test entry point
│   ├── export_onnx.py                # ONNX export
│   └── requirements.txt
└── OBBNet/                           # Stage 2: oriented-box watermark detection
    ├── config/config.yaml            # Inference, preprocessing, test, and export settings
    ├── test/input/                   # Test input
    ├── test/output/                  # JSON and annotated-image output
    ├── trainResult/                  # Training metrics and visualizations
    ├── weight/                       # PyTorch, TorchScript, and ONNX weights
    ├── model.py                      # Standalone model architecture and weight loading
    ├── preprocess.py                 # Letterbox preprocessing
    ├── postprocess.py                # Filtering, oriented-box NMS, and coordinate restoration
    ├── pipeline.py                   # Image, directory, and test inference APIs
    ├── test_pipeline.py              # Automated test entry point
    ├── convert_checkpoint.py         # Standalone TorchScript conversion
    ├── export_onnx.py                # ONNX export
    └── requirements.txt
```

## 2. Installation

Python 3.10 or later is recommended. Because the subprojects have different dependencies, separate virtual environments are recommended; alternatively, install both requirement files in one environment.

### 2.1 DetectNet

```powershell
cd DetectNet
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2.2 OBBNet

```powershell
cd OBBNet
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux/macOS, activate an environment with:

```bash
source .venv/bin/activate
```

Both projects require PyTorch 2.5 or later. For CUDA, install the PyTorch build matching the target CUDA environment. CPU and ONNX inference do not require CUDA.

## 3. Stage 1: DetectNet Screening

### 3.1 Purpose and Scope

DetectNet is a fine-tuned MobileNetV3-Large image-level screening model. It was trained on synthetic watermark data; real-world watermark style, opacity, compression, scaling, screenshot noise, and capture pipelines may differ and produce false negatives or false positives.

Use it to reduce the number of images sent to OBBNet, not as a standalone final decision.

### 3.2 Input and Output

| Item | Description |
|---|---|
| ONNX input | `images` |
| Input shape | `[N, 3, 320, 320]`, where `N` is the tile count |
| Input type | FP32 |
| Channel order | RGB |
| Normalization | Divide by 255, then apply ImageNet mean/std |
| Output | `logits` |
| Output shape | `[N]` |
| Meaning | Unnormalized classification logit for each 320 × 320 tile |

The pipeline applies sigmoid and takes the maximum probability across all tiles:

```text
image_probability = max(sigmoid(tile_logits))
is_watermark = image_probability >= threshold
```

The ONNX model contains only the classifier. Image decoding, tiling, normalization, image-level aggregation, and thresholding must be reproduced outside the model for mobile deployment.

### 3.3 Preprocessing

1. Correct orientation from EXIF and convert to RGB.
2. Create 320 × 320 tiles at native resolution with default stride 240 and 80-pixel overlap.
3. Align the last tile in each direction with the image edge.
4. If width or height is below 320 pixels, do not upscale or stretch; pad only the right and bottom with black.
5. Convert pixels to `[0, 1]`, then normalize with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.

This preserves local textures of translucent watermarks at native scale. Large images create more tiles, so inference time is approximately proportional to image area.

### 3.4 Configuration and Threshold

Configuration: [`DetectNet/config/config.yaml`](DetectNet/config/config.yaml).

```yaml
model:
  weights: ../weights/mobilenet_v3_large.pt
  device: auto

preprocess:
  tile_size: 320
  stride: 240

inference:
  threshold: 0.30
  tile_batch_size: 64
```

- `device: auto` prefers CUDA and otherwise uses CPU.
- Reduce `tile_batch_size` if GPU memory is insufficient.
- Start threshold calibration in the 0.2–0.4 range. Lower values usually improve recall but send more candidates to OBBNet; higher values usually reduce false positives but increase missed-detection risk.
- The configuration threshold takes precedence over any historical threshold in the checkpoint.

Choose the production threshold using independent, real business data excluded from training and tuning.

### 3.5 Inference

Run these commands from `DetectNet`.

Configured test path:

```powershell
python pipeline.py
```

Single image:

```powershell
python pipeline.py --input D:\images\example.jpg --output D:\images\result.json
```

Directory:

```powershell
python pipeline.py --input D:\images --output D:\images\result.json
```

Test entry point:

```powershell
python test_pipeline.py
```

Python API:

```python
from pipeline import DetectionPipeline

detector = DetectionPipeline("config/config.yaml")

one = detector.infer_file("D:/images/example.jpg")
print(one.has_watermark, one.probability)

many = detector.infer_directory("D:/images")
results = detector.run("D:/images", "D:/images/result.json")
test_results = detector.test()
```

The summary JSON records paths, candidate decisions, image probability, threshold, original dimensions, tile count, highest-scoring tile coordinates, elapsed time, and errors.

### 3.6 Test and Training Results

The current [`DetectNet/test/result.json`](DetectNet/test/result.json) is a historical run over 15 images at threshold 0.3: 9 candidates, 6 non-candidates, and 0 errors. The directory now contains more images; rerunning the test overwrites the file, so this snapshot does not represent every current sample.

[`DetectNet/trainResult/training_history.csv`](DetectNet/trainResult/training_history.csv) contains 30 epochs of training and validation metrics:

- Minimum validation loss: 0.3052 at epoch 6.
- Epoch 6: validation F1 ~0.8885, accuracy ~0.887, recall 0.900, specificity 0.874.
- Epoch 30: recall 0.904, specificity 0.848, accuracy 0.876, F1 ~0.8794.
- Training loss continued to fall without matching validation gains, indicating a generalization gap.

![DetectNet training metrics](DetectNet/trainResult/training_curves.png)

### 3.7 Weights and ONNX

The default PyTorch weight is `DetectNet/weights/mobilenet_v3_large.pt`. `DetectNet/weights/onnx/` contains:

| File | Description |
|---|---|
| `mobilenet_v3_large_fp32.onnx` | FP32 baseline |
| `mobilenet_v3_large_fp16.onnx` | FP16 internal computation with FP32 input/output |
| `mobilenet_v3_large_fp8.onnx` | E4M3FN weight storage, restored to FP32 in the graph |
| `mobilenet_v3_large_int8.onnx` | Static QDQ INT8 model |
| Matching `.json`, `export_summary.json` | Source, hash, tensor contract, and numerical validation reports |

Re-export with:

```powershell
python export_onnx.py --config config/config.yaml
```

**Quantization may reduce model quality.** FP8 here primarily compresses weight storage and does not imply native FP8 acceleration. Replace INT8 calibration data with representative real data excluded from final testing. Validate operator compatibility, latency, memory, and image-level recall on target hardware before releasing any low-precision model.

## 4. Stage 2: OBBNet Fine Detection

### 4.1 Purpose and Status

OBBNet detects and annotates watermarks in images selected by DetectNet.

- Default input: 960 × 960 with fixed batch 1
- Default weight: `OBBNet/weight/best.pt`
- Supports individual images and directories
- Supports JPG, JPEG, PNG, BMP, WebP, and TIFF
- Produces one prediction JSON and one annotated image per input
- Current class: `watermark` only

### 4.2 Input and Raw Output

| Item | Description |
|---|---|
| ONNX input | `images` |
| Input shape | `[1, 3, 960, 960]` |
| Input type | FP32; PyTorch CUDA can optionally try FP16 |
| Channel order | RGB |
| Value range | `[0, 1]` |
| Class | `0: watermark` |
| Raw output shape | `[1, 6, 18900]` |

The six values are:

```text
center_x, center_y, width, height, watermark_score, angle_radians
```

The raw tensor has not undergone confidence filtering, oriented-box NMS, or coordinate restoration and is not a final annotation result.

### 4.3 Preprocessing and Postprocessing

Preprocessing:

1. Decode with OpenCV.
2. Preserve aspect ratio and fit the complete image within 960 × 960 without cropping or stretching.
3. Center-pad with BGR `[114, 114, 114]`.
4. Convert BGR to RGB, HWC to CHW, add the batch dimension, and divide pixels by 255.

```text
scale = min(960 / original_width, 960 / original_height)
```

Default postprocessing:

1. Remove candidates with confidence at or below 0.25.
2. If necessary, retain the top 3,000 candidates by score.
3. Apply probabilistic-IoU oriented-box NMS at IoU threshold 0.45.
4. Retain at most 300 detections per image.
5. Convert center, width, height, and angle into four corners.
6. Remove Letterbox padding, divide by scale, and map coordinates back to the source image.
7. Output pixel and normalized corners, center, size, radians/degrees, and an annotated image.

### 4.4 Final JSON

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

`obb_label` contains `class_id + 4 normalized corners`, for nine values total. Results also record weights, device, thresholds, scale, padding, and model inference time.

### 4.5 Configuration

Configuration: [`OBBNet/config/config.yaml`](OBBNet/config/config.yaml).

| Section | Purpose |
|---|---|
| `model` | Weights, device, input size, FP16 switch, and class names |
| `inference` | Confidence threshold, oriented-box NMS threshold, and maximum detections |
| `preprocess` | Padding color and small-image upscaling |
| `visualization` | Line width, font, colors, and JPEG quality |
| `test` | Test input/output and recursive scanning |
| `export` | ONNX output, opset, quantization, calibration, and numerical validation |

Relative paths are resolved from the configuration directory. You may lower the confidence threshold to evaluate recall when real-world missed detections are high, but determine the production value on independent real data.

### 4.6 Inference and Tests

Run from `OBBNet`.

```powershell
python pipeline.py
```

Single image:

```python
from pipeline import OBBInferencePipeline

detector = OBBInferencePipeline("config/config.yaml")
result = detector.predict_file(
    "test/input/example.jpg",
    output_dir="test/output",
)
print(result["watermarks"])
```

Directory:

```python
from pipeline import OBBInferencePipeline

detector = OBBInferencePipeline("config/config.yaml")
results = detector.predict_folder(
    input_dir="path/to/images",
    output_dir="path/to/output",
    recursive=False,
)
```

Automated tests:

```powershell
python -m unittest test_pipeline.py
```

Tests cover weight loading, TorchScript validation, large/small-image Letterbox behavior, oriented-box corner conversion, and configured end-to-end inference. The end-to-end test overwrites matching artifacts in `test/output/`.

### 4.7 Weights and Export

| File | Format and purpose |
|---|---|
| `best.pt` | Default PyTorch checkpoint; load only trusted files |
| `best_standalone.pt` | Standalone TorchScript inference weight |
| `best.onnx` | FP32 baseline ONNX with fixed `[1,3,960,960]` input |
| `best_fp16.onnx` | FP16 graph with FP32 input/output |
| `best_fp8.onnx` | FP8 E4M3FN weight storage restored to FP32 at runtime |
| `best_int8.onnx` | INT8 weight storage restored to FP32 at runtime |
| `best*.json` | Source SHA-256, tensor contract, precision mode, and numerical validation |

```powershell
python convert_checkpoint.py
python export_onnx.py
```

The current pipeline loads `.pt` files. ONNX files are for ONNX Runtime, mobile conversion, or hardware compilers and cannot directly replace `model.weights`. FP8/INT8 defaults are storage quantization and do not imply native low-precision acceleration.

### 4.8 Training Metrics

Training ran for 60 epochs; raw data is in [`OBBNet/trainResult/results.csv`](OBBNet/trainResult/results.csv).

| Metric | Best value | Epoch |
|---|---:|---:|
| Precision | 0.98469 | 30 |
| Recall | 0.95558 | 60 |
| mAP50 | 0.98048 | 40 |
| mAP50-95 | 0.89379 | 60 |

The validation confusion matrix records 981 correct watermark detections, 53 background false positives, and 38 missed watermarks. These validation-set metrics do not replace evaluation on real business data.

| Confusion matrix | Loss trends |
|---|---|
| ![OBBNet confusion matrix](OBBNet/trainResult/confusion_matrix.png) | ![OBBNet loss trends](OBBNet/trainResult/training_loss_trends.png) |

![OBBNet training metrics](OBBNet/trainResult/results.png)

| Validation labels | Predictions |
|---|---|
| ![Validation labels](OBBNet/trainResult/val_batch0_labels.jpg) | ![Validation predictions](OBBNet/trainResult/val_batch0_pred.jpg) |

## 5. Recommended Two-Stage Integration

The subprojects currently provide independent pipelines and no unified end-to-end orchestrator. For integration:

1. Screen input images with DetectNet.
2. Send only images with `has_watermark = true` to OBBNet.
3. Treat OBBNet `detection_count` and `watermarks` as the final machine-detection result.
4. Retain manual review or annotation for high-risk images not detected by OBBNet.
5. Record latency, thresholds, and model versions for both stages to diagnose quality and performance issues.

The preprocessing contracts differ and the models cannot share the same input tensor:

| Item | DetectNet | OBBNet |
|---|---|---|
| Task | Image-level candidate screening | Watermark localization and confirmation |
| Input strategy | Native-scale 320 × 320 tiles | Aspect-ratio resize and padding to 960 × 960 |
| Normalization | ImageNet mean/std | Divide by 255 only |
| Batch | Dynamic tile count `N` | Fixed at 1 |
| Result | Image probability and candidate decision | Count, confidence, and oriented boxes |

## 6. Pre-release Checklist

- Evaluate independent real data covering low-opacity, small, multiple, edge, rotated, platform-logo, timestamp, text-heavy-background, and differently compressed watermarks; fine-tune if required.
- Tune DetectNet for recall while measuring the fraction of images entering OBBNet and its throughput load.
- Evaluate OBBNet Precision, Recall, mAP, image-level missed-detection rate, and manual annotation cost.
- Reproduce both preprocessing/postprocessing pipelines on target hardware and test large-image latency and peak memory.
- Run end-to-end regression for FP32, FP16, FP8, and INT8; successful loading is not enough.
- Validate quantized models on the actual NNAPI, Core ML, ONNX Runtime Mobile, or vendor NPU backend.
- `.pt` loading uses pickle; load weights only from trusted sources.

## 7. Known Limitations and Licensing

1. Both models were trained primarily on synthetic or limited real data. Training/validation metrics do not equal performance on real internet data.
2. Low-opacity, very small, irregularly placed, overlapping, or subtitle-like watermarks may still be missed or misclassified. Retain manual review and annotation, and fine-tune on real data at a low learning rate when needed.
3. OBBNet was retrained from YOLO11m-OBB, and its training environment used Ultralytics and related components. The inference stack has been reimplemented in PyTorch without a runtime Ultralytics dependency, but the project owner must still review licensing obligations for the models, training code, and weights before commercial use.

## 8. Detailed Subproject Documentation

This README is the unified entry point. See the language-matched subproject documents for details:

- [`DetectNet/README.en.md`](DetectNet/README.en.md)
- [`OBBNet/README.en.md`](OBBNet/README.en.md)
