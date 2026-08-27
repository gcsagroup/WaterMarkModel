[简体中文](README.zh-Simplified.md) | [繁體中文](README.zh-Traditional.md) | [English](README.en.md)

# Visible Watermark Screening Model

## 1. Purpose and Scope

This project fine-tunes MobileNetV3-Large to perform **preliminary screening** for visible watermarks. Its output indicates whether an image is a potential watermark candidate; it is not a strict or final determination.

Images selected as candidates must still be passed to the OBB watermark detector, which makes the final determination and reports whether watermarks exist, their count, and the position and orientation of each watermark.

The model was trained on synthetic watermark data. Real images may differ in watermark style, opacity, compression, scaling, screenshot noise, and capture pipeline, so false negatives and false positives remain possible. Do not use screening results as the final business decision.

## 2. Directory Structure

```text
Detect/
├── config/
│   └── config.yaml                    # Inference, preprocessing, test, and export settings
├── test/
│   ├── *.png                          # Test images shipped with the project
│   └── result.json                    # Test summary
├── trainResult/
│   ├── training_history.csv           # Raw metrics from 30 training epochs
│   └── training_curves.png            # Training metric visualization
├── weights/
│   ├── mobilenet_v3_large.pt           # PyTorch V3 weights
│   └── onnx/                           # ONNX exports and metadata
├── model.py                            # Model definition and PT weight loading
├── preprocess.py                       # Native-scale tiling and normalization
├── pipeline.py                         # Image, directory, and test inference APIs
├── test_pipeline.py                    # One-command test runner
├── export_onnx.py                      # FP16, FP8, and INT8 ONNX conversion
├── settings.py                         # YAML loading and validation
└── requirements.txt
```

No open-source license is included. Before publishing the repository, the project owner should add an appropriate `LICENSE` and verify distribution rights for training data, test images, and weights. Large `.pt` and `.onnx` files should be managed with Git LFS; matching `.gitattributes` rules are already provided.

## 3. Installation

The project has been validated with PyTorch 2.13, TorchVision 0.28, ONNX 1.22, and ONNX Runtime 1.29.

```powershell
python -m pip install -r requirements.txt
```

For CUDA, install a PyTorch build that matches the target CUDA environment. CUDA is not required for CPU inference or mobile ONNX deployment.

## 4. Model Input and Output

### 4.1 Input

The PyTorch/ONNX model receives:

- Tensor name: `images`
- Shape: `[N, 3, 320, 320]`
- Channel order: RGB
- Data type: FP32
- Value processing: divide by 255, then normalize using ImageNet mean/std

`N` is the number of tiles in a model batch. The ONNX graph contains only the classifier; image decoding, tiling, normalization, and image-level aggregation must be implemented outside the model on mobile platforms.

### 4.2 Output

- Tensor name: `logits`
- Shape: `[N]`
- Meaning: unnormalized classification logit for each 320 × 320 tile

The pipeline applies sigmoid to every logit and uses the maximum tile probability as the image probability:

```text
image_probability = max(sigmoid(tile_logits))
is_watermark = image_probability >= threshold
```

Maximum aggregation prioritizes recall: if any local region exhibits watermark features, the image proceeds to the OBB stage.

## 5. Input Preprocessing

Preprocessing matches the V3 training strategy and does not force arbitrary aspect ratios into 320 × 320:

1. Read the image, correct orientation using EXIF, and convert it to RGB.
2. If either dimension exceeds 320 pixels, create 320 × 320 tiles at native resolution. The default stride is 240, producing an 80-pixel overlap.
3. Align the last tile in each direction with the image edge to avoid missing the right or bottom boundary.
4. If either dimension is below 320 pixels, do not upscale or stretch; pad only the right and bottom with black.
5. Convert pixels to `[0, 1]`, then normalize with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.

This preserves local textures of translucent watermarks at their original scale. Larger images produce more tiles, so inference time is approximately proportional to image area.

## 6. Postprocessing and Threshold

The recommended starting range is **0.2–0.4**, configured by `inference.threshold` in `config/config.yaml`. The configuration value takes precedence over any historical threshold stored in the checkpoint.

- A higher threshold is stricter and generally reduces false positives, but may increase missed detections.
- A lower threshold generally improves recall, but reduces precision and sends more images to the OBB stage.
- When missed detections are unacceptable, evaluate thresholds below 0.3 on an independent real-world validation set and account for OBB throughput.

Recalibrate the threshold on actual business data rather than relying only on a synthetic validation set. Because this model is a screening stage, keep it low enough to prioritize **recall**.

## 7. Configuration

All inference settings come from `config/config.yaml`, including weights, device, tile size, stride, normalization, threshold, tile batch size, test paths, and ONNX export options. Relative paths are resolved from the directory containing `config.yaml`.

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

`device: auto` prefers CUDA and falls back to CPU. Reduce `tile_batch_size` if GPU memory is insufficient.

## 8. Inference

### 8.1 Command Line

With no input argument, the pipeline runs the built-in test using `test.input` and `test.result` from YAML:

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

### 8.2 Python API

```python
from pipeline import DetectionPipeline

detector = DetectionPipeline("config/config.yaml")

# One image: returns DetectionResult
one = detector.infer_file("D:/images/example.jpg")
print(one.has_watermark, one.probability)

# Directory: returns a list of DetectionResult values
many = detector.infer_directory("D:/images")

# Detect image/directory automatically, write JSON, and return the result list
results = detector.run("D:/images", "D:/images/result.json")

# Use the test paths in config.yaml and return the result list
test_results = detector.test()
```

Each `result.json` record includes the image path, candidate decision, image probability, threshold, original dimensions, tile count, highest-scoring tile coordinates, elapsed time, and error details. Top-level fields include the model, weights, summary counts, and runtime configuration.

## 9. Test Set and Current Results

The `test` directory contains 15 images shipped with the project. Running `python test_pipeline.py` overwrites `test/result.json`.

Current results with `mobilenet_v3_large.pt` at threshold 0.3:

- Total images: 15
- Watermark candidates: 9
- Non-candidates: 6
- Read/inference errors: 0

The test directory has no independent ground-truth labels. These values describe only the output distribution and pipeline acceptance result; they do not represent accuracy, recall, or specificity.

## 10. Weight Files

`weights/mobilenet_v3_large.pt` is the DetectV3 MobileNetV3-Large PyTorch checkpoint. It includes a model-structure identifier, parameters, epoch number, and training/validation metadata, and currently records epoch 30. The pipeline loads `model_state_dict` strictly and fails on structural mismatches.

Exports in `weights/onnx` include:

- `mobilenet_v3_large_fp32.onnx`: FP32 baseline
- `mobilenet_v3_large_fp16.onnx`: FP16 internal computation with FP32 input/output
- `mobilenet_v3_large_fp8.onnx`: per-output-channel E4M3FN weight storage, restored to FP32 in the graph
- `mobilenet_v3_large_int8.onnx`: static QDQ INT8 model calibrated with tiles from `test`
- Matching `.json` files: hashes, I/O contracts, source weights, and numerical validation
- `export_summary.json`: summary of all exports

Approximate sizes are 16.03 MiB (FP32), 8.05 MiB (FP16), 4.18 MiB (FP8), and 4.51 MiB (INT8). FP32, FP16, and INT8 passed the current ONNX Runtime post-export validation. FP8 passes ONNX structural checks and runs, but its sample logit error exceeds the configured tolerance, so it remains experimental and must be revalidated on target hardware and real data.

FP8 operator support varies across mobile runtimes. This export primarily compresses weight storage and does not imply native FP8 acceleration. The current INT8 calibration uses shipped test images only to demonstrate the conversion pipeline. For release, point `export.calibration_input` to representative, compliant real calibration data excluded from the final test set, then export again and evaluate recall.

## 11. ONNX Conversion

```powershell
python export_onnx.py --config config/config.yaml
```

The conversion process:

1. Rebuilds MobileNetV3-Large from the PT weight specified in YAML.
2. Exports the FP32 baseline ONNX model.
3. Produces FP16, FP8 weight-storage, and static INT8 QDQ variants.
4. Performs ONNX structural checks and ONNX Runtime numerical comparisons.
5. Writes models, per-model metadata, and a summary to `weights/onnx`.

Quantization support differs by mobile chipset and runtime. Before release, test operator compatibility, end-to-end latency, memory use, and image-level recall on the actual NNAPI, Core ML, ONNX Runtime Mobile, or vendor NPU backend. Mobile integrations must reproduce tiling, padding, normalization, sigmoid, maximum aggregation, and thresholding outside the model.

## 12. Training Results

`trainResult/training_history.csv` contains training and validation metrics for 30 epochs:

- Total training loss fell from 1.3311 to 0.1863, with a minimum of 0.1855 at epoch 28.
- Image-level training loss fell from 0.5903 to 0.0046.
- Hard-negative loss fell from 0.4380 to 0.0003.
- Ranking loss reached 0 at epoch 5.
- Auxiliary loss fell from 0.7264 to 0.3632.
- Minimum validation loss was 0.3052 at epoch 6.
- Epoch 6 also produced the highest validation F1 (~0.8885), accuracy (~0.887), recall (0.900), and specificity (0.874), with a selected threshold of 0.34.
- Epoch 30 recorded validation loss 0.3583, threshold 0.17, recall 0.904, specificity 0.848, accuracy 0.876, and F1 ~0.8794.

Training loss continued to fall while validation performance did not improve correspondingly, indicating a generalization gap. The deployment configuration uses the recommended threshold 0.3 rather than copying an epoch-specific automatic threshold, and must still be recalibrated on real business data.

![MobileNetV3-Large training metrics](trainResult/training_curves.png)

## 13. Pre-release Checklist

- Evaluate low-opacity, small, multiple, timestamp, platform-logo, text-heavy-background, and differently compressed watermarks separately.
- Select the threshold for missed-detection priority rather than accuracy alone.
- Reproduce native-scale tiling on target phones and measure large-image batch latency and peak memory.
- Run end-to-end regression tests for FP16, FP8, and INT8; successful model loading alone is insufficient.
- Retain the second-stage OBB confirmation so screening false positives are not treated as final watermark results.
