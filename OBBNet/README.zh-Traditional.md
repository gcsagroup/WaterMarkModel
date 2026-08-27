[簡體中文](README.zh-Simplified.md) | [繁體中文](README.zh-Traditional.md) | [English](README.en.md)

# 可見水印 OBB 檢測模型

> 本目錄提供可見水印旋轉目標框（Oriented Bounding Box，OBB）檢測模型的獨立 PyTorch 推理、測試、權重轉換及 ONNX 導出能力。</br>
> 當前模型的輸入僅包含一個類別：`watermark`，即應使用 `Detect` 模型對圖像進行一次初篩后，將識別為 `watermark` 的圖像再輸入當前 `OBB` 模型進行水印識別與標注，以降低算力消耗，最終結果以 `OBB` 模型輸出為準。

## 1. 項目狀態

- 默認輸入尺寸：`960 × 960`，批量大小固定為 `1`。
- 默認推理權重：`weight/best.pt`。
- 支持輸入：單張圖片或圖片文件夾。
- 支持圖片格式：JPG、JPEG、PNG、BMP、WebP、TIFF。
- 輸出內容：每張圖片對應一份 OBB 預測 JSON 和一張繪制旋轉框的結果圖。
- 推理代碼只依賴 PyTorch、OpenCV、NumPy 和 PyYAML；加載兼容權重時不需要安裝訓練框架。
- 已提供 FP32、FP16、FP8 權重存儲量化和 INT8 權重存儲量化 ONNX 文件。

## 2. 目錄結構

```text
OBB/
├── README.zh-Simplified.md    # 本交接文檔（簡體中文）
├── README.zh-Traditional.md   # 本交接文檔（繁體中文）
├── README.en.md               # 本交接文檔（英文）
├── requirements.txt          # Python 依賴
├── .gitignore                # Git 忽略規則
├── .gitattributes            # 二進制文件屬性
├── __init__.py               # Python 包入口
├── config/
│   └── config.yaml           # 推理、測試與導出配置
├── weight/                   # PyTorch、TorchScript、ONNX 權重及校驗報告
├── trainResult/              # 訓練指標、損失曲線和批次可視化
├── test/
│   ├── input/                # 測試圖片
│   └── output/               # 推理結果，不提交到 Git
├── model.py                  # 獨立模型結構和權重加載邏輯
├── preprocess.py             # 讀圖、等比例縮放、填充和張量轉換
├── postprocess.py            # 置信度過濾、旋轉框 NMS 和坐標轉換
├── pipeline.py               # 單圖、文件夾及配置化測試接口
├── test_pipeline.py          # 單元測試和測試集批量推理入口
├── convert_checkpoint.py     # 原始權重轉獨立 TorchScript 權重
└── export_onnx.py            # FP32/FP16/FP8/INT8 ONNX 導出
```

為保持既有調用方兼容，本次沒有更改 `pipeline.py`、`config/config.yaml` 等入口路徑，也保留了既有的 `trainResult` 和 `Interference` 目錄名稱。

## 3. 環境安裝

建議使用 Python 3.10 或更高版本。進入本目錄后執行：

```bash
python -m venv .venv
```

Windows：

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

若僅運行 `.pt` 推理，可以不安裝 ONNX 相關依賴；按當前 `requirements.txt` 完整安裝可直接使用全部功能。

## 4. 模型輸入與輸出

### 4.1 輸入契約

| 項目 | 說明 |
|---|---|
| 輸入名稱 | `images`（ONNX） |
| 輸入形狀 | `[1, 3, 960, 960]`，依次為 batch、通道、高、寬 |
| 數據類型 | FP32；PyTorch CUDA 推理可按配置嘗試 FP16 |
| 色彩順序 | RGB |
| 數值范圍 | `[0, 1]` |
| 類別 | `0: watermark` |

輸入尺寸由 [`config/config.yaml`](config/config.yaml) 中的 `model.input_size` 控制，寬高必須為 32 的正整數倍。當前導出的 ONNX 文件使用固定輸入尺寸和固定 batch。

### 4.2 模型原始輸出

當前單類別模型的原始輸出形狀為：

```text
[1, 6, 18900]
```

第二維的 6 個值依次表示：

```text
center_x, center_y, width, height, watermark_score, angle_radians
```

- `18900` 是三個尺度特征圖展開后的候選點總數。
- 中心點和寬高位于預處理后的 `960 × 960` 坐標系。
- 角度由模型角度分支給出，模型內部范圍約為 `[-π/4, 3π/4]`。
- 原始張量尚未執行閾值過濾、旋轉框 NMS 或原圖坐標還原，不能直接作為最終標注結果。

### 4.3 最終 JSON 輸出

后處理后的 JSON 主要字段如下：

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

其中 `obb_label` 為 `class_id + 4 個歸一化角點`，共 9 個數。JSON 還會記錄權重路徑、運行設備、閾值、縮放比例、填充量和單次模型推理耗時，便于問題追蹤。

## 5. 輸入預處理

預處理實現在 [`preprocess.py`](preprocess.py)，流程如下：

1. 使用 OpenCV 解碼圖片，原始內存色彩順序為 BGR。
2. 按原寬高比縮放到能夠完整放入 `960 × 960` 的最大尺寸，不拉伸、不裁剪。
3. 在上下或左右居中填充，默認填充值為 BGR `[114, 114, 114]`。
4. 大圖縮小時采用 `INTER_AREA`，小圖放大時采用 `INTER_LINEAR`。
5. 將 BGR 轉為 RGB，布局從 HWC 轉為 CHW，并增加 batch 維。
6. 轉為 FP32/FP16，將像素值除以 255，歸一化到 `[0, 1]`。

縮放比例為：

```text
scale = min(960 / original_width, 960 / original_height)
```

`preprocess.scale_up: true` 時允許放大小圖；設置為 `false` 時小圖只進行居中填充。該處理與訓練時的等比例縮放加灰邊策略保持一致。

## 6. 輸出后處理

后處理實現在 [`postprocess.py`](postprocess.py) 和 [`pipeline.py`](pipeline.py)，默認流程如下：

1. 讀取類別置信度，過濾 `confidence ≤ 0.25` 的候選框。
2. 若候選過多，僅保留最高分的前 `3000` 個候選。
3. 使用基于概率 IoU 的旋轉框 NMS，默認 IoU 閾值為 `0.45`。
4. 每張圖片最多保留 `300` 個檢測結果。
5. 從中心點、寬、高、角度轉換為四個旋轉框角點。
6. 減去 Letterbox 填充量并除以縮放比例，將結果映射回原圖坐標。
7. 同時生成像素角點、歸一化角點、中心點、尺寸和弧度/角度表示。
8. 在原圖副本上繪制旋轉框、類別名稱及置信度。

置信度閾值、NMS 閾值及最大框數量均可在 [`config/config.yaml`](config/config.yaml) 的 `inference` 段修改。真實場景漏檢較多時可先降低置信度閾值做召回評估，但正式閾值必須在獨立真實驗證集上確定。

## 7. 權重文件說明

`weight/` 中的文件如下：

| 文件 | 精度/格式 | 用途與注意事項 |
|---|---|---|
| `best.pt` | PyTorch checkpoint，約 40.50 MiB | 當前默認權重，保留原始模型對象；只加載可信來源的文件 |
| `best_standalone.pt` | TorchScript，約 41.18 MiB | 由 `convert_checkpoint.py` 生成的獨立推理權重，適合減少訓練框架對象依賴 |
| `best.onnx` | ONNX FP32，約 79.97 MiB | 精度基準及通用部署版本，固定輸入 `[1,3,960,960]` |
| `best_fp16.onnx` | ONNX FP16，約 40.08 MiB | FP16 圖，輸入輸出仍保持 FP32；部署端需要支持圖中的 FP16 算子 |
| `best_fp8.onnx` | ONNX FP8 E4M3FN 權重，約 20.46 MiB | 逐輸出通道權重存儲量化，執行時恢復 FP32；不代表原生 FP8 加速 |
| `best_int8.onnx` | ONNX INT8 權重，約 20.44 MiB | 逐輸出通道權重存儲量化，執行時恢復 FP32；默認優先保證行為穩定 |
| `best*.json` | JSON | 相應導出文件的來源 SHA-256、張量契約、精度模式及數值校驗結果 |

注意：

- `best.json` 是 `best.onnx` 的導出報告，不是 `best.pt` 的標簽文件。
- 默認 [`config/config.yaml`](config/config.yaml) 仍指向 `best.pt`。如需使用獨立 TorchScript 權重，將 `model.weights` 改為 `../weight/best_standalone.pt`。
- 當前推理 Pipeline 加載的是 `.pt`；ONNX 文件用于后續 ONNX Runtime、移動端轉換或硬件編譯鏈，不能直接替換 `model.weights`。
- FP8/INT8 默認是權重存儲量化而非端到端低精度計算。若將 `export.int8_mode` 改為 `static_qdq`，會使用校準圖執行混合靜態量化；必須重新評估 mAP 和漏檢率。
- GitHub Web 單文件上傳限制可能小于這些權重文件。正式倉庫建議使用 Git LFS 管理 `*.pt` 和 `*.onnx`，或在 Release 中發布權重并在 README 記錄校驗值。

重新生成獨立權重：

```bash
python convert_checkpoint.py
```

重新導出全部 ONNX 版本：

```bash
python export_onnx.py
```

## 8. 訓練結果

本次訓練共 60 個周期，總訓練時間約 `18144.1 s`（約 5 小時 2 分）。逐周期的訓練損失、驗證損失、學習率及檢測指標均保存在 [`trainResult/results.csv`](trainResult/results.csv)，交接時以該 CSV 為原始記錄，不在本文重復列出全部 60 行。

### 8.1 關鍵指標摘要

| 指標 | 最佳值 | 最佳周期 |
|---|---:|---:|
| Precision | 0.98469 | 30 |
| Recall | 0.95558 | 60 |
| mAP50 | 0.98048 | 40 |
| mAP50-95 | 0.89379 | 60 |

第 60 周期的損失與指標：

| train/box | train/cls | train/dfl | train/angle | val/box | val/cls | val/dfl | val/angle | Precision | Recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.41093 | 0.31891 | 1.01352 | 0.01248 | 0.41423 | 0.32093 | 1.05277 | 0.01842 | 0.97692 | 0.95558 | 0.98012 | 0.89379 |

### 8.2 混淆矩陣

混淆矩陣記錄 981 個正確水印檢測、53 個背景誤檢和 38 個水印漏檢。該矩陣來自訓練流程的驗證集統計，不能替代真實業務數據評估。

![混淆矩陣](trainResult/confusion_matrix.png)

### 8.3 損失趨勢

訓練與驗證的 box、cls、dfl、angle 損失總體收斂。角度驗證損失在后期穩定在約 0.018，訓練損失繼續下降，部署前仍應關注真實數據上的域差異。

![損失趨勢](trainResult/training_loss_trends.png)

### 8.4 綜合訓練指標

該圖展示損失、Precision、Recall、mAP 和學習率隨周期的變化。

![綜合訓練指標](trainResult/results.png)

### 8.5 訓練集標注可視化

以下圖片用于快速核對訓練樣本的 Letterbox、旋轉框方向、類別編號和多目標標注是否合理。

| train batch 0 | train batch 1 |
|---|---|
| ![訓練批次0](trainResult/train_batch0.jpg) | ![訓練批次1](trainResult/train_batch1.jpg) |

### 8.6 驗證集標簽與預測可視化

`labels` 為人工/生成標簽，`pred` 為模型預測。交接驗收時應成對查看，重點關注透明水印、小水印、貼邊水印、旋轉角度及相鄰水印。

| 驗證標簽 batch 0 | 驗證預測 batch 0 |
|---|---|
| ![驗證標簽0](trainResult/val_batch0_labels.jpg) | ![驗證預測0](trainResult/val_batch0_pred.jpg) |

| 驗證標簽 batch 1 | 驗證預測 batch 1 |
|---|---|
| ![驗證標簽1](trainResult/val_batch1_labels.jpg) | ![驗證預測1](trainResult/val_batch1_pred.jpg) |

## 9. 測試集與使用方法

### 9.1 測試目錄

- `test/input/`：當前包含 40 張合成數據、自然圖片和真實來源圖片，用于冒煙測試與效果觀察。
- `test/output/`：保存推理 JSON 和標注圖，屬于可重復生成的運行產物，默認不納入 Git。
- 測試圖片僅用于工程驗證。上傳公開倉庫前，應確認所有圖片的版權、隱私和數據授權。

每張輸入圖片會生成：

```text
test/output/<原文件名>.json
test/output/<原文件名>_annotated.<原后綴>
```

### 9.2 使用配置運行全部測試圖片

在本目錄運行：

```bash
python pipeline.py
```

程序讀取 `config.test.input`，并將結果寫入 `config.test.output`。是否遞歸掃描子目錄由 `config.test.recursive` 控制。

也可直接調用測試方法：

```python
from test_pipeline import run_configured_input_test

results = run_configured_input_test()
print(f"完成 {len(results)} 張圖片")
```

### 9.3 單張圖片推理

```python
from pipeline import OBBInferencePipeline

detector = OBBInferencePipeline("config/config.yaml")
result = detector.predict_file(
    "test/input/example.jpg",
    output_dir="test/output",
)
print(result["watermarks"])
```

### 9.4 文件夾推理

```python
from pipeline import OBBInferencePipeline

detector = OBBInferencePipeline("config/config.yaml")
results = detector.predict_folder(
    input_dir="path/to/images",
    output_dir="path/to/output",
    recursive=False,
)
```

### 9.5 自動化測試

```bash
python -m unittest test_pipeline.py
```

測試內容包括：權重加載、獨立 TorchScript 權重檢查、大/小圖 Letterbox、旋轉框角點轉換，以及配置目錄下全部測試圖片的端到端推理。最后一項會實際覆蓋 `test/output/` 中的同名結果，運行時間取決于設備和測試圖片數量。

## 10. 配置說明

主要參數位于 [`config/config.yaml`](config/config.yaml)：

| 配置段 | 用途 |
|---|---|
| `model` | 權重路徑、設備、輸入尺寸、FP16 開關和類別名稱 |
| `inference` | 置信度閾值、旋轉框 NMS 閾值和最大檢測數 |
| `preprocess` | 灰邊顏色及是否允許放大小圖 |
| `visualization` | 線寬、字體、顏色和 JPEG 質量 |
| `test` | 測試輸入、輸出路徑及遞歸掃描開關 |
| `export` | ONNX 輸出路徑、opset、量化模式、校準目錄及數值驗證閾值 |

相對路徑均以 `config/config.yaml` 所在目錄為基準，而不是調用程序時的工作目錄。

## 11. 潛在問題
1. 當前 OBB 模型在合成水印數據集上表現良好，在已知的數據分布中達到 mAP50-95 ≈ 90% 的成績，受限于真實數據集數量，但在真實互聯網數據中測試僅達到 mAP50-95 ≥ 70%。 
2. 當前版本模型存在未能正確標注識別的水印，包括不限于分布位置不規律水印、占比區域較小水印、重疊水印、透明度過低或過高水印（過高會被認定為字幕而非水印），存在漏檢可能，仍需支持最終用戶手動添加標記。


## 12. 注意事項
1. 訓練/驗證指標主要反映當前數據分布；上線前必須使用獨立、真實、未參與調參的數據統計 Precision、Recall、mAP 和圖片級漏檢率。
2. 對于水印檢測業務，通常應優先控制漏檢。閾值調整必須同時觀察誤檢數量和后續處理成本。
3. FP16、FP8、INT8 文件都需要在目標 NPU/GPU 的實際運行時上驗證算子支持、延遲、內存和精度，文件更小不等于推理一定更快。
4. `.pt` 使用基于 pickle 的加載機制，只加載可信權重；移動端優先使用經過目標工具鏈驗證的 ONNX/Core ML/TFLite/廠商格式。
5. 當前模型基于 YoloV11m-OBB 再訓練，訓練環境中使用了 Ultralytics 庫及其衍生組件；當前項目中已使用 Torch 重構推理鏈路，去除了 Ultralytics 相關依賴。
