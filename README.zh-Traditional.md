[簡體中文](README.zh-Simplified.md) | [繁體中文](README.zh-Traditional.md) | [English](README.en.md)

# 可見水印檢測

本目錄包含兩個相互銜接的可見水印模型：

1. **DetectNet**：基于 `MobileNetV3-Large` 的圖像級初篩模型，以召回率優先，判斷圖像是否可能包含可見水印。
2. **OBBNet**：基于 `Yolo v11 OBB` 旋轉目標框（Oriented Bounding Box，OBB）的精檢模型，對候選圖像進行最終確認，并輸出水印數量、位置、方向與置信度。

推薦的業務流程如下：

```text
輸入圖像
   │
   ▼
DetectNet 初篩
   ├── 非水印候選 ──? 結束
   │
   └── 水印候選
          │
          ▼
       OBBNet 精檢
          │
          ├── JSON：水印數量、置信度、旋轉框坐標
          └── 標注圖：在原圖上繪制旋轉框
```

> DetectNet 的結果不是最終水印結論。為降低漏檢風險，初篩閾值應以召回率優先；最終結果以 OBBNet 輸出為準。

## 1. 項目目錄

```text
water_mark_detect/
├── README.zh-Simplified.md           # 兩階段項目統一說明（簡體中文）
├── README.zh-Traditional.md          # 兩階段項目統一說明（繁體中文）
├── README.en.md                      # 兩階段項目統一說明（英文）
├── DetectNet/                        # 第一階段：圖像級水印初篩
│   ├── config/config.yaml            # 推理、預處理、測試及導出配置
│   ├── test/                         # 測試圖像及匯總結果
│   ├── trainResult/                  # 訓練記錄與曲線
│   ├── weights/                      # PyTorch 與 ONNX 權重
│   ├── model.py                      # 模型定義及 PT 權重加載
│   ├── preprocess.py                 # 滑窗、補邊和歸一化
│   ├── pipeline.py                   # 單圖、目錄及測試推理接口
│   ├── test_pipeline.py              # 測試入口
│   ├── export_onnx.py                # ONNX 導出
│   └── requirements.txt
└── OBBNet/                           # 第二階段：旋轉框水印檢測
    ├── config/config.yaml            # 推理、預處理、測試及導出配置
    ├── test/input/                   # 測試輸入
    ├── test/output/                  # JSON 與標注圖輸出
    ├── trainResult/                  # 訓練指標和可視化
    ├── weight/                       # PyTorch、TorchScript 與 ONNX 權重
    ├── model.py                      # 獨立模型結構及權重加載
    ├── preprocess.py                 # Letterbox 預處理
    ├── postprocess.py                # 閾值過濾、旋轉框 NMS 和坐標還原
    ├── pipeline.py                   # 單圖、目錄及測試推理接口
    ├── test_pipeline.py              # 自動化測試入口
    ├── convert_checkpoint.py         # 獨立 TorchScript 權重轉換
    ├── export_onnx.py                # ONNX 導出
    └── requirements.txt
```

## 2. 環境安裝

建議使用 Python 3.10 或更高版本。兩個子項目依賴不同，建議分別創建虛擬環境；也可以在同一環境中依次安裝兩份依賴。

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

Linux/macOS 激活虛擬環境時使用：

```bash
source .venv/bin/activate
```

兩項目均要求 PyTorch 2.5 或更高版本。使用 CUDA 時，請安裝與目標 CUDA 環境匹配的 PyTorch；CPU 或 ONNX 推理不要求 CUDA。

## 3. 第一階段：DetectNet 初篩

### 3.1 用途與邊界

DetectNet 基于 MobileNetV3-Large 微調，對輸入圖像執行圖像級可見水印初篩。模型使用合成水印數據訓練，真實圖像在水印樣式、透明度、壓縮、縮放、截圖噪聲和拍攝鏈路上可能存在域差異，因此可能出現漏檢或誤檢。

它適合在 OBBNet 前減少需要精檢的圖像數量，不應單獨作為業務最終判定。

### 3.2 輸入與輸出

| 項目 | 說明 |
|---|---|
| ONNX 輸入名稱 | `images` |
| 輸入形狀 | `[N, 3, 320, 320]`，`N` 為滑窗數量 |
| 輸入類型 | FP32 |
| 色彩順序 | RGB |
| 歸一化 | 像素除以 255，再按 ImageNet mean/std 歸一化 |
| 輸出名稱 | `logits` |
| 輸出形狀 | `[N]` |
| 輸出含義 | 每個 320 × 320 滑窗的未歸一化分類 logit |

Pipeline 對每個 logit 執行 sigmoid，并取整張圖所有滑窗概率的最大值：

```text
image_probability = max(sigmoid(tile_logits))
is_watermark = image_probability >= threshold
```

ONNX 僅包含分類模型，不包含圖像解碼、滑窗切分、歸一化、整圖聚合和閾值判定；在移動端部署時需要在模型外復現這些步驟。

### 3.3 預處理

1. 根據 EXIF 修正方向并轉換為 RGB。
2. 按原始分辨率執行 320 × 320 滑窗，默認步長 240，相鄰窗口重疊 80 像素。
3. 每個方向的最后一個窗口貼合圖像末端，避免遺漏右邊緣或下邊緣。
4. 寬或高不足 320 像素時不放大、不拉伸，僅在右側和底部補黑。
5. 像素轉換到 `[0, 1]`，再使用 ImageNet 參數歸一化：mean `[0.485, 0.456, 0.406]`，std `[0.229, 0.224, 0.225]`。

該策略保留半透明水印在原始尺度下的局部紋理，但大圖會產生更多滑窗，推理耗時和圖像面積近似正相關。

### 3.4 配置與閾值

配置文件為 [`DetectNet/config/config.yaml`](DetectNet/config/config.yaml)。常用配置如下：

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

- `device: auto` 優先使用 CUDA，否則使用 CPU。
- 顯存不足時可減小 `tile_batch_size`。
- 推薦從 0.2～0.4 范圍標定閾值。閾值越低，召回通常越高，但會有更多候選進入 OBBNet；閾值越高，誤報通常越少，但漏檢風險增加。
- 配置文件中的閾值優先于權重檢查點內記錄的歷史閾值。

正式閾值必須使用獨立、真實、未參與訓練和調參的業務驗證集確定。

### 3.5 推理用法

進入 `DetectNet` 目錄后執行。

使用配置中的測試路徑：

```powershell
python pipeline.py
```

輸入單張圖像：

```powershell
python pipeline.py --input D:\images\example.jpg --output D:\images\result.json
```

輸入文件夾：

```powershell
python pipeline.py --input D:\images --output D:\images\result.json
```

測試入口：

```powershell
python test_pipeline.py
```

Python 接口：

```python
from pipeline import DetectionPipeline

detector = DetectionPipeline("config/config.yaml")

one = detector.infer_file("D:/images/example.jpg")
print(one.has_watermark, one.probability)

many = detector.infer_directory("D:/images")
results = detector.run("D:/images", "D:/images/result.json")
test_results = detector.test()
```

匯總 JSON 會記錄圖像路徑、候選判定、圖像概率、閾值、原始尺寸、滑窗數量、最高分滑窗坐標、耗時和錯誤信息。

### 3.6 測試與訓練結果

當前 [`DetectNet/test/result.json`](DetectNet/test/result.json) 是一次歷史運行快照，記錄 15 張圖像在閾值 0.3 下的輸出：水印候選 9 張、非水印候選 6 張、錯誤 0。測試目錄目前包含更多圖像，重新執行測試后會覆蓋該文件，因此該快照不能代表當前目錄全部樣本的結果。

[`DetectNet/trainResult/training_history.csv`](DetectNet/trainResult/training_history.csv) 保存 30 個周期的訓練和驗證指標。摘要如下：

- 最低驗證損失 0.3052（第 6 周期）。
- 第 6 周期驗證 F1 約 0.8885、準確率約 0.887、召回率 0.900、特異性 0.874。
- 第 30 周期召回率 0.904、特異性 0.848、準確率 0.876、F1 約 0.8794。
- 后期訓練損失繼續下降而驗證指標未同步改善，說明存在訓練/驗證泛化差距。

![DetectNet 訓練指標](DetectNet/trainResult/training_curves.png)

### 3.7 權重與 ONNX

默認 PyTorch 權重為 `DetectNet/weights/mobilenet_v3_large.pt`。`DetectNet/weights/onnx/` 提供：

| 文件 | 說明 |
|---|---|
| `mobilenet_v3_large_fp32.onnx` | FP32 基準模型 |
| `mobilenet_v3_large_fp16.onnx` | FP16 內部計算，保留 FP32 輸入輸出 |
| `mobilenet_v3_large_fp8.onnx` | E4M3FN 權重存儲格式，圖內恢復 FP32 計算 |
| `mobilenet_v3_large_int8.onnx` | 靜態 QDQ INT8 模型 |
| 同名 `.json`、`export_summary.json` | 來源、哈希、張量契約和數值校驗報告 |

重新導出：

```powershell
python export_onnx.py --config config/config.yaml
```

**注意：量化可能會造成模型性能下降**，FP8 主要用于壓縮權重存儲，不等同于原生 FP8 加速；INT8 校準數據需要替換為具有代表性且不參與最終測試的真實數據。所有低精度模型上線前都必須在目標設備上驗證算子兼容性、延遲、內存及圖像級召回率。


## 4. 第二階段：OBBNet 精檢

### 4.1 用途與狀態

OBBNet 對 DetectNet 篩出的候選圖像執行水印識別和旋轉框標注。

- 默認輸入尺寸：960 × 960，batch 固定為 1。
- 默認推理權重：`OBBNet/weight/best.pt`。
- 支持單張圖片或圖片文件夾。
- 支持 JPG、JPEG、PNG、BMP、WebP 和 TIFF。
- 每張圖片輸出一份預測 JSON 和一張旋轉框標注圖。
- 當前僅包含一個類別：`watermark`。

### 4.2 輸入與模型原始輸出

| 項目 | 說明 |
|---|---|
| ONNX 輸入名稱 | `images` |
| 輸入形狀 | `[1, 3, 960, 960]` |
| 輸入類型 | FP32；PyTorch CUDA 可按配置嘗試 FP16 |
| 色彩順序 | RGB |
| 數值范圍 | `[0, 1]` |
| 類別 | `0: watermark` |
| 原始輸出形狀 | `[1, 6, 18900]` |

原始輸出第二維依次為：

```text
center_x, center_y, width, height, watermark_score, angle_radians
```

原始張量尚未執行置信度過濾、旋轉框 NMS 或原圖坐標還原，不能直接作為最終標注結果。

### 4.3 預處理與后處理

預處理：

1. OpenCV 解碼圖像。
2. 保持原寬高比縮放，使其完整放入 960 × 960，不裁剪、不拉伸。
3. 上下或左右居中填充，默認填充值為 BGR `[114, 114, 114]`。
4. BGR 轉 RGB、HWC 轉 CHW、增加 batch 維，像素除以 255。

縮放比例：

```text
scale = min(960 / original_width, 960 / original_height)
```

后處理默認流程：

1. 過濾置信度不高于 0.25 的候選框。
2. 候選過多時保留最高分的前 3000 個。
3. 執行基于概率 IoU 的旋轉框 NMS，默認 IoU 閾值 0.45。
4. 每張圖片最多保留 300 個檢測結果。
5. 將中心點、寬、高和角度轉換為四個旋轉框角點。
6. 去除 Letterbox 填充并除以縮放比例，將坐標映射回原圖。
7. 輸出像素角點、歸一化角點、中心點、尺寸和弧度/角度表示，并生成標注圖。

### 4.4 最終 JSON

每張圖像的結果主要結構如下：

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

`obb_label` 為 `class_id + 4 個歸一化角點`，共 9 個數。結果還會記錄權重路徑、運行設備、閾值、縮放比例、填充量和模型推理耗時。

### 4.5 配置

配置文件為 [`OBBNet/config/config.yaml`](OBBNet/config/config.yaml)。主要參數：

| 配置段 | 用途 |
|---|---|
| `model` | 權重、設備、輸入尺寸、FP16 開關和類別名稱 |
| `inference` | 置信度閾值、旋轉框 NMS 閾值和最大檢測數 |
| `preprocess` | 填充顏色及是否允許放大小圖 |
| `visualization` | 線寬、字體、顏色和 JPEG 質量 |
| `test` | 測試輸入、輸出及遞歸掃描開關 |
| `export` | ONNX 輸出、opset、量化、校準及數值驗證參數 |

相對路徑均以配置文件所在目錄為基準。真實場景漏檢較多時可以降低置信度閾值進行召回評估，但正式值必須在獨立真實驗證集上確定。

### 4.6 推理與測試

進入 `OBBNet` 目錄后執行。

使用配置中的測試目錄：

```powershell
python pipeline.py
```

單張圖片：

```python
from pipeline import OBBInferencePipeline

detector = OBBInferencePipeline("config/config.yaml")
result = detector.predict_file(
    "test/input/example.jpg",
    output_dir="test/output",
)
print(result["watermarks"])
```

文件夾：

```python
from pipeline import OBBInferencePipeline

detector = OBBInferencePipeline("config/config.yaml")
results = detector.predict_folder(
    input_dir="path/to/images",
    output_dir="path/to/output",
    recursive=False,
)
```

自動化測試：

```powershell
python -m unittest test_pipeline.py
```

測試覆蓋權重加載、TorchScript 權重檢查、大/小圖 Letterbox、旋轉框角點轉換，以及配置目錄的端到端推理。端到端測試會覆蓋 `test/output/` 中的同名產物。

### 4.7 權重與導出

| 文件 | 格式與用途 |
|---|---|
| `best.pt` | 默認 PyTorch checkpoint；只加載可信來源文件 |
| `best_standalone.pt` | 獨立 TorchScript 推理權重 |
| `best.onnx` | FP32 基準 ONNX，固定輸入 `[1,3,960,960]` |
| `best_fp16.onnx` | FP16 圖，輸入輸出保持 FP32 |
| `best_fp8.onnx` | FP8 E4M3FN 權重存儲，執行時恢復 FP32 |
| `best_int8.onnx` | INT8 權重存儲，執行時恢復 FP32 |
| `best*.json` | 來源 SHA-256、張量契約、精度模式和數值校驗報告 |

重新生成 TorchScript 權重并導出 ONNX：

```powershell
python convert_checkpoint.py
python export_onnx.py
```

當前 Pipeline 加載 `.pt`，ONNX 文件用于 ONNX Runtime、移動端轉換或硬件編譯鏈，不能直接替換配置中的 `model.weights`。FP8/INT8 默認屬于權重存儲量化，并不代表原生低精度加速。

### 4.8 訓練指標

訓練共 60 個周期，原始記錄位于 [`OBBNet/trainResult/results.csv`](OBBNet/trainResult/results.csv)。

| 指標 | 最佳值 | 最佳周期 |
|---|---:|---:|
| Precision | 0.98469 | 30 |
| Recall | 0.95558 | 60 |
| mAP50 | 0.98048 | 40 |
| mAP50-95 | 0.89379 | 60 |

驗證混淆矩陣記錄 981 個正確水印檢測、53 個背景誤檢和 38 個水印漏檢。這些指標來自訓練流程的驗證集，不能替代真實業務數據評估。

| 混淆矩陣 | 損失趨勢 |
|---|---|
| ![OBBNet 混淆矩陣](OBBNet/trainResult/confusion_matrix.png) | ![OBBNet 損失趨勢](OBBNet/trainResult/training_loss_trends.png) |

![OBBNet 綜合訓練指標](OBBNet/trainResult/results.png)

| 驗證標簽 | 模型預測 |
|---|---|
| ![驗證標簽](OBBNet/trainResult/val_batch0_labels.jpg) | ![驗證預測](OBBNet/trainResult/val_batch0_pred.jpg) |

## 5. 推薦的兩階段接入方式

當前兩個子項目分別提供獨立 Pipeline，尚未提供統一的端到端編排入口。業務接入時建議：

1. 使用 DetectNet 對輸入圖像初篩。
2. 僅將 `has_watermark = true` 的候選圖像傳給 OBBNet。
3. 以 OBBNet 的 `detection_count` 和 `watermarks` 作為最終機器檢測結果。
4. 對 OBBNet 未檢出但風險較高的圖像保留人工復核或手動補標能力。
5. 分別記錄兩階段耗時、閾值和模型版本，以便定位誤檢、漏檢與性能問題。

兩個模型的預處理契約不同，不能復用同一份模型輸入張量：

| 項目 | DetectNet | OBBNet |
|---|---|---|
| 任務 | 圖像級候選篩選 | 水印定位與最終確認 |
| 輸入策略 | 原始尺度 320 × 320 滑窗 | 等比例縮放并填充到 960 × 960 |
| 歸一化 | ImageNet mean/std | 僅除以 255 |
| batch | 動態滑窗數 `N` | 固定為 1 |
| 結果 | 圖片概率和候選判定 | 數量、置信度和旋轉框 |

## 6. 上線前檢查清單

- 使用獨立真實數據覆蓋低透明度、小水印、多水印、貼邊水印、旋轉水印、平臺角標、時間戳、復雜文字背景和不同壓縮質量，進行評估，必要時進行額外微調。
- DetectNet 閾值以召回優先，并同時評估進入 OBBNet 的圖像比例和吞吐壓力。
- OBBNet 同時評估 Precision、Recall、mAP、圖片級漏檢率及人工補標成本。
- 在目標設備復現兩套不同的預處理與后處理流程，并測試超大圖的耗時和峰值內存。
- 對 FP32、FP16、FP8 和 INT8 分別進行端到端回歸；不能只驗證文件可加載。
- 量化模型必須在實際 NNAPI、Core ML、ONNX Runtime Mobile 或廠商 NPU 后端驗證算子支持和真實精度。
- `.pt` 基于 pickle 加載，只使用可信來源權重。

## 7. 已知限制與授權注意事項

1. 兩個模型主要基于合成數據或有限真實數據訓練，訓練/驗證指標不等同于真實互聯網數據表現。
2. 低透明度、面積過小、位置不規律、互相重疊或與字幕相似的水印仍可能漏檢或誤檢，系統應保留人工復核和補標能力，必要時需在真實數據上進行低學習率微調。
3. OBBNet 基于 YOLO11m-OBB 再訓練，訓練環境使用過 Ultralytics 及相關組件；當前推理鏈路已用 PyTorch 重構并移除運行時 Ultralytics 依賴，正式商用前仍應由項目所有者核查所用模型、訓練代碼和權重對應的授權義務。

## 8. 子項目詳細文檔

本 README 是兩個項目的統一入口。需要查看原始的獨立說明時，可參考：

- [`DetectNet/README.zh-Traditional.md`](DetectNet/README.zh-Traditional.md)
- [`OBBNet/README.zh-Traditional.md`](OBBNet/README.zh-Traditional.md)
