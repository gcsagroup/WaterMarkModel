[簡體中文](README.zh-Simplified.md) | [繁體中文](README.zh-Traditional.md) | [English](README.en.md)

# 可見水印初篩模型推理項目

## 1. 項目用途與邊界

本項目基于 MobileNetV3-Large 微調實現對圖像初步是水印判別，對輸入圖像執行可見水印**初步篩選**。輸出表示“該圖像是否為潛在水印圖像”，不是嚴格、最終的水印判定。

被本模型篩選為水印候選的圖像，后續仍須送入 OBB 水印目標識別模型，由 OBB 模型最終確認：

- 是否確實存在水印；
- 水印數量；
- 每個水印的位置和方向框。

本模型使用合成水印數據訓練。真實圖像的水印樣式、透明度、壓縮、縮放、截圖噪聲和拍攝鏈路可能與訓練數據存在域差異，因此在真實數據上仍可能出現漏檢或誤檢，不應把初篩結果直接作為業務最終結論。

## 2. 目錄結構

```text
Detect/
├── config/
│   └── config.yaml                    # 推理、預處理、測試和導出配置
├── test/
│   ├── *.png                          # 隨項目交付的測試圖像
│   └── result.json                    # 測試匯總結果
├── trainResult/
│   ├── training_history.csv           # 30 個訓練周期的原始指標
│   └── training_curves.png            # 訓練指標可視化
├── weights/
│   ├── mobilenet_v3_large.pt           # PyTorch V3 權重
│   └── onnx/                           # ONNX 導出文件及元數據
├── model.py                            # 模型定義與 PT 權重加載
├── preprocess.py                       # 原始尺度滑窗與歸一化
├── pipeline.py                         # 單圖、文件夾和測試推理接口
├── test_pipeline.py                    # 一鍵執行 test 測試
├── export_onnx.py                      # FP16、FP8、INT8 ONNX 轉換
├── settings.py                         # YAML 配置讀取與校驗
└── requirements.txt
```

項目沒有附帶開源許可證。上傳公開倉庫前，應由項目所有者補充合適的 `LICENSE`，并確認訓練數據、測試圖像和權重具備對應的分發權限。較大的 `.pt`、`.onnx` 文件建議通過 Git LFS 管理，倉庫已經提供相應的 `.gitattributes` 規則。

## 3. 環境安裝

當前項目已驗證，核心版本為 PyTorch 2.13、TorchVision 0.28、ONNX 1.22 和 ONNX Runtime 1.29。

```powershell
python -m pip install -r requirements.txt
```

如需使用 CUDA，請安裝與目標 CUDA 環境匹配的 PyTorch；僅做 CPU 或移動端 ONNX 推理時不要求 CUDA。

## 4. 模型輸入與輸出

### 4.1 輸入

PyTorch/ONNX 模型本體接收：

- 張量名稱：`images`；
- 形狀：`[N, 3, 320, 320]`；
- 通道順序：RGB；
- 數據類型：FP32 輸入；
- 數值處理：先除以 255，再使用 ImageNet mean/std 歸一化。

其中 `N` 是一次送入模型的滑窗數量。ONNX 只包含分類模型，不包含圖像解碼、滑窗切分、歸一化和整圖結果聚合；移動端需要復現第 5 節的流程。

### 4.2 輸出

模型本體輸出：

- 張量名稱：`logits`；
- 形狀：`[N]`；
- 含義：每個 320 × 320 滑窗的未歸一化分類 logit。

Pipeline 對每個 logit 執行 sigmoid，得到滑窗水印概率，再以整張圖所有滑窗概率的最大值作為圖像概率：

```text
image_probability = max(sigmoid(tile_logits))
is_watermark = image_probability >= threshold
```

使用最大值聚合是為了優先保證召回：只要任一局部區域呈現水印特征，整圖即進入后續 OBB 檢查階段。

## 5. 輸入預處理

預處理與 V3 訓練策略保持一致，不把任意比例圖像強制縮放為 320 × 320：

1. 讀取圖像并根據 EXIF 修正方向，統一轉換為 RGB。
2. 當寬或高超過 320 像素時，按原始分辨率執行 320 × 320 滑窗，默認步長為 240，即相鄰窗口重疊 80 像素。
3. 每個方向最后一個窗口強制貼合圖像末端，避免右邊緣和下邊緣遺漏。
4. 當圖像寬或高不足 320 像素時，不放大、不拉伸，僅在右側和底部補黑至 320 × 320。
5. 像素轉換為 `[0, 1]`，隨后按 ImageNet 參數歸一化：mean 為 `[0.485, 0.456, 0.406]`，std 為 `[0.229, 0.224, 0.225]`。

這種處理能保留半透明水印在原始尺度下的局部紋理，但大圖會產生更多滑窗，推理耗時與圖像面積近似正相關。

## 6. 輸出后處理與閾值

推薦閾值為 **0.2-0.4** 之間，配置位置為 `config/config.yaml` 中的 `inference.threshold`。配置文件中的值優先于權重檢查點內記錄的歷史閾值。

- 閾值越高，判別越嚴格，誤報通常減少，但可能增加漏檢；
- 閾值越低，判別越寬松，召回率通常提高，但精度下降、更多圖像會進入 OBB 階段；
- 當前任務不能接受漏檢時，可在獨立真實驗證集上嘗試低于 0.3 的閾值，并結合 OBB 階段的吞吐壓力確定最終工作點。

閾值需要按實際業務數據重新標定，不能僅根據合成驗證集決定，考慮到當前模型僅作初篩，閾值應盡可能低，確保**召回率優先**。

## 7. 配置文件

全部推理配置從 `config/config.yaml` 讀取，包括：權重路徑、設備、滑窗大小、步長、歸一化參數、判別閾值、滑窗批大小、測試輸入輸出路徑以及 ONNX 導出參數。相對路徑均相對于 `config.yaml` 所在目錄解析。

常用配置：

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

`device: auto` 會優先使用 CUDA，否則使用 CPU。顯存不足時可減小 `tile_batch_size`。

## 8. 推理用法

### 8.1 命令行

不傳輸入路徑時，按 YAML 的 `test.input` 和 `test.result` 執行內置測試：

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

也可以執行測試入口：

```powershell
python test_pipeline.py
```

### 8.2 Python 接口

```python
from pipeline import DetectionPipeline

detector = DetectionPipeline("config/config.yaml")

# 單圖：返回DetectionResult
one = detector.infer_file("D:/images/example.jpg")
print(one.has_watermark, one.probability)

# 文件夾：返回DetectionResult列表
many = detector.infer_directory("D:/images")

# 自動判斷單圖/文件夾，并將結果寫入JSON；返回結果列表
results = detector.run("D:/images", "D:/images/result.json")

# 使用config.yaml中的test路徑；返回結果列表
test_results = detector.test()
```

`result.json` 的每條記錄包含圖像路徑、是否為水印候選、圖像概率、閾值、原始尺寸、滑窗數量、最高分滑窗坐標、耗時和錯誤信息。文件頂層還包含模型、權重、匯總數量和運行配置。

## 9. 測試集與當前測試結果

`test` 文件夾包含 15 張隨項目交付的測試圖像。執行 `python test_pipeline.py` 后，會覆蓋生成 `test/result.json`。

當前使用 `mobilenet_v3_large.pt`、閾值 0.3 的結果為：

- 總圖像數：15；
- 水印候選：9；
- 非水印候選：6；
- 讀取/推理錯誤：0。

該測試目錄沒有獨立真值標簽，因此以上數字只是輸出分布和流程驗收結果，不代表準確率、召回率或特異性。

## 10. 權重文件

`weights/mobilenet_v3_large.pt` 是 DetectV3 的 MobileNetV3-Large PyTorch 檢查點，包含模型結構標識、模型參數、訓練周期和訓練/驗證元數據。當前文件記錄的訓練周期為第 30 周期。Pipeline 以嚴格模式加載 `model_state_dict`，結構不匹配時直接報錯，避免靜默使用錯誤權重。

`weights/onnx` 中的導出物包括：

- `mobilenet_v3_large_fp32.onnx`：FP32 基準模型；
- `mobilenet_v3_large_fp16.onnx`：FP16 內部計算，保留 FP32 輸入輸出；
- `mobilenet_v3_large_fp8.onnx`：E4M3FN 逐輸出通道權重存儲，圖內恢復為 FP32 計算；
- `mobilenet_v3_large_int8.onnx`：使用 `test` 圖像滑窗校準的靜態 QDQ INT8 模型；
- 同名 `.json`：每個模型的哈希、輸入輸出、來源權重和數值校驗結果；
- `export_summary.json`：全部導出物匯總。

當前導出大小約為 FP32 16.03 MiB、FP16 8.05 MiB、FP8 4.18 MiB、INT8 4.51 MiB。FP32、FP16、INT8 已通過當前 ONNX Runtime 的導出后數值校驗；FP8 可以通過 ONNX 結構檢查并運行，但當前樣本 logit 誤差未通過配置門限，因此屬于實驗格式，部署前必須在目標設備和真實驗證集上重新驗證。

FP8 在不同移動端推理框架中的算子支持差異較大；此文件主要壓縮權重存儲，并不等同于原生 FP8 加速。INT8 校準目前使用交付測試圖像，僅用于演示完整轉換流程。正式發布應把 `export.calibration_input` 指向具有代表性、合規且不參與最終測試的真實校準集，再重新導出并評估召回率。

## 11. ONNX 轉換

```powershell
python export_onnx.py --config config/config.yaml
```

轉換流程會：

1. 從 YAML 指定的 PT 權重重建 MobileNetV3-Large；
2. 導出 FP32 基準 ONNX；
3. 由基準模型生成 FP16、FP8 權重存儲和靜態 INT8 QDQ 版本；
4. 執行 ONNX 結構檢查和 ONNX Runtime 數值比較；
5. 在 `weights/onnx` 寫入模型、逐模型元數據和匯總文件。

不同手機芯片和移動推理框架支持的量化格式不同。上線前需要在實際 NNAPI、Core ML、ONNX Runtime Mobile 或廠商 NPU 后端上測試算子兼容性、端到端延遲、內存占用以及圖像級召回率。移動端還必須在模型外實現完全相同的滑窗、補邊、歸一化、sigmoid、最大值聚合和閾值判定。

## 12. 訓練結果說明

`trainResult/training_history.csv` 保存 30 個周期的訓練和驗證指標。核心趨勢如下：

- 總訓練損失從 1.3311 降至 0.1863，最低 0.1855（第 28 周期）；
- 圖像級訓練損失從 0.5903 降至 0.0046；
- 難負樣本損失從 0.4380 降至 0.0003；
- 排序損失在第 5 周期降至 0；
- 輔助損失從 0.7264 降至 0.3632；
- 驗證損失最低為 0.3052（第 6 周期）；
- 第 6 周期同時取得最高驗證 F1 約 0.8885、準確率約 0.887、召回率 0.900、特異性 0.874，對應當周期選擇閾值 0.34；
- 第 30 周期驗證損失為 0.3583，記錄閾值 0.17、召回率 0.904、特異性 0.848、準確率 0.876、F1 約 0.8794。

后期訓練損失繼續下降而驗證集未同步改善，說明存在一定訓練/驗證泛化差距。部署配置采用推薦閾值 0.3，而不是照搬某一訓練周期自動選擇的閾值，并且仍需以真實業務驗證集重新標定。

![MobileNetV3-Large訓練指標可視化](trainResult/training_curves.png)

## 13. 上線前檢查清單

- 單獨覆蓋低透明度、小水印、多水印、時間戳、平臺角標、復雜文字背景和不同壓縮質量；
- 依據“漏檢優先”目標重新選擇閾值，不以 accuracy 單指標決策；
- 在目標手機上復現原始尺度滑窗，并測量超大圖批量處理的耗時與峰值內存；
- 對 FP16/FP8/INT8 分別做端到端回歸測試，不能只驗證模型能被加載；
- 保留 OBB 二階段確認，避免把初篩誤報直接作為最終水印結論。
