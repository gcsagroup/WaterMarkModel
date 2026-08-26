# 可见水印两阶段检测项目

本目录包含两个相互衔接的可见水印模型：

1. **DetectNet**：基于 `MobileNetV3-Large` 的图像级初筛模型，以召回率优先，判断图像是否可能包含可见水印。
2. **OBBNet**：基于 `Yolo v11 OBB` 旋转目标框（Oriented Bounding Box，OBB）的精检模型，对候选图像进行最终确认，并输出水印数量、位置、方向与置信度。

推荐的业务流程如下：

```text
输入图像
   │
   ▼
DetectNet 初筛
   ├── 非水印候选 ──► 结束
   │
   └── 水印候选
          │
          ▼
       OBBNet 精检
          │
          ├── JSON：水印数量、置信度、旋转框坐标
          └── 标注图：在原图上绘制旋转框
```

> DetectNet 的结果不是最终水印结论。为降低漏检风险，初筛阈值应以召回率优先；最终结果以 OBBNet 输出为准。

## 1. 项目目录

```text
water_mark_detect/
├── README.md                         # 本文：两阶段项目统一说明
├── DetectNet/                        # 第一阶段：图像级水印初筛
│   ├── config/config.yaml            # 推理、预处理、测试及导出配置
│   ├── test/                         # 测试图像及汇总结果
│   ├── trainResult/                  # 训练记录与曲线
│   ├── weights/                      # PyTorch 与 ONNX 权重
│   ├── model.py                      # 模型定义及 PT 权重加载
│   ├── preprocess.py                 # 滑窗、补边和归一化
│   ├── pipeline.py                   # 单图、目录及测试推理接口
│   ├── test_pipeline.py              # 测试入口
│   ├── export_onnx.py                # ONNX 导出
│   └── requirements.txt
└── OBBNet/                           # 第二阶段：旋转框水印检测
    ├── config/config.yaml            # 推理、预处理、测试及导出配置
    ├── test/input/                   # 测试输入
    ├── test/output/                  # JSON 与标注图输出
    ├── trainResult/                  # 训练指标和可视化
    ├── weight/                       # PyTorch、TorchScript 与 ONNX 权重
    ├── model.py                      # 独立模型结构及权重加载
    ├── preprocess.py                 # Letterbox 预处理
    ├── postprocess.py                # 阈值过滤、旋转框 NMS 和坐标还原
    ├── pipeline.py                   # 单图、目录及测试推理接口
    ├── test_pipeline.py              # 自动化测试入口
    ├── convert_checkpoint.py         # 独立 TorchScript 权重转换
    ├── export_onnx.py                # ONNX 导出
    └── requirements.txt
```

## 2. 环境安装

建议使用 Python 3.10 或更高版本。两个子项目依赖不同，建议分别创建虚拟环境；也可以在同一环境中依次安装两份依赖。

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

Linux/macOS 激活虚拟环境时使用：

```bash
source .venv/bin/activate
```

两项目均要求 PyTorch 2.5 或更高版本。使用 CUDA 时，请安装与目标 CUDA 环境匹配的 PyTorch；CPU 或 ONNX 推理不要求 CUDA。

## 3. 第一阶段：DetectNet 初筛

### 3.1 用途与边界

DetectNet 基于 MobileNetV3-Large 微调，对输入图像执行图像级可见水印初筛。模型使用合成水印数据训练，真实图像在水印样式、透明度、压缩、缩放、截图噪声和拍摄链路上可能存在域差异，因此可能出现漏检或误检。

它适合在 OBBNet 前减少需要精检的图像数量，不应单独作为业务最终判定。

### 3.2 输入与输出

| 项目 | 说明 |
|---|---|
| ONNX 输入名称 | `images` |
| 输入形状 | `[N, 3, 320, 320]`，`N` 为滑窗数量 |
| 输入类型 | FP32 |
| 色彩顺序 | RGB |
| 归一化 | 像素除以 255，再按 ImageNet mean/std 归一化 |
| 输出名称 | `logits` |
| 输出形状 | `[N]` |
| 输出含义 | 每个 320 × 320 滑窗的未归一化分类 logit |

Pipeline 对每个 logit 执行 sigmoid，并取整张图所有滑窗概率的最大值：

```text
image_probability = max(sigmoid(tile_logits))
is_watermark = image_probability >= threshold
```

ONNX 仅包含分类模型，不包含图像解码、滑窗切分、归一化、整图聚合和阈值判定；在移动端部署时需要在模型外复现这些步骤。

### 3.3 预处理

1. 根据 EXIF 修正方向并转换为 RGB。
2. 按原始分辨率执行 320 × 320 滑窗，默认步长 240，相邻窗口重叠 80 像素。
3. 每个方向的最后一个窗口贴合图像末端，避免遗漏右边缘或下边缘。
4. 宽或高不足 320 像素时不放大、不拉伸，仅在右侧和底部补黑。
5. 像素转换到 `[0, 1]`，再使用 ImageNet 参数归一化：mean `[0.485, 0.456, 0.406]`，std `[0.229, 0.224, 0.225]`。

该策略保留半透明水印在原始尺度下的局部纹理，但大图会产生更多滑窗，推理耗时和图像面积近似正相关。

### 3.4 配置与阈值

配置文件为 [`DetectNet/config/config.yaml`](DetectNet/config/config.yaml)。常用配置如下：

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

- `device: auto` 优先使用 CUDA，否则使用 CPU。
- 显存不足时可减小 `tile_batch_size`。
- 推荐从 0.2～0.4 范围标定阈值。阈值越低，召回通常越高，但会有更多候选进入 OBBNet；阈值越高，误报通常越少，但漏检风险增加。
- 配置文件中的阈值优先于权重检查点内记录的历史阈值。

正式阈值必须使用独立、真实、未参与训练和调参的业务验证集确定。

### 3.5 推理用法

进入 `DetectNet` 目录后执行。

使用配置中的测试路径：

```powershell
python pipeline.py
```

输入单张图像：

```powershell
python pipeline.py --input D:\images\example.jpg --output D:\images\result.json
```

输入文件夹：

```powershell
python pipeline.py --input D:\images --output D:\images\result.json
```

测试入口：

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

汇总 JSON 会记录图像路径、候选判定、图像概率、阈值、原始尺寸、滑窗数量、最高分滑窗坐标、耗时和错误信息。

### 3.6 测试与训练结果

当前 [`DetectNet/test/result.json`](DetectNet/test/result.json) 是一次历史运行快照，记录 15 张图像在阈值 0.3 下的输出：水印候选 9 张、非水印候选 6 张、错误 0。测试目录目前包含更多图像，重新执行测试后会覆盖该文件，因此该快照不能代表当前目录全部样本的结果。

测试数据没有完整、独立的真值标签，输出数量只用于流程验收，不能解释为准确率、召回率或特异性。

[`DetectNet/trainResult/training_history.csv`](DetectNet/trainResult/training_history.csv) 保存 30 个周期的训练和验证指标。摘要如下：

- 最低验证损失 0.3052（第 6 周期）。
- 第 6 周期验证 F1 约 0.8885、准确率约 0.887、召回率 0.900、特异性 0.874。
- 第 30 周期召回率 0.904、特异性 0.848、准确率 0.876、F1 约 0.8794。
- 后期训练损失继续下降而验证指标未同步改善，说明存在训练/验证泛化差距。

![DetectNet 训练指标](DetectNet/trainResult/training_curves.png)

### 3.7 权重与 ONNX

默认 PyTorch 权重为 `DetectNet/weights/mobilenet_v3_large.pt`。`DetectNet/weights/onnx/` 提供：

| 文件 | 说明 |
|---|---|
| `mobilenet_v3_large_fp32.onnx` | FP32 基准模型 |
| `mobilenet_v3_large_fp16.onnx` | FP16 内部计算，保留 FP32 输入输出 |
| `mobilenet_v3_large_fp8.onnx` | E4M3FN 权重存储格式，图内恢复 FP32 计算 |
| `mobilenet_v3_large_int8.onnx` | 静态 QDQ INT8 模型 |
| 同名 `.json`、`export_summary.json` | 来源、哈希、张量契约和数值校验报告 |

重新导出：

```powershell
python export_onnx.py --config config/config.yaml
```

**注意：量化可能会造成模型性能下降**，FP8 主要用于压缩权重存储，不等同于原生 FP8 加速；INT8 校准数据需要替换为具有代表性且不参与最终测试的真实数据。所有低精度模型上线前都必须在目标设备上验证算子兼容性、延迟、内存及图像级召回率。


## 4. 第二阶段：OBBNet 精检

### 4.1 用途与状态

OBBNet 对 DetectNet 筛出的候选图像执行水印识别和旋转框标注。

- 默认输入尺寸：960 × 960，batch 固定为 1。
- 默认推理权重：`OBBNet/weight/best.pt`。
- 支持单张图片或图片文件夹。
- 支持 JPG、JPEG、PNG、BMP、WebP 和 TIFF。
- 每张图片输出一份预测 JSON 和一张旋转框标注图。
- 当前仅包含一个类别：`watermark`。

### 4.2 输入与模型原始输出

| 项目 | 说明 |
|---|---|
| ONNX 输入名称 | `images` |
| 输入形状 | `[1, 3, 960, 960]` |
| 输入类型 | FP32；PyTorch CUDA 可按配置尝试 FP16 |
| 色彩顺序 | RGB |
| 数值范围 | `[0, 1]` |
| 类别 | `0: watermark` |
| 原始输出形状 | `[1, 6, 18900]` |

原始输出第二维依次为：

```text
center_x, center_y, width, height, watermark_score, angle_radians
```

原始张量尚未执行置信度过滤、旋转框 NMS 或原图坐标还原，不能直接作为最终标注结果。

### 4.3 预处理与后处理

预处理：

1. OpenCV 解码图像。
2. 保持原宽高比缩放，使其完整放入 960 × 960，不裁剪、不拉伸。
3. 上下或左右居中填充，默认填充值为 BGR `[114, 114, 114]`。
4. BGR 转 RGB、HWC 转 CHW、增加 batch 维，像素除以 255。

缩放比例：

```text
scale = min(960 / original_width, 960 / original_height)
```

后处理默认流程：

1. 过滤置信度不高于 0.25 的候选框。
2. 候选过多时保留最高分的前 3000 个。
3. 执行基于概率 IoU 的旋转框 NMS，默认 IoU 阈值 0.45。
4. 每张图片最多保留 300 个检测结果。
5. 将中心点、宽、高和角度转换为四个旋转框角点。
6. 去除 Letterbox 填充并除以缩放比例，将坐标映射回原图。
7. 输出像素角点、归一化角点、中心点、尺寸和弧度/角度表示，并生成标注图。

### 4.4 最终 JSON

每张图像的结果主要结构如下：

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

`obb_label` 为 `class_id + 4 个归一化角点`，共 9 个数。结果还会记录权重路径、运行设备、阈值、缩放比例、填充量和模型推理耗时。

### 4.5 配置

配置文件为 [`OBBNet/config/config.yaml`](OBBNet/config/config.yaml)。主要参数：

| 配置段 | 用途 |
|---|---|
| `model` | 权重、设备、输入尺寸、FP16 开关和类别名称 |
| `inference` | 置信度阈值、旋转框 NMS 阈值和最大检测数 |
| `preprocess` | 填充颜色及是否允许放大小图 |
| `visualization` | 线宽、字体、颜色和 JPEG 质量 |
| `test` | 测试输入、输出及递归扫描开关 |
| `export` | ONNX 输出、opset、量化、校准及数值验证参数 |

相对路径均以配置文件所在目录为基准。真实场景漏检较多时可以降低置信度阈值进行召回评估，但正式值必须在独立真实验证集上确定。

### 4.6 推理与测试

进入 `OBBNet` 目录后执行。

使用配置中的测试目录：

```powershell
python pipeline.py
```

单张图片：

```python
from pipeline import OBBInferencePipeline

detector = OBBInferencePipeline("config/config.yaml")
result = detector.predict_file(
    "test/input/example.jpg",
    output_dir="test/output",
)
print(result["watermarks"])
```

文件夹：

```python
from pipeline import OBBInferencePipeline

detector = OBBInferencePipeline("config/config.yaml")
results = detector.predict_folder(
    input_dir="path/to/images",
    output_dir="path/to/output",
    recursive=False,
)
```

自动化测试：

```powershell
python -m unittest test_pipeline.py
```

测试覆盖权重加载、TorchScript 权重检查、大/小图 Letterbox、旋转框角点转换，以及配置目录的端到端推理。端到端测试会覆盖 `test/output/` 中的同名产物。

### 4.7 权重与导出

| 文件 | 格式与用途 |
|---|---|
| `best.pt` | 默认 PyTorch checkpoint；只加载可信来源文件 |
| `best_standalone.pt` | 独立 TorchScript 推理权重 |
| `best.onnx` | FP32 基准 ONNX，固定输入 `[1,3,960,960]` |
| `best_fp16.onnx` | FP16 图，输入输出保持 FP32 |
| `best_fp8.onnx` | FP8 E4M3FN 权重存储，执行时恢复 FP32 |
| `best_int8.onnx` | INT8 权重存储，执行时恢复 FP32 |
| `best*.json` | 来源 SHA-256、张量契约、精度模式和数值校验报告 |

重新生成 TorchScript 权重并导出 ONNX：

```powershell
python convert_checkpoint.py
python export_onnx.py
```

当前 Pipeline 加载 `.pt`，ONNX 文件用于 ONNX Runtime、移动端转换或硬件编译链，不能直接替换配置中的 `model.weights`。FP8/INT8 默认属于权重存储量化，并不代表原生低精度加速。

### 4.8 训练指标

训练共 60 个周期，原始记录位于 [`OBBNet/trainResult/results.csv`](OBBNet/trainResult/results.csv)。

| 指标 | 最佳值 | 最佳周期 |
|---|---:|---:|
| Precision | 0.98469 | 30 |
| Recall | 0.95558 | 60 |
| mAP50 | 0.98048 | 40 |
| mAP50-95 | 0.89379 | 60 |

验证混淆矩阵记录 981 个正确水印检测、53 个背景误检和 38 个水印漏检。这些指标来自训练流程的验证集，不能替代真实业务数据评估。

| 混淆矩阵 | 损失趋势 |
|---|---|
| ![OBBNet 混淆矩阵](OBBNet/trainResult/confusion_matrix.png) | ![OBBNet 损失趋势](OBBNet/trainResult/training_loss_trends.png) |

![OBBNet 综合训练指标](OBBNet/trainResult/results.png)

| 验证标签 | 模型预测 |
|---|---|
| ![验证标签](OBBNet/trainResult/val_batch0_labels.jpg) | ![验证预测](OBBNet/trainResult/val_batch0_pred.jpg) |

## 5. 推荐的两阶段接入方式

当前两个子项目分别提供独立 Pipeline，尚未提供统一的端到端编排入口。业务接入时建议：

1. 使用 DetectNet 对输入图像初筛。
2. 仅将 `has_watermark = true` 的候选图像传给 OBBNet。
3. 以 OBBNet 的 `detection_count` 和 `watermarks` 作为最终机器检测结果。
4. 对 OBBNet 未检出但风险较高的图像保留人工复核或手动补标能力。
5. 分别记录两阶段耗时、阈值和模型版本，以便定位误检、漏检与性能问题。

两个模型的预处理契约不同，不能复用同一份模型输入张量：

| 项目 | DetectNet | OBBNet |
|---|---|---|
| 任务 | 图像级候选筛选 | 水印定位与最终确认 |
| 输入策略 | 原始尺度 320 × 320 滑窗 | 等比例缩放并填充到 960 × 960 |
| 归一化 | ImageNet mean/std | 仅除以 255 |
| batch | 动态滑窗数 `N` | 固定为 1 |
| 结果 | 图片概率和候选判定 | 数量、置信度和旋转框 |

## 6. 上线前检查清单

- 使用独立真实数据覆盖低透明度、小水印、多水印、贴边水印、旋转水印、平台角标、时间戳、复杂文字背景和不同压缩质量，进行评估，必要时进行额外微调。
- DetectNet 阈值以召回优先，并同时评估进入 OBBNet 的图像比例和吞吐压力。
- OBBNet 同时评估 Precision、Recall、mAP、图片级漏检率及人工补标成本。
- 在目标设备复现两套不同的预处理与后处理流程，并测试超大图的耗时和峰值内存。
- 对 FP32、FP16、FP8 和 INT8 分别进行端到端回归；不能只验证文件可加载。
- 量化模型必须在实际 NNAPI、Core ML、ONNX Runtime Mobile 或厂商 NPU 后端验证算子支持和真实精度。
- `.pt` 基于 pickle 加载，只使用可信来源权重。

## 7. 已知限制与授权注意事项

1. 两个模型主要基于合成数据或有限真实数据训练，训练/验证指标不等同于真实互联网数据表现。
2. 低透明度、面积过小、位置不规律、互相重叠或与字幕相似的水印仍可能漏检或误检，系统应保留人工复核和补标能力，必要时需在真实数据上进行低学习率微调。
3. OBBNet 基于 YOLO11m-OBB 再训练，训练环境使用过 Ultralytics 及相关组件；当前推理链路已用 PyTorch 重构并移除运行时 Ultralytics 依赖，正式商用前仍应由项目所有者核查所用模型、训练代码和权重对应的授权义务。
4. 当前目录未提供开源许可证。公开或对外分发前应补充合适的 `LICENSE`，并完成数据、依赖及模型权重的授权审查。

## 8. 子项目详细文档

本 README 是两个项目的统一入口。需要查看原始的独立说明时，可参考：

- [`DetectNet/Readme.md`](DetectNet/Readme.md)
- [`OBBNet/README.md`](OBBNet/README.md)
