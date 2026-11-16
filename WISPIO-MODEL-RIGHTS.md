# Wispio 模型分发许可提案与第三方声明

**状态：DetectNet 和 OBBNet 的 Wispio 专用许可待权利人确认。** 本仓库包含两类可见水印检测模型。训练权重、推理代码与第三方依赖可能有不同权利人。当前没有覆盖全仓库和所有权重的开源许可。

## 提请权利人批准的范围

对下列模型中签署人拥有或有权许可的部分，拟授予 Global Cybersecurity Alliance Limited 一项非独占许可，使其可为 Wispio 复制、修改、转换相关推理代码与权重，将指定衍生模型随 Wispio 分发，并允许应用商店为交付 Wispio 而托管和复制这些产物。范围包括 Wispio 的测试分发和正式应用商店分发。独立下载模型、公开训练集和测试图片、为其他产品分发均不在本提案范围内。

DetectNet 与 OBBNet 应分别由有权代表相应权利人的人员书面确认。每份确认应引用本文件的提交、具体代码路径和模型摘要，并载明权利人、覆盖的分发渠道、生效日期、第三方条款与随包声明。对一个模型的决定只适用于该项记录。公开可读、权重格式转换和合并这份提案均不能单独证明授权生效。

## DetectNet

Wispio 使用 MobileNetV3-Large 做图像级筛查。上游推理代码以 `weights=None` 初始化网络，再载入项目权重。这能说明加载方式，无法确定训练时从哪个 checkpoint 开始。请训练方确认初始化来源、训练数据使用条件、项目代码权属，以及训练权重和以下转换件的分发范围。TorchVision 的代码条款适用于实际使用的 TorchVision 代码。[推理代码](https://github.com/gcsagroup/WaterMarkModel/blob/411cb342c4f63b039716f55ea9f16c1e29164e11/DetectNet/model.py) · [TorchVision 代码许可](https://github.com/pytorch/vision/blob/v0.22.0/LICENSE)

| 资产 | Wispio 固定的 SHA-256 |
| --- | --- |
| `DetectNet/weights/mobilenet_v3_large.pt` | `4ef772ab087d07e0a90dd9b5e9bf2cab8a06a2c239814e93f9d27c645d23677c` |
| `DetectNet/weights/onnx/mobilenet_v3_large_fp32.onnx` | `f6f5a35d54eb8b3731591bab18810aa2ca251a2bfa8a5617210073ec7dc44374` |
| iOS Core ML FP32 | `19435ef00b248978476ebe5b4cf10769af9b3e2c2d222098dcc9d6ebbc4dca41` |
| Android LiteRT FP32 | `d8ad4d175ea83ca026467b8884a972bc48fa4e83c828909eb647609494ae6318` |

## OBBNet

上游将模型描述为 YOLO11 OBB。Wispio 的来源记录列出 AGPL-3.0，同时将具体基础 checkpoint 的来源标记为待核实。请训练方提供基础 checkpoint 的版本和摘要，区分原创代码、Ultralytics 组件、微调贡献和训练数据，并说明分发时如何履行适用条款，或提供涵盖实际部署及衍生权重的另行授权。将推理图导出为 ONNX、TorchScript 或移动端格式不会自行解决这些问题。[来源记录](https://github.com/gcsagroup/watermark/blob/6c33c3cd4dc493fae3d982257782c9a5700e5cf7/packages/model-assets/tools/visible_watermark_obb/source.lock.json) · [Ultralytics 许可选项](https://www.ultralytics.com/license)

| 资产 | Wispio 固定的 SHA-256 |
| --- | --- |
| `OBBNet/weight/best.pt` | `2c705caa5575cd1413df3d9e9b4b762aeb5207cf3f9a1caebf74af09ebb82585` |
| `OBBNet/weight/best_standalone.pt` | `3084df755258b28b680b15f4426ff133c3f80bd20cb748f2c6e7751d7c44a858` |
| `OBBNet/weight/best.onnx` | `1d2fe56774eb2e73dc5c028eb9fd0c211d036c4b106e985c4a57047879c5f19d` |
| iOS Core ML FP32 | `1f4ffdc72cca64e288759f71fc1d670a52a5e54fb105483b3a46dbc38d99f920` |
| Android LiteRT FP32 | `5f73cfbee3247c7978464c316b37e08694333ce60d5983c8020e104d24e09edd` |

上表固定上游源码 `411cb342c4f63b039716f55ea9f16c1e29164e11`。移动端指纹来自对应转换清单，Core ML 目录使用 `sha256-tree-v1`，其余使用文件 SHA-256。[Wispio 移动端转换清单](https://github.com/gcsagroup/watermark/blob/6c33c3cd4dc493fae3d982257782c9a5700e5cf7/packages/model-assets/mobile/research/visible-watermark-obb-native-models.json)

## Wispio 的第三方模型

以下条款来自各自的发布者。Wispio 使用这些模型时，应保留对应版权与许可全文。本仓库的 Wispio 专用许可提案仅覆盖签署人可授权的贡献，不能收窄第三方已授予的 MIT 权利。具体转换件和随包声明见 [Wispio 权重及转换记录](https://github.com/gcsagroup/watermark/blob/6c33c3cd4dc493fae3d982257782c9a5700e5cf7/docs/ios/apple_model_asset_rights.md)。

| 移动端检测器 | 代码与权重依据 |
| --- | --- |
| TrustMark P、Q | Adobe 明确 MIT 覆盖代码及首次使用时下载的模型。[代码许可](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/LICENSE) · [权重说明](https://github.com/adobe/trustmark/blob/59bde8b41c973d2fac3be7ef89e3ac0fea2254fa/README.md) |
| AudioSeal 16-bit | Meta 明确 MIT 覆盖模型权重。[代码许可](https://github.com/facebookresearch/audioseal/blob/e63a8a0e5cdf7bb797159c92ba15961557fe9bd2/LICENSE) · [权重说明](https://github.com/facebookresearch/audioseal/blob/e63a8a0e5cdf7bb797159c92ba15961557fe9bd2/README.md) |
| WavMark default | 代码和实际 checkpoint 的模型卡分别标记 MIT。保留上游许可中原有的版权占位文字。[代码许可](https://github.com/wavmark/wavmark/blob/6ab3bf7ce0679e5b5cfeff3a62e8df9cd2024b37/LICENSE) · [权重模型卡](https://huggingface.co/M4869/WavMark/blob/0ad3c7b74f641bddb61f6b85cdf2de0d93a5bfef/README.md) |
| VideoSeal v0 | Meta 明确模型适用 MIT。Wispio 固定 v0 的转换输入和独立的移动端产物摘要。[代码许可](https://github.com/facebookresearch/videoseal/blob/870ca7fb33578b90f14c602016b6c2788096226e/LICENSE) · [模型说明](https://github.com/facebookresearch/videoseal/blob/870ca7fb33578b90f14c602016b6c2788096226e/README.md) |

VAE Invisible 的授权提案由[专门的 VAE 模型仓库](https://github.com/gcsagroup/Image-Steganography-Invisible-Watermark-Pipeline)维护。Wispio 另有可选的 VideoSeal 1.0 TorchScript 和 Watermark Anything 候选，两者未列入上表的内置移动端清单。Watermark Anything 的 `wam_mit.pth` 使用 MIT，原论文权重有不同的非商业条款，复用时须区分具体 checkpoint。[VideoSeal 1.0 来源](https://github.com/facebookresearch/videoseal/blob/870ca7fb33578b90f14c602016b6c2788096226e/README.md) · [Watermark Anything 权重说明](https://github.com/facebookresearch/watermark-anything/blob/2c08af04d037d5667c02f6ddebbda9ff04581c3e/README.md)
