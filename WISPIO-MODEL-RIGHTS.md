# Wispio 模型分发权限与第三方声明

**DetectNet 和 OBBNet 的完整模型包均待补齐第三方分发依据。** 2026-11-16，GCSA 组织管理员 `GCSA-ToolsTeam` 分别确认这两项模型中 GCSA 拥有或有权许可的原创推理代码、训练贡献及对应移动端转换件，可由 Global Cybersecurity Alliance Limited 为 Wispio 非独占地复制、修改、转换和分发。确认绑定提交 `a8e594f3da30069bdd6ad4554eb04e1b0959ee06` 和以下两表的资产摘要。[管理员书面确认](https://github.com/gcsagroup/WaterMarkModel/pull/1#pullrequestreview-5918021136)

### Wispio 的分发渠道

| 版本与渠道 | GCSA 授权的用途和产物 |
| --- | --- |
| 1.0.0 · Apple | 允许 Wispio 测试分发、TestFlight、App Store，以及 Apple 专供该 App 下载的托管资源。范围以[原书面确认](https://github.com/gcsagroup/WaterMarkModel/pull/1#pullrequestreview-5918021136)为准。 |
| 1.1.0 · GitHub | GCSA 于 2026-11-16 补充授权 Global Cybersecurity Alliance Limited 通过官方 `gcsagroup/WaterMarkModel` 的 GitHub Releases 公开托管模型包，供 Wispio 用户下载并在 Wispio 中使用。GitHub 可存储、交付这些附件，任何人均可下载。[管理员补充授权](https://github.com/gcsagroup/WaterMarkModel/pull/1#issuecomment-6913749701) |

GitHub 渠道仅涵盖下列两表中 OBBNet、DetectNet 各自的 **iOS Core ML FP32** 和 **Android LiteRT FP32** 转换件，以及固定清单中对应的四个 adapter 和随包许可声明。两表中的源 checkpoint、TorchScript 与 ONNX 用于识别转换来源。训练与测试图片、其他产品及镜像分发均在该项补充授权范围之外。[转换件及 adapter 的固定清单](https://github.com/gcsagroup/watermark/blob/6c33c3cd4dc493fae3d982257782c9a5700e5cf7/packages/model-assets/mobile/research/visible-watermark-obb-native-models.json)

两项确认只授予 GCSA 有权许可的贡献，须保留适用的版权、许可、署名及修改声明。第三方原许可直接授予接收方的权利继续适用。两个完整模型包在 Apple 或 GitHub 分发前，仍须分别落实下列事项。

## DetectNet

Wispio 使用 MobileNetV3-Large 做图像级筛查。GCSA 的确认涉及 `DetectNet/model.py`、`DetectNet/pipeline.py` 中可许可的部分及对应转换件。固定权重文件的训练记录显示 `epoch=30`、`pretrained=True`。推理代码另以 `weights=None` 创建网络再加载项目权重。上游 README 记载训练使用合成水印数据。[固定权重](https://github.com/gcsagroup/WaterMarkModel/blob/411cb342c4f63b039716f55ea9f16c1e29164e11/DetectNet/weights/mobilenet_v3_large.pt) · [推理代码](https://github.com/gcsagroup/WaterMarkModel/blob/411cb342c4f63b039716f55ea9f16c1e29164e11/DetectNet/model.py) · [训练说明](https://github.com/gcsagroup/WaterMarkModel/blob/411cb342c4f63b039716f55ea9f16c1e29164e11/DetectNet/README.en.md#L11)

发行前仍需由训练方列明预训练权重的版本、摘要及条款，说明合成数据所用素材的权利来源，并补齐训练权重及其转换件的授权链。TorchVision 的代码许可适用于实际使用的库代码。TorchVision 提供多个 MobileNetV3-Large 预训练权重，并说明权重可能另受训练数据条款约束；现有记录不足以指定 V1 或 V2 为本模型的训练起点。[TorchVision 代码许可](https://github.com/pytorch/vision/blob/v0.22.0/LICENSE) · [预训练权重条款说明](https://docs.pytorch.org/vision/stable/models.html#general-information-on-pre-trained-weights) · [MobileNetV3-Large 权重版本](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html)

| 资产 | Wispio 固定的 SHA-256 |
| --- | --- |
| `DetectNet/weights/mobilenet_v3_large.pt` | `4ef772ab087d07e0a90dd9b5e9bf2cab8a06a2c239814e93f9d27c645d23677c` |
| `DetectNet/weights/onnx/mobilenet_v3_large_fp32.onnx` | `f6f5a35d54eb8b3731591bab18810aa2ca251a2bfa8a5617210073ec7dc44374` |
| iOS Core ML FP32 | `19435ef00b248978476ebe5b4cf10769af9b3e2c2d222098dcc9d6ebbc4dca41` |
| Android LiteRT FP32 | `d8ad4d175ea83ca026467b8884a972bc48fa4e83c828909eb647609494ae6318` |

## OBBNet

Wispio 使用 YOLO11 OBB 做可见水印定位。GCSA 的确认涉及 `OBBNet/model.py`、`OBBNet/pipeline.py` 中可许可的部分及对应转换件。固定 `best.pt` 文件内直接标注 `AGPL-3.0` 和 Ultralytics 版本 `8.4.120`，并记录 `pretrained=True`、`yolo11m-obb.yaml` 和前一轮训练权重。该配置名与前序权重尚不能确定最初的基础 checkpoint 版本及摘要。[固定权重](https://github.com/gcsagroup/WaterMarkModel/blob/411cb342c4f63b039716f55ea9f16c1e29164e11/OBBNet/weight/best.pt) · [来源记录](https://github.com/gcsagroup/watermark/blob/6c33c3cd4dc493fae3d982257782c9a5700e5cf7/packages/model-assets/tools/visible_watermark_obb/source.lock.json)

发行前须核实基础权重和训练数据来源，并记录 Wispio 如何履行适用的 AGPL 义务，或提供覆盖实际部署及衍生权重的另行授权。上游导出记录把同一份 `best.pt` 的 SHA-256 分别绑定到 TorchScript 与 ONNX 产物。授权应覆盖 Wispio 实际交付的移动端衍生权重及相应分发渠道；仅取得训练工具或托管平台的使用权不足以确认该范围。[TorchScript 导出记录](https://github.com/gcsagroup/WaterMarkModel/blob/411cb342c4f63b039716f55ea9f16c1e29164e11/OBBNet/weight/best_standalone.json) · [ONNX 导出记录](https://github.com/gcsagroup/WaterMarkModel/blob/411cb342c4f63b039716f55ea9f16c1e29164e11/OBBNet/weight/best.json) · [Ultralytics 许可选项](https://www.ultralytics.com/license)

| 资产 | Wispio 固定的 SHA-256 |
| --- | --- |
| `OBBNet/weight/best.pt` | `2c705caa5575cd1413df3d9e9b4b762aeb5207cf3f9a1caebf74af09ebb82585` |
| `OBBNet/weight/best_standalone.pt` | `3084df755258b28b680b15f4426ff133c3f80bd20cb748f2c6e7751d7c44a858` |
| `OBBNet/weight/best.onnx` | `1d2fe56774eb2e73dc5c028eb9fd0c211d036c4b106e985c4a57047879c5f19d` |
| iOS Core ML FP32 | `1f4ffdc72cca64e288759f71fc1d670a52a5e54fb105483b3a46dbc38d99f920` |
| Android LiteRT FP32 | `5f73cfbee3247c7978464c316b37e08694333ce60d5983c8020e104d24e09edd` |

上表固定上游源码 `411cb342c4f63b039716f55ea9f16c1e29164e11`。移动端指纹来自对应转换清单，Core ML 目录使用 `sha256-tree-v1`，其余使用文件 SHA-256。[Wispio 移动端转换清单](https://github.com/gcsagroup/watermark/blob/6c33c3cd4dc493fae3d982257782c9a5700e5cf7/packages/model-assets/mobile/research/visible-watermark-obb-native-models.json)

## 完整模型包的授权记录

请由能确认下列事实的训练方、权利人或授权代表分别补录。每项记录应引用对应的模型摘要，注明签署方、日期、覆盖的产物、渠道和随包声明。保密协议可引用可供发行方核验的编号及适用条款。

| 模型 | 需要取得的记录 |
| --- | --- |
| OBBNet | 由训练方列明基础 checkpoint 的来源、版本、摘要和训练素材权利。若采用 Ultralytics 另行许可，请由许可方确认条款是否覆盖 Wispio 的移动端衍生模型、Apple 托管及公众可直接下载的 GitHub 附件。若按 AGPL 分发，请记录适用范围、对应源码的提供位置、许可及 notices，并核实接收方能行使原许可权利。 |
| DetectNet | 由训练方提供初始化权重的来源、版本、摘要及适用条款，说明用于生成合成样本的素材来源和授权范围，再由有权方确认训练贡献及指定转换件可通过哪些渠道分发。当前 checkpoint 中的 `pretrained=True` 只证实使用了预训练初始化。 |

缺少原始记录时，可使用权利明确的基础模型和素材重新训练，按新产物登记版本、摘要、权利与验证结果。重新导出现有权重不会补足其来源授权。

## Wispio 的第三方模型

以下条款来自各自的发布者。上述 GCSA 渠道授权仅涉及 DetectNet 和 OBBNet 的相应贡献。下列五个移动端检测器继续按各自上游的 MIT 条款分发，须随包保留版权与许可全文。具体转换件和随包声明见 [Wispio 权重及转换记录](https://github.com/gcsagroup/watermark/blob/6c33c3cd4dc493fae3d982257782c9a5700e5cf7/docs/ios/apple_model_asset_rights.md)。

| 移动端检测器 | 代码与权重依据 |
| --- | --- |
| TrustMark P、Q | Adobe 明确 MIT 覆盖代码及首次使用时下载的模型。[代码许可](https://github.com/adobe/trustmark/blob/0ed40cbe8188f664fd9cbbeacd969807de27440a/LICENSE) · [权重说明](https://github.com/adobe/trustmark/blob/59bde8b41c973d2fac3be7ef89e3ac0fea2254fa/README.md) |
| AudioSeal 16-bit | Meta 明确 MIT 覆盖模型权重。[代码许可](https://github.com/facebookresearch/audioseal/blob/e63a8a0e5cdf7bb797159c92ba15961557fe9bd2/LICENSE) · [权重说明](https://github.com/facebookresearch/audioseal/blob/e63a8a0e5cdf7bb797159c92ba15961557fe9bd2/README.md) |
| WavMark default | 代码和实际 checkpoint 的模型卡分别标记 MIT。保留上游许可中原有的版权占位文字。[代码许可](https://github.com/wavmark/wavmark/blob/6ab3bf7ce0679e5b5cfeff3a62e8df9cd2024b37/LICENSE) · [权重模型卡](https://huggingface.co/M4869/WavMark/blob/0ad3c7b74f641bddb61f6b85cdf2de0d93a5bfef/README.md) |
| VideoSeal v0 | Meta 明确模型适用 MIT。Wispio 固定 v0 的转换输入和独立的移动端产物摘要。[代码许可](https://github.com/facebookresearch/videoseal/blob/870ca7fb33578b90f14c602016b6c2788096226e/LICENSE) · [模型说明](https://github.com/facebookresearch/videoseal/blob/870ca7fb33578b90f14c602016b6c2788096226e/README.md) |

VAE Invisible 的分发权限由[专门的 VAE 模型仓库](https://github.com/gcsagroup/Image-Steganography-Invisible-Watermark-Pipeline)维护。Wispio 另有可选的 VideoSeal 1.0 TorchScript 和 Watermark Anything 候选，两者未列入上表的内置移动端清单。Watermark Anything 的 `wam_mit.pth` 使用 MIT，原论文权重有不同的非商业条款，复用时须区分具体 checkpoint。[VideoSeal 1.0 来源](https://github.com/facebookresearch/videoseal/blob/870ca7fb33578b90f14c602016b6c2788096226e/README.md) · [Watermark Anything 权重说明](https://github.com/facebookresearch/watermark-anything/blob/2c08af04d037d5667c02f6ddebbda9ff04581c3e/README.md)
