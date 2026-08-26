from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

import torch
from torch import nn


class Conv(nn.Module):
    """Convolution block whose attributes are restored from the checkpoint."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class DWConv(Conv):
    pass


class Bottleneck(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k2(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(module(y[-1]) for module in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(getattr(self, "n", 3)))
        output = self.cv2(torch.cat(y, 1))
        return output + x if getattr(self, "add", False) else output


class Attention(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        count = height * width
        q, k, v = self.qkv(x).view(
            batch, self.num_heads, self.key_dim * 2 + self.head_dim, count
        ).split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        attention = ((q * self.scale).transpose(-2, -1) @ k).softmax(dim=-1)
        output = (v @ attention.transpose(-2, -1)).view(batch, channels, height, width)
        output = output + self.pe(v.reshape(batch, channels, height, width))
        return self.proj(output)


class PSABlock(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) if self.add else self.attn(x)
        return x + self.ffn(x) if self.add else self.ffn(x)


class C2PSA(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        return self.cv2(torch.cat((a, self.m(b)), 1))


class Concat(nn.Module):
    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, self.d)


class DFL(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, anchors = x.shape
        return self.conv(
            x.view(batch, 4, self.c1, anchors).transpose(2, 1).softmax(1)
        ).view(batch, 4, anchors)


def make_anchors(
    features: list[torch.Tensor], strides: torch.Tensor, offset: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    anchor_points: list[torch.Tensor] = []
    stride_tensors: list[torch.Tensor] = []
    dtype = features[0].dtype
    device = features[0].device
    for index, feature in enumerate(features):
        height, width = feature.shape[2:]
        sx = torch.arange(width, device=device, dtype=dtype) + offset
        sy = torch.arange(height, device=device, dtype=dtype) + offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensors.append(
            torch.full((height * width, 1), strides[index], device=device, dtype=dtype)
        )
    return torch.cat(anchor_points), torch.cat(stride_tensors)


def dist2rbox(
    distances: torch.Tensor,
    angle: torch.Tensor,
    anchor_points: torch.Tensor,
    dim: int = 1,
) -> torch.Tensor:
    left_top, right_bottom = distances.split(2, dim=dim)
    cos, sin = torch.cos(angle), torch.sin(angle)
    xf, yf = ((right_bottom - left_top) / 2).split(1, dim=dim)
    x = xf * cos - yf * sin
    y = xf * sin + yf * cos
    center = torch.cat((x, y), dim=dim) + anchor_points
    return torch.cat((center, left_top + right_bottom), dim=dim)


class Detect(nn.Module):
    dynamic = False
    export = False
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    @property
    def end2end(self) -> bool:
        return getattr(self, "_end2end", True) and hasattr(self, "one2one_cv2")

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: nn.Module | None = None,
        cls_head: nn.Module | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if box_head is None or cls_head is None:
            return {}
        batch = x[0].shape[0]
        boxes = torch.cat(
            [box_head[i](x[i]).view(batch, 4 * self.reg_max, -1) for i in range(self.nl)],
            dim=-1,
        )
        scores = torch.cat(
            [cls_head[i](x[i]).view(batch, self.nc, -1) for i in range(self.nl)], dim=-1
        )
        return {"boxes": boxes, "scores": scores, "feats": x}

    def _decode_boxes(self, predictions: dict[str, Any]) -> torch.Tensor:
        features = predictions["feats"]
        current_shape = features[0].shape
        if self.dynamic or self.shape != current_shape:
            anchors, strides = make_anchors(features, self.stride, 0.5)
            self.anchors, self.strides = anchors.transpose(0, 1), strides.transpose(0, 1)
            self.shape = current_shape
        boxes = self.decode_bboxes(self.dfl(predictions["boxes"]), self.anchors.unsqueeze(0))
        return boxes * self.strides

    def _inference(self, predictions: dict[str, Any]) -> torch.Tensor:
        return torch.cat((self._decode_boxes(predictions), predictions["scores"].sigmoid()), 1)


class OBB(Detect):
    @property
    def one2many(self) -> dict[str, nn.Module]:
        return {"box_head": self.cv2, "cls_head": self.cv3, "angle_head": self.cv4}

    def forward_head(
        self,
        x: list[torch.Tensor],
        box_head: nn.Module | None = None,
        cls_head: nn.Module | None = None,
        angle_head: nn.Module | None = None,
    ) -> dict[str, Any]:
        predictions = super().forward_head(x, box_head, cls_head)
        if angle_head is not None:
            batch = x[0].shape[0]
            angle = torch.cat(
                [angle_head[i](x[i]).view(batch, self.ne, -1) for i in range(self.nl)], 2
            )
            predictions["angle"] = (angle.sigmoid() - 0.25) * math.pi
        return predictions

    def decode_bboxes(self, boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        return dist2rbox(boxes, self.angle, anchors, dim=1)

    def _inference(self, predictions: dict[str, Any]) -> torch.Tensor:
        self.angle = predictions["angle"]
        return torch.cat((super()._inference(predictions), self.angle), dim=1)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        predictions = self.forward_head(x, **self.one2many)
        if self.training:
            return predictions
        output = self._inference(predictions)
        return output if self.export else (output, predictions)


class OBBModel(nn.Module):
    """Minimal inference-only replacement for Ultralytics OBBModel."""

    def forward(self, x: torch.Tensor) -> Any:
        outputs: list[torch.Tensor | None] = []
        for module in self.model:
            if module.f != -1:
                x = (
                    outputs[module.f]
                    if isinstance(module.f, int)
                    else [x if index == -1 else outputs[index] for index in module.f]
                )
            x = module(x)
            outputs.append(x if module.i in self.save else None)
        return x


_CLASS_MAP: dict[tuple[str, str], type] = {
    ("ultralytics.nn.tasks", "OBBModel"): OBBModel,
    ("ultralytics.nn.modules.conv", "Conv"): Conv,
    ("ultralytics.nn.modules.conv", "DWConv"): DWConv,
    ("ultralytics.nn.modules.conv", "Concat"): Concat,
    ("ultralytics.nn.modules.block", "Bottleneck"): Bottleneck,
    ("ultralytics.nn.modules.block", "C3k"): C3k,
    ("ultralytics.nn.modules.block", "C3k2"): C3k2,
    ("ultralytics.nn.modules.block", "SPPF"): SPPF,
    ("ultralytics.nn.modules.block", "Attention"): Attention,
    ("ultralytics.nn.modules.block", "PSABlock"): PSABlock,
    ("ultralytics.nn.modules.block", "C2PSA"): C2PSA,
    ("ultralytics.nn.modules.block", "DFL"): DFL,
    ("ultralytics.nn.modules.head", "OBB"): OBB,
}


class _StandaloneUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        replacement = _CLASS_MAP.get((module, name))
        if replacement is not None:
            return replacement
        if module.startswith("ultralytics"):
            raise pickle.UnpicklingError(
                f"Unsupported Ultralytics checkpoint class {module}.{name}. "
                "This pipeline currently supports compatible OBB checkpoints."
            )
        return super().find_class(module, name)


class _StandalonePickleModule:
    __name__ = "pickle"
    Unpickler = _StandaloneUnpickler
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)
    dump = staticmethod(pickle.dump)
    dumps = staticmethod(pickle.dumps)


def load_obb_checkpoint(weights: str | Path, device: torch.device) -> nn.Module:
    """Load TorchScript or a trusted OBB training checkpoint."""

    weights = Path(weights).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Weight file not found: {weights}")
    try:
        scripted = torch.jit.load(str(weights), map_location=device)
    except (RuntimeError, ValueError):
        scripted = None
    if scripted is not None:
        return scripted.eval()

    checkpoint = torch.load(
        weights,
        map_location="cpu",
        pickle_module=_StandalonePickleModule,
        weights_only=False,
    )
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        model = checkpoint.get("ema") or checkpoint.get("model")
    else:
        model = None
    if not isinstance(model, OBBModel):
        raise TypeError(
            f"{weights} does not contain a supported OBB model; got {type(model).__name__}"
        )
    model.to(device).eval()
    return model
