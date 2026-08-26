from __future__ import annotations

import math

import torch


def _covariance(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.cat((boxes[:, 2:4].pow(2) / 12, boxes[:, 4:]), dim=-1)
    a, b, angle = values.split(1, dim=-1)
    cos, sin = angle.cos(), angle.sin()
    cos2, sin2 = cos.pow(2), sin.pow(2)
    return a * cos2 + b * sin2, a * sin2 + b * cos2, (a - b) * cos * sin


def batch_probiou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    x1, y1 = boxes1[..., :2].split(1, dim=-1)
    x2, y2 = (value.squeeze(-1)[None] for value in boxes2[..., :2].split(1, dim=-1))
    a1, b1, c1 = _covariance(boxes1)
    a2, b2, c2 = (value.squeeze(-1)[None] for value in _covariance(boxes2))
    denominator = (a1 + a2) * (b1 + b2) - (c1 + c2).pow(2) + eps
    t1 = ((a1 + a2) * (y1 - y2).pow(2) + (b1 + b2) * (x1 - x2).pow(2)) / denominator * 0.25
    t2 = (c1 + c2) * (x2 - x1) * (y1 - y2) / denominator * 0.5
    t3 = (
        denominator
        / (4 * ((a1 * b1 - c1.pow(2)).clamp_(0) * (a2 * b2 - c2.pow(2)).clamp_(0)).sqrt() + eps)
        + eps
    ).log() * 0.5
    distance = (t1 + t2 + t3).clamp(eps, 100.0)
    return 1 - (1.0 - (-distance).exp() + eps).sqrt()


def rotated_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    sorted_boxes = boxes[order]
    overlaps = batch_probiou(sorted_boxes, sorted_boxes).triu_(diagonal=1)
    keep = torch.nonzero((overlaps >= iou_threshold).sum(0) == 0).squeeze(-1)
    return order[keep]


def non_max_suppression_obb(
    prediction: torch.Tensor,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
    max_nms: int,
    class_count: int,
) -> list[torch.Tensor]:
    """Return [cx, cy, w, h, confidence, class_id, angle] for each image."""

    outputs: list[torch.Tensor] = []
    for item in prediction.transpose(1, 2):
        boxes = item[:, :4]
        class_scores = item[:, 4 : 4 + class_count]
        angle = item[:, 4 + class_count : 5 + class_count]
        confidence, class_id = class_scores.max(dim=1)
        keep = confidence > confidence_threshold
        detections = torch.cat(
            (boxes, confidence[:, None], class_id.float()[:, None], angle), dim=1
        )[keep]
        if not len(detections):
            outputs.append(detections)
            continue
        if len(detections) > max_nms:
            detections = detections[detections[:, 4].argsort(descending=True)[:max_nms]]
        class_offset = detections[:, 5:6] * 7680
        nms_boxes = torch.cat(
            (detections[:, :2] + class_offset, detections[:, 2:4], detections[:, 6:7]), dim=1
        )
        selected = rotated_nms(nms_boxes, detections[:, 4], iou_threshold)[:max_detections]
        outputs.append(detections[selected])
    return outputs


def xywhr_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    center = boxes[..., :2]
    width, height, angle = (boxes[..., index : index + 1] for index in range(2, 5))
    cos, sin = angle.cos(), angle.sin()
    vector1 = torch.cat((width / 2 * cos, width / 2 * sin), -1)
    vector2 = torch.cat((-height / 2 * sin, height / 2 * cos), -1)
    return torch.stack(
        (center + vector1 + vector2, center + vector1 - vector2, center - vector1 - vector2, center - vector1 + vector2),
        dim=-2,
    )


def normalize_rbox(box: torch.Tensor) -> torch.Tensor:
    x, y, width, height, angle = box.unbind(dim=-1)
    swap = angle % math.pi >= math.pi / 2
    return torch.stack(
        (x, y, torch.where(swap, height, width), torch.where(swap, width, height), angle % (math.pi / 2)),
        dim=-1,
    )
