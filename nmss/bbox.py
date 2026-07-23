"""Bounding-box non-maximum suppression."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import (
    validate_boxes,
    validate_offset,
    validate_scores,
    validate_threshold,
)


def bbox_iou(
    box: ArrayLike,
    boxes: ArrayLike,
    *,
    offset: float = 0.0,
) -> NDArray[np.float64]:
    """Return IoU between one ``xyxy`` box and an array of ``xyxy`` boxes."""
    offset = validate_offset(offset)
    box_array = np.asarray(box, dtype=np.float64)
    if box_array.shape != (4,) or not np.isfinite(box_array).all():
        raise ValueError("box must contain four finite xyxy coordinates")
    boxes_array = validate_boxes(boxes)
    if box_array[2] < box_array[0] or box_array[3] < box_array[1]:
        raise ValueError("box must satisfy x2 >= x1 and y2 >= y1")

    top_left = np.maximum(box_array[:2], boxes_array[:, :2])
    bottom_right = np.minimum(box_array[2:], boxes_array[:, 2:])
    intersection_size = np.maximum(0.0, bottom_right - top_left + offset)
    intersection = intersection_size[:, 0] * intersection_size[:, 1]

    box_size = box_array[2:] - box_array[:2] + offset
    boxes_size = boxes_array[:, 2:] - boxes_array[:, :2] + offset
    box_area = box_size[0] * box_size[1]
    boxes_area = boxes_size[:, 0] * boxes_size[:, 1]
    union = box_area + boxes_area - intersection

    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def nms(
    boxes: ArrayLike,
    scores: ArrayLike,
    score_threshold: float = 0.0,
    iou_threshold: float = 0.5,
    *,
    offset: float = 0.0,
    max_detections: int | None = None,
) -> NDArray[np.int64]:
    """Run deterministic single-class greedy NMS.

    Scores equal to ``score_threshold`` are retained. Equal-score boxes are
    processed in original index order.
    """
    boxes_array = validate_boxes(boxes)
    scores_array = validate_scores(scores, len(boxes_array), ndim=1)
    score_threshold = validate_threshold("score_threshold", score_threshold)
    iou_threshold = validate_threshold("iou_threshold", iou_threshold)
    offset = validate_offset(offset)
    if max_detections is not None and max_detections < 0:
        raise ValueError("max_detections must be non-negative or None")

    candidate_indices = np.flatnonzero(scores_array >= score_threshold)
    if candidate_indices.size == 0 or max_detections == 0:
        return np.empty(0, dtype=np.int64)

    # lexsort uses the last key as primary: descending score, then index.
    order = np.lexsort(
        (candidate_indices, -scores_array[candidate_indices])
    )
    candidate_indices = candidate_indices[order]

    keep: list[int] = []
    while candidate_indices.size:
        current = int(candidate_indices[0])
        keep.append(current)
        if (
            candidate_indices.size == 1
            or (max_detections is not None and len(keep) >= max_detections)
        ):
            break
        remaining = candidate_indices[1:]
        overlaps = bbox_iou(
            boxes_array[current],
            boxes_array[remaining],
            offset=offset,
        )
        candidate_indices = remaining[overlaps <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)


def multiclass_nms_class_aware(
    boxes: ArrayLike,
    scores: ArrayLike,
    score_threshold: float = 0.0,
    iou_threshold: float = 0.5,
    *,
    offset: float = 0.0,
    max_detections: int | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Run NMS independently per class and sort all results by score."""
    boxes_array = validate_boxes(boxes)
    scores_array = validate_scores(scores, len(boxes_array), ndim=2)
    score_threshold = validate_threshold("score_threshold", score_threshold)
    iou_threshold = validate_threshold("iou_threshold", iou_threshold)
    offset = validate_offset(offset)
    if max_detections is not None and max_detections < 0:
        raise ValueError("max_detections must be non-negative or None")

    box_parts: list[NDArray[np.int64]] = []
    class_parts: list[NDArray[np.int64]] = []
    score_parts: list[NDArray[np.float64]] = []
    for class_id in range(scores_array.shape[1]):
        kept = nms(
            boxes_array,
            scores_array[:, class_id],
            score_threshold,
            iou_threshold,
            offset=offset,
        )
        if kept.size:
            box_parts.append(kept)
            class_parts.append(np.full(kept.size, class_id, dtype=np.int64))
            score_parts.append(scores_array[kept, class_id])

    if not box_parts or max_detections == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()

    box_indices = np.concatenate(box_parts)
    class_ids = np.concatenate(class_parts)
    kept_scores = np.concatenate(score_parts)
    order = np.lexsort((class_ids, box_indices, -kept_scores))
    if max_detections is not None:
        order = order[:max_detections]
    return box_indices[order], class_ids[order]


def multiclass_nms_class_unaware(
    boxes: ArrayLike,
    scores: ArrayLike,
    score_threshold: float = 0.0,
    iou_threshold: float = 0.5,
    *,
    offset: float = 0.0,
    max_detections: int | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Assign each box to its best class, then suppress across all classes."""
    boxes_array = validate_boxes(boxes)
    scores_array = validate_scores(scores, len(boxes_array), ndim=2)
    score_threshold = validate_threshold("score_threshold", score_threshold)
    iou_threshold = validate_threshold("iou_threshold", iou_threshold)
    offset = validate_offset(offset)
    if max_detections is not None and max_detections < 0:
        raise ValueError("max_detections must be non-negative or None")
    if len(boxes_array) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()
    class_ids = np.argmax(scores_array, axis=1).astype(np.int64, copy=False)
    best_scores = scores_array[np.arange(len(scores_array)), class_ids]
    kept = nms(
        boxes_array,
        best_scores,
        score_threshold,
        iou_threshold,
        offset=offset,
        max_detections=max_detections,
    )
    return kept, class_ids[kept]


def multiclass_nms(
    boxes: ArrayLike,
    scores: ArrayLike,
    score_threshold: float = 0.0,
    iou_threshold: float = 0.5,
    *,
    class_aware: bool = True,
    offset: float = 0.0,
    max_detections: int | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Run class-aware or class-unaware bounding-box NMS."""
    implementation = (
        multiclass_nms_class_aware
        if class_aware
        else multiclass_nms_class_unaware
    )
    return implementation(
        boxes,
        scores,
        score_threshold,
        iou_threshold,
        offset=offset,
        max_detections=max_detections,
    )


# Backwards-compatible call signatures used by the original scripts.
def nms_cpu(
    boxes: ArrayLike,
    scores: ArrayLike,
    score_thr: float,
    nms_thr: float,
) -> NDArray[np.int64]:
    return nms(
        boxes,
        scores,
        score_threshold=score_thr,
        iou_threshold=nms_thr,
        offset=1.0,
    )


def multiclass_nms_class_aware_cpu(
    boxes: ArrayLike,
    scores: ArrayLike,
    score_thr: float,
    nms_thr: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    return multiclass_nms_class_aware(
        boxes,
        scores,
        score_threshold=score_thr,
        iou_threshold=nms_thr,
        offset=1.0,
    )


def multiclass_nms_class_unaware_cpu(
    boxes: ArrayLike,
    scores: ArrayLike,
    score_thr: float,
    nms_thr: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    return multiclass_nms_class_unaware(
        boxes,
        scores,
        score_threshold=score_thr,
        iou_threshold=nms_thr,
        offset=1.0,
    )
