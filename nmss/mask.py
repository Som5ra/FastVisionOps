"""Boolean-mask non-maximum suppression."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import (
    validate_max_detections,
    validate_masks,
    validate_scores,
    validate_threshold,
)


def mask_iou(mask: ArrayLike, masks: ArrayLike) -> NDArray[np.float64]:
    """Return IoU between one boolean mask and a batch of boolean masks."""
    mask_array = np.asarray(mask)
    masks_array = validate_masks(masks)
    if mask_array.dtype != np.bool_:
        raise TypeError(f"mask must have boolean dtype, got {mask_array.dtype}")
    if mask_array.shape != masks_array.shape[1:]:
        raise ValueError(
            "mask spatial shape must match masks, "
            f"got {mask_array.shape} and {masks_array.shape[1:]}"
        )
    if len(masks_array) == 0:
        return np.empty(0, dtype=np.float64)
    flattened = masks_array.reshape(len(masks_array), -1)
    mask_flattened = mask_array.reshape(-1)
    intersection = np.count_nonzero(flattened & mask_flattened, axis=1)
    union = np.count_nonzero(flattened | mask_flattened, axis=1)
    return np.divide(
        intersection,
        union,
        out=np.zeros(len(masks_array), dtype=np.float64),
        where=union > 0,
    )


def mask_nms(
    masks: ArrayLike,
    scores: ArrayLike,
    score_threshold: float = 0.0,
    iou_threshold: float = 0.5,
    *,
    max_detections: int | None = None,
) -> NDArray[np.int64]:
    """Run deterministic single-class NMS over boolean masks."""
    masks_array = validate_masks(masks)
    scores_array = validate_scores(scores, len(masks_array), ndim=1)
    score_threshold = validate_threshold("score_threshold", score_threshold)
    iou_threshold = validate_threshold("iou_threshold", iou_threshold)
    max_detections = validate_max_detections(max_detections)

    candidates = np.flatnonzero(scores_array >= score_threshold)
    if candidates.size == 0 or max_detections == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((candidates, -scores_array[candidates]))
    candidates = candidates[order]

    keep: list[int] = []
    while candidates.size:
        current = int(candidates[0])
        keep.append(current)
        if (
            candidates.size == 1
            or (max_detections is not None and len(keep) >= max_detections)
        ):
            break
        remaining = candidates[1:]
        overlaps = mask_iou(masks_array[current], masks_array[remaining])
        candidates = remaining[overlaps <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def multiclass_mask_nms(
    masks: ArrayLike,
    scores: ArrayLike,
    score_threshold: float = 0.0,
    iou_threshold: float = 0.5,
    *,
    max_detections: int | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Run mask NMS independently per class and sort by score."""
    masks_array = validate_masks(masks)
    scores_array = validate_scores(scores, len(masks_array), ndim=2)
    score_threshold = validate_threshold("score_threshold", score_threshold)
    iou_threshold = validate_threshold("iou_threshold", iou_threshold)
    max_detections = validate_max_detections(max_detections)

    mask_parts: list[NDArray[np.int64]] = []
    class_parts: list[NDArray[np.int64]] = []
    score_parts: list[NDArray[np.float64]] = []
    for class_id in range(scores_array.shape[1]):
        kept = mask_nms(
            masks_array,
            scores_array[:, class_id],
            score_threshold,
            iou_threshold,
        )
        if kept.size:
            mask_parts.append(kept)
            class_parts.append(np.full(kept.size, class_id, dtype=np.int64))
            score_parts.append(scores_array[kept, class_id])

    if not mask_parts or max_detections == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()

    mask_indices = np.concatenate(mask_parts)
    class_ids = np.concatenate(class_parts)
    kept_scores = np.concatenate(score_parts)
    order = np.lexsort((class_ids, mask_indices, -kept_scores))
    if max_detections is not None:
        order = order[:max_detections]
    return mask_indices[order], class_ids[order]


# Backwards-compatible call signatures used by the original script.
def mask_overlap(mask1: ArrayLike, mask2: ArrayLike) -> float:
    return float(mask_iou(mask1, np.asarray([mask2]))[0])


def mask_nms_cpu(
    masks: ArrayLike,
    scores: ArrayLike,
    score_thr: float = 0.5,
    nms_thr: float = 0.5,
) -> NDArray[np.int64]:
    return mask_nms(
        masks,
        scores,
        score_threshold=score_thr,
        iou_threshold=nms_thr,
    )


def multiclass_mask_nms_class_aware_cpu(
    masks: ArrayLike,
    scores: ArrayLike,
    score_thr: float,
    nms_thr: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    return multiclass_mask_nms(
        masks,
        scores,
        score_threshold=score_thr,
        iou_threshold=nms_thr,
    )
