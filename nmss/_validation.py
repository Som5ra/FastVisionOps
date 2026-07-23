"""Shared validation helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def validate_threshold(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}")
    return value


def validate_offset(offset: float) -> float:
    offset = float(offset)
    if offset not in (0.0, 1.0):
        raise ValueError(f"offset must be 0 or 1, got {offset!r}")
    return offset


def validate_max_detections(value: int | None) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
    ):
        raise ValueError("max_detections must be a non-negative integer or None")
    return int(value)


def validate_boxes(boxes: ArrayLike) -> NDArray[np.float64]:
    result = np.ascontiguousarray(boxes, dtype=np.float64)
    if result.ndim != 2 or result.shape[1:] != (4,):
        raise ValueError(f"boxes must have shape (N, 4), got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("boxes must contain only finite values")
    if result.size and (
        np.any(result[:, 2] < result[:, 0])
        or np.any(result[:, 3] < result[:, 1])
    ):
        raise ValueError("each box must satisfy x2 >= x1 and y2 >= y1")
    return result


def validate_scores(
    scores: ArrayLike,
    num_items: int,
    *,
    ndim: int,
) -> NDArray[np.float64]:
    result = np.ascontiguousarray(scores, dtype=np.float64)
    if result.ndim != ndim:
        shape = "(N,)" if ndim == 1 else "(N, C)"
        raise ValueError(f"scores must have shape {shape}, got {result.shape}")
    if result.shape[0] != num_items:
        raise ValueError(
            "boxes/masks and scores must contain the same number of items, "
            f"got {num_items} and {result.shape[0]}"
        )
    if ndim == 2 and result.shape[1] == 0:
        raise ValueError("scores must contain at least one class")
    if not np.isfinite(result).all():
        raise ValueError("scores must contain only finite values")
    return result


def validate_masks(masks: ArrayLike) -> NDArray[np.bool_]:
    result = np.asarray(masks)
    if result.ndim < 2:
        raise ValueError(f"masks must have shape (N, ...), got {result.shape}")
    if result.dtype != np.bool_:
        raise TypeError(f"masks must have boolean dtype, got {result.dtype}")
    return np.ascontiguousarray(result)


def validate_batch(
    boxes: Sequence[ArrayLike],
    scores: Sequence[ArrayLike],
) -> None:
    if len(boxes) != len(scores):
        raise ValueError(
            "boxes and scores batches must have equal length, "
            f"got {len(boxes)} and {len(scores)}"
        )
