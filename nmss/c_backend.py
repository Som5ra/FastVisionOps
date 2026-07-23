"""ctypes bindings for the optional C bounding-box NMS backend."""

from __future__ import annotations

from collections.abc import Sequence
import ctypes
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import os
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer
from numpy.typing import ArrayLike, NDArray

from ._validation import (
    validate_batch,
    validate_boxes,
    validate_offset,
    validate_scores,
    validate_threshold,
)
from .build import DEFAULT_OUTPUT


class CBackend:
    """Loaded native backend with validated NumPy-facing methods."""

    def __init__(self, library: str | os.PathLike[str] = DEFAULT_OUTPUT) -> None:
        library_path = Path(library).resolve()
        if not library_path.is_file():
            raise FileNotFoundError(
                f"native backend not found at {library_path}; "
                "run `python -m nmss.build` first"
            )
        self.library_path = library_path
        self._library = ctypes.CDLL(str(library_path))
        self._library.nmss_nms.argtypes = [
            ndpointer(np.float64, ndim=2, flags="C_CONTIGUOUS"),
            ndpointer(np.float64, ndim=1, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ndpointer(np.int64, ndim=1, flags="C_CONTIGUOUS"),
        ]
        self._library.nmss_nms.restype = ctypes.c_size_t

    def nms(
        self,
        boxes: ArrayLike,
        scores: ArrayLike,
        score_threshold: float = 0.0,
        iou_threshold: float = 0.5,
        *,
        offset: float = 0.0,
        max_detections: int | None = None,
    ) -> NDArray[np.int64]:
        """Run single-class NMS in C."""
        boxes_array = validate_boxes(boxes)
        scores_array = validate_scores(scores, len(boxes_array), ndim=1)
        score_threshold = validate_threshold(
            "score_threshold", score_threshold
        )
        iou_threshold = validate_threshold("iou_threshold", iou_threshold)
        offset = validate_offset(offset)
        if max_detections is not None and max_detections < 0:
            raise ValueError("max_detections must be non-negative or None")
        if len(boxes_array) == 0 or max_detections == 0:
            return np.empty(0, dtype=np.int64)

        output = np.empty(len(boxes_array), dtype=np.int64)
        result_size = self._library.nmss_nms(
            boxes_array,
            scores_array,
            len(boxes_array),
            score_threshold,
            iou_threshold,
            offset,
            output,
        )
        if result_size == ctypes.c_size_t(-1).value:
            raise MemoryError("native NMS could not allocate working memory")
        if max_detections is not None:
            result_size = min(result_size, max_detections)
        return output[:result_size].copy()

    def multiclass_nms(
        self,
        boxes: ArrayLike,
        scores: ArrayLike,
        score_threshold: float = 0.0,
        iou_threshold: float = 0.5,
        *,
        offset: float = 0.0,
        max_detections: int | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Run class-aware NMS in C and globally sort the detections."""
        boxes_array = validate_boxes(boxes)
        scores_array = validate_scores(scores, len(boxes_array), ndim=2)
        if max_detections is not None and max_detections < 0:
            raise ValueError("max_detections must be non-negative or None")

        box_parts: list[NDArray[np.int64]] = []
        class_parts: list[NDArray[np.int64]] = []
        score_parts: list[NDArray[np.float64]] = []
        for class_id in range(scores_array.shape[1]):
            kept = self.nms(
                boxes_array,
                scores_array[:, class_id],
                score_threshold,
                iou_threshold,
                offset=offset,
            )
            if kept.size:
                box_parts.append(kept)
                class_parts.append(
                    np.full(kept.size, class_id, dtype=np.int64)
                )
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

    def batch_multiclass_nms(
        self,
        boxes: Sequence[ArrayLike],
        scores: Sequence[ArrayLike],
        score_threshold: float = 0.0,
        iou_threshold: float = 0.5,
        *,
        offset: float = 0.0,
        max_detections: int | None = None,
        workers: int | None = None,
    ) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
        """Run independent images concurrently."""
        validate_batch(boxes, scores)
        if not boxes:
            return []
        if workers is None:
            workers = min(len(boxes), os.cpu_count() or 1)
        if workers < 1:
            raise ValueError("workers must be at least 1")

        def run(item: tuple[ArrayLike, ArrayLike]):
            image_boxes, image_scores = item
            return self.multiclass_nms(
                image_boxes,
                image_scores,
                score_threshold,
                iou_threshold,
                offset=offset,
                max_detections=max_detections,
            )

        if workers == 1:
            return [run(item) for item in zip(boxes, scores)]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(run, zip(boxes, scores)))


@lru_cache(maxsize=None)
def load_backend(
    library: str | os.PathLike[str] = DEFAULT_OUTPUT,
) -> CBackend:
    """Load and cache a native backend instance."""
    return CBackend(library)


def nms(*args, library: str | os.PathLike[str] = DEFAULT_OUTPUT, **kwargs):
    """Run native single-class NMS with the default or requested library."""
    return load_backend(library).nms(*args, **kwargs)


def multiclass_nms(
    *args,
    library: str | os.PathLike[str] = DEFAULT_OUTPUT,
    **kwargs,
):
    """Run native class-aware NMS with the default or requested library."""
    return load_backend(library).multiclass_nms(*args, **kwargs)


def batch_multiclass_nms(
    *args,
    library: str | os.PathLike[str] = DEFAULT_OUTPUT,
    **kwargs,
):
    """Run native class-aware NMS for a batch."""
    return load_backend(library).batch_multiclass_nms(*args, **kwargs)
