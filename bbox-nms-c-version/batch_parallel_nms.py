"""Compatibility adapter for the original accelerated API."""

from __future__ import annotations

from nmss.c_backend import CBackend


class Batch_Parallel_Nms:
    """Deprecated adapter around :class:`nmss.c_backend.CBackend`."""

    def __init__(self, dll=None) -> None:
        self._backend = CBackend(dll) if dll else CBackend()

    def nms(self, bboxes, scores, score_thr, nms_thr):
        return self._backend.multiclass_nms(
            bboxes,
            scores,
            score_threshold=score_thr,
            iou_threshold=nms_thr,
            offset=1.0,
        )

    def batch_parallel_nms(
        self,
        bboxes,
        scores,
        score_thr,
        nms_thr,
    ):
        results = self._backend.batch_multiclass_nms(
            bboxes,
            scores,
            score_threshold=score_thr,
            iou_threshold=nms_thr,
            offset=1.0,
        )
        return (
            [indices for indices, _ in results],
            [class_ids for _, class_ids in results],
        )
