"""Compatibility wrapper for the original Python reference implementation."""

from nmss.bbox import (
    multiclass_nms_class_aware_cpu,
    multiclass_nms_class_unaware_cpu,
    nms_cpu,
)

__all__ = [
    "multiclass_nms_class_aware_cpu",
    "multiclass_nms_class_unaware_cpu",
    "nms_cpu",
]
