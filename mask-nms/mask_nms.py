"""Compatibility wrapper for the original module path."""

from nmss.mask import (
    mask_nms_cpu,
    mask_overlap,
    multiclass_mask_nms_class_aware_cpu,
)

__all__ = [
    "mask_nms_cpu",
    "mask_overlap",
    "multiclass_mask_nms_class_aware_cpu",
]
