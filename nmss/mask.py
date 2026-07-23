"""Backward-compatible mask NMS imports."""

from fastvisionops.postprocess.mask import (
    mask_iou,
    mask_nms,
    mask_nms_cpu,
    mask_overlap,
    multiclass_mask_nms,
    multiclass_mask_nms_class_aware_cpu,
)

__all__ = [
    "mask_iou",
    "mask_nms",
    "mask_nms_cpu",
    "mask_overlap",
    "multiclass_mask_nms",
    "multiclass_mask_nms_class_aware_cpu",
]
