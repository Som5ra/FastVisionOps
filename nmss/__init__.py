"""Fast, dependency-light non-maximum suppression utilities."""

from .bbox import (
    bbox_iou,
    multiclass_nms,
    multiclass_nms_class_aware,
    multiclass_nms_class_unaware,
    nms,
)
from .mask import mask_iou, mask_nms, multiclass_mask_nms

__all__ = [
    "bbox_iou",
    "mask_iou",
    "mask_nms",
    "multiclass_mask_nms",
    "multiclass_nms",
    "multiclass_nms_class_aware",
    "multiclass_nms_class_unaware",
    "nms",
]

__version__ = "1.0.0"
