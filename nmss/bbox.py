"""Backward-compatible bounding-box NMS imports."""

from fastvisionops.postprocess.bbox import (
    bbox_iou,
    multiclass_nms,
    multiclass_nms_class_aware,
    multiclass_nms_class_aware_cpu,
    multiclass_nms_class_unaware,
    multiclass_nms_class_unaware_cpu,
    nms,
    nms_cpu,
)

__all__ = [
    "bbox_iou",
    "multiclass_nms",
    "multiclass_nms_class_aware",
    "multiclass_nms_class_aware_cpu",
    "multiclass_nms_class_unaware",
    "multiclass_nms_class_unaware_cpu",
    "nms",
    "nms_cpu",
]
