"""Fast, validated CPU operations for computer-vision inference."""

from nmss.bbox import (
    bbox_iou,
    multiclass_nms,
    multiclass_nms_class_aware,
    multiclass_nms_class_unaware,
    nms,
)
from nmss.mask import mask_iou, mask_nms, multiclass_mask_nms

from .preprocess import (
    chw_channel_normalize,
    hwc_to_chw,
    hwc_to_chw_normalize,
    hwc_to_chw_normalize_batched,
)

__all__ = [
    "bbox_iou",
    "chw_channel_normalize",
    "hwc_to_chw",
    "hwc_to_chw_normalize",
    "hwc_to_chw_normalize_batched",
    "mask_iou",
    "mask_nms",
    "multiclass_mask_nms",
    "multiclass_nms",
    "multiclass_nms_class_aware",
    "multiclass_nms_class_unaware",
    "nms",
]

__version__ = "1.0.0"
