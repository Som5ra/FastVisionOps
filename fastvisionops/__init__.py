"""Fast, validated CPU operations for computer-vision inference."""

from typing import TYPE_CHECKING

from .postprocess.bbox import (
    bbox_iou,
    multiclass_nms,
    multiclass_nms_class_aware,
    multiclass_nms_class_unaware,
    nms,
)
from .postprocess.mask import mask_iou, mask_nms, multiclass_mask_nms

from .preprocess import (
    chw_channel_normalize,
    hwc_to_chw,
    hwc_to_chw_normalize,
    hwc_to_chw_normalize_batched,
)

if TYPE_CHECKING:
    from .native import NativeBackend

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
    "NativeBackend",
    "nms",
]

__version__ = "1.0.0"


def __getattr__(name: str):
    if name == "NativeBackend":
        from .native import NativeBackend

        return NativeBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
