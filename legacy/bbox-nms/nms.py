"""Compatibility wrapper for the original module path.

New code should import these functions from :mod:`nmss`.
"""

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
