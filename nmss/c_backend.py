"""Backward-compatible native API.

Use :mod:`fastvisionops.native` in new code.
"""

from fastvisionops.native import (
    CBackend,
    NativeBackend,
    batch_multiclass_nms,
    load_backend,
    multiclass_nms,
    nms,
)

__all__ = [
    "CBackend",
    "NativeBackend",
    "batch_multiclass_nms",
    "load_backend",
    "multiclass_nms",
    "nms",
]
