"""Validated NumPy image preprocessing operations."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_image(
    image: ArrayLike,
    *,
    ndim: int,
    layout: str,
) -> NDArray[np.uint8]:
    result = np.asarray(image)
    if result.dtype != np.uint8:
        raise TypeError(f"{layout} input must have uint8 dtype, got {result.dtype}")
    if result.ndim != ndim:
        raise ValueError(
            f"{layout} input must be {ndim}D, got shape {result.shape}"
        )
    if result.shape[-1 if layout in {"HWC", "NHWC"} else 0] == 0:
        raise ValueError(f"{layout} input must contain at least one channel")
    return result


def _validate_statistics(
    mean: ArrayLike,
    std: ArrayLike,
    channels: int,
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    mean_array = np.asarray(mean, dtype=np.float32)
    std_array = np.asarray(std, dtype=np.float32)
    expected_shape = (channels,)
    if mean_array.shape != expected_shape or std_array.shape != expected_shape:
        raise ValueError(
            "mean and std must each have shape "
            f"{expected_shape}, got {mean_array.shape} and {std_array.shape}"
        )
    if not np.isfinite(mean_array).all() or not np.isfinite(std_array).all():
        raise ValueError("mean and std must contain only finite values")
    if np.any(std_array == 0):
        raise ValueError("std values must be non-zero")
    return (
        np.ascontiguousarray(mean_array),
        np.ascontiguousarray(std_array),
    )


def _validate_flip(flip_rb: bool, channels: int) -> bool:
    if not isinstance(flip_rb, (bool, np.bool_)):
        raise TypeError("flip_rb must be a boolean")
    if flip_rb and channels != 3:
        raise ValueError("flip_rb requires exactly three channels")
    return bool(flip_rb)


def hwc_to_chw(
    image: ArrayLike,
    *,
    flip_rb: bool = False,
) -> NDArray[np.uint8]:
    """Convert one uint8 image from HWC to contiguous CHW layout."""
    image_array = _validate_image(image, ndim=3, layout="HWC")
    flip_rb = _validate_flip(flip_rb, image_array.shape[2])
    result = image_array.transpose(2, 0, 1)
    if flip_rb:
        result = result[::-1]
    return np.ascontiguousarray(result)


def chw_channel_normalize(
    image: ArrayLike,
    mean: ArrayLike,
    std: ArrayLike,
    *,
    flip_rb: bool = False,
) -> NDArray[np.float32]:
    """Normalize one uint8 CHW image, optionally reversing three channels."""
    image_array = _validate_image(image, ndim=3, layout="CHW")
    channels = image_array.shape[0]
    mean_array, std_array = _validate_statistics(mean, std, channels)
    flip_rb = _validate_flip(flip_rb, channels)
    result = (
        image_array.astype(np.float32)
        - mean_array[:, np.newaxis, np.newaxis]
    ) / std_array[:, np.newaxis, np.newaxis]
    if flip_rb:
        result = result[::-1]
    return np.ascontiguousarray(result)


def hwc_to_chw_normalize(
    image: ArrayLike,
    mean: ArrayLike,
    std: ArrayLike,
    *,
    flip_rb: bool = False,
) -> NDArray[np.float32]:
    """Fuse uint8 HWC-to-CHW conversion and per-channel normalization."""
    image_array = _validate_image(image, ndim=3, layout="HWC")
    channels = image_array.shape[2]
    mean_array, std_array = _validate_statistics(mean, std, channels)
    flip_rb = _validate_flip(flip_rb, channels)
    result = (
        image_array.astype(np.float32)
        - mean_array[np.newaxis, np.newaxis, :]
    ) / std_array[np.newaxis, np.newaxis, :]
    result = result.transpose(2, 0, 1)
    if flip_rb:
        result = result[::-1]
    return np.ascontiguousarray(result)


def hwc_to_chw_normalize_batched(
    images: ArrayLike,
    mean: ArrayLike,
    std: ArrayLike,
    *,
    flip_rb: bool = False,
) -> NDArray[np.float32]:
    """Fuse uint8 NHWC-to-NCHW conversion and channel normalization."""
    image_array = _validate_image(images, ndim=4, layout="NHWC")
    channels = image_array.shape[3]
    mean_array, std_array = _validate_statistics(mean, std, channels)
    flip_rb = _validate_flip(flip_rb, channels)
    result = (
        image_array.astype(np.float32)
        - mean_array[np.newaxis, np.newaxis, np.newaxis, :]
    ) / std_array[np.newaxis, np.newaxis, np.newaxis, :]
    result = result.transpose(0, 3, 1, 2)
    if flip_rb:
        result = result[:, ::-1]
    return np.ascontiguousarray(result)
