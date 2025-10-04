"""Utility functions for data preprocessing."""
from __future__ import annotations

from typing import Tuple

import tensorflow as tf


def _prepare_mask(mask: tf.Tensor) -> tf.Tensor:
    mask = tf.convert_to_tensor(mask)
    # Allow masks coming as (H, W, 1) and squeeze the channel dim if present.
    if mask.shape.rank == 3 and mask.shape[-1] == 1:
        mask = tf.squeeze(mask, axis=-1)
    return mask


def resize_image_and_mask(
    image: tf.Tensor,
    mask: tf.Tensor,
    height: int,
    width: int,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Resize an image/mask pair with strict interpolation and validation.

    The image is always resized with bilinear interpolation while the mask uses
    nearest-neighbour to preserve discrete label values. The returned mask is an
    ``int32`` tensor and lightweight assertions verify rank, dtype, and value
    ranges so that misaligned or malformed tensors are surfaced early.
    """

    image = tf.convert_to_tensor(image, dtype=tf.float32)
    mask = _prepare_mask(mask)
    mask = tf.cast(mask, tf.int32)

    tf.debugging.assert_rank(image, 3, message="Image must be rank-3 (H, W, C)")
    tf.debugging.assert_rank(mask, 2, message="Mask must be rank-2 (H, W)")

    resized_image = tf.image.resize(
        image,
        size=[height, width],
        method=tf.image.ResizeMethod.BILINEAR,
        preserve_aspect_ratio=False,
    )
    resized_mask = tf.image.resize(
        mask[..., tf.newaxis],
        size=[height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        preserve_aspect_ratio=False,
    )
    resized_mask = tf.squeeze(resized_mask, axis=-1)
    resized_mask = tf.cast(resized_mask, tf.int32)

    resized_image = tf.ensure_shape(resized_image, [height, width, 3])
    resized_mask = tf.ensure_shape(resized_mask, [height, width])

    tf.debugging.assert_type(resized_image, tf.float32)
    tf.debugging.assert_type(resized_mask, tf.int32)

    min_val = tf.reduce_min(resized_image)
    max_val = tf.reduce_max(resized_image)
    tf.debugging.assert_greater_equal(min_val, 0.0, message="Image values < 0")
    tf.debugging.assert_less_equal(max_val, 1.0, message="Image values > 1")

    return resized_image, resized_mask