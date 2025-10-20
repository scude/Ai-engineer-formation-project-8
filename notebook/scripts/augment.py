# /scripts/augment.py
from __future__ import annotations

from typing import Tuple

import albumentations as A
import cv2
import numpy as np
import tensorflow as tf

from .config import AugmentConfig
from .data_utils import resize_image_and_mask


def _build_albu_pipeline(
    cfg: AugmentConfig, height: int, width: int, ignore_index: int
) -> A.Compose | None:
    if not cfg.enabled:
        return None

    transforms: list[A.BasicTransform] = []

    raw_min_scale, raw_max_scale = cfg.random_resized_crop_scale
    min_scale, max_scale = sorted((float(raw_min_scale), float(raw_max_scale)))
    min_scale = max(min_scale, 1e-6)
    max_scale = max(max_scale, min_scale)

    # Albumentations expects the crop scale to stay within (0, 1].  Clamping the
    # configured bounds keeps the transform valid while still exposing the
    # caller’s intent when the upper bound is above one.
    max_scale = min(max_scale, 1.0)
    if min_scale > max_scale:
        min_scale = max_scale

    base_ratio = float(width) / float(height)
    lock_ratio = bool(cfg.lock_random_resized_crop_ratio)

    if lock_ratio:
        ratio_min = ratio_max = base_ratio
    else:
        raw_ratio_min, raw_ratio_max = cfg.random_resized_crop_ratio
        raw_ratio_min, raw_ratio_max = sorted(
            (float(raw_ratio_min), float(raw_ratio_max))
        )
        raw_ratio_min = max(raw_ratio_min, 1e-6)
        raw_ratio_max = max(raw_ratio_max, raw_ratio_min)

        ratio_min = base_ratio * raw_ratio_min
        ratio_max = base_ratio * raw_ratio_max

        jitter = max(float(cfg.max_ratio_jitter), 0.0)
        if jitter > 0.0:
            jitter_low = base_ratio * (1.0 - jitter)
            jitter_high = base_ratio * (1.0 + jitter)
            ratio_min = float(np.clip(ratio_min, jitter_low, jitter_high))
            ratio_max = float(np.clip(ratio_max, ratio_min, jitter_high))

    transforms.append(
        A.RandomResizedCrop(
            size=(height, width),
            scale=(float(min_scale), float(max_scale)),
            ratio=(float(ratio_min), float(ratio_max)),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            p=1.0,
        )
    )

    if cfg.horizontal_flip_prob > 0.0:
        transforms.append(A.HorizontalFlip(p=cfg.horizontal_flip_prob))

    # Rotation sans déformation : Pad -> Rotate -> Crop
    if cfg.shift_scale_rotate_prob > 0.0 and cfg.rotate_limit > 0:
        pad_factor = 1.12  # ~12% de marge suffit pour ±5..10°
        transforms.extend([
            A.PadIfNeeded(
                min_height=int(height * pad_factor),
                min_width=int(width * pad_factor),
                border_mode=cv2.BORDER_REFLECT_101,  # évite les bandes noires
                p=1.0,
            ),
            A.Rotate(
                limit=cfg.rotate_limit,  # ex: 5 ou 10 degrés
                border_mode=cv2.BORDER_REFLECT_101,
                p=cfg.shift_scale_rotate_prob,  # proba d’appliquer la rotation
            ),
            A.RandomCrop(height=height, width=width, p=1.0),  # on recadre à (h,w)
        ])

    color_params = [
        cfg.color_jitter_brightness,
        cfg.color_jitter_contrast,
        cfg.color_jitter_saturation,
        cfg.color_jitter_hue,
    ]
    if any(param > 0.0 for param in color_params):
        transforms.append(
            A.ColorJitter(
                brightness=cfg.color_jitter_brightness,
                contrast=cfg.color_jitter_contrast,
                saturation=cfg.color_jitter_saturation,
                hue=cfg.color_jitter_hue,
                p=1.0,
            )
        )

    if cfg.gaussian_blur_prob > 0.0:
        transforms.append(
            A.GaussianBlur(
                blur_limit=cfg.gaussian_blur_kernel,
                p=cfg.gaussian_blur_prob,
            )
        )

    if cfg.gauss_noise_prob > 0.0:
        noise_min, noise_max = cfg.gauss_noise_var_limit
        if noise_min > noise_max:
            noise_min, noise_max = noise_max, noise_min

        noise_min = max(noise_min, 0.0)
        noise_max = max(noise_max, noise_min)

        # Albumentations 2.x expects the Gaussian noise standard deviation to be
        # normalised to the image dynamic range. The configuration expresses the
        # interval directly as pixel-space standard deviations, so we only need
        # to divide by 255 to obtain the expected fraction.
        std_range = (noise_min / 255.0, noise_max / 255.0)

        transforms.append(
            A.GaussNoise(
                std_range=std_range,
                mean_range=(0.0, 0.0),
                p=cfg.gauss_noise_prob,
            )
        )

    if cfg.grid_dropout_prob > 0.0:
        unit_size = max(int(cfg.grid_dropout_unit_size), 2)
        grid_ratio = float(np.clip(cfg.grid_dropout_ratio, 0.0, 1.0))

        transforms.append(
            A.GridDropout(
                ratio=grid_ratio,
                unit_size_range=(unit_size, unit_size + 1),
                random_offset=True,
                fill=0,
                fill_mask=ignore_index,
                p=cfg.grid_dropout_prob,
            )
        )

    return A.Compose(transforms)


def build_augment_fn(cfg: AugmentConfig, h: int, w: int, ignore_index: int):
    """Create an augmentation function for images and masks.

    Args:
        cfg: Augmentation configuration that defines the pipeline.
        h: Target image height.
        w: Target image width.
        ignore_index: Value used to mark pixels that should be ignored during loss
            computation. This value is propagated to Albumentations transforms that
            need it (e.g. rotations).
    """
    pipeline = _build_albu_pipeline(cfg, h, w, ignore_index)

    def _augment_numpy(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        image = np.asarray(image, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.int32)

        if pipeline is None:
            return image, mask

        image_u8 = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
        mask_u8 = np.ascontiguousarray(mask.astype(np.uint8))
        augmented = pipeline(image=image_u8, mask=mask_u8)
        aug_img = augmented["image"].astype(np.float32) / 255.0
        aug_mask = augmented["mask"].astype(np.int32)
        return aug_img, aug_mask

    def _fn(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.cast(y, tf.int32)

        if pipeline is not None:
            aug_x, aug_y = tf.numpy_function(
                func=_augment_numpy,
                inp=[x, y],
                Tout=[tf.float32, tf.int32],
            )
            aug_x = tf.ensure_shape(aug_x, [None, None, 3])
            aug_y = tf.ensure_shape(aug_y, [None, None])
        else:
            aug_x, aug_y = x, y

        resized_x, resized_y = resize_image_and_mask(aug_x, aug_y, h, w)
        resized_x.set_shape((h, w, 3))
        resized_y.set_shape((h, w))
        return resized_x, resized_y

    return _fn
