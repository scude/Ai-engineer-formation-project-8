# /scripts/augment.py
from __future__ import annotations

from typing import Tuple

import albumentations as A
import cv2
import numpy as np
import tensorflow as tf

from .config import AugmentConfig
from .data_utils import resize_image_and_mask


class SoftSepia(A.ImageOnlyTransform):
    """Blend a sepia-toned version of the image with the original.

    Albumentations' :class:`~albumentations.augmentations.transforms.ToSepia`
    applies a full-strength sepia filter. This transform keeps the original
    image as the base layer and blends in a sepia version with a randomly
    sampled opacity, resulting in a gentler effect that better matches the
    user's expectations.
    """

    def __init__(self, alpha_range: Tuple[float, float], always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        low, high = alpha_range
        low, high = sorted((float(low), float(high)))
        self.alpha_range = (
            max(0.0, low),
            min(1.0, high),
        )
        self._kernel = np.array(
            [
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131],
            ],
            dtype=np.float32,
        )

    def apply(self, image: np.ndarray, alpha: float = 0.3, **params) -> np.ndarray:  # type: ignore[override]
        sepia_image = image.astype(np.float32) @ self._kernel.T
        sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
        blended = cv2.addWeighted(image, 1.0 - alpha, sepia_image, alpha, 0.0)
        return blended

    def get_params(self) -> dict:
        low, high = self.alpha_range
        alpha = float(np.random.uniform(low, high))
        return {"alpha": alpha}


def _build_albu_pipeline(
    cfg: AugmentConfig, height: int, width: int, ignore_index: int
) -> A.Compose | None:
    if not cfg.enabled:
        return None

    transforms = []

    scale_min = min(max(cfg.random_scale_min, 0.0), 1.0)
    scale_max = min(max(cfg.random_scale_max, scale_min), 1.0)

    if cfg.random_crop or scale_min < 1.0 or scale_max < 1.0:
        # Match the crop aspect ratio to the requested output size so that the
        # subsequent resize step does not stretch the image horizontally or
        # vertically.
        aspect_ratio = float(height) / float(width)
        transforms.append(
            A.RandomResizedCrop(
                size=(height, width),
                scale=(scale_min, scale_max),
                ratio=(aspect_ratio, aspect_ratio),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=1.0,
            )
        )

    if cfg.hflip:
        transforms.append(A.HorizontalFlip(p=0.5))
    if cfg.vflip:
        transforms.append(A.VerticalFlip(p=0.5))

    if cfg.random_rotate_deg > 0:
        transforms.append(
            A.Rotate(
                limit=(-cfg.random_rotate_deg, cfg.random_rotate_deg),
                border_mode=cv2.BORDER_CONSTANT,
                fill=0.0,
                # Fill exposed mask regions with the dataset ignore label so that
                # they can be masked out during loss computation.
                fill_mask=ignore_index,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
            )
        )

    color_params = [
        cfg.brightness_delta,
        cfg.contrast_delta,
        cfg.saturation_delta,
        cfg.hue_delta,
    ]
    if any(delta > 0 for delta in color_params):
        transforms.append(
            A.ColorJitter(
                brightness= cfg.brightness_delta,
                contrast=cfg.contrast_delta,
                saturation=cfg.saturation_delta,
                hue=cfg.hue_delta,
            )
        )

    if cfg.gaussian_noise_std > 0:
        transforms.append(A.GaussNoise(std_range=(0, cfg.gaussian_noise_std), p=1.0))

    if cfg.sepia_probability > 0:
        transforms.append(SoftSepia(alpha_range=(0.25, 0.45), p=cfg.sepia_probability))

    if not transforms:
        transforms.append(A.NoOp(always_apply=True))

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
