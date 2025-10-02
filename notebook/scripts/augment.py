# /scripts/augment.py
from typing import Tuple

import albumentations as A
import cv2
import numpy as np
import tensorflow as tf

from .config import AugmentConfig


class RandomScaleCrop(A.DualTransform):
    """Albumentations transform mimicking the TF random scale+crop pipeline."""

    def __init__(
        self,
        scale_min: float,
        scale_max: float,
        target_height: int,
        target_width: int,
        always_apply: bool = False,
        p: float = 1.0,
    ) -> None:
        super().__init__(always_apply=always_apply, p=p)
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.target_height = target_height
        self.target_width = target_width

    def apply(self, image, scale=1.0, nh=0, nw=0, top=0, left=0, **params):
        return self._apply(image, scale, nh, nw, top, left, cv2.INTER_LINEAR)

    def apply_to_mask(self, mask, scale=1.0, nh=0, nw=0, top=0, left=0, **params):
        return self._apply(mask, scale, nh, nw, top, left, cv2.INTER_NEAREST)

    def _apply(self, arr, scale, nh, nw, top, left, interpolation):
        resized = cv2.resize(arr, (nw, nh), interpolation=interpolation)
        bottom = min(top + self.target_height, nh)
        right = min(left + self.target_width, nw)
        cropped = resized[top:bottom, left:right]
        if cropped.shape[0] != self.target_height or cropped.shape[1] != self.target_width:
            cropped = cv2.resize(
                cropped,
                (self.target_width, self.target_height),
                interpolation=interpolation,
            )
        return cropped

    def get_params_dependent_on_targets(self, params):
        image = params["image"]
        height, width = image.shape[:2]
        scale = float(np.random.uniform(self.scale_min, self.scale_max))
        nh = max(1, int(round(height * scale)))
        nw = max(1, int(round(width * scale)))
        max_top = max(nh - self.target_height, 0)
        max_left = max(nw - self.target_width, 0)
        top = int(np.random.randint(0, max_top + 1)) if max_top > 0 else 0
        left = int(np.random.randint(0, max_left + 1)) if max_left > 0 else 0
        return {"scale": scale, "nh": nh, "nw": nw, "top": top, "left": left}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("scale_min", "scale_max", "target_height", "target_width")


def _build_albu_pipeline(cfg: AugmentConfig, height: int, width: int) -> A.Compose:
    transforms = []

    if not cfg.enabled:
        transforms.append(
            A.Resize(height=height, width=width, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST)
        )
        return A.Compose(transforms)

    scale_min = cfg.random_scale_min
    scale_max = cfg.random_scale_max
    if cfg.random_crop or scale_min != 1.0 or scale_max != 1.0:
        transforms.append(RandomScaleCrop(scale_min, scale_max, height, width))
    else:
        transforms.append(
            A.Resize(height=height, width=width, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST)
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
                value=0.0,
                mask_value=255,
                interpolation=cv2.INTER_LINEAR,
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
                brightness=cfg.brightness_delta,
                contrast=cfg.contrast_delta,
                saturation=cfg.saturation_delta,
                hue=cfg.hue_delta,
            )
        )

    if cfg.gaussian_noise_std > 0:
        std255 = cfg.gaussian_noise_std * 255.0
        transforms.append(A.GaussNoise(var_limit=(0.0, float(std255**2)), mean=0.0))

    transforms.append(
        A.Resize(height=height, width=width, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST)
    )

    return A.Compose(transforms)


def build_augment_fn(cfg: AugmentConfig, h: int, w: int):
    pipeline = _build_albu_pipeline(cfg, h, w)

    def _augment_numpy(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        image = np.asarray(image, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.int32)

        image_scaled = np.clip(image * 255.0, 0.0, 255.0).astype(np.float32)
        mask_for_aug = np.ascontiguousarray(mask.astype(np.uint8))
        augmented = pipeline(image=image_scaled, mask=mask_for_aug)
        aug_img = np.clip(augmented["image"] / 255.0, 0.0, 1.0).astype(np.float32)
        aug_mask = augmented["mask"].astype(np.int32)
        return aug_img, aug_mask

    def _fn(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.cast(y, tf.int32)

        aug_x, aug_y = tf.numpy_function(
            func=_augment_numpy,
            inp=[x, y],
            Tout=[tf.float32, tf.int32],
        )
        aug_x.set_shape((h, w, 3))
        aug_y.set_shape((h, w))
        return aug_x, aug_y

    return _fn
