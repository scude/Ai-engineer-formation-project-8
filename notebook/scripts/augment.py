# /scripts/augment.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import albumentations as A
import cv2
import numpy as np
import tensorflow as tf

from .config import AugmentConfig
from .data_utils import resize_image_and_mask


@dataclass(frozen=True)
class NamedTransformSpec:
    """Factory describing a Bhuiya et al. (2022) corruption."""

    name: str
    builder: Callable[[int], A.BasicTransform]

    def build(self, ignore_index: int) -> A.BasicTransform:
        return self.builder(ignore_index)


def _bhuiya_transform_specs() -> Sequence[NamedTransformSpec]:
    """Return the Albumentations transforms used in Bhuiya et al. (2022)."""

    return (
        NamedTransformSpec(
            name="Blur",
            builder=lambda _ignore: A.Blur(blur_limit=(10, 25), p=1.0),
        ),
        NamedTransformSpec(
            name="GaussianBlur",
            builder=lambda _ignore: A.GaussianBlur(
                blur_limit=(10, 20), sigma_limit=(2, 4.5), p=1.0
            ),
        ),
        NamedTransformSpec(
            name="GlassBlur",
            builder=lambda _ignore: A.GlassBlur(
                sigma=1.0, max_delta=4, iterations=4, p=1.0
            ),
        ),
        NamedTransformSpec(
            name="MotionBlur",
            builder=lambda _ignore: A.MotionBlur(blur_limit=(15, 30), p=1.0),
        ),
        NamedTransformSpec(
            name="CLAHE",
            builder=lambda _ignore: A.CLAHE(clip_limit=(2.0, 3.0), tile_grid_size=(8, 8), p=1.0),
        ),
        NamedTransformSpec(
            name="Equalize",
            builder=lambda _ignore: A.Equalize(mode="cv", by_channels=True, p=1.0),
        ),
        NamedTransformSpec(
            name="ColorJitter",
            builder=lambda _ignore: A.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=1.0
            ),
        ),
        NamedTransformSpec(
            name="HueSaturationValue",
            builder=lambda _ignore: A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=25,
                val_shift_limit=20,
                p=1.0,
            ),
        ),
        NamedTransformSpec(
            name="Posterize",
            builder=lambda _ignore: A.Posterize(num_bits=3, p=1.0),
        ),
        NamedTransformSpec(
            name="ISONoise",
            builder=lambda _ignore: A.ISONoise(
                color_shift=(0.02, 0.08), intensity=(0.3, 0.40), p=1.0
            ),
        ),
        NamedTransformSpec(
            name="OpticalDistortion",
            builder=lambda ignore: A.OpticalDistortion(
                distort_limit=0.50,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_REFLECT101,
                fill=0,
                fill_mask=int(ignore),
                p=1.0,
            ),
        ),
        NamedTransformSpec(
            name="RandomRain",
            builder=lambda _ignore: A.RandomRain(
                drop_length=24,
                drop_width=1,
                drop_color=(200, 200, 200),
                blur_value=3,
                brightness_coefficient=0.95,
                p=1.0,
            ),
        ),
        NamedTransformSpec(
            name="RandomFog",
            builder=lambda _ignore: A.RandomFog(
                fog_coef_range=(0.4, 0.50),
                alpha_coef=0.15,
                p=1.0,
            ),
        ),
        NamedTransformSpec(
            name="RandomSnow",
            builder=lambda _ignore: A.RandomSnow(
                snow_point_range=(0.3, 0.6),
                brightness_coeff=2,
                p=1.0,
            ),
        ),
        NamedTransformSpec(
            name="RandomSunflare",
            builder=lambda _ignore: A.RandomSunFlare(
                flare_roi=(0.0, 0.0, 1.0, 0.8),
                angle_range=(0.0, 1.0),
                src_radius=600,
                src_color=(255, 240, 200),
                p=1,
            ),
        )

    )


def get_bhuiya_transform_specs() -> Sequence[NamedTransformSpec]:
    """Public accessor for the Bhuiya et al. transform definitions."""

    return _bhuiya_transform_specs()


def _build_albu_pipeline(
    cfg: AugmentConfig, height: int, width: int, ignore_index: int
) -> A.Compose | None:
    if not cfg.enabled:
        return None

    transforms = [spec.build(ignore_index) for spec in _bhuiya_transform_specs()]
    return A.Compose([A.OneOf(transforms, p=1.0)])


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
