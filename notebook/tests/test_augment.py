import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from notebook.scripts.augment import build_augment_fn
from notebook.scripts.config import AugmentConfig


def test_color_jitter_output_not_black():
    cfg = AugmentConfig(
        enabled=True,
        random_crop=False,
        random_scale_min=1.0,
        random_scale_max=1.0,
        brightness_delta=0.2,
        contrast_delta=0.2,
        saturation_delta=0.2,
        hue_delta=0.1,
    )

    augment_fn = build_augment_fn(cfg, h=32, w=32)

    image = np.full((32, 32, 3), 0.5, dtype=np.float32)
    mask = np.zeros((32, 32), dtype=np.uint8)

    aug_image, aug_mask = augment_fn(tf.constant(image), tf.constant(mask))

    aug_image_np = aug_image.numpy()
    aug_mask_np = aug_mask.numpy()

    assert aug_image_np.dtype == np.float32
    assert aug_mask_np.dtype == np.int32
    assert aug_image_np.shape == image.shape
    assert aug_mask_np.shape == mask.shape

    assert np.any(aug_image_np > 0.0), "Augmented image should not be all zeros"
    assert np.any(aug_image_np < 1.0), "Augmented image should not be all ones"

    def test_random_scale_crop_keeps_mask_alignment():
        np.random.seed(1234)

        cfg = AugmentConfig(
            enabled=True,
            random_crop=True,
            random_scale_min=0.5,
            random_scale_max=1.5,
            hflip=False,
            vflip=False,
            random_rotate_deg=0.0,
        )

        augment_fn = build_augment_fn(cfg, h=32, w=32)

        image = np.zeros((32, 32, 3), dtype=np.float32)
        mask = np.zeros((32, 32), dtype=np.uint8)

        image[10:14, 6:10] = 1.0
        mask[10:14, 6:10] = 1

        aug_image, aug_mask = augment_fn(tf.constant(image), tf.constant(mask))

        aug_image_np = aug_image.numpy()
        aug_mask_np = aug_mask.numpy()

        assert aug_image_np.shape == image.shape
        assert aug_mask_np.shape == mask.shape

        mask_positive = aug_mask_np > 0
        assert np.any(mask_positive), "Mask should retain a positive region after augmentation"

        image_gray = aug_image_np.mean(axis=-1)
        max_val = float(image_gray.max())
        peak_coords = np.argwhere(np.isclose(image_gray, max_val, atol=1e-6))

        assert peak_coords.size > 0
        for r, c in peak_coords:
            assert mask_positive[r, c], "Image peak intensity must align with mask"

        assert np.any(mask_positive & (image_gray >= max_val - 1e-6)), "Mask should overlap image peak"
