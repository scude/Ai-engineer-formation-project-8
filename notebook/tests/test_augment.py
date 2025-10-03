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