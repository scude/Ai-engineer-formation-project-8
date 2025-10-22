import albumentations as A
import numpy as np
import pytest
import tensorflow as tf

from notebook.scripts.augment import (
    _build_albu_pipeline,
    build_augment_fn,
    get_bhuiya_transform_specs,
)
from notebook.scripts.config import AugmentConfig


IGNORE_INDEX = 255


def test_pipeline_contains_single_oneof_with_expected_transforms():
    cfg = AugmentConfig(enabled=True)
    pipeline = _build_albu_pipeline(cfg, height=32, width=64, ignore_index=IGNORE_INDEX)

    assert isinstance(pipeline, A.Compose)
    assert len(pipeline.transforms) == 1

    one_of = pipeline.transforms[0]
    assert isinstance(one_of, A.OneOf)
    assert one_of.p == pytest.approx(1.0)

    spec_instances = [spec.build(IGNORE_INDEX) for spec in get_bhuiya_transform_specs()]
    expected_types = {type(instance) for instance in spec_instances}
    actual_types = {type(transform) for transform in one_of.transforms}

    assert actual_types == expected_types


def test_each_transform_preserves_geometry_and_mask_dtype():
    specs = get_bhuiya_transform_specs()
    gradient = np.linspace(0, 255, 64, dtype=np.uint8)
    image = np.stack([np.tile(gradient, (32, 1))] * 3, axis=-1)
    mask = np.zeros((32, 64), dtype=np.uint8)
    mask[:, 16:48] = 1

    for spec in specs:
        transform = spec.build(IGNORE_INDEX)
        augmented = transform(image=image, mask=mask)

        aug_img = augmented["image"]
        aug_mask = augmented["mask"]

        assert aug_img.shape == image.shape
        assert aug_mask.shape == mask.shape
        assert aug_img.dtype == np.uint8
        assert aug_mask.dtype == np.uint8

        if spec.name != "OpticalDistortion":
            np.testing.assert_array_equal(aug_mask, mask)


def test_tensorflow_wrapper_keeps_shapes_and_ranges():
    cfg = AugmentConfig(enabled=True)
    augment_fn = build_augment_fn(cfg, h=32, w=64, ignore_index=IGNORE_INDEX)

    image = np.random.rand(32, 64, 3).astype(np.float32)
    mask = np.zeros((32, 64), dtype=np.uint8)

    aug_image, aug_mask = augment_fn(tf.constant(image), tf.constant(mask))

    assert tuple(aug_image.shape) == (32, 64, 3)
    assert tuple(aug_mask.shape) == (32, 64)
    assert aug_image.dtype == tf.float32
    assert aug_mask.dtype == tf.int32

    aug_np = aug_image.numpy()
    assert np.all(aug_np >= 0.0)
    assert np.all(aug_np <= 1.0)
