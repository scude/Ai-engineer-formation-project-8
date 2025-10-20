import albumentations as A
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from notebook.scripts.augment import _build_albu_pipeline, build_augment_fn
from notebook.scripts.config import AugmentConfig
from notebook.scripts.data import make_weights


IGNORE_INDEX = 255


def test_color_jitter_output_not_black():
    cfg = AugmentConfig(
        enabled=True,
        random_resized_crop_scale=(1.0, 1.0),
        random_resized_crop_ratio=(1.0, 1.0),
        horizontal_flip_prob=0.0,
        shift_scale_rotate_prob=0.0,
        gaussian_blur_prob=0.0,
        gauss_noise_prob=0.0,
        grid_dropout_prob=0.0,
        color_jitter_brightness=0.2,
        color_jitter_contrast=0.2,
        color_jitter_saturation=0.2,
        color_jitter_hue=0.1,
    )

    augment_fn = build_augment_fn(cfg, h=32, w=32, ignore_index=IGNORE_INDEX)

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


def test_random_resized_crop_keeps_mask_alignment():
    np.random.seed(1234)

    cfg = AugmentConfig(
        enabled=True,
        random_resized_crop_scale=(0.5, 1.0),
        random_resized_crop_ratio=(0.75, 1.33),
        horizontal_flip_prob=0.0,
        shift_scale_rotate_prob=0.0,
        gaussian_blur_prob=0.0,
        gauss_noise_prob=0.0,
        grid_dropout_prob=0.0,
    )

    augment_fn = build_augment_fn(cfg, h=32, w=32, ignore_index=IGNORE_INDEX)

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
    rows, cols = peak_coords.T
    assert np.any(mask_positive[rows, cols]), "At least one image peak must overlap the mask"

    assert np.any(mask_positive & (image_gray >= max_val - 1e-3)), "Mask should overlap image peak"


def test_locked_ratio_pipeline_uses_fixed_ratio_crop():
    cfg = AugmentConfig(
        enabled=True,
        random_resized_crop_scale=(0.5, 2.0),
        random_resized_crop_ratio=(0.5, 1.5),
        max_ratio_jitter=0.25,
        horizontal_flip_prob=0.0,
        shift_scale_rotate_prob=0.0,
        gaussian_blur_prob=0.0,
        gauss_noise_prob=0.0,
        grid_dropout_prob=0.0,
    )

    pipeline = _build_albu_pipeline(cfg, height=32, width=64, ignore_index=IGNORE_INDEX)
    assert pipeline is not None

    crop = next(t for t in pipeline.transforms if isinstance(t, A.RandomResizedCrop))

    assert crop.size == (32, 64)
    assert crop.scale == pytest.approx((0.5, 1.0))

    base_ratio = 64 / 32
    assert crop.ratio == pytest.approx((base_ratio, base_ratio))


def test_locked_ratio_pipeline_emits_fixed_geometry():
    cfg = AugmentConfig(
        enabled=True,
        random_resized_crop_scale=(0.5, 2.0),
        horizontal_flip_prob=0.0,
        shift_scale_rotate_prob=0.0,
        gaussian_blur_prob=0.0,
        gauss_noise_prob=0.0,
        grid_dropout_prob=0.0,
    )

    pipeline = _build_albu_pipeline(cfg, height=32, width=64, ignore_index=IGNORE_INDEX)
    assert pipeline is not None

    image = np.zeros((70, 90, 3), dtype=np.uint8)
    mask = np.zeros((70, 90), dtype=np.uint8)

    for _ in range(5):
        augmented = pipeline(image=image, mask=mask)
        aug_image = augmented["image"]
        aug_mask = augmented["mask"]

        assert aug_image.shape[:2] == (32, 64)
        assert aug_mask.shape == (32, 64)
        assert aug_mask.dtype == np.uint8


def test_random_resized_crop_ratio_clamps_to_jitter_budget():
    cfg = AugmentConfig(
        enabled=True,
        random_resized_crop_scale=(1.0, 1.0),
        random_resized_crop_ratio=(0.5, 1.5),
        max_ratio_jitter=0.05,
        lock_random_resized_crop_ratio=False,
        horizontal_flip_prob=0.0,
        shift_scale_rotate_prob=0.0,
        gaussian_blur_prob=0.0,
        gauss_noise_prob=0.0,
        grid_dropout_prob=0.0,
    )

    pipeline = _build_albu_pipeline(cfg, height=32, width=64, ignore_index=IGNORE_INDEX)
    assert pipeline is not None

    rrc = next(t for t in pipeline.transforms if isinstance(t, A.RandomResizedCrop))

    base_ratio = 64 / 32
    expected_min = pytest.approx(base_ratio * (1.0 - cfg.max_ratio_jitter))
    expected_max = pytest.approx(base_ratio * (1.0 + cfg.max_ratio_jitter))

    actual_min, actual_max = rrc.ratio

    assert actual_min == expected_min
    assert actual_max == expected_max


def test_random_resized_crop_ratio_respects_requested_window_when_safe():
    cfg = AugmentConfig(
        enabled=True,
        random_resized_crop_scale=(1.0, 1.0),
        random_resized_crop_ratio=(0.95, 1.05),
        max_ratio_jitter=0.2,
        lock_random_resized_crop_ratio=False,
        horizontal_flip_prob=0.0,
        shift_scale_rotate_prob=0.0,
        gaussian_blur_prob=0.0,
        gauss_noise_prob=0.0,
        grid_dropout_prob=0.0,
    )

    pipeline = _build_albu_pipeline(cfg, height=32, width=64, ignore_index=IGNORE_INDEX)
    assert pipeline is not None

    rrc = next(t for t in pipeline.transforms if isinstance(t, A.RandomResizedCrop))

    base_ratio = 64 / 32
    expected_min = pytest.approx(base_ratio * 0.95)
    expected_max = pytest.approx(base_ratio * 1.05)

    actual_min, actual_max = rrc.ratio

    assert actual_min == expected_min
    assert actual_max == expected_max


def test_locked_ratio_pipeline_clamps_scales_above_one():
    cfg = AugmentConfig(
        enabled=True,
        random_resized_crop_scale=(0.8, 1.5),
        horizontal_flip_prob=0.0,
        shift_scale_rotate_prob=0.0,
        gaussian_blur_prob=0.0,
        gauss_noise_prob=0.0,
        grid_dropout_prob=0.0,
    )

    pipeline = _build_albu_pipeline(cfg, height=32, width=64, ignore_index=IGNORE_INDEX)
    assert pipeline is not None

    crop = next(t for t in pipeline.transforms if isinstance(t, A.RandomResizedCrop))

    assert crop.scale == pytest.approx((0.8, 1.0))


def test_gauss_noise_var_limit_translates_to_std_range():
    cfg = AugmentConfig(
        enabled=True,
        random_resized_crop_scale=(1.0, 1.0),
        random_resized_crop_ratio=(1.0, 1.0),
        horizontal_flip_prob=0.0,
        shift_scale_rotate_prob=0.0,
        gaussian_blur_prob=0.0,
        gauss_noise_prob=1.0,
        gauss_noise_var_limit=(4.0, 9.0),
        grid_dropout_prob=0.0,
    )

    pipeline = _build_albu_pipeline(cfg, height=32, width=64, ignore_index=IGNORE_INDEX)
    assert pipeline is not None

    noise = next(t for t in pipeline.transforms if isinstance(t, A.GaussNoise))

    assert noise.std_range == pytest.approx(
        (
            4.0 / 255.0,
            9.0 / 255.0,
        )
    )
    assert noise.mean_range == (0.0, 0.0)


def test_grid_dropout_uses_unit_size_range_and_mask_fill():
    cfg = AugmentConfig(
        enabled=True,
        random_resized_crop_scale=(1.0, 1.0),
        random_resized_crop_ratio=(1.0, 1.0),
        horizontal_flip_prob=0.0,
        shift_scale_rotate_prob=0.0,
        gaussian_blur_prob=0.0,
        gauss_noise_prob=0.0,
        grid_dropout_prob=1.0,
        grid_dropout_ratio=0.65,
        grid_dropout_unit_size=37,
    )

    pipeline = _build_albu_pipeline(cfg, height=32, width=64, ignore_index=IGNORE_INDEX)
    assert pipeline is not None

    grid = next(t for t in pipeline.transforms if isinstance(t, A.GridDropout))

    assert grid.ratio == pytest.approx(0.65)
    assert grid.unit_size_range == (37, 38)
    assert grid.random_offset is True
    assert grid.fill == 0
    assert grid.fill_mask == IGNORE_INDEX


def test_rotation_masks_fill_with_ignore_label_and_zero_weights():
    np.random.seed(42)

    cfg = AugmentConfig(
        enabled=True,
        random_resized_crop_scale=(1.0, 1.0),
        random_resized_crop_ratio=(1.0, 1.0),
        horizontal_flip_prob=0.0,
        shift_scale_rotate_prob=1.0,
        shift_limit=0.0,
        scale_limit=0.0,
        rotate_limit=45.0,
        gaussian_blur_prob=0.0,
        gauss_noise_prob=0.0,
        grid_dropout_prob=0.0,
    )

    augment_fn = build_augment_fn(cfg, h=32, w=32, ignore_index=IGNORE_INDEX)

    image = np.zeros((32, 32, 3), dtype=np.float32)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1

    for _ in range(10):
        _, aug_mask = augment_fn(tf.constant(image), tf.constant(mask))
        aug_mask_np = aug_mask.numpy()
        unique_values = np.unique(aug_mask_np)

        assert unique_values.min() >= 0
        if IGNORE_INDEX in unique_values:
            newly_exposed = aug_mask_np == IGNORE_INDEX
            assert np.any(newly_exposed), "Rotation should create ignored pixels"

            weights = make_weights(tf.constant(aug_mask_np, dtype=tf.int32), IGNORE_INDEX)
            weights_np = weights.numpy()

            assert np.all(weights_np[newly_exposed] == 0.0)
            assert np.all(weights_np[~newly_exposed] == 1.0)
            break
    else:
        pytest.fail("Rotate did not produce ignore-index pixels within iterations")
