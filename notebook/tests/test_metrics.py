import pytest

tf = pytest.importorskip("tensorflow")

from notebook.scripts.metrics import MaskedMeanIoU, masked_mean_iou


def test_masked_mean_iou_serialization_preserves_ignore_index():
    metric = MaskedMeanIoU(num_classes=3, ignore_index=255, dtype=tf.float32)

    config = metric.get_config()
    rebuilt_metric = MaskedMeanIoU.from_config(config)

    assert int(rebuilt_metric.ignore_index.numpy()) == 255


def test_masked_mean_iou_factory_creates_metric_with_expected_config():
    factory = masked_mean_iou(num_classes=4, ignore_index=7, name="custom_mIoU", dtype=tf.float64)

    metric = factory()

    assert isinstance(metric, MaskedMeanIoU)
    assert metric.name == "custom_mIoU"
    assert metric.dtype == tf.float64
    assert int(metric.ignore_index.numpy()) == 7