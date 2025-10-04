import pytest

tf = pytest.importorskip("tensorflow")

from notebook.scripts.metrics import MaskedMeanIoU


def test_masked_mean_iou_serialization_preserves_ignore_index():
    metric = MaskedMeanIoU(num_classes=3, ignore_index=255, dtype=tf.float32)

    config = metric.get_config()
    rebuilt_metric = MaskedMeanIoU.from_config(config)

    assert int(rebuilt_metric.ignore_index.numpy()) == 255