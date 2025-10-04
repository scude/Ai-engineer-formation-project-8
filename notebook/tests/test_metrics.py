import pytest

tf = pytest.importorskip("tensorflow")

from notebook.scripts.metrics import MaskedMeanIoU, masked_mean_iou, masked_pixel_accuracy


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
    assert int(metric.ignore_index.numpy()) == 7


def test_masked_pixel_accuracy_excludes_ignore_index_from_denominator():
    y_true = tf.constant(
        [[0, 1, 255], [1, 2, 255]],
        dtype=tf.int32,
    )
    y_pred = tf.constant(
        [
            [[10.0, 0.0, 0.0], [0.1, 0.2, 0.7], [0.0, 1.0, 0.0]],
            [[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.3, 0.2, 0.5]],
        ],
        dtype=tf.float32,
    )

    accuracy = masked_pixel_accuracy(y_true, y_pred, ignore_index=255)

    expected_correct = 2  # positions (0, 0) and (1, 0)
    expected_total = 4  # two pixels are ignored
    expected_accuracy = expected_correct / expected_total

    assert accuracy.numpy() == pytest.approx(expected_accuracy)


def test_masked_mean_iou_excludes_ignore_index_from_confusion_matrix():
    factory = masked_mean_iou(num_classes=3, ignore_index=255, dtype=tf.float32)
    metric = factory()

    y_true = tf.constant(
        [[0, 1, 255], [1, 2, 255]],
        dtype=tf.int32,
    )
    y_pred = tf.constant(
        [
            [[10.0, 0.0, 0.0], [0.1, 0.2, 0.7], [0.0, 1.0, 0.0]],
            [[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.3, 0.2, 0.5]],
        ],
        dtype=tf.float32,
    )

    metric.update_state(y_true, y_pred)

    expected_confusion = tf.constant(
        [[1, 0, 0], [0, 1, 1], [1, 0, 0]],
        dtype=metric.total_cm.dtype,
    )
    expected_mean_iou = (0.5 + 0.5 + 0.0) / 3

    assert tf.reduce_all(tf.equal(metric.total_cm, expected_confusion))
    assert metric.result().numpy() == pytest.approx(expected_mean_iou)