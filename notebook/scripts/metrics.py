import tensorflow as tf

__all__ = [
    "MaskedMeanIoU",
    "masked_mean_iou",
    "masked_pixel_accuracy",
    "dice_coefficient",
    "dice_loss",
    "dice_coef_wrapper",
]

class MaskedMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes: int, ignore_index: int, name="masked_mIoU", **kwargs):
        self._ignore_index = int(ignore_index)
        super().__init__(num_classes=num_classes, name=name, **kwargs)
        self.ignore_index = tf.cast(self._ignore_index, tf.int32)

    def get_config(self):
        config = super().get_config()
        config.update({"ignore_index": self._ignore_index})
        return config

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)

        mask = tf.not_equal(y_true, self.ignore_index)

        # on compresse les tenseurs sur les pixels valides
        y_true_valid = tf.boolean_mask(y_true, mask)
        y_pred_valid = tf.boolean_mask(y_pred, mask)

        # évite l’avertissement de Keras (pas de sample_weight float/int)
        return super().update_state(y_true_valid, y_pred_valid, sample_weight=None)


def masked_mean_iou(num_classes, ignore_index, name="masked_mIoU", dtype=None):
    """Return a callable that builds :class:`MaskedMeanIoU` with preset arguments."""

    def factory():
        return MaskedMeanIoU(
            num_classes=num_classes,
            ignore_index=ignore_index,
            name=name,
            dtype=dtype,
        )

    factory.__name__ = name
    return factory


def masked_pixel_accuracy(y_true, y_pred, ignore_index):
    """Compute pixel accuracy while ignoring a specific class index."""

    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    ignore_index = tf.cast(ignore_index, tf.int32)
    mask = tf.not_equal(y_true, ignore_index)

    y_true_valid = tf.boolean_mask(y_true, mask)
    y_pred_valid = tf.boolean_mask(y_pred, mask)

    valid_count = tf.size(y_true_valid)

    def compute_accuracy():
        matches = tf.equal(y_true_valid, y_pred_valid)
        return tf.reduce_mean(tf.cast(matches, tf.float32))

    return tf.cond(valid_count > 0, compute_accuracy, lambda: tf.constant(0.0, dtype=tf.float32))


def dice_coefficient(y_true, y_pred, num_classes, ignore_index):
    """Compute the mean Dice coefficient across classes.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Logits or probabilities over classes.
        num_classes: Number of semantic classes (excluding the ignore index).
        ignore_index: Label value to exclude from the computation.

    Returns:
        tf.Tensor containing the average Dice score across valid classes.
    """

    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    ignore_index = tf.cast(ignore_index, tf.int32)
    mask = tf.not_equal(y_true, ignore_index)

    # Filter out ignored pixels
    y_true_valid = tf.boolean_mask(y_true, mask)
    y_pred_valid = tf.boolean_mask(y_pred, mask)

    # Handle the edge-case where all pixels are ignored
    valid_pixel_count = tf.size(y_true_valid)

    def compute_dice():
        y_true_one_hot = tf.one_hot(y_true_valid, depth=num_classes, dtype=tf.float32)
        y_pred_one_hot = tf.one_hot(y_pred_valid, depth=num_classes, dtype=tf.float32)

        intersection = tf.reduce_sum(y_true_one_hot * y_pred_one_hot, axis=0)
        total = tf.reduce_sum(y_true_one_hot, axis=0) + tf.reduce_sum(y_pred_one_hot, axis=0)

        dice_per_class = tf.where(total > 0, (2.0 * intersection) / total, tf.zeros_like(total))
        valid_classes = tf.reduce_sum(tf.cast(total > 0, tf.float32))

        return tf.where(
            valid_classes > 0,
            tf.reduce_sum(dice_per_class) / valid_classes,
            tf.constant(0.0, dtype=tf.float32),
        )

    return tf.cond(valid_pixel_count > 0, compute_dice, lambda: tf.constant(0.0, dtype=tf.float32))


def dice_loss(y_true, y_pred, num_classes, ignore_index):
    """Dice loss derived from the Dice coefficient."""

    dice = dice_coefficient(y_true, y_pred, num_classes=num_classes, ignore_index=ignore_index)
    return 1.0 - dice


def dice_coef_wrapper(num_classes, ignore_index):
    """Return a callable that computes the Dice coefficient with preset arguments."""

    def metric(y_true, y_pred):
        return dice_coefficient(y_true, y_pred, num_classes=num_classes, ignore_index=ignore_index)

    metric.__name__ = "dice_coefficient"
    return metric
