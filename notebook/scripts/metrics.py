import tensorflow as tf

__all__ = ["MaskedMeanIoU", "masked_pixel_accuracy"]

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
