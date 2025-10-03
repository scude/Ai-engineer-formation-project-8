import tensorflow as tf

class MaskedMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes: int, ignore_index: int, name="masked_mIoU"):
        super().__init__(num_classes=num_classes, name=name)
        self.ignore_index = tf.cast(ignore_index, tf.int32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        mask = tf.not_equal(y_true, self.ignore_index)  # bool

        if sample_weight is not None:
            sample_weight = tf.convert_to_tensor(sample_weight)
            weight_mask = tf.greater(sample_weight, 0)
            mask = tf.logical_and(mask, weight_mask)

        # on compresse les tenseurs sur les pixels valides
        y_true_valid = tf.boolean_mask(y_true, mask)
        y_pred_valid = tf.boolean_mask(y_pred, mask)

        # évite l’avertissement de Keras (pas de sample_weight float/int)
        return super().update_state(y_true_valid, y_pred_valid, sample_weight=None)
