# /scripts/remap.py
import tensorflow as tf

def build_cityscapes_8cls_lut(ignore_index: int = 255) -> tf.Tensor:
    table = [ignore_index] * 34
    for k in [6,7,9,10]: table[k] = 0       # road
    table[8] = 1                            # sidewalk
    for k in [11,12,13,14,15,16]: table[k] = 2
    for k in [17,18,19,20]: table[k] = 3
    for k in [21,22]: table[k] = 4
    table[23] = 5
    for k in [24,25]: table[k] = 6
    for k in [26,27,28,29,30,31,32,33]: table[k] = 7
    return tf.constant(table, dtype=tf.int32)

@tf.function
def remap_labels(y: tf.Tensor, lut: tf.Tensor) -> tf.Tensor:
    y = tf.clip_by_value(y, 0, tf.shape(lut)[0]-1)
    return tf.gather(lut, y)
