# /scripts/remap.py
import tensorflow as tf

def build_cityscapes_8cls_lut(ignore_index: int = 255) -> tf.Tensor:
    """Return a 256-entry lookup table mapping Cityscapes IDs to 8 classes.

    The Cityscapes label IDs are encoded on ``uint8`` PNG masks, which means
    that any remapping performed at training time must cope with indices in the
    full ``[0, 255]`` range.  We therefore initialise a 256-entry table filled
    with ``ignore_index`` and explicitly set the entries that should map to one
    of the eight training classes.  Every other ID (including the official void
    value ``255``) will naturally fall back to ``ignore_index`` when gathered.
    """

    table = [ignore_index] * 256

    for k in [6, 7, 9, 10]:
        table[k] = 0                      # road
    table[8] = 1                          # sidewalk
    for k in [11, 12, 13, 14, 15, 16]:
        table[k] = 2                      # building
    for k in [17, 18, 19, 20]:
        table[k] = 3                      # vegetation
    for k in [21, 22]:
        table[k] = 4                      # terrain
    table[23] = 5                         # sky
    for k in [24, 25]:
        table[k] = 6                      # human
    for k in [26, 27, 28, 29, 30, 31, 32, 33]:
        table[k] = 7                      # vehicle

    table[ignore_index] = ignore_index    # make the intent explicit for void

    return tf.constant(table, dtype=tf.int32)

@tf.function
def remap_labels(y: tf.Tensor, lut: tf.Tensor) -> tf.Tensor:
    return tf.gather(lut, y)
