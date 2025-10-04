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


def remap_label_ids(mask: tf.Tensor, ignore_index: int = 255) -> tf.Tensor:
    """Remap raw Cityscapes IDs in ``mask`` to contiguous training IDs.

    Parameters
    ----------
    mask:
        Tensor containing raw Cityscapes IDs, typically decoded from the PNG
        ground-truth annotations.
    ignore_index:
        Label used to denote ignored/void pixels. Defaults to the Cityscapes
        convention of ``255``.

    Returns
    -------
    tf.Tensor
        Tensor of type ``tf.int32`` with the same shape as ``mask`` where all
        valid IDs are mapped to the ``[0, 7]`` range and void pixels keep the
        ``ignore_index`` value.
    """

    mask = tf.cast(mask, tf.int32)
    lut = build_cityscapes_8cls_lut(ignore_index)
    return tf.cast(remap_labels(mask, lut), tf.int32)
