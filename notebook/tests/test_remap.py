import pytest

tf = pytest.importorskip("tensorflow")

from notebook.scripts.remap import build_cityscapes_8cls_lut, remap_labels


def test_cityscapes_lut_has_full_uint8_coverage():
    lut = build_cityscapes_8cls_lut()
    assert lut.shape[0] == 256


def test_remap_labels_preserves_ignore_for_void_pixels():
    ignore_index = 255
    lut = build_cityscapes_8cls_lut(ignore_index=ignore_index)
    sample = tf.constant([[6, 8, 24, 255, 1, 200]], dtype=tf.int32)

    remapped = remap_labels(sample, lut)

    expected = tf.constant([[0, 1, 6, ignore_index, ignore_index, ignore_index]], dtype=tf.int32)
    tf.debugging.assert_equal(remapped, expected)
