import os, glob
from typing import List, Tuple
import tensorflow as tf
from .config import DataConfig, AugmentConfig
from .remap import build_cityscapes_8cls_lut, remap_labels
from .augment import build_augment_fn

def left_to_label_path(lp, left_dir, gt_dir, img_suffix, lbl_suffix):
    parts = lp.split(os.sep)
    parts[parts.index(left_dir)] = gt_dir
    parts[-1] = parts[-1].replace(img_suffix, lbl_suffix)
    return os.path.join(*parts)

def gather_pairs(cfg: DataConfig, split: str) -> Tuple[List[str], List[str], List[str]]:
    img_glob = os.path.join(cfg.data_root, cfg.left_dir, split, "*", f"*{cfg.img_suffix}")
    left_files = sorted(glob.glob(img_glob))
    left_ok, lbl_ok, miss = [], [], []
    for lp in left_files:
        lb = left_to_label_path(lp, cfg.left_dir, cfg.gt_dir, cfg.img_suffix, cfg.lbl_suffix)
        if os.path.exists(lb): left_ok.append(lp); lbl_ok.append(lb)
        else: miss.append(lb)
    return left_ok, lbl_ok, miss

def decode_img(path: tf.Tensor):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    # donner un indice de shape dynamique à TF (H,W,3)
    img = tf.ensure_shape(img, [None, None, 3])
    return img

def decode_lbl(path: tf.Tensor):
    raw = tf.io.read_file(path)
    y = tf.image.decode_png(raw, channels=1)
    y = tf.squeeze(tf.cast(y, tf.int32), -1)
    # (H,W)
    y = tf.ensure_shape(y, [None, None])
    return y

def make_weights(y, ignore_index):
    return tf.where(tf.equal(y, ignore_index), 0.0, 1.0)

def prepare_labels_for_loss(y, ignore_index):
    return tf.where(tf.equal(y, ignore_index), tf.zeros_like(y), y)


# Backwards compatibility for older notebooks/imports relying on the previous name.
def sanitize_labels_for_loss(y, ignore_index):
    return prepare_labels_for_loss(y, ignore_index)

def build_dataset(cfg: DataConfig, aug: AugmentConfig, split: str, training: bool):
    xs, ys, miss = gather_pairs(cfg, split)
    print(f"[info] {split}: {len(xs)} pairs | missing: {len(miss)}")
    if xs: print("ex:", xs[0], "->", ys[0])

    lut = build_cityscapes_8cls_lut(cfg.ignore_index)
    aug_fn = build_augment_fn(aug, cfg.height, cfg.width, cfg.ignore_index)
    AUTOTUNE = tf.data.AUTOTUNE if cfg.autotune is None else cfg.autotune

    def _parse(xp, yp):
        x = decode_img(xp)
        y = decode_lbl(yp)
        y = remap_labels(y, lut)                   # {0..7,255}
        x, y = aug_fn(x, y)                        # resize + aug (shapes imposées)
        w = make_weights(y, cfg.ignore_index)      # 0 on ignore
        y = tf.cast(y, tf.int32)
        # shapes finales strictes
        x = tf.ensure_shape(x, [cfg.height, cfg.width, 3])
        y = tf.ensure_shape(y, [cfg.height, cfg.width])
        w = tf.ensure_shape(w, [cfg.height, cfg.width])
        return x, y, w

    ds = tf.data.Dataset.from_tensor_slices((xs, ys))
    if training:
        if cfg.max_train_samples is not None:
            ds = ds.take(cfg.max_train_samples)
        ds = ds.shuffle(len(xs), seed=cfg.seed, reshuffle_each_iteration=True)
    else:
        if cfg.max_val_samples is not None:
            ds = ds.take(cfg.max_val_samples)
    ds = ds.map(_parse, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(cfg.batch_size).prefetch(AUTOTUNE)
    return ds
