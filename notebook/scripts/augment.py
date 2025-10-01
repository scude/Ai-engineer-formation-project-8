# /scripts/augment.py
import tensorflow as tf
from typing import Tuple
from .config import AugmentConfig

# ---------- helpers sûrs 3D <-> 4D ----------
def _resize_3d(img3: tf.Tensor, size, method="bilinear") -> tf.Tensor:
    img4 = tf.expand_dims(img3, 0)                    # [1,H,W,C]
    out4 = tf.image.resize(img4, size, method=method) # [1,h,w,C]
    return tf.squeeze(out4, 0)                        # [h,w,C]

def _crop_to_box_3d(img3: tf.Tensor, offset_h, offset_w, target_h, target_w) -> tf.Tensor:
    img4 = tf.expand_dims(img3, 0)  # [1,H,W,C]
    out4 = tf.image.crop_to_bounding_box(img4, offset_h, offset_w, target_h, target_w)
    return tf.squeeze(out4, 0)      # [h,w,C]

# ---------- rotation (sans TFA) ----------
def _rotate_3d(img3: tf.Tensor, angle_rad: tf.Tensor, method: str) -> tf.Tensor:
    """
    Rotation autour du centre via ImageProjectiveTransformV3 (op natif TF).
    img3: [H,W,C] float32
    angle_rad: scalaire radians
    method: "bilinear" ou "nearest"
    """
    h = tf.cast(tf.shape(img3)[0], tf.float32)
    w = tf.cast(tf.shape(img3)[1], tf.float32)
    cy = (h - 1.0) / 2.0
    cx = (w - 1.0) / 2.0

    c = tf.cos(angle_rad)
    s = tf.sin(angle_rad)

    # matrice qui mappe (x_out, y_out) -> (x_in, y_in)
    # x' = c*x - s*y + tx
    # y' = s*x + c*y + ty
    tx = (1.0 - c) * cx + s * cy
    ty = (1.0 - c) * cy - s * cx

    # TF attend [a0, a1, a2, a3, a4, a5, a6, a7] (a6=a7=0 pour affine)
    transform = tf.stack([c, -s, tx, s, c, ty, 0.0, 0.0], axis=0)
    transform = tf.expand_dims(transform, 0)  # [1,8]

    img4 = tf.expand_dims(img3, 0)  # [1,H,W,C]
    out4 = tf.raw_ops.ImageProjectiveTransformV3(
        images=img4,
        transforms=transform,
        output_shape=tf.stack([tf.cast(h, tf.int32), tf.cast(w, tf.int32)]),
        interpolation=method.upper(),  # "BILINEAR" / "NEAREST"
        fill_value=0.0
    )
    return tf.squeeze(out4, 0)  # [H,W,C]

# ---------- flips ----------
def _maybe_hflip(x, y):
    do = tf.less(tf.random.uniform([]), 0.5)
    return tf.cond(do,
                   lambda: (tf.image.flip_left_right(x), tf.image.flip_left_right(y)),
                   lambda: (x, y))

def _maybe_vflip(x, y):
    do = tf.less(tf.random.uniform([]), 0.5)
    return tf.cond(do,
                   lambda: (tf.image.flip_up_down(x), tf.image.flip_up_down(y)),
                   lambda: (x, y))

def _random_rotate(x, y, deg: float):
    if deg <= 0: return x, y
    angle = tf.random.uniform([], -deg, deg) * 3.14159265 / 180.0
    x = _rotate_3d(x, angle, method="bilinear")
    y = _rotate_3d(y, angle, method="nearest")
    return x, y

# ---------- scale + crop ----------
def _random_rescale_and_crop(x, y, h, w, smin, smax):
    if smin == 1.0 and smax == 1.0:
        return _resize_3d(x, (h, w), "bilinear"), _resize_3d(y, (h, w), "nearest")
    s = tf.random.uniform([], smin, smax)
    nh = tf.cast(tf.round(tf.cast(h, tf.float32) * s), tf.int32)
    nw = tf.cast(tf.round(tf.cast(w, tf.float32) * s), tf.int32)

    x = _resize_3d(x, (nh, nw), "bilinear")
    y = _resize_3d(y, (nh, nw), "nearest")

    off_h = tf.cond(nh > h, lambda: tf.random.uniform([], 0, nh - h + 1, dtype=tf.int32), lambda: 0)
    off_w = tf.cond(nw > w, lambda: tf.random.uniform([], 0, nw - w + 1, dtype=tf.int32), lambda: 0)
    crop_h = tf.minimum(h, nh); crop_w = tf.minimum(w, nw)

    x = _crop_to_box_3d(x, off_h, off_w, crop_h, crop_w)
    y = _crop_to_box_3d(y, off_h, off_w, crop_h, crop_w)

    x = _resize_3d(x, (h, w), "bilinear")
    y = _resize_3d(y, (h, w), "nearest")
    return x, y

# ---------- photométrie ----------
def _color_jitter(x, cfg: AugmentConfig):
    if cfg.brightness_delta > 0: x = tf.image.random_brightness(x, cfg.brightness_delta)
    if cfg.contrast_delta > 0:   x = tf.image.random_contrast(x, 1-cfg.contrast_delta, 1+cfg.contrast_delta)
    if cfg.saturation_delta > 0: x = tf.image.random_saturation(x, 1-cfg.saturation_delta, 1+cfg.saturation_delta)
    if cfg.hue_delta > 0:        x = tf.image.random_hue(x, cfg.hue_delta)
    return tf.clip_by_value(x, 0.0, 1.0)

def _gaussian_noise(x, std: float):
    if std <= 0: return x
    noise = tf.random.normal(tf.shape(x), 0.0, std, dtype=x.dtype)
    return tf.clip_by_value(x + noise, 0.0, 1.0)

# ---------- API ----------
def build_augment_fn(cfg: AugmentConfig, h: int, w: int):
    def _fn(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = tf.convert_to_tensor(x, dtype=tf.float32)  # [H,W,3]
        y = tf.cast(y, tf.int32)                       # [H,W]
        y = tf.expand_dims(y, -1)                      # [H,W,1]
        y = tf.cast(y, tf.float32)                     # géométrie en float32

        if not cfg.enabled:
            x = _resize_3d(x, (h, w), "bilinear")
            y = _resize_3d(y, (h, w), "nearest")
            y = tf.squeeze(tf.cast(tf.round(y), tf.int32), -1)   # [h,w]
            return x, y

        if cfg.random_crop or cfg.random_scale_min != 1.0 or cfg.random_scale_max != 1.0:
            x, y = _random_rescale_and_crop(x, y, h, w, cfg.random_scale_min, cfg.random_scale_max)
        else:
            x = _resize_3d(x, (h, w), "bilinear")
            y = _resize_3d(y, (h, w), "nearest")

        if cfg.hflip: x, y = _maybe_hflip(x, y)
        if cfg.vflip: x, y = _maybe_vflip(x, y)

        x, y = _random_rotate(x, y, cfg.random_rotate_deg)

        x = _color_jitter(x, cfg)
        x = _gaussian_noise(x, cfg.gaussian_noise_std)

        y = tf.squeeze(tf.cast(tf.round(y), tf.int32), -1)       # [h,w]
        return x, y
    return _fn
