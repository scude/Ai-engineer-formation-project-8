# /scripts/models.py
from typing import Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def ASPP(x, out_channels=256, rates=(6,12,18), name="aspp"):
    h, w = keras.backend.int_shape(x)[1:3]
    img_pool = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_gp")(x)
    img_pool = layers.Conv2D(out_channels, 1, use_bias=False)(img_pool)
    img_pool = layers.BatchNormalization()(img_pool); img_pool = layers.ReLU()(img_pool)
    img_pool = layers.Resizing(h, w, interpolation="bilinear")(img_pool)

    conv1 = layers.Conv2D(out_channels, 1, padding="same", use_bias=False)(x)
    conv1 = layers.BatchNormalization()(conv1); conv1 = layers.ReLU()(conv1)

    atr = []
    for r in rates:
        a = layers.Conv2D(out_channels, 3, padding="same", dilation_rate=r, use_bias=False)(x)
        a = layers.BatchNormalization()(a); a = layers.ReLU()(a); atr.append(a)

    x = layers.Concatenate()([conv1, *atr, img_pool])
    x = layers.Conv2D(out_channels, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Dropout(0.1)(x)
    return x

def deeplab_resnet50(input_shape, num_classes, output_stride=16, imagenet=True):
    assert output_stride in (8,16)
    base = keras.applications.ResNet50(include_top=False, weights="imagenet" if imagenet else None,
                                       input_shape=input_shape)
    low  = base.get_layer("conv2_block3_out").output     # H/4
    high = base.get_layer("conv4_block6_out").output     # H/16

    x = ASPP(high, 256, (6,12,18))
    low_h, low_w = keras.backend.int_shape(low)[1:3]
    x = layers.Resizing(low_h, low_w, interpolation="bilinear")(x)

    lowp = layers.Conv2D(48, 1, use_bias=False)(low)
    lowp = layers.BatchNormalization()(lowp); lowp = layers.ReLU()(lowp)

    x = layers.Concatenate()([x, lowp])
    for i in range(2):
        x = layers.SeparableConv2D(256, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)

    logits = layers.Conv2D(num_classes, 1, name="logits")(x)
    logits = layers.Resizing(input_shape[0], input_shape[1], interpolation="bilinear")(logits)
    return keras.Model(base.input, logits, name="deeplabv3plus_resnet50")

def deeplab_mobilenetv2(input_shape, num_classes, imagenet=True):
    base = keras.applications.MobileNetV2(include_top=False, weights="imagenet" if imagenet else None,
                                          input_shape=input_shape, alpha=1.0)
    low  = base.get_layer("block_3_expand_relu").output   # H/4
    high = base.get_layer("block_13_expand_relu").output  # H/16

    x = ASPP(high, 128, (6,12,18))
    low_h, low_w = keras.backend.int_shape(low)[1:3]
    x = layers.Resizing(low_h, low_w, interpolation="bilinear")(x)

    lowp = layers.Conv2D(32, 1, use_bias=False)(low)
    lowp = layers.BatchNormalization()(lowp); lowp = layers.ReLU()(lowp)

    x = layers.Concatenate()([x, lowp])
    for _ in range(2):
        x = layers.SeparableConv2D(128, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)

    logits = layers.Conv2D(num_classes, 1, name="logits")(x)
    logits = layers.Resizing(input_shape[0], input_shape[1], interpolation="bilinear")(logits)
    return keras.Model(base.input, logits, name="deeplabv3plus_mobilenetv2")

def unet_small(input_shape, num_classes, base_filters=32):
    inputs = keras.Input(shape=input_shape)
    # enc
    c1 = layers.Conv2D(base_filters, 3, padding="same", activation="relu")(inputs)
    c1 = layers.Conv2D(base_filters, 3, padding="same", activation="relu")(c1)
    p1 = layers.MaxPool2D()(c1)
    c2 = layers.Conv2D(base_filters*2, 3, padding="same", activation="relu")(p1)
    c2 = layers.Conv2D(base_filters*2, 3, padding="same", activation="relu")(c2)
    p2 = layers.MaxPool2D()(c2)
    c3 = layers.Conv2D(base_filters*4, 3, padding="same", activation="relu")(p2)
    c3 = layers.Conv2D(base_filters*4, 3, padding="same", activation="relu")(c3)
    p3 = layers.MaxPool2D()(c3)
    # bottleneck
    b = layers.Conv2D(base_filters*8, 3, padding="same", activation="relu")(p3)
    b = layers.Conv2D(base_filters*8, 3, padding="same", activation="relu")(b)
    # dec
    u3 = layers.UpSampling2D()(b); u3 = layers.Concatenate()([u3, c3])
    c4 = layers.Conv2D(base_filters*4, 3, padding="same", activation="relu")(u3)
    c4 = layers.Conv2D(base_filters*4, 3, padding="same", activation="relu")(c4)
    u2 = layers.UpSampling2D()(c4); u2 = layers.Concatenate()([u2, c2])
    c5 = layers.Conv2D(base_filters*2, 3, padding="same", activation="relu")(u2)
    c5 = layers.Conv2D(base_filters*2, 3, padding="same", activation="relu")(c5)
    u1 = layers.UpSampling2D()(c5); u1 = layers.Concatenate()([u1, c1])
    c6 = layers.Conv2D(base_filters, 3, padding="same", activation="relu")(u1)
    c6 = layers.Conv2D(base_filters, 3, padding="same", activation="relu")(c6)
    logits = layers.Conv2D(num_classes, 1, name="logits")(c6)
    return keras.Model(inputs, logits, name="unet_small")

def _dw_sep(x, out, s=1):
    x = layers.DepthwiseConv2D(3, strides=s, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv2D(out, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    return x

def fast_scnn(input_shape, num_classes):
    inputs = keras.Input(input_shape)
    # learning to downsample
    x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = _dw_sep(x, 48, s=2)  # H/4
    x = _dw_sep(x, 64, s=2)  # H/8

    # global feature extractor
    for _ in range(3): x = _dw_sep(x, 64, s=1)
    x = ASPP(x, out_channels=128, rates=(2,4,6))

    # feature fusion
    low = layers.Conv2D(64, 1, use_bias=False)(inputs); low = layers.BatchNormalization()(low); low = layers.ReLU()(low)
    low = layers.AveragePooling2D(pool_size=4, strides=4, padding="same")(low)  # ~H/4
    x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)                # ~H/4
    x = layers.Concatenate()([x, low])
    x = _dw_sep(x, 128)

    # classifier
    x = layers.UpSampling2D(size=4, interpolation="bilinear")(x)
    logits = layers.Conv2D(num_classes, 1, name="logits")(x)
    return keras.Model(inputs, logits, name="fast_scnn")

def build_model(name: str, input_shape, num_classes: int):
    name = name.lower()
    if name == "deeplab_resnet50":
        return deeplab_resnet50(input_shape, num_classes)
    if name == "deeplab_mobilenetv2":
        return deeplab_mobilenetv2(input_shape, num_classes)
    if name == "unet_small":
        return unet_small(input_shape, num_classes)
    if name == "fast_scnn":
        return fast_scnn(input_shape, num_classes)
    raise ValueError(f"Unknown model: {name}")
