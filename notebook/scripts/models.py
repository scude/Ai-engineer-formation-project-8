"""Segmentation model builders used by the training scripts.

The module keeps everything in a single place to stay easy to reason about
while still providing reusable decoder utilities so the custom heads share the
same implementation blocks."""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from tensorflow import keras
from tensorflow.keras import layers

TensorShape = Tuple[int, int, int]


def conv_bn_act(
    x: keras.layers.Layer,
    filters: int,
    kernel_size: int = 3,
    strides: int = 1,
    activation: str | None = "relu",
    name: str | None = None,
) -> keras.layers.Layer:
    """Conv2D followed by batch-normalisation and optional activation."""
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        name=None if name is None else f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(name=None if name is None else f"{name}_bn")(x)
    if activation:
        x = layers.Activation(activation, name=None if name is None else f"{name}_act")(x)
    return x


def separable_conv_block(
    x: keras.layers.Layer,
    filters: int,
    kernel_size: int = 3,
    activation: str = "relu",
    name: str | None = None,
) -> keras.layers.Layer:
    x = layers.SeparableConv2D(
        filters,
        kernel_size,
        padding="same",
        use_bias=False,
        name=None if name is None else f"{name}_sepconv",
    )(x)
    x = layers.BatchNormalization(name=None if name is None else f"{name}_bn")(x)
    return layers.Activation(activation, name=None if name is None else f"{name}_act")(x)


def double_conv_block(x: keras.layers.Layer, filters: int, name: str | None = None) -> keras.layers.Layer:
    """Two conv-bn-relu blocks used by the U-Net style decoders."""
    x = conv_bn_act(x, filters, name=None if name is None else f"{name}_1")
    return conv_bn_act(x, filters, name=None if name is None else f"{name}_2")


def residual_block(x: keras.layers.Layer, filters: int, name: str | None = None) -> keras.layers.Layer:
    shortcut = x
    y = conv_bn_act(x, filters, name=None if name is None else f"{name}_conv1")
    y = conv_bn_act(y, filters, activation=None, name=None if name is None else f"{name}_conv2")
    if keras.backend.int_shape(shortcut)[-1] != filters:
        shortcut = conv_bn_act(shortcut, filters, activation=None, name=None if name is None else f"{name}_proj")
    x = layers.Add(name=None if name is None else f"{name}_add")([shortcut, y])
    return layers.Activation("relu", name=None if name is None else f"{name}_out")(x)


def decoder_block(
    x: keras.layers.Layer,
    skip: keras.layers.Layer | None,
    filters: int,
    separable: bool = False,
    name: str | None = None,
) -> keras.layers.Layer:
    """Upsample + concat skip connection followed by a conv block."""
    x = layers.UpSampling2D(size=2, interpolation="bilinear", name=None if name is None else f"{name}_up")(x)
    if skip is not None:
        x = layers.Concatenate(name=None if name is None else f"{name}_concat")([x, skip])
    block = separable_conv_block if separable else double_conv_block
    return block(x, filters, name=name)


def aspp_block(
    x: keras.layers.Layer,
    out_channels: int = 256,
    rates: Tuple[int, int, int] = (6, 12, 18),
    activation: str = "relu",
    dropout: float = 0.1,
    name: str = "aspp",
) -> keras.layers.Layer:
    """Atrous Spatial Pyramid Pooling head used by DeepLab-like models."""
    h, w = keras.backend.int_shape(x)[1:3]
    img_pool = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name}_gp")(x)
    img_pool = layers.Conv2D(out_channels, 1, use_bias=False, name=f"{name}_gp_conv")(img_pool)
    img_pool = layers.BatchNormalization(name=f"{name}_gp_bn")(img_pool)
    img_pool = layers.Activation(activation, name=f"{name}_gp_act")(img_pool)
    img_pool = layers.Resizing(h, w, interpolation="bilinear", name=f"{name}_gp_resize")(img_pool)

    branches = [
        conv_bn_act(x, out_channels, kernel_size=1, name=f"{name}_conv1"),
    ]
    for i, rate in enumerate(rates, start=1):
        b = layers.Conv2D(
            out_channels,
            3,
            padding="same",
            dilation_rate=rate,
            use_bias=False,
            name=f"{name}_atrous{i}",
        )(x)
        b = layers.BatchNormalization(name=f"{name}_atrous{i}_bn")(b)
        b = layers.Activation(activation, name=f"{name}_atrous{i}_act")(b)
        branches.append(b)

    x = layers.Concatenate(name=f"{name}_concat")([*branches, img_pool])
    x = layers.Conv2D(out_channels, 1, use_bias=False, name=f"{name}_proj_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_proj_bn")(x)
    x = layers.Activation(activation, name=f"{name}_proj_act")(x)
    return layers.Dropout(dropout, name=f"{name}_dropout")(x)


def deeplab_resnet50(
    input_shape: TensorShape,
    num_classes: int,
    output_stride: int = 16,
    imagenet: bool = True,
    aspp_dilations: Tuple[int, int, int] = (6, 12, 18),
    decoder_filters: int = 256,
    aspp_dropout: float = 0.5,
    decoder_activation: str = "relu",
    aspp_activation: str = "relu",
    **_: Any,
) -> keras.Model:
    assert output_stride in (8, 16)
    base = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet" if imagenet else None,
        input_shape=input_shape,
    )
    low = base.get_layer("conv2_block3_out").output
    if len(aspp_dilations) != 3:
        raise ValueError("'aspp_dilations' must contain exactly three dilation rates")
    if output_stride == 16:
        high = base.get_layer("conv4_block6_out").output
    else:
        high = base.get_layer("conv3_block4_out").output

    x = aspp_block(
        high,
        256,
        aspp_dilations,
        activation=aspp_activation,
        dropout=aspp_dropout,
    )
    low_h, low_w = keras.backend.int_shape(low)[1:3]
    x = layers.Resizing(low_h, low_w, interpolation="bilinear")(x)

    lowp = layers.Conv2D(48, 1, use_bias=False)(low)
    lowp = layers.BatchNormalization()(lowp)
    lowp = layers.Activation(decoder_activation)(lowp)

    x = layers.Concatenate()([x, lowp])
    for _ in range(2):
        x = layers.SeparableConv2D(decoder_filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(decoder_activation)(x)

    logits = layers.Conv2D(num_classes, 1, name="logits")(x)
    logits = layers.Resizing(input_shape[0], input_shape[1], interpolation="bilinear")(logits)
    return keras.Model(base.input, logits, name="deeplabv3plus_resnet50")


def deeplab_mobilenetv2(
    input_shape: TensorShape,
    num_classes: int,
    imagenet: bool = True,
    **_: Any,
) -> keras.Model:
    base = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet" if imagenet else None,
        input_shape=input_shape,
        alpha=1.0,
    )
    low = base.get_layer("block_3_expand_relu").output
    high = base.get_layer("block_13_expand_relu").output

    x = aspp_block(high, 128, (6, 12, 18))
    low_h, low_w = keras.backend.int_shape(low)[1:3]
    x = layers.Resizing(low_h, low_w, interpolation="bilinear")(x)

    lowp = layers.Conv2D(32, 1, use_bias=False)(low)
    lowp = layers.BatchNormalization()(lowp)
    lowp = layers.ReLU()(lowp)

    x = layers.Concatenate()([x, lowp])
    for _ in range(2):
        x = layers.SeparableConv2D(128, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    logits = layers.Conv2D(num_classes, 1, name="logits")(x)
    logits = layers.Resizing(input_shape[0], input_shape[1], interpolation="bilinear")(logits)
    return keras.Model(base.input, logits, name="deeplabv3plus_mobilenetv2")


def unet_small(
    input_shape: TensorShape,
    num_classes: int,
    base_filters: int = 32,
    **_: Any,
) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    c1 = double_conv_block(inputs, base_filters, name="enc1")
    p1 = layers.MaxPool2D(name="pool1")(c1)

    c2 = double_conv_block(p1, base_filters * 2, name="enc2")
    p2 = layers.MaxPool2D(name="pool2")(c2)

    c3 = double_conv_block(p2, base_filters * 4, name="enc3")
    p3 = layers.MaxPool2D(name="pool3")(c3)

    b = double_conv_block(p3, base_filters * 8, name="bottleneck")

    u3 = decoder_block(b, c3, base_filters * 4, name="dec3")
    u2 = decoder_block(u3, c2, base_filters * 2, name="dec2")
    u1 = decoder_block(u2, c1, base_filters, name="dec1")

    logits = layers.Conv2D(num_classes, 1, name="logits")(u1)
    return keras.Model(inputs, logits, name="unet_small")


def unet_mini(
    num_classes: int,
    input_shape: TensorShape,
    base_filters: int = 24,
    **_: Any,
) -> keras.Model:
    inputs = keras.Input(shape=input_shape)
    c1 = double_conv_block(inputs, base_filters, name="mini_enc1")
    p1 = layers.MaxPool2D(name="mini_pool1")(c1)

    c2 = double_conv_block(p1, base_filters * 2, name="mini_enc2")
    p2 = layers.MaxPool2D(name="mini_pool2")(c2)

    b = double_conv_block(p2, base_filters * 4, name="mini_bottleneck")

    u2 = decoder_block(b, c2, base_filters * 2, name="mini_dec2")
    u1 = decoder_block(u2, c1, base_filters, name="mini_dec1")

    logits = layers.Conv2D(num_classes, 1, name="logits")(u1)
    return keras.Model(inputs, logits, name="unet_mini")


def unet_vgg16(
    num_classes: int,
    input_shape: TensorShape,
    imagenet: bool = True,
    **_: Any,
) -> keras.Model:
    base = keras.applications.VGG16(
        include_top=False,
        weights="imagenet" if imagenet else None,
        input_shape=input_shape,
    )
    skips = [
        base.get_layer("block1_conv2").output,
        base.get_layer("block2_conv2").output,
        base.get_layer("block3_conv3").output,
        base.get_layer("block4_conv3").output,
    ]
    bottleneck = base.get_layer("block5_conv3").output

    # The original decoder mirrored VGG16's channel counts (512->64) which
    # yields very large feature maps when training on 1024x512 crops.  Those
    # activations easily blow up the GPU memory usage when the batch size is
    # greater than one.  To keep the receptive field while reducing the memory
    # footprint we progressively shrink the number of filters in the decoder.
    # This keeps the model expressive enough for the task but lowers the peak
    # tensor size so the training no longer OOMs on common GPUs.
    x = decoder_block(bottleneck, skips[-1], 128, name="vgg_dec4")
    x = decoder_block(x, skips[-2], 64, name="vgg_dec3")
    x = decoder_block(x, skips[-3], 48, name="vgg_dec2")
    x = decoder_block(x, skips[-4], 36, name="vgg_dec1")
    x = double_conv_block(x, 36, name="vgg_final")

    logits = layers.Conv2D(num_classes, 1, name="logits")(x)
    return keras.Model(base.input, logits, name="unet_vgg16")


def mobiledet_seg(
    num_classes: int,
    input_shape: TensorShape,
    imagenet: bool = True,
    **_: Any,
) -> keras.Model:
    backbone = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet" if imagenet else None,
        input_shape=input_shape,
        alpha=1.0,
    )
    skip2 = backbone.get_layer("block_6_expand_relu").output
    skip1 = backbone.get_layer("block_3_expand_relu").output
    skip0 = backbone.get_layer("block_1_expand_relu").output
    bottleneck = backbone.get_layer("block_13_expand_relu").output

    x = separable_conv_block(bottleneck, 256, name="mobiledet_bottleneck")
    x = decoder_block(x, skip2, 192, separable=True, name="mobiledet_dec3")
    x = decoder_block(x, skip1, 128, separable=True, name="mobiledet_dec2")
    x = decoder_block(x, skip0, 96, separable=True, name="mobiledet_dec1")
    x = layers.UpSampling2D(size=2, interpolation="bilinear", name="mobiledet_up_final")(x)
    x = separable_conv_block(x, 64, name="mobiledet_head")

    logits = layers.Conv2D(num_classes, 1, name="logits")(x)
    return keras.Model(backbone.input, logits, name="mobiledet_seg")


def _yolo_stage(x: keras.layers.Layer, filters: int, blocks: int, name: str) -> keras.layers.Layer:
    x = conv_bn_act(x, filters, kernel_size=3, strides=2, name=f"{name}_down")
    for i in range(blocks):
        x = residual_block(x, filters, name=f"{name}_res{i + 1}")
    return x


def yolov9_seg(
    num_classes: int,
    input_shape: TensorShape,
    **_: Any,
) -> keras.Model:
    inputs = keras.Input(shape=input_shape)

    stem = conv_bn_act(inputs, 32, kernel_size=3, name="stem1")
    stem = conv_bn_act(stem, 32, kernel_size=3, name="stem2")

    stage1 = _yolo_stage(stem, 64, blocks=1, name="stage1")
    stage2 = _yolo_stage(stage1, 128, blocks=2, name="stage2")
    stage3 = _yolo_stage(stage2, 256, blocks=2, name="stage3")
    stage4 = _yolo_stage(stage3, 512, blocks=1, name="stage4")

    neck = conv_bn_act(stage4, 512, kernel_size=3, name="neck")

    x = decoder_block(neck, stage3, 128, name="yolo_dec3")
    x = decoder_block(x, stage2, 64, name="yolo_dec2")
    x = decoder_block(x, stage1, 48, name="yolo_dec1")
    x = decoder_block(x, stem, 32, name="yolo_dec0")

    logits = layers.Conv2D(num_classes, 1, name="logits")(x)
    return keras.Model(inputs, logits, name="yolov9_seg")


def _dw_sep(x: keras.layers.Layer, out_channels: int, strides: int = 1) -> keras.layers.Layer:
    x = layers.DepthwiseConv2D(3, strides=strides, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(out_channels, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def fast_scnn(
    input_shape: TensorShape,
    num_classes: int,
    **_: Any,
) -> keras.Model:
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, strides=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = _dw_sep(x, 48, strides=2)
    x = _dw_sep(x, 64, strides=2)

    for _ in range(3):
        x = _dw_sep(x, 64, strides=1)
    x = aspp_block(x, out_channels=128, rates=(2, 4, 6))

    low = layers.Conv2D(64, 1, use_bias=False)(inputs)
    low = layers.BatchNormalization()(low)
    low = layers.ReLU()(low)
    low = layers.AveragePooling2D(pool_size=4, strides=4, padding="same")(low)

    x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
    x = layers.Concatenate()([x, low])
    x = _dw_sep(x, 128)

    x = layers.UpSampling2D(size=4, interpolation="bilinear")(x)
    logits = layers.Conv2D(num_classes, 1, name="logits")(x)
    return keras.Model(inputs, logits, name="fast_scnn")


ASPP = aspp_block


AVAILABLE_MODELS = (
    "unet_small",
    "unet_mini",
    "unet_vgg16",
    "mobiledet_seg",
    "yolov9_seg",
    "deeplab_resnet50",
    "deeplab_mobilenetv2",
    "fast_scnn",
)


def _registry() -> Dict[str, Callable[[int, TensorShape], keras.Model]]:
    return {
        "unet_small": lambda num_classes, input_shape, **kwargs: unet_small(input_shape, num_classes, **kwargs),
        "unet_mini": unet_mini,
        "unet_vgg16": unet_vgg16,
        "mobiledet_seg": mobiledet_seg,
        "yolov9_seg": yolov9_seg,
        "deeplab_resnet50": lambda num_classes, input_shape, **kwargs: deeplab_resnet50(input_shape, num_classes, **kwargs),
        "deeplab_mobilenetv2": lambda num_classes, input_shape, **kwargs: deeplab_mobilenetv2(input_shape, num_classes, **kwargs),
        "fast_scnn": lambda num_classes, input_shape, **kwargs: fast_scnn(input_shape, num_classes, **kwargs),
    }


def build_model(name: str, *args: Any, **kwargs: Any) -> keras.Model:
    """Instantiate a segmentation model by its registered name."""
    if len(args) >= 2:
        first, second = args[0], args[1]
        if isinstance(first, int):
            num_classes, input_shape = first, second
            remaining = args[2:]
        else:
            input_shape, num_classes = first, second
            remaining = args[2:]
    else:
        num_classes = kwargs.pop("num_classes")
        input_shape = kwargs.pop("input_shape")
        remaining = ()

    registry = _registry()
    key = name.lower()
    if key not in registry:
        raise ValueError(f"Unknown model architecture: {name}")
    builder = registry[key]
    return builder(num_classes, input_shape, *remaining, **kwargs)


__all__ = [
    "AVAILABLE_MODELS",
    "ASPP",
    "aspp_block",
    "build_model",
    "deeplab_mobilenetv2",
    "deeplab_resnet50",
    "fast_scnn",
    "mobiledet_seg",
    "unet_mini",
    "unet_small",
    "unet_vgg16",
    "yolov9_seg",
]
