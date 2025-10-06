# /scripts/config.py
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

@dataclass
class DataConfig:
    data_root: str = "../data"
    left_dir: str = "leftImg8bit"
    gt_dir: str = "gtFine"
    img_suffix: str = "_leftImg8bit.png"
    lbl_suffix: str = "_gtFine_labelIds.png"
    height: int = 512
    width: int = 1024
    num_classes: int = 8
    ignore_index: int = 255
    seed: int = 1337
    batch_size: int = 2
    autotune: Optional[int] = None  # tf.data.AUTOTUNE set in code
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None

@dataclass
class TrainConfig:
    lr: float = 3e-4
    epochs: int = 60
    optimizer: str = "adam"
    early_stop_patience: int = 10
    output_dir: str = "artifacts"
    exp_name: str = "cityscapes-seg-8cls"
    arch: str = "unet_small"
    loss: str = "ce"
    deterministic_ops: bool = False
    precision_policy: str = "auto"  # "auto" chooses mixed_float16 on GPU, float32 otherwise

@dataclass
class AugmentConfig:
    """Configuration for the Albumentations-based augmentation pipeline.

    The geometric parameters are converted to Albumentations transforms:

    * ``random_rotate_deg`` becomes the symmetric degree range passed to
      :class:`albumentations.Rotate` (``[-deg, +deg]``).
    * ``random_scale_min`` and ``random_scale_max`` are multiplicative factors
      consumed by :class:`~notebook.scripts.augment.RandomScaleCrop` to mimic
      the former TensorFlow resize/scale/crop pipeline.

    Photometric parameters map to :class:`albumentations.ColorJitter`. The
    ``*_delta`` values follow the PyTorch-style semantics used by
    Albumentations: a value of ``d`` yields multiplicative factors drawn from
    ``[max(0, 1 - d), 1 + d]``. ``gaussian_noise_std`` represents the desired
    standard deviation in the ``[0, 1]`` range and is converted to a variance
    interval for :class:`albumentations.GaussNoise`.
    """

    # geometric
    hflip: bool = True
    vflip: bool = False
    random_rotate_deg: float = 0.0     # rotation amplitude in degrees (symmetric)
    random_scale_min: float = 1.0      # multiplicative lower bound for scaling
    random_scale_max: float = 1.0      # multiplicative upper bound for scaling
    random_crop: bool = False          # if True, enable random scale+crop to (H,W)
    # photometric (image-only)
    brightness_delta: float = 0.0      # ColorJitter brightness factor delta
    contrast_delta: float = 0.0        # ColorJitter contrast factor delta
    saturation_delta: float = 0.0      # ColorJitter saturation factor delta
    hue_delta: float = 0.0             # ColorJitter hue factor delta (0..0.5)
    gaussian_noise_std: float = 0.0    # standard deviation of additive Gaussian noise in [0,1]
    # enable/disable all aug
    enabled: bool = True

DEFAULT_PALETTE_8 = {
    0:(128, 64,128), 1:(244, 35,232), 2:( 70, 70, 70), 3:(220,220,  0),
    4:(107,142, 35), 5:( 70,130,180), 6:(220, 20, 60), 7:(  0,  0,142),
}
