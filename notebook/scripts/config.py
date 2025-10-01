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

@dataclass
class AugmentConfig:
    # geometric
    hflip: bool = True
    vflip: bool = False
    random_rotate_deg: float = 0.0     # e.g., 5.0
    random_scale_min: float = 1.0      # e.g., 0.75
    random_scale_max: float = 1.0      # e.g., 1.25
    random_crop: bool = False          # if True, do scale+crop to (H,W)
    # photometric (image-only)
    brightness_delta: float = 0.0      # e.g., 0.1
    contrast_delta: float = 0.0        # e.g., 0.1
    saturation_delta: float = 0.0      # e.g., 0.1
    hue_delta: float = 0.0             # e.g., 0.02
    gaussian_noise_std: float = 0.0    # e.g., 0.01
    # enable/disable all aug
    enabled: bool = True

DEFAULT_PALETTE_8 = {
    0:(128, 64,128), 1:(244, 35,232), 2:( 70, 70, 70), 3:(220,220,  0),
    4:(107,142, 35), 5:( 70,130,180), 6:(220, 20, 60), 7:(  0,  0,142),
}
