# /scripts/config.py
from dataclasses import dataclass
from typing import Optional

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
    deterministic_input: bool = False
    cache_val: bool = False
    cache_val_path: Optional[str] = None
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    verbose: bool = True

@dataclass
class TrainConfig:
    lr: float = 5e-4
    epochs: int = 80
    optimizer: str = "adamw"
    momentum: Optional[float] = None
    weight_decay: Optional[float] = 1e-4
    poly_power: Optional[float] = None
    lr_schedule: str = "cosine_warmup"
    warmup_epochs: float = 5.0
    min_lr_ratio: float = 0.05
    cosine_cycles: float = 1.0
    early_stop_patience: int = 12
    output_dir: str = "artifacts"
    exp_name: str = "cityscapes-seg-8cls"
    arch: str = "unet_small"
    loss: str = "ce"
    deterministic_ops: bool = False
    precision_policy: str = "float32"

@dataclass
class AugmentConfig:
    """Parameters of the Albumentations pipeline used for Cityscapes training.

    The defaults mirror the production settings for DeepLabV3+ (ResNet-50) and
    implement the data augmentation recipe requested in the project brief.  The
    individual probabilities can be overridden in tests or experiments to
    isolate the effect of a particular transform while keeping the rest of the
    pipeline unchanged.
    """

    enabled: bool = True

    # geometric
    horizontal_flip_prob: float = 0.5
    random_resized_crop_scale: tuple[float, float] = (0.25, 0.6)
    # Intervalle d'aire relatif utilisé par RandomResizedCrop. Les bornes sont
    # automatiquement triées, bornées dans ]0, 1] pour respecter les exigences
    # d'Albumentations et appliquées telles quelles lorsque le ratio n'est pas
    # verrouillé.
    random_resized_crop_ratio: tuple[float, float] = (0.75, 1.33)
    # Verrouille le ratio cible sur la géométrie Cityscapes. Quand ce drapeau
    # est actif (par défaut), RandomResizedCrop fonctionne avec un ratio fixe de
    # 1024/512 tout en respectant la fenêtre d'échelle configurée.
    lock_random_resized_crop_ratio: bool = True
    # Budget optionnel permettant de rapprocher les bornes de ratio du format
    # Cityscapes lorsque le verrou est désactivé. Une valeur nulle conserve le
    # comportement historique tandis qu'une valeur positive limite l'écart
    # relatif autour du ratio 2.0.
    max_ratio_jitter: float = 0.0


    shift_scale_rotate_prob: float = 1.0
    shift_limit: float = 0.1
    scale_limit: float = 0.5
    rotate_limit: float = 15.0

    # photometric
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.5
    color_jitter_saturation: float = 0.5
    color_jitter_hue: float = 0.2

    # stochastic image corruption
    gaussian_blur_prob: float = 0.3
    gaussian_blur_kernel: tuple[int, int] = (3, 5)

    gauss_noise_prob: float = 0.3
    gauss_noise_var_limit: tuple[float, float] = (1.0, 25.0)

    grid_dropout_prob: float = 0.0 #0.3
    grid_dropout_ratio: float = 0.2
    grid_dropout_unit_size: int = 120

DEFAULT_AUGMENT_CONFIG = AugmentConfig()

DEFAULT_PALETTE_8 = {
    0:(128, 64,128), 1:(244, 35,232), 2:( 70, 70, 70), 3:(220,220,  0),
    4:(107,142, 35), 5:( 70,130,180), 6:(220, 20, 60), 7:(  0,  0,142),
}
