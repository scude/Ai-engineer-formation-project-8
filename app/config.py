"""Configuration constants for the Flask segmentation demo."""
from __future__ import annotations

from pathlib import Path
from typing import Final

from notebook.scripts.config import AugmentConfig, DEFAULT_PALETTE_8

BASE_DIR: Final[Path] = Path(__file__).resolve().parent
MODEL_PATH: Final[Path] = (
    BASE_DIR / "models" / "deeplab_resnet50_final.keras"
)

CLASS_NAMES: Final[list[str]] = [
    "Road",
    "Sidewalk",
    "Building",
    "Vegetation",
    "Terrain",
    "Sky",
    "Person",
    "Vehicle",
]

PALETTE: Final[dict[int, tuple[int, int, int]]] = DEFAULT_PALETTE_8

AUGMENT_CONFIG: Final[AugmentConfig] = AugmentConfig(
    enabled=True,
    hflip=True,
    vflip=False,
    random_rotate_deg=10.0,
    random_scale_min=0.75,
    random_scale_max=1.25,
    random_crop=True,
    brightness_delta=0.2,
    contrast_delta=0.2,
    saturation_delta=0.2,
    hue_delta=0.05,
    gaussian_noise_std=0.01,
)

APP_STATIC_FOLDER: Final[str] = str(BASE_DIR / "static")
APP_TEMPLATE_FOLDER: Final[str] = str(BASE_DIR / "templates")

MAX_CONTENT_LENGTH: Final[int] = 16 * 1024 * 1024
"""Maximum upload size (16 MiB)."""