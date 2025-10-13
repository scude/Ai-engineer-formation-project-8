"""Configuration constants for the Flask segmentation demo."""
from __future__ import annotations

from pathlib import Path
from typing import Final

from dataclasses import replace

from notebook.scripts.config import (
    AugmentConfig,
    DEFAULT_AUGMENT_CONFIG,
    DEFAULT_PALETTE_8,
)

BASE_DIR: Final[Path] = Path(__file__).resolve().parent
MODEL_PATH: Final[Path] = (
    BASE_DIR / "models" / "deeplab_resnet50_final.keras"
)

CLASS_NAMES: Final[list[str]] = [
    "Road",
    "Sidewalk",
    "Building",
    "Street furniture",
    "Vegetation",
    "Sky",
    "Person",
    "Vehicle",
]

PALETTE: Final[dict[int, tuple[int, int, int]]] = DEFAULT_PALETTE_8

AUGMENT_CONFIG: Final[AugmentConfig] = replace(DEFAULT_AUGMENT_CONFIG)

APP_STATIC_FOLDER: Final[str] = str(BASE_DIR / "static")
APP_TEMPLATE_FOLDER: Final[str] = str(BASE_DIR / "templates")

MAX_CONTENT_LENGTH: Final[int] = 16 * 1024 * 1024
"""Maximum upload size (16 MiB)."""