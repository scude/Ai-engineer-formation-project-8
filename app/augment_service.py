"""Augmentation helpers using the notebook Albumentations pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image

from notebook.scripts.augment import _build_albu_pipeline
from notebook.scripts.config import DataConfig

from .config import AUGMENT_CONFIG


@dataclass(frozen=True)
class AugmentedImage:
    """Representation of a single augmented image."""

    name: str
    image: Image.Image


class AugmentationService:
    """Generate augmentation previews compatible with the training pipeline."""

    def __init__(self) -> None:
        data_cfg = DataConfig()
        pipeline = _build_albu_pipeline(
            AUGMENT_CONFIG, data_cfg.height, data_cfg.width, data_cfg.ignore_index
        )
        if pipeline is None:
            raise ValueError("Augmentation pipeline is disabled in configuration")
        self._pipeline = pipeline

    @staticmethod
    def _read_image(data: bytes) -> Image.Image:
        with BytesIO(data) as buffer:
            return Image.open(buffer).convert("RGB")

    def generate(self, data: bytes, samples: int = 6) -> list[AugmentedImage]:
        """Produce a list of augmented versions of the given image."""

        base_image = self._read_image(data)
        image_array = np.asarray(base_image)
        augmented_images: list[AugmentedImage] = []

        for idx in range(samples):
            augmented = self._pipeline(image=image_array)
            aug_img = Image.fromarray(augmented["image"])
            augmented_images.append(AugmentedImage(name=f"Augmentation {idx + 1}", image=aug_img))
        return augmented_images