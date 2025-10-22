"""Augmentation helpers using the notebook Albumentations pipeline."""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image

from notebook.scripts.augment import (
    _build_albu_pipeline,
    get_bhuiya_transform_specs,
)
from notebook.scripts.config import DataConfig

from .config import AUGMENT_CONFIG


@dataclass(frozen=True)
class AugmentedImage:
    """Representation of a single augmented image."""

    name: str
    image: Image.Image


@dataclass(frozen=True)
class AugmentationPreview:
    """Container for the original and augmented images."""

    original: Image.Image
    augmentations: list[AugmentedImage]


class AugmentationService:
    """Generate augmentation previews compatible with the training pipeline."""

    def __init__(self) -> None:
        data_cfg = DataConfig()
        pipeline = _build_albu_pipeline(
            AUGMENT_CONFIG, data_cfg.height, data_cfg.width, data_cfg.ignore_index
        )
        if pipeline is None:
            raise ValueError("Augmentation pipeline is disabled in configuration")
        self._ignore_index = data_cfg.ignore_index
        self._pipeline = pipeline
        self._transform_specs = tuple(get_bhuiya_transform_specs())
        self._spec_by_name = {spec.name: spec for spec in self._transform_specs}

    @staticmethod
    def _read_image(data: bytes) -> Image.Image:
        with BytesIO(data) as buffer:
            return Image.open(buffer).convert("RGB")

    def _apply_transform(self, image_array: np.ndarray, spec_name: str) -> Image.Image:
        spec = self._spec_by_name[spec_name]
        transform = spec.build(self._ignore_index)
        augmented = transform(image=image_array)
        return Image.fromarray(augmented["image"])

    def generate(self, data: bytes, samples: int = 6) -> AugmentationPreview:
        """Produce a preview containing random Bhuiya et al. augmentations."""

        base_image = self._read_image(data)
        image_array = np.asarray(base_image)
        augmented_images: list[AugmentedImage] = []
        usage_counter: Counter[str] = Counter()

        for _ in range(samples):
            spec = random.choice(self._transform_specs)
            usage_counter[spec.name] += 1
            label = (
                spec.name
                if usage_counter[spec.name] == 1
                else f"{spec.name} ({usage_counter[spec.name]})"
            )
            transform = spec.build(self._ignore_index)
            augmented = transform(image=image_array)
            aug_img = Image.fromarray(augmented["image"])
            augmented_images.append(AugmentedImage(name=label, image=aug_img))

        return AugmentationPreview(original=base_image, augmentations=augmented_images)

    def gallery(self, data: bytes) -> AugmentationPreview:
        """Return one preview per Bhuiya et al. transform."""

        base_image = self._read_image(data)
        image_array = np.asarray(base_image)
        augmented_images = [
            AugmentedImage(name=spec.name, image=self._apply_transform(image_array, spec.name))
            for spec in self._transform_specs
        ]
        return AugmentationPreview(original=base_image, augmentations=augmented_images)
