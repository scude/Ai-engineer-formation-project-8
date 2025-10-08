"""Segmentation inference helpers for the Flask API."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from tensorflow import keras

from .config import CLASS_NAMES, PALETTE


@dataclass(frozen=True)
class SegmentationResult:
    """Container for the images produced by a segmentation inference."""

    original: Image.Image
    mask: Image.Image
    overlay: Image.Image


class SegmentationService:
    """High level service that performs image segmentation with a Keras model."""

    def __init__(self, model: keras.Model) -> None:
        self._model = model
        input_shape = getattr(self._model, "input_shape", None)
        if not isinstance(input_shape, Iterable) or len(input_shape) != 4:
            raise ValueError("Model input shape must be a 4D iterable")
        _, height, width, _ = input_shape
        if height is None or width is None:
            raise ValueError("Model input shape must define height and width")
        self._height = int(height)
        self._width = int(width)

    @staticmethod
    def _read_image(data: bytes) -> Image.Image:
        with BytesIO(data) as buffer:
            image = Image.open(buffer).convert("RGB")
        return image

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        resized = image.resize((self._width, self._height), Image.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def _predict_mask(self, batch: np.ndarray) -> np.ndarray:
        logits = self._model.predict(batch, verbose=0)
        mask = np.argmax(logits, axis=-1)[0]
        return mask.astype(np.uint8)

    def _resize_mask(self, mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        resized = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        return resized

    def _colorize_mask(self, mask: np.ndarray) -> Image.Image:
        color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for cls_id, color in PALETTE.items():
            color_mask[mask == cls_id] = color
        return Image.fromarray(color_mask)

    @staticmethod
    def _blend_images(image: Image.Image, mask_image: Image.Image, alpha: float = 0.5) -> Image.Image:
        return Image.blend(image, mask_image, alpha)

    def predict(self, data: bytes) -> SegmentationResult:
        """Run model inference on the given image bytes."""

        original = self._read_image(data)
        batch = self._preprocess(original)
        mask = self._predict_mask(batch)
        resized_mask = self._resize_mask(mask, original.size)
        mask_image = self._colorize_mask(resized_mask)
        overlay = self._blend_images(original, mask_image)
        return SegmentationResult(original=original, mask=mask_image, overlay=overlay)

    @staticmethod
    def legend() -> list[tuple[str, str]]:
        """Return the class legend as name/hex pairs for rendering."""

        legend_entries: list[tuple[str, str]] = []
        for idx, name in enumerate(CLASS_NAMES):
            color = PALETTE.get(idx, (255, 255, 255))
            legend_entries.append((name, "#%02x%02x%02x" % color))
        return legend_entries