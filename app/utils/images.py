"""Image utility helpers."""
from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image


def image_to_data_url(image: Image.Image, format: str = "PNG") -> str:
    """Convert a PIL image to a base64 data URL."""

    with BytesIO() as buffer:
        image.save(buffer, format=format)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"