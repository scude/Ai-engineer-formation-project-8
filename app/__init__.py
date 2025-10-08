"""Application factory for the segmentation demo UI and API."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask
from tensorflow import keras

from .config import APP_STATIC_FOLDER, APP_TEMPLATE_FOLDER, MODEL_PATH
from .inference import SegmentationService
from .augment_service import AugmentationService


def create_app(config: dict[str, Any] | None = None) -> Flask:
    """Create and configure the Flask application."""

    app = Flask(__name__, static_folder=APP_STATIC_FOLDER, template_folder=APP_TEMPLATE_FOLDER)

    if config:
        app.config.update(config)

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Segmentation model not found at {model_path}")
    model = keras.models.load_model(model_path, compile=False)
    app.extensions["segmentation_service"] = SegmentationService(model=model)
    app.extensions["augmentation_service"] = AugmentationService()

    from .routes import bp as routes_bp

    app.register_blueprint(routes_bp)

    return app