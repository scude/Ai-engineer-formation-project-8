"""Flask route and API definitions."""
from __future__ import annotations

from http import HTTPStatus

from flask import Blueprint, Response, current_app, jsonify, render_template, request

from .inference import SegmentationService
from .utils.images import image_to_data_url

bp = Blueprint("main", __name__)


@bp.route("/")
def index() -> str:
    service: SegmentationService = current_app.extensions["segmentation_service"]
    legend = service.legend()
    return render_template("index.html", legend=legend)


@bp.route("/predict", methods=["POST"])
def predict() -> Response:
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "Image file is required"}), HTTPStatus.BAD_REQUEST

    service: SegmentationService = current_app.extensions["segmentation_service"]
    result = service.predict(file.read())

    payload = {
        "original": image_to_data_url(result.original),
        "mask": image_to_data_url(result.mask),
        "overlay": image_to_data_url(result.overlay),
    }
    return jsonify(payload)


@bp.route("/augment", methods=["POST"])
def augment() -> Response:
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "Image file is required"}), HTTPStatus.BAD_REQUEST

    service = current_app.extensions["augmentation_service"]
    preview = service.generate(file.read())
    payload = {
        "original": image_to_data_url(preview.original),
        "augmentations": [
            {"name": item.name, "image": image_to_data_url(item.image)}
            for item in preview.augmentations
        ]
    }
    return jsonify(payload)