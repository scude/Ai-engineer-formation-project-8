"""Validation diagnostics for segmentation models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image

from .config import DEFAULT_PALETTE_8, DataConfig
from .data import decode_img, decode_lbl
from .data_utils import resize_image_and_mask
from .remap import remap_label_ids

__all__ = [
    "BatchSample",
    "colorize_mask",
    "visualize_val_batch",
    "compute_val_confusion_matrix",
]


@dataclass
class BatchSample:
    """Container bundling the tensors needed for validation diagnostics."""

    image: tf.Tensor
    label: tf.Tensor
    path: str


def _ensure_palette(palette: Optional[Mapping[int, Tuple[int, int, int]]]) -> Mapping[int, Tuple[int, int, int]]:
    return palette if palette is not None else DEFAULT_PALETTE_8


def _load_sample(image_path: str, label_path: str, cfg: DataConfig) -> BatchSample:
    """Load and preprocess a single validation sample."""

    image = decode_img(tf.constant(image_path))
    label = decode_lbl(tf.constant(label_path))
    label = remap_label_ids(label, cfg.ignore_index)
    image, label = resize_image_and_mask(image, label, cfg.height, cfg.width)

    image = tf.ensure_shape(image, [cfg.height, cfg.width, 3])
    label = tf.ensure_shape(label, [cfg.height, cfg.width])

    return BatchSample(image=image, label=label, path=image_path)


def colorize_mask(
    mask: np.ndarray,
    palette: Optional[Mapping[int, Tuple[int, int, int]]] = None,
    ignore_index: Optional[int] = None,
) -> np.ndarray:
    """Convert a label mask to a colour image."""

    palette = _ensure_palette(palette)
    mask = np.asarray(mask)
    coloured = np.zeros((*mask.shape, 3), dtype=np.uint8)

    for cls_id, colour in palette.items():
        coloured[mask == cls_id] = colour

    if ignore_index is not None:
        coloured[mask == ignore_index] = (0, 0, 0)

    return coloured


def visualize_val_batch(
    image_paths: Sequence[str],
    label_paths: Sequence[str],
    model: tf.keras.Model,
    cfg: DataConfig,
    palette: Optional[Mapping[int, Tuple[int, int, int]]] = None,
    max_items: Optional[int] = None,
) -> List[Image.Image]:
    """Render a batch of validation samples with predictions."""

    palette = _ensure_palette(palette)
    count = len(image_paths)
    if max_items is not None:
        count = min(count, max_items)

    outputs: List[Image.Image] = []

    for idx in range(count):
        sample = _load_sample(image_paths[idx], label_paths[idx], cfg)

        logits = model(sample.image[tf.newaxis, ...], training=False)
        pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
        pred_mask = tf.squeeze(pred, axis=0)

        image_np = tf.clip_by_value(sample.image * 255.0, 0.0, 255.0)
        image_np = tf.cast(image_np, tf.uint8).numpy()

        gt_np = sample.label.numpy()
        pred_np = pred_mask.numpy()

        gt_color = colorize_mask(gt_np, palette, cfg.ignore_index)
        pred_color = colorize_mask(pred_np, palette, cfg.ignore_index)

        stacked = np.concatenate([image_np, gt_color, pred_color], axis=1)
        outputs.append(Image.fromarray(stacked))

    return outputs


def _iter_batches(
    image_paths: Sequence[str],
    label_paths: Sequence[str],
    batch_size: int,
    cfg: DataConfig,
) -> Iterable[Tuple[tf.Tensor, tf.Tensor]]:
    for start in range(0, len(image_paths), batch_size):
        end = min(start + batch_size, len(image_paths))
        images, labels = [], []
        for idx in range(start, end):
            sample = _load_sample(image_paths[idx], label_paths[idx], cfg)
            images.append(sample.image)
            labels.append(sample.label)
        yield tf.stack(images, axis=0), tf.stack(labels, axis=0)


def compute_val_confusion_matrix(
    image_paths: Sequence[str],
    label_paths: Sequence[str],
    model: tf.keras.Model,
    cfg: DataConfig,
    batch_size: int = 4,
) -> Tuple[np.ndarray, Dict[int, float]]:
    """Compute the confusion matrix and per-class IoU on the validation set."""

    num_classes = cfg.num_classes
    ignore_index = cfg.ignore_index
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for images, labels in _iter_batches(image_paths, label_paths, batch_size, cfg):
        logits = model(images, training=False)
        preds = tf.argmax(logits, axis=-1, output_type=tf.int32)

        labels_np = labels.numpy()
        preds_np = preds.numpy()

        valid_mask = labels_np != ignore_index
        y_true = labels_np[valid_mask]
        y_pred = preds_np[valid_mask]

        if y_true.size == 0:
            continue

        batch_confusion = tf.math.confusion_matrix(
            y_true,
            y_pred,
            num_classes=num_classes,
            dtype=tf.int64,
        ).numpy()
        confusion += batch_confusion

    per_class_iou: Dict[int, float] = {}
    for cls_idx in range(num_classes):
        intersection = confusion[cls_idx, cls_idx]
        ground_truth = confusion[cls_idx, :].sum()
        predicted = confusion[:, cls_idx].sum()
        union = ground_truth + predicted - intersection
        iou = float(intersection / union) if union > 0 else float("nan")
        per_class_iou[cls_idx] = iou

    return confusion, per_class_iou
notebook/scripts/validate.py
Nouveau
+127
-0

"""Run validation diagnostics on a trained segmentation model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

from .config import DataConfig, DEFAULT_PALETTE_8
from .data import gather_pairs
from .diagnostics import compute_val_confusion_matrix, visualize_val_batch


def _build_data_config(args: argparse.Namespace) -> DataConfig:
    cfg = DataConfig(
        data_root=args.data_root,
        height=args.height,
        width=args.width,
        batch_size=args.batch_size,
        max_val_samples=args.max_val_samples,
    )
    return cfg


def run_validation(
    model_path: str,
    data_cfg: DataConfig,
    output_dir: str,
    split: str = "val",
    num_visuals: Optional[int] = 4,
    batch_size: int = 4,
) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {model_path}")
    model = keras.models.load_model(model_path, compile=False)

    image_paths, label_paths, missing = gather_pairs(data_cfg, split)
    if missing:
        print(f"[warn] {len(missing)} label files missing for split '{split}'")
    if not image_paths:
        raise RuntimeError(f"No samples found for split '{split}'")

    if data_cfg.max_val_samples is not None:
        image_paths = image_paths[: data_cfg.max_val_samples]
        label_paths = label_paths[: data_cfg.max_val_samples]

    print(f"Evaluating {len(image_paths)} samples")

    # Visualise a few samples
    visualisations = visualize_val_batch(
        image_paths,
        label_paths,
        model,
        data_cfg,
        palette=DEFAULT_PALETTE_8,
        max_items=num_visuals,
    )

    for idx, pil_img in enumerate(visualisations):
        out_path = Path(output_dir) / f"sample_{idx:03d}.png"
        pil_img.save(out_path)
        print(f"Saved visualisation -> {out_path}")

    confusion, per_class_iou = compute_val_confusion_matrix(
        image_paths,
        label_paths,
        model,
        data_cfg,
        batch_size=batch_size,
    )

    np.save(Path(output_dir) / "confusion.npy", confusion)

    summary = {
        "per_class_iou": {str(k): float(v) for k, v in per_class_iou.items()},
        "mean_iou": float(np.nanmean(list(per_class_iou.values()))),
        "num_samples": len(image_paths),
        "ignore_index": data_cfg.ignore_index,
    }

    with open(Path(output_dir) / "validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Validation summary:")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation diagnostics")
    parser.add_argument("--model_path", required=True, help="Path to the trained model (.keras)")
    parser.add_argument("--data_root", default="../data")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--output_dir", default="artifacts/validation")
    parser.add_argument("--split", default="val")
    parser.add_argument("--num_visuals", type=int, default=4)
    parser.add_argument("--conf_batch_size", type=int, default=4)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(1337)
    tf.config.experimental.enable_op_determinism()
    tf.config.optimizer.set_jit(False)
    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

    data_cfg = _build_data_config(args)
    run_validation(
        model_path=args.model_path,
        data_cfg=data_cfg,
        output_dir=args.output_dir,
        split=args.split,
        num_visuals=args.num_visuals,
        batch_size=args.conf_batch_size,
    )


if __name__ == "__main__":
    main()