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