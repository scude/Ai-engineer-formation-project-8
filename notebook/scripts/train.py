# /scripts/train.py

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # masque INFO & WARNING C++
# Désactive quelques optimisations bruyantes
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_XLA_FLAGS", "--xla_cpu_enable_xla=false")

import re, shutil, argparse, csv, datetime
from typing import Callable
import tensorflow as tf
from tensorflow import keras
from .config import DataConfig, TrainConfig, AugmentConfig
from .data import build_dataset, prepare_labels_for_loss
from .models import AVAILABLE_MODELS, build_model
from .metrics import masked_mean_iou, masked_pixel_accuracy, dice_coef_wrapper
from .mlflow_utils import init_mlflow, start_run, KerasMlflowLogger
import mlflow, mlflow.keras

def _canonical_arch(name: str) -> str:
    """Normalise architecture aliases to the registered model keys."""

    if not name:
        raise ValueError("Architecture name must be a non-empty string")
    canonical = name.strip().lower().replace("-", "_")
    if canonical not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model architecture: {name}")
    return canonical


def _checkpoint_regex(monitor: str) -> re.Pattern[str]:
    escaped = re.escape(monitor)
    return re.compile(rf"([\w\-]+)\.{escaped}\.(\d+)-([0-9]*\.[0-9]+)\.keras$")


CKPT_RE_CACHE: dict[str, re.Pattern[str]] = {}


def _best_ckpt(path: str, monitor: str):
    if not os.path.isdir(path):
        return None, -1.0
    if monitor not in CKPT_RE_CACHE:
        CKPT_RE_CACHE[monitor] = _checkpoint_regex(monitor)
    pattern = CKPT_RE_CACHE[monitor]
    best, score = None, -1.0
    for f in os.listdir(path):
        m = pattern.match(f)
        if m:
            s = float(m.group(3))
            if s > score:
                best, score = os.path.join(path, f), s
    return best, score

def build_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "adam": return keras.optimizers.Adam(lr)
    if name == "adamw": return keras.optimizers.AdamW(lr)
    if name == "sgd": return keras.optimizers.SGD(lr, momentum=0.9, nesterov=True)
    raise ValueError(f"Unknown optimizer: {name}")

def train(model_name: str = "unet_small",
          data_cfg: DataConfig = DataConfig(),
          train_cfg: TrainConfig = TrainConfig(),
          aug_cfg: AugmentConfig = AugmentConfig()):
    model_name = _canonical_arch(model_name)
    train_cfg.arch = model_name
    # reproducibility
    tf.keras.utils.set_random_seed(data_cfg.seed)
    if train_cfg.deterministic_ops:
        tf.config.experimental.enable_op_determinism()
    # désactive JIT/XLA et layout optimizer pour éviter les messages "layout failed"
    tf.config.optimizer.set_jit(False)
    tf.config.optimizer.set_experimental_options({
        "layout_optimizer": False,
    })

    from tensorflow.keras import mixed_precision
    try:
        mixed_precision.set_global_policy(train_cfg.precision_policy)
    except ValueError as exc:
        raise ValueError(
            f"Invalid mixed precision policy '{train_cfg.precision_policy}'."
            " Accepted values include 'float32' and 'mixed_float16'."
        ) from exc
    policy = mixed_precision.global_policy()
    current_policy = policy.name
    compute_dtype = tf.dtypes.as_dtype(policy.compute_dtype)
    print(f"Using mixed precision policy: {current_policy} (compute dtype: {compute_dtype.name})")

    print(f"TF {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")
    os.makedirs(train_cfg.output_dir, exist_ok=True)
    ckpt_root = os.path.join(train_cfg.output_dir, "checkpoints")
    os.makedirs(ckpt_root, exist_ok=True)

    # data
    train_ds_with_weights = build_dataset(data_cfg, aug_cfg, split="train", training=True)
    val_ds_with_weights = build_dataset(
        data_cfg,
        aug_cfg.__class__(enabled=False),
        split="val",
        training=False,
    )

    for xb, yb, wb in train_ds_with_weights.take(1):
        print(f"probe -> x:{xb.shape} {xb.dtype} | y:{yb.shape} {yb.dtype} | w:{wb.shape} {wb.dtype}")

    def _drop_weights(ds):
        return ds.map(lambda x, y, w: (x, y), num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = _drop_weights(train_ds_with_weights)
    val_ds = _drop_weights(val_ds_with_weights)

    if compute_dtype != tf.float32:
        def _cast_inputs(ds):
            return ds.map(
                lambda x, y: (tf.cast(x, compute_dtype), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        train_ds = _cast_inputs(train_ds)
        val_ds = _cast_inputs(val_ds)

    # model
    input_shape = (data_cfg.height, data_cfg.width, 3)
    model = build_model(model_name, data_cfg.num_classes, input_shape)

    # compile
    base_loss = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE,
    )

    ignore_index = data_cfg.ignore_index
    num_classes = data_cfg.num_classes

    def _cross_entropy_component(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.int32)
        valid_mask = tf.not_equal(y_true, ignore_index)
        y_true_clean = prepare_labels_for_loss(y_true, ignore_index)
        per_pixel = base_loss(y_true_clean, y_pred)
        mask = tf.cast(valid_mask, per_pixel.dtype)
        per_pixel = per_pixel * mask
        per_example = tf.reduce_sum(per_pixel, axis=[1, 2])
        valid_counts = tf.reduce_sum(mask, axis=[1, 2])
        per_example = tf.where(
            valid_counts > 0,
            per_example / tf.maximum(valid_counts, tf.ones_like(valid_counts)),
            tf.zeros_like(per_example),
        )
        return per_example

    def _dice_component(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.int32)
        valid_mask = tf.not_equal(y_true, ignore_index)
        y_true_clean = prepare_labels_for_loss(y_true, ignore_index)
        y_true_one_hot = tf.one_hot(y_true_clean, depth=num_classes, dtype=tf.float32)
        probs = tf.nn.softmax(y_pred, axis=-1)
        mask = tf.cast(valid_mask, tf.float32)[..., tf.newaxis]

        y_true_one_hot = y_true_one_hot * mask
        probs = probs * mask

        intersection = tf.reduce_sum(y_true_one_hot * probs, axis=[1, 2])
        totals = tf.reduce_sum(y_true_one_hot + probs, axis=[1, 2])
        smooth = 1e-6
        dice_per_class = tf.where(
            totals > 0.0,
            (2.0 * intersection + smooth) / (totals + smooth),
            tf.zeros_like(totals),
        )
        valid_classes = tf.reduce_sum(tf.cast(totals > 0.0, tf.float32), axis=-1)
        dice_scores = tf.where(
            valid_classes > 0.0,
            tf.reduce_sum(dice_per_class, axis=-1) / tf.maximum(valid_classes, tf.ones_like(valid_classes)),
            tf.zeros_like(valid_classes),
        )
        return 1.0 - dice_scores

    loss_choice = (train_cfg.loss or "ce").lower()

    if loss_choice == "ce":
        loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] = _cross_entropy_component
    elif loss_choice == "dice":
        loss_fn = _dice_component
    elif loss_choice == "ce+dice":
        def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            ce = _cross_entropy_component(y_true, y_pred)
            dl = _dice_component(y_true, y_pred)
            return 0.5 * ce + 0.5 * dl
    else:
        raise ValueError(f"Unsupported loss choice: {train_cfg.loss}")

    train_cfg.loss = loss_choice

    metrics = [
        keras.metrics.MeanMetricWrapper(
            masked_pixel_accuracy,
            name="pix_acc",
            ignore_index=ignore_index,
        ),
        masked_mean_iou(
            num_classes=num_classes,
            ignore_index=ignore_index,
        )(),
        keras.metrics.MeanMetricWrapper(
            dice_coef_wrapper(num_classes=num_classes, ignore_index=ignore_index),
            name="dice_coef",
        ),
    ]
    opt = build_optimizer(train_cfg.optimizer, train_cfg.lr)
    if compute_dtype == tf.float16:

        # ``LossScaleOptimizer`` keeps gradients in float32 to avoid underflow when
        # using float16 tensors. Keras automatically applies dynamic loss scaling
        # for most built-in optimizers, but wrapping here guarantees the behaviour
        # for custom optimizers as well.
        if not isinstance(opt, mixed_precision.LossScaleOptimizer):
            opt = mixed_precision.LossScaleOptimizer(opt)
    model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)

    # MLflow
    init_mlflow(train_cfg.exp_name)
    run = start_run(run_name=f"{model_name}")
    run_id = getattr(getattr(run, "info", None), "run_id", None)
    if not run_id:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join(ckpt_root, run_id)
    os.makedirs(ckpt_dir, exist_ok=True)
    max_train_samples = getattr(train_cfg, "max_train_samples", None)
    max_val_samples = getattr(train_cfg, "max_val_samples", None)
    mlf_logger = KerasMlflowLogger({
        "arch": model_name,
        "model": model_name,
        "height": data_cfg.height,
        "width": data_cfg.width,
        "batch_size": data_cfg.batch_size,
        "lr": train_cfg.lr,
        "epochs": train_cfg.epochs,
        "optimizer": train_cfg.optimizer,
        "aug": vars(aug_cfg),
        "ignore_index": data_cfg.ignore_index,
        "loss": loss_choice,
        "precision_policy": current_policy,
    }, max_train_samples=max_train_samples, max_val_samples=max_val_samples)

    monitor_metric = "val_masked_mIoU"
    if loss_choice in {"dice", "ce+dice"}:
        monitor_metric = "val_dice_coef"

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                ckpt_dir,
                f"{model_name}.{monitor_metric}.{{epoch:03d}}-{{{monitor_metric}:.4f}}.keras",
            ),
            monitor=monitor_metric,
            mode="max",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            mode="max",
            patience=train_cfg.early_stop_patience,
            restore_best_weights=True,
        ),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.CSVLogger(os.path.join(train_cfg.output_dir, "train_log.csv")),
        mlf_logger,
    ]

    hist = model.fit(train_ds, validation_data=val_ds, epochs=train_cfg.epochs, callbacks=callbacks, verbose=1)

    # evaluate restored weights (EarlyStopping may have reloaded best checkpoint)
    restored_metrics = model.evaluate(val_ds, return_dict=True)
    print("Restored weights evaluation:")
    for name, value in restored_metrics.items():
        print(f"  {name}: {value:.6f}")
        mlflow.log_metric(f"restored_{name}", float(value))

    hist_len = len(hist.history.get("loss", []))
    if hist_len:
        for name, value in restored_metrics.items():
            float_value = float(value)
            mlflow.log_metric(f"val_{name}", float_value, step=hist_len)

            val_key = f"val_{name}"
            if val_key in hist.history and hist.history[val_key]:
                hist.history[val_key][-1] = float_value
            elif val_key in hist.history:
                hist.history[val_key] = [float_value]
            else:
                hist.history[val_key] = [float("nan")] * (hist_len - 1) + [float_value]

    csv_path = os.path.join(train_cfg.output_dir, "train_log.csv")
    if hist_len and os.path.exists(csv_path):
        try:
            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                fieldnames = rows[0].keys()
                updated = False
                for name, value in restored_metrics.items():
                    val_key = f"val_{name}"
                    if val_key in rows[-1]:
                        rows[-1][val_key] = str(float(value))
                        updated = True
                if updated:
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)
        except Exception as e:
            print(f"Failed to update CSV log with restored metrics: {e}")

    # save final
    final_path = os.path.join(train_cfg.output_dir, f"{model_name}_final.keras")
    model.save(final_path)
    print(f"saved -> {final_path}")

    # best ckpt copy
    best_path, best_score = _best_ckpt(ckpt_dir, monitor_metric)
    best_export = None
    if best_path:
        best_export = os.path.join(train_cfg.output_dir, f"{model_name}_best.keras")
        shutil.copy2(best_path, best_export)
        print(f"best -> {os.path.basename(best_path)} ({monitor_metric}={best_score:.4f}) | exported: {best_export}")

    # log artifacts
    mlflow.log_artifact(os.path.join(train_cfg.output_dir, "train_log.csv"), artifact_path="logs")
    if os.path.exists(final_path): mlflow.log_artifact(final_path, artifact_path="models")
    if best_export and os.path.exists(best_export):
        mlflow.log_artifact(best_export, artifact_path="models")
        try:
            bm = keras.models.load_model(best_export, compile=False)
            mlflow.keras.log_model(bm, artifact_path="best_model")
            mlflow.log_metric(f"best_{monitor_metric}", float(best_score))
        except Exception as e:
            print("mlflow keras flavor failed:", e)

    if os.path.isdir(ckpt_dir):
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    mlflow.end_run()
    print("MLflow run ended.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="Legacy alias for --arch")
    p.add_argument("--arch", default=None, help="Model architecture to train")
    p.add_argument("--data_root", default="../data")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_val_samples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--optimizer", default="adam", choices=["adam","adamw","sgd"])
    p.add_argument("--loss", default="ce", choices=["ce", "dice", "ce+dice"])
    p.add_argument(
        "--deterministic_ops",
        action="store_true",
        help="Enable TensorFlow deterministic ops (may reduce performance and require extra scratch space)",
    )
    p.add_argument(
        "--precision_policy",
        default="float32",
        choices=["float32", "mixed_float16"],
        help=(
            "Mixed precision policy to apply. 'mixed_float16' keeps variables in float32 while "
            "using float16 activations to reduce GPU memory usage. When enabling it, ensure the "
            "optimizer supports loss scaling (the script wraps common optimizers with "
            "tf.keras.mixed_precision.LossScaleOptimizer)."
        ),
    )

    # Aug params
    p.add_argument("--aug_enabled", type=int, default=1)
    p.add_argument("--hflip", type=int, default=1)
    p.add_argument("--vflip", type=int, default=0)
    p.add_argument("--rotate", type=float, default=0.0)
    p.add_argument("--scale_min", type=float, default=1.0)
    p.add_argument("--scale_max", type=float, default=1.0)
    p.add_argument("--random_crop", type=int, default=0)
    p.add_argument("--brightness", type=float, default=0.0)
    p.add_argument("--contrast", type=float, default=0.0)
    p.add_argument("--saturation", type=float, default=0.0)
    p.add_argument("--hue", type=float, default=0.0)
    p.add_argument("--noise_std", type=float, default=0.0)

    args = p.parse_args()

    arch_arg = args.arch or args.model or "unet_small"
    try:
        arch = _canonical_arch(arch_arg)
    except ValueError as exc:
        p.error(str(exc))

    loss_choice = args.loss.lower()

    data_cfg = DataConfig(
        data_root=args.data_root, height=args.height, width=args.width,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples
    )
    train_cfg = TrainConfig(
        lr=args.lr,
        epochs=args.epochs,
        optimizer=args.optimizer,
        arch=arch,
        loss=loss_choice,
        deterministic_ops=bool(args.deterministic_ops),
        precision_policy=args.precision_policy,
    )
    aug_cfg = AugmentConfig(
        enabled=bool(args.aug_enabled), hflip=bool(args.hflip), vflip=bool(args.vflip),
        random_rotate_deg=args.rotate, random_scale_min=args.scale_min, random_scale_max=args.scale_max,
        random_crop=bool(args.random_crop), brightness_delta=args.brightness, contrast_delta=args.contrast,
        saturation_delta=args.saturation, hue_delta=args.hue, gaussian_noise_std=args.noise_std
    )
    train(arch, data_cfg, train_cfg, aug_cfg)
