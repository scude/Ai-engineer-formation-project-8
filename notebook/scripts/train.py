# /scripts/train.py

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # masque INFO & WARNING C++
# Désactive quelques optimisations bruyantes
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_XLA_FLAGS", "--xla_cpu_enable_xla=false")

import re, shutil, argparse, csv, datetime, gc, math
from dataclasses import replace
from typing import Any, Callable, Optional
import tensorflow as tf
from tensorflow import keras
from .config import (
    DataConfig,
    TrainConfig,
    AugmentConfig,
    DEFAULT_AUGMENT_CONFIG,
)
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


class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine decay with an optional linear warm-up phase."""

    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        *,
        min_lr_ratio: float = 0.0,
        cycles: float = 1.0,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be > 0 for WarmupCosineSchedule")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0 for WarmupCosineSchedule")
        if min_lr_ratio < 0.0:
            raise ValueError("min_lr_ratio must be >= 0")
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self.cycles = float(cycles)

    def __call__(self, step: tf.Tensor) -> tf.Tensor:
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        base_lr = tf.cast(self.base_lr, tf.float32)
        min_lr = base_lr * tf.cast(self.min_lr_ratio, tf.float32)

        if self.warmup_steps > 0:
            warmup_progress = tf.clip_by_value(step / tf.maximum(warmup_steps, 1.0), 0.0, 1.0)
            warmup_lr = base_lr * warmup_progress
        else:
            warmup_lr = base_lr

        decay_steps = tf.maximum(total_steps - warmup_steps, 1.0)
        post_warmup_step = tf.maximum(step - warmup_steps, 0.0)
        progress = tf.clip_by_value(post_warmup_step / decay_steps, 0.0, 1.0)
        cosine_argument = math.pi * progress * self.cycles
        cosine_decay = 0.5 * (1.0 + tf.cos(cosine_argument))
        decayed_lr = min_lr + (base_lr - min_lr) * cosine_decay

        return tf.where(step < warmup_steps, warmup_lr, decayed_lr)

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "cycles": self.cycles,
        }

def build_optimizer(name: str, lr, momentum: float | None = None, weight_decay: float | None = None):
    name = name.lower()
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    if name == "adamw":
        kwargs = {}
        if weight_decay is not None:
            kwargs["weight_decay"] = weight_decay
        return keras.optimizers.AdamW(learning_rate=lr, **kwargs)
    if name == "sgd":
        kwargs = {"momentum": 0.9, "nesterov": True}
        if momentum is not None:
            kwargs["momentum"] = momentum
        if weight_decay is not None:
            kwargs["weight_decay"] = weight_decay
        try:
            return keras.optimizers.SGD(learning_rate=lr, **kwargs)
        except TypeError:
            kwargs.pop("weight_decay", None)
            opt = keras.optimizers.SGD(learning_rate=lr, **kwargs)
            if weight_decay is not None:
                setattr(opt, "weight_decay", weight_decay)
            return opt
    raise ValueError(f"Unknown optimizer: {name}")

def train(model_name: str = "unet_small",
          data_cfg: DataConfig = DataConfig(),
          train_cfg: TrainConfig = TrainConfig(),
          aug_cfg: Optional[AugmentConfig] = None,
          model_kwargs: Optional[dict[str, Any]] = None,
          *,
          use_mlflow: bool = True,
          keep_artifacts: bool = True,
          cleanup_after: bool = False,
          probe_dataset: bool = True):
    """Train a segmentation model.

    Parameters
    ----------
    model_name:
        Architecture key registered in :data:`AVAILABLE_MODELS`.
    data_cfg:
        Dataset configuration describing input sizes, sampling limits, etc.
    train_cfg:
        Training hyper-parameters (optimizer, learning rate schedule, output
        paths, ...).
    aug_cfg:
        Optional augmentation configuration. When ``None`` the default
        augmentation settings are copied to avoid mutating shared state.
    model_kwargs:
        Extra keyword arguments forwarded to :func:`build_model`.
    use_mlflow:
        When ``False`` the run avoids initialising MLflow, which keeps the
        Optuna objective lightweight.
    keep_artifacts:
        Toggle checkpoint and CSV exports. For Optuna sweeps it is common to
        disable artifact retention so temporary files do not accumulate.
    cleanup_after:
        When ``True`` datasets, compiled models and temporary directories are
        explicitly released at the end of training. This is designed for the
        Optuna objective where repeated trials in a single Python process would
        otherwise retain large objects in memory.
    probe_dataset:
        When ``True`` the loader fetches a single batch to report the shapes and
        data types flowing through the pipeline. Disable it to skip the
        additional warm-up pass when minimising start-up time (e.g. within
        Optuna trials).
    """
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except (RuntimeError, ValueError):
            # Memory growth must be set before GPUs are initialized. If this
            # fails we simply continue with the default allocation strategy.
            pass
    if aug_cfg is None:
        aug_cfg = replace(DEFAULT_AUGMENT_CONFIG)
    if model_kwargs is None:
        model_kwargs = {}
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
    if keep_artifacts:
        os.makedirs(ckpt_root, exist_ok=True)

    # data
    train_ds_with_weights = build_dataset(data_cfg, aug_cfg, split="train", training=True)
    val_ds_with_weights = build_dataset(
        data_cfg,
        aug_cfg.__class__(enabled=False),
        split="val",
        training=False,
    )

    if probe_dataset:
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

    steps_per_epoch = None
    cardinality = None
    try:
        cardinality = tf.data.experimental.cardinality(train_ds)
        card_value = int(cardinality.numpy())
        if card_value >= 0:
            steps_per_epoch = card_value
    except (TypeError, AttributeError):
        try:
            card_value = int(cardinality)
            if card_value >= 0:
                steps_per_epoch = card_value
        except Exception:
            steps_per_epoch = None

    # model
    input_shape = (data_cfg.height, data_cfg.width, 3)
    model = build_model(model_name, data_cfg.num_classes, input_shape, **model_kwargs)

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
    effective_lr = train_cfg.lr
    lr_schedule_name = "constant"
    schedule_choice = (train_cfg.lr_schedule or "").lower()
    if train_cfg.poly_power is not None:
        if steps_per_epoch is None:
            total_steps = train_cfg.epochs
        else:
            total_steps = max(1, steps_per_epoch * max(train_cfg.epochs, 1))
        effective_lr = keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=train_cfg.lr,
            decay_steps=total_steps,
            end_learning_rate=0.0,
            power=train_cfg.poly_power,
        )
        lr_schedule_name = "polynomial_decay"
    else:
        if schedule_choice in {"cosine_warmup", "warmup_cosine"} and steps_per_epoch is not None:
            warmup_epochs = max(0.0, float(train_cfg.warmup_epochs))
            warmup_steps = int(round(warmup_epochs * steps_per_epoch))
            total_steps = max(1, int(train_cfg.epochs * max(steps_per_epoch, 1)))
            min_ratio = max(0.0, float(train_cfg.min_lr_ratio))
            cycles = max(1e-3, float(train_cfg.cosine_cycles))
            effective_lr = WarmupCosineSchedule(
                base_lr=train_cfg.lr,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr_ratio=min_ratio,
                cycles=cycles,
            )
            lr_schedule_name = "cosine_warmup"

    opt = build_optimizer(
        train_cfg.optimizer,
        effective_lr,
        momentum=train_cfg.momentum,
        weight_decay=train_cfg.weight_decay,
    )
    if compute_dtype == tf.float16:

        # ``LossScaleOptimizer`` keeps gradients in float32 to avoid underflow when
        # using float16 tensors. Keras automatically applies dynamic loss scaling
        # for most built-in optimizers, but wrapping here guarantees the behaviour
        # for custom optimizers as well.
        if not isinstance(opt, mixed_precision.LossScaleOptimizer):
            opt = mixed_precision.LossScaleOptimizer(opt)
    model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)

    # MLflow / bookkeeping
    mlflow_enabled = bool(use_mlflow)
    run = None
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if mlflow_enabled:
        init_mlflow(train_cfg.exp_name)
        run = start_run(run_name=f"{model_name}")
        mlflow_run_id = getattr(getattr(run, "info", None), "run_id", None)
        if mlflow_run_id:
            run_id = mlflow_run_id

    ckpt_dir = os.path.join(ckpt_root, run_id) if keep_artifacts else None
    if keep_artifacts and ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    mlf_logger = None
    if mlflow_enabled:
        max_train_samples = getattr(train_cfg, "max_train_samples", None)
        max_val_samples = getattr(train_cfg, "max_val_samples", None)
        model_params = {f"model__{k}": v for k, v in model_kwargs.items()}
        params = {
            "arch": model_name,
            "model": model_name,
            "height": data_cfg.height,
            "width": data_cfg.width,
            "batch_size": data_cfg.batch_size,
            "lr": train_cfg.lr,
            "epochs": train_cfg.epochs,
            "optimizer": train_cfg.optimizer,
            "momentum": train_cfg.momentum,
            "weight_decay": train_cfg.weight_decay,
            "lr_scheduler": lr_schedule_name,
            "poly_power": train_cfg.poly_power,
            "lr_schedule_choice": schedule_choice,
            "warmup_epochs": train_cfg.warmup_epochs,
            "min_lr_ratio": train_cfg.min_lr_ratio,
            "cosine_cycles": train_cfg.cosine_cycles,
            "aug": vars(aug_cfg),
            "ignore_index": data_cfg.ignore_index,
            "loss": loss_choice,
            "precision_policy": current_policy,
        }
        params.update(model_params)
        mlf_logger = KerasMlflowLogger(
            params,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
        )

    monitor_metric = "val_masked_mIoU"
    if loss_choice in {"dice", "ce+dice"}:
        monitor_metric = "val_dice_coef"

    callbacks: list[keras.callbacks.Callback] = []
    if keep_artifacts and ckpt_dir is not None:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    ckpt_dir,
                    f"{model_name}.{monitor_metric}.{{epoch:03d}}-{{{monitor_metric}:.4f}}.keras",
                ),
                monitor=monitor_metric,
                mode="max",
                save_best_only=True,
            )
        )
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            mode="max",
            patience=train_cfg.early_stop_patience,
            restore_best_weights=True,
        )
    )
    callbacks.append(keras.callbacks.TerminateOnNaN())
    csv_path = None
    if keep_artifacts:
        csv_path = os.path.join(train_cfg.output_dir, "train_log.csv")
        callbacks.append(keras.callbacks.CSVLogger(csv_path))
    if mlflow_enabled and mlf_logger is not None:
        callbacks.append(mlf_logger)

    # --- training loop & bookkeeping -----------------------------------------------------
    final_path = os.path.join(train_cfg.output_dir, f"{model_name}_final.keras") if keep_artifacts else None
    best_export = None
    hist = None
    restored_metrics: dict[str, float] = {}

    try:
        hist = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=train_cfg.epochs,
            callbacks=callbacks,
            verbose=1,
        )

        # evaluate restored weights (EarlyStopping may have reloaded best checkpoint)
        restored_metrics = model.evaluate(val_ds, return_dict=True)
        print("Restored weights evaluation:")
        for name, value in restored_metrics.items():
            print(f"  {name}: {value:.6f}")
            if mlflow_enabled:
                mlflow.log_metric(f"restored_{name}", float(value))

        hist_len = len(hist.history.get("loss", [])) if hist is not None else 0
        if hist is not None and hist_len:
            for name, value in restored_metrics.items():
                float_value = float(value)
                if mlflow_enabled:
                    mlflow.log_metric(f"val_{name}", float_value, step=hist_len)

                val_key = f"val_{name}"
                if val_key in hist.history and hist.history[val_key]:
                    hist.history[val_key][-1] = float_value
                elif val_key in hist.history:
                    hist.history[val_key] = [float_value]
                else:
                    hist.history[val_key] = [float("nan")] * (hist_len - 1) + [float_value]

        if keep_artifacts and csv_path and hist is not None and hist_len and os.path.exists(csv_path):
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

        if keep_artifacts and final_path:
            model.save(final_path)
            print(f"saved -> {final_path}")

            best_path, best_score = _best_ckpt(ckpt_dir, monitor_metric) if ckpt_dir else (None, -1.0)
            if best_path:
                best_export = os.path.join(train_cfg.output_dir, f"{model_name}_best.keras")
                shutil.copy2(best_path, best_export)
                print(
                    f"best -> {os.path.basename(best_path)} ({monitor_metric}={best_score:.4f}) | exported: {best_export}"
                )

            # log artifacts
            if csv_path and os.path.exists(csv_path) and mlflow_enabled:
                mlflow.log_artifact(os.path.join(train_cfg.output_dir, "train_log.csv"), artifact_path="logs")
            if mlflow_enabled:
                if os.path.exists(final_path):
                    mlflow.log_artifact(final_path, artifact_path="models")
                if best_export and os.path.exists(best_export):
                    mlflow.log_artifact(best_export, artifact_path="models")
                    try:
                        bm = keras.models.load_model(best_export, compile=False)
                        mlflow.keras.log_model(bm, artifact_path="best_model")
                        mlflow.log_metric(f"best_{monitor_metric}", float(best_score))
                    except Exception as e:
                        print("mlflow keras flavor failed:", e)

            if ckpt_dir and os.path.isdir(ckpt_dir):
                shutil.rmtree(ckpt_dir, ignore_errors=True)

    finally:
        if cleanup_after:
            # Explicitly release dataset and model references to curb memory usage in Optuna trials
            del train_ds_with_weights, val_ds_with_weights, train_ds, val_ds, model
            keras.backend.clear_session()
            gc.collect()
            if not keep_artifacts:
                shutil.rmtree(train_cfg.output_dir, ignore_errors=True)

    if mlflow_enabled:
        mlflow.end_run()
        print("MLflow run ended.")

    return restored_metrics

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
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--optimizer", default="adamw", choices=["adam","adamw","sgd"])
    p.add_argument("--momentum", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument(
        "--poly_power",
        type=float,
        default=None,
        help=(
            "Polynomial decay power. When provided it overrides --lr_schedule."
        ),
    )
    p.add_argument(
        "--lr_schedule",
        default="cosine_warmup",
        choices=["constant", "cosine_warmup"],
        help=(
            "Learning-rate schedule to apply when --poly_power is not set. "
            "'cosine_warmup' enables a short warm-up followed by cosine decay, "
            "while 'constant' keeps the base learning rate for the full training."
        ),
    )
    p.add_argument(
        "--warmup_epochs",
        type=float,
        default=5.0,
        help="Number of epochs used for linear LR warm-up when cosine scheduling is active.",
    )
    p.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.05,
        help="Final learning rate expressed as a fraction of the base LR for cosine scheduling.",
    )
    p.add_argument(
        "--cosine_cycles",
        type=float,
        default=1.0,
        help="Number of cosine cycles to run after the warm-up phase (1.0 corresponds to a single decay).",
    )
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
    default_aug = DEFAULT_AUGMENT_CONFIG
    p.add_argument("--aug_enabled", type=int, default=int(default_aug.enabled))
    p.add_argument("--flip_prob", type=float, default=default_aug.horizontal_flip_prob)
    p.add_argument("--rrc_scale_min", type=float, default=default_aug.random_resized_crop_scale[0])
    p.add_argument("--rrc_scale_max", type=float, default=default_aug.random_resized_crop_scale[1])
    p.add_argument("--rrc_ratio_min", type=float, default=default_aug.random_resized_crop_ratio[0])
    p.add_argument("--rrc_ratio_max", type=float, default=default_aug.random_resized_crop_ratio[1])
    p.add_argument(
        "--rrc_lock_ratio",
        type=int,
        default=int(default_aug.lock_random_resized_crop_ratio),
        help=(
            "When set to 1 (default), force RandomResizedCrop to keep the dataset aspect ratio. "
            "Disable by passing 0 to honour the configured ratio window and optional jitter."
        ),
    )
    p.add_argument("--shift_prob", type=float, default=default_aug.shift_scale_rotate_prob)
    p.add_argument("--shift_limit", type=float, default=default_aug.shift_limit)
    p.add_argument("--scale_limit", type=float, default=default_aug.scale_limit)
    p.add_argument("--rotate_limit", type=float, default=default_aug.rotate_limit)
    p.add_argument("--jitter_brightness", type=float, default=default_aug.color_jitter_brightness)
    p.add_argument("--jitter_contrast", type=float, default=default_aug.color_jitter_contrast)
    p.add_argument("--jitter_saturation", type=float, default=default_aug.color_jitter_saturation)
    p.add_argument("--jitter_hue", type=float, default=default_aug.color_jitter_hue)
    p.add_argument("--blur_prob", type=float, default=default_aug.gaussian_blur_prob)
    p.add_argument("--blur_kernel_min", type=int, default=default_aug.gaussian_blur_kernel[0])
    p.add_argument("--blur_kernel_max", type=int, default=default_aug.gaussian_blur_kernel[1])
    p.add_argument("--noise_prob", type=float, default=default_aug.gauss_noise_prob)
    p.add_argument("--noise_var_min", type=float, default=default_aug.gauss_noise_var_limit[0])
    p.add_argument("--noise_var_max", type=float, default=default_aug.gauss_noise_var_limit[1])
    p.add_argument("--grid_prob", type=float, default=default_aug.grid_dropout_prob)
    p.add_argument("--grid_ratio", type=float, default=default_aug.grid_dropout_ratio)
    p.add_argument("--grid_unit", type=int, default=default_aug.grid_dropout_unit_size)
    p.add_argument("--aspp_dropout", type=float, default=0.5,
                   help="Dropout rate applied in the DeepLab ASPP head")

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
    poly_power = args.poly_power
    train_cfg = TrainConfig(
        lr=args.lr,
        epochs=args.epochs,
        optimizer=args.optimizer,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        poly_power=poly_power,
        lr_schedule=args.lr_schedule,
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=args.min_lr_ratio,
        cosine_cycles=args.cosine_cycles,
        arch=arch,
        loss=loss_choice,
        deterministic_ops=bool(args.deterministic_ops),
        precision_policy=args.precision_policy,
    )
    aug_cfg = AugmentConfig(
        enabled=bool(args.aug_enabled),
        horizontal_flip_prob=args.flip_prob,
        random_resized_crop_scale=(args.rrc_scale_min, args.rrc_scale_max),
        random_resized_crop_ratio=(args.rrc_ratio_min, args.rrc_ratio_max),
        lock_random_resized_crop_ratio=bool(args.rrc_lock_ratio),
        shift_scale_rotate_prob=args.shift_prob,
        shift_limit=args.shift_limit,
        scale_limit=args.scale_limit,
        rotate_limit=args.rotate_limit,
        color_jitter_brightness=args.jitter_brightness,
        color_jitter_contrast=args.jitter_contrast,
        color_jitter_saturation=args.jitter_saturation,
        color_jitter_hue=args.jitter_hue,
        gaussian_blur_prob=args.blur_prob,
        gaussian_blur_kernel=(args.blur_kernel_min, args.blur_kernel_max),
        gauss_noise_prob=args.noise_prob,
        gauss_noise_var_limit=(args.noise_var_min, args.noise_var_max),
        grid_dropout_prob=args.grid_prob,
        grid_dropout_ratio=args.grid_ratio,
        grid_dropout_unit_size=args.grid_unit,
    )
    model_kwargs: dict[str, Any] = {}
    if arch == "deeplab_resnet50":
        model_kwargs["aspp_dropout"] = args.aspp_dropout

    train(arch, data_cfg, train_cfg, aug_cfg, model_kwargs=model_kwargs)
