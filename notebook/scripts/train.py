# /scripts/train.py

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # masque INFO & WARNING C++
# Désactive quelques optimisations bruyantes
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_XLA_FLAGS", "--xla_cpu_enable_xla=false")

import os, re, shutil, argparse, csv
import tensorflow as tf
from tensorflow import keras
from .config import DataConfig, TrainConfig, AugmentConfig
from .data import build_dataset
from .models import build_model
from .metrics import MaskedMeanIoU
from .mlflow_utils import init_mlflow, start_run, KerasMlflowLogger
import mlflow, mlflow.keras

CKPT_RE = re.compile(r"weights\.(\d+)-([0-9]*\.[0-9]+)\.keras$")

def _best_ckpt(path: str):
    if not os.path.isdir(path): return None, -1.0
    best, score = None, -1.0
    for f in os.listdir(path):
        m = CKPT_RE.match(f)
        if m:
            s = float(m.group(2))
            if s > score: best, score = os.path.join(path, f), s
    return best, score

def build_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "adam": return keras.optimizers.Adam(lr)
    if name == "adamw": return keras.optimizers.AdamW(lr)
    if name == "sgd": return keras.optimizers.SGD(lr, momentum=0.9, nesterov=True)
    raise ValueError(f"Unknown optimizer: {name}")

def train(model_name: str = "deeplab_resnet50",
          data_cfg: DataConfig = DataConfig(),
          train_cfg: TrainConfig = TrainConfig(),
          aug_cfg: AugmentConfig = AugmentConfig()):
    # reproducibility
    tf.keras.utils.set_random_seed(data_cfg.seed)
    tf.config.experimental.enable_op_determinism()
    # désactive JIT/XLA et layout optimizer pour éviter les messages "layout failed"
    tf.config.optimizer.set_jit(False)
    tf.config.optimizer.set_experimental_options({
        "layout_optimizer": False,
    })

    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy("float32")

    print(f"TF {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")
    os.makedirs(train_cfg.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(train_cfg.output_dir, "checkpoints"); os.makedirs(ckpt_dir, exist_ok=True)

    # data
    train_ds = build_dataset(data_cfg, aug_cfg, split="train", training=True)
    val_ds   = build_dataset(data_cfg, aug_cfg.__class__(enabled=False), split="val", training=False)  # no aug at val
    xb, yb, wb = next(iter(train_ds))
    print(f"probe -> x:{xb.shape} {xb.dtype} | y:{yb.shape} {yb.dtype} | w:{wb.shape} {wb.dtype}")

    # model
    input_shape = (data_cfg.height, data_cfg.width, 3)
    model = build_model(model_name, input_shape, data_cfg.num_classes)

    # compile
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [keras.metrics.SparseCategoricalAccuracy(name="pix_acc"),
               MaskedMeanIoU(num_classes=data_cfg.num_classes, ignore_index=data_cfg.ignore_index)]
    opt = build_optimizer(train_cfg.optimizer, train_cfg.lr)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)

    # MLflow
    init_mlflow(train_cfg.exp_name)
    run = start_run(run_name=f"{model_name}")
    max_train_samples = getattr(train_cfg, "max_train_samples", None)
    max_val_samples = getattr(train_cfg, "max_val_samples", None)
    mlf_logger = KerasMlflowLogger({
        "model": model_name, "height": data_cfg.height, "width": data_cfg.width,
        "batch_size": data_cfg.batch_size, "lr": train_cfg.lr, "epochs": train_cfg.epochs,
        "optimizer": train_cfg.optimizer, "aug": vars(aug_cfg), "ignore_index": data_cfg.ignore_index
    }, max_train_samples=max_train_samples, max_val_samples=max_val_samples)

    callbacks = [keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(ckpt_dir, "weights.{epoch:03d}-{val_masked_mIoU:.4f}.keras"),
        monitor="val_masked_mIoU", mode="max", save_best_only=True
    ), keras.callbacks.EarlyStopping(monitor="val_masked_mIoU", mode="max",
                                     patience=train_cfg.early_stop_patience, restore_best_weights=True),
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.CSVLogger(os.path.join(train_cfg.output_dir, "train_log.csv")), mlf_logger]

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
            mlflow.log_metric(name, float_value, step=hist_len)
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
    best_path, best_score = _best_ckpt(ckpt_dir)
    best_export = None
    if best_path:
        best_export = os.path.join(train_cfg.output_dir, f"{model_name}_best.keras")
        shutil.copy2(best_path, best_export)
        print(f"best -> {os.path.basename(best_path)} (mIoU={best_score:.4f}) | exported: {best_export}")

    # log artifacts
    mlflow.log_artifact(os.path.join(train_cfg.output_dir, "train_log.csv"), artifact_path="logs")
    if os.path.exists(final_path): mlflow.log_artifact(final_path, artifact_path="models")
    if best_export and os.path.exists(best_export):
        mlflow.log_artifact(best_export, artifact_path="models")
        try:
            bm = keras.models.load_model(best_export, compile=False)
            mlflow.keras.log_model(bm, artifact_path="best_model")
            mlflow.log_metric("best_val_masked_mIoU", float(best_score))
        except Exception as e:
            print("mlflow keras flavor failed:", e)

    mlflow.end_run()
    print("MLflow run ended.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deeplab_resnet50",
                   choices=["deeplab_resnet50","deeplab_mobilenetv2","unet_small","fast_scnn"])
    p.add_argument("--data_root", default="../data")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_val_samples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--optimizer", default="adam", choices=["adam","adamw","sgd"])

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

    data_cfg = DataConfig(
        data_root=args.data_root, height=args.height, width=args.width,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples
    )
    train_cfg = TrainConfig(lr=args.lr, epochs=args.epochs, optimizer=args.optimizer)
    aug_cfg = AugmentConfig(
        enabled=bool(args.aug_enabled), hflip=bool(args.hflip), vflip=bool(args.vflip),
        random_rotate_deg=args.rotate, random_scale_min=args.scale_min, random_scale_max=args.scale_max,
        random_crop=bool(args.random_crop), brightness_delta=args.brightness, contrast_delta=args.contrast,
        saturation_delta=args.saturation, hue_delta=args.hue, gaussian_noise_std=args.noise_std
    )
    train(args.model, data_cfg, train_cfg, aug_cfg)
