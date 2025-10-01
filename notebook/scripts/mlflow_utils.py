# /scripts/mlflow_utils.py
import os, datetime
from typing import Dict, Optional

import mlflow
from tensorflow import keras

def init_mlflow(exp_name: str, base_dir: str = "artifacts/mlruns"):
    os.makedirs(base_dir, exist_ok=True)
    mlflow.set_tracking_uri("file://" + os.path.abspath(base_dir))
    mlflow.set_experiment(exp_name)

def start_run(run_name: Optional[str] = None):
    if run_name is None:
        run_name = f"run-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return mlflow.start_run(run_name=run_name)

class KerasMlflowLogger(keras.callbacks.Callback):
    def __init__(self, params: Dict,
                 max_train_samples: Optional[int] = None,
                 max_val_samples: Optional[int] = None):
        super().__init__()
        self.params = params
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples

    def on_train_begin(self, logs=None):
        mlflow.log_params(self.params)
        metric_payload = {}
        if self.max_train_samples is not None:
            metric_payload["max_train_samples"] = float(self.max_train_samples)
        if self.max_val_samples is not None:
            metric_payload["max_val_samples"] = float(self.max_val_samples)
        if metric_payload:
            mlflow.log_metrics(metric_payload, step=0)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        if not logs:
            return
        metrics = {}
        for key, value in logs.items():
            try:
                metrics[key] = float(value)
            except (TypeError, ValueError):
                continue
        if metrics:
            mlflow.log_metrics(metrics, step=epoch + 1)
