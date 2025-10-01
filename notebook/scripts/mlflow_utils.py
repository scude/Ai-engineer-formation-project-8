# /scripts/mlflow_utils.py
import os, datetime
from typing import Dict, Optional
import mlflow

def init_mlflow(exp_name: str, base_dir: str = "artifacts/mlruns"):
    os.makedirs(base_dir, exist_ok=True)
    mlflow.set_tracking_uri("file://" + os.path.abspath(base_dir))
    mlflow.set_experiment(exp_name)

def start_run(run_name: Optional[str] = None):
    if run_name is None:
        run_name = f"run-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return mlflow.start_run(run_name=run_name)

class KerasMlflowLogger:
    def __init__(self, params: Dict):
        self.params = params

    def on_train_begin(self):
        mlflow.log_params(self.params)

    def on_epoch_end(self, epoch: int, logs: Dict):
        if not logs: return
        mlflow.log_metrics({k: float(v) for k,v in logs.items() if isinstance(v, (int,float))}, step=epoch+1)
