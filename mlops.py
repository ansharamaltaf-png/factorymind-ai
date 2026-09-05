"""
Stage VIII (MLOps half) - Experiment tracking with MLflow.
Wraps mlflow.start_run/log_param/log_metric/log_artifact. If mlflow is not
installed in the current environment, falls back to a local JSON-lines
logger with an IDENTICAL call signature, so the rest of the app never needs
to know which backend is active -- and no experiment silently goes
unrecorded. Install `mlflow` and run `mlflow ui` to get the real dashboard.
"""

import os
import json
import time

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)
LOCAL_LOG = f"{ARTIFACT_DIR}/mlflow_fallback_log.jsonl"

try:
    import mlflow
    HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False


class ExperimentRun:
    def __init__(self, run_name: str):
        self.run_name = run_name
        self.record = {"run_name": run_name, "timestamp": time.time(),
                        "params": {}, "metrics": {}, "artifacts": []}
        self._mlflow_run = None

    def __enter__(self):
        if HAS_MLFLOW:
            mlflow.set_experiment("ai_factory_intelligence")
            self._mlflow_run = mlflow.start_run(run_name=self.run_name)
        return self

    def log_param(self, key, value):
        self.record["params"][key] = value
        if HAS_MLFLOW:
            mlflow.log_param(key, value)

    def log_metric(self, key, value):
        self.record["metrics"][key] = value

        if HAS_MLFLOW:
            try:
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, float(value))
            except Exception:
                pass

    def log_artifact(self, path):
        self.record["artifacts"].append(path)
        if HAS_MLFLOW:
            mlflow.log_artifact(path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if HAS_MLFLOW:
            mlflow.end_run()
        else:
            with open(LOCAL_LOG, "a") as f:
                f.write(json.dumps(self.record, default=str) + "\n")
        return False


def log_training_run(run_name: str, params: dict, metrics: dict, artifact_paths=None):
    with ExperimentRun(run_name) as run:
        for k, v in params.items():
            run.log_param(k, v)
        for k, v in metrics.items():
            if v is not None:
                run.log_metric(k, v)
        for p in (artifact_paths or []):
            if os.path.exists(p):
                run.log_artifact(p)


def load_local_runs():
    if not os.path.exists(LOCAL_LOG):
        return []
    with open(LOCAL_LOG) as f:
        return [json.loads(line) for line in f if line.strip()]


def backend_name():
    return "MLflow" if HAS_MLFLOW else "Local JSONL fallback (install `mlflow` for full UI)"


if __name__ == "__main__":
    log_training_run("baseline_rf_demo", {"n_estimators": 300, "max_depth": 10},
                      {"f1": 0.63, "roc_auc": 0.96})
    log_training_run("dl_ann_demo", {"epochs": 30, "hidden": "64-32"},
                      {"f1": 0.56, "roc_auc": 0.92})
    log_training_run("dl_ann_demo_v2", {"epochs": 50, "hidden": "128-64"},
                      {"f1": 0.60, "roc_auc": 0.94})
    print("Backend:", backend_name())
    print("Runs logged:", len(load_local_runs()))
