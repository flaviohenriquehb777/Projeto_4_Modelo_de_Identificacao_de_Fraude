import json
import sys
from pathlib import Path

import yaml

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from src.config.paths import MLFLOW_PUBLIC_SUMMARY_PATH, PROJECT_ROOT


DAGSHUB_BASE_URL = "https://dagshub.com/flaviohenriquehb777/Modelo_de_Deteccao_de_Fraude-Financeiro"


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def read_metric(metric_path: Path) -> float | None:
    if not metric_path.exists():
        return None
    line = metric_path.read_text(encoding="utf-8").strip().splitlines()[-1]
    parts = line.split()
    return float(parts[1]) if len(parts) >= 2 else None


def read_param(param_path: Path) -> str | None:
    if not param_path.exists():
        return None
    return param_path.read_text(encoding="utf-8").strip()


def build_run_summary(run_dir: Path) -> dict:
    meta = read_yaml(run_dir / "meta.yaml")
    return {
        "run_id": meta["run_id"],
        "run_name": meta["run_name"],
        "status": meta["status"],
        "created_at": meta["start_time"],
        "mean_accuracy": read_metric(run_dir / "metrics" / "mean_accuracy"),
        "mean_precision": read_metric(run_dir / "metrics" / "mean_precision"),
        "mean_recall": read_metric(run_dir / "metrics" / "mean_recall"),
    }


def build_model_summary(model_dir: Path) -> dict:
    meta = read_yaml(model_dir / "meta.yaml")
    return {
        "model_id": meta["model_id"],
        "name": meta["name"],
        "source_run_id": meta["source_run_id"],
        "creation_timestamp": meta["creation_timestamp"],
        "mean_accuracy": read_metric(model_dir / "metrics" / "mean_accuracy"),
        "mean_precision": read_metric(model_dir / "metrics" / "mean_precision"),
        "mean_recall": read_metric(model_dir / "metrics" / "mean_recall"),
        "params": {
            "max_depth": read_param(model_dir / "params" / "max_depth"),
            "n_estimators": read_param(model_dir / "params" / "n_estimators"),
            "learning_rate": read_param(model_dir / "params" / "learning_rate"),
            "random_state": read_param(model_dir / "params" / "random_state"),
            "scale_pos_weight": read_param(model_dir / "params" / "scale_pos_weight"),
        },
    }


def select_best_model(models: list[dict]) -> dict:
    return max(
        models,
        key=lambda model: (
            model["mean_precision"] if model["mean_precision"] is not None else -1.0,
            model["mean_recall"] if model["mean_recall"] is not None else -1.0,
            model["mean_accuracy"] if model["mean_accuracy"] is not None else -1.0,
            1 if model["name"] != "best_model" else 0,
            model["creation_timestamp"],
        ),
    )


def main() -> None:
    mlruns_dir = PROJECT_ROOT / "mlruns"
    experiment_dirs = [path for path in mlruns_dir.iterdir() if path.is_dir() and path.name != ".trash"]
    experiment_dir = next(path for path in experiment_dirs if (path / "meta.yaml").exists())

    experiment_meta = read_yaml(experiment_dir / "meta.yaml")

    run_dirs = [
        path
        for path in experiment_dir.iterdir()
        if path.is_dir() and path.name != "models" and (path / "meta.yaml").exists()
    ]
    runs = sorted((build_run_summary(path) for path in run_dirs), key=lambda run: run["created_at"])

    model_dirs = [
        path for path in (experiment_dir / "models").iterdir() if path.is_dir() and (path / "meta.yaml").exists()
    ]
    models = sorted((build_model_summary(path) for path in model_dirs), key=lambda model: model["creation_timestamp"])
    best_model = select_best_model(models)

    summary = {
        "dagshub": {
            "repository_url": DAGSHUB_BASE_URL,
            "experiments_url": f"{DAGSHUB_BASE_URL}/experiments",
            "models_url": f"{DAGSHUB_BASE_URL}/models",
            "repository_visibility": "public",
        },
        "mlflow_experiment": {
            "experiment_id": experiment_meta["experiment_id"],
            "name": experiment_meta["name"],
            "total_public_runs": len(runs),
            "total_public_models": len(models),
        },
        "historical_best_model": {
            **best_model,
            "selection_rule": "highest_mean_precision_then_mean_recall_then_mean_accuracy_preferring_named_and_latest_public_model",
            "business_baseline_alignment": {
                "target_precision": 0.9227,
                "target_recall": 0.8102,
            },
        },
        "public_runs": runs,
        "public_models": models,
    }

    MLFLOW_PUBLIC_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MLFLOW_PUBLIC_SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Resumo publico de MLflow/DagsHub salvo em {MLFLOW_PUBLIC_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
