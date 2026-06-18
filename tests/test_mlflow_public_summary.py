import json

from src.config.paths import MLFLOW_PUBLIC_SUMMARY_PATH


def test_mlflow_public_summary_exists():
    assert MLFLOW_PUBLIC_SUMMARY_PATH.exists()


def test_mlflow_public_summary_has_expected_public_evidence():
    with MLFLOW_PUBLIC_SUMMARY_PATH.open("r", encoding="utf-8") as file:
        summary = json.load(file)

    assert summary["dagshub"]["repository_visibility"] == "public"
    assert summary["dagshub"]["experiments_url"].endswith("/experiments")
    assert summary["dagshub"]["models_url"].endswith("/models")
    assert summary["mlflow_experiment"]["total_public_runs"] >= 7
    assert summary["mlflow_experiment"]["total_public_models"] >= 1
    assert summary["historical_best_model"]["mean_precision"] >= 0.9227
    assert summary["historical_best_model"]["mean_recall"] >= 0.8101
