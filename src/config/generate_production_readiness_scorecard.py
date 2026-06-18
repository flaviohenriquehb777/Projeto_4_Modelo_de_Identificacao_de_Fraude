import json
import sys
from pathlib import Path

BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from src.config.paths import (
    DATA_QUALITY_REPORT_PATH,
    MLFLOW_PUBLIC_SUMMARY_PATH,
    MODEL_METADATA_PATH,
    PIPELINE_MODEL_PATH,
    PRODUCTION_READINESS_SCORECARD_PATH,
    PROJECT_ROOT,
)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    metrics = read_json(PROJECT_ROOT / "metrics.json")
    model_metadata = read_json(MODEL_METADATA_PATH)
    data_quality = read_json(DATA_QUALITY_REPORT_PATH)
    mlflow_public = read_json(MLFLOW_PUBLIC_SUMMARY_PATH)

    acceptance = metrics["acceptance"]
    selected_threshold = metrics["evaluation"]["threshold_calibration"]["selected"]

    scorecard = {
        "repository": {
            "project_name": "Modelo_de_Deteccao_de_Fraude-Financeiro",
            "dvc_clone_zero_friction_branch": True,
            "dagshub_public_repository": mlflow_public["dagshub"]["repository_visibility"] == "public",
        },
        "model_readiness": {
            "acceptance_passed": acceptance["passed"],
            "decision_threshold_versioned": 0.0 <= float(model_metadata["decision_threshold"]) <= 1.0,
            "operational_model_present": PIPELINE_MODEL_PATH.exists(),
            "calibrated_precision": selected_threshold["metrics"]["precision"],
            "calibrated_recall": selected_threshold["metrics"]["recall"],
            "alignment_score": selected_threshold["business_alignment"]["alignment_score"],
        },
        "data_readiness": {
            "quality_report_passed": data_quality["status"] == "passed",
            "processed_rows": data_quality["prepared_dataset"]["row_count"],
            "null_count": data_quality["prepared_dataset"]["null_count"],
            "time_scaled_to_unit_interval": (
                data_quality["prepared_dataset"]["time_min"] == 0.0
                and data_quality["prepared_dataset"]["time_max"] == 1.0
            ),
        },
        "public_evidence": {
            "public_experiments": mlflow_public["mlflow_experiment"]["total_public_runs"],
            "public_models": mlflow_public["mlflow_experiment"]["total_public_models"],
            "best_public_model_id": mlflow_public["historical_best_model"]["model_id"],
            "best_public_model_name": mlflow_public["historical_best_model"]["name"],
        },
        "serving_readiness": {
            "cli_entrypoint": "python -m src.serving.cli",
            "threshold_source": str(MODEL_METADATA_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "inference_contract_module": "src/serving/inference.py",
        },
        "known_environment_notes": {
            "clean_venv_recommended": True,
            "warning_sources": [
                "ambiente local Anaconda com numexpr desatualizado",
                "coexistencia local entre PyFPDF e fpdf2 fora do repositório",
            ],
        },
    }

    PRODUCTION_READINESS_SCORECARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PRODUCTION_READINESS_SCORECARD_PATH.open("w", encoding="utf-8") as file:
        json.dump(scorecard, file, indent=2)

    print(f"Scorecard de prontidao salvo em {PRODUCTION_READINESS_SCORECARD_PATH}")


if __name__ == "__main__":
    main()
