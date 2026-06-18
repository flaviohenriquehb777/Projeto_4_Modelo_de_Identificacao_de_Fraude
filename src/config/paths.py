from pathlib import Path


def get_project_root() -> Path:
    """Localiza a raiz do projeto a partir deste arquivo."""
    current = Path(__file__).resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / ".git").exists() and (candidate / "src").exists():
            return candidate
    raise RuntimeError("Raiz do projeto não encontrada.")


PROJECT_ROOT = get_project_root()

DATA_DIR = PROJECT_ROOT / "dados"
RAW_DATA_PATH = DATA_DIR / "credicard.csv.gz"
PROCESSED_DATA_PATH = DATA_DIR / "credicard_tratado.parquet"

MODELS_DIR = PROJECT_ROOT / "models"
PIPELINE_MODEL_PATH = MODELS_DIR / "model.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
HISTORICAL_MODEL_REFERENCE_PATH = MODELS_DIR / "best_model_xgboost.reference.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_QUALITY_REPORT_PATH = REPORTS_DIR / "data_quality_report.json"
MLFLOW_PUBLIC_SUMMARY_PATH = REPORTS_DIR / "mlflow_public_registry_summary.json"
PRODUCTION_READINESS_SCORECARD_PATH = REPORTS_DIR / "production_readiness_scorecard.json"
SHAP_EXPLANATIONS_DIR = REPORTS_DIR / "shap_explanations"
SHAP_MANIFEST_PATH = SHAP_EXPLANATIONS_DIR / "shap_manifest.json"
SHAP_REPORT_PATH = REPORTS_DIR / "shap_report.pdf"
