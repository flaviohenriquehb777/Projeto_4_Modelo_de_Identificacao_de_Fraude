import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from src.config.paths import MODEL_METADATA_PATH, PIPELINE_MODEL_PATH


@dataclass(frozen=True)
class InferenceBundle:
    model: object
    threshold: float
    feature_names: list[str]


def load_inference_bundle(
    model_path: Path = PIPELINE_MODEL_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
) -> InferenceBundle:
    model = joblib.load(model_path)
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_names = list(getattr(model, "feature_names_in_", []))
    if not feature_names:
        raise ValueError("O modelo nao expoe feature_names_in_.")

    threshold = float(metadata["decision_threshold"])
    return InferenceBundle(model=model, threshold=threshold, feature_names=feature_names)


def validate_inference_input(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    missing_features = [feature for feature in feature_names if feature not in df.columns]
    if missing_features:
        raise ValueError(f"Features obrigatorias ausentes para inferencia: {missing_features}")

    extra_columns = [column for column in df.columns if column not in feature_names]
    ordered_df = df[feature_names].copy()

    return ordered_df.assign(**{"_ignored_columns_count": len(extra_columns)}).drop(
        columns="_ignored_columns_count"
    )


def predict_dataframe(df: pd.DataFrame, bundle: InferenceBundle) -> pd.DataFrame:
    features = validate_inference_input(df, bundle.feature_names)
    probabilities = bundle.model.predict_proba(features)[:, 1]
    predictions = (probabilities >= bundle.threshold).astype(int)

    return pd.DataFrame(
        {
            "fraud_probability": probabilities,
            "fraud_prediction": predictions,
            "decision_threshold": bundle.threshold,
        }
    )

