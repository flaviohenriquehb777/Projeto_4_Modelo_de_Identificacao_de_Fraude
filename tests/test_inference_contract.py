import json

import joblib
import pandas as pd

from src.config.paths import MODEL_METADATA_PATH, PIPELINE_MODEL_PATH, PROCESSED_DATA_PATH


def load_threshold() -> float:
    with MODEL_METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    return float(metadata["decision_threshold"])


def test_model_supports_predict_proba():
    model = joblib.load(PIPELINE_MODEL_PATH)
    assert hasattr(model, "predict_proba")


def test_probability_output_contract():
    model = joblib.load(PIPELINE_MODEL_PATH)
    df = pd.read_parquet(PROCESSED_DATA_PATH).head(32)
    X = df.drop("Class", axis=1)

    probabilities = model.predict_proba(X)[:, 1]
    assert len(probabilities) == len(X)
    assert (probabilities >= 0.0).all()
    assert (probabilities <= 1.0).all()


def test_thresholded_predictions_are_binary():
    model = joblib.load(PIPELINE_MODEL_PATH)
    threshold = load_threshold()
    df = pd.read_parquet(PROCESSED_DATA_PATH).head(64)
    X = df.drop("Class", axis=1)

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    assert set(predictions.tolist()).issubset({0, 1})

