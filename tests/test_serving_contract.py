import pandas as pd
import pytest

from src.config.paths import PROCESSED_DATA_PATH
from src.serving.inference import load_inference_bundle, predict_dataframe, validate_inference_input


def test_inference_bundle_loads_threshold_and_features():
    bundle = load_inference_bundle()
    assert 0.0 <= bundle.threshold <= 1.0
    assert len(bundle.feature_names) > 0


def test_validate_inference_input_rejects_missing_feature():
    bundle = load_inference_bundle()
    sample = pd.read_parquet(PROCESSED_DATA_PATH).drop(columns="Class").head(3)
    broken = sample.drop(columns=[bundle.feature_names[0]])

    with pytest.raises(ValueError, match="Features obrigatorias ausentes"):
        validate_inference_input(broken, bundle.feature_names)


def test_predict_dataframe_output_contract():
    bundle = load_inference_bundle()
    sample = pd.read_parquet(PROCESSED_DATA_PATH).drop(columns="Class").head(10)
    predictions = predict_dataframe(sample, bundle)

    assert list(predictions.columns) == [
        "fraud_probability",
        "fraud_prediction",
        "decision_threshold",
    ]
    assert len(predictions) == 10
    assert predictions["fraud_probability"].between(0.0, 1.0).all()
    assert set(predictions["fraud_prediction"].unique()).issubset({0, 1})
    assert (predictions["decision_threshold"] == bundle.threshold).all()
