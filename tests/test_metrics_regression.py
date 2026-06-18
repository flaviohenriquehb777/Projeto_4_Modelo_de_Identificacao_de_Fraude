import json

import pytest
import yaml

from src.config.paths import PROJECT_ROOT


def load_metrics() -> dict:
    with (PROJECT_ROOT / "metrics.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def load_params() -> dict:
    with (PROJECT_ROOT / "params.yaml").open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def test_acceptance_gate_passed():
    metrics = load_metrics()
    assert metrics["acceptance"]["passed"] is True


def test_metrics_meet_acceptance_thresholds():
    params = load_params()
    acceptance = params["train"]["acceptance"]
    metrics = load_metrics()

    cv_metrics = metrics["evaluation"]["cross_validation_mean_metrics"]
    calibrated_metrics = metrics["evaluation"]["threshold_calibration"]["selected"]["metrics"]
    weighted_gap = metrics["evaluation"]["threshold_calibration"]["selected"]["business_alignment"]["weighted_gap"]

    assert cv_metrics["precision"] >= acceptance["min_cv_precision"]
    assert cv_metrics["recall"] >= acceptance["min_cv_recall"]
    assert calibrated_metrics["precision"] >= acceptance["min_calibrated_precision"]
    assert calibrated_metrics["recall"] >= acceptance["min_calibrated_recall"]
    assert weighted_gap <= acceptance["max_business_weighted_gap"]


def test_threshold_within_search_space():
    params = load_params()
    evaluation = params["train"]["evaluation"]
    metrics = load_metrics()

    threshold = metrics["evaluation"]["threshold_calibration"]["selected"]["threshold"]
    assert evaluation["threshold_search_start"] <= threshold <= evaluation["threshold_search_end"]
    assert evaluation["threshold_search_steps"] >= 2

