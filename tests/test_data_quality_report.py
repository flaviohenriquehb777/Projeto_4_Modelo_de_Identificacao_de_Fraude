import json

from src.config.paths import DATA_QUALITY_REPORT_PATH


def test_data_quality_report_exists():
    assert DATA_QUALITY_REPORT_PATH.exists()


def test_data_quality_report_has_expected_fields():
    with DATA_QUALITY_REPORT_PATH.open("r", encoding="utf-8") as file:
        report = json.load(file)

    assert report["status"] == "passed"
    assert report["raw_dataset"]["row_count"] >= 1000
    assert report["raw_dataset"]["null_count"] == 0
    assert 0.0 < report["raw_dataset"]["positive_rate"] < 1.0

