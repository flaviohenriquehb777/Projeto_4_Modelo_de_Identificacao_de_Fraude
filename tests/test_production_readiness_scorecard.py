import json

from src.config.paths import PROJECT_ROOT, PRODUCTION_READINESS_SCORECARD_PATH


def test_production_readiness_scorecard_exists():
    assert PRODUCTION_READINESS_SCORECARD_PATH.exists()


def test_production_readiness_scorecard_core_flags():
    with PRODUCTION_READINESS_SCORECARD_PATH.open("r", encoding="utf-8") as file:
        scorecard = json.load(file)

    assert scorecard["repository"]["dvc_clone_zero_friction_branch"] is True
    assert scorecard["repository"]["dagshub_public_repository"] is True
    assert scorecard["model_readiness"]["acceptance_passed"] is True
    assert scorecard["model_readiness"]["decision_threshold_versioned"] is True
    assert scorecard["data_readiness"]["quality_report_passed"] is True
    assert scorecard["public_evidence"]["public_experiments"] >= 7


def test_phase9_docs_exist():
    docs_dir = PROJECT_ROOT / "docs"
    assert (docs_dir / "evaluator-guide.md").exists()
    assert (docs_dir / "environment-setup.md").exists()
    assert (docs_dir / "release-update-draft.md").exists()
