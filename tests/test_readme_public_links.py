from src.config.paths import PROJECT_ROOT


def test_readme_does_not_use_local_file_links():
    content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "file:///" not in content


def test_readme_exposes_public_dagshub_links():
    content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://dagshub.com/flaviohenriquehb777/Modelo_de_Deteccao_de_Fraude-Financeiro/experiments" in content
    assert "https://dagshub.com/flaviohenriquehb777/Modelo_de_Deteccao_de_Fraude-Financeiro/models" in content
