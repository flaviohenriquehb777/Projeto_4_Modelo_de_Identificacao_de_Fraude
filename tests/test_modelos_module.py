import pytest
import sys
import os
from pathlib import Path

# Configuração para sua estrutura REAL
PROJECT_ROOT = Path(r"C:\Users\flavi\Documents\GitHub\Projeto_4_Modelo_de_Identificacao_de_Fraude")

def test_config_files_exist():
    """Testa se os arquivos em src/config/ existem"""
    config_files = [
        PROJECT_ROOT / "src" / "config" / "prepare_data.py",
        PROJECT_ROOT / "src" / "config" / "train_model.py"
    ]
    
    for file_path in config_files:
        assert file_path.exists(), f"Arquivo não encontrado: {file_path}"
        assert file_path.stat().st_size > 0, f"Arquivo vazio: {file_path}"
        print(f"✅ {file_path.name} encontrado!")

def test_src_structure():
    """Testa a estrutura completa da pasta src/"""
    src_path = PROJECT_ROOT / "src"
    config_path = src_path / "config"
    
    assert src_path.exists(), "Pasta src/ não existe"
    assert config_path.exists(), "Pasta src/config/ não existe"
    
    # Verifica arquivos em config/
    config_files = list(config_path.glob("*.py"))
    assert len(config_files) > 0, "Nenhum arquivo .py encontrado em src/config/"
    
    print("✅ Estrutura src/config/ está correta!")
    for file in config_files:
        print(f"   → {file.name}")

def test_python_imports():
    """Testa imports básicos de Python"""
    try:
        import pandas
        import sklearn
        import xgboost
        print("✅ Todos os imports básicos funcionam!")
        assert True
    except ImportError as e:
        pytest.fail(f"Falha em import básico: {e}")