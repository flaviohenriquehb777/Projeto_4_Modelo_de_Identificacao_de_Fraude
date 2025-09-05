import pytest
import sys
import os
from pathlib import Path

# SOLUÇÃO: Usar caminho relativo em vez de absoluto do Windows
PROJECT_ROOT = Path(__file__).parent.parent  # ← CORREÇÃO AQUI!

def test_config_files_exist():
    """Testa se os arquivos em src/config/ existem"""
    config_files = [
        PROJECT_ROOT / "src" / "config" / "prepare_data.py",
        PROJECT_ROOT / "src" / "config" / "train_model.py"
    ]
    
    for file_path in config_files:
        assert file_path.exists(), f"Arquivo não encontrado: {file_path}"
        print(f"✅ {file_path.name} encontrado!")

def test_src_structure():
    """Testa a estrutura completa da pasta src/"""
    src_path = PROJECT_ROOT / "src"
    config_path = src_path / "config"
    
    assert src_path.exists(), "Pasta src/ não existe"
    assert config_path.exists(), "Pasta src/config/ não existe"
    
    print("✅ Estrutura src/config/ está correta!")

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