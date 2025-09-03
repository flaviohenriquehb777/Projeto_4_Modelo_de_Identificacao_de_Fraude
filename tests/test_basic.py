import pytest

def test_imports():
    """Testa se todas as dependências podem ser importadas"""
    try:
        import pandas
        import numpy
        import sklearn
        import xgboost
        import mlflow
        import dvc
        import joblib
        print("✅ Todos os imports funcionaram!")
        assert True
    except ImportError as e:
        pytest.fail(f"❌ Falha ao importar: {str(e)}")

def test_python_version():
    """Testa se estamos usando Python 3.7+"""
    import sys
    version_valid = sys.version_info >= (3, 7)
    print(f"✅ Python version: {sys.version}")
    assert version_valid, "Python version must be 3.7 or higher"