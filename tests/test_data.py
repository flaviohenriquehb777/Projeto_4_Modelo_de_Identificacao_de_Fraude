import pandas as pd
import pytest
from src.config.paths import PROCESSED_DATA_PATH

def test_data_file_exists():
    """Testa se o arquivo de dados existe após DVC pull"""
    assert PROCESSED_DATA_PATH.exists(), "Arquivo de dados não encontrado - execute 'dvc pull' primeiro"

def test_data_quality():
    """Testa a qualidade básica dos dados processados"""
    try:
        df = pd.read_parquet(PROCESSED_DATA_PATH)
        
        # Testes de qualidade
        assert not df.empty, "DataFrame está vazio"
        assert 'Class' in df.columns, "Coluna target 'Class' não encontrada"
        assert df['Class'].dtype in ['int64', 'int32', 'float64'], "Coluna Class deve ser numérica"
        assert set(df['Class'].unique()).issubset({0, 1}), "Coluna Class deve conter apenas 0 e 1"
        assert df.isnull().sum().sum() == 0, "Existem valores nulos no dataset"
        
        # Verifica se há dados suficientes
        assert len(df) > 1000, "Dataset muito pequeno"

        assert 'Time' in df.columns, "Coluna Time nao encontrada"
        assert 'Amount' in df.columns, "Coluna Amount nao encontrada"
        
    except Exception as e:
        pytest.fail(f"Falha ao carregar ou validar dados: {str(e)}")
