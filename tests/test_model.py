import pytest
import joblib
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def test_model_file_exists():
    """Testa se o modelo treinado existe"""
    assert os.path.exists('models/model.pkl'), "Modelo não encontrado - execute 'dvc pull' ou treine o modelo"

def test_model_inference():
    """Testa se o modelo faz inferências corretamente"""
    try:
        # Carrega modelo
        model = joblib.load('models/model.pkl')
        
        # Carrega dados
        df = pd.read_parquet('dados/credicard_tratado.parquet')
        X = df.drop('Class', axis=1)
        
        # Testa previsões em um subset
        sample_data = X.head(5)
        predictions = model.predict(sample_data)
        
        # Validações
        assert len(predictions) == 5, "Número incorreto de previsões"
        assert predictions.dtype in ['int64', 'int32', 'float64'], "Previsões devem ser numéricas"
        assert set(predictions).issubset({0, 1}), "Previsões devem ser 0 ou 1 (classificação binária)"
        
    except Exception as e:
        pytest.fail(f"Falha no teste de inferência do modelo: {str(e)}")

def test_model_metrics_exist():
    """Testa se o arquivo de métricas existe"""
    assert os.path.exists('metrics.json'), "Arquivo de métricas não encontrado"