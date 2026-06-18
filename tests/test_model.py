import pytest
import joblib
import pandas as pd
import os
import json
from src.config.paths import MODEL_METADATA_PATH, PIPELINE_MODEL_PATH, PROCESSED_DATA_PATH

def test_model_file_exists():
    """Testa se o modelo treinado existe"""
    assert PIPELINE_MODEL_PATH.exists(), "Modelo não encontrado - execute 'dvc pull' ou treine o modelo"

def test_model_inference():
    """Testa se o modelo faz inferências corretamente"""
    try:
        # Carrega modelo
        model = joblib.load(PIPELINE_MODEL_PATH)
        
        # Carrega dados
        df = pd.read_parquet(PROCESSED_DATA_PATH)
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


def test_model_metadata_exists():
    """Testa se os metadados operacionais do modelo existem"""
    assert MODEL_METADATA_PATH.exists(), "Metadados do modelo não encontrados"


def test_model_metadata_threshold_is_valid():
    """Testa se o threshold calibrado salvo e as metricas operacionais sao validos"""
    try:
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        threshold = metadata["decision_threshold"]
        calibrated_metrics = metadata["out_of_fold_calibrated_metrics"]

        assert 0.0 <= threshold <= 1.0, "Threshold deve estar entre 0 e 1"
        assert 0.0 <= calibrated_metrics["precision"] <= 1.0, "Precisao invalida"
        assert 0.0 <= calibrated_metrics["recall"] <= 1.0, "Recall invalido"
        assert 0.0 <= calibrated_metrics["accuracy"] <= 1.0, "Accuracy invalida"
        assert metadata["acceptance"]["passed"] is True, "Modelo deveria atender aos criterios de aceite"
    except Exception as e:
        pytest.fail(f"Falha ao validar metadados do modelo: {str(e)}")
