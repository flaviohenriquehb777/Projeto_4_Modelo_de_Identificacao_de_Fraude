import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import json

# O DVC passa os caminhos dos arquivos como argumentos
data_path = "dados/credicard_tratado.parquet"
model_path = "models/model.pkl"

# Carrega os dados
df = pd.read_parquet(data_path)

# Prepara os dados (divisão entre features e target)
X = df.drop("Class", axis=1)
y = df["Class"]

# Divide os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treina o modelo (XGBoost)
model = XGBClassifier()
model.fit(X_train, y_train)

# Faz as previsões e salva o modelo
y_pred = model.predict(X_test)
joblib.dump(model, model_path)

# Calcula as métricas
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
}

# Salva as métricas em um arquivo JSON para o DVC rastrear
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print("Modelo treinado e métricas salvas.")