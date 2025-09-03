# 🚨 Modelo de Identificação de Fraude em Transações (Credicard - Brasil)

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
[![DVC Managed](https://img.shields.io/badge/DVC-Managed-blue)]()
[![CI Tests](https://github.com/flaviohenriquehb777/Projeto_4_Modelo_de_Identificacao_de_Fraude/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/flaviohenriquehb777/Projeto_4_Modelo_de_Identificacao_de_Fraude/actions/workflows/ci-tests.yml)
[![MLflow Tracking](https://img.shields.io/badge/MLflow-Tracking-orange)]()
[![DagsHub Repository](https://img.shields.io/badge/DagsHub-Repository-purple)](https://dagshub.com/flaviohenriquehb777/Projeto_4_Modelo_de_Identificacao_de_Fraude)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-blueviolet)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-success)

Utilizando algoritmos de Machine Learning e práticas de MLOps para criar um modelo robusto de identificação de transações fraudulentas.

---

## 📋 Sumário

- [Descrição do Projeto](#descrição-do-projeto)
- [Contexto dos Dados](#contexto-dos-dados)
- [Algoritmos e Técnicas Utilizadas](#algoritmos-e-técnicas-utilizadas)
- [MLOps e Infraestrutura Profissional](#mlops-e-infraestrutura-profissional)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Processo de Desenvolvimento](#processo-de-desenvolvimento)
- [Instalação e Uso](#instalação-e-uso)
- [Diferenciais do Projeto](#diferenciais-do-projeto)
- [Resultados e Conclusão](#resultados-e-conclusão)
- [Interpretação das Métricas](#interpretação-das-métricas)
- [Roadmap](#roadmap)
- [Licença](#licença)
- [Contato](#contato)

---

## 📊 Descrição do Projeto

Este projeto tem como objetivo desenvolver um modelo de Machine Learning capaz de identificar transações financeiras fraudulentas, com foco em otimizar a balança entre a detecção de fraudes e a minimização de falsos positivos. O modelo final foi ajustado para atender às necessidades de negócio, garantindo **alta precisão** para evitar bloqueios indevidos.

---

## 📁 Contexto dos Dados

A base de dados contém transações financeiras anonimizadas por PCA (Principal Component Analysis). As principais colunas são:

- **Time**: Segundos decorridos entre cada transação e a primeira transação.
- **Amount**: Valor da transação.
- **Class**: Variável de resposta (1 = fraude, 0 = legítima).
- **V1, V2, ... V28**: Componentes principais do PCA.

O dataset é altamente desbalanceado, com pouquíssimas fraudes em relação às transações legítimas.

---

## 🤖 Algoritmos e Técnicas Utilizadas

### Algoritmos Principais
- **XGBoost**  
- **Logistic Regression**  
- **Random Forest**  
- **Support Vector Classifier (SVC)**  

### Técnicas de Balanceamento
- Under-sampling: RandomUnderSampler, ClusterCentroids, NearMiss
- Over-sampling: RandomOverSampler, SMOTE

### Outras Técnicas
- Normalização de dados  
- Cross-validation  
- GridSearchCV para otimização de hiperparâmetros  
- Curva Precision-Recall (PR AUC)  
- Rastreamento de experimentos com MLflow  

---

## 🚀 MLOps e Infraestrutura Profissional

- **Versionamento com DVC**: dados e modelos versionados
- **MLflow**: rastreamento de experimentos e model registry
- **CI/CD com GitHub Actions**: testes automatizados e validação
- **Integração com DagsHub**: repositório central e colaboração

---

## 📂 Estrutura do Projeto

```text
Projeto_4_Modelo_de_Identificacao_de_Fraude/
├── .dvc/                       # Configurações do DVC
├── .github/workflows/          # CI/CD com GitHub Actions
│   └── ci-tests.yml            # Pipeline de testes automatizados
├── .dvcignore                  # Arquivos ignorados pelo DVC
├── .gitignore                  # Arquivos ignorados pelo Git
├── config/                     # Configurações do projeto
├── dados/                      # Dados versionados com DVC
│   ├── creditcard.csv.gz.dvc       # Ponteiro para dados originais
│   └── .gitignore              # Ignora arquivos de dados
├── mlruns/                     # Experimentos do MLflow
├── models/                     # Modelos versionados
│   ├── best_model_xgboost.pkl.dvc  # Ponteiro para modelo final
│   └── .gitignore              # Ignora arquivos de modelo
├── notebooks/                  # Jupyter notebooks
│   ├── 01_777_Initial_Model.ipynb
│   ├── 02_777_Alg_ML_unbalanced.ipynb
│   ├── 03_777_Alg_ML_USamp.ipynb
│   ├── 04_777_Alg_ML_OSamp.ipynb
│   ├── 05_777_Alg_ML_normalization.ipynb
│   ├── 06_777_Alg_ML_Cross_validation.ipynb
│   ├── 07_777_Alg_ML_Parameters.ipynb
│   ├── 08_777_Model_Extension_pkl.ipynb
│   ├── 09_777_MLFlow_Deployment.ipynb
│   └── Prediction_Fraud_Test_New_Data.ipynb
├── src/                        # Código fonte
│   └── config/
│       ├── __init__.py
│       ├── auxiliares.py
│       ├── auxiliares_ml.py
│       ├── graficos.py
│       ├── modelos.py
│       ├── paths.py
│       ├── prepare_data.py
│       ├── train_model.py
│       └── .gitkeep
├── tests/                      # Testes automatizados
│   ├── test_basic.py
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_modelos_module.py
│   ├── .dvcignore
│   └── .gitignore
├── LICENSE.md                  # Licença MIT
├── README.md                   # Este arquivo
├── conftest.py                 # Configuração do pytest
├── dvc.lock                    # Lock file do DVC
├── dvc.yaml                    # Pipeline do DVC
├── metrics.json                # Métricas do modelo
└── requirements.txt            # Dependências do projeto
```

---

## 🔬 Processo de Desenvolvimento

1. Análise Exploratória Inicial  
2. Modelagem com dados desbalanceados  
3. Balanceamento (Over/Under-sampling)  
4. Normalização  
5. Cross-validation  
6. Otimização de hiperparâmetros  
7. Consolidação do modelo final (XGBoost)  
8. MLflow + DagsHub para rastreamento e deploy  

---

## 💻 Instalação e Uso

### Pré-requisitos
- Python 3.8+
- pip
- git
- DVC (`pip install dvc`)

### Passos
```bash
# Clone o repositório
git clone https://github.com/flaviohenriquehb777/Projeto_4_Modelo_de_Identificacao_de_Fraude.git
cd Projeto_4_Modelo_de_Identificacao_de_Fraude

# Baixe os dados
dvc pull

# Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Instale dependências
pip install -r requirements.txt

# Rode testes
pytest tests/ -v

# Execute notebooks
jupyter lab
```

---

## ✨ Diferenciais do Projeto
- Implementação de **MLOps completo** (DVC + MLflow + GitHub Actions + DagsHub)
- **Modelo final otimizado** com XGBoost
- Equilíbrio entre **alta precisão** e **satisfação do cliente**

---

## 📈 Resultados e Conclusão

| Métrica   | Valor (%) |
|-----------|-----------|
| Acurácia  | 99,96     |
| Precisão  | 92,27     |
| Recall    | 81,02     |

O modelo final alcança um equilíbrio estratégico entre **detecção eficaz de fraudes** e **minimização de falsos positivos**.

---

## 🎯 Interpretação das Métricas

- **Alta Precisão (92,27%)** → garante que poucas transações legítimas sejam bloqueadas.  
- **Recall Moderado (81,02%)** → captura a maioria das fraudes, aceitando pequena perda em prol da experiência do cliente.  
- **Alta Acurácia (99,96%)** → consistente, mas menos relevante em datasets desbalanceados.  

---

## 🛠 Roadmap
- [ ] Deploy em API REST com FastAPI
- [ ] Dashboard interativo com Streamlit
- [ ] Explicabilidade com SHAP

---

## 📜 Licença

Este projeto está licenciado sob a Licença MIT. Consulte o arquivo [LICENSE.md](LICENSE.md).

---

## 📬 Contato

Se tiver alguma dúvida, sugestão ou quiser colaborar, sinta-se à vontade para entrar em contato:
- **Nome:** Flávio Henrique Barbosa  
- **LinkedIn:** [Flávio Henrique Barbosa | LinkedIn](https://www.linkedin.com/in/fl%C3%A1vio-henrique-barbosa-38465938)  
- **Email:** flaviohenriquehb777@outlook.com  
- **DagsHub:** https://dagshub.com/flaviohenriquehb777  
