# 🚨 Modelo de Identificação de Fraude em Transações (Credicard - Brasil)

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
[![DVC Managed](https://img.shields.io/badge/DVC-Managed-blue)]()
[![CI Tests](https://github.com/flaviohenriquehb777/Projeto_4_Modelo_de_Identificacao_de_Fraude/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/flaviohenriquehb777/Projeto_4_Modelo_de_Identificacao_de_Fraude/actions/workflows/ci-tests.yml)
[![MLflow Tracking](https://img.shields.io/badge/MLflow-Tracking-orange)]()
[![SHAP Explainability](https://img.shields.io/badge/SHAP-Explainability-purple)]()
[![DagsHub Repository](https://img.shields.io/badge/DagsHub-Repository-purple)](https://dagshub.com/flaviohenriquehb777/Projeto_4_Modelo_de_Identificacao_de_Fraude)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-blueviolet)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-success)

Utilizando algoritmos de Machine Learning e práticas de MLOps para criar um modelo robusto de identificação de transações fraudulentas com explainability completa via SHAP.

---

## 📋 Sumário

- [Descrição do Projeto](#-descrição-do-projeto)
- [Contexto dos Dados](#-contexto-dos-dados)
- [Algoritmos e Técnicas Utilizadas](#-algoritmos-e-técnicas-utilizadas)
- [MLOps e Infraestrutura Profissional](#-mlops-e-infraestrutura-profissional)
- [SHAP Explainability](#-shap-explainability)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Processo de Desenvolvimento](#-processo-de-desenvolvimento)
- [Instalação e Uso](#-instalação-e-uso)
- [Diferenciais do Projeto](#-diferenciais-do-projeto)
- [Resultados e Conclusão](#-resultados-e-conclusão)
- [Interpretação das Métricas](#-interpretação-das-métricas)
- [Roadmap](#-roadmap)
- [Licença](#-licença)
- [Contato](#-contato)

---

## 📊 Descrição do Projeto

Este projeto tem como objetivo desenvolver um modelo de Machine Learning capaz de identificar transações financeiras fraudulentas, com foco em otimizar a balança entre a detecção de fraudes e a minimização de falsos positivos. O modelo final foi ajustado para atender às necessidades de negócio, garantindo **alta precisão** para evitar bloqueios indevidos e **explicabilidade completa** via SHAP.

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
- **SHAP para explainability** ← NOVO!

---

## 🚀 MLOps e Infraestrutura Profissional

O projeto utiliza um pipeline de MLOps robusto para garantir a reprodutibilidade, rastreabilidade e facilidade de deployment dos modelos.

- **Versionamento com DVC:** dados, modelos e artefatos de explainability são versionados, garantindo que o pipeline de ML possa ser reproduzido com qualquer versão do dataset.
- **MLflow e DagsHub:** a combinação perfeita para rastrear e gerenciar experimentos. Cada execução de modelo é registrada com seus parâmetros, métricas e o próprio modelo. O MLflow Model Registry atua como o repositório central para modelos prontos para produção.
- **CI/CD com GitHub Actions:** testes automatizados e validação de código, assegurando a qualidade antes do merge.
- **Integração total com DagsHub:** um hub de MLOps que centraliza o código (via Git), os dados (via DVC) e os experimentos (via MLflow) em um único local, facilitando a colaboração e a transparência.
- **SHAP Integration:** explicações de modelo automatizadas e versionadas ← NOVO!

---

## 🔍 SHAP Explainability

O projeto inclui explicações completas do modelo usando SHAP (SHapley Additive exPlanations):

### 📊 Artefatos Gerados:
- **Summary Plots:** Visualização da importância global das features
- **Feature Importance:** Ranking das features mais importantes
- **Dependence Plots:** Relação entre features e suas contribuições
- **Force Plots:** Explicações individuais para previsões específicas
- **Relatório PDF Automático:** Relatório executivo para stakeholders ← NOVO!

### 🎯 Insights Obtidos:
- **Top 5 Features:** V14, V12, V10, V16, V17 (mais influentes para detecção de fraude)
- **Direção do Impacto:** Como cada feature influencia a previsão de fraude
- **Transparência:** Explicações completas para compliance regulatório

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
│   ├── creditcard.csv.gz.dvc   # Ponteiro para dados originais
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
├── reports/                    # Artefatos de explainability ← NOVO!
│   ├── shap_explanations/      # Explicações SHAP versionadas
│   │   ├── shap_summary_plot.png
│   │   ├── shap_feature_importance.png
│   │   ├── shap_values.csv
│   │   ├── shap_stats.json
│   │   ├── shap_force_plot.html
│   │   ├── feature_importance_ranking.csv
│   │   └── dependence_plots/
│   └── shap_report.pdf         # Relatório executivo em PDF
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
│       ├── generate_shap_insights.py     # NOVO!
│       ├── generate_shap_report.py       # NOVO!
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

## 🔬 Processo de Desenvolvimento

1. Análise Exploratória Inicial  
2. Modelagem com dados desbalanceados  
3. Balanceamento (Over/Under-sampling)  
4. Normalização  
5. Cross-validation  
6. Rastreamento de Experimentos com MLflow  
7. Seleção do Melhor Modelo a partir do Histórico de Experimentos  
8. Registro do Modelo no MLflow Model Registry
9. Explainability com SHAP
10. Geração de Relatórios Automáticos

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

# Baixe os dados e artefatos
dvc pull

# Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Instale dependências
pip install -r requirements.txt

# Execute o pipeline completo
dvc repro

# Ou execute estágios específicos
dvc repro prepare_data      # Prepara dados
dvc repro train_model       # Treina modelo
dvc repro generate_shap_insights  # Gera explicações SHAP
dvc repro generate_shap_report    # Gera relatório PDF

# Rode testes
pytest tests/ -v

# Execute notebooks
jupyter lab
```

## ✨ Diferenciais do Projeto

- Implementação de **MLOps completo** (DVC + MLflow + GitHub Actions + DagsHub)
- **Modelo final otimizado** com XGBoost
- Equilíbrio estratégico entre **alta precisão** e **satisfação do cliente**
- Ciclo de vida do modelo gerenciado profissionalmente com o **MLflow Model Registry**
- Explainability completa com **SHAP** para transparência e compliance
- Relatórios automáticos em **PDF** para stakeholders **não-técnicos**
- Pipeline 100% reprodutível com versionamento de todos os artefatos

---

## 📈 Resultados e Conclusão

| Métrica   | Valor (%) |
|-----------|-----------|
| Acurácia  | 99,96     |
| Precisão  | 92,27     |
| Recall    | 81,02     |

O modelo final alcança um equilíbrio estratégico entre **detecção eficaz de fraudes** e **minimização de falsos positivos**, com explicações completas via **SHAP** para transparência total.

---

## 🎯 Insights do SHAP

- **Feature mais importante:** V14 (importância: 1.1781)
- **Top 5 features:** V14, V12, V10, V16, V17
- **Impacto direcional:** Features específicas aumentam/diminuem probabilidade de fraude

---

## 🎯 Interpretação das Métricas

- **Alta Precisão (92,27%)** → garante que poucas transações legítimas sejam bloqueadas.  
- **Recall Moderado (81,02%)** → captura a maioria das fraudes, aceitando pequena perda em prol da experiência do cliente.  
- **Alta Acurácia (99,96%)** → consistente, mas menos relevante em datasets desbalanceados.
- **Explicabilidade Completa** → transparência total nas decisões do modelo

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
