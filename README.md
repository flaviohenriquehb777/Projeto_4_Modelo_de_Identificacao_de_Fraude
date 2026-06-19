# 🚨 Modelo de Detecção de Fraude em Transações (Credicard - Brasil)

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
[![DVC Managed](https://img.shields.io/badge/DVC-Managed-blue)](https://dagshub.com/flaviohenriquehb777/Modelo_de_Deteccao_de_Fraude-Financeiro)
[![CI Tests](https://github.com/flaviohenriquehb777/Modelo_de_Deteccao_de_Fraude-Financeiro/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/flaviohenriquehb777/Modelo_de_Deteccao_de_Fraude-Financeiro/actions/workflows/ci-tests.yml)
[![MLflow Tracking](https://img.shields.io/badge/MLflow-Tracking-orange)](https://dagshub.com/flaviohenriquehb777/Modelo_de_Deteccao_de_Fraude-Financeiro/experiments)
[![SHAP Explainability](https://img.shields.io/badge/SHAP-Explainability-purple)]()
[![DagsHub Repository](https://img.shields.io/badge/DagsHub-Repository-purple)](https://dagshub.com/flaviohenriquehb777/Modelo_de_Deteccao_de_Fraude-Financeiro)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-blueviolet)
![XGBoost](https://img.shields.io/badge/XGBoost-Model-success)

Modelo de Machine Learning para detecção de transações financeiras fraudulentas, combinando evolução experimental, versionamento com DVC, rastreamento de experimentos com MLflow/DagsHub e evidências automatizadas de qualidade via GitHub Actions.

## 🏷️ Releases

- [snapshot — Entrega original (referência)](https://github.com/flaviohenriquehb777/Modelo_de_Deteccao_de_Fraude-Financeiro/releases/tag/snapshot)
- [v1.0.0 — Entrega original do modelo](https://github.com/flaviohenriquehb777/Modelo_de_Deteccao_de_Fraude-Financeiro/releases/tag/v1.0.0)
- [v1.1.0 — Rename + DagsHub/DVC/CI](https://github.com/flaviohenriquehb777/Modelo_de_Deteccao_de_Fraude-Financeiro/releases/tag/v1.1.0)
- [v1.2.0 — Production Hardening Update](https://github.com/flaviohenriquehb777/Modelo_de_Deteccao_de_Fraude-Financeiro/releases/tag/v1.2.0)

## 🧭 Status Atual

- **Entrega histórica preservada:** o commit original segue explicitamente marcado pelas tags `snapshot` e `v1.0.0`.
- **Atualização técnica posterior:** a release `v1.1.0` consolida o rename do repositório e o alinhamento entre GitHub, DVC, CI e DagsHub.
- **Update profissional atual:** a release `v1.2.0` consolida reprodutibilidade, hardening, testes de produção, serving e evidências públicas para avaliação técnica.
- **Baseline de negócio comprovado:** o melhor experimento histórico registrado no projeto atingiu **Precisão 92,27%** e **Recall 81,02%**.
- **Evidência pública confirmada:** o DagsHub público expõe **7 experimentos** e **1 modelo registrado** visíveis para avaliadores sem autenticação.
- **Artefatos organizados por papel:** `models/model.pkl` representa o artefato operacional atual do pipeline DVC, enquanto `models/best_model_xgboost.reference.json` preserva a referência histórica do melhor modelo experimental sem depender de um ponteiro DVC quebrado.
- **Posicionamento honesto:** o repositório já demonstra maturidade forte de portfólio em MLOps, enquanto a evolução para prontidão plena de produção segue como trilha ativa de hardening.

## ⚡ Como Avaliar Em 3 Minutos

1. Veja a linha do tempo nas [Releases](https://github.com/flaviohenriquehb777/Modelo_de_Deteccao_de_Fraude-Financeiro/releases).
2. Confira o pipeline versionado em [dvc.yaml](dvc.yaml).
3. Valide a automação de testes em [ci-tests.yml](.github/workflows/ci-tests.yml).
4. Consulte o experimento histórico em [09_777_MLFlow_Deployment.ipynb](notebooks/09_777_MLFlow_Deployment.ipynb).
5. Abra os [Experiments](https://dagshub.com/flaviohenriquehb777/Modelo_de_Deteccao_de_Fraude-Financeiro/experiments) e os [Models](https://dagshub.com/flaviohenriquehb777/Modelo_de_Deteccao_de_Fraude-Financeiro/models) públicos no DagsHub.
6. Veja o resumo consolidado em [mlflow_public_registry_summary.json](reports/mlflow_public_registry_summary.json).

---

## 📋 Sumário

- [Descrição do Modelo](#-descrição-do-modelo)
- [Contexto dos Dados](#-contexto-dos-dados)
- [Algoritmos e Técnicas Utilizadas](#-algoritmos-e-técnicas-utilizadas)
- [MLOps e Infraestrutura Profissional](#-mlops-e-infraestrutura-profissional)
- [SHAP Explainability](#-shap-explainability)
- [Estrutura do Modelo](#-estrutura-do-modelo)
- [Processo de Desenvolvimento](#-processo-de-desenvolvimento)
- [Instalação e Uso](#-instalação-e-uso)
- [Evidências Técnicas](#-evidências-técnicas)
- [Guia do Avaliador](#-guia-do-avaliador)
- [Diferenciais do Modelo](#-diferenciais-do-modelo)
- [Resultados e Conclusão](#-resultados-e-conclusão)
- [Interpretação das Métricas](#-interpretação-das-métricas)
- [Roadmap](#-roadmap)
- [Licença](#-licença)
- [Contato](#-contato)

---

## 📊 Descrição do Modelo

Este modelo tem como objetivo desenvolver um modelo de Machine Learning capaz de identificar transações financeiras fraudulentas, com foco em otimizar a balança entre a detecção de fraudes e a minimização de falsos positivos. O modelo final foi ajustado para atender às necessidades de negócio, garantindo **alta precisão** para evitar bloqueios indevidos e **explicabilidade completa** via SHAP.

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

O modelo utiliza uma base sólida de MLOps para garantir reprodutibilidade, rastreabilidade e clareza de evolução técnica do ciclo de vida do modelo.

- **Versionamento com DVC:** dados, modelo treinado e artefatos derivados podem ser reconstruídos e sincronizados com storage remoto.
- **MLflow + DagsHub:** experimentos históricos, métricas e modelos ficam registrados para análise comparativa e auditoria técnica.
- **GitHub Actions:** o pipeline automatizado executa `dvc pull` e testes para verificar integridade mínima do repositório.
- **Releases versionadas:** a linha do tempo do modelo está publicada para separar explicitamente a entrega original das atualizações posteriores.
- **SHAP explainability:** o pipeline prevê a geração de artefatos explicativos e relatório executivo para interpretação do comportamento do modelo.

> **Nota de transparência:** o melhor score histórico do modelo já está registrado nos notebooks/artefatos de experimentação. A trilha atual de evolução do repositório busca tornar esse mesmo desempenho totalmente reproduzível também no pipeline operacional automatizado.

---

## 🔍 SHAP Explainability

O modelo inclui explicações completas usando SHAP (SHapley Additive exPlanations):

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

## 📂 Estrutura do Modelo

```text
Modelo_de_Deteccao_de_Fraude-Financeiro/
├── .dvc/                       # Configurações do DVC
├── .github/workflows/          # CI/CD com GitHub Actions
│   ├── ci-tests.yml            # Pipeline de testes automatizados
│   └── publish-release.yml     # Publicação automática de releases
├── .dvcignore                  # Arquivos ignorados pelo DVC
├── .gitignore                  # Arquivos ignorados pelo Git
├── dados/                      # Dados versionados com DVC
│   ├── credicard.csv.gz.dvc    # Ponteiro para dados originais
│   └── .gitignore              # Ignora arquivos de dados
├── mlruns/                     # Experimentos do MLflow
├── models/                     # Modelos versionados
│   ├── best_model_xgboost.reference.json  # Referência histórica do melhor modelo promovido
│   └── .gitignore              # Ignora apenas os artefatos operacionais gerados
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
├── reports/                    # Gerado por `dvc repro` quando o pipeline completo e os artefatos locais estão disponíveis
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
│       ├── generate_shap_insights.py     
│       ├── generate_shap_report.py       
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
├── metrics.json                # Gerado/atualizado pelo estágio de treino
└── requirements.txt            # Dependências do modelo
```

> **Convenção atual de artefatos:** o pipeline DVC treina e usa `models/model.pkl` como saída operacional. A referência histórica do melhor modelo promovido foi preservada em `models/best_model_xgboost.reference.json`, enquanto o ponteiro DVC legado foi removido do branch atual para eliminar atrito em `dvc pull` e `clone` por terceiros.

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
git clone https://github.com/flaviohenriquehb777/Modelo_de_Deteccao_de_Fraude-Financeiro.git
cd Modelo_de_Deteccao_de_Fraude-Financeiro

# Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Instale dependências
pip install -r requirements.txt

# Baixe os dados e artefatos
dvc pull

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

Se o `dvc pull` solicitar autenticação, configure as credenciais do DagsHub localmente (sem versionar segredos):

```bash
dvc remote modify --local dagshub auth basic
dvc remote modify --local dagshub user <seu_usuario_dagshub>
dvc remote modify --local dagshub password <seu_token_dagshub>
```

> **Fluxo zero fricção para avaliadores:** o branch atual foi limpo para que `git clone` + `pip install -r requirements.txt` + `dvc pull` funcione sem depender de ponteiros DVC órfãos. A trilha histórica continua preservada nas tags, notebooks e no arquivo `models/best_model_xgboost.reference.json`.

## 🧪 Evidências Técnicas

- **Pipeline versionado:** [dvc.yaml](dvc.yaml)
- **Integração contínua:** [ci-tests.yml](.github/workflows/ci-tests.yml)
- **Publicação de releases:** [publish-release.yml](.github/workflows/publish-release.yml)
- **Melhor experimento histórico:** [09_777_MLFlow_Deployment.ipynb](notebooks/09_777_MLFlow_Deployment.ipynb)
- **Resumo público de MLflow/DagsHub:** [mlflow_public_registry_summary.json](reports/mlflow_public_registry_summary.json)
- **Scorecard de prontidão:** [production_readiness_scorecard.json](reports/production_readiness_scorecard.json)
- **Experimentos públicos no DagsHub:** [Experiments](https://dagshub.com/flaviohenriquehb777/Modelo_de_Deteccao_de_Fraude-Financeiro/experiments)
- **Modelo público registrado no DagsHub:** [Models](https://dagshub.com/flaviohenriquehb777/Modelo_de_Deteccao_de_Fraude-Financeiro/models)
- **Modelo e dados via DVC/DagsHub:** [DagsHub](https://dagshub.com/flaviohenriquehb777/Modelo_de_Deteccao_de_Fraude-Financeiro)

## 🌐 MLflow e DagsHub Públicos

- **Experimento público principal:** `Fraud Detection XGBoost Experiment` (`experiment_id = 128518284461930217`)
- **Best-model histórico público:** `Treinamento do Modelo - XGBoost Otimizado` (`model_id = m-695031f059db422fb7c12c7e47ef58e9`)
- **Métricas históricas auditáveis:** `precision = 0.922736`, `recall = 0.810171`, `accuracy = 0.999561`
- **Execuções visíveis para avaliadores:** `7` runs públicos no DagsHub
- **Registry público visível:** `1` modelo registrado publicamente no DagsHub

## 🧭 Guia do Avaliador

- **Passo a passo objetivo:** [docs/evaluator-guide.md](docs/evaluator-guide.md)
- **Setup limpo de ambiente:** [docs/environment-setup.md](docs/environment-setup.md)
- **Draft da release final:** [docs/release-update-draft.md](docs/release-update-draft.md)
- **Interface oficial de inferência:** `python -m src.serving.cli`
- **Critério operacional do modelo:** threshold oficial salvo em `models/model_metadata.json`

## ✨ Diferenciais do Modelo

- Linha do tempo pública com **releases** que preservam a entrega original e as atualizações posteriores
- Base consistente de **MLOps aplicado** com DVC, MLflow/DagsHub e GitHub Actions
- Melhor experimento histórico alinhado ao objetivo de negócio de **alta precisão com recall competitivo**
- Trilha metodológica completa, do baseline até ajuste de hiperparâmetros e deployment experimental
- Explainability com **SHAP** e geração de artefatos interpretáveis para análise técnica e executiva
- Estrutura pronta para evolução em direção a **production readiness** sem apagar a história do projeto

---

## 📈 Resultados e Conclusão

| Métrica   | Valor (%) |
|-----------|-----------|
| Acurácia  | 99,96     |
| Precisão  | 92,27     |
| Recall    | 81,02     |

O modelo final alcança um equilíbrio estratégico entre **detecção eficaz de fraudes** e **minimização de falsos positivos**, com explicações completas via **SHAP** para transparência total.

> **Contexto das métricas:** estes valores representam o melhor experimento histórico documentado no projeto e servem como baseline de negócio para a evolução do modelo. O pipeline operacional atual segue sendo endurecido para reproduzir esse patamar de forma automatizada e auditável.

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

## 🛣️ Roadmap

- **Fase 1 — Diagnóstico e Baseline:** concluída
- **Fase 2 — Alinhamento de Narrativa Pública:** concluída
- **Fase 3 — Higienização e Organização Técnica:** concluída
- **Fase 4 — Reprodutibilidade do Melhor Modelo:** concluída
- **Fase 5 — Hardening de Dados e Modelo:** concluída
- **Fase 6 — Testes de Produção:** concluída
- **Fase 7 — Serving e Prontidão para Deploy:** concluída
- **Fase 8 — MLflow e DagsHub Profissionais:** concluída
- **Fase 9 — Polimento Final:** concluída
- **Fase 10 — Entrega Oficial do Update:** publicar novo release profissional sem tocar no histórico antigo

---

## 📜 Licença

Este repositório está licenciado sob a Licença MIT. Consulte o arquivo [LICENSE.md](LICENSE.md).

---

## 📬 Contato

Se tiver alguma dúvida, sugestão ou quiser colaborar, sinta-se à vontade para entrar em contato:
- **Nome:** Flávio Henrique Barbosa  
- **LinkedIn:** [Flávio Henrique Barbosa | LinkedIn](https://www.linkedin.com/in/fl%C3%A1vio-henrique-barbosa-38465938) 
- **Email:** flaviohenriquehb777@outlook.com  
- **DagsHub:** https://dagshub.com/flaviohenriquehb777  
