# Draft do Update Profissional

## Título sugerido

`v1.2.0 - Production hardening update for fraud detection model`

## Resumo executivo

Este update profissional consolida a evolução do modelo original para um estado muito mais auditável e próximo de produção, sem apagar o histórico antigo do projeto.

## Destaques

- Reprodutibilidade do melhor modelo integrada ao pipeline operacional.
- Threshold calibrado e versionado em metadados do modelo.
- Gates de aceite para precisão, recall e aderência ao objetivo de negócio.
- Contratos automatizados para dados, inferência, serving e links públicos.
- DVC sem ponteiro órfão no branch atual, reduzindo atrito no clone.
- DagsHub público alinhado com experimentos e model registry auditáveis.

## Evidências para a release final

- [metrics.json](../metrics.json)
- [model_metadata.json](../models/model_metadata.json)
- [data_quality_report.json](../reports/data_quality_report.json)
- [mlflow_public_registry_summary.json](../reports/mlflow_public_registry_summary.json)
- [production_readiness_scorecard.json](../reports/production_readiness_scorecard.json)

## Mensagem recomendada para avaliadores

Este repositório preserva a entrega histórica original e adiciona um update profissional solicitado para endurecer a trilha operacional, de validação e de evidência pública do modelo.
