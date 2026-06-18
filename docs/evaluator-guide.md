# Guia do Avaliador

## Objetivo

Este guia resume o caminho mais curto para validar que o repositório está profissional, reproduzível e pronto para discussão com times técnicos.

## Avaliação em 5 minutos

1. Abra o [README](../README.md) para entender a linha do tempo entre `snapshot`, `v1.0.0` e a trilha de update.
2. Verifique os experimentos públicos no DagsHub:
   - [Experiments](https://dagshub.com/flaviohenriquehb777/Modelo_de_Deteccao_de_Fraude-Financeiro/experiments)
   - [Models](https://dagshub.com/flaviohenriquehb777/Modelo_de_Deteccao_de_Fraude-Financeiro/models)
3. Consulte as evidências locais:
   - [metrics.json](../metrics.json)
   - [model_metadata.json](../models/model_metadata.json)
   - [data_quality_report.json](../reports/data_quality_report.json)
   - [mlflow_public_registry_summary.json](../reports/mlflow_public_registry_summary.json)
   - [production_readiness_scorecard.json](../reports/production_readiness_scorecard.json)
4. Confirme os contratos automáticos em [tests](../tests).
5. Execute a inferência oficial:

```bash
python -m src.serving.cli --input dados/credicard_tratado.parquet --output predictions.csv --limit 5 --format csv
```

## O que observar

- O threshold operacional está versionado em `models/model_metadata.json`.
- O pipeline exige aceite mínimo de precisão e recall antes de promover o modelo.
- O DVC do branch atual não depende mais de ponteiros órfãos.
- O MLflow/DagsHub público expõe o histórico experimental e o best-model.
- A suíte de testes cobre dados, métricas, inferência, links públicos e serving.

## Leitura recomendada

- [README](../README.md)
- [environment-setup.md](./environment-setup.md)
- [release-update-draft.md](./release-update-draft.md)
