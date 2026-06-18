# Setup de Ambiente

## Recomendação

Para evitar warnings do ambiente local legado, use um ambiente virtual limpo em vez do `base` do Anaconda.

## Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python -m pip check
```

## Verificações recomendadas

```powershell
python -m pytest -q
dvc pull
python -m src.serving.cli --input dados/credicard_tratado.parquet --output predictions.csv --limit 5 --format csv
```

## Notas de ambiente

- O repositório já declara `numexpr>=2.10.2`, mas um ambiente local antigo pode manter `2.10.1` instalado e continuar emitindo warning.
- O warning de `fpdf` aparece quando `PyFPDF` e `fpdf2` coexistem no mesmo ambiente global.
- O projeto fica mais limpo para avaliadores quando essas dependências são instaladas em `.venv` recém-criado.
- A suíte `pytest` do repositório filtra apenas warnings pendentes conhecidos de bibliotecas terceiras (`shap`) para manter a saída de validação objetiva.

## Boas práticas para avaliação

- Não rode a validação a partir do ambiente `base` do Anaconda.
- Use `python -m pip check` para detectar conflitos instalados no ambiente.
- Se necessário, recrie `.venv` do zero antes da demonstração final.
- Para este projeto, prefira `python -m pytest -q` em vez de `pytest -q` quando quiser garantir que a execução use o interpretador da `.venv`.
