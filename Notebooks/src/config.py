from pathlib import Path


PASTA_PROJETO = Path(__file__).resolve().parents[2]

PASTA_DADOS = PASTA_PROJETO / "dados"

# coloque abaixo o caminho para os arquivos de dados de seu projeto
DADOS_CREDICARD = PASTA_DADOS / "credicard.csv.gz"
DADOS_CREDICARD_TRATADO = PASTA_DADOS / "credicard_tratado.parquet"

# DADOS_DIABETES_TRATADOS = PASTA_DADOS / "diabetes.parquet"
# DADOS_DIABETES_CATEGORIZADOS = PASTA_DADOS / "diabetes_categorizado.parquet"

# coloque abaixo o caminho para os arquivos de modelos de seu projeto
# PASTA_MODELOS = PASTA_PROJETO / "modelos"

# coloque abaixo outros caminhos que você julgar necessário
# PASTA_RELATORIOS = PASTA_PROJETO / "relatorios"
# PASTA_IMAGENS = PASTA_RELATORIOS / "imagens"

