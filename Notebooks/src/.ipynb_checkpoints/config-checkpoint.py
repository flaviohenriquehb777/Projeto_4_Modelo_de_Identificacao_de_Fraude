from pathlib import Path


PASTA_PROJETO = Path(__file__).resolve().parents[2]

PASTA_DADOS = PASTA_PROJETO / "dados"

# coloque abaixo o caminho para os arquivos de dados de seu projeto
DADOS_MASTERCARD = PASTA_DADOS / "creditcard.csv.gz"
DADOS_MASTERCARD_TRATADO = PASTA_DADOS / "creditcard_tratado.parquet"

# DADOS_DIABETES_TRATADOS = PASTA_DADOS / "diabetes.parquet"
# DADOS_DIABETES_CATEGORIZADOS = PASTA_DADOS / "diabetes_categorizado.parquet"

# coloque abaixo o caminho para os arquivos de modelos de seu projeto
# PASTA_MODELOS = PASTA_PROJETO / "modelos"

# coloque abaixo outros caminhos que você julgar necessário
# PASTA_RELATORIOS = PASTA_PROJETO / "relatorios"
# PASTA_IMAGENS = PASTA_RELATORIOS / "imagens"

