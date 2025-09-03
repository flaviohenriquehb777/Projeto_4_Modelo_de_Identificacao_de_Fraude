import pandas as pd
import sys

# O DVC passa os caminhos dos arquivos como argumentos de linha de comando
input_path = sys.argv[1]
output_path = sys.argv[2]

# Carrega os dados
df = pd.read_csv(input_path)

# Exemplo de pré-processamento (apenas salva o arquivo, você pode adicionar seu código aqui)
df.to_parquet(output_path)

print(f"Dados preparados e salvos em {output_path}")