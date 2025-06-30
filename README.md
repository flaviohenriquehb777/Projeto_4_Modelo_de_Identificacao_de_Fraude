# Modelo de Identificação de Fraude (Credicard - Brasil)

**Utilizando algoritmos de Machine Learning para criar um modelo robusto de identificação de transações fraudulentas.**

## Sumário
- [Descrição do Projeto](#descrição-do-projeto)
- [Contexto dos Dados](#contexto-dos-dados)
- [Algoritmos e Técnicas Utilizadas](#algoritmos-e-técnicas-utilizadas)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Processo de Desenvolvimento](#processo-de-desenvolvimento)
- [Instalação e Uso](#instalação-e-uso)
- [Resultados e Conclusão](#resultados-e-conclusão)
- [Interpretação das Métricas](#interpretação-das-métricas)
- [Licença](#licença)
- [Contato](#contato)

## DESCRIÇÃO DO PROJETO:

Este projeto tem como objetivo desenvolver um modelo de Machine Learning capaz de identificar transações financeiras fraudulentas, com foco em otimizar a balança entre a detecção de fraudes e a minimização de falsos positivos, crucial para a experiência do cliente em transações diárias de baixo valor. O modelo final foi ajustado para atender às necessidades de negócio, garantindo alta precisão para evitar bloqueios indevidos.

## Contexto dos Dados:

A base de dados utilizada contém transações financeiras e, devido a questões de confidencialidade, os recursos originais foram transformados utilizando o PCA (Principal Component Analysis). As principais colunas são:
- `'Time'` (tempo): Segundos decorridos entre cada transação e a primeira transação no conjunto de dados.
- `'Amount'` (valor): O valor da transação.
- `'Class'` (classe): Variável de resposta, onde `1` indica fraude e `0` indica transação legítima.
- `'V1, V2, ... V28'`: Principais componentes obtidos com PCA, representando características anonimizadas da transação.

A natureza altamente desbalanceada do dataset (poucas fraudes em relação a transações legítimas) foi um desafio central abordado ao longo do desenvolvimento do projeto.

## Algoritmos e Técnicas Utilizadas:

Para a construção e avaliação do modelo, foram explorados diversos algoritmos de Machine Learning e técnicas de balanceamento:

### Algoritmos Principais:
- **XGBoost**: Frequentemente um dos melhores em problemas de classificação.
- **Logistic Regression**: Um modelo linear robusto e interpretável.
- **Random Forest**: Algoritmo baseado em árvores de decisão, conhecido por sua robustez e bom desempenho.
- **Support Vector Classifier (SVC)**: Efetivo em espaços de alta dimensão.

### Técnicas de Balanceamento (Exploradas):
Para lidar com o desbalanceamento dos dados, foram testadas as seguintes técnicas:

- **Under-sampling**:
    - `RandomUnderSampler`
    - `ClusterCentroids`
    - `NearMiss`
- **Over-sampling**:
    - `RandomOverSampler`
    - `SMOTE` (Synthetic Minority Over-sampling Technique)

### Outras Técnicas:
- **Normalização de Dados**: Utilizada para padronizar as características.
- **Validação Cruzada (Cross-Validation)**: Empregado para avaliar a robustez do modelo, dividindo o dataset em múltiplas dobras.
- **Otimização de Hiperparâmetros**: Utilização de `GridSearchCV` para encontrar a melhor combinação de parâmetros para os algoritmos.
- **Curva Precision-Recall (PR AUC)**: Métrica crucial para avaliar o desempenho em datasets desbalanceados, focando na capacidade do modelo de identificar corretamente os positivos.

## Estrutura do Projeto:
Este repositório está organizado nos seguintes arquivos e diretórios:

- `data/`: (Assumindo que seus dados estejam aqui ou sejam baixados/gerados) Contém a base de dados utilizada no projeto.
- `notebooks/`: Contém os notebooks Jupyter que documentam todo o processo de desenvolvimento.
    - `01_777_Initial_Model.ipynb`: Análise exploratória inicial e primeiros testes.
    - `02_777_Alg_ML_unbalanced.ipynb`: Avaliação dos algoritmos com o dataset desbalanceado.
    - `03_777_Alg_ML_USamp.ipynb`: Experimentação com técnicas de *Under-sampling*.
    - `04_777_Alg_ML_OSamp.ipynb`: Experimentação com técnicas de *Over-sampling*.
    - `05_777_Alg_ML_normalization.ipynb`: Análise do impacto da normalização nos dados.
    - `06_777_Alg_ML_Cross_validation.ipynb`: Aplicação de validação cruzada para maior robustez.
    - `07_777_Alg_ML_Parameters.ipynb`: Otimização de hiperparâmetros usando `GridSearchCV`.
    - `00_777_Final_Model.ipynb`: O notebook final que consolida o melhor modelo e as conclusões.
    - `Prediction_Fraud_Test_New_Data.ipynb`: Notebook para testar o modelo final com novos dados simulados.
- `README.md`: Este arquivo.
- `LICENSE.md`: Arquivo contendo a licença do projeto (MIT).
- `requirements.txt`: Lista de dependências Python para o projeto.

## Processo de Desenvolvimento:

O projeto passou por várias etapas de experimentação e refino:

1.  **Análise Exploratória Inicial (`01_777_Initial_Model.ipynb`):**
    * Compreensão da estrutura dos dados e identificação do desbalanceamento na variável `Class`.
    * Primeiros testes com algoritmos para estabelecer uma linha de base.

2.  **Modelagem com Dados Desbalanceados (`02_777_Alg_ML_unbalanced.ipynb`):**
    * Avaliação dos algoritmos de Machine Learning com a base de dados original.
    * Análise das métricas de Acurácia, Precisão e Recall.

3.  **Experimentação com Balanceamento (`03_777_Alg_ML_USamp.ipynb` e `04_777_Alg_ML_OSamp.ipynb`):**
    * Aplicação de diversas técnicas de Under-sampling e Over-sampling para tentar melhorar o desempenho em dados desbalanceados.
    * Comparação dos resultados com a base desbalanceada.

4.  **Normalização de Dados (`05_777_Alg_ML_normalization.ipynb`):**
    * Investigação do impacto da normalização das características no desempenho dos modelos.

5.  **Validação Cruzada (`06_777_Alg_ML_Cross_validation.ipynb`):**
    * Implementação de validação cruzada com os algoritmos de melhor desempenho para garantir a estabilidade e generalização do modelo.

6.  **Otimização de Hiperparâmetros (`07_777_Alg_ML_Parameters.ipynb`):**
    * Utilização de `GridSearchCV` para encontrar os melhores hiperparâmetros para o algoritmo escolhido (XGBoost) para maximizar as métricas desejadas.

7.  **Modelo Final e Conclusão (`00_777_Final_Model.ipynb`):**
    * Consolidação do melhor modelo, o qual revelou que a base desbalanceada, com o ajuste de hiperparâmetros, se mostrou a melhor abordagem para o objetivo de negócio (minimizar falsos positivos), mesmo após testar técnicas de balanceamento.

## Instalação e Uso:

Para configurar e executar este projeto em seu ambiente local, siga as instruções abaixo:

1.  **Pré-requisitos:**
    * Python 3.8+
    * `pip` (gerenciador de pacotes do Python)

2.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/Projeto_4_Modelo_de_Identificacao_de_Fraude.git](https://github.com/seu-usuario/Projeto_4_Modelo_de_Identificacao_de_Fraude.git)
    cd Projeto_4_Modelo_de_Identificacao_de_Fraude
    ```
    *(Lembre-se de substituir `seu-usuario` pelo seu nome de usuário do GitHub ao clonar.)*

3.  **Crie e ative um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    # No Windows:
    venv\Scripts\activate
    # No macOS/Linux:
    source venv/bin/activate
    ```

4.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
    **(Importante: Gere o arquivo `requirements.txt` executando `pip freeze > requirements.txt` no seu ambiente virtual após instalar todas as bibliotecas usadas nos notebooks.)**

5.  **Como usar:**
    Os notebooks Jupyter (`.ipynb` na pasta `notebooks/`) podem ser abertos e executados sequencialmente para replicar o desenvolvimento do projeto. O notebook `00_777_Final_Model.ipynb` contém o modelo final e a avaliação consolidada. O `Prediction_Fraud_Test_New_Data.ipynb` pode ser usado para testar o modelo com novos dados.

    Para executar os notebooks, use Jupyter Lab ou Jupyter Notebook:
    ```bash
    jupyter lab
    # ou
    jupyter notebook
    ```

## Resultados e Conclusão:

O modelo de detecção de fraudes foi avaliado utilizando validação cruzada com 10 dobras, resultando nas seguintes métricas médias para o modelo final (XGBoost com hiperparâmetros otimizados na base desbalanceada):

-   **Acurácia:** 99,96%
-   **Precisão:** 92,27%
-   **Recall:** 81,02%

## Interpretação das Métricas:

Dado que o objetivo principal da empresa é minimizar falsos positivos para evitar reclamações de clientes, especialmente para transações corriqueiras com um valor máximo de R$5.000,00, a interpretação das métricas é a seguinte:

-   ### Alta Precisão (92,27%)
    Isso indica que a maioria das transações identificadas como fraude realmente são fraudes. Um alto valor de precisão significa que pouquíssimos clientes legítimos terão suas transações bloqueadas indevidamente. Este é um ponto crucial, pois reduz o impacto negativo sobre a experiência do usuário, um objetivo primário do negócio.

-   ### Recall Moderado (81,02%)
    O modelo consegue capturar 81% das fraudes reais, o que é um bom resultado. Isso implica que cerca de 19% das fraudes reais não são detectadas. No contexto de negócio, onde o foco principal é a satisfação do cliente (redução de falsos positivos), um recall ligeiramente menor é aceitável, contanto que a precisão seja alta. A empresa pode aceitar uma pequena porcentagem de fraudes não detectadas em troca de um fluxo de transações legítimas mais suave.

-   ### Alta Acurácia (99,96%)
    Esse valor é muito alto, mas deve ser interpretado com cautela em datasets desbalanceados. Embora o modelo classifique corretamente quase todas as transações, a acurácia pode ser inflada pelo grande número de transações legítimas (a classe majoritária) corretamente classificadas. As métricas de Precisão e Recall são mais informativas para problemas de classificação desbalanceados como a detecção de fraude.

**Conclusão:** O modelo final alcança um equilíbrio estratégico entre a detecção eficaz de fraudes e a minimização de interrupções para clientes legítimos, alinhando-se perfeitamente com os requisitos de negócio para a Credicard no Brasil.

## Licença:

Este projeto está licenciado sob a Licença MIT. Para mais detalhes, consulte o arquivo [LICENSE.md](LICENSE.md) na raiz do repositório.

## Contato:

Se tiver alguma dúvida, sugestão ou quiser colaborar, sinta-se à vontade para entrar em contato:
- **Nome:** Flávio Henrique Barbosa
- **LinkedIn:** [Flávio Henrique Barbosa | LinkedIn](https://www.linkedin.com/in/fl%C3%A1vio-henrique-barbosa-38465938)
- **Email:** flaviohenriquehb777@outlook.com