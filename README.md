# Projeto_4_Modelo_de_Identificacao_de_Fraude<br>
## Usando algoritmos de Machine Learning para criar um modelo de identificação de fraude. <br><br>
Para esse modelo, foram utilizados, em princípio, quatro algoritmos de Marchine Learning para analisar os resultados.<br><br>
NOTA: Para realizar o projeto, favor descompactar a base de dados.<br><br>
## Algoritmos utilizados no projeto: <br><br>
- XGBoost
- Logistic Regression;<br> 
- Random Forest; <br> 
- Support Vector Classifier (SVC). <br><br>

O projeto contou com várias etapas. Estão todas expostas aqui no GitHub, passo a passo até o arquivo final.<br><br>

O projeto contou uma uma análise exploratória detalhada. Nesse momento, foi observada a necessidade de balancear o dataset. Pois o mesmo se encontrava desbalanceado, alterando demais os resultados.<br> 
Porém, no final do projeto, foi concluído que seria melhor, para obtermos o resultado acordado com a empresa, que não utilizássemos o balanceamento. Isso poderá ser conferido ao longo do projeto.<br> 

## Foram utilizadas algumas técnicas de balanceamento: <br><br>
## Under-sampling: <br> 
- RandomUnderSampler; <br> 
- ClusterCentroids; <br>
- NearMiss; <br>
## Over-sampling: <br>
- RandomOverSampler; <br> 
- SMOTE. <br><br>

Para escolher o melhor algoritmo de Machine Learning utilizei uma função de minha autoria chamada: testar_modelos_com_undersampling:<br>
Esse função utiliza o RandomUnderSampler como pré-processamento e os seguintes algoritmos de Machine Learning:<br>
- RandomForest;
- DecisionTree;
- LogisticRegression;
- SVC;
- KNN;
- XGBoost;
- ANN.<br>

A função retorna os três melhores modelos, em ordem decrescente, com base na área sob a curva precisão-recall (PR AUC).<br>

Por fim, também foram utilizados vários hiperparâmetros através do GridSearchCV para se chegar ao melhor resultado.
