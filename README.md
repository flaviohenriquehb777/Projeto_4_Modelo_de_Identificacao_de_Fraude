# Projeto_4_Modelo_de_Identificacao_de_Fraude<br>
## Usando algoritmos de Machine Learning para criar um modelo de identificação de fraude. <br><br>
Para esse modelo, foram utilizados, em princípio, cinco algoritmos de Marchine Learning para analisar os resultados.<br><br>
## Algoritmos utilizados no projeto: <br><br>
- Logistic regression;<br> 
- Decision Trees; <br> 
- Support Vector Machines (SVC); <br>
- KNN; <br>
- Random Forest Classifier.<br><br>

O projeto contou com várias etapas. Estão todas expostas aqui no GitHub, passo a passo até o arquivo final.<br><br>

O projeto contou uma uma análise exploratória detalhada. Nesse momento, foi observada a necessidade de balancear o dataset. Pois o mesmo se encontrava desbalanceado, alterando demais os resultados.<br> 

## Foram utilizadas algumas técnicas de balanceamento: <br><br>
## Under Sampler: <br> 
- Random Under Sampler; <br> 
- ClusterCentroids; <br>
- NearMiss; <br>
## Over Sampler: <br>
- Random Over Sampler; <br> 
- SMOTE; <br>
- ADASYN; <br>
## Combinação de métodos (Under Sampler e Over Sampler): <br>
- SMOTEENN; <br><br>

Para escolher o melhor algoritmo de Machine Learning utilizamos:<br>
- roc_curve; <br>
- roc_auc_score; <br>
- precision_recall_curve; <br>
- precision_recall_auc_score <br>

Por fim, também foram utilizados vários hiperparâmetros através do GridSearchCV para se chegar ao melhor resultado.
