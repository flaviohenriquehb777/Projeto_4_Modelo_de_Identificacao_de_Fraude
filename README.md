# Projeto_4_Modelo_de_Identificacao_de_Fraude<br>

### <br> Usando algoritmos de Machine Learning para criar um modelo de identificação de fraude. <br><br>

Para esse modelo, foram utilizados, em princípio, quatro algoritmos de Marchine Learning para analisar os resultados.<br><br>

## Algoritmos utilizados no projeto: <br>

- XGBoost
- Logistic Regression;<br> 
- Random Forest; <br> 
- Support Vector Classifier (SVC). <br><br>

O projeto contou com várias etapas. Estão todas expostas aqui no GitHub, passo a passo até o arquivo final.<br>

O projeto contou uma uma análise exploratória detalhada. Nesse momento, foi observada a necessidade de balancear o dataset. Pois o mesmo se encontrava desbalanceado, alterando demais os resultados.<br> 
Porém, no final do projeto, foi concluído que seria melhor, para obtermos o resultado acordado com a empresa, que não utilizássemos o balanceamento. Isso poderá ser conferido ao longo do projeto.<br> 

## Foram utilizadas algumas técnicas de balanceamento: <br>

### Under-sampling: <br> 
- RandomUnderSampler; <br> 
- ClusterCentroids; <br>
- NearMiss; <br>

### Over-sampling: <br>
- RandomOverSampler; <br> 
- SMOTE. <br><br>

Para escolher o melhor algoritmo de Machine Learning utilizei uma função de minha autoria chamada: testar_modelos_com_undersampling.<br>
Essa função utiliza o RandomUnderSampler como pré-processamento e os seguintes algoritmos de Machine Learning:<br>

- RandomForest;
- DecisionTree;
- LogisticRegression;
- SVC;
- KNN;
- XGBoost;
- ANN.<br><br>

A função retorna os três melhores modelos, em ordem decrescente, com base na área sob a curva precisão-recall (PR AUC).<br>
Por fim, também foram utilizados vários hiperparâmetros através do GridSearchCV para se chegar ao melhor resultado.<br><br>

## CONCLUSÃO:<br>

<br>O modelo de detecção de fraudes foi avaliado utilizando validação cruzada com 10 dobras, resultando nas seguintes métricas médias:<br>

- ### Acurácia: 99,96% 
- ### Precisão: 92,27% 
- ### Recall: 81,02% <br><br>

Dado que a empresa aplicará esse modelo para transações corriqueiras com um valor máximo de R$5.000,00, a principal preocupação é minimizar falsos positivos para evitar reclamações de clientes. <br><br>

## INTERPRETAÇÃO DAS MÉTRICAS:<br>

- ### Alta Precisão (92,27%)<br>
  
Isso indica que a maioria das transações identificadas como fraude realmente são fraudes, o que significa que poucos clientes legítimos terão suas transações bloqueadas indevidamente. Esse é um ponto positivo, pois reduz o impacto negativo sobre a experiência do usuário.<br><br>

- ### Recall Moderado (81,02%)<br>

O modelo consegue capturar 81% das fraudes reais, o que é bom, mas há um 19% de fraudes não detectadas. Como o foco principal é reduzir falsos positivos, esse recall é aceitável dentro do contexto do negócio.<br><br>

- ### Alta Acurácia (99,96%)<br>

Esse valor é muito alto, mas deve ser interpretado com cuidado, pois fraudes são eventos raros. Como o dataset é desbalanceado, a acurácia pode estar sendo inflada pelo grande número de transações legítimas corretamente classificadas.
