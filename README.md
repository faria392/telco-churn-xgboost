# Previsão de Churn de Clientes:Telco Customer Churn

Projeto de Machine Learning para prever a evasão (*churn*) de clientes de uma empresa de telecomunicações, a partir de dados cadastrais, de contrato e de consumo de serviços.

## 🎯 Objetivo

Construir um pipeline completo de classificação binária capaz de identificar, com antecedência, quais clientes têm maior probabilidade de cancelar o serviço — permitindo que o time de retenção atue de forma proativa.

## 🗂️ Dataset

- **Fonte:** [Telco Customer Churn (IBM Sample Dataset)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Arquivo:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Dimensões:** 7.043 linhas × 21 colunas
- **Variável-alvo:** `Churn` (Yes/No) — desbalanceamento moderado (~26,5% de churn)
- **Variáveis:** dados demográficos (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), tempo de contrato (`tenure`), serviços contratados (`PhoneService`, `InternetService`, `OnlineSecurity`, `StreamingTV` etc.), forma de contrato e pagamento (`Contract`, `PaymentMethod`, `PaperlessBilling`) e cobrança (`MonthlyCharges`, `TotalCharges`)

## 🧠 Metodologia

O notebook segue um pipeline de ponta a ponta, documentado célula a célula:

1. **Análise Exploratória (EDA)** — tipos de dados, valores ausentes, duplicatas, cardinalidade e inconsistências (ex: `TotalCharges` armazenado como texto com espaços em branco).
2. **Limpeza dos dados** — conversão de `TotalCharges` para numérico, remoção de `customerID` (identificador, não-feature), padronização de categorias redundantes (`"No internet service"` → `"No"`).
3. **Split estratificado** — 70% treino / 15% validação / 15% teste, mantendo a proporção de churn em cada partição.
4. **Pipeline de pré-processamento** — `ColumnTransformer` do scikit-learn com imputação de nulos e `OrdinalEncoder` para variáveis categóricas, encapsulado para evitar *data leakage*.
5. **Modelagem** — `XGBoostClassifier`, escolhido por ser o estado da arte para dados tabulares estruturados de porte pequeno/médio.
6. **Tuning de hiperparâmetros** — `RandomizedSearchCV` com validação cruzada estratificada (5 folds, 50 combinações).
7. **Avaliação final** — métricas no conjunto de teste (nunca antes utilizado), incluindo ROC-AUC, matriz de confusão, curva ROC e curva Precision-Recall.
8. **Análise de threshold** — trade-off entre precision e recall para apoiar a decisão de negócio sobre o ponto de corte ideal.

## 📈 Resultados

| Modelo | ROC-AUC (Validação) | ROC-AUC (Teste) |
|---|---|---|
| XGBoost (sem tuning) | 0,8138 | — |
| XGBoost (tunado via RandomizedSearchCV) | 0,8485 | **0,8410** |

**Relatório de classificação (threshold padrão = 0,5) no conjunto de teste:**

| Classe | Precision | Recall | F1-score |
|---|---|---|---|
| Não Churn | 0,90 | 0,74 | 0,81 |
| Churn | 0,52 | 0,78 | 0,62 |

O modelo prioriza **recall** na classe de interesse (Churn) — identifica 78% dos clientes que de fato cancelariam o serviço, o que é geralmente desejável em cenários de retenção, onde o custo de não identificar um cliente em risco tende a ser maior que o de uma ação de retenção "desperdiçada".

## 🛠️ Tecnologias utilizadas

- Python 3
- pandas / numpy
- scikit-learn (`ColumnTransformer`, `Pipeline`, `RandomizedSearchCV`)
- XGBoost
- matplotlib
- Jupyter Notebook

## 🚀 Como executar

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/nome-do-repositorio.git
cd nome-do-repositorio

# Crie um ambiente virtual (opcional, mas recomendado)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt

# Abra o notebook
jupyter notebook churn_pipeline_final.ipynb
```

Certifique-se de que o arquivo `WA_Fn-UseC_-Telco-Customer-Churn.csv` esteja na mesma pasta do notebook.

## 📦 requirements.txt sugerido

```
pandas
numpy
scikit-learn
xgboost
matplotlib
scipy
jupyter
```

## 📁 Estrutura do repositório

```
├── churn_pipeline_final.ipynb   # Notebook principal com todo o pipeline
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset
├── requirements.txt
└── README.md
```

## 🔭 Próximos passos

- Explicabilidade do modelo com SHAP (importância de features por predição)
- Otimização do threshold de decisão com base em custo real de negócio (custo de ação de retenção vs. custo de perda de cliente)
- Deploy do modelo (API ou aplicação simples para consumo pelo time de retenção)
- Monitoramento de *data drift* em produção

## ✍️ Autor

**Gustavo Faria**
Estudante de Ciência de Dados — PUC-Campinas
