README
# Projeto IAAC — Cybersecurity Attack Classification

## Descrição

Este projeto tem como objetivo desenvolver um modelo de Machine Learning capaz de classificar cenários de cibersegurança em diferentes categorias de ataque com base em informação textual.

O projeto segue uma abordagem baseada nas etapas de Machine Learning estudadas na unidade curricular, incluindo:

- Business Understanding
- Data Understanding / Exploratory Data Analysis (EDA)
- Data Preparation
- Supervised Learning
- Model Evaluation

---

## Objetivo

O problema foi definido como um problema de **classificação multiclasse**.

A variável alvo selecionada é:

`Category`

O objetivo é investigar se as características textuais de um cenário de cibersegurança permitem prever corretamente a categoria à qual pertence.

---

# Sprint 1 — Data Understanding / EDA

O ficheiro:

`eda.py`

é responsável pela análise exploratória do dataset.

### Análises realizadas

- Dimensão do dataset
- Tipos de dados
- Valores em falta
- Valores duplicados
- Valores únicos por coluna
- Análise da variável `Category`
- Distribuição das categorias
- Análise do desequilíbrio entre classes
- Análise dos tipos de alvo
- Análise das técnicas MITRE ATT&CK
- Análise das fontes dos dados
- Análise do comprimento dos campos textuais

Também foram produzidas visualizações para compreender melhor a distribuição dos dados.

---

# Sprint 2 — Data Preparation

O ficheiro:

`data_preparation.py`

é responsável pela preparação dos dados antes do treino dos modelos de Machine Learning.

### Data Cleaning

Foram realizadas operações como:

- Remoção de colunas residuais do CSV (`Unnamed`)
- Remoção de linhas completamente vazias
- Verificação e remoção de duplicados
- Limpeza de strings
- Tratamento de valores em falta

### Feature Selection

Foram selecionadas características textuais relevantes para o problema de classificação, incluindo:

- Title
- Scenario Description
- Tools Used
- Attack Steps
- Target Type
- Vulnerability
- MITRE Technique
- Impact
- Detection Method
- Tags

A coluna `ID` não é utilizada como feature por funcionar apenas como identificador.

---

## Feature Engineering

As diferentes características textuais são combinadas numa nova feature:

`Combined_Text`

Também são criadas características auxiliares:

- `Text_Length`
- `Word_Count`

---

## Tratamento de classes raras

O dataset original contém **64 categorias** com uma distribuição bastante desigual.

Para o primeiro experimento de classificação são consideradas categorias com pelo menos:

`100 amostras`

Esta decisão reduz o impacto de classes extremamente raras na fase inicial de treino e avaliação.

---

## Train/Test Split

Os dados são separados em:

- 80% Training
- 20% Testing

É utilizado `stratify=y` para preservar aproximadamente a distribuição das categorias nos conjuntos de treino e teste.

---

## Feature Extraction — TF-IDF

Como os dados utilizados são principalmente textuais, é utilizado **TF-IDF (Term Frequency-Inverse Document Frequency)** para transformar texto em representações numéricas que possam ser utilizadas pelos algoritmos de Machine Learning.

Configuração inicial:

- Unigrams e Bigrams
- Máximo de 20.000 features
- Remoção de stopwords inglesas
- `min_df = 2`
- `max_df = 0.95`

O TF-IDF é ajustado apenas aos dados de treino para evitar **data leakage**.

---

# Estrutura atual

```text
Projeto-IAAC/
│
├── cyber_dataset/
│   └── Attack_Dataset.csv
│
├── eda.py
├── data_preparation.py
├── .gitignore
└── README.md
```

---

# Próxima etapa

A próxima fase do projeto corresponde ao módulo de **Supervised Learning**.

Serão treinados e comparados diferentes algoritmos de classificação, utilizando métricas como:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Os resultados serão utilizados para analisar o comportamento dos diferentes modelos no problema de classificação de cenários de cibersegurança.

---

## Tecnologias

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Git / GitHub
