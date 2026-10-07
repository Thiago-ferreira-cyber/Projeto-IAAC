# IAAC-Repo

Projeto de classificação de executáveis PE em benignos e maliciosos, organizado segundo as etapas CRISP-ML.

## Estrutura

- `dataset/raw/`: CSV de entrada.
- `src/`: funções reutilizáveis para cada etapa.
- `notebooks/`: exploração, preparação, modeling e avaliação/otimização, por ordem.
- `models/`: pipelines treinados, metadados e relatórios gerados.

## Configuração

Requer Python 3.10 ou superior. Crie um ambiente virtual e instale as dependências listadas:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Abra os notebooks com Jupyter a partir de `notebooks/`, ou execute as funções de `src/` com os caminhos relativos à raiz do repositório. A sequência recomendada é `01_data_understanding.ipynb`, `02_data_preparation.ipynb`, `03_modeling.ipynb` e `04_model_evaluation_optimization.ipynb`.

## Dados e avaliação

O pipeline divide os dados de forma estratificada em 70% treino, 15% validação e 15% teste. A comparação inicial dos modelos usa validação cruzada no treino. Na etapa de Evaluation & Optimization, os hiperparâmetros são afinados por validação cruzada apenas no treino; a validação escolhe o limiar da classe maliciosa; e o teste fica reservado para a avaliação final.

A comparação está implementada em `src/data_modelling.py`. A afinação, otimização do limiar e avaliação final estão em `src/data_evaluation.py`. Os modelos, metadados e gráficos são guardados em `models/`. Os ficheiros `*_latest` são atualizados a cada execução; os ficheiros com timestamp preservam execuções anteriores.
