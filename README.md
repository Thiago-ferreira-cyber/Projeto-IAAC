# IAAC-Repo

Projeto de classifica??o de execut?veis PE em benignos e maliciosos. O fluxo est? organizado em tr?s etapas: explora??o, prepara??o dos dados e compara??o/avalia??o de modelos.

## Estrutura

- `dataset/raw/`: CSV de entrada.
- `src/`: fun??es reutiliz?veis para cada etapa.
- `notebooks/`: explora??o, prepara??o e modela??o, por ordem.
- `models/`: pipelines treinados, metadados e relat?rios gerados.

## Configura??o

Requer Python 3.10 ou superior. Crie um ambiente virtual e instale as depend?ncias listadas:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Abra os notebooks com Jupyter a partir de `notebooks/`, ou execute as fun??es de `src/` com os caminhos relativos ? raiz do reposit?rio. A sequ?ncia recomendada ? `01_data_understanding.ipynb`, `02_data_preparation.ipynb` e `03_modeling.ipynb`.

## Dados e avalia??o

O pipeline divide os dados de forma estratificada em 70% treino, 15% validacao e 15% teste. A comparacao dos modelos usa validacao cruzada nos dados de treino; a afinacao de hiperparametros usa o conjunto de validacao. O modelo final e reajustado com treino + validacao e avaliado uma vez no teste.

A compara??o e avalia??o s?o implementadas em `src/data_modelling.py`. Modelos, metadados e gr?ficos s?o guardados em `models/`. Os ficheiros `*_latest` s?o substitu?dos a cada execu??o; os ficheiros com timestamp preservam execu??es anteriores.
