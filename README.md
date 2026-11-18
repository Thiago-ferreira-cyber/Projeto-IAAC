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

## Divisão permanente dos dados

A partição train/validation/test fica registada em `dataset/processed/split_manifest.csv`. O manifesto guarda cada `File`, um SHA-256 das features para manter vetores idênticos na mesma partição, outro hash das features e do alvo para detetar alterações e a partição atribuída. O alvo não é armazenado em claro no manifesto.

Os notebooks 02, 03, 04 e 05 e o relatório Telegram usam este manifesto. A preparação falha se forem adicionadas/removidas amostras ou se as features/rótulos de um ficheiro mudarem; assim, não recalcula silenciosamente o conjunto de teste. O notebook 05 avalia precisamente as linhas marcadas como `test`.

O manifesto foi criado uma vez com divisão estratificada, `random_state=42` e proporções de 70%/15%/15%. Para criar uma divisão nova de forma deliberada, usa `data_preparation.create_split_manifest` indicando um caminho de saída novo, revê o resultado e só depois substitui o manifesto ativo. A função recusa sobrescrever o ficheiro existente.

## Dados e avaliação

O pipeline divide os dados de forma estratificada em 70% treino, 15% validação e 15% teste. A comparação inicial dos modelos usa validação cruzada no treino. Na etapa de Evaluation & Optimization, os hiperparâmetros são afinados por validação cruzada apenas no treino; a validação escolhe o limiar da classe maliciosa; e o teste fica reservado para a avaliação final.

A comparação está implementada em `src/data_modelling.py`. A afinação, otimização do limiar e avaliação final estão em `src/data_evaluation.py`. Os modelos, metadados e gráficos são guardados em `models/`. Os ficheiros `*_latest` são atualizados a cada execução; os ficheiros com timestamp preservam execuções anteriores.

Na preparação, linhas duplicadas em todas as colunas exceto `File` são removidas antes da divisão. A auditoria também assinala features constantes ou idênticas sem as remover, para manter explícito o esquema de features. A versão das regras de preparação fica guardada nos metadados do modelo; depois desta alteração, executa novamente o notebook `04_model_evaluation_optimization.ipynb` antes de gerar o relatório final do teste.

## Relatório de teste via Hermes

O script `src/test_report_telegram.py` valida a proveniência do modelo `malware_random_forest_latest.joblib`, calcula as métricas no conjunto `test` e prepara um relatório Markdown. Os valores são calculados por Python; o Hermes redige o texto final e recebe instrução para o enviar como mensagem através da integração Telegram configurada na própria instalação do Hermes. O script guarda também uma cópia local em `models/`.

Instala e configura o Hermes CLI e a sua integração Telegram fora deste repositório. Copia `.env.example` para `.env` e confirma `REPORT_AGENT=hermes`. Se `hermes` não estiver no `PATH`, define `REPORT_AGENT_EXECUTABLE` com o caminho completo. `REPORT_AGENT_TIMEOUT` controla o limite de espera em segundos. O `.env` está excluído do Git.

Executa a partir da raiz do repositório:

```powershell
python src/test_report_telegram.py
```

O Hermes é chamado em modo não interativo com o relatório calculado via stdin. O script termina com erro se o Hermes falhar ou exceder o limite de espera. Um código de saída bem-sucedido confirma que o Hermes concluiu a tarefa, mas o envio só é possível se a integração Telegram estiver configurada e disponível para o agente.

O modelo `latest` deve ter sido guardado pelo notebook `04_model_evaluation_optimization.ipynb`, treinado apenas na partição `train`, com o limiar selecionado na validação. O script rejeita modelos sem esta proveniência ou que não correspondam ao manifesto ativo.
