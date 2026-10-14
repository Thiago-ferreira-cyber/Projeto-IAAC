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

## Relatório de teste no Telegram

O script `src/test_report_telegram.py` carrega o modelo `malware_random_forest_latest.joblib`, avalia-o no conjunto `test` do manifesto permanente e verifica a proveniência do modelo. Python calcula as métricas; o agente Codex CLI redige a análise final num ficheiro `.md`, que é enviado como documento para o Telegram.

Cria um bot através do BotFather, inicia uma conversa com o bot e preenche `TELEGRAM_BOT_TOKEN` e `TELEGRAM_CHAT_ID` no `.env` da raiz do projeto. O `.env` está excluído pelo `.gitignore`; `.env.example` serve de modelo. Depois executa:

```powershell
python src/test_report_telegram.py
```

Configura `REPORT_AGENT=codex` no `.env`; se necessário, define também
`REPORT_AGENT_EXECUTABLE` com o caminho do executável e
`REPORT_AGENT_TIMEOUT` com o limite de espera em segundos. A sessão Codex CLI
tem de estar autenticada antes da execução.

O modelo `latest` tem de ser o guardado pelo notebook `04_model_evaluation_optimization.ipynb`, que foi ajustado apenas no treino. O script recusa modelos sem metadados de seleção do limiar na validação, para evitar apresentar como avaliação independente um modelo treinado também com validação. O token não deve ser guardado no repositório.
