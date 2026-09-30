"""
Funções para a fase de Modeling (CRISP-ML).
Treina e compara vários modelos de classificação, afina o melhor,
avalia overfitting, guarda métricas/CSV e o pipeline final (com metadados)
para ser usado depois pelos agentes.
"""

import os
import json
import joblib
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ==========================================
# MODELOS A COMPARAR
# ==========================================

def get_models() -> dict:
    """
    Devolve os modelos candidatos. O booleano indica se o modelo
    precisa de normalização (True) ou não (False: modelos baseados em
    árvores, e Naive Bayes, que não é sensível à escala).
    """
    return {
        "Regressão Logística": (LogisticRegression(max_iter=1000, random_state=42), True),
        "kNN":                 (KNeighborsClassifier(), True),
        "SVM":                 (SVC(random_state=42), True),
        "Naive Bayes":         (GaussianNB(), False),
        "MLP":                 (MLPClassifier(max_iter=500, random_state=42), True),
        "Árvore de Decisão":   (DecisionTreeClassifier(random_state=42), False),
        "Random Forest":       (RandomForestClassifier(random_state=42), False),
        "Gradient Boosting":   (GradientBoostingClassifier(random_state=42), False),
    }


# Grelhas de hiperparâmetros, pequenas para manter o tempo de execução razoável.
# Os nomes dos parâmetros levam o prefixo "model__" porque o estimador vive
# dentro de um Pipeline (passo chamado "model").
PARAM_GRIDS = {
    "Regressão Logística": {"model__C": [0.01, 0.1, 1, 10]},
    "kNN":                 {"model__n_neighbors": [3, 5, 7, 9]},
    "SVM":                 {"model__C": [0.1, 1, 10], "model__kernel": ["rbf", "linear"]},
    "Naive Bayes":         {},  # sem hiperparâmetros relevantes a afinar
    "MLP":                 {"model__hidden_layer_sizes": [(50,), (100,)], "model__alpha": [0.0001, 0.001]},
    "Árvore de Decisão":   {"model__max_depth": [None, 5, 10, 20]},
    "Random Forest":       {"model__n_estimators": [100, 300], "model__max_depth": [None, 10, 20]},
    "Gradient Boosting":   {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1]},
}


def build_pipeline(model, needs_scaling: bool) -> Pipeline:
    """Constrói um Pipeline com (ou sem) normalização, consoante o modelo."""
    steps = []
    if needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


# ==========================================
# COMPARAÇÃO DE MODELOS (VALIDAÇÃO CRUZADA)
# ==========================================

def evaluate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = "f1_macro",
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Avalia todos os modelos candidatos com validação cruzada estratificada.
    Devolve um dataframe ordenado pela média da métrica escolhida.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []

    for name, (model, needs_scaling) in get_models().items():
        pipe = build_pipeline(model, needs_scaling)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=n_jobs)
        results.append({"modelo": name, "media": scores.mean(), "desvio_padrao": scores.std()})
        print(f"{name:22s} | {scoring} = {scores.mean():.4f} ± {scores.std():.4f}")

    results_df = pd.DataFrame(results).sort_values("media", ascending=False).reset_index(drop=True)
    return results_df


def save_results_csv(
    results_df: pd.DataFrame,
    dataset_name: str = "dataset",
    output_dir: str = "../models",
    timestamp: str = None,
) -> str:
    """Guarda a tabela de comparação de modelos em CSV, sem sobrescrever execuções anteriores."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = timestamp or _timestamp()
    file_path = os.path.join(output_dir, f"comparacao_modelos_{dataset_name}_{timestamp}.csv")
    results_df.to_csv(file_path, index=False)
    print(f"Comparação de modelos guardada em: {file_path}")
    return file_path


def plot_model_comparison(
    results_df: pd.DataFrame,
    scoring: str = "f1_macro",
    dataset_name: str = "dataset",
    output_dir: str = "../models",
    timestamp: str = None,
) -> str:
    """Guarda um gráfico de barras a comparar os modelos pela métrica escolhida, sem sobrescrever execuções anteriores."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = timestamp or _timestamp()
    save_path = os.path.join(output_dir, f"comparacao_modelos_{dataset_name}_{timestamp}.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(results_df["modelo"], results_df["media"], xerr=results_df["desvio_padrao"])
    ax.invert_yaxis()  # melhor modelo no topo, já que results_df vem ordenado por "media" desc
    ax.set_xlabel(scoring)
    ax.set_title(f"Comparação de modelos — {dataset_name} (validação cruzada)")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Gráfico guardado em: {save_path}")
    return save_path


# ==========================================
# MODELO FINAL (COM AFINAÇÃO DE HIPERPARÂMETROS)
# ==========================================

def select_best_model(results_df: pd.DataFrame) -> str:
    """Devolve o nome do melhor modelo (primeira linha do dataframe ordenado)."""
    best_name = results_df.iloc[0]["modelo"]
    print(f"\nMelhor modelo: {best_name}")
    return best_name


def train_final_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune: bool = True,
    scoring: str = "f1_macro",
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    """
    Treina o pipeline do modelo escolhido em todo o conjunto de treino.
    Se tune=True (omissão) e houver grelha de hiperparâmetros definida para
    o modelo, afina-os por GridSearchCV antes de ajustar o pipeline final.
    """
    model, needs_scaling = get_models()[model_name]
    pipe = build_pipeline(model, needs_scaling)

    param_grid = PARAM_GRIDS.get(model_name, {})
    if tune and param_grid:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        grid = GridSearchCV(pipe, param_grid, scoring=scoring, cv=cv, n_jobs=n_jobs)
        grid.fit(X_train, y_train)
        print(f"Melhores hiperparâmetros para {model_name}: {grid.best_params_}")
        return grid.best_estimator_

    pipe.fit(X_train, y_train)
    return pipe


# ==========================================
# AVALIAÇÃO (TESTE + DETEÇÃO DE OVERFITTING)
# ==========================================

def evaluate_on_test(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scoring: str = "f1_macro",
    dataset_name: str = "dataset",
    output_dir: str = "../models",
    timestamp: str = None,
) -> dict:
    """
    Avalia o modelo final no conjunto de teste: relatório + matriz de confusão,
    e compara com o desempenho no treino para sinalizar overfitting.
    Devolve um dicionário com as métricas, para ser guardado como metadados.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = timestamp or _timestamp()

    y_pred_test = pipe.predict(X_test)
    y_pred_train = pipe.predict(X_train)

    print("\n--- Classification report (teste) ---")
    print(classification_report(y_test, y_pred_test))

    score_fn = f1_score if scoring == "f1_macro" else None
    if score_fn is not None:
        train_score = f1_score(y_train, y_pred_train, average="macro")
        test_score = f1_score(y_test, y_pred_test, average="macro")
    else:
        train_score = (y_pred_train == y_train).mean()
        test_score = (y_pred_test == y_test).mean()

    gap = train_score - test_score
    print(f"\n--- Overfitting check ({scoring}) ---")
    print(f"Treino: {train_score:.4f} | Teste: {test_score:.4f} | Diferença: {gap:.4f}")
    if gap > 0.05:
        print("Atenção: diferença treino-teste > 0.05 — possível overfitting.")
    else:
        print("Diferença treino-teste pequena — sem sinais fortes de overfitting.")

    cm = confusion_matrix(y_test, y_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Matriz de confusão — {dataset_name} (teste)")
    save_path = os.path.join(output_dir, f"matriz_confusao_{dataset_name}_{timestamp}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Gráfico guardado em: {save_path}")

    return {
        "scoring": scoring,
        "train_score": round(float(train_score), 4),
        "test_score": round(float(test_score), 4),
        "overfitting_gap": round(float(gap), 4),
        "classification_report": classification_report(y_test, y_pred_test, output_dict=True),
    }


# ==========================================
# GUARDAR MODELO (+ METADADOS)
# ==========================================

def save_model(
    pipe: Pipeline,
    model_name: str,
    dataset_name: str = "dataset",
    output_dir: str = "../models",
    metrics: dict = None,
    feature_names: list = None,
    timestamp: str = None,
) -> str:
    """
    Guarda o pipeline treinado em <output_dir>/<dataset>_<modelo>_<timestamp>.joblib,
    mais um ficheiro .json de metadados ao lado (métricas, features, data/hora),
    e uma cópia sem timestamp, "..._latest.joblib" / "..._latest.json".

    Por omissão usa '../models', assumindo que o código está numa pasta
    (ex.: 'src/'/'notebooks/') e a pasta 'models' já existe ao lado dela.

    Cada chamada gera ficheiros novos (com timestamp), para que voltar a
    correr o notebook/script de Modeling nunca apague nem sobrescreva os
    modelos e métricas guardados em execuções anteriores.
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")
    timestamp = timestamp or _timestamp()

    base_versioned = f"{dataset_name}_{safe_name}_{timestamp}"
    base_latest = f"{dataset_name}_{safe_name}_latest"

    versioned_path = os.path.join(output_dir, f"{base_versioned}.joblib")
    latest_path = os.path.join(output_dir, f"{base_latest}.joblib")
    joblib.dump(pipe, versioned_path)
    joblib.dump(pipe, latest_path)

    metadata = {
        "dataset": dataset_name,
        "modelo": model_name,
        "timestamp": timestamp,
        "hiperparametros": pipe.named_steps["model"].get_params(),
        "features": list(feature_names) if feature_names is not None else None,
        "metricas": metrics or {},
    }
    meta_versioned_path = os.path.join(output_dir, f"{base_versioned}.json")
    meta_latest_path = os.path.join(output_dir, f"{base_latest}.json")
    with open(meta_versioned_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
    with open(meta_latest_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

    print(f"Modelo guardado em: {versioned_path}")
    print(f"Metadados guardados em: {meta_versioned_path}")
    print(f"Cópias 'latest' atualizadas em: {latest_path} / {meta_latest_path}")
    return versioned_path


# ==========================================
# ORQUESTRAÇÃO (uso como script)
# ==========================================

def run_modeling(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    dataset_name: str = "dataset",
    scoring: str = "f1_macro",
    output_dir: str = "../models",
    tune: bool = True,
) -> Pipeline:
    """Corre a fase de Modeling do início ao fim: compara, afina, treina, avalia e guarda (com metadados)."""
    timestamp = _timestamp()

    results_df = evaluate_models(X_train, y_train, scoring=scoring)
    save_results_csv(results_df, dataset_name=dataset_name, output_dir=output_dir, timestamp=timestamp)
    plot_model_comparison(results_df, scoring=scoring, dataset_name=dataset_name, output_dir=output_dir, timestamp=timestamp)

    best_name = select_best_model(results_df)
    best_pipe = train_final_model(best_name, X_train, y_train, tune=tune, scoring=scoring)

    metrics = evaluate_on_test(
        best_pipe, X_train, y_train, X_test, y_test,
        scoring=scoring, dataset_name=dataset_name, output_dir=output_dir, timestamp=timestamp,
    )
    metrics["cv_score_medio"] = round(float(results_df.iloc[0]["media"]), 4)
    metrics["cv_score_desvio_padrao"] = round(float(results_df.iloc[0]["desvio_padrao"]), 4)

    save_model(
        best_pipe, best_name, dataset_name=dataset_name, output_dir=output_dir,
        metrics=metrics, feature_names=list(X_train.columns), timestamp=timestamp,
    )

    print("\n--- Modeling concluído ---")
    return best_pipe


if __name__ == "__main__":
    import data_preparation as dp

    X_train, X_test, y_train, y_test = dp.run_data_preparation("../dataset/raw/Malware_and_benign_recognition.csv")
    run_modeling(X_train, X_test, y_train, y_test, dataset_name="malware")