"""
Funções para a fase de Modeling (CRISP-ML).
Treina e compara vários modelos de classificação, afina o melhor,
avalia overfitting, guarda métricas/CSV e o pipeline final (com metadados)
para ser usado depois pelos agentes.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, ParameterGrid
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, get_scorer

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _reserve_artifact_path(output_dir: str, filename: str) -> str:
    """Reserva um nome de artefacto livre para impedir que uma execu??o o substitua."""
    stem, extension = os.path.splitext(filename)
    suffix = 1
    while True:
        candidate_name = filename if suffix == 1 else f"{stem}_{suffix}{extension}"
        candidate_path = os.path.join(output_dir, candidate_name)
        try:
            descriptor = os.open(candidate_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            suffix += 1
            continue
        os.close(descriptor)
        return candidate_path


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


def _resolve_output_dir(output_dir: str) -> Path:
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parents[1] / output_path
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_results_csv(
    results_df: pd.DataFrame,
    dataset_name: str = "dataset",
    output_dir: str = "../models",
    timestamp: str = None,
) -> str:
    """Guarda a tabela de comparação de modelos em CSV, sem sobrescrever execuções anteriores."""
    output_path = _resolve_output_dir(output_dir)
    timestamp = timestamp or _timestamp()
    file_path = output_path / f"comparacao_modelos_{dataset_name}_{timestamp}.csv"
    results_df.to_csv(file_path, index=False)
    print(f"Comparação de modelos guardada em: {file_path}")
    return str(file_path)


def plot_model_comparison(
    results_df: pd.DataFrame,
    scoring: str = "f1_macro",
    dataset_name: str = "dataset",
    output_dir: str = "../models",
    timestamp: str = None,
) -> str:
    """Guarda um gráfico de barras a comparar os modelos pela métrica escolhida, sem sobrescrever execuções anteriores."""
    output_path = _resolve_output_dir(output_dir)
    timestamp = timestamp or _timestamp()
    save_path = _reserve_artifact_path(str(output_path), f"comparacao_modelos_{dataset_name}_{timestamp}.png")

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
    X_val: pd.DataFrame,
    y_val: pd.Series,
    tune: bool = True,
    scoring: str = "f1_macro",
) -> Pipeline:
    """Escolhe hiperpar?metros na valida??o e devolve o pipeline ajustado no treino."""
    param_grid = PARAM_GRIDS.get(model_name, {}) if tune else {}
    candidates = list(ParameterGrid(param_grid)) if param_grid else [{}]
    scorer = get_scorer(scoring)
    best_score = float("-inf")
    best_params = {}

    for params in candidates:
        model, needs_scaling = get_models()[model_name]
        candidate = build_pipeline(model, needs_scaling)
        candidate.set_params(**params)
        candidate.fit(X_train, y_train)
        score = scorer(candidate, X_val, y_val)
        if score > best_score:
            best_score, best_params = score, params

    model, needs_scaling = get_models()[model_name]
    best_pipe = build_pipeline(model, needs_scaling)
    best_pipe.set_params(**best_params)
    best_pipe.fit(X_train, y_train)
    print(f"Melhores hiperpar?metros para {model_name}: {best_params}")
    print(f"{scoring} na valida??o: {best_score:.4f}")
    return best_pipe


def refit_final_model(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple:
    """Ajusta o pipeline escolhido com treino + valida??o antes do teste final."""
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)
    final_pipe = clone(pipe)
    final_pipe.fit(X_train_val, y_train_val)
    return final_pipe, X_train_val, y_train_val


# ==========================================

def evaluate_on_validation(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scoring: str = "f1_macro",
) -> dict:
    """Regista desempenho de treino/valida??o antes de reajustar o modelo final."""
    scorer = get_scorer(scoring)
    train_score = scorer(pipe, X_train, y_train)
    validation_score = scorer(pipe, X_val, y_val)
    gap = train_score - validation_score
    print(f"\n--- Treino vs valida??o ({scoring}) ---")
    print(f"Treino: {train_score:.4f} | Valida??o: {validation_score:.4f} | Diferen?a: {gap:.4f}")
    return {
        "scoring": scoring,
        "train_score": round(float(train_score), 4),
        "validation_score": round(float(validation_score), 4),
        "overfitting_gap": round(float(gap), 4),
        "best_params": pipe.named_steps["model"].get_params(),
    }


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

    print("\n--- Classification report (teste) ---")
    print(classification_report(y_test, y_pred_test))

    scorer = get_scorer(scoring)
    train_score = scorer(pipe, X_train, y_train)
    test_score = scorer(pipe, X_test, y_test)

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
    save_path = _reserve_artifact_path(output_dir, f"matriz_confusao_{dataset_name}_{timestamp}.png")
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
    """Guarda o pipeline e seus metadados em versões persistentes e latest."""
    output_path = _resolve_output_dir(output_dir)
    safe_name = model_name.lower().replace(" ", "_")
    timestamp = timestamp or _timestamp()

    base_versioned = f"{dataset_name}_{safe_name}_{timestamp}"
    base_latest = f"{dataset_name}_{safe_name}_latest"

    versioned_path = output_path / f"{base_versioned}.joblib"
    latest_path = output_path / f"{base_latest}.joblib"
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
    meta_versioned_path = output_path / f"{base_versioned}.json"
    meta_latest_path = output_path / f"{base_latest}.json"
    with open(meta_versioned_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
    with open(meta_latest_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

    print(f"Modelo guardado em: {versioned_path}")
    print(f"Metadados guardados em: {meta_versioned_path}")
    print(f"Cópias 'latest' atualizadas em: {latest_path} / {meta_latest_path}")
    return str(versioned_path)


# ==========================================
# ORQUESTRAÇÃO (uso como script)
# ==========================================

def run_modeling(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    dataset_name: str = "dataset",
    scoring: str = "f1_macro",
    output_dir: str = "../models",
    tune: bool = True,
) -> Pipeline:
    """Compara, afina na valida??o, reajusta em treino+valida??o e avalia no teste."""
    timestamp = _timestamp()
    results_df = evaluate_models(X_train, y_train, scoring=scoring)
    save_results_csv(results_df, dataset_name=dataset_name, output_dir=output_dir, timestamp=timestamp)
    plot_model_comparison(results_df, scoring=scoring, dataset_name=dataset_name, output_dir=output_dir, timestamp=timestamp)

    best_name = select_best_model(results_df)
    tuned_pipe = train_final_model(best_name, X_train, y_train, X_val, y_val, tune=tune, scoring=scoring)
    validation_metrics = evaluate_on_validation(tuned_pipe, X_train, y_train, X_val, y_val, scoring=scoring)
    best_pipe, X_train_val, y_train_val = refit_final_model(tuned_pipe, X_train, y_train, X_val, y_val)
    test_metrics = evaluate_on_test(
        best_pipe, X_train_val, y_train_val, X_test, y_test,
        scoring=scoring, dataset_name=dataset_name, output_dir=output_dir, timestamp=timestamp,
    )
    metrics = {**validation_metrics, **{f"test_{key}": value for key, value in test_metrics.items()}}
    metrics["cv_score_medio"] = round(float(results_df.iloc[0]["media"]), 4)
    metrics["cv_score_desvio_padrao"] = round(float(results_df.iloc[0]["desvio_padrao"]), 4)
    save_model(
        best_pipe, best_name, dataset_name=dataset_name, output_dir=output_dir,
        metrics=metrics, feature_names=list(X_train.columns), timestamp=timestamp,
    )
    print("\n--- Modeling conclu?do ---")
    return best_pipe

if __name__ == "__main__":
    import data_preparation as dp

    X_train, X_val, X_test, y_train, y_val, y_test = dp.run_data_preparation(
        "dataset/raw/MalwareMemoryDump.csv",
        target="Label",
    )
    run_modeling(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        dataset_name="malware_memory_dump",
    )