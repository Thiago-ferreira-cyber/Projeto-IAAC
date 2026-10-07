"""
Funcoes para a fase de Model Evaluation & Optimization (CRISP-ML).
Afina hiperparametros por validacao cruzada, otimiza o limiar na validacao
e calcula metricas finais no conjunto de teste.
"""

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import data_modelling as dm


CLASS_NAMES = {0: "Benigno", 1: "Malicioso"}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _reserve_artifact_path(output_dir: str, filename: str) -> str:
    """Reserva um nome de artefacto livre para preservar execucoes anteriores."""
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


def tune_hyperparameters(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = "f1_macro",
    n_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> GridSearchCV:
    """Afina o modelo selecionado usando apenas CV estratificada no treino."""
    if model_name not in dm.get_models():
        raise ValueError(f"Modelo desconhecido: {model_name}")

    model, needs_scaling = dm.get_models()[model_name]
    pipeline = dm.build_pipeline(model, needs_scaling)
    param_grid = dm.PARAM_GRIDS.get(model_name, {})
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    print(f"Melhores hiperparametros por CV ({model_name}): {search.best_params_}")
    print(f"{scoring} medio na CV: {search.best_score_:.4f}")
    return search


def get_positive_class_scores(estimator, X: pd.DataFrame, positive_label=1) -> tuple:
    """Obtém probabilidades ou scores de decisão orientados para a classe positiva."""
    classes = np.asarray(estimator.classes_)
    positive_indices = np.flatnonzero(classes == positive_label)
    if len(positive_indices) != 1:
        raise ValueError(f"A classe positiva {positive_label!r} nao existe no modelo.")

    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(X)[:, positive_indices[0]]
        return np.asarray(scores, dtype=float), "probabilidade"

    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(X), dtype=float)
        if scores.ndim == 1:
            if classes[1] != positive_label:
                scores = -scores
        else:
            scores = scores[:, positive_indices[0]]
        return scores, "score de decisao"

    raise TypeError("O estimador tem de disponibilizar predict_proba ou decision_function.")


def optimize_threshold(
    y_true: pd.Series,
    positive_scores: np.ndarray,
    positive_label=1,
) -> dict:
    """Seleciona na validacao o limiar que maximiza F1 da classe positiva."""
    if positive_label not in (0, 1):
        raise ValueError("As classes têm de estar codificadas como 0 e 1.")
    y_binary = np.asarray(y_true) == positive_label
    if not y_binary.any() or y_binary.all():
        raise ValueError("A validacao tem de conter exemplos das duas classes.")

    precision, recall, thresholds = precision_recall_curve(
        y_binary, positive_scores, pos_label=True,
    )
    if len(thresholds) == 0:
        raise ValueError("Nao foi possivel calcular limiares a partir dos scores.")

    precision_at_threshold = precision[:-1]
    recall_at_threshold = recall[:-1]
    denominator = precision_at_threshold + recall_at_threshold
    f1 = np.divide(
        2 * precision_at_threshold * recall_at_threshold,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.argmax(f1))
    curve = pd.DataFrame({
        "limiar": thresholds,
        "precisao_malware": precision_at_threshold,
        "recall_malware": recall_at_threshold,
        "f1_malware": f1,
    })
    selected = {
        "threshold": float(thresholds[best_index]),
        "precision": float(precision_at_threshold[best_index]),
        "recall": float(recall_at_threshold[best_index]),
        "f1": float(f1[best_index]),
    }
    print(
        "Limiar escolhido na validacao: "
        f"{selected['threshold']:.6f} | F1 malware={selected['f1']:.4f} | "
        f"recall malware={selected['recall']:.4f}"
    )
    return {"selected": selected, "curve": curve}


def _metrics_at_threshold(y_true, scores, threshold: float, positive_label=1) -> dict:
    y_true = np.asarray(y_true)
    if positive_label not in (0, 1):
        raise ValueError("As classes têm de estar codificadas como 0 e 1.")
    y_pred = np.where(scores >= threshold, positive_label, 1 - positive_label)
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    positive_metrics = report[CLASS_NAMES[positive_label]]
    positive_index = int(positive_label)
    negative_index = 1 - positive_index
    return {
        "threshold": float(threshold),
        "accuracy": float(report["accuracy"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "precision_malware": float(positive_metrics["precision"]),
        "recall_malware": float(positive_metrics["recall"]),
        "f1_malware": float(positive_metrics["f1-score"]),
        "false_negatives": int(matrix[positive_index, negative_index]),
        "false_positives": int(matrix[negative_index, positive_index]),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }


def evaluate_test_thresholds(
    estimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    selected_threshold: float,
    score_type: str,
    positive_label=1,
    dataset_name: str = "malware",
    output_dir: str = "../models",
    timestamp: str = None,
) -> dict:
    """Compara no teste o limiar padrao com o limiar escolhido na validacao."""
    y_test_array = np.asarray(y_test)
    if positive_label not in (0, 1) or set(np.unique(y_test_array)) != {0, 1}:
        raise ValueError("A avaliacao atual espera classes binárias codificadas como 0 e 1.")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = timestamp or _timestamp()
    scores, actual_score_type = get_positive_class_scores(estimator, X_test, positive_label)
    if score_type != actual_score_type:
        raise ValueError("O tipo de score da validacao nao corresponde ao do modelo.")

    default_threshold = 0.5 if score_type == "probabilidade" else 0.0
    baseline = _metrics_at_threshold(y_test_array, scores, default_threshold, positive_label)
    optimized = _metrics_at_threshold(y_test_array, scores, selected_threshold, positive_label)
    optimized["average_precision_malware"] = float(
        average_precision_score(y_test_array == positive_label, scores)
    )

    figure, axis = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=np.asarray(optimized["confusion_matrix"]),
        display_labels=[CLASS_NAMES[0], CLASS_NAMES[1]],
    )
    display.plot(cmap="Blues", ax=axis, colorbar=False)
    axis.set_title(f"Matriz de confusao — {dataset_name} (teste, limiar otimizado)")
    figure.tight_layout()
    matrix_path = _reserve_artifact_path(
        output_dir, f"matriz_confusao_{dataset_name}_{timestamp}.png"
    )
    figure.savefig(matrix_path, bbox_inches="tight")
    plt.close(figure)

    return {
        "score_type": score_type,
        "baseline": baseline,
        "optimized": optimized,
        "confusion_matrix_path": matrix_path,
    }


def save_evaluation_artifacts(
    metrics: dict,
    threshold_curve: pd.DataFrame,
    dataset_name: str = "malware",
    output_dir: str = "../models",
    timestamp: str = None,
) -> dict:
    """Guarda curva de limiar e resultados finais em CSV, PNG e JSON."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = timestamp or _timestamp()

    curve_path = _reserve_artifact_path(
        output_dir, f"curva_limiar_{dataset_name}_{timestamp}.csv"
    )
    threshold_curve.to_csv(curve_path, index=False)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(threshold_curve["limiar"], threshold_curve["precisao_malware"], label="Precisao malware")
    axis.plot(threshold_curve["limiar"], threshold_curve["recall_malware"], label="Recall malware")
    axis.plot(threshold_curve["limiar"], threshold_curve["f1_malware"], label="F1 malware")
    selected_threshold = metrics["threshold_selection"]["threshold"]
    axis.axvline(selected_threshold, color="black", linestyle="--", label="Limiar escolhido")
    axis.set_xlabel("Limiar de decisao")
    axis.set_ylabel("Metrica")
    axis.set_title(f"Otimizacao do limiar — {dataset_name} (validacao)")
    axis.legend()
    figure.tight_layout()
    plot_path = _reserve_artifact_path(
        output_dir, f"otimizacao_limiar_{dataset_name}_{timestamp}.png"
    )
    figure.savefig(plot_path, bbox_inches="tight")
    plt.close(figure)

    json_path = _reserve_artifact_path(
        output_dir, f"avaliacao_{dataset_name}_{timestamp}.json"
    )
    with open(json_path, "w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, ensure_ascii=False, indent=2, default=str)

    print(f"Curva de limiar guardada em: {curve_path}")
    print(f"Grafico de limiar guardado em: {plot_path}")
    print(f"Relatorio de avaliacao guardado em: {json_path}")
    return {"threshold_curve": curve_path, "threshold_plot": plot_path, "report": json_path}
