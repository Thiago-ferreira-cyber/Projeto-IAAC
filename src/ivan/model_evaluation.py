
"""Avaliação dos modelos de classificação."""

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def evaluate_model(model, X_test, y_test, output_dir="models",
                   dataset_name="cybersecurity_attacks",
                   model_name="Linear SVM", save_plots=True):

    # Fazer previsões
    y_pred = model.predict(X_test)

    # Calcular métricas
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_test, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_test, y_pred, average="macro", zero_division=0
        )
    }

    print("\nResultados no teste:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Guardar resultados
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    name = model_name.lower().replace(" ", "_")
    prefix = f"{dataset_name}_{name}"

    with open(output / f"{prefix}_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # Matriz de confusão
    labels = model.classes_
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        output / f"{prefix}_confusion_matrix.csv"
    )

    if save_plots:
        fig, ax = plt.subplots(figsize=(16, 14))
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=labels
        ).plot(ax=ax, xticks_rotation=90, colorbar=False)

        ax.set_title(f"Matriz de confusão - {model_name}")
        plt.tight_layout()
        plt.savefig(output / f"{prefix}_confusion_matrix.png")
        plt.close()

    return metrics
