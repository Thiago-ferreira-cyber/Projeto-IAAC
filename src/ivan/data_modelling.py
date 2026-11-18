
"""Modeling - Classificação de ataques de cibersegurança."""

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from model_evaluation import evaluate_model


# 1. Algoritmos que vamos comparar
def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, class_weight="balanced"
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5, metric="cosine"
        ),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(
            class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=1
        ),
    }


# 2. TF-IDF + algoritmo
def build_pipeline(model):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            stop_words="english"
        )),
        ("model", model)
    ])


# 3. Comparação com cross-validation
def evaluate_models(X_train, y_train):
    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )

    results = []

    for name, model in get_models().items():
        pipeline = build_pipeline(model)

        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv, scoring="f1_macro", n_jobs=1
        )

        results.append({
            "modelo": name,
            "f1_macro": scores.mean(),
            "desvio_padrao": scores.std()
        })

        print(f"{name}: F1 = {scores.mean():.4f}")

    return pd.DataFrame(results).sort_values(
        "f1_macro", ascending=False
    ).reset_index(drop=True)


# 4. Treino, validação e avaliação final
def run_modeling(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    dataset_name="cybersecurity_attacks",
    output_dir="models"
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Comparar os seis algoritmos
    results = evaluate_models(X_train, y_train)
    results.to_csv(output / "comparacao_modelos.csv", index=False)

    # Selecionar o melhor modelo
    best_name = results.iloc[0]["modelo"]
    print("\nMelhor modelo:", best_name)

    pipeline = build_pipeline(get_models()[best_name])
    pipeline.fit(X_train, y_train)

    # Avaliar na validação
    val_pred = pipeline.predict(X_val)
    val_f1 = f1_score(y_val, val_pred, average="macro")
    print(f"F1 na validação: {val_f1:.4f}")

    # Treinar com treino + validação
    X_final = pd.concat([X_train, X_val])
    y_final = pd.concat([y_train, y_val])
    pipeline.fit(X_final, y_final)

    # Avaliar no teste
    test_metrics = evaluate_model(
        model=pipeline,
        X_test=X_test,
        y_test=y_test,
        output_dir=output_dir,
        dataset_name=dataset_name,
        model_name=best_name
    )

    # Guardar modelo
    name = best_name.lower().replace(" ", "_")
    model_path = output / f"{dataset_name}_{name}_latest.joblib"
    joblib.dump(pipeline, model_path)

    print("\nModelo guardado:", model_path)
    print("F1 no teste:", round(test_metrics["f1_macro"], 4))

    return pipeline


if __name__ == "__main__":
    import data_preparation as dp

    splits = dp.run_data_preparation(
        "dataset/raw/Attack_Dataset.csv"
    )

    run_modeling(*splits)
