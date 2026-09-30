"""
Funções para a fase de Modeling (CRISP-ML).
Treina e compara vários modelos de classificação, escolhe o melhor
e guarda-o em disco para ser usado depois pelos agentes.
"""

import os
import joblib
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


# ==========================================
# MODELOS A COMPARAR
# ==========================================

def get_models() -> dict:
    """
    Devolve os modelos candidatos. O booleano indica se o modelo
    precisa de normalização (True) ou não (False, modelos baseados em árvores).
    """
    return {
        "Regressão Logística": (LogisticRegression(max_iter=1000, random_state=42), True),
        "kNN":                 (KNeighborsClassifier(), True),
        "SVM":                 (SVC(random_state=42), True),
        "Naive Bayes":         (GaussianNB(), True),
        "MLP":                 (MLPClassifier(max_iter=500, random_state=42), True),
        "Árvore de Decisão":   (DecisionTreeClassifier(random_state=42), False),
        "Random Forest":       (RandomForestClassifier(random_state=42), False),
        "Gradient Boosting":   (GradientBoostingClassifier(random_state=42), False),
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
) -> pd.DataFrame:
    """
    Avalia todos os modelos candidatos com validação cruzada estratificada.
    Devolve um dataframe ordenado pela média da métrica escolhida.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []

    for name, (model, needs_scaling) in get_models().items():
        pipe = build_pipeline(model, needs_scaling)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring)
        results.append({"modelo": name, "media": scores.mean(), "desvio_padrao": scores.std()})
        print(f"{name:22s} | {scoring} = {scores.mean():.4f} ± {scores.std():.4f}")

    results_df = pd.DataFrame(results).sort_values("media", ascending=False).reset_index(drop=True)
    return results_df


def plot_model_comparison(results_df: pd.DataFrame, scoring: str = "f1_macro", save_path: str = "comparacao_modelos.png") -> None:
    """Guarda um gráfico de barras a comparar os modelos pela métrica escolhida."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(results_df["modelo"], results_df["media"], xerr=results_df["desvio_padrao"])
    ax.invert_yaxis()  # melhor modelo no topo, já que results_df vem ordenado por "media" desc
    ax.set_xlabel(scoring)
    ax.set_title("Comparação de modelos (validação cruzada)")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Gráfico guardado em: {save_path}")


# ==========================================
# MODELO FINAL
# ==========================================

def select_best_model(results_df: pd.DataFrame) -> str:
    """Devolve o nome do melhor modelo (primeira linha do dataframe ordenado)."""
    best_name = results_df.iloc[0]["modelo"]
    print(f"\nMelhor modelo: {best_name}")
    return best_name


def train_final_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Treina o pipeline do modelo escolhido em todo o conjunto de treino."""
    model, needs_scaling = get_models()[model_name]
    pipe = build_pipeline(model, needs_scaling)
    pipe.fit(X_train, y_train)
    return pipe


def evaluate_on_test(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, save_path: str = "matriz_confusao.png") -> None:
    """Avalia o modelo final no conjunto de teste: relatório + matriz de confusão."""
    y_pred = pipe.predict(X_test)

    print("\n--- Classification report (teste) ---")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Matriz de confusão (teste)")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Gráfico guardado em: {save_path}")


# ==========================================
# GUARDAR MODELO
# ==========================================

def save_model(pipe: Pipeline, model_name: str, dataset_name: str = "dataset", output_dir: str = "../models") -> str:
    """
    Guarda o pipeline treinado em <output_dir>/<dataset>_<modelo>_<timestamp>.joblib.
    Por omissão usa '../models', assumindo que o código está numa pasta
    (ex.: 'src/') e a pasta 'models' já existe ao lado dela, no repositório.

    Cada chamada gera um ficheiro novo (com timestamp), para que voltar a
    correr o notebook/script de Modeling nunca apague nem sobrescreva os
    modelos guardados em execuções anteriores.

    Também é atualizada uma cópia sem timestamp, "<dataset>_<modelo>_latest.joblib",
    para os agentes carregarem sempre a versão mais recente sem terem de saber o timestamp.
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    versioned_path = os.path.join(output_dir, f"{dataset_name}_{safe_name}_{timestamp}.joblib")
    latest_path = os.path.join(output_dir, f"{dataset_name}_{safe_name}_latest.joblib")

    joblib.dump(pipe, versioned_path)
    joblib.dump(pipe, latest_path)

    print(f"Modelo guardado em: {versioned_path}")
    print(f"Cópia 'latest' atualizada em: {latest_path}")
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
) -> Pipeline:
    """Corre a fase de Modeling do início ao fim: compara, escolhe, treina, avalia e guarda."""
    results_df = evaluate_models(X_train, y_train, scoring=scoring)
    plot_model_comparison(results_df, scoring=scoring)

    best_name = select_best_model(results_df)
    best_pipe = train_final_model(best_name, X_train, y_train)

    evaluate_on_test(best_pipe, X_test, y_test)
    save_model(best_pipe, best_name, dataset_name=dataset_name, output_dir=output_dir)

    print("\n--- Modeling concluído ---")
    return best_pipe


if __name__ == "__main__":
    import data_preparation as dp

    X_train, X_test, y_train, y_test = dp.run_data_preparation("Malware_and_benign_recognition.csv")
    run_modeling(X_train, X_test, y_train, y_test, dataset_name="malware")