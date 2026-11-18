"""
Funções para a fase de Data Understanding (CRISP-ML).
Pensado para ser importado quer por um script, quer por um notebook.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# CARREGAMENTO
# ==========================================

def load_data(path: str) -> pd.DataFrame:
    """Carrega o dataset a partir de um ficheiro CSV."""
    dataset_path = Path(path)
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).resolve().parents[1] / dataset_path
    df = pd.read_csv(dataset_path)
    print("Dimensões do dataset:", df.shape)
    print("\nColunas:")
    print(df.columns.tolist())
    return df


# ==========================================
# VISÃO GERAL
# ==========================================

def overview(df: pd.DataFrame) -> None:
    """Mostra uma primeira vista sobre os dados: head, tipos e estatísticas."""
    print("\n--- Primeiras linhas ---")
    print(df.head())

    print("\n--- Tipos de dados ---")
    print(df.dtypes)

    print("\n--- Estatísticas descritivas ---")
    print(df.describe())


def check_missing_duplicates(df: pd.DataFrame) -> None:
    """Verifica valores nulos e linhas duplicadas."""
    print("\n--- Valores nulos ---")
    print(df.isnull().sum())

    print("\n--- Duplicados ---")
    print("Número de duplicados:", df.duplicated().sum())


# ==========================================
# VARIÁVEL-ALVO
# ==========================================

def class_distribution(df: pd.DataFrame, target: str) -> None:
    """Imprime a contagem e a percentagem de cada classe do target."""
    print(f"\n--- Distribuição das classes ({target}) ---")
    print(df[target].value_counts())
    print(df[target].value_counts(normalize=True).round(3) * 100)


def plot_class_distribution(df: pd.DataFrame, target: str, save_path: str = "distribuicao_classes.png") -> None:
    """Guarda um gráfico de barras com a distribuição das classes."""
    output_path = Path(save_path)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parents[1] / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(5, 4))
    sns.countplot(x=target, data=df)
    plt.title("Distribuição das classes")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Gráfico guardado em: {output_path}")


# ==========================================
# CORRELAÇÕES
# ==========================================

def plot_correlation_matrix(df: pd.DataFrame, save_path: str = "matriz_correlacao.png") -> pd.DataFrame:
    """Guarda o heatmap de correlação entre todas as features numéricas e devolve a matriz."""
    numeric_df = df.select_dtypes(include="number")
    output_path = Path(save_path)
    if not output_path.is_absolute():
        output_path = Path(__file__).resolve().parents[1] / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 10))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 7})
    plt.title("Matriz de correlação")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Gráfico guardado em: {output_path}")

    return numeric_df.corr()


def target_correlation(df: pd.DataFrame, target: str) -> pd.Series:
    """Devolve a correlação de cada feature numérica com o target, ordenada."""
    numeric_df = df.select_dtypes(include="number").copy()
    encoded_target = (df[target] != "Benign").astype(int)
    numeric_df["_target_"] = encoded_target
    corr = numeric_df.corr()["_target_"].drop("_target_").sort_values(ascending=False)
    print(f"\n--- Correlação de cada feature com {target} ---")
    print(corr)
    return corr


# ==========================================
# DISTRIBUIÇÕES POR FEATURE
# ==========================================

def plot_feature_distributions(df: pd.DataFrame, features: list, target: str, save_dir: str = ".") -> None:
    """Guarda um histograma por feature, separado por classe do target."""
    features = [f for f in features if f in df.columns]
    output_dir = Path(save_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parents[1] / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for feature in features:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x=feature, hue=target, bins=30, kde=True)
        plt.title(f"Distribuição de {feature} por classe")
        save_path = output_dir / f"dist_{feature}.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"Gráfico guardado em: {save_path}")


# ==========================================
# ORQUESTRAÇÃO (uso como script)
# ==========================================

def run_data_understanding(
    path: str,
    target: str = "Label",
    features_to_plot: list = None,
) -> pd.DataFrame:
    """Executa a fase de Data Understanding com o dataset atual."""
    if features_to_plot is None:
        features_to_plot = [
            "dlllist_avg_dlls_per_proc",
            "handles_nevent",
            "svcscan_nservices",
            "svcscan_kernel_drivers",
        ]

    df = load_data(path)
    overview(df)
    check_missing_duplicates(df)
    class_distribution(df, target)
    plot_class_distribution(df, target, "models/distribuicao_classes.png")
    plot_correlation_matrix(df, "models/matriz_correlacao.png")
    target_correlation(df, target)
    plot_feature_distributions(df, features_to_plot, target, "models")

    print("\n--- Data Understanding concluído ---")
    return df


if __name__ == "__main__":
    run_data_understanding("dataset/raw/MalwareMemoryDump.csv")