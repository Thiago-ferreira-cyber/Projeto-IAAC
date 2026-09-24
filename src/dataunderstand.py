"""
Funções para a fase de Data Understanding (CRISP-ML).
Adaptado para o dataset MalwareMemoryDump.csv
(features extraídas com Volatility a partir de dumps de memória RAM).

Colunas-chave do dataset:
    - Raw_Type : identificador da amostra/ficheiro de origem (não é feature)
    - SubType  : Benign / Ransomware / Trojan / Spyware
    - Label    : Benign / Malware  <- variável-alvo (target)
    - 55 features numéricas (pslist_*, dlllist_*, handles_*, ldrmodules_*,
      malfind_*, psxview_*, modules_*, svcscan_*, callbacks_*)

Pensado para ser importado quer por um script, quer por um notebook.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# CARREGAMENTO
# ==========================================

def load_data(path: str) -> pd.DataFrame:
    """Carrega o dataset a partir de um ficheiro CSV."""
    df = pd.read_csv("../dataset/raw/MalwareMemoryDump.csv")
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

def class_distribution(df: pd.DataFrame, target: str = "Label") -> None:
    """Imprime a contagem e a percentagem de cada classe do target.

    Funciona tanto para o target binário (Label: Benign/Malware)
    como para o multiclasse (SubType: Benign/Ransomware/Trojan/Spyware).
    """
    print(f"\n--- Distribuição das classes ({target}) ---")
    print(df[target].value_counts())
    print(df[target].value_counts(normalize=True).round(3) * 100)


def plot_class_distribution(df: pd.DataFrame, target: str = "Label",
                             save_path: str = "distribuicao_classes.png") -> None:
    """Guarda um gráfico de barras com a distribuição das classes."""
    plt.figure(figsize=(5, 4))
    sns.countplot(x=target, data=df)
    plt.title(f"Distribuição das classes ({target})")
    plt.xticks(rotation=20)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Gráfico guardado em: {save_path}")


# ==========================================
# CORRELAÇÕES
# ==========================================

def _encode_target(df: pd.DataFrame, target: str) -> pd.Series:
    """Converte o target categórico (ex: Label 'Benign'/'Malware') em 0/1.

    Se o target já for numérico, devolve-o sem alterações.
    Por convenção, a classe 'positiva' (1) é a que NÃO é 'Benign'.
    """
    if pd.api.types.is_numeric_dtype(df[target]):
        return df[target]
    return (df[target] != "Benign").astype(int)


def plot_correlation_matrix(df: pd.DataFrame, save_path: str = "matriz_correlacao.png") -> pd.DataFrame:
    """Guarda o heatmap de correlação entre todas as features numéricas e devolve a matriz.

    Nota: o dataset tem ~55 features numéricas, por isso o heatmap fica denso.
    Para uma leitura mais limpa, ver `target_correlation()` abaixo, que ordena
    apenas as correlações com a variável-alvo.
    """
    numeric_df = df.select_dtypes(include="number")

    plt.figure(figsize=(18, 14))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 7})
    plt.title("Matriz de correlação (features numéricas)")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Gráfico guardado em: {save_path}")

    return numeric_df.corr()


def target_correlation(df: pd.DataFrame, target: str = "Label", method: str = "pearson") -> pd.Series:
    """Devolve a correlação de cada feature numérica com o target, ordenada.

    method: "pearson" (relações lineares) ou "spearman" (relações monótonas,
    mais robusto a outliers — importante neste dataset, ver `compare_correlations`).
    """
    numeric_df = df.select_dtypes(include="number").copy()
    numeric_df["_target_"] = _encode_target(df, target)

    corr = numeric_df.corr(method=method)["_target_"].drop("_target_")
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index)

    print(f"\n--- Correlação ({method}) de cada feature com {target} ---")
    print(corr)
    return corr


def compare_correlations(df: pd.DataFrame, target: str = "Label") -> pd.DataFrame:
    """Compara Pearson vs Spearman lado a lado.

    Uma diferença grande entre os dois métodos é um sinal de que a relação
    não é linear e/ou de que a feature tem outliers a distorcer a Pearson
    (ex.: handles_nfile neste dataset).
    """
    pearson = target_correlation(df, target, method="pearson")
    spearman = target_correlation(df, target, method="spearman")

    comp = pd.DataFrame({"pearson": pearson, "spearman": spearman})
    comp["diff_abs"] = (comp["pearson"] - comp["spearman"]).abs()
    comp = comp.reindex(comp["diff_abs"].sort_values(ascending=False).index)

    print("\n--- Maiores divergências entre Pearson e Spearman ---")
    print(comp.head(10))
    return comp


# ==========================================
# DISTRIBUIÇÃO / ASSIMETRIA (skewness & kurtosis)
# ==========================================

def check_skewness_kurtosis(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Calcula skewness e kurtosis de cada feature numérica.

    |skew| > 1 já é considerado assimétrico; valores muito altos (dezenas ou
    centenas, como acontece neste dataset) indicam outliers extremos.
    """
    numeric_df = df.select_dtypes(include="number")
    stats = pd.DataFrame({
        "skew": numeric_df.skew(),
        "kurtosis": numeric_df.kurt(),
    })
    stats = stats.reindex(stats["skew"].abs().sort_values(ascending=False).index)

    print(f"\n--- Top {top_n} features mais assimétricas (skewness) ---")
    print(stats.head(top_n))
    return stats


# ==========================================
# DISTRIBUIÇÕES POR FEATURE
# ==========================================

def plot_feature_distributions(df: pd.DataFrame, features: list, target: str = "Label",
                                save_dir: str = ".") -> None:
    """Guarda um histograma por feature, separado por classe do target."""
    features = [f for f in features if f in df.columns]

    for feature in features:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x=feature, hue=target, bins=30, kde=True)
        plt.title(f"Distribuição de {feature} por classe ({target})")
        save_path = f"{save_dir}/dist_{feature}.png"
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
    """Corre a fase de Data Understanding do início ao fim e devolve o dataframe."""
    if features_to_plot is None:
        # As 4 features mais correlacionadas com Label neste dataset (ver EDA)
        features_to_plot = [
            "dlllist_avg_dlls_per_proc",
            "handles_nevent",
            "svcscan_nservices",
            "svcscan_kernel_drivers",
        ]

    df = load_data(path)
    overview(df)
    check_missing_duplicates(df)

    class_distribution(df, target="Label")
    class_distribution(df, target="SubType")  # visão multiclasse também
    plot_class_distribution(df, target="Label")

    plot_correlation_matrix(df)
    compare_correlations(df, target)
    check_skewness_kurtosis(df)

    plot_feature_distributions(df, features_to_plot, target)

    print("\n--- Data Understanding concluído ---")
    return df


if __name__ == "__main__":
    run_data_understanding("MalwareMemoryDump.csv")