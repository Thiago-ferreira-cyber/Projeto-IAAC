
"""Análise exploratória do Cybersecurity Attack Dataset."""

import pandas as pd
import matplotlib.pyplot as plt


def run_data_understanding(path="dataset/raw/Attack_Dataset.csv"):
    # Carregar o dataset
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    print("=== DATA UNDERSTANDING ===")
    print("Dimensões:", df.shape)
    print("\nPrimeiras linhas:")
    print(df.head())

    # Verificar qualidade dos dados
    print("\nTipos de dados:")
    print(df.dtypes)

    print("\nValores nulos:")
    print(df.isnull().sum())

    print("\nLinhas duplicadas:", df.duplicated().sum())

    # Analisar a variável alvo
    categories = df["Category"].dropna().astype(str).str.strip()
    counts = categories.value_counts()

    print("\nNúmero de categorias:", len(counts))
    print("\nDistribuição das categorias:")
    print(counts)

    print("\nClasses com menos de 100 amostras:")
    print(counts[counts < 100])

    # Gráfico das categorias mais frequentes
    plt.figure(figsize=(11, 7))
    counts.head(20).sort_values().plot(kind="barh")
    plt.title("20 categorias de ataques mais frequentes")
    plt.xlabel("Número de registos")
    plt.tight_layout()
    plt.show()

    # Analisar comprimento das descrições
    lengths = df["Scenario Description"].fillna("").str.len()

    print("\nComprimento das descrições:")
    print(lengths.describe())

    plt.figure(figsize=(9, 5))
    plt.hist(lengths, bins=30)
    plt.title("Comprimento das descrições dos ataques")
    plt.xlabel("Número de caracteres")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.show()

    print("\n=== RESUMO ===")
    print("Registos:", len(df))
    print("Categorias:", len(counts))
    print("Categorias com pelo menos 100 registos:", (counts >= 100).sum())

    return df


if __name__ == "__main__":
    run_data_understanding()

