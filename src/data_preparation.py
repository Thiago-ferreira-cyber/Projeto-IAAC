
"""Preparação dos dados para classificação de ataques."""

import pandas as pd
from sklearn.model_selection import train_test_split

PREPARATION_VERSION = "cybersecurity_text_multiclass_v1"

TEXT_FEATURES = [
    "Title", "Scenario Description", "Tools Used",
    "Attack Steps", "Target Type", "Vulnerability",
    "MITRE Technique", "Impact", "Detection Method", "Tags"
]


def load_data(path):
    df = pd.read_csv(path)
    print("Dimensões originais:", df.shape)
    return df


def clean_data(df, target="Category", min_samples_per_class=100):
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Remover colunas desnecessárias e duplicados
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
    df = df.dropna(how="all").drop_duplicates()

    # Limpar a variável alvo
    df = df.dropna(subset=[target])
    df[target] = df[target].astype(str).str.strip()
    df = df[df[target] != ""]

    # Preparar as colunas textuais
    columns = [c for c in TEXT_FEATURES if c in df.columns]

    for col in columns:
        df[col] = (
            df[col].fillna("")
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    # Juntar as características num único texto
    df["Combined_Text"] = df[columns].agg(" ".join, axis=1).str.strip()
    df = df[df["Combined_Text"] != ""]

    # Remover textos repetidos
    df = df.drop_duplicates(subset=["Combined_Text"])

    # Manter classes com pelo menos 100 exemplos
    counts = df[target].value_counts()
    valid_classes = counts[counts >= min_samples_per_class].index
    df = df[df[target].isin(valid_classes)].copy()

    print("Registos após limpeza:", len(df))
    print("Número de classes:", df[target].nunique())

    return df


def split_features_target(df, target="Category"):
    X = df["Combined_Text"]
    y = df[target]

    print("\nDistribuição das classes:")
    print(y.value_counts())

    return X, y


def split_train_validation_test(
    X, y, validation_size=0.15, test_size=0.15, random_state=42
):
    if validation_size <= 0 or test_size <= 0:
        raise ValueError("As proporções devem ser positivas.")

    if validation_size + test_size >= 1:
        raise ValueError("A soma deve ser inferior a 1.")

    # Separar o conjunto de teste
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Separar a validação do treino
    val_fraction = validation_size / (1 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_fraction,
        random_state=random_state,
        stratify=y_temp
    )

    print("\nDivisão dos dados:")
    print("Treino:", len(X_train))
    print("Validação:", len(X_val))
    print("Teste:", len(X_test))

    return X_train, X_val, X_test, y_train, y_val, y_test


def run_data_preparation(
    path,
    target="Category",
    min_samples_per_class=100,
    validation_size=0.15,
    test_size=0.15,
    random_state=42
):
    df = load_data(path)
    df = clean_data(df, target, min_samples_per_class)
    X, y = split_features_target(df, target)

    return split_train_validation_test(
        X, y, validation_size, test_size, random_state
    )


if __name__ == "__main__":
    run_data_preparation("dataset/raw/Attack_Dataset.csv")


