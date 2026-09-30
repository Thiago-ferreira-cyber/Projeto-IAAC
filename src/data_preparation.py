"""
Funções para a fase de Data Preparation (CRISP-ML).
Pensado para ser importado quer por um script, quer por um notebook.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ==========================================
# CARREGAMENTO
# ==========================================

def load_data(path: str) -> pd.DataFrame:
    """Carrega o dataset a partir de um ficheiro CSV."""
    df = pd.read_csv(path)
    print("Dimensões do dataset:", df.shape)
    return df


# ==========================================
# VERIFICAÇÕES
# ==========================================

def check_missing_duplicates(df: pd.DataFrame) -> None:
    """Verifica valores nulos e linhas duplicadas."""
    print("\n--- Valores nulos ---")
    print(df.isnull().sum())

    print("\n--- Duplicados ---")
    print("Número de duplicados:", df.duplicated().sum())


# ==========================================
# LIMPEZA
# ==========================================

def clean_data(df: pd.DataFrame, id_column: str = "File") -> pd.DataFrame:
    """Corrige tipos de dados e remove o identificador único."""
    df = df.copy()

    # ImageBase vem como uint64 (valores muito grandes) — passar para int64
    if "ImageBase" in df.columns:
        df["ImageBase"] = df["ImageBase"].astype(np.int64)

    # Identificador único, não deve ser usado como feature
    if id_column in df.columns:
        df = df.drop(columns=[id_column])

    return df


# ==========================================
# FEATURES / TARGET
# ==========================================

def split_features_target(df: pd.DataFrame, target: str) -> tuple:
    """Separa o dataframe em features (X) e target (y)."""
    X = df.drop(columns=[target])
    y = df[target]

    print(f"\n--- Distribuição das classes ({target}) ---")
    print((y.value_counts(normalize=True).round(3) * 100))

    return X, y


# ==========================================
# SPLIT TREINO/TESTE
# ==========================================

def split_train_test(X: pd.DataFrame, y: pd.Series, test_size: float = 0.20, random_state: int = 42) -> tuple:
    """Divide os dados em treino e teste, mantendo a proporção de classes."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print("\n--- Divisão dos dados ---")
    print("Treino:", X_train.shape)
    print("Teste:", X_test.shape)

    return X_train, X_test, y_train, y_test


# ==========================================
# ORQUESTRAÇÃO (uso como script)
# ==========================================

def run_data_preparation(
    path: str,
    target: str = "Malicious",
    id_column: str = "File",
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple:
    """Corre a fase de Data Preparation do início ao fim e devolve os dados prontos."""
    df = load_data(path)
    check_missing_duplicates(df)
    df = clean_data(df, id_column=id_column)
    X, y = split_features_target(df, target)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size, random_state)

    print("\n--- Data Preparation concluído ---")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_data_preparation("Malware_and_benign_recognition.csv")