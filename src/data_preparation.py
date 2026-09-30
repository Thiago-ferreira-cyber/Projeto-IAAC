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
# SPLIT TREINO/VALIDA??O/TESTE
# ==========================================

def split_train_validation_test(
    X: pd.DataFrame,
    y: pd.Series,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple:
    """Divide os dados em treino/valida??o/teste com propor??es estratificadas."""
    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size e test_size t?m de ser positivos e a soma inferior a 1.")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    validation_fraction = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=validation_fraction,
        random_state=random_state, stratify=y_train_val,
    )

    print("\n--- Divis?o dos dados ---")
    print(f"Treino: {X_train.shape} ({len(X_train) / len(X):.1%})")
    print(f"Valida??o: {X_val.shape} ({len(X_val) / len(X):.1%})")
    print(f"Teste: {X_test.shape} ({len(X_test) / len(X):.1%})")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ==========================================
# ORQUESTRAÇÃO (uso como script)
# ==========================================

def run_data_preparation(
    path: str,
    target: str = "Malicious",
    id_column: str = "File",
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple:
    """Prepara e devolve treino, valida??o e teste estratificados."""
    df = load_data(path)
    check_missing_duplicates(df)
    df = clean_data(df, id_column=id_column)
    X, y = split_features_target(df, target)
    splits = split_train_validation_test(
        X, y, validation_size=validation_size, test_size=test_size, random_state=random_state,
    )

    print("\n--- Data Preparation conclu?do ---")
    for name, split in zip(("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"), splits):
        print(f"{name}: {split.shape}")
    return splits

if __name__ == "__main__":
    run_data_preparation("Malware_and_benign_recognition.csv")