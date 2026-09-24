"""
Funções para a fase de Data Preparation (CRISP-ML).
Adaptado para o dataset MalwareMemoryDump.csv
(features extraídas com Volatility a partir de dumps de memória RAM).

Passos aplicados (decididos na fase de Data Understanding):
    1. Remover linhas duplicadas
    2. Remover o identificador único (Raw_Type) e a coluna redundante com
       o target (SubType, que teria fuga de informação em relação a Label)
    3. Remover colunas constantes (variância zero)
    4. Reduzir multicolinearidade (pares de features com |r| >= 0.95)
    5. Tratar outliers extremos por winsorização (cap no percentil 1º/99º)
    6. Separar X / y e dividir em treino / teste

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
    df = pd.read_csv("../dataset/raw/MalwareMemoryDump.csv")
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

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas 100% duplicadas."""
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicados removidos: {before - len(df)} (novo shape: {df.shape})")
    return df


def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove colunas com um único valor (variância zero) — não ajudam o modelo."""
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    print("Colunas constantes removidas:", const_cols)
    return df.drop(columns=const_cols)


def reduce_multicollinearity(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """Remove, de cada par de features numéricas com |correlação| >= threshold,
    a que tiver maior correlação média com todas as outras (mais redundante).
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    corr = df[num_cols].corr().abs()

    to_drop = set()
    for i, c1 in enumerate(num_cols):
        for c2 in num_cols[i + 1:]:
            if c1 in to_drop or c2 in to_drop:
                continue
            if corr.loc[c1, c2] >= threshold:
                avg1 = corr[c1].drop(c1).mean()
                avg2 = corr[c2].drop(c2).mean()
                to_drop.add(c1 if avg1 >= avg2 else c2)

    print(f"Colunas removidas por multicolinearidade (|r|>={threshold}): {len(to_drop)}")
    print(sorted(to_drop))
    return df.drop(columns=list(to_drop))


def handle_outliers(df: pd.DataFrame, skew_threshold: float = 5.0,
                     lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    """Winsoriza (limita) as features cuja assimetria (skew) ultrapassa o limiar,
    fazendo cap nos percentis lower_q / upper_q em vez de remover as linhas.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    skew = df[num_cols].skew()
    cols_to_cap = skew[skew.abs() > skew_threshold].index.tolist()

    print(f"Colunas winsorizadas (|skew|>{skew_threshold}): {cols_to_cap}")
    for col in cols_to_cap:
        lo, hi = df[col].quantile(lower_q), df[col].quantile(upper_q)
        df[col] = df[col].clip(lo, hi)

    return df


def clean_data(df: pd.DataFrame, id_column: str = "Raw_Type",
                drop_columns: list = None) -> pd.DataFrame:
    """Aplica toda a limpeza: duplicados, colunas constantes, identificador,
    colunas redundantes, multicolinearidade e outliers.

    drop_columns: colunas adicionais a remover antes das features (por omissão,
    'SubType', que teria fuga de informação em relação ao target 'Label').
    """
    if drop_columns is None:
        drop_columns = ["SubType"]

    df = df.copy()
    df = remove_duplicates(df)
    df = drop_constant_columns(df)

    if id_column in df.columns:
        df = df.drop(columns=[id_column])

    existing_drop = [c for c in drop_columns if c in df.columns]
    if existing_drop:
        df = df.drop(columns=existing_drop)

    df = reduce_multicollinearity(df)
    df = handle_outliers(df)

    return df


# ==========================================
# FEATURES / TARGET
# ==========================================

def split_features_target(df: pd.DataFrame, target: str = "Label") -> tuple:
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
    target: str = "Label",
    id_column: str = "Raw_Type",
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
    run_data_preparation("MalwareMemoryDump.csv")