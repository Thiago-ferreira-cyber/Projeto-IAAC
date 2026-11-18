"""
Funções para a fase de Data Preparation (CRISP-ML).
Pensado para ser importado quer por um script, quer por um notebook.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Increment this when the rules used to build the modeling data change.
PREPARATION_VERSION = "deduplicate_exact_samples_v1"
SPLIT_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "dataset" / "processed" / "split_manifest.csv"
SPLIT_NAMES = ("train", "validation", "test")


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

def clean_data(
    df: pd.DataFrame,
    id_column: str = "File",
    remove_exact_duplicates: bool = True,
) -> pd.DataFrame:
    """Corrige tipos, remove o identificador e deduplica linhas idênticas."""
    df = df.copy()

    # ImageBase vem como uint64 (valores muito grandes) — passar para int64
    if "ImageBase" in df.columns:
        df["ImageBase"] = df["ImageBase"].astype(np.int64)

    # Identificador único, não deve ser usado como feature
    if id_column in df.columns:
        df = df.drop(columns=[id_column])

    if remove_exact_duplicates:
        original_rows = len(df)
        df = df.drop_duplicates(keep="first")
        removed = original_rows - len(df)
        print(f"Duplicados exatos removidos (sem {id_column}): {removed}")

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

    constant_features = X.columns[X.nunique(dropna=False) <= 1].tolist()
    duplicate_features = []
    columns = list(X.columns)
    for index, column in enumerate(columns):
        for other in columns[index + 1:]:
            if X[column].equals(X[other]):
                duplicate_features.append((column, other))

    print("\n--- Auditoria de features (informativa; não remove colunas) ---")
    print(f"Features constantes: {constant_features or 'nenhuma'}")
    print(f"Pares de features idênticas: {duplicate_features or 'nenhum'}")

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


def _canonical_value(value):
    """Convert pandas/numpy scalars into stable JSON values for row fingerprints."""
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return format(value, ".17g")
    return value


def _sample_fingerprints(df: pd.DataFrame, id_column: str) -> pd.Series:
    """Hash all sample data except its identifier, including its target."""
    sample_columns = sorted(column for column in df.columns if column != id_column)
    fingerprints = []
    for row in df.loc[:, sample_columns].itertuples(index=False, name=None):
        canonical = [_canonical_value(value) for value in row]
        payload = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
        fingerprints.append(hashlib.sha256(payload.encode("utf-8")).hexdigest())
    return pd.Series(fingerprints, index=df.index, name="sample_hash")


def _feature_group_fingerprints(
    df: pd.DataFrame,
    id_column: str,
    target: str,
) -> pd.Series:
    """Hash model inputs only, so identical feature vectors stay in one partition."""
    feature_columns = sorted(
        column for column in df.columns if column not in {id_column, target}
    )
    fingerprints = []
    for row in df.loc[:, feature_columns].itertuples(index=False, name=None):
        canonical = [_canonical_value(value) for value in row]
        payload = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
        fingerprints.append(hashlib.sha256(payload.encode("utf-8")).hexdigest())
    return pd.Series(fingerprints, index=df.index, name="feature_hash")


def create_split_manifest(
    path: str,
    target: str = "Malicious",
    id_column: str = "File",
    manifest_path: str | Path = SPLIT_MANIFEST_PATH,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Path:
    """Create a permanent stratified assignment manifest once; refuse overwrite."""
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        raise FileExistsError(
            f"O manifesto já existe: {manifest_path}. Não o substituir automaticamente."
        )

    df = load_data(path)
    if id_column not in df.columns or target not in df.columns:
        raise ValueError(f"O dataset tem de incluir {id_column!r} e {target!r}.")
    if df[id_column].isna().any() or df[id_column].astype(str).duplicated().any():
        raise ValueError(f"{id_column!r} tem de estar preenchido e ser único por linha.")

    row_hashes = _sample_fingerprints(df, id_column)
    feature_hashes = _feature_group_fingerprints(df, id_column, target)
    clean_df = clean_data(df, id_column=id_column)
    X, y = split_features_target(clean_df, target)
    split_parts = split_train_validation_test(
        X, y, validation_size=validation_size, test_size=test_size,
        random_state=random_state,
    )

    assignment_by_hash = {}
    for split_name, X_split in zip(SPLIT_NAMES, split_parts[:3]):
        for index in X_split.index:
            sample_hash = feature_hashes.loc[index]
            prior = assignment_by_hash.setdefault(sample_hash, split_name)
            if prior != split_name:
                raise ValueError("Uma amostra duplicada foi atribuída a mais de uma partição.")

    manifest = pd.DataFrame({
        id_column: df[id_column].astype(str),
        "sample_hash": feature_hashes,
        "row_hash": row_hashes,
        "split": feature_hashes.map(assignment_by_hash),
    })
    if manifest["split"].isna().any():
        raise ValueError("Não foi possível atribuir uma partição a todas as amostras.")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    print(f"Manifesto permanente criado: {manifest_path}")
    print(manifest["split"].value_counts().reindex(SPLIT_NAMES, fill_value=0))
    return manifest_path


def _load_and_validate_manifest(
    df: pd.DataFrame,
    id_column: str,
    target: str,
    manifest_path: str | Path,
) -> pd.DataFrame:
    """Validate identifiers and row fingerprints before applying saved partitions."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Manifesto de divisão não encontrado: {manifest_path}. "
            "Cria-o uma vez com create_split_manifest; não é gerado automaticamente."
        )
    required = {id_column, "sample_hash", "row_hash", "split"}
    manifest = pd.read_csv(manifest_path, dtype={id_column: "string"})
    missing_columns = required - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"Colunas em falta no manifesto: {sorted(missing_columns)}")
    if manifest[id_column].isna().any() or manifest[id_column].duplicated().any():
        raise ValueError(f"O manifesto contém identificadores {id_column!r} inválidos.")
    if manifest["split"].isna().any() or not set(manifest["split"].unique()).issubset(SPLIT_NAMES):
        raise ValueError("O manifesto contém partições ausentes ou desconhecidas.")

    current_ids = df[id_column].astype("string")
    manifest_ids = manifest[id_column]
    if set(current_ids) != set(manifest_ids):
        missing = sorted(set(manifest_ids) - set(current_ids))[:5]
        new = sorted(set(current_ids) - set(manifest_ids))[:5]
        raise ValueError(
            "O dataset não corresponde ao manifesto. "
            f"Amostras ausentes (até 5): {missing}; novas (até 5): {new}. "
            "Revê e atualiza a divisão de forma explícita."
        )

    current = pd.DataFrame({
        id_column: current_ids.to_numpy(),
        "sample_hash": _feature_group_fingerprints(df, id_column, target).to_numpy(),
        "row_hash": _sample_fingerprints(df, id_column).to_numpy(),
    })
    checked = current.merge(
        manifest[[id_column, "sample_hash", "row_hash", "split"]],
        on=id_column,
        how="left",
        suffixes=("_current", "_saved"),
        validate="one_to_one",
    )
    changed_hash = (
        (checked["sample_hash_current"] != checked["sample_hash_saved"])
        | (checked["row_hash_current"] != checked["row_hash_saved"])
    )
    if changed_hash.any():
        changed_ids = checked.loc[changed_hash, id_column].head(5).tolist()
        raise ValueError(
            "Features ou rótulos mudaram desde a criação do manifesto para "
            f"estas amostras: {changed_ids}. A divisão exige revisão explícita."
        )

    return manifest.set_index(id_column)


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
    manifest_path: str | Path = SPLIT_MANIFEST_PATH,
) -> tuple:
    """Use the saved assignment; size and seed arguments remain for compatibility."""
    df = load_data(path)
    check_missing_duplicates(df)
    saved_manifest = _load_and_validate_manifest(df, id_column, target, manifest_path)
    clean_df = clean_data(df, id_column=id_column)
    X, y = split_features_target(clean_df, target)

    retained_ids = df.loc[clean_df.index, id_column].astype(str)
    assignment = saved_manifest.loc[retained_ids, "split"].to_numpy()
    split_data = {}
    for split_name in SPLIT_NAMES:
        indices = clean_df.index[assignment == split_name]
        split_data[split_name] = (X.loc[indices], y.loc[indices])
    X_train, y_train = split_data["train"]
    X_val, y_val = split_data["validation"]
    X_test, y_test = split_data["test"]
    splits = (X_train, X_val, X_test, y_train, y_val, y_test)

    print(f"\n--- Divisão permanente ({Path(manifest_path)}) ---")
    for split_name in SPLIT_NAMES:
        X_split, _ = split_data[split_name]
        print(f"{split_name}: {X_split.shape}")
    print("\n--- Data Preparation concluído ---")
    for name, split in zip(("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"), splits):
        print(f"{name}: {split.shape}")
    return splits

if __name__ == "__main__":
    run_data_preparation("Malware_and_benign_recognition.csv")
