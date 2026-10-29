"""Generate a text report on the held-out test split and optionally send it to Telegram."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import data_preparation as dp


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "raw" / "Malware_and_benign_recognition.csv"
MODEL_PATH = ROOT / "models" / "malware_random_forest_latest.joblib"
METADATA_PATH = ROOT / "models" / "malware_random_forest_latest.json"
SPLIT_MANIFEST_PATH = ROOT / "dataset" / "processed" / "split_manifest.csv"
OUTPUT_DIR = ROOT / "models"
RANDOM_STATE = 42
TARGET = "Malicious"
POSITIVE_LABEL = 1
TELEGRAM_TEXT_LIMIT = 3500
MESSAGE_SEPARATOR = "\n\n----- PRÓXIMA MENSAGEM -----\n\n"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Load KEY=VALUE pairs from .env without overriding existing environment values."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, separator, value = line.partition("=")
        if not separator or not key.strip():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '\"'):
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def build_test_report() -> tuple[str, Path]:
    """Evaluate latest model on the project's reproducible, held-out test split."""
    if not DATA_PATH.is_file() or not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
        raise FileNotFoundError("Dataset, modelo latest ou metadados não encontrados.")

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if metadata.get("preparation_version") != dp.PREPARATION_VERSION:
        raise ValueError(
            "O modelo latest foi treinado com outra versão da preparação dos dados. "
            "Executa novamente o notebook 04 para treinar e guardar o modelo após a deduplicação."
        )
    split_provenance = metadata.get("split_manifest", {})
    current_manifest_hash = hashlib.sha256(SPLIT_MANIFEST_PATH.read_bytes()).hexdigest()
    if split_provenance.get("sha256") != current_manifest_hash:
        raise ValueError(
            "O modelo latest não foi treinado com o manifesto de divisão ativo. "
            "Executa novamente o notebook 04 antes de avaliar o teste."
        )
    if split_provenance.get("training_partitions") != ["train"]:
        raise ValueError(
            "O modelo latest não declara treino exclusivo na partição train. "
            "Executa novamente o notebook 04 antes de avaliar o teste."
        )

    metrics = metadata.get("metricas", {})
    # Notebook 04 records its validation-selected threshold. Notebook 03 refits on
    # train + validation, so its latest model must not be presented as test-independent.
    if "threshold_selection" not in metrics:
        raise ValueError(
            "O modelo latest não documenta a seleção do limiar na validação. "
            "Para uma avaliação independente, guarda primeiro o modelo treinado apenas no treino (notebook 04)."
        )

    df = pd.read_csv(DATA_PATH)
    feature_names = metadata.get("features")
    if not feature_names:
        raise ValueError("Os metadados do modelo não incluem a lista de features.")
    missing = sorted(set(feature_names) - set(df.columns))
    if missing:
        raise ValueError(f"Faltam features no dataset: {missing}")

    _, _, X_test, _, _, y_test = dp.run_data_preparation(
        str(DATA_PATH), target=TARGET, random_state=RANDOM_STATE
    )
    model = joblib.load(MODEL_PATH)
    model_features = list(getattr(model, "feature_names_in_", feature_names))
    if model_features != list(feature_names):
        raise ValueError("Features nos metadados não correspondem à ordem esperada pelo modelo.")
    X_test = X_test.loc[:, feature_names]

    threshold_info = metrics["threshold_selection"]
    threshold = float(threshold_info["threshold"])
    classes = list(model.classes_)
    positive_index = classes.index(POSITIVE_LABEL)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, positive_index]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        if getattr(scores, "ndim", 1) == 1 and classes[1] != POSITIVE_LABEL:
            scores = -scores
        elif getattr(scores, "ndim", 1) > 1:
            scores = scores[:, positive_index]
    else:
        raise TypeError("O modelo não fornece predict_proba nem decision_function.")

    y_true = y_test.astype(int).to_numpy()
    y_pred = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    class_report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=["Benigno", "Malicioso"],
        output_dict=True, zero_division=0,
    )
    model_name = metadata.get("modelo", MODEL_PATH.name)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "RELATÓRIO DE AVALIAÇÃO — CONJUNTO DE TESTE",
        f"Modelo: {model_name} | Data: {timestamp}",
        f"Samples: {len(y_true)} | Limiar malware: {threshold:.4f}",
        f"Accuracy: {accuracy_score(y_true, y_pred):.4f} | F1 macro: {f1_score(y_true, y_pred, average='macro'):.4f}",
        "",
        "MATRIZ DE CONFUSÃO (real × previsto)",
        "                 Previsto benigno | Previsto malware",
        f"Real benigno     {matrix[0, 0]:>15} | {matrix[0, 1]:>15}",
        f"Real malware     {matrix[1, 0]:>15} | {matrix[1, 1]:>15}",
        "",
        "MÉTRICAS POR CLASSE",
        f"{'Classe':<12} {'Precisão':>10} {'Recall':>10} {'F1':>10} {'N':>6}",
    ]
    for label, key in (("Benigno", "Benigno"), ("Malicioso", "Malicioso")):
        item = class_report[key]
        lines.append(
            f"{label:<12} {item['precision']:>10.3f} {item['recall']:>10.3f} "
            f"{item['f1-score']:>10.3f} {int(item['support']):>6}"
        )
    lines.extend([
        "",
        "PREVISÕES POR SAMPLE",
        f"{'Ficheiro':<30} {'Real':<10} {'Previsto':<10} {'Score':>7} {'Resultado'}",
        "-" * 75,
    ])

    sample_names = df.loc[X_test.index, "File"].astype(str).to_numpy() if "File" in df else X_test.index.astype(str).to_numpy()
    row_lines = []
    for name, actual, predicted, score in zip(sample_names, y_true, y_pred, scores):
        filename = name if len(name) <= 29 else name[:26] + "..."
        actual_name = "Malware" if actual == 1 else "Benigno"
        predicted_name = "Malware" if predicted == 1 else "Benigno"
        result = "CERTO" if actual == predicted else "ERRO"
        row_lines.append(f"{filename:<30} {actual_name:<10} {predicted_name:<10} {score:>7.3f} {result}")

    # Split the full text into Telegram-safe HTML <pre> messages, preserving rows.
    chunks: list[str] = []
    current = lines[:]
    for row in row_lines:
        if len("\n".join(current + [row])) > TELEGRAM_TEXT_LIMIT - 20:
            chunks.append("\n".join(current))
            current = ["PREVISÕES POR SAMPLE (continuação)", lines[-2]]
        current.append(row)
    if current:
        chunks.append("\n".join(current))

    source_report = MESSAGE_SEPARATOR.join(chunks)
    report_text = generate_report_with_agent(source_report, metadata, current_manifest_hash)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"relatorio_teste_agente_{datetime.now():%Y%m%d_%H%M%S}.md"
    report_path.write_text(report_text, encoding="utf-8")
    return report_text, report_path


def generate_report_with_agent(source_report: str, metadata: dict, manifest_hash: str) -> str:
    """Use the configured local agent to write the report and send it via Telegram."""
    agent_name = os.getenv("REPORT_AGENT", "hermes").strip().lower()
    executable = os.getenv("REPORT_AGENT_EXECUTABLE") or shutil.which(agent_name)
    if not executable:
        raise RuntimeError(
            f"Agente '{agent_name}' não encontrado. Instala/configura o agente ou "
            "define REPORT_AGENT_EXECUTABLE no .env."
        )
    prompt = (
        "Send the final report as a text message using this agent's configured Telegram "
        "integration. If no Telegram tool is available, do not claim delivery; return "
        "the Markdown report only.\n\n"
        "Redige um relatório conciso (máximo 3000 caracteres) em Markdown e português "
        "europeu com base nos resultados abaixo. Preserva os valores; não inventes dados. "
        "Resume as métricas e matriz de confusão; lista só as previsões erradas, não "
        "reproduzas a lista completa de previsões corretas. "
        "Indica que o modelo usou apenas a partição train e que a avaliação usa o manifesto "
        f"SHA-256 {manifest_hash}. Acrescenta limitações: avaliação num único split não "
        "garante desempenho futuro e resultados individuais não explicam a causa dos erros. "
        "Entrega apenas Markdown, sem bloco de código.\n\n"
        f"Metadados do modelo (JSON):\n{json.dumps(metadata, ensure_ascii=False, indent=2)}\n\n"
        f"Resultados calculados (não alterar):\n{source_report}"
    )
    agent_environment = os.environ.copy()
    if agent_name == "codex":
        agent_environment.setdefault("CODEX_HOME", str(Path.home() / ".codex"))
    if agent_name == "hermes":
        command = [executable, "chat", "--query-file", "-", "--oneshot", "--quiet"]
    elif agent_name == "codex":
        command = [executable, "exec", "--ephemeral", "--sandbox", "read-only", "--cd", str(ROOT), "-"]
    else:
        raise ValueError(
            f"Agent '{agent_name}' not supported. Use REPORT_AGENT=hermes or codex."
        )

    try:
        process = subprocess.run(
            command,
            input=prompt,
            env=agent_environment,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=int(os.getenv("REPORT_AGENT_TIMEOUT", "300")),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Não foi possível obter o relatório do agente: {exc}") from exc
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(f"O agente falhou (código {process.returncode}): {detail}")
    report = process.stdout.strip()
    if not report:
        raise RuntimeError("O agente terminou sem produzir o relatório.")
    return report + "\n"


def main() -> None:
    load_dotenv()
    report_text, report_path = build_test_report()
    print(f"Relatório gerado e guardado em: {report_path}")
    print("\n" + "="*40 + "\n")
    print(report_text)
    print("\n" + "="*40)


if __name__ == "__main__":
    main()
