"""Generate a text report on the held-out test split and optionally send it to Telegram."""

from __future__ import annotations

import html
import json
import os
import urllib.error
import urllib.request
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

    report_text = MESSAGE_SEPARATOR.join(chunks)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"relatorio_teste_telegram_{datetime.now():%Y%m%d_%H%M%S}.txt"
    report_path.write_text(report_text, encoding="utf-8")
    return report_text, report_path


def send_telegram_report(report_text: str, bot_token: str, chat_id: str) -> None:
    """Send report chunks as plain HTML-preformatted Telegram messages."""
    chunks = report_text.split(MESSAGE_SEPARATOR)
    for chunk in chunks:
        payload = json.dumps({
            "chat_id": chat_id,
            "text": f"<pre>{html.escape(chunk)}</pre>",
            "parse_mode": "HTML",
        }).encode("utf-8")
        request = urllib.request.Request(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Telegram respondeu HTTP {exc.code}: {details}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Falha ao contactar o Telegram: {exc}") from exc
        if not result.get("ok"):
            raise RuntimeError(f"Telegram rejeitou o envio: {result}")


def main() -> None:
    load_dotenv()
    report_text, report_path = build_test_report()
    print(f"Relatório guardado: {report_path}")
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if (
        not bot_token
        or not chat_id
        or "coloca_aqui" in bot_token.lower()
        or "coloca_aqui" in chat_id.lower()
    ):
        raise SystemExit(
            "Relatório criado, mas não enviado. Preenche TELEGRAM_BOT_TOKEN e "
            "TELEGRAM_CHAT_ID no ficheiro .env."
        )
    send_telegram_report(report_text, bot_token, chat_id)
    print("Relatório enviado para o Telegram.")


if __name__ == "__main__":
    main()
