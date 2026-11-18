
"""Inferência dos modelos de classificação."""

from pathlib import Path
import joblib


# 1. Encontrar modelos guardados
def discover_models(models_dir="models"):
    return list(Path(models_dir).glob("*_latest.joblib"))


# 2. Carregar um modelo
def load_model(model_path):
    return joblib.load(model_path)


# 3. Fazer previsões
def predict(model, texts):
    if isinstance(texts, str):
        texts = [texts]

    if not texts:
        raise ValueError("É necessário fornecer texto.")

    predictions = model.predict(texts)
    return [{"prediction": str(p)} for p in predictions]


# 4. Carregar o modelo e fazer previsões
def predict_from_file(model_path, texts):
    model = load_model(model_path)
    return predict(model, texts)


if __name__ == "__main__":
    models = discover_models()

    if not models:
        print("Nenhum modelo encontrado.")
    else:
        print("Modelo disponível:", models[0].name)

        text = input("Descreve o ataque: ")
        result = predict_from_file(models[0], text)

        print("Categoria prevista:", result[0]["prediction"])

