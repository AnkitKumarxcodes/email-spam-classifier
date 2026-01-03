from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

from preprocessing import preprocess_text


# Load model & vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

app = FastAPI(title="Spam Classifier API")


class TextInput(BaseModel):
    message: str


# Prediction endpoint
@app.post("/predict")
def predict_spam(data: TextInput):

    raw_text = data.message

    cleaned_text = preprocess_text(raw_text)
    X = vectorizer.transform([cleaned_text])

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    return {
        "label": "Spam" if pred == 1 else "Not Spam",
        "confidence": round(proba[1] * 100, 2)
    }
