from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

from preprocessing import preprocess_text

# -----------------------
# Load model & vectorizer
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

app = FastAPI(title="Spam Classifier API")

# -----------------------
# Input schema
# -----------------------
class TextInput(BaseModel):
    message: str

# -----------------------
# Prediction endpoint
# -----------------------
@app.post("/predict")
def predict_spam(data: TextInput):

    # 1️⃣ Take input
    raw_text = data.message

    # 2️⃣ Preprocess
    cleaned_text = preprocess_text(raw_text)

    # 3️⃣ Vectorize
    X = vectorizer.transform([cleaned_text])

    # 4️⃣ Predict
    prediction = model.predict(X)[0]

    # 5️⃣ Output
    return {
        "prediction": "Spam" if prediction == 1 else "Not Spam"
    }
