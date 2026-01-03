
````md
# 📧 Email Spam Classifier – FastAPI API

This project provides a FastAPI-based REST API that serves a trained email spam classification model. The API accepts raw email text and returns whether the message is Spam or Not Spam, along with a confidence score.

## 🛠 Tech Stack

- FastAPI  
- Scikit-learn  
- TF-IDF Vectorizer  
- Logistic Regression  
- Joblib  
- NLTK  

---

## 📂 Directory Structure

```text
spam_classifier_api/
│
├── app.py
├── preprocessing.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md
````

---

## ▶️ Running the API

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start the Server

```bash
uvicorn app:app --reload
```

The API will be available at:

```
http://127.0.0.1:8000
```

---

## 🔗 API Endpoint

### POST /predict

#### Request

```json
{
  "message": "Your email text here"
}
```

#### Response

```json
{
  "label": "Spam",
  "confidence": 67.45
}
```

---

## 📘 API Documentation

Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

## ⚠️ Notes

* Ensure model.pkl and vectorizer.pkl are present before starting the API.
* Confidence values are derived from the model’s probability estimates.



