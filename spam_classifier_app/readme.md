
##  spam_classifier_app/README.md

```md
# Email Spam Classifier – Streamlit App

This directory contains a **Streamlit web application** for interactively testing the email spam classification model.

---

## 🎯 Features
- Email text input
- Spam / Not Spam prediction
- Confidence score display
- Sample email loading
- Interactive UI

---

## 🛠 Tech Stack
- Streamlit
- Scikit-learn
- TF-IDF
- Logistic Regression
- Joblib
- NLTK

---

## 📂 Directory Structure

spam_classifier_app/
│
├── streamlit_app.py
├── preprocessing.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
└── README.md

yaml
Copy code

---

## ▶ Run the App

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
🧪 Usage
Enter an email message

Click Analyze Email

View prediction and confidence score

📌 Notes
Runs locally by default

Uses the same model as the FastAPI API

Designed for ML demonstration and evaluation

