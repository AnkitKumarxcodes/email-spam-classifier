# Email Spam Classifier

This project implements an **Email Spam Classification system** using **Machine Learning**.
The model is trained using **TF-IDF vectorization** and **Logistic Regression** and is exposed through:

- A **Streamlit web application** for user interaction
- A **FastAPI REST API** for programmatic access

Both components use the same trained model and preprocessing pipeline.

---

## 📂 Project Structure

EMAIL_SPAM_CLASSIFIER_PROJECT/
│
├── spam_classifier_app/ # Streamlit UI
├── spam_classifier_api/ # FastAPI REST API
├── .gitignore
└── README.md


---

## 🧠 Model Details
- Vectorizer: TF-IDF
- Classifier: Logistic Regression
- Task: Binary classification (Spam / Not Spam)
- Focus: High precision spam detection

---

## 🚀 Run Locally

### Streamlit App
```bash
cd spam_classifier_app
streamlit run streamlit_app.py
```
### FastAPI API
```
cd spam_classifier_api
uvicorn app:app --reload

https://email-spam-classifier-th2v.onrender.com/docs
```



