# 📧 Email Spam Classifier

An end-to-end **Email Spam Classification system** built using **Machine Learning and NLP**. The project trains and evaluates multiple ML models using both **Count Vectorizer** and **TF-IDF** representations, and deploys the best-performing model via:

* 🖥 **Streamlit Web Application** for user interaction
* ⚙️ **FastAPI REST API** for programmatic access

Both the web app and API use the **same trained model and preprocessing pipeline**, ensuring consistent predictions.

---

## 📂 Project Structure

```
EMAIL_SPAM_CLASSIFIER_PROJECT/
│
├── spam_classifier_app/        # Streamlit UI
├── spam_classifier_api/        # FastAPI REST API
├── .gitignore
└── README.md
```

---

## 🧠 Dataset & Training Details

* **Task:** Binary Classification (Spam / Not Spam)
* **Text Vectorization Techniques Used:**

  * Count Vectorizer
  * TF-IDF Vectorizer
* **Models Evaluated:**

  * Logistic Regression (LR)
  * Random Forest (RF)
  * Extra Trees Classifier (ETC)
  * Naive Bayes (NB)
  * Gradient Boosted Decision Trees (GBDT)
  * XGBoost (XGB)
  * AdaBoost
  * Decision Tree (DT)
  * K-Nearest Neighbors (KNN)

The dataset was split into **training and validation sets**, and all models were evaluated primarily on **accuracy and precision**, with special emphasis on **high precision for spam detection**.

---

## 📊 Model Performance Snapshot

The top 5 models that were trained and evaluated using **two different text vectorization techniques**:

### 🔹 Count Vectorizer (CV)

| Model                  | Accuracy   | Precision  |
| ---------------------- | ---------- | ---------- |
| Logistic Regression    | **98.24%** | **98.64%** |
| Extra Trees Classifier | 98.14%     | 98.42%     |
| Random Forest          | 97.81%     | 98.63%     |
| Naive Bayes            | 94.86%     | **99.02%** |
| XGBoost                | 96.50%     | 96.62%     |

### 🔹 TF-IDF Vectorizer

| Model                  | Accuracy   | Precision   |
| ---------------------- | ---------- | ----------- |
| Logistic Regression    | **95.62%** | **100.00%** |
| Extra Trees Classifier | 98.58%     | 99.54%      |
| Random Forest          | 97.26%     | 98.39%      |
| Naive Bayes            | 97.37%     | 97.74%      |
| XGBoost                | 96.49%     | 97.05%      |

---

📌 **Key Observations**

* **TF-IDF consistently improved precision**, which is critical for spam detection
* **Logistic Regression showed stable and high precision across both CV and TF-IDF**
* Naive Bayes achieved very high precision but lower overall accuracy
* Tree-based models performed well but were heavier for deployment

---

## ✅ Final Model Selection

* **Vectorizer:** TF-IDF
* **Classifier:** Logistic Regression
* **Final Performance:**

  * Accuracy ≈ **97%**
  * Precision ≈ **100%**

📌 **Why Logistic Regression?**

* High precision (minimizes false positives)
* Simple and interpretable model
* Fast inference and low memory footprint
* Well-suited for real-time web and API deployment

This configuration provides an optimal balance between **performance, simplicity, and deployment readiness**.

---

## 🚀 Run Locally

### ▶️ Streamlit App

```bash
cd spam_classifier_app
streamlit run streamlit_app.py
```

### ▶️ FastAPI API

```bash
cd spam_classifier_api
uvicorn app:app --reload
```

Once running, open:

* **API Docs:** `http://127.0.0.1:8000/docs`

---

## 🔗 Live Demo

* **Streamlit App:** *Add live app link here*
* **API Docs:** *Add deployed API docs link here*

---

## 🛠 Tech Stack

* Python
* Scikit-learn
* TF-IDF & Count Vectorizer
* Streamlit
* FastAPI
* Uvicorn

---

## 📌 Key Highlights

* Compared multiple ML classifiers for spam detection
* Used **two different text vectorization techniques**
* Selected model based on **precision-focused evaluation**
* Deployed both **UI and API** using a shared ML pipeline
* Clean, modular, and reproducible project structure

---

## 📄 License

This project is open-source and available for learning and experimentation.







