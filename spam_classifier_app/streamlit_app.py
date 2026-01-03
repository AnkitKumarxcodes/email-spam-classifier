import streamlit as st
import joblib
import os

from preprocessing import preprocess_text

#loading model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))


# Page config
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered"
)

#  Sidebar
st.sidebar.title("📌 About")
st.sidebar.write(
    """
    This application uses **TF-IDF** and **Logistic Regression**
    to classify emails as **Spam** or **Not Spam**.

    - High precision model
    - Optimized for spam detection
    """
)

st.sidebar.markdown("---")
st.sidebar.write("**Try sample emails below 👇**")

if st.sidebar.button("Load Spam Example"):
    st.session_state.sample_text = (
        "Dear Customer,\n\n"
        "Congratulations! Your email has been selected to receive "
        "a cash reward of $10,000. Click the link below to claim your prize.\n\n"
        "Regards,\nRewards Team"
    )

if st.sidebar.button("Load Ham Example"):
    st.session_state.sample_text = (
        "Hi Team,\n\n"
        "This is a reminder about the project meeting scheduled "
        "for tomorrow at 11 AM. Please be prepared with your updates.\n\n"
        "Thanks,\nAnkit"
    )


st.title("📧 Email Spam Classifier")
st.write("Paste an email below to check whether it is **Spam** or **Not Spam**.")

# Text input
email_text = st.text_area(
    "✉️ Email content",
    height=220,
    value=st.session_state.get("sample_text", "")
)

# Live stats
char_count = len(email_text)
word_count = len(email_text.split())

st.caption(f"📝 Characters: {char_count} | Words: {word_count}")

# Predict button
if st.button("🔍 Analyze Email", use_container_width=True):
    if email_text.strip() == "":
        st.warning("Please enter some email text.")
    else:
        # Preprocess
        cleaned_text = preprocess_text(email_text)

        # Vectorize
        X = vectorizer.transform([cleaned_text])

        # Predict
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]

        spam_confidence = probability[1] * 100
        ham_confidence = probability[0] * 100

        st.markdown("---")

        #  Output
        if prediction == 1:
            st.error("🚨 **This email is classified as SPAM**")
            st.progress(int(spam_confidence))
            st.write(f"**Confidence:** {spam_confidence:.2f}%")
        else:
            st.success("✅ **This email is NOT spam**")
            st.progress(int(ham_confidence))
            st.write(f"**Confidence:** {ham_confidence:.2f}%")

st.markdown("---")
st.caption("🔬 Model: TF-IDF + Logistic Regression | Built with Streamlit")
