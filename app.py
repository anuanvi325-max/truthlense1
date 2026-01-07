# app.py

import streamlit as st
import os
import re
import pickle

# -----------------------------
# Load model & vectorizer
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "ensemble_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# -----------------------------
# Text cleaning
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -----------------------------
# UI DESIGN
# -----------------------------
st.set_page_config(page_title="TruthLens", layout="centered")

st.title("🕵️‍♂️ TruthLens")
st.subheader("Ensemble-based Fake News Detection System")
st.write("Paste a news article or headline below to check whether it is **FAKE or REAL**.")

news_input = st.text_area("📰 Enter News Text", height=200)

if st.button("🔍 Predict"):
    if news_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        cleaned = clean_text(news_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction.upper() == "FAKE":
            st.error("🚨 This news is likely **FAKE**")
        else:
            st.success("✅ This news appears to be **REAL**")

st.markdown("---")
st.caption("Mini Project | TruthLens | NLP + Ensemble ML")
