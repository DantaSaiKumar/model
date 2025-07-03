import streamlit as st
import joblib
import re
from sentence_transformers import SentenceTransformer

# Load BERT model only once
@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedder = load_embedder()

@st.cache_resource
def load_models():
    return {
        'IE': joblib.load('mbti_models/model_IE.joblib'),
        'NS': joblib.load('mbti_models/model_NS.joblib'),
        'TF': joblib.load('mbti_models/model_TF.joblib'),
        'JP': joblib.load('mbti_models/model_JP.joblib'),
    }

models = load_models()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\|\|\|', ' ', text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_mbti(text):
    clean = clean_text(text)
    emb = embedder.encode([clean])[0]

    mbti = ""
    mbti += "I" if models['IE'].predict([emb])[0] == 1 else "E"
    mbti += "N" if models['NS'].predict([emb])[0] == 1 else "S"
    mbti += "T" if models['TF'].predict([emb])[0] == 1 else "F"
    mbti += "J" if models['JP'].predict([emb])[0] == 1 else "P"
    
    return mbti

# Personality trait descriptions (simple example)
mbti_descriptions = {
    "INTP": "Thinkers: Analytical, logical, and curious.",
    "INFJ": "Advocates: Insightful, altruistic, and idealistic.",
    "ENFP": "Campaigners: Enthusiastic, creative, and sociable.",
    "ESTJ": "Executives: Organized, direct, and practical.",
    # Add more descriptions as you want
}

st.sidebar.title("About")
st.sidebar.info("""
This app predicts your MBTI personality type  
based on the text you enter.  
Using logistic regression models on BERT embeddings.
""")

st.title("🔮 MBTI Personality Predictor")

st.markdown("Type or paste text that reflects your personality or writing style. For example, a short paragraph about your interests or how you think.")

user_input = st.text_area("📝 Your Text Here:", height=200)

if st.button("Predict MBTI"):
    if user_input.strip() == "":
        st.warning("Please enter some text to predict your MBTI type.")
    else:
        mbti_type = predict_mbti(user_input)
        st.success(f"🎉 Predicted MBTI Type: **{mbti_type}**")
        desc = mbti_descriptions.get(mbti_type, "Description not available for this type yet.")
        st.info(desc)
