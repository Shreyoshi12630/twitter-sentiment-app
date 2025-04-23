import streamlit as st
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import pickle

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="💬 Tweet Sentiment Predictor", layout="centered")

# ========== CSS STYLE ==========
st.markdown("""
    <style>
    body {
        background-color: #FAF9F6;
        color: #222831;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        max-width: 700px;
        margin: auto;
    }
    .title {
        font-size: 2.5rem;
        text-align: center;
        color: #6C4AB6;
        font-weight: bold;
        margin-top: 1rem;
    }
    .subtitle {
        font-size: 1.3rem;
        text-align: center;
        color: #FF6B35;
        margin-bottom: 2rem;
    }
    .stTextInput > div > div > input, .stTextArea textarea {
        background-color: #ffffff;
        color: #000000;
        border: 2px solid #6C4AB6;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton > button {
        background-color: #6C4AB6;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ========== TITLES ==========
st.markdown("<div class='title'>🧠 Try Predicting a Tweet Sentiment</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>💬 Enter a tweet and get its Predicted Sentiment!</div>", unsafe_allow_html=True)

# ========== LOAD TRAINED COMPONENTS ==========
model = load_model("model.h5")
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# ========== CLEAN FUNCTION ==========
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ========== PREDICT FUNCTION ==========
def predict_sentiment(tweet):
    cleaned = clean_text(tweet)
    vect = vectorizer.transform([cleaned]).toarray()
    pred = model.predict(vect)[0]
    return encoder.inverse_transform([np.argmax(pred)])[0]

# ========== INPUT ==========
tweet_input = st.text_area("📝 Write your tweet here...", height=150)

if st.button("🔍 Predict Sentiment"):
    if tweet_input.strip() == "":
        st.warning("⚠️ Please enter a tweet first!")
    else:
        sentiment = predict_sentiment(tweet_input)
        emoji_map = {
            "positive": "😊💖",
            "negative": "😞💔",
            "neutral": "😐🌀"
        }
        emoji = emoji_map.get(sentiment.lower(), "🔍")
        st.success(f"{emoji} **Predicted Sentiment:** `{sentiment.upper()}`")
