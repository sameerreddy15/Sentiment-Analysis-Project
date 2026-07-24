import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import re

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="🎬 Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

.title{
    text-align:center;
    color:#00E5FF;
    font-size:42px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:white;
    font-size:18px;
}

.result-positive{
    background:#0f5132;
    padding:20px;
    border-radius:12px;
    color:white;
    font-size:24px;
    text-align:center;
}

.result-negative{
    background:#842029;
    padding:20px;
    border-radius:12px;
    color:white;
    font-size:24px;
    text-align:center;
}

.info-box{
    background:#1e293b;
    padding:15px;
    border-radius:10px;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sentiment_analysis.keras")

model = load_model()

# -----------------------------
# IMDb Word Index
# -----------------------------
word_index = imdb.get_word_index()

VOCAB_SIZE = 50000
MAX_LENGTH = 200

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="title">🎬 Movie Review Sentiment Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Predict whether a movie review is Positive or Negative using a Bidirectional LSTM model.</div>',
    unsafe_allow_html=True
)

st.write("")

#part 2
# -----------------------------
# Text Preprocessing
# -----------------------------
def preprocess_review(review):

    review = review.lower()

    review = re.sub(r"[^a-zA-Z0-9\s]", "", review)

    words = review.split()

    encoded_review = []

    for word in words:

        if word in word_index:

            index = word_index[word] + 3

            if index < VOCAB_SIZE:
                encoded_review.append(index)

        else:
            encoded_review.append(2)

    padded_review = pad_sequences(
        [encoded_review],
        maxlen=MAX_LENGTH
    )

    return padded_review


# -----------------------------
# Prediction Function
# -----------------------------
def predict_sentiment(review):

    processed_review = preprocess_review(review)

    prediction = model.predict(
        processed_review,
        verbose=0
    )[0][0]

    positive_probability = float(prediction)
    negative_probability = 1 - positive_probability

    if positive_probability >= 0.5:
        sentiment = "Positive 😊"
        confidence = positive_probability * 100
    else:
        sentiment = "Negative 😞"
        confidence = negative_probability * 100

    return (
        sentiment,
        confidence,
        positive_probability,
        negative_probability
    )


# -----------------------------
# Initialize Prediction History
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

#part 3
    # -----------------------------
# User Input
# -----------------------------
st.write("")

review = st.text_area(
    "📝 Enter Movie Review",
    height=180,
    placeholder="Example: This movie was amazing. The acting and story were excellent..."
)

col1, col2 = st.columns(2)

with col1:
    predict = st.button("🚀 Analyze Sentiment", use_container_width=True)

with col2:
    clear = st.button("🗑 Clear", use_container_width=True)

if clear:
    st.rerun()

# -----------------------------
# Prediction
# -----------------------------
if predict:

    if review.strip() == "":
        st.warning("⚠ Please enter a movie review.")

    else:

        with st.spinner("Analyzing review..."):

            sentiment, confidence, positive, negative = predict_sentiment(review)

        st.write("")

        if "Positive" in sentiment:

            st.markdown(
                f"""
                <div class="result-positive">
                😊 <b>{sentiment}</b><br><br>
                Confidence: {confidence:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result-negative">
                😞 <b>{sentiment}</b><br><br>
                Confidence: {confidence:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        st.subheader("📊 Confidence")

        st.progress(confidence / 100)

        chart = pd.DataFrame(
            {
                "Probability": [
                    positive * 100,
                    negative * 100
                ]
            },
            index=["Positive", "Negative"]
        )

        st.bar_chart(chart)

        st.session_state.history.append(
            {
                "Review": review,
                "Prediction": sentiment,
                "Confidence": round(confidence, 2)
            }
        )

# -----------------------------
# Example Reviews
# -----------------------------
st.divider()

st.subheader("📝 Example Reviews")

st.info("Positive: This movie was amazing with brilliant acting and an excellent story.")

st.info("Negative: This movie was boring, slow and a complete waste of time.")
# -----------------------------
# Model Information
# -----------------------------
st.divider()

st.subheader("ℹ Model Information")

st.markdown("""
<div class="info-box">
<b>Model:</b> Bidirectional LSTM<br><br>

<b>Dataset:</b> IMDb Movie Reviews<br><br>

<b>Vocabulary Size:</b> 50,000 words<br><br>

<b>Maximum Review Length:</b> 200 words<br><br>

<b>Output:</b> Positive or Negative Sentiment
</div>
""", unsafe_allow_html=True)
