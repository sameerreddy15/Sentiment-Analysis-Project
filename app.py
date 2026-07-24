import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Set Page Config (optional but recommended for Streamlit)
st.set_page_config(page_title="Movie Review Sentiment Analysis", page_icon="🎬", layout="wide")

import tensorflow as tf
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

model = tf.keras.models.load_model("sentiment_analysis.keras")

word_index = imdb.get_word_index()

vocab_size = 50000
max_length = 200


if "history" not in st.session_state:
    st.session_state.history = []

# ==========================================
# PREDICTION FUNCTION 
# ==========================================
positive_words = {
    "blockbuster", "decent", "highlight", "excellent", "awesome",
    "amazing", "fantastic", "superb", "brilliant", "masterpiece",
    "great", "good", "best", "love", "loved", "enjoyed",
    "worth", "recommended", "outstanding", "perfect",
    "beautiful", "wonderful", "super", "hit",
    "classic", "incredible", "fabulous", "epic","mindblowing"
}

negative_words = {
    "bad", "worst", "boring", "terrible", "awful",
    "poor", "waste", "disappointing", "hate", "hated",
    "horrible", "flop", "disaster", "useless",
    "dull", "pathetic", "mediocre",
    "predictable", "slow", "weak",
    "failure", "forgettable", "annoying"
}


def predict_sentiment(review):

    review = review.lower()
    review = re.sub(r"[^a-zA-Z0-9\s]", "", review)

    words = review.split()


    # Phrase-based rules (ADD HERE)
    negative_phrases = {
        "very bad",
        "bad movie",
        "worst movie",
        "waste of time",
        "not good",
        "not worth"
    }

    positive_phrases = {
        "very good",
        "great movie",
        "amazing movie",
        "excellent movie",
        "loved this movie"
    }


    for phrase in negative_phrases:
        if phrase in review:
            return "Negative Review", 95.0, 0.05, 0.95


    for phrase in positive_phrases:
        if phrase in review:
            return "Positive Review", 95.0, 0.95, 0.05

    # Rule-based for single word
    if len(words) == 1:

        if words[0] in positive_words:
            return "Positive Review", 99.0, 0.99, 0.01

        if words[0] in negative_words:
            return "Negative Review", 99.0, 0.01, 0.99


    encoded_review = []

    for word in words:
        if word in word_index and word_index[word] < vocab_size:
            encoded_review.append(word_index[word] + 3)
        else:
            encoded_review.append(2)


    padded_review = pad_sequences(
        [encoded_review],
        maxlen=max_length,
        padding="post",
        truncating="post"
    )


    prediction = model.predict(
        padded_review,
        verbose=0
    )[0][0]


    if prediction >= 0.5:

        sentiment = "Positive Review"
        confidence = prediction * 100
        positive = prediction
        negative = 1 - prediction

    else:

        sentiment = "Negative Review"
        confidence = (1 - prediction) * 100
        positive = prediction
        negative = 1 - prediction


    return sentiment, confidence, positive, negative
# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("🎬 Sentiment Analysis")

option = st.sidebar.radio(
    "Choose Prediction Mode",
    (
        "Single Review",
        "Multiple Reviews",
        "CSV Upload"
    )
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Details")
st.sidebar.write("**Model:** Bidirectional LSTM")
st.sidebar.write("**Dataset:** IMDb")
st.sidebar.write("**Vocabulary:** 50,000")
st.sidebar.write("**Sequence Length:** 200")
st.sidebar.write("**Framework:** TensorFlow")

st.sidebar.markdown("---")
st.sidebar.subheader("🎬 Sample Reviews")

samples = {
    "Excellent Movie": "This movie was amazing with brilliant acting and an excellent storyline.",
    "Bad Movie": "This movie was boring and a complete waste of time.",
    "Average Movie": "The movie was decent with some good moments."
}

choice = st.sidebar.selectbox(
    "Load Sample Review",
    ["None"] + list(samples.keys())
)

sample_text = ""
if choice != "None":
    sample_text = samples[choice]

if st.sidebar.button("🗑 Clear History"):
    st.session_state.history = []
    st.sidebar.success("History Cleared")

# ==========================================
# MAIN HEADER
# ==========================================
st.title("🎬 Movie Review Sentiment Analysis")
st.caption("Predict movie review sentiment using a Bidirectional LSTM model")
st.divider()

# ==========================================
# SINGLE REVIEW MODE
# ==========================================
if option == "Single Review":

    st.subheader("📝 Single Review Analysis")
    
    review = st.text_area(
        "Enter Movie Review:",
        value=sample_text,
        height=180
    )

    if review.strip():
        st.markdown("**Entered Review:**")
        st.write(review)

    if st.button("🚀 Analyze"):

        if not review.strip():
            st.warning("Please enter a review.")
        else:
            start_time = time.time()
            sentiment, confidence, positive, negative = predict_sentiment(review)
            end_time = time.time()

            # Dynamic Emoji Indicator
            if confidence >= 90:
                emoji = "🌟"
            elif confidence >= 75:
                emoji = "😊"
            elif confidence >= 60:
                emoji = "🙂"
            else:
                emoji = "🤔"

            if "Positive" in sentiment:
                st.success(f"{emoji} {sentiment}")
            else:
                st.error(f"😞 {sentiment}")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Confidence", f"{confidence:.2f}%")
                st.progress(float(confidence) / 100)
            
            with col2:
                chart = pd.DataFrame(
                    {"Probability": [positive * 100, negative * 100]},
                    index=["Positive", "Negative"]
                )
                st.bar_chart(chart)

            st.caption(f"Prediction Time: {(end_time - start_time):.3f} seconds")

# ==========================================
# MULTIPLE REVIEWS MODE
# ==========================================
elif option == "Multiple Reviews":

    st.subheader("📝 Batch Analysis (Multiple Lines)")

    reviews = st.text_area(
        "Enter one review per line:",
        height=200,
        placeholder="""This movie was amazing
The acting was terrible
I loved every minute
Worst movie ever"""
    )

    if st.button("🚀 Analyze All Reviews"):

        if not reviews.strip():
            st.warning("Please enter at least one review.")
        else:
            review_list = [r.strip() for r in reviews.split("\n") if r.strip()]

            results = []
            positive_count = 0
            negative_count = 0

            for r in review_list:
                sentiment, confidence, pos, neg = predict_sentiment(r)
                results.append({
                    "Review": r,
                    "Prediction": sentiment,
                    "Confidence (%)": round(confidence, 2)
                })
                if "Positive" in sentiment:
                    positive_count += 1
                else:
                    negative_count += 1

            df = pd.DataFrame(results)

            st.subheader("Prediction Results")
            st.dataframe(df, use_container_width=True)

            total = len(review_list)
            positive_percent = (positive_count / total) * 100
            negative_percent = (negative_count / total) * 100

            st.subheader("Overall Sentiment Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Reviews", total)
            col2.metric("😊 Positive", f"{positive_percent:.2f}%")
            col3.metric("😞 Negative", f"{negative_percent:.2f}%")

            # Verdict Section
            if positive_percent >= 70:
                st.success("⭐ Overall Verdict: Highly Recommended Movie")
            elif positive_percent >= 50:
                st.info("👍 Overall Verdict: Good Movie")
            elif positive_percent >= 30:
                st.warning("😐 Overall Verdict: Average Movie")
            else:
                st.error("👎 Overall Verdict: Not Recommended")

            # Visualizations
            chart_col, pie_col = st.columns(2)

            with chart_col:
                chart = pd.DataFrame(
                    {"Reviews": [positive_count, negative_count]},
                    index=["Positive", "Negative"]
                )
                st.bar_chart(chart)

            with pie_col:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie(
                    [positive_count, negative_count],
                    labels=["Positive", "Negative"],
                    autopct="%1.1f%%",
                    startangle=90
                )
                ax.axis("equal")
                st.pyplot(fig)

            # File Downloads
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download CSV Results",
                csv_data,
                "prediction_results.csv",
                "text/csv"
            )

            report = f"""Movie Review Sentiment Analysis Report
--------------------------------------
Total Reviews      : {total}
Positive Reviews   : {positive_count} ({positive_percent:.2f}%)
Negative Reviews   : {negative_count} ({negative_percent:.2f}%)
"""
            st.download_button(
                "📄 Download Text Report",
                report,
                "report.txt"
            )

# ==========================================
# CSV UPLOAD MODE
# ==========================================
elif option == "CSV Upload":

    st.subheader("📂 Upload CSV File")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "review" not in df.columns and "text" in df.columns:
            df.rename(columns={"text":"review"}, inplace=True)

        st.write("### Preview")
        st.dataframe(df.head())

        st.info("The CSV must contain a column named 'review'.")

        if "review" not in df.columns:
            st.error("Column 'review' not found in uploaded file.")
        else:
            if st.button("🚀 Predict CSV Reviews"):

                predictions = []
                confidences = []

                positive_count = 0
                negative_count = 0

                progress = st.progress(0)
                total_rows = len(df)

                for i, rev in enumerate(df["review"]):
                    sentiment, confidence, pos, neg = predict_sentiment(str(rev))

                    predictions.append(sentiment)
                    confidences.append(round(confidence, 2))

                    if "Positive" in sentiment:
                        positive_count += 1
                    else:
                        negative_count += 1

                    progress.progress((i + 1) / total_rows)

                df["Prediction"] = predictions
                df["Confidence (%)"] = confidences

                st.success("Prediction Completed!")

                st.subheader("Prediction Results")
                st.dataframe(df, use_container_width=True)

                positive_percent = (positive_count / total_rows) * 100
                negative_percent = (negative_count / total_rows) * 100

                st.subheader("Overall Sentiment Summary")
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Reviews", total_rows)
                col2.metric("😊 Positive Reviews", f"{positive_percent:.2f}%")
                col3.metric("😞 Negative Reviews", f"{negative_percent:.2f}%")

                # Visualizations
                chart_col, pie_col = st.columns(2)
                
                with chart_col:
                    chart = pd.DataFrame(
                        {"Count": [positive_count, negative_count]},
                        index=["Positive", "Negative"]
                    )
                    st.bar_chart(chart)

                with pie_col:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.pie(
                        [positive_count, negative_count],
                        labels=["Positive", "Negative"],
                        autopct="%1.1f%%",
                        startangle=90
                    )
                    ax.axis("equal")
                    st.pyplot(fig)

                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Results CSV",
                    csv_data,
                    "prediction_results.csv",
                    "text/csv"
                )

# ==========================================
# FOOTER
# ==========================================
st.divider()

st.markdown(
    """
    ### 📌 About This Project
    This application predicts the sentiment of movie reviews using a
    **Bidirectional LSTM Deep Learning Model** trained on the **IMDb Movie Review Dataset**.

    **Technologies Used:**
    - 🐍 Python
    - 🤖 TensorFlow / Keras
    - 🎬 Bidirectional LSTM
    - 📊 Streamlit
    - 📝 IMDb Dataset
    """
)

st.caption("© 2026 Sameer | Movie Review Sentiment Analysis")
