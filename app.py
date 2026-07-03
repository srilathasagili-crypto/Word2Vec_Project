import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------

st.set_page_config(
    page_title="Twitter Airline Sentiment Analysis",
    page_icon="✈️",
    layout="centered"
)

# ---------------------------------------------------
# Load Files
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sentiment_model.keras")


@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_encoder():
    with open("label_encoder.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_embeddings():
    with open("embedding_lookup.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()
scaler = load_scaler()
encoder = load_encoder()
embedding_lookup = load_embeddings()

# ---------------------------------------------------
# Text Cleaning
# ---------------------------------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", " ", text)

    text = re.sub(r"@\w+", " @ ", text)

    text = re.sub(r"[^a-z@ ]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text.split()

# ---------------------------------------------------
# Sentence Embedding
# ---------------------------------------------------

def sentence_vector(sentence):

    words = clean_text(sentence)

    vectors = []

    for word in words:

        if word in embedding_lookup:
            vectors.append(embedding_lookup[word])

    if len(vectors) == 0:
        return np.zeros(200)

    return np.mean(vectors, axis=0)

# ---------------------------------------------------
# UI
# ---------------------------------------------------

st.title("✈️ Twitter Airline Sentiment Analysis")

st.write(
    "Predict whether an airline-related tweet expresses "
    "Negative, Neutral, or Positive sentiment."
)

tweet = st.text_area(
    "Enter Tweet",
    height=150,
    placeholder="Example: I loved the customer service today!"
)

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

if st.button("Predict Sentiment"):

    if tweet.strip() == "":

        st.warning("Please enter a tweet.")

    else:

        vector = sentence_vector(tweet)

        vector = scaler.transform(vector.reshape(1, -1))

        prediction = model.predict(vector, verbose=0)

        pred_index = np.argmax(prediction)

        sentiment = encoder.inverse_transform([pred_index])[0]

        confidence = prediction[0][pred_index]

        if sentiment == "positive":
            st.success(f"😊 Prediction: {sentiment.capitalize()}")

        elif sentiment == "neutral":
            st.info(f"😐 Prediction: {sentiment.capitalize()}")

        else:
            st.error(f"😠 Prediction: {sentiment.capitalize()}")

        st.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

        st.subheader("Prediction Probabilities")

        for label, prob in zip(encoder.classes_, prediction[0]):

            st.progress(float(prob))

            st.write(f"{label.capitalize()} : {prob*100:.2f}%")