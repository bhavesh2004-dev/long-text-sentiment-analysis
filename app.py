import streamlit as st
from transformers import pipeline, AutoTokenizer
import numpy as np


# CONFIG
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_LENGTH = 512
BATCH_SIZE = 4

# LOAD MODEL (CACHE FOR SPEED)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    classifier = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=tokenizer
    )
    return tokenizer, classifier

tokenizer, classifier = load_model()

# CHUNKING FUNCTION

def chunk_text(text, tokenizer, max_length=512):
    token_ids = tokenizer(
        text,
        truncation=False,
        add_special_tokens=False
    )["input_ids"]

    chunks = []
    for i in range(0, len(token_ids), max_length):
        chunk_ids = token_ids[i:i + max_length]
        chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk)

    return chunks


# AGGREGATE SENTIMENT
def aggregate_sentiment(chunk_results):
    scores = []

    for result in chunk_results:
        if result["label"] == "POSITIVE":
            scores.append(result["score"])
        else:
            scores.append(-result["score"])

    avg_score = np.mean(scores)
    final_label = "POSITIVE" if avg_score > 0 else "NEGATIVE"

    return final_label, abs(avg_score)

# STREAMLIT UI
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")

st.title("🧠 Sentiment Analysis (Chunking Based)")
st.write(
    "This app analyzes sentiment for **short and long text** using "
    "a Transformer model with **chunking** to handle long inputs."
)

user_text = st.text_area(
    "Enter your text below:",
    height=200,
    placeholder="Paste a review, feedback, or paragraph here..."
)
token_count = len(tokenizer(user_text, add_special_tokens=False)["input_ids"])
st.caption(f"Token count: {token_count}")


if st.button("Analyze Sentiment"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing sentiment..."):
            chunks = chunk_text(user_text, tokenizer, MAX_LENGTH)
            chunk_results = classifier(chunks, batch_size=BATCH_SIZE)
            final_label, confidence = aggregate_sentiment(chunk_results)

        st.subheader("Result")
        if final_label == "POSITIVE":
            st.success(f"Sentiment: {final_label}")
        else:
            st.error(f"Sentiment: {final_label}")

        st.write(f"Confidence Score: **{round(confidence, 3)}**")

        st.caption(f"Processed in {len(chunks)} chunk(s)")
