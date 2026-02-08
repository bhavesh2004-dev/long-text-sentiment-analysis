
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
import numpy as np

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_LENGTH = 512
BATCH_SIZE = 4

# LOAD TOKENIZER & MODEL

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

classifier = pipeline(
    task="sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=tokenizer
)

# FUNCTION: SPLIT TEXT INTO CHUNKS

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

# FUNCTION: AGGREGATE SENTIMENT

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

# LOAD DATASET (HUGGING FACE)
dataset = load_dataset("imdb", split="test[:5]")


# RUN SENTIMENT ANALYSIS

for idx, sample in enumerate(dataset):
    text = sample["text"]

    chunks = chunk_text(text, tokenizer, MAX_LENGTH)
    chunk_results = classifier(chunks, batch_size=BATCH_SIZE)

    final_label, confidence = aggregate_sentiment(chunk_results)

    print(f"\nReview {idx + 1}")
    print("Final Sentiment :", final_label)
    print("Confidence      :", round(confidence, 3))
    print("-" * 60)
