# Long Text Sentiment Analysis (Transformer-Based)

A **Transformer-based sentiment analysis system** designed to handle **long text inputs beyond token limits** using a chunking and aggregation strategy.

The project demonstrates a practical NLP engineering solution using a pre-trained Hugging Face model and includes both a CLI script and a Streamlit web interface.

---

## Overview

Transformer models have a fixed token limit, which makes analyzing long documents challenging.  
This project solves the problem by:

1. Splitting long text into manageable chunks
2. Running sentiment analysis on each chunk
3. Aggregating chunk-level predictions into a final sentiment score

---

## Key Features

- Handles long text beyond Transformer token limits
- Chunk-based text processing
- Sentiment aggregation logic
- Pre-trained Hugging Face Transformer model
- Streamlit-based interactive UI
- CLI-based batch evaluation support

---

## How It Works

1. Input text is tokenized without truncation
2. Tokens are split into fixed-size chunks
3. Each chunk is analyzed independently
4. Sentiment scores are aggregated
5. Final sentiment and confidence score are returned

---

## Tech Stack

- Python
- Hugging Face Transformers
- Datasets (Hugging Face)
- NumPy
- Streamlit

---


## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt

# 2. Run Streamlit app
streamlit run app.py

# 3. Run CLI script
python main.py
