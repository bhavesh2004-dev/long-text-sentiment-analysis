import streamlit as st
from transformers import pipeline, AutoTokenizer
import numpy as np

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_LENGTH = 512
BATCH_SIZE = 4

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentIQ",
    page_icon="◎",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #1a1a2e;
}
.stApp {
    background: #f5f4f0;
    background-image:
        radial-gradient(circle at 20% 20%, rgba(218,210,195,0.4) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(200,215,210,0.3) 0%, transparent 50%);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 3rem;
    padding-bottom: 4rem;
    max-width: 800px;
}

/* ── Masthead ── */
.masthead {
    text-align: center;
    margin-bottom: 2.5rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid #dddad4;
}
.masthead .eyebrow {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #7a7060;
    background: #eae8e2;
    border: 1px solid #d8d4cc;
    padding: 0.28rem 0.9rem;
    border-radius: 2rem;
    margin-bottom: 1.1rem;
}
.masthead h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    font-weight: 400;
    letter-spacing: -0.02em;
    color: #12111a;
    margin: 0 0 0.6rem 0;
    line-height: 1.1;
}
.masthead h1 em {
    font-style: italic;
    color: #4a6fa5;
}
.masthead .subtitle {
    font-size: 0.93rem;
    color: #7a7060;
    font-weight: 300;
    line-height: 1.6;
}

/* ── Section label ── */
.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #9a9288;
    margin-bottom: 0.7rem;
}

/* ── Example tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent;
    border-bottom: 1.5px solid #e2dfd8;
    padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #8a8070 !important;
    background: transparent !important;
    border: 1px solid #e2dfd8 !important;
    border-bottom: none !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 0.45rem 1.1rem !important;
    transition: all 0.15s ease !important;
}
.stTabs [aria-selected="true"] {
    color: #12111a !important;
    background: #ffffff !important;
    border-color: #d0ccc4 !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #ffffff;
    border: 1.5px solid #d0ccc4;
    border-top: none;
    border-radius: 0 0 14px 14px;
    padding: 1.4rem 1.6rem 1.2rem;
}

/* ── Example text inside panel ── */
.example-preview {
    font-size: 0.88rem;
    line-height: 1.7;
    color: #3a3830;
    background: #faf9f7;
    border: 1px dashed #d8d4cc;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.9rem;
    font-style: italic;
}
.example-tag {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.2rem 0.65rem;
    border-radius: 2rem;
    margin-bottom: 0.6rem;
}
.tag-positive { background: #e8f4f0; color: #2d7a4f; border: 1px solid #a8d5b0; }
.tag-negative { background: #fdf2f2; color: #a33030; border: 1px solid #f0b8b8; }
.tag-mixed    { background: #fdf6e8; color: #8a5e20; border: 1px solid #e8cc88; }

/* ── Main textarea ── */
.stTextArea textarea {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.93rem !important;
    line-height: 1.7 !important;
    background: #ffffff !important;
    border: 1.5px solid #dddad4 !important;
    border-radius: 12px !important;
    color: #1a1a2e !important;
    padding: 1rem 1.1rem !important;
    transition: border-color 0.2s ease !important;
}
.stTextArea textarea:focus {
    border-color: #4a6fa5 !important;
    box-shadow: 0 0 0 3px rgba(74,111,165,0.1) !important;
}

/* ── Token badge ── */
.token-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.75rem;
    color: #7a7060;
    background: #f0ede8;
    border: 1px solid #e2dfd8;
    padding: 0.22rem 0.7rem;
    border-radius: 2rem;
    margin-top: 0.45rem;
}
.token-badge .dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #b0c4de;
}

/* ── Analyze button ── */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    background: #12111a !important;
    color: #f5f4f0 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #2a2840 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(18,17,26,0.2) !important;
}

/* ── Result card ── */
.result-positive {
    background: linear-gradient(135deg, #f0f7f0 0%, #e8f4f0 100%);
    border: 1.5px solid #a8d5b0;
    border-radius: 16px;
    padding: 2rem 2.2rem;
    margin-top: 1.4rem;
}
.result-negative {
    background: linear-gradient(135deg, #fdf2f2 0%, #fae8e8 100%);
    border: 1.5px solid #f0b8b8;
    border-radius: 16px;
    padding: 2rem 2.2rem;
    margin-top: 1.4rem;
}
.result-verdict-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.result-verdict-label.pos { color: #2d7a4f; }
.result-verdict-label.neg { color: #a33030; }
.result-sentiment {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    font-weight: 400;
    line-height: 1;
    margin-bottom: 1.4rem;
}
.result-sentiment.pos { color: #1d5c3a; }
.result-sentiment.neg { color: #7a2020; }
.conf-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: #5a5550;
    font-weight: 500;
    margin-bottom: 0.4rem;
}
.conf-bar-bg {
    background: rgba(255,255,255,0.6);
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
    margin-bottom: 1.2rem;
}
.conf-bar-pos {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #4caf7d, #2d7a4f);
}
.conf-bar-neg {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #e57373, #a33030);
}
.stats-row {
    display: flex;
    gap: 0.8rem;
}
.stat-pill {
    flex: 1;
    background: rgba(255,255,255,0.7);
    border: 1px solid rgba(0,0,0,0.07);
    border-radius: 10px;
    padding: 0.7rem 0.8rem;
    text-align: center;
}
.stat-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #12111a;
    line-height: 1;
}
.stat-key {
    font-size: 0.65rem;
    color: #9a9288;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* ── Warning ── */
.warn-box {
    text-align: center;
    padding: 1.1rem;
    background: #fdf8f0;
    border: 1.5px solid #f0ddb0;
    border-radius: 12px;
    color: #8a6820;
    font-size: 0.86rem;
    margin-top: 1rem;
}

/* ── Footer ── */
.model-footer {
    text-align: center;
    margin-top: 2.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid #dddad4;
    font-size: 0.73rem;
    color: #b0a898;
    letter-spacing: 0.04em;
}
.model-footer strong { color: #8a8070; }

.stSpinner > div { border-top-color: #4a6fa5 !important; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    classifier = pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=tokenizer)
    return tokenizer, classifier

tokenizer, classifier = load_model()

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def chunk_text(text, tokenizer, max_length=512):
    token_ids = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = []
    for i in range(0, len(token_ids), max_length):
        chunk = tokenizer.decode(token_ids[i:i + max_length], skip_special_tokens=True)
        chunks.append(chunk)
    return chunks

def aggregate_sentiment(chunk_results):
    scores = [r["score"] if r["label"] == "POSITIVE" else -r["score"] for r in chunk_results]
    avg = np.mean(scores)
    return ("POSITIVE" if avg > 0 else "NEGATIVE"), abs(avg)

# ─── EXAMPLES ─────────────────────────────────────────────────────────────────
EXAMPLES = [
    {
        "tab": "😊 Positive",
        "tag": "tag-positive",
        "tag_label": "Likely Positive",
        "text": (
            "I recently visited this restaurant and it was an absolutely delightful experience. "
            "The food was fresh, flavorful, and beautifully presented. The staff were attentive "
            "and warm, making us feel genuinely welcomed. I'll definitely be coming back!"
        ),
    },
    {
        "tab": "😠 Negative",
        "tag": "tag-negative",
        "tag_label": "Likely Negative",
        "text": (
            "Terrible experience from start to finish. The product arrived broken, customer "
            "support was completely unresponsive, and when I finally reached someone they were "
            "rude and dismissive. I would not recommend this company to anyone."
        ),
    },
    {
        "tab": "😐 Mixed",
        "tag": "tag-mixed",
        "tag_label": "Mixed Sentiment",
        "text": (
            "The conference had some genuinely inspiring talks, and I appreciated the effort "
            "put into the venue. However, the scheduling was chaotic, several sessions ran "
            "over time, and the catering left a lot to be desired. Overall a mixed bag."
        ),
    },
]

# ─── SESSION STATE ────────────────────────────────────────────────────────────
if "user_text" not in st.session_state:
    st.session_state.user_text = ""

# ─── MASTHEAD ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
    <div class="eyebrow">◎ NLP · Transformer-Based</div>
    <h1>Sentiment<em>IQ</em></h1>
    <p class="subtitle">
        Understand the emotional tone of any text — short or long —<br>
        powered by chunked transformer inference.
    </p>
</div>
""", unsafe_allow_html=True)

# ─── EXAMPLE TABS ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Quick Examples — click to load</p>', unsafe_allow_html=True)

tabs = st.tabs([ex["tab"] for ex in EXAMPLES])
for i, (tab, ex) in enumerate(zip(tabs, EXAMPLES)):
    with tab:
        st.markdown(
            f'<span class="example-tag {ex["tag"]}">{ex["tag_label"]}</span>'
            f'<div class="example-preview">{ex["text"]}</div>',
            unsafe_allow_html=True
        )
        if st.button("Use this example →", key=f"use_{i}"):
            st.session_state.user_text = ex["text"]
            st.rerun()

# ─── TEXT INPUT ───────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top:1.6rem'></div>", unsafe_allow_html=True)
st.markdown('<p class="section-label">Or type / paste your own text</p>', unsafe_allow_html=True)

user_text = st.text_area(
    label="",
    value=st.session_state.user_text,
    height=180,
    placeholder="Paste a review, article, feedback, or any paragraph here…",
    label_visibility="collapsed",
)

token_count = len(tokenizer(user_text, add_special_tokens=False)["input_ids"]) if user_text.strip() else 0
chunk_preview = max(1, -(-token_count // MAX_LENGTH)) if token_count else 1

st.markdown(f"""
<div class="token-badge">
    <span class="dot"></span>
    {token_count} tokens &nbsp;·&nbsp; {chunk_preview} chunk{"s" if chunk_preview != 1 else ""} estimated
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
analyze = st.button("Analyze Sentiment →")

# ─── RESULTS ──────────────────────────────────────────────────────────────────
if analyze:
    if not user_text.strip():
        st.markdown(
            '<div class="warn-box">⚠ Please enter or select some text to analyze.</div>',
            unsafe_allow_html=True
        )
    else:
        with st.spinner("Running inference…"):
            chunks = chunk_text(user_text, tokenizer, MAX_LENGTH)
            chunk_results = classifier(chunks, batch_size=BATCH_SIZE)
            final_label, confidence = aggregate_sentiment(chunk_results)

        conf_pct = round(confidence * 100, 1)
        is_pos   = final_label == "POSITIVE"
        card_cls = "result-positive" if is_pos else "result-negative"
        lbl_cls  = "pos" if is_pos else "neg"
        emoji    = "◆" if is_pos else "◇"
        bar_cls  = "conf-bar-pos" if is_pos else "conf-bar-neg"

        st.markdown(f"""
<div class="{card_cls}">
    <div class="result-verdict-label {lbl_cls}">Sentiment Verdict</div>
    <div class="result-sentiment {lbl_cls}">{emoji} {final_label.capitalize()}</div>
    <div class="conf-header">
        <span>Confidence</span><span>{conf_pct}%</span>
    </div>
    <div class="conf-bar-bg">
        <div class="{bar_cls}" style="width:{conf_pct}%"></div>
    </div>
    <div class="stats-row">
        <div class="stat-pill">
            <div class="stat-val">{len(chunks)}</div>
            <div class="stat-key">Chunks</div>
        </div>
        <div class="stat-pill">
            <div class="stat-val">{token_count}</div>
            <div class="stat-key">Tokens</div>
        </div>
        <div class="stat-pill">
            <div class="stat-val">{conf_pct}%</div>
            <div class="stat-key">Confidence</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="model-footer">
    Model &nbsp;·&nbsp; <strong>{MODEL_NAME}</strong>
    &nbsp;·&nbsp; Max chunk {MAX_LENGTH} tokens &nbsp;·&nbsp; Batch size {BATCH_SIZE}
</div>
""", unsafe_allow_html=True)