# InsightForge: GenAI-Powered Customer Insight Tool

Turns raw Amazon product reviews into structured, LLM-generated insight cards (top praises,
complaints, recurring themes, recommended action) — plus a classical VADER sentiment baseline
and an interactive Streamlit dashboard, including a "paste your own review" live-analysis demo.

## Why this project

Product and planning teams can't read thousands of reviews per product. This project builds
the pipeline that does it for them: structured, consistent, and fast — the kind of tool a
Data Analytics team would build to turn unstructured customer feedback into decisions.

## Dataset

Amazon consumer electronics reviews (~34,600 reviews across 48 products — Fire tablets, Echo,
Kindle, etc.), sourced from a public GitHub mirror of Amazon's product review data.

## Approach

1. **Clean** review data (drop missing text/ratings, normalize product names)
2. **Classical baseline** — score every review with VADER sentiment (free, instant, validated
   against star ratings before trusting it)
3. **GenAI layer** — for each product, sample reviews and prompt an LLM to return **strict JSON**:
   top praises, top complaints, recurring themes, overall sentiment, and one recommended action
4. **Compare** the quantitative (VADER) and qualitative (LLM) signals side by side
5. **Dashboard** — Streamlit app to browse insight cards per product and run live analysis on
   any pasted review text

## LLM provider: Groq

This project calls the LLM via [Groq](https://console.groq.com), which offers a **permanent free
tier** (14,400 requests/day) serving fast open-weight models like Llama 3.3 70B.

**To enable live LLM calls:**
1. Go to console.groq.com → sign up (free) → API Keys → Create Key
2. Set it as an environment variable: `export GROQ_API_KEY=your_key_here`
   (or put it in a `.env` file — see `.env.example`)
3. Re-run the notebook / launch the dashboard — it will automatically use live calls

**Without a key:** the notebook and dashboard still run end-to-end using a lightweight
keyword-frequency fallback, clearly labeled as offline mode, so the project is always
demonstrable even without setup.

## Project structure

```
genai_insights/
├── data/
│   └── amazon_reviews.csv           # raw review data
├── notebooks/
│   └── genai_insight_tool.ipynb     # full pipeline, executed with outputs
├── outputs/
│   ├── product_insight_cards.json   # LLM-generated (or fallback) insight cards
│   ├── product_sentiment_summary.csv
│   └── reviews_with_sentiment.csv
├── app/
│   └── dashboard.py                 # Streamlit dashboard incl. live review analysis
├── .env.example
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt

# optional but recommended: enable live LLM calls
export GROQ_API_KEY=your_key_here

# Explore the analysis
jupyter notebook notebooks/genai_insight_tool.ipynb

# Launch the dashboard
cd app
streamlit run dashboard.py
```

## Tech stack

Python · pandas · VADER Sentiment · Groq API (Llama 3.3 70B) via OpenAI-compatible client ·
Streamlit · Plotly
