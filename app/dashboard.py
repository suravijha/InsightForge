"""
GenAI-Powered Customer Insight Dashboard
-----------------------------------------
Interactive Streamlit app for exploring LLM-generated product insight cards
and running live sentiment/theme extraction on new review text.

Run with: streamlit run dashboard.py
Requires GROQ_API_KEY environment variable for live LLM features
(free key: console.groq.com). Falls back gracefully if not set.
"""

import os
import json
import re

import pandas as pd
import streamlit as st
import plotly.express as px

from dotenv import load_dotenv
load_dotenv()  # Load GROQ_API_KEY from .env file if present

st.set_page_config(page_title="GenAI Product Insight Dashboard", layout="wide")

GROQ_MODEL = "llama-3.3-70b-versatile"


@st.cache_resource
def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


@st.cache_data
def load_data():
    summary = pd.read_csv("../outputs/product_sentiment_summary.csv")
    with open("../outputs/product_insight_cards.json") as f:
        cards = json.load(f)
    reviews = pd.read_csv("../outputs/reviews_with_sentiment.csv")
    return summary, cards, reviews


client = get_client()
summary_df, insight_cards, reviews_df = load_data()

st.title("🤖 GenAI-Powered Customer Insight Dashboard")
st.caption(
    "LLM-generated product insight cards (Groq / Llama 3.3 70B) + a classical VADER "
    "sentiment baseline, built on Amazon product reviews."
)

if client is None:
    st.warning(
        "No `GROQ_API_KEY` found — showing pre-generated / offline-mode insight cards. "
        "Get a free key at console.groq.com and set it as an environment variable to "
        "enable live analysis below.",
        icon="⚠️",
    )

# ---------------- Product overview ----------------
st.subheader("Product Overview")

col1, col2 = st.columns(2)
with col1:
    fig_rating = px.bar(
        summary_df.sort_values("avg_rating"),
        x="avg_rating", y="product", orientation="h",
        title="Average Star Rating by Product",
        color="avg_rating", color_continuous_scale="Blues",
    )
    st.plotly_chart(fig_rating, use_container_width=True)

with col2:
    fig_sent = px.bar(
        summary_df.sort_values("avg_vader_score"),
        x="avg_vader_score", y="product", orientation="h",
        title="Average VADER Sentiment by Product",
        color="avg_vader_score", color_continuous_scale="Oranges",
    )
    st.plotly_chart(fig_sent, use_container_width=True)

# ---------------- Insight cards ----------------
st.divider()
st.subheader("LLM-Generated Insight Cards")

product = st.selectbox("Select a product", options=list(insight_cards.keys()))
card = insight_cards[product]

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**✅ Top Praises**")
    for p in card.get("top_praises", []):
        st.markdown(f"- {p}")
with c2:
    st.markdown("**⚠️ Top Complaints**")
    for cpl in card.get("top_complaints", []):
        st.markdown(f"- {cpl}")
with c3:
    st.markdown("**🔁 Recurring Themes**")
    for t in card.get("recurring_themes", []):
        st.markdown(f"- {t}")

st.info(f"**Recommended Action:** {card.get('recommended_action', 'n/a')}")
st.caption(f"Overall sentiment: **{card.get('overall_sentiment', 'n/a')}**")

# ---------------- Live analysis ----------------
st.divider()
st.subheader("Try It Live — Analyze a New Review")

user_review = st.text_area(
    "Paste any product review text:",
    placeholder="e.g. 'Battery life is way shorter than advertised, but the screen quality is amazing...'",
    height=100,
)

if st.button("Analyze with LLM", type="primary"):
    if not user_review.strip():
        st.error("Paste some review text first.")
    elif client is None:
        st.error("No GROQ_API_KEY configured — set one to enable live analysis.")
    else:
        with st.spinner("Calling Groq..."):
            prompt = f"""Analyze this product review and return ONLY valid JSON (no markdown):
{{
  "sentiment": "positive" | "mixed" | "negative",
  "key_points": ["...", "..."],
  "one_line_summary": "..."
}}

Review: {user_review}
"""
            try:
                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=300,
                )
                raw = response.choices[0].message.content.strip()
                raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
                result = json.loads(raw)
                st.success(f"Sentiment: **{result['sentiment']}**")
                st.write(result["one_line_summary"])
                for kp in result.get("key_points", []):
                    st.markdown(f"- {kp}")
            except Exception as e:
                st.error(f"Error calling LLM: {e}")

# ---------------- Raw data ----------------
st.divider()
st.subheader("Review-Level Data")
selected_product_reviews = reviews_df[reviews_df["product_short"] == product]
st.dataframe(
    selected_product_reviews[["rating", "vader_score", "text"]].sort_values(
        "vader_score", ascending=False
    ).reset_index(drop=True),
    use_container_width=True,
)
