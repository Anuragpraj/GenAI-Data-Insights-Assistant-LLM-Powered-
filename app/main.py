"""
AI-Powered Data Insights Assistant
Author : Anurag Kumar Prajapati
Stack  : Python · OpenAI GPT API · NLP (spaCy/NLTK) · Pandas · SQL · Streamlit
"""

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_loader import load_data, load_sql_table
from utils.nlp_pipeline import NLPPipeline
from utils.query_engine import QueryEngine
from utils.insight_generator import InsightGenerator
from utils.visualizer import (
    plot_distribution,
    plot_correlation_heatmap,
    plot_top_categories,
    plot_time_series,
    plot_outliers_boxplot,
)

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Insights Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .stTextInput > div > div > input { background-color: #1e2130; color: #e0e0e0; }
    .stButton > button {
        background: linear-gradient(135deg, #00e5ff, #7c3aed);
        color: #000; font-weight: 700; border: none; border-radius: 8px;
    }
    .insight-card {
        background: #1e2130; border-left: 3px solid #00e5ff;
        padding: 12px 16px; border-radius: 6px; margin-bottom: 10px;
    }
    .metric-box {
        background: #1a1d2e; border: 1px solid #2a2d3e;
        border-radius: 8px; padding: 14px; text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Session State Init ──────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None
if "nlp" not in st.session_state:
    st.session_state.nlp = NLPPipeline()


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64
    )
    st.title("⚡ AI Insights Assistant")
    st.caption("By Anurag Kumar Prajapati")
    st.divider()

    # API Key
    api_key = st.text_input(
        "🔑 OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Paste your OpenAI key. It stays in-session only.",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()

    # Data Source
    st.subheader("📂 Data Source")
    source_type = st.radio(
        "Choose source", ["Upload CSV/Excel", "Load Sample Data", "SQLite DB"]
    )

    df = None

    if source_type == "Upload CSV/Excel":
        uploaded = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"])
        if uploaded:
            df, msg = load_data(uploaded)
            if df is not None:
                st.success(msg)
            else:
                st.error(msg)

    elif source_type == "Load Sample Data":
        sample = st.selectbox(
            "Pick a sample dataset",
            ["Airline Operations", "Sales Data", "HR Analytics"],
        )
        sample_map = {
            "Airline Operations": "data/sample_airline.csv",
            "Sales Data": "data/sample_sales.csv",
            "HR Analytics": "data/sample_hr.csv",
        }
        sample_path = sample_map[sample]
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path)
            st.success(f"Loaded {sample} — {len(df):,} rows")
        else:
            st.warning("Sample file not found. Upload your own CSV.")

    elif source_type == "SQLite DB":
        db_file = st.file_uploader("Upload SQLite DB", type=["db", "sqlite", "sqlite3"])
        if db_file:
            table_name = st.text_input("Table name", "data")
            df, msg = load_sql_table(db_file, table_name)
            if df is not None:
                st.success(msg)
            else:
                st.error(msg)

    if df is not None and not df.equals(st.session_state.get("df", pd.DataFrame())):
        st.session_state.df = df
        st.session_state.query_engine = QueryEngine(df, api_key)
        st.session_state.chat_history = []
        st.rerun()

    if st.session_state.df is not None:
        st.divider()
        st.metric("Rows", f"{len(st.session_state.df):,}")
        st.metric("Columns", len(st.session_state.df.columns))

    st.divider()
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()


# ─── Main Area ───────────────────────────────────────────────────────────────
st.title("🤖 AI-Powered Data Insights Assistant")
st.caption(
    "Ask anything about your dataset in **plain English** — no SQL or code required."
)

if st.session_state.df is None:
    st.info(
        "👈  Upload a CSV / Excel file or load a sample dataset from the sidebar to begin."
    )
    st.stop()

df = st.session_state.df

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["💬 Chat", "📊 Explore", "🔍 Auto Insights", "📋 Data Preview"]
)

# ══════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Ask your data anything")

    # Quick suggestion chips
    suggestions = [
        "What are the top trends?",
        "Show me outliers and anomalies",
        "Which columns have highest correlation?",
        "Summarize key business metrics",
        "Give revenue optimisation recommendations",
    ]
    cols = st.columns(len(suggestions))
    clicked = None
    for i, s in enumerate(suggestions):
        if cols[i].button(s, key=f"chip_{i}", use_container_width=True):
            clicked = s

    st.divider()

    # Chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
                st.markdown(msg["content"])

    # Input
    user_input = st.chat_input("Ask me about your data…") or clicked

    if user_input:
        if not api_key:
            st.warning("Please enter your OpenAI API key in the sidebar.")
        else:
            # NLP preprocessing
            nlp_result = st.session_state.nlp.process(user_input)

            # Show in chat
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )

            with st.chat_message("user", avatar="👤"):
                st.markdown(user_input)

            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Analysing your query…"):
                    # Safety check for query engine
                    if st.session_state.query_engine is None:
                        st.error("Query engine not initialized. Please reload the dataset.")
                    else:
                        response = st.session_state.query_engine.answer(
                            user_input, nlp_result
                        )
                        st.markdown(response)

                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": response}
                        )
                    
                st.markdown(response)

            st.session_state.chat_history.append(
                {"role": "assistant", "content": response}
            )

# ══════════════════════════════════════════════════════════════════
# TAB 2 — EXPLORE
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Dataset Explorer")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    date_cols = df.select_dtypes(include="datetime").columns.tolist()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("Numeric", len(num_cols))
    col4.metric("Categorical", len(cat_cols))

    st.divider()

    viz_type = st.selectbox(
        "Choose visualisation",
        [
            "Distribution (Histogram)",
            "Correlation Heatmap",
            "Top Categories (Bar)",
            "Outliers (Box Plot)",
            "Time Series",
        ],
    )

    if viz_type == "Distribution (Histogram)" and num_cols:
        col = st.selectbox("Select numeric column", num_cols)
        fig = plot_distribution(df, col)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Correlation Heatmap" and len(num_cols) >= 2:
        fig = plot_correlation_heatmap(df)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Top Categories (Bar)" and cat_cols:
        col = st.selectbox("Select categorical column", cat_cols)
        top_n = st.slider("Top N", 5, 20, 10)
        fig = plot_top_categories(df, col, top_n)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Outliers (Box Plot)" and num_cols:
        col = st.selectbox("Select numeric column", num_cols, key="box_col")
        fig = plot_outliers_boxplot(df, col)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Time Series":
        if date_cols or any("date" in c.lower() or "time" in c.lower() for c in df.columns):
            possible = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
            date_col = st.selectbox("Date/Time column", possible or df.columns.tolist())
            val_col = st.selectbox("Value column", num_cols)
            fig = plot_time_series(df, date_col, val_col)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No date/time columns detected.")

    # Descriptive stats table
    st.subheader("📋 Descriptive Statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — AUTO INSIGHTS
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🔍 Auto-Generated Insights Report")
    st.caption(
        "GPT analyses your dataset and surfaces trends, anomalies, and business recommendations."
    )

    if not api_key:
        st.warning("Add your OpenAI API key in the sidebar to generate insights.")
    else:
        if st.button("⚡ Generate Full Insights Report", use_container_width=True):
            with st.spinner("Analysing dataset… this may take a few seconds."):
                generator = InsightGenerator(df, api_key)
                report = generator.generate()

            st.success("Report ready!")

            for section in report.get("sections", []):
                with st.expander(section["title"], expanded=True):
                    st.markdown(section["content"])

            # Data quality panel
            st.divider()
            st.subheader("🧹 Data Quality Report")
            num_df = df.select_dtypes(include="number")
            for col in num_df.columns:
                null_pct = df[col].isna().mean() * 100
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                n_outliers = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
                quality = max(0, 100 - int(null_pct) - min(n_outliers, 20))
                color = "#22c55e" if quality > 85 else "#f59e0b" if quality > 65 else "#ef4444"
                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
                      <span style="width:130px;font-size:12px;color:#aaa">{col[:18]}</span>
                      <div style="flex:1;height:8px;background:#1a1d2e;border-radius:4px;overflow:hidden">
                        <div style="width:{quality}%;height:100%;background:{color};border-radius:4px"></div>
                      </div>
                      <span style="width:40px;text-align:right;font-size:12px;color:{color}">{quality}%</span>
                      <span style="font-size:11px;color:#666">{n_outliers} outliers | {null_pct:.1f}% null</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ══════════════════════════════════════════════════════════════════
# TAB 4 — DATA PREVIEW
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📋 Raw Data Preview")
    rows = st.slider("Rows to show", 10, min(500, len(df)), 50)
    st.dataframe(df.head(rows), use_container_width=True)

    st.download_button(
        "⬇ Download as CSV",
        df.to_csv(index=False).encode("utf-8"),
        "data_export.csv",
        "text/csv",
    )
