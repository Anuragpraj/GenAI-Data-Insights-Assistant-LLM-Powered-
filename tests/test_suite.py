"""
Unit tests for AI Data Insights Assistant
Run: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import pytest

from utils.nlp_pipeline import NLPPipeline
from utils.data_loader import _clean_dataframe


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "route":          ["DEL-BOM", "BOM-BLR", "DEL-DXB", "DEL-BOM", "BOM-LHR"],
        "ticket_price":   [4500, 3200, 18500, 4700, 52000],
        "occupancy_rate": [0.90, 0.80, 0.90, 0.95, 0.80],
        "revenue":        [729000, 384000, 4995000, 803700, 11648000],
        "season":         ["Winter", "Winter", "Winter", "Winter", "Winter"],
    })

@pytest.fixture
def nlp():
    return NLPPipeline()


# ─── NLP Pipeline Tests ──────────────────────────────────────────────────────

class TestNLPPipeline:

    def test_process_returns_expected_keys(self, nlp):
        result = nlp.process("What are the top 5 routes by revenue?")
        for key in ("original", "clean", "tokens", "intent", "entities", "keywords", "numbers", "confidence"):
            assert key in result, f"Missing key: {key}"

    def test_intent_top_n(self, nlp):
        result = nlp.process("Show me top 5 routes by revenue")
        assert result["intent"] == "top_n"

    def test_intent_outlier(self, nlp):
        result = nlp.process("Show outliers and anomalies in the data")
        assert result["intent"] == "outlier"

    def test_intent_correlation(self, nlp):
        result = nlp.process("Which columns are correlated?")
        assert result["intent"] == "correlation"

    def test_intent_summary(self, nlp):
        result = nlp.process("Give me a summary of this dataset")
        assert result["intent"] == "summary"

    def test_intent_trend(self, nlp):
        result = nlp.process("How does revenue change over time?")
        assert result["intent"] == "trend"

    def test_number_extraction(self, nlp):
        result = nlp.process("Show top 10 items")
        assert 10.0 in result["numbers"]

    def test_confidence_range(self, nlp):
        result = nlp.process("What is the average price?")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_clean_lowercases(self, nlp):
        result = nlp.process("SHOW ME OUTLIERS")
        assert result["clean"] == result["clean"].lower()

    def test_tokens_list(self, nlp):
        result = nlp.process("show distribution")
        assert isinstance(result["tokens"], list)
        assert len(result["tokens"]) > 0


# ─── Data Loader Tests ───────────────────────────────────────────────────────

class TestDataLoader:

    def test_clean_strips_column_names(self):
        df = pd.DataFrame({" price ": [1, 2], "  name  ": ["a", "b"]})
        cleaned = _clean_dataframe(df)
        assert "price" in cleaned.columns
        assert "name" in cleaned.columns

    def test_clean_drops_all_null_rows(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, None, 6]})
        # Manually create a fully null row
        df.loc[3] = [None, None]
        cleaned = _clean_dataframe(df)
        assert len(cleaned) == 3  # row 3 dropped

    def test_clean_strips_string_whitespace(self):
        df = pd.DataFrame({"city": ["  Delhi  ", " Mumbai "]})
        cleaned = _clean_dataframe(df)
        assert cleaned["city"].tolist() == ["Delhi", "Mumbai"]

    def test_clean_parses_date_columns(self):
        df = pd.DataFrame({"date": ["2024-01-01", "2024-02-15"], "value": [1, 2]})
        cleaned = _clean_dataframe(df)
        assert str(cleaned["date"].dtype).startswith("datetime")

    def test_clean_handles_numeric_columns(self, sample_df):
        cleaned = _clean_dataframe(sample_df)
        assert cleaned["ticket_price"].dtype in [np.float64, np.int64]


# ─── Insight Generator (stat functions) ──────────────────────────────────────

class TestInsightGeneratorStats:

    def test_outlier_detection(self, sample_df):
        """Revenue has a clear high outlier (BOM-LHR = 11.6M)."""
        from utils.insight_generator import InsightGenerator
        gen = InsightGenerator(sample_df)
        result = gen._outliers()
        assert "revenue" in result.lower()
        assert "outlier" in result.lower()

    def test_correlation_output(self, sample_df):
        from utils.insight_generator import InsightGenerator
        gen = InsightGenerator(sample_df)
        result = gen._correlations()
        assert "correlation" in result.lower() or "r =" in result.lower()

    def test_overview_output(self, sample_df):
        from utils.insight_generator import InsightGenerator
        gen = InsightGenerator(sample_df)
        result = gen._overview()
        assert "5" in result  # 5 rows
        assert "columns" in result.lower()


# ─── Query Engine (fallback) ─────────────────────────────────────────────────

class TestQueryEngineFallback:

    def test_summary_answer(self, sample_df):
        from utils.query_engine import QueryEngine
        from utils.nlp_pipeline import NLPPipeline
        engine = QueryEngine(sample_df, api_key=None)
        nlp = NLPPipeline()
        nlp_result = nlp.process("Give me a summary")
        response = engine.answer("Give me a summary", nlp_result)
        assert len(response) > 50
        assert "rows" in response.lower() or "columns" in response.lower()

    def test_outlier_answer(self, sample_df):
        from utils.query_engine import QueryEngine
        from utils.nlp_pipeline import NLPPipeline
        engine = QueryEngine(sample_df, api_key=None)
        nlp = NLPPipeline()
        nlp_result = nlp.process("Show me outliers")
        response = engine.answer("Show me outliers", nlp_result)
        assert "outlier" in response.lower()

    def test_top_n_answer(self, sample_df):
        from utils.query_engine import QueryEngine
        from utils.nlp_pipeline import NLPPipeline
        engine = QueryEngine(sample_df, api_key=None)
        nlp = NLPPipeline()
        nlp_result = nlp.process("Top 3 by revenue")
        response = engine.answer("Top 3 by revenue", nlp_result)
        assert len(response) > 10
