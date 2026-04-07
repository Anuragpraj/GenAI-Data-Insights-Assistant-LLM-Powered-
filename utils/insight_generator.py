"""
Insight Generator
Auto-generates data insight reports: trends, distributions, outliers.
Cuts manual analysis time by ~50% (as per project spec).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import scipy.stats as stats

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class InsightGenerator:
    """
    Generates a structured insight report using:
    1. Statistical analysis (always runs locally)
    2. GPT narrative (if API key available)
    """

    def __init__(self, df: pd.DataFrame, api_key: str | None = None):
        self.df = df
        self.api_key = api_key
        self._client = None
        if OPENAI_AVAILABLE and api_key:
            self._client = OpenAI(api_key=api_key)

    # ── Public ───────────────────────────────────────────────────────────────

    def generate(self) -> Dict[str, Any]:
        """Returns a dict: { sections: [ {title, content}, ... ] }"""
        sections = []

        sections.append({"title": "📊 Dataset Overview", "content": self._overview()})
        sections.append({"title": "📈 Trends & Patterns", "content": self._trends()})
        sections.append({"title": "🔴 Outliers & Anomalies", "content": self._outliers()})
        sections.append({"title": "🔗 Correlations", "content": self._correlations()})
        sections.append({"title": "📉 Distribution Analysis", "content": self._distributions()})

        if self._client:
            try:
                sections.append({
                    "title": "🤖 GPT Business Insights & Recommendations",
                    "content": self._gpt_insights(),
                })
            except Exception as e:
                sections.append({
                    "title": "🤖 GPT Insights",
                    "content": f"⚠️ Could not generate GPT insights: {e}",
                })
        else:
            sections.append({
                "title": "💡 Rule-Based Recommendations",
                "content": self._rule_based_recommendations(),
            })

        return {"sections": sections}

    # ── Sections ─────────────────────────────────────────────────────────────

    def _overview(self) -> str:
        df = self.df
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        null_pct = df.isna().mean().mean() * 100
        dup_rows = df.duplicated().sum()

        return (
            f"- **{len(df):,} rows** and **{len(df.columns)} columns** loaded successfully.\n"
            f"- **{len(num_cols)} numeric** columns: `{'`, `'.join(num_cols) or 'none'}`\n"
            f"- **{len(cat_cols)} categorical** columns: `{'`, `'.join(cat_cols) or 'none'}`\n"
            f"- **Missing data**: {null_pct:.1f}% of all cells are null.\n"
            f"- **Duplicate rows**: {dup_rows:,} detected.\n"
        )

    def _trends(self) -> str:
        df = self.df
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return "No numeric columns available for trend analysis."

        lines = []
        for col in num_cols[:5]:
            series = df[col].dropna()
            if len(series) < 4:
                continue
            # Linear trend via scipy
            x = np.arange(len(series))
            slope, intercept, r, p, se = stats.linregress(x, series)
            direction = "📈 upward" if slope > 0 else "📉 downward"
            significance = "significant" if p < 0.05 else "not statistically significant"
            lines.append(
                f"- `{col}`: **{direction} trend** (slope={slope:.4f}, r²={r**2:.3f}, p={p:.4f}) — {significance}."
            )

        return "\n".join(lines) if lines else "Insufficient data for trend analysis."

    def _outliers(self) -> str:
        df = self.df
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return "No numeric columns for outlier detection."

        lines = ["Using **IQR method** (1.5× fence):\n"]
        total_outliers = 0
        for col in num_cols[:6]:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (df[col] < lower) | (df[col] > upper)
            n = mask.sum()
            total_outliers += n
            pct = n / len(df) * 100
            flag = "⚠️" if pct > 5 else "✅"
            lines.append(
                f"- {flag} `{col}`: **{n} outliers** ({pct:.1f}%) — "
                f"valid range [{lower:.2f}, {upper:.2f}]"
            )

        lines.append(f"\n**Total outlier cells detected: {total_outliers:,}**")
        return "\n".join(lines)

    def _correlations(self) -> str:
        df = self.df
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) < 2:
            return "Need at least 2 numeric columns for correlation analysis."

        corr = df[num_cols].corr()
        pairs = []
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                r = corr.iloc[i, j]
                if not np.isnan(r):
                    pairs.append((num_cols[i], num_cols[j], r))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        lines = []
        for a, b, r in pairs[:8]:
            if abs(r) > 0.7:
                level = "🔴 **Strong**"
            elif abs(r) > 0.4:
                level = "🟡 **Moderate**"
            else:
                level = "⚪ **Weak**"
            direction = "positive ↗" if r > 0 else "negative ↘"
            lines.append(f"- {level} {direction} correlation between `{a}` and `{b}`: r = **{r:.3f}**")

        return "\n".join(lines) if lines else "No significant correlations found."

    def _distributions(self) -> str:
        df = self.df
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return "No numeric columns."

        lines = []
        for col in num_cols[:6]:
            series = df[col].dropna()
            skew = series.skew()
            kurt = series.kurtosis()
            _, p_normal = stats.normaltest(series) if len(series) >= 8 else (None, 1.0)
            normality = "✅ approx. normal" if p_normal > 0.05 else "❌ non-normal"

            if skew > 1:
                skew_desc = "heavily right-skewed (long tail on right)"
            elif skew > 0.5:
                skew_desc = "mildly right-skewed"
            elif skew < -1:
                skew_desc = "heavily left-skewed (long tail on left)"
            elif skew < -0.5:
                skew_desc = "mildly left-skewed"
            else:
                skew_desc = "approximately symmetric"

            lines.append(
                f"- `{col}`: {skew_desc} (skew={skew:.2f}, kurtosis={kurt:.2f}) — {normality}"
            )

        return "\n".join(lines)

    def _gpt_insights(self) -> str:
        summary = self._build_gpt_summary()
        response = self._client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior data analyst. Given dataset statistics, generate "
                        "exactly 6 actionable business insights in bullet-point Markdown. "
                        "Be specific with numbers. Bold key findings. "
                        "End with 3 concrete recommendations. Max 400 words."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Dataset statistics:\n\n{summary}\n\nGenerate insights and recommendations.",
                },
            ],
            max_tokens=700,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()

    def _rule_based_recommendations(self) -> str:
        df = self.df
        num_cols = df.select_dtypes(include="number").columns.tolist()
        recs = []

        null_pct = df.isna().mean().mean() * 100
        if null_pct > 5:
            recs.append(f"⚠️ **Data Quality**: {null_pct:.1f}% missing values — consider imputation or removal.")

        if df.duplicated().sum() > 0:
            recs.append(f"🔁 **Duplicates**: {df.duplicated().sum()} duplicate rows found — deduplicate before modelling.")

        for col in num_cols[:3]:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            n_out = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
            if n_out / len(df) > 0.05:
                recs.append(f"📌 **`{col}`**: {n_out} outliers (>{5}%) — investigate or cap/floor values.")

        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    if abs(corr.iloc[i, j]) > 0.85:
                        recs.append(
                            f"🔗 **High collinearity**: `{num_cols[i]}` & `{num_cols[j]}` (r={corr.iloc[i,j]:.2f}) — "
                            "consider dropping one for regression models."
                        )

        if not recs:
            recs.append("✅ Dataset looks clean. Add your OpenAI API key for GPT-powered insights.")

        return "\n".join(recs)

    # ── Helper ───────────────────────────────────────────────────────────────

    def _build_gpt_summary(self) -> str:
        df = self.df
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        parts = [
            f"Rows: {len(df)}, Columns: {len(df.columns)}",
            f"Numeric: {', '.join(num_cols)}",
            f"Categorical: {', '.join(cat_cols)}",
        ]
        for col in num_cols[:6]:
            d = df[col].describe()
            parts.append(
                f"{col}: mean={d['mean']:.2f}, std={d['std']:.2f}, min={d['min']:.2f}, max={d['max']:.2f}"
            )
        for col in cat_cols[:3]:
            top = df[col].value_counts().head(3).to_dict()
            parts.append(f"{col}: {top}")
        return "\n".join(parts)
