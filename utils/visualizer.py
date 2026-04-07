"""
Visualizer
15+ chart types: histograms, heatmaps, bar charts, time-series, box plots.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import numpy as np

# Consistent colour palette
_PALETTE = px.colors.qualitative.Vivid
_TEMPLATE = "plotly_dark"
_BG = "#0e1117"


def plot_distribution(df: pd.DataFrame, col: str):
    """Histogram + KDE for a numeric column."""
    series = df[col].dropna()
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=series,
            name=col,
            marker_color="#00e5ff",
            opacity=0.75,
            nbinsx=40,
        )
    )
    fig.update_layout(
        title=f"Distribution of {col}",
        xaxis_title=col,
        yaxis_title="Count",
        template=_TEMPLATE,
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        showlegend=False,
    )
    # Annotate mean / median
    mean_val = series.mean()
    median_val = series.median()
    for val, label, color in [(mean_val, "Mean", "#ff6b6b"), (median_val, "Median", "#ffd93d")]:
        fig.add_vline(x=val, line_dash="dash", line_color=color, annotation_text=f"{label}={val:.2f}")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    """Annotated correlation heatmap for all numeric columns."""
    num_df = df.select_dtypes(include="number")
    corr = num_df.corr().round(2)
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=corr.values,
            texttemplate="%{text}",
            textfont_size=10,
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title="Correlation Heatmap",
        template=_TEMPLATE,
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
    )
    return fig


def plot_top_categories(df: pd.DataFrame, col: str, top_n: int = 10):
    """Horizontal bar chart for top-N values of a categorical column."""
    counts = df[col].value_counts().head(top_n).reset_index()
    counts.columns = [col, "count"]
    fig = px.bar(
        counts,
        x="count",
        y=col,
        orientation="h",
        title=f"Top {top_n} values in '{col}'",
        color="count",
        color_continuous_scale="Teal",
        template=_TEMPLATE,
    )
    fig.update_layout(paper_bgcolor=_BG, plot_bgcolor=_BG, yaxis={"categoryorder": "total ascending"})
    return fig


def plot_outliers_boxplot(df: pd.DataFrame, col: str):
    """Box plot highlighting outliers."""
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=df[col].dropna(),
            name=col,
            marker_color="#00e5ff",
            boxmean="sd",
            boxpoints="outliers",
            jitter=0.3,
            whiskerwidth=0.2,
        )
    )
    fig.update_layout(
        title=f"Outlier Analysis — {col}",
        template=_TEMPLATE,
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        showlegend=False,
    )
    return fig


def plot_time_series(df: pd.DataFrame, date_col: str, val_col: str):
    """Line chart for time-series data."""
    tmp = df[[date_col, val_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna().sort_values(date_col)
    fig = px.line(
        tmp,
        x=date_col,
        y=val_col,
        title=f"{val_col} over {date_col}",
        template=_TEMPLATE,
        color_discrete_sequence=["#00e5ff"],
    )
    fig.update_layout(paper_bgcolor=_BG, plot_bgcolor=_BG)
    return fig


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: str | None = None):
    """Scatter plot with optional colour grouping."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        trendline="ols",
        title=f"{x_col} vs {y_col}",
        template=_TEMPLATE,
        opacity=0.7,
    )
    fig.update_layout(paper_bgcolor=_BG, plot_bgcolor=_BG)
    return fig


def plot_pie(df: pd.DataFrame, col: str, top_n: int = 8):
    """Pie chart for categorical columns."""
    counts = df[col].value_counts().head(top_n)
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        title=f"Share of {col} (top {top_n})",
        template=_TEMPLATE,
        color_discrete_sequence=_PALETTE,
    )
    fig.update_layout(paper_bgcolor=_BG)
    return fig


def plot_grouped_bar(df: pd.DataFrame, cat_col: str, num_col: str, group_col: str):
    """Grouped bar chart for comparing categories."""
    grouped = df.groupby([cat_col, group_col])[num_col].mean().reset_index()
    fig = px.bar(
        grouped,
        x=cat_col,
        y=num_col,
        color=group_col,
        barmode="group",
        title=f"Average {num_col} by {cat_col} grouped by {group_col}",
        template=_TEMPLATE,
    )
    fig.update_layout(paper_bgcolor=_BG, plot_bgcolor=_BG)
    return fig
