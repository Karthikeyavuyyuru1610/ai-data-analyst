"""Plotly visualization — automatic chart selection based on result shape."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def should_visualize(df: pd.DataFrame) -> bool:
    """Determine if a result set benefits from a chart."""
    if df is None or df.empty:
        return False
    # Single scalar value — show as KPI, not a chart
    if len(df) == 1 and len(df.columns) == 1:
        return False
    # Need at least 2 rows for a meaningful chart
    if len(df) < 2:
        return False
    # Need at least one numeric column
    if df.select_dtypes(include=["number"]).columns.tolist() == []:
        return False
    return True


def detect_chart_type(df: pd.DataFrame) -> str:
    """
    Determine the best chart type based on column types.
    Returns: 'bar', 'line', 'scatter', 'kpi', or 'none'.
    """
    if df is None or df.empty:
        return "none"

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    non_num_cols = [c for c in df.columns if c not in num_cols]

    # Single value → KPI
    if len(df) == 1 and len(num_cols) >= 1:
        return "kpi"

    # Check if any non-numeric column looks like a date/time
    has_time = False
    time_col = None
    for col in non_num_cols:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > len(df) * 0.5:
                has_time = True
                time_col = col
                break
        except Exception:
            pass

    # Also check numeric columns that might be years/months
    if not has_time:
        for col in non_num_cols:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ["month", "year", "date", "week", "quarter", "period"]):
                has_time = True
                time_col = col
                break

    # Time + numeric → line chart
    if has_time and len(num_cols) >= 1:
        return "line"

    # Category + numeric → bar chart
    if len(non_num_cols) >= 1 and len(num_cols) >= 1:
        return "bar"

    # Two numeric columns → scatter
    if len(num_cols) >= 2 and len(non_num_cols) == 0:
        return "scatter"

    return "bar"  # Default fallback


def create_chart(df: pd.DataFrame, question: str = "") -> go.Figure | None:
    """
    Create an appropriate Plotly chart for the result set.
    Returns a Figure or None if visualization isn't appropriate.
    """
    if not should_visualize(df):
        return None

    chart_type = detect_chart_type(df)

    if chart_type == "none" or chart_type == "kpi":
        return None

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    non_num_cols = [c for c in df.columns if c not in num_cols]

    # Clean styling
    layout_defaults = dict(
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=13, color="#334155"),
        title=dict(font=dict(size=16, color="#1e293b")),
        margin=dict(l=60, r=30, t=50, b=50),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    color_sequence = ["#0d9488", "#6366f1", "#f59e0b", "#ef4444", "#8b5cf6",
                      "#06b6d4", "#10b981", "#f97316", "#ec4899", "#84cc16"]

    try:
        if chart_type == "line":
            x_col = non_num_cols[0] if non_num_cols else df.columns[0]
            y_col = num_cols[0]

            fig = px.line(
                df, x=x_col, y=y_col,
                title=_make_title(question, y_col, "over", x_col),
                color_discrete_sequence=color_sequence,
                markers=True,
            )
            fig.update_traces(line=dict(width=2.5))

        elif chart_type == "bar":
            x_col = non_num_cols[0] if non_num_cols else df.columns[0]
            y_col = num_cols[0]

            # Horizontal bar for many categories
            if len(df) > 8:
                fig = px.bar(
                    df, x=y_col, y=x_col,
                    orientation="h",
                    title=_make_title(question, y_col, "by", x_col),
                    color_discrete_sequence=color_sequence,
                )
            else:
                fig = px.bar(
                    df, x=x_col, y=y_col,
                    title=_make_title(question, y_col, "by", x_col),
                    color_discrete_sequence=color_sequence,
                )
            fig.update_traces(marker_line_width=0)

        elif chart_type == "scatter":
            x_col = num_cols[0]
            y_col = num_cols[1]
            fig = px.scatter(
                df, x=x_col, y=y_col,
                title=_make_title(question, y_col, "vs", x_col),
                color_discrete_sequence=color_sequence,
            )
            fig.update_traces(marker=dict(size=8, opacity=0.7))

        else:
            return None

        fig.update_layout(**layout_defaults)
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#f1f5f9")

        return fig

    except Exception:
        return None


def _make_title(question: str, *parts: str) -> str:
    """Generate a clean chart title."""
    if question:
        # Use first 60 chars of the question as title
        title = question[:60]
        if len(question) > 60:
            title += "..."
        return title
    return " ".join(p.replace("_", " ").title() for p in parts if p)
