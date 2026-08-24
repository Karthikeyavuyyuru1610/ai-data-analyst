"""LangChain + Google Gemini integration with dedicated Context Builder and System Prompts."""
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from core.ingestion import DatasetProfile, ColumnProfile

load_dotenv()

# ─── System Prompts ─────────────────────────────────────────────────────────

DATA_ANALYSIS_SYSTEM_PROMPT = """You are a structured-data analysis assistant.

You answer questions about CSV and Excel datasets provided by the application.

The application provides authoritative dataset metadata including:
- file names
- table names
- row counts
- column names
- physical data types
- semantic column types
- statistics
- sample values

Treat the provided schema and dataset metadata as the source of truth.

Never invent a column.
Never invent a table.
Never invent a numerical value.
Never assume a column exists unless it is present in the provided dataset context.

Respect the analysis scope selected by the user.
If one dataset is selected, analyze only that dataset.
If multiple datasets are selected, analyze only those datasets.
If combined analysis is requested, only combine datasets that are compatible.

When a question requires calculation, aggregation, filtering, comparison, ranking, grouping, or trend analysis, rely on the analytical database rather than estimating values.

Do not perform arithmetic mentally when the database can calculate it.

If the requested information does not exist, clearly explain what is missing.

If the question is ambiguous, ask a concise clarification question.

Do not hallucinate.

Base every numerical claim on actual analytical results.

Return concise, factual, easy-to-understand explanations.

When appropriate, mention which columns or datasets were used."""


SQL_GENERATION_SYSTEM_PROMPT = """You are an expert SQL analyst. Convert the natural language question into a single DuckDB SQL query.

RULES:
1. Return ONLY the DuckDB SQL query inside a markdown ```sql ... ``` block or as plain text. No commentary.
2. Use ONLY the tables and column names provided in the dataset context.
3. Use DuckDB SQL dialect.
4. For "top N", use ORDER BY ... DESC LIMIT N.
5. For aggregations (SUM, AVG, COUNT), always assign clear aliases (e.g. AS total_revenue, AS avg_price).
6. If combined scope is requested across compatible files, query the combined view or use UNION ALL as specified.
7. Only generate SELECT or WITH statements."""


EXPLANATION_SYSTEM_PROMPT = """You are a data analyst explaining analytical query results to a business user.

RULES:
1. State the key finding first.
2. Reference exact numerical values from the provided query results.
3. Do NOT invent, extrapolate, or estimate any numbers. All figures must originate from the result table.
4. Be concise and direct. Use bullet points or short paragraphs.
5. Format numbers cleanly (e.g. $12.4M, 25,420, 15.2%)."""


SUMMARY_SYSTEM_PROMPT = """You are a senior data analyst summarizing tabular datasets.

RULES:
1. Base your summary STRICTLY on the provided pre-computed statistics and dataset metadata.
2. Never invent numbers or metrics.
3. Format your response cleanly into three distinct markdown sections:

### Overview
Brief summary of the dataset scope, record count, and time period covered (if date columns exist).

### Key Metrics
List the actual computed totals, averages, or record counts (e.g. Total Revenue: $X, Average Price: $Y).

### Key Observations
List 3-5 notable insights, top performers, or missing data observations derived from the metrics.
"""


COMBINED_SUMMARY_SYSTEM_PROMPT = """You are a senior data analyst providing a combined summary across multiple datasets.

RULES:
1. Base your summary STRICTLY on the provided combined pre-computed statistics.
2. Never invent numbers.
3. Format your response cleanly into these markdown sections:

### Overall Picture
Combined record count and scope summary.

### Key Combined Metrics
Overall totals and averages across all datasets.

### File-to-File Comparison
Compare performance, metrics, or record counts across individual files.

### Key Observations
Notable trends, changes between datasets, or top performers.
"""


def _get_llm() -> ChatGoogleGenerativeAI:
    """Initialize the Gemini LLM via LangChain."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is not set. Please add your API key to the .env file."
        )

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0,  # Deterministic execution
    )


# ─── Data Context Builder ───────────────────────────────────────────────────

def build_analysis_context(
    question: str,
    selected_profiles: list[DatasetProfile],
    analysis_scope: str,
    all_available_tables: list[str] | None = None,
) -> str:
    """
    Build structured dataset context to send to Gemini.
    Prompt Section 8 requirement.
    """
    context_lines = [
        f"ANALYSIS SCOPE:\n{analysis_scope}\n",
        "DATASETS IN SCOPE:",
    ]

    for profile in selected_profiles:
        context_lines.append(
            f"- Table: {profile.table_name} (Source File: {profile.file_name}, "
            f"Rows: {profile.row_count:,}, Columns: {profile.column_count})"
        )

    context_lines.append("\nCOLUMN PROFILES & METADATA:")

    for profile in selected_profiles:
        context_lines.append(f"\n--- Table: {profile.table_name} ({profile.file_name}) ---")
        for col in profile.columns:
            meta_str = f"  - {col.name} ({col.physical_dtype}) | Semantic: {col.semantic_type} | Nulls: {col.null_count} ({col.null_pct}%)"

            if col.semantic_type in ("date", "datetime") and (col.min_date or col.max_date):
                meta_str += f" | Date Range: {col.min_date} to {col.max_date}"
            elif col.min_val is not None or col.max_val is not None:
                meta_str += f" | Min: {col.min_val}, Max: {col.max_val}"
                if col.mean_val is not None:
                    meta_str += f", Mean: {col.mean_val}"

            if col.sample_values:
                samples_str = ", ".join(str(s) for s in col.sample_values[:4])
                meta_str += f" | Examples: [{samples_str}]"

            context_lines.append(meta_str)

    if all_available_tables:
        context_lines.append(f"\nALL AVAILABLE TABLES IN DATABASE:\n{', '.join(all_available_tables)}")

    context_lines.append(f"\nUSER QUESTION:\n{question}")

    return "\n".join(context_lines)


# ─── LLM Operations ────────────────────────────────────────────────────────

def generate_sql(
    question: str,
    selected_profiles: list[DatasetProfile],
    analysis_scope: str,
    all_available_tables: list[str] | None = None,
) -> tuple[str, str]:
    """
    Generate DuckDB SQL query using LangChain + Gemini.
    Returns (sql, error_message).
    """
    try:
        llm = _get_llm()
        context = build_analysis_context(question, selected_profiles, analysis_scope, all_available_tables)

        user_content = (
            f"{context}\n\n"
            f"Generate a single DuckDB SQL query to compute the answer for the user question above."
        )

        messages = [
            SystemMessage(content=DATA_ANALYSIS_SYSTEM_PROMPT + "\n\n" + SQL_GENERATION_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        response = llm.invoke(messages)
        sql = response.content.strip()
        return sql, ""

    except ValueError as e:
        return "", str(e)
    except Exception as e:
        return "", f"Gemini API error: {str(e)[:300]}"


def retry_sql_generation(
    question: str,
    selected_profiles: list[DatasetProfile],
    analysis_scope: str,
    failed_sql: str,
    error_message: str,
    all_available_tables: list[str] | None = None,
) -> tuple[str, str]:
    """
    Retry SQL generation with error context feedback.
    Returns (sql, error_message).
    """
    try:
        llm = _get_llm()
        context = build_analysis_context(question, selected_profiles, analysis_scope, all_available_tables)

        user_content = (
            f"{context}\n\n"
            f"PREVIOUS GENERATED SQL THAT FAILED:\n{failed_sql}\n\n"
            f"EXECUTION ERROR:\n{error_message}\n\n"
            f"Please fix the query. Return ONLY the corrected DuckDB SQL query."
        )

        messages = [
            SystemMessage(content=DATA_ANALYSIS_SYSTEM_PROMPT + "\n\n" + SQL_GENERATION_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        response = llm.invoke(messages)
        sql = response.content.strip()
        return sql, ""

    except Exception as e:
        return "", f"Gemini API error during retry: {str(e)[:300]}"


def explain_result(
    question: str,
    sql: str,
    result_summary: str,
    selected_profiles: list[DatasetProfile],
    analysis_scope: str,
) -> tuple[str, str]:
    """
    Generate a natural language explanation of the computed DuckDB result.
    Returns (explanation, error_message).
    """
    try:
        llm = _get_llm()
        context = build_analysis_context(question, selected_profiles, analysis_scope)

        user_content = (
            f"DATASET CONTEXT:\n{context}\n\n"
            f"EXECUTED SQL:\n{sql}\n\n"
            f"ACTUAL COMPUTED RESULTS FROM DATABASE:\n{result_summary}\n\n"
            f"Explain the computed result clearly to answer the user question."
        )

        messages = [
            SystemMessage(content=DATA_ANALYSIS_SYSTEM_PROMPT + "\n\n" + EXPLANATION_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        response = llm.invoke(messages)
        return response.content.strip(), ""

    except Exception as e:
        return "", f"Could not generate explanation: {str(e)[:300]}"


def generate_summary(
    selected_profiles: list[DatasetProfile],
    statistics_str: str,
    is_combined: bool = False,
) -> tuple[str, str]:
    """
    Generate a structured dataset summary (individual or combined).
    Returns (summary_markdown, error_message).
    """
    try:
        llm = _get_llm()
        scope_str = "Combined Analysis (Multiple Files)" if is_combined else f"Single File Analysis ({selected_profiles[0].file_name})"

        context_lines = ["DATASET METADATA:"]
        for p in selected_profiles:
            context_lines.append(f"- File: {p.file_name}, Table: {p.table_name}, Rows: {p.row_count:,}, Cols: {p.column_count}")

        context_str = "\n".join(context_lines)

        user_content = (
            f"SCOPE: {scope_str}\n\n"
            f"{context_str}\n\n"
            f"COMPUTED DATABASE STATISTICS:\n{statistics_str}\n\n"
            f"Generate a structured summary following the specified sections."
        )

        sys_prompt = COMBINED_SUMMARY_SYSTEM_PROMPT if is_combined else SUMMARY_SYSTEM_PROMPT

        messages = [
            SystemMessage(content=DATA_ANALYSIS_SYSTEM_PROMPT + "\n\n" + sys_prompt),
            HumanMessage(content=user_content),
        ]

        response = llm.invoke(messages)
        return response.content.strip(), ""

    except Exception as e:
        return "", f"Could not generate summary: {str(e)[:300]}"
