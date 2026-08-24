"""DataLens — AI-Powered Professional Data Q&A Workspace."""
import streamlit as st
import pandas as pd
from typing import Any, Optional

from core.ingestion import (
    validate_file,
    load_file,
    profile_dataset,
    check_schema_compatibility,
    sanitize_table_name,
    DatasetProfile,
)
from core.database import Database
from core.sql_safety import validate_sql, clean_sql_from_llm
from core.llm import (
    generate_sql,
    retry_sql_generation,
    explain_result,
    generate_summary,
    build_analysis_context,
)

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DataLens — Data Analysis Workspace",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Professional UI Design System (CSS) ─────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    *, html, body, [class*="st-"] {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    .stApp {
        background-color: #ffffff;
        color: #0f172a;
    }

    /* Hide Streamlit Chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 1.75rem;
        padding-bottom: 3rem;
        max-width: 60rem;
    }

    /* ── Header ── */
    .app-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 1.25rem;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 1.75rem;
    }

    .app-brand {
        display: flex;
        align-items: center;
        gap: 0.625rem;
    }

    .app-logo {
        width: 32px;
        height: 32px;
        background: #0f766e;
        color: #ffffff;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.1rem;
    }

    .app-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.2;
    }

    .app-subtitle {
        font-size: 0.8125rem;
        color: #64748b;
        font-weight: 400;
    }

    .session-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        padding: 0.25rem 0.625rem;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 100px;
        font-size: 0.75rem;
        color: #475569;
        font-weight: 500;
    }

    .session-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #10b981;
    }

    /* ── Section Labels ── */
    .section-header {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        margin-bottom: 0.75rem;
        margin-top: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .section-header::after {
        content: "";
        flex: 1;
        height: 1px;
        background: #f1f5f9;
    }

    /* ── Dataset Card ── */
    .dataset-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.875rem 1.125rem;
        margin-bottom: 0.625rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }

    .dataset-info {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .dataset-icon {
        width: 36px;
        height: 36px;
        background: #f0fdfa;
        border: 1px solid #ccfbf1;
        color: #0f766e;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        font-size: 0.8125rem;
    }

    .dataset-name {
        font-weight: 600;
        font-size: 0.875rem;
        color: #0f172a;
    }

    .dataset-meta {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 0.125rem;
    }

    /* ── Schema Table Styling ── */
    .schema-container {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.75rem;
        margin-bottom: 1rem;
    }

    .semantic-badge {
        display: inline-block;
        padding: 0.125rem 0.5rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    .sem-currency { background: #dcfce7; color: #166534; }
    .sem-measure { background: #e0f2fe; color: #075985; }
    .sem-quantity { background: #fef3c7; color: #92400e; }
    .sem-percentage { background: #fae8ff; color: #86198f; }
    .sem-date { background: #ccfbf1; color: #115e59; }
    .sem-categorical { background: #f1f5f9; color: #334155; }
    .sem-identifier { background: #f3f4f6; color: #4b5563; }
    .sem-boolean { background: #fee2e2; color: #991b1b; }
    .sem-text { background: #f8fafc; color: #64748b; }

    /* ── Result Card ── */
    .answer-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-left: 4px solid #0d9488;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    .answer-title {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #0f766e;
        margin-bottom: 0.5rem;
    }

    .answer-body {
        font-size: 0.9375rem;
        color: #1e293b;
        line-height: 1.65;
    }

    /* ── Supporting Metrics Grid ── */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.875rem;
        margin-bottom: 1.25rem;
    }

    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 1rem 1.125rem;
    }

    .metric-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.02em;
    }

    .metric-lbl {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 0.25rem;
    }

    /* ── Error Card ── */
    .error-box {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        color: #991b1b;
        font-size: 0.875rem;
        line-height: 1.6;
        margin: 1rem 0;
    }

    .error-box strong {
        color: #7f1d1d;
    }

    /* ── Code / SQL Block ── */
    .sql-code {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8125rem;
        background: #0f172a;
        color: #f8fafc;
        padding: 0.875rem 1rem;
        border-radius: 6px;
        overflow-x: auto;
    }

    /* ── Scope Badge ── */
    .scope-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        padding: 0.375rem 0.75rem;
        background: #f0fdfa;
        border: 1px solid #ccfbf1;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #0f766e;
    }

    /* ── Button styling overrides ── */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.8125rem;
        transition: all 0.15s ease;
    }

    [data-testid="stFileUploader"] {
        border: 2px dashed #cbd5e1;
        border-radius: 8px;
        padding: 0.75rem;
        background: #f8fafc;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #0d9488;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State ────────────────────────────────────────────────────────────

def init_session():
    """Initialize state store."""
    if "db" not in st.session_state:
        st.session_state.db = Database()
    if "dataset_profiles" not in st.session_state:
        st.session_state.dataset_profiles = {}  # table_name -> DatasetProfile
    if "dataframes" not in st.session_state:
        st.session_state.dataframes = {}  # table_name -> DataFrame
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "uploaded_files_set" not in st.session_state:
        st.session_state.uploaded_files_set = set()
    if "schema_view_toggles" not in st.session_state:
        st.session_state.schema_view_toggles = {}


init_session()
db: Database = st.session_state.db


# ─── Helper Functions ────────────────────────────────────────────────────────

def compute_statistics_for_summary(selected_profiles: list[DatasetProfile]) -> str:
    """Compute detailed aggregate statistics using DuckDB for dataset summarization."""
    stats_lines = []

    for profile in selected_profiles:
        tn = profile.table_name
        stats_lines.append(f"--- Table: {tn} (File: {profile.file_name}, Rows: {profile.row_count:,}) ---")

        # Numeric / Measures stats
        for col in profile.columns:
            if col.semantic_type in ("measure", "currency", "quantity", "percentage") or col.min_val is not None:
                col_name = col.name
                try:
                    res, err = db.execute_query(
                        f'SELECT COUNT("{col_name}") as cnt, '
                        f'ROUND(AVG("{col_name}"), 2) as avg_val, '
                        f'ROUND(SUM("{col_name}"), 2) as sum_val, '
                        f'ROUND(MIN("{col_name}"), 2) as min_val, '
                        f'ROUND(MAX("{col_name}"), 2) as max_val '
                        f'FROM "{tn}" WHERE "{col_name}" IS NOT NULL'
                    )
                    if res is not None and not res.empty:
                        r = res.iloc[0]
                        stats_lines.append(
                            f"Column '{col_name}' ({col.semantic_type}): "
                            f"Sum={r['sum_val']:,}, Avg={r['avg_val']:,}, Min={r['min_val']:,}, Max={r['max_val']:,}"
                        )
                except Exception:
                    pass

        # Categorical top counts
        for col in profile.columns:
            if col.semantic_type == "categorical":
                col_name = col.name
                try:
                    res, err = db.execute_query(
                        f'SELECT "{col_name}", COUNT(*) as cnt '
                        f'FROM "{tn}" GROUP BY "{col_name}" ORDER BY cnt DESC LIMIT 5'
                    )
                    if res is not None and not res.empty:
                        top_items = [f"{row[col_name]} ({row['cnt']:,})" for _, row in res.iterrows()]
                        stats_lines.append(f"Top '{col_name}': {', '.join(top_items)}")
                except Exception:
                    pass

        # Missing values summary
        null_cols = [f"{c.name} ({c.null_count} nulls, {c.null_pct}%)" for c in profile.columns if c.null_count > 0]
        if null_cols:
            stats_lines.append(f"Missing Data: {', '.join(null_cols)}")

    return "\n".join(stats_lines)


def format_dataframe_result(df: pd.DataFrame) -> str:
    """Format DataFrame as text representation for LLM context."""
    if df is None or df.empty:
        return "No rows returned."

    if len(df) <= 20:
        return df.to_string(index=False)
    else:
        return df.head(15).to_string(index=False) + f"\n... ({len(df)} total rows returned, showing top 15)"


def render_supporting_metrics(df: pd.DataFrame):
    """Render key computed metrics from a query result."""
    if df is None or df.empty:
        return

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Case A: Single row result with numeric columns
    if len(df) == 1 and num_cols:
        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        cols = st.columns(min(len(num_cols), 4))
        for idx, cname in enumerate(num_cols[:4]):
            val = df.iloc[0][cname]
            if isinstance(val, float):
                formatted = f"{val:,.2f}" if abs(val) >= 100 else f"{val:.2f}"
            elif isinstance(val, int):
                formatted = f"{val:,}"
            else:
                formatted = str(val)

            with cols[idx]:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-val">{formatted}</div>
                        <div class="metric-lbl">{cname.replace('_', ' ').title()}</div>
                    </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


def handle_user_query(question: str, scope: str, selected_tables: list[str]):
    """Process user question or summary request."""

    # Resolve target dataset profiles
    all_profiles = st.session_state.dataset_profiles
    selected_profiles = [all_profiles[tn] for tn in selected_tables if tn in all_profiles]

    if not selected_profiles:
        st.markdown(
            '<div class="error-box"><strong>No Dataset Selected:</strong> Please select at least one dataset to analyze.</div>',
            unsafe_allow_html=True,
        )
        return

    scope_desc = f"Scope: {scope} ({', '.join([p.file_name for p in selected_profiles])})"

    # Check for summary intent
    is_summary = any(kw in question.lower() for kw in ["summarize", "summary", "overview", "describe dataset"])

    if is_summary:
        with st.status("Generating Dataset Summary...", expanded=True) as status:
            st.write("Reading dataset structure...")
            st.write("Computing database statistics...")
            stats_str = compute_statistics_for_summary(selected_profiles)

            st.write("Preparing summary...")
            is_combined = len(selected_profiles) > 1
            summary_text, err = generate_summary(selected_profiles, stats_str, is_combined)

            if err:
                status.update(label="Summary Generation Failed", state="error")
                st.markdown(f'<div class="error-box"><strong>Error:</strong> {err}</div>', unsafe_allow_html=True)
                return

            status.update(label="Summary Ready", state="complete")

            # Structured result architecture (no visualization)
            st.session_state.last_result = {
                "question": question,
                "answer": summary_text,
                "sql": None,
                "result_df": None,
                "tables_used": [p.table_name for p in selected_profiles],
                "scope": scope_desc,
                "status": "Success",
                "visualization": None,
            }
        return

    # ── Standard Analytical Flow ──
    with st.status("Analyzing your request...", expanded=True) as status:
        st.write("Understanding your question...")

        all_available_table_names = list(st.session_state.dataset_profiles.keys())
        raw_sql, err = generate_sql(question, selected_profiles, scope_desc, all_available_table_names)

        if err:
            status.update(label="Analysis Failed", state="error")
            st.markdown(f'<div class="error-box"><strong>Gemini API Error:</strong> {err}</div>', unsafe_allow_html=True)
            return

        clean_sql = clean_sql_from_llm(raw_sql)

        # Validate SQL Safety
        is_safe, safety_err = validate_sql(clean_sql)
        if not is_safe:
            status.update(label="Security Error", state="error")
            st.markdown(f'<div class="error-box"><strong>SQL Safety Violation:</strong> {safety_err}</div>', unsafe_allow_html=True)
            return

        st.write("Running analysis against DuckDB...")
        max_retries = 2
        result_df = None
        exec_err = ""
        final_sql = clean_sql

        for attempt in range(max_retries + 1):
            if attempt > 0:
                st.write(f"Refining query (Attempt {attempt + 1})...")

            result_df, exec_err = db.execute_query(final_sql)
            if result_df is not None:
                break

            if attempt < max_retries:
                st.write("Correcting SQL query...")
                retry_sql, retry_err = retry_sql_generation(
                    question, selected_profiles, scope_desc, final_sql, exec_err, all_available_table_names
                )
                if retry_err:
                    break
                final_sql = clean_sql_from_llm(retry_sql)
                is_safe, safety_err = validate_sql(final_sql)
                if not is_safe:
                    exec_err = safety_err
                    break

        if result_df is None:
            status.update(label="Query Execution Error", state="error")

            # Friendly column suggestion
            available_cols = []
            for p in selected_profiles:
                available_cols.extend([c.name for c in p.columns])

            st.markdown(f"""
                <div class="error-box">
                    <strong>Query Execution Failed:</strong> {exec_err}<br><br>
                    <strong>Available Columns in Selected Dataset:</strong> {', '.join(sorted(set(available_cols)))}
                </div>
            """, unsafe_allow_html=True)
            return

        st.write("Preparing answer...")
        result_text = format_dataframe_result(result_df)
        explanation, explain_err = explain_result(question, final_sql, result_text, selected_profiles, scope_desc)

        if explain_err:
            explanation = f"Query returned {len(result_df)} row(s)."

        status.update(label="Analysis Complete", state="complete")

        # Result object (no visualization)
        st.session_state.last_result = {
            "question": question,
            "answer": explanation,
            "sql": final_sql,
            "result_df": result_df,
            "tables_used": [p.table_name for p in selected_profiles],
            "scope": scope_desc,
            "status": "Success",
            "visualization": None,  # Reserved for future
        }


# ─── Main Interface ──────────────────────────────────────────────────────────

# Workspace Header
st.markdown("""
    <div class="app-header">
        <div class="app-brand">
            <div class="app-logo">◈</div>
            <div>
                <div class="app-title">DataLens</div>
                <div class="app-subtitle">Professional Data Analysis & Q&A Workspace</div>
            </div>
        </div>
        <div class="session-badge">
            <span class="session-dot"></span> Session Active
        </div>
    </div>
""", unsafe_allow_html=True)


# ── Section 1: YOUR DATA ─────────────────────────────────────────────────────

st.markdown('<div class="section-header">1. Your Data</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Drop CSV or Excel files here",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

# Process uploads & sync session tables
if uploaded_files:
    current_names = {f.name for f in uploaded_files}

    # Clean up removed files
    for old_file in list(st.session_state.uploaded_files_set):
        if old_file not in current_names:
            tbl_to_remove = sanitize_table_name(old_file)
            db.drop_table(tbl_to_remove)
            st.session_state.dataset_profiles.pop(tbl_to_remove, None)
            st.session_state.dataframes.pop(tbl_to_remove, None)

    st.session_state.uploaded_files_set = current_names

    for ufile in uploaded_files:
        tbl_name = sanitize_table_name(ufile.name)
        if tbl_name in st.session_state.dataset_profiles:
            continue

        valid, err_msg = validate_file(ufile.name)
        if not valid:
            st.markdown(f'<div class="error-box"><strong>Invalid File:</strong> {err_msg}</div>', unsafe_allow_html=True)
            continue

        df, load_err = load_file(ufile, ufile.name)
        if load_err:
            st.markdown(f'<div class="error-box"><strong>Error loading {ufile.name}:</strong> {load_err}</div>', unsafe_allow_html=True)
            continue

        # Profile dataset
        profile = profile_dataset(df, ufile.name)

        # Register in DuckDB
        db.register_dataframe(tbl_name, df, profile.to_dict())

        st.session_state.dataset_profiles[tbl_name] = profile
        st.session_state.dataframes[tbl_name] = df

# Render Uploaded Datasets List & Schema Views
if st.session_state.dataset_profiles:
    for tn, prof in st.session_state.dataset_profiles.items():
        col_c1, col_c2, col_c3 = st.columns([3, 1, 1])

        with col_c1:
            st.markdown(f"""
                <div class="dataset-card">
                    <div class="dataset-info">
                        <div class="dataset-icon">{prof.file_type}</div>
                        <div>
                            <div class="dataset-name">{prof.file_name}</div>
                            <div class="dataset-meta">{prof.row_count:,} rows · {prof.column_count} columns · Table: <code>{prof.table_name}</code></div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col_c2:
            view_schema_key = f"toggle_schema_{tn}"
            if st.button("View schema", key=f"btn_schema_{tn}", use_container_width=True):
                st.session_state.schema_view_toggles[tn] = not st.session_state.schema_view_toggles.get(tn, False)

        with col_c3:
            if st.button("Summarize", key=f"btn_sum_{tn}", use_container_width=True):
                handle_user_query(f"Summarize {prof.file_name}", "Current file", [tn])

        # Schema View Expander
        if st.session_state.schema_view_toggles.get(tn, False):
            with st.container():
                st.markdown(f"**Schema Extraction Profile for `{prof.file_name}`**")
                schema_data = []
                for c in prof.columns:
                    sample_str = ", ".join(str(s) for s in c.sample_values[:3]) if c.sample_values else "N/A"
                    if c.semantic_type in ("date", "datetime") and c.min_date:
                        sample_str = f"Range: {c.min_date} to {c.max_date}"
                    elif c.min_val is not None and c.max_val is not None:
                        sample_str = f"Min: {c.min_val}, Max: {c.max_val}, Mean: {c.mean_val}"

                    schema_data.append({
                        "Column": c.name,
                        "Physical Type": c.physical_dtype,
                        "Semantic Type": c.semantic_type,
                        "Null %": f"{c.null_pct}% ({c.null_count})",
                        "Unique Values": c.unique_count,
                        "Samples / Range": sample_str,
                    })

                st.dataframe(pd.DataFrame(schema_data), use_container_width=True, hide_index=True)
                st.markdown("---")

else:
    st.markdown("""
        <div style="text-align: center; padding: 2.5rem 1rem; color: #64748b; background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 8px;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem; color: #0f766e;">◈</div>
            <div style="font-size: 0.9375rem; font-weight: 600; color: #1e293b;">Start with your data</div>
            <div style="font-size: 0.8125rem; color: #64748b; margin-top: 0.25rem;">Upload one or multiple CSV or Excel files to begin analysis.</div>
        </div>
    """, unsafe_allow_html=True)


# ── Section 2: ANALYZE ───────────────────────────────────────────────────────

if st.session_state.dataset_profiles:
    st.markdown('<div class="section-header">2. Analyze</div>', unsafe_allow_html=True)

    all_table_names = list(st.session_state.dataset_profiles.keys())
    all_profiles_list = list(st.session_state.dataset_profiles.values())
    compatible, compat_err = check_schema_compatibility(all_profiles_list)

    # Scope Selection
    scope_options = ["Current file"]
    if len(all_table_names) > 1:
        scope_options.append("Selected files")
        if compatible:
            scope_options.append("All compatible files")

    col_scope, col_dataset_sel = st.columns([1, 2])

    with col_scope:
        selected_scope = st.selectbox(
            "Analysis scope",
            scope_options,
            index=0,
            help="Choose dataset scope for the analysis",
        )

    with col_dataset_sel:
        filename_map = {prof.file_name: tn for tn, prof in st.session_state.dataset_profiles.items()}

        if selected_scope == "Current file":
            chosen_file = st.selectbox("Target dataset", list(filename_map.keys()))
            target_tables = [filename_map[chosen_file]] if chosen_file else []

        elif selected_scope == "Selected files":
            chosen_files = st.multiselect("Target datasets", list(filename_map.keys()), default=list(filename_map.keys()))
            target_tables = [filename_map[f] for f in chosen_files]

        else:  # All compatible files
            target_tables = all_table_names
            files_str = ", ".join([p.file_name for p in all_profiles_list])
            st.markdown(
                f'<div class="scope-pill">◈ Combined scope across {len(all_table_names)} files: {files_str}</div>',
                unsafe_allow_html=True,
            )

    if not compatible and selected_scope in ["Selected files", "All compatible files"] and len(all_table_names) > 1:
        st.warning(f"⚠️ {compat_err}")

    # Question Input
    user_question = st.text_input(
        "Ask a question about your data",
        placeholder="e.g. Which region generated the highest revenue? Summarize this file.",
        key="main_question_input",
    )

    # Action buttons & Suggested questions
    col_act1, col_act2 = st.columns([4, 1])

    with col_act2:
        run_clicked = st.button("Run analysis", type="primary", use_container_width=True)

    # Suggested Chips
    suggestions = [
        "Summarize this file",
        "What is the total revenue?",
        "Which product performed best?",
        "Show revenue by region",
    ]
    if len(all_table_names) > 1 and compatible:
        suggestions.append("Give me a combined summary")
        suggestions.append("Compare selected files")

    st.markdown('<div style="font-size: 0.75rem; color: #64748b; font-weight: 600; margin-top: 0.5rem; margin-bottom: 0.375rem;">SUGGESTED QUESTIONS:</div>', unsafe_allow_html=True)
    chip_cols = st.columns(min(len(suggestions), 5))
    for i, sugg in enumerate(suggestions[:5]):
        with chip_cols[i]:
            if st.button(sugg, key=f"chip_{i}", use_container_width=True):
                user_question = sugg
                run_clicked = True

    if run_clicked and user_question:
        handle_user_query(user_question, selected_scope, target_tables)


# ── Section 3: RESULT ────────────────────────────────────────────────────────

if st.session_state.last_result:
    res = st.session_state.last_result

    st.markdown('<div class="section-header">3. Result</div>', unsafe_allow_html=True)

    # Key computed metrics grid (if single row result)
    if res["result_df"] is not None:
        render_supporting_metrics(res["result_df"])

    # Main Answer Card
    if res["answer"]:
        st.markdown(f"""
            <div class="answer-card">
                <div class="answer-title">Answer</div>
                <div class="answer-body">{res["answer"]}</div>
            </div>
        """, unsafe_allow_html=True)

    # Supporting Data Table
    if res["result_df"] is not None and not res["result_df"].empty:
        with st.expander(f"Supporting Data Table ({len(res['result_df']):,} rows)", expanded=len(res["result_df"]) <= 15):
            st.dataframe(res["result_df"], use_container_width=True, hide_index=True)

    # Analysis Details
    with st.expander("Analysis details"):
        det_c1, det_c2, det_c3 = st.columns(3)
        with det_c1:
            st.markdown("**Datasets Analyzed**")
            for tn in res["tables_used"]:
                prof = st.session_state.dataset_profiles.get(tn)
                st.markdown(f"- {prof.file_name if prof else tn}")

        with det_c2:
            st.markdown("**Scope**")
            st.markdown(res["scope"])

        with det_c3:
            st.markdown("**Status**")
            st.markdown(f"✓ {res['status']}")

        if res["sql"]:
            st.markdown("**Generated SQL**")
            st.markdown(f'<div class="sql-code">{res["sql"]}</div>', unsafe_allow_html=True)
