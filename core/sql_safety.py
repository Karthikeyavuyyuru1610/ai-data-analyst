"""SQL safety validation — only allow read-only analytical queries."""
import re

# Destructive SQL keywords that must be rejected
BLOCKED_KEYWORDS = [
    r"\bINSERT\b",
    r"\bUPDATE\b",
    r"\bDELETE\b",
    r"\bDROP\b",
    r"\bALTER\b",
    r"\bTRUNCATE\b",
    r"\bCREATE\b",
    r"\bEXECUTE\b",
    r"\bEXEC\b",
    r"\bGRANT\b",
    r"\bREVOKE\b",
    r"\bATTACH\b",
    r"\bDETACH\b",
    r"\bCOPY\b",
    r"\bEXPORT\b",
    r"\bIMPORT\b",
    r"\bCALL\b",
    r"\bSET\b",
    r"\bPRAGMA\b",
]

# Allowed statement starters
ALLOWED_STARTERS = [
    r"^\s*SELECT\b",
    r"^\s*WITH\b",
]


def validate_sql(sql: str) -> tuple[bool, str]:
    """
    Validate that SQL is read-only and safe to execute.
    Returns (is_valid, error_message).
    """
    if not sql or not sql.strip():
        return False, "Empty SQL query."

    cleaned = sql.strip()

    # Remove SQL comments
    cleaned_no_comments = re.sub(r"--.*$", "", cleaned, flags=re.MULTILINE)
    cleaned_no_comments = re.sub(r"/\*.*?\*/", "", cleaned_no_comments, flags=re.DOTALL)
    cleaned_no_comments = cleaned_no_comments.strip()

    if not cleaned_no_comments:
        return False, "Empty SQL query after removing comments."

    # Check for blocked keywords
    for pattern in BLOCKED_KEYWORDS:
        if re.search(pattern, cleaned_no_comments, re.IGNORECASE):
            keyword = re.search(pattern, cleaned_no_comments, re.IGNORECASE).group()
            return False, f"Destructive SQL operation '{keyword.strip()}' is not allowed. Only SELECT queries are permitted."

    # Check that query starts with an allowed keyword
    is_allowed = any(
        re.match(pat, cleaned_no_comments, re.IGNORECASE)
        for pat in ALLOWED_STARTERS
    )

    if not is_allowed:
        return False, "Only SELECT and WITH (CTE) queries are allowed."

    # Check for multiple statements (semicolon-separated)
    # Allow trailing semicolon but not multiple statements
    statements = [s.strip() for s in cleaned_no_comments.split(";") if s.strip()]
    if len(statements) > 1:
        return False, "Multiple SQL statements are not allowed. Please use a single query."

    return True, ""


def clean_sql_from_llm(raw: str) -> str:
    """
    Extract SQL from LLM response that may contain markdown code blocks.
    """
    # Remove markdown code blocks
    sql = raw.strip()

    # Pattern: ```sql ... ``` or ``` ... ```
    match = re.search(r"```(?:sql)?\s*\n?(.*?)\n?\s*```", sql, re.DOTALL | re.IGNORECASE)
    if match:
        sql = match.group(1).strip()

    # Remove trailing semicolons (DuckDB doesn't need them)
    sql = sql.rstrip(";").strip()

    return sql
