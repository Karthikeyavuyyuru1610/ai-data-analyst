"""File ingestion: validation, loading, schema profiling, and semantic type detection."""
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import pandas as pd


@dataclass
class ColumnProfile:
    """Detailed profile for a single column."""
    name: str
    physical_dtype: str
    semantic_type: str
    null_count: int
    null_pct: float
    unique_count: int
    sample_values: list[Any]
    min_val: Optional[Any] = None
    max_val: Optional[Any] = None
    mean_val: Optional[float] = None
    min_date: Optional[str] = None
    max_date: Optional[str] = None


@dataclass
class DatasetProfile:
    """Structured representation of an ingested dataset."""
    file_name: str
    file_type: str  # "CSV" or "XLSX"
    table_name: str
    row_count: int
    column_count: int
    columns: list[ColumnProfile]
    numeric_columns: list[str] = field(default_factory=list)
    date_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)

    def get_column(self, name: str) -> Optional[ColumnProfile]:
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_name": self.file_name,
            "file_type": self.file_type,
            "table_name": self.table_name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": [asdict(c) for c in self.columns],
        }


# Backward compatibility alias for TableMetadata
TableMetadata = DatasetProfile

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def validate_file(filename: str) -> tuple[bool, str]:
    """Validate that a file has an allowed extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file format '{ext}'. Please upload CSV or Excel files."
    return True, ""


def sanitize_table_name(filename: str) -> str:
    """Convert a filename into a valid SQL table name."""
    name = os.path.splitext(filename)[0]
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    name = re.sub(r"^[0-9]+", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "uploaded_table"
    return name.lower()


def load_file(file_obj, filename: str) -> tuple[pd.DataFrame | None, str]:
    """
    Load a CSV or Excel file into a DataFrame.
    Returns (DataFrame, error_message).
    """
    ext = os.path.splitext(filename)[1].lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(file_obj, encoding="utf-8")
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(file_obj, engine="openpyxl")
        else:
            return None, f"Unsupported file format: {ext}"

        if df.empty:
            return None, "The file is empty or contains no data rows."

        if len(df.columns) == 0:
            return None, "The file contains no columns."

        return df, ""

    except pd.errors.EmptyDataError:
        return None, "The file is empty."
    except pd.errors.ParserError as e:
        return None, f"Could not parse the file. It may be malformed. Details: {str(e)[:200]}"
    except UnicodeDecodeError:
        if ext == ".csv":
            try:
                file_obj.seek(0)
                df = pd.read_csv(file_obj, encoding="latin-1")
                if df.empty:
                    return None, "The file is empty or contains no data rows."
                return df, ""
            except Exception:
                return None, "Could not decode the file. Please ensure it uses UTF-8 or Latin-1 encoding."
        return None, "Could not decode the file encoding."
    except Exception as e:
        return None, f"Failed to read the file: {str(e)[:200]}"


def infer_semantic_type(col_name: str, df_series: pd.Series, row_count: int) -> str:
    """
    Infer semantic column type using deterministic Python logic.
    Returns: date, datetime, identifier, categorical, measure, quantity, percentage, currency, text, boolean, unknown.
    """
    name_lower = col_name.lower().strip()
    clean_series = df_series.dropna()

    if len(clean_series) == 0:
        return "unknown"

    # 1. Date / Datetime
    if pd.api.types.is_datetime64_any_dtype(df_series):
        return "datetime"

    if any(kw in name_lower for kw in ["date", "timestamp", "time"]):
        try:
            pd.to_datetime(clean_series.head(20))
            return "date"
        except (ValueError, TypeError):
            pass

    # 2. Boolean
    if pd.api.types.is_bool_dtype(df_series):
        return "boolean"

    if clean_series.dtype == "object":
        sample_set = set(clean_series.head(50).astype(str).str.lower().str.strip())
        if sample_set.issubset({"true", "false", "yes", "no", "y", "n", "0", "1"}):
            return "boolean"

    # 3. Identifier
    unique_count = clean_series.nunique()
    if (
        name_lower.endswith("_id")
        or name_lower == "id"
        or name_lower.endswith("id")
        or any(kw in name_lower for kw in ["code", "sku", "uuid", "guid", "key", "number"])
    ):
        return "identifier"

    if unique_count == row_count and row_count > 10 and not pd.api.types.is_numeric_dtype(df_series):
        return "identifier"

    # 4. Numeric types (Currency, Percentage, Quantity, Measure)
    if pd.api.types.is_numeric_dtype(df_series):
        if any(kw in name_lower for kw in ["pct", "percent", "percentage", "rate", "ratio", "%"]):
            return "percentage"

        if any(kw in name_lower for kw in [
            "revenue", "profit", "price", "cost", "salary", "amount", "sales",
            "fee", "tax", "spend", "val", "income", "margin", "discount", "total_price"
        ]):
            return "currency"

        if any(kw in name_lower for kw in ["qty", "quantity", "count", "units", "items", "stock", "volume", "num_"]):
            return "quantity"

        if "int" in str(df_series.dtype).lower():
            if unique_count <= min(20, max(2, int(row_count * 0.1))):
                return "categorical"
            return "quantity"

        return "measure"

    # 5. Categorical vs Text
    if unique_count <= min(50, max(5, int(row_count * 0.5))):
        return "categorical"

    return "text"


def profile_dataset(df: pd.DataFrame, filename: str) -> DatasetProfile:
    """Extract full column profiling and dataset metadata."""
    table_name = sanitize_table_name(filename)
    ext = os.path.splitext(filename)[1].lower()
    file_type = "CSV" if ext == ".csv" else "XLSX"
    row_count = len(df)
    col_count = len(df.columns)

    col_profiles: list[ColumnProfile] = []
    numeric_cols: list[str] = []
    date_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in df.columns:
        series = df[col]
        clean = series.dropna()
        null_cnt = int(series.isna().sum())
        null_pct = round((null_cnt / row_count) * 100, 2) if row_count > 0 else 0.0
        unique_cnt = int(series.nunique())

        physical_dtype = str(series.dtype)
        semantic_type = infer_semantic_type(col, series, row_count)

        # Sample non-null values
        sample_vals = clean.head(4).tolist()
        # Convert timestamps or non-serializable objects to string
        sample_vals = [
            str(v) if isinstance(v, (pd.Timestamp, pd.Timedelta)) else v
            for v in sample_vals
        ]

        min_val, max_val, mean_val = None, None, None
        min_date, max_date = None, None

        if semantic_type in ("date", "datetime"):
            date_cols.append(col)
            try:
                dt_series = pd.to_datetime(clean)
                min_date = str(dt_series.min().strftime("%Y-%m-%d")) if len(dt_series) > 0 else None
                max_date = str(dt_series.max().strftime("%Y-%m-%d")) if len(dt_series) > 0 else None
            except Exception:
                pass

        elif pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
            if len(clean) > 0:
                try:
                    min_val = round(float(clean.min()), 2)
                    max_val = round(float(clean.max()), 2)
                    mean_val = round(float(clean.mean()), 2)
                except Exception:
                    pass

        elif semantic_type == "categorical":
            categorical_cols.append(col)

        col_profiles.append(
            ColumnProfile(
                name=col,
                physical_dtype=physical_dtype,
                semantic_type=semantic_type,
                null_count=null_cnt,
                null_pct=null_pct,
                unique_count=unique_cnt,
                sample_values=sample_vals,
                min_val=min_val,
                max_val=max_val,
                mean_val=mean_val,
                min_date=min_date,
                max_date=max_date,
            )
        )

    return DatasetProfile(
        file_name=filename,
        file_type=file_type,
        table_name=table_name,
        row_count=row_count,
        column_count=col_count,
        columns=col_profiles,
        numeric_columns=numeric_cols,
        date_columns=date_cols,
        categorical_columns=categorical_cols,
    )


# Alias for backward compatibility
detect_schema = profile_dataset


def check_schema_compatibility(meta_list: list[DatasetProfile]) -> tuple[bool, str]:
    """
    Check if multiple tables have compatible schemas for combined analysis.
    Compatible = same column names (order doesn't matter).
    """
    if len(meta_list) < 2:
        return True, ""

    base_cols = set(c.name for c in meta_list[0].columns)
    base_name = meta_list[0].file_name

    for meta in meta_list[1:]:
        other_cols = set(c.name for c in meta.columns)
        if base_cols != other_cols:
            missing = base_cols - other_cols
            extra = other_cols - base_cols
            details = []
            if missing:
                details.append(f"missing columns: {', '.join(sorted(missing))}")
            if extra:
                details.append(f"extra columns: {', '.join(sorted(extra))}")
            return False, (
                f"'{meta.file_name}' is not compatible with '{base_name}'. "
                f"Schema differences: {'; '.join(details)}."
            )

    return True, ""
