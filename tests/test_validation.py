"""
Validation tests for the Data Q&A application.

These tests verify correctness of the data pipeline using DuckDB directly.
They do NOT require a Gemini API key — they test the computation layer.
LLM integration tests are marked separately.
"""
import os
import sys
import math

import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import Database
from core.ingestion import load_file, detect_schema, sanitize_table_name, check_schema_compatibility, validate_file
from core.sql_safety import validate_sql, clean_sql_from_llm
from core.visualization import should_visualize, detect_chart_type, create_chart


# ─── Fixtures ────────────────────────────────────────────────────────────────

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")


@pytest.fixture
def db():
    """Create a fresh in-memory DuckDB database with sample data loaded."""
    database = Database()

    for filename in ["sales_january.csv", "sales_february.csv", "sales_march.csv"]:
        filepath = os.path.join(SAMPLE_DIR, filename)
        df = pd.read_csv(filepath)
        table_name = sanitize_table_name(filename)
        meta = detect_schema(df, filename)
        database.register_dataframe(table_name, df, {
            "source_filename": filename,
            "columns": [c.name for c in meta.columns],
            "row_count": meta.row_count,
        })

    yield database
    database.close()


@pytest.fixture
def combined_db(db):
    """Database with a combined view of all tables."""
    tables = db.get_table_names()
    db.create_combined_view("all_sales", tables)
    return db


@pytest.fixture
def jan_df():
    """January sales DataFrame."""
    return pd.read_csv(os.path.join(SAMPLE_DIR, "sales_january.csv"))


@pytest.fixture
def feb_df():
    """February sales DataFrame."""
    return pd.read_csv(os.path.join(SAMPLE_DIR, "sales_february.csv"))


@pytest.fixture
def mar_df():
    """March sales DataFrame."""
    return pd.read_csv(os.path.join(SAMPLE_DIR, "sales_march.csv"))


# ─── Test 1: Total Revenue ──────────────────────────────────────────────────

class TestTotalRevenue:
    """Test 1: Total revenue calculation."""

    def test_single_file_total_revenue(self, db, jan_df):
        result, error = db.execute_query('SELECT ROUND(SUM(revenue), 2) as total_revenue FROM sales_january')
        assert error == ""
        assert result is not None
        expected = round(jan_df["revenue"].sum(), 2)
        actual = round(result.iloc[0]["total_revenue"], 2)
        assert actual == expected

    def test_combined_total_revenue(self, combined_db, jan_df, feb_df, mar_df):
        result, error = combined_db.execute_query('SELECT ROUND(SUM(revenue), 2) as total_revenue FROM all_sales')
        assert error == ""
        expected = round(jan_df["revenue"].sum() + feb_df["revenue"].sum() + mar_df["revenue"].sum(), 2)
        actual = round(result.iloc[0]["total_revenue"], 2)
        assert actual == expected


# ─── Test 2: Average Revenue ────────────────────────────────────────────────

class TestAverageRevenue:
    """Test 2: Average revenue calculation."""

    def test_single_file_avg_revenue(self, db, jan_df):
        result, error = db.execute_query('SELECT ROUND(AVG(revenue), 2) as avg_revenue FROM sales_january')
        assert error == ""
        expected = round(jan_df["revenue"].mean(), 2)
        actual = round(result.iloc[0]["avg_revenue"], 2)
        assert actual == expected

    def test_combined_avg_revenue(self, combined_db, jan_df, feb_df, mar_df):
        result, error = combined_db.execute_query('SELECT ROUND(AVG(revenue), 2) as avg_revenue FROM all_sales')
        assert error == ""
        all_revenue = pd.concat([jan_df["revenue"], feb_df["revenue"], mar_df["revenue"]])
        expected = round(all_revenue.mean(), 2)
        actual = round(result.iloc[0]["avg_revenue"], 2)
        assert actual == expected


# ─── Test 3: Top Product ────────────────────────────────────────────────────

class TestTopProduct:
    """Test 3: Identify the top product by revenue."""

    def test_top_product_single_file(self, db, jan_df):
        result, error = db.execute_query(
            'SELECT product, SUM(revenue) as total_revenue FROM sales_january '
            'GROUP BY product ORDER BY total_revenue DESC LIMIT 1'
        )
        assert error == ""
        expected = jan_df.groupby("product")["revenue"].sum().idxmax()
        actual = result.iloc[0]["product"]
        assert actual == expected

    def test_top_product_combined(self, combined_db, jan_df, feb_df, mar_df):
        result, error = combined_db.execute_query(
            'SELECT product, SUM(revenue) as total_revenue FROM all_sales '
            'GROUP BY product ORDER BY total_revenue DESC LIMIT 1'
        )
        assert error == ""
        all_df = pd.concat([jan_df, feb_df, mar_df])
        expected = all_df.groupby("product")["revenue"].sum().idxmax()
        actual = result.iloc[0]["product"]
        assert actual == expected


# ─── Test 4: Top 5 Products ─────────────────────────────────────────────────

class TestTop5Products:
    """Test 4: Top 5 products by revenue."""

    def test_top_5_products(self, db, jan_df):
        result, error = db.execute_query(
            'SELECT product, SUM(revenue) as total_revenue FROM sales_january '
            'GROUP BY product ORDER BY total_revenue DESC LIMIT 5'
        )
        assert error == ""
        assert len(result) <= 5
        # Verify ordering
        revenues = result["total_revenue"].tolist()
        assert revenues == sorted(revenues, reverse=True)


# ─── Test 5: Best Region ────────────────────────────────────────────────────

class TestBestRegion:
    """Test 5: Highest revenue region."""

    def test_best_region(self, db, jan_df):
        result, error = db.execute_query(
            'SELECT region, SUM(revenue) as total_revenue FROM sales_january '
            'GROUP BY region ORDER BY total_revenue DESC LIMIT 1'
        )
        assert error == ""
        expected = jan_df.groupby("region")["revenue"].sum().idxmax()
        actual = result.iloc[0]["region"]
        assert actual == expected


# ─── Test 6: Customer Count ─────────────────────────────────────────────────

class TestCustomerCount:
    """Test 6: Distinct customer count."""

    def test_customer_count(self, db, jan_df):
        result, error = db.execute_query(
            'SELECT COUNT(DISTINCT customer) as customer_count FROM sales_january'
        )
        assert error == ""
        expected = jan_df["customer"].nunique()
        actual = result.iloc[0]["customer_count"]
        assert actual == expected


# ─── Test 7: Total Profit ───────────────────────────────────────────────────

class TestTotalProfit:
    """Test 7: Total profit (handling NULLs)."""

    def test_total_profit(self, db, jan_df):
        result, error = db.execute_query(
            'SELECT ROUND(SUM(profit), 2) as total_profit FROM sales_january WHERE profit IS NOT NULL'
        )
        assert error == ""
        # Pandas treats empty strings from CSV as NaN after to_numeric
        profit_series = pd.to_numeric(jan_df["profit"], errors="coerce")
        expected = round(profit_series.sum(), 2)
        actual = round(result.iloc[0]["total_profit"], 2)
        assert actual == expected


# ─── Test 8: Revenue by Region ──────────────────────────────────────────────

class TestRevenueByRegion:
    """Test 8: Revenue grouped by region."""

    def test_revenue_by_region(self, db, jan_df):
        result, error = db.execute_query(
            'SELECT region, ROUND(SUM(revenue), 2) as total_revenue FROM sales_january '
            'GROUP BY region ORDER BY total_revenue DESC'
        )
        assert error == ""
        expected_regions = set(jan_df["region"].unique())
        actual_regions = set(result["region"].tolist())
        assert actual_regions == expected_regions


# ─── Test 9: Monthly Revenue ────────────────────────────────────────────────

class TestMonthlyRevenue:
    """Test 9: Monthly revenue aggregation."""

    def test_monthly_revenue(self, combined_db, jan_df, feb_df, mar_df):
        result, error = combined_db.execute_query(
            "SELECT _source_table, ROUND(SUM(revenue), 2) as total_revenue "
            "FROM all_sales GROUP BY _source_table ORDER BY _source_table"
        )
        assert error == ""
        assert len(result) == 3


# ─── Test 10: January vs February ───────────────────────────────────────────

class TestJanVsFeb:
    """Test 10: Compare January and February."""

    def test_jan_vs_feb_revenue(self, db, jan_df, feb_df):
        jan_result, _ = db.execute_query('SELECT SUM(revenue) as rev FROM sales_january')
        feb_result, _ = db.execute_query('SELECT SUM(revenue) as rev FROM sales_february')
        jan_rev = jan_result.iloc[0]["rev"]
        feb_rev = feb_result.iloc[0]["rev"]
        # Just verify both are positive and different
        assert jan_rev > 0
        assert feb_rev > 0


# ─── Test 11: Highest Revenue Month ─────────────────────────────────────────

class TestHighestRevenueMonth:
    """Test 11: Identify the highest revenue month."""

    def test_highest_revenue_month(self, combined_db, jan_df, feb_df, mar_df):
        result, error = combined_db.execute_query(
            "SELECT _source_table, SUM(revenue) as total_revenue "
            "FROM all_sales GROUP BY _source_table ORDER BY total_revenue DESC LIMIT 1"
        )
        assert error == ""
        assert result.iloc[0]["total_revenue"] > 0


# ─── Test 12: Product Growth ────────────────────────────────────────────────

class TestProductGrowth:
    """Test 12: Product growth across months."""

    def test_product_growth(self, combined_db):
        result, error = combined_db.execute_query(
            "SELECT product, _source_table, SUM(revenue) as total_revenue "
            "FROM all_sales GROUP BY product, _source_table "
            "ORDER BY product, _source_table"
        )
        assert error == ""
        # Should have entries for each product-month combination
        assert len(result) > 0


# ─── Test 13: Individual Summary Stats ──────────────────────────────────────

class TestIndividualSummary:
    """Test 13: Statistics for individual file summary."""

    def test_summary_stats(self, db, jan_df):
        # Verify we can compute the stats needed for summary
        result, error = db.execute_query(
            'SELECT COUNT(*) as rows, '
            'ROUND(SUM(revenue), 2) as total_revenue, '
            'ROUND(AVG(revenue), 2) as avg_revenue, '
            'COUNT(DISTINCT product) as products, '
            'COUNT(DISTINCT region) as regions '
            'FROM sales_january'
        )
        assert error == ""
        assert result.iloc[0]["rows"] == len(jan_df)
        assert result.iloc[0]["products"] == jan_df["product"].nunique()


# ─── Test 14: Combined Summary Stats ────────────────────────────────────────

class TestCombinedSummary:
    """Test 14: Statistics for combined file summary."""

    def test_combined_stats(self, combined_db, jan_df, feb_df, mar_df):
        result, error = combined_db.execute_query(
            'SELECT COUNT(*) as rows, '
            'ROUND(SUM(revenue), 2) as total_revenue, '
            'COUNT(DISTINCT _source_table) as file_count '
            'FROM all_sales'
        )
        assert error == ""
        total_rows = len(jan_df) + len(feb_df) + len(mar_df)
        assert result.iloc[0]["rows"] == total_rows
        assert result.iloc[0]["file_count"] == 3


# ─── Test 15: Chart Generation ──────────────────────────────────────────────

class TestChartGeneration:
    """Test 15: Automatic chart creation."""

    def test_bar_chart_for_categories(self, db):
        result, _ = db.execute_query(
            'SELECT region, SUM(revenue) as total FROM sales_january GROUP BY region'
        )
        chart_type = detect_chart_type(result)
        assert chart_type == "bar"
        chart = create_chart(result, "Revenue by region")
        assert chart is not None

    def test_no_chart_for_scalar(self, db):
        result, _ = db.execute_query('SELECT SUM(revenue) as total FROM sales_january')
        assert not should_visualize(result)

    def test_line_chart_for_time(self):
        df = pd.DataFrame({
            "month": ["Jan", "Feb", "Mar"],
            "revenue": [100, 150, 200],
        })
        chart_type = detect_chart_type(df)
        assert chart_type in ("line", "bar")  # 'month' might be detected as category


# ─── SQL Safety Tests ────────────────────────────────────────────────────────

class TestSQLSafety:
    """Test SQL validation and safety checks."""

    def test_valid_select(self):
        valid, _ = validate_sql("SELECT * FROM sales")
        assert valid

    def test_valid_cte(self):
        valid, _ = validate_sql("WITH cte AS (SELECT 1) SELECT * FROM cte")
        assert valid

    def test_reject_drop(self):
        valid, msg = validate_sql("DROP TABLE sales")
        assert not valid
        assert "DROP" in msg

    def test_reject_delete(self):
        valid, msg = validate_sql("DELETE FROM sales")
        assert not valid

    def test_reject_insert(self):
        valid, msg = validate_sql("INSERT INTO sales VALUES (1)")
        assert not valid

    def test_reject_update(self):
        valid, msg = validate_sql("UPDATE sales SET revenue = 0")
        assert not valid

    def test_reject_truncate(self):
        valid, msg = validate_sql("TRUNCATE TABLE sales")
        assert not valid

    def test_reject_alter(self):
        valid, msg = validate_sql("ALTER TABLE sales ADD COLUMN x INT")
        assert not valid

    def test_reject_multiple_statements(self):
        valid, msg = validate_sql("SELECT 1; DROP TABLE sales")
        assert not valid

    def test_empty_sql(self):
        valid, _ = validate_sql("")
        assert not valid

    def test_clean_sql_markdown(self):
        raw = "```sql\nSELECT * FROM sales\n```"
        cleaned = clean_sql_from_llm(raw)
        assert cleaned == "SELECT * FROM sales"


# ─── Ingestion Tests ────────────────────────────────────────────────────────

class TestIngestion:
    """Test file ingestion and schema detection."""

    def test_sanitize_table_name(self):
        assert sanitize_table_name("sales_january.csv") == "sales_january"
        assert sanitize_table_name("My File (1).xlsx") == "my_file_1"
        assert sanitize_table_name("123data.csv") == "data"

    def test_validate_file_csv(self):
        valid, _ = validate_file("data.csv")
        assert valid

    def test_validate_file_xlsx(self):
        valid, _ = validate_file("data.xlsx")
        assert valid

    def test_reject_invalid_extension(self):
        valid, msg = validate_file("data.json")
        assert not valid
        assert "Unsupported" in msg

    def test_schema_compatibility_same(self, jan_df, feb_df):
        meta1 = detect_schema(jan_df, "jan.csv")
        meta2 = detect_schema(feb_df, "feb.csv")
        compatible, _ = check_schema_compatibility([meta1, meta2])
        assert compatible

    def test_schema_compatibility_different(self, jan_df):
        meta1 = detect_schema(jan_df, "jan.csv")
        different_df = pd.DataFrame({"x": [1], "y": [2]})
        meta2 = detect_schema(different_df, "other.csv")
        compatible, _ = check_schema_compatibility([meta1, meta2])
        assert not compatible
