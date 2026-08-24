"""DuckDB session management and query execution."""
from __future__ import annotations

from typing import Any

import duckdb
import pandas as pd


class Database:
    """In-memory DuckDB database for session-scoped analytics."""

    def __init__(self):
        self.conn = duckdb.connect(database=":memory:")
        self._tables: dict[str, dict] = {}  # table_name -> metadata dict

    def register_dataframe(self, table_name: str, df: pd.DataFrame, metadata: dict) -> None:
        """Register a pandas DataFrame as a DuckDB table."""
        # Drop existing table if re-uploading
        self.conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        self.conn.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM df')
        self._tables[table_name] = metadata

    def get_table_names(self) -> list[str]:
        """Return list of registered table names."""
        return list(self._tables.keys())

    def get_table_metadata(self, table_name: str) -> dict | None:
        """Return metadata for a specific table."""
        return self._tables.get(table_name)

    def get_all_metadata(self) -> dict[str, dict]:
        """Return metadata for all tables."""
        return dict(self._tables)

    def get_schema_info(self, table_name: str) -> list[dict]:
        """Get column names and types from DuckDB for a table."""
        try:
            result = self.conn.execute(
                f"SELECT column_name, data_type FROM information_schema.columns "
                f"WHERE table_name = '{table_name}' ORDER BY ordinal_position"
            ).fetchall()
            return [{"column": row[0], "type": row[1]} for row in result]
        except Exception:
            return []

    def get_sample_values(self, table_name: str, limit: int = 3) -> dict[str, list]:
        """Get sample values for each column (for LLM context)."""
        try:
            df = self.conn.execute(
                f'SELECT * FROM "{table_name}" LIMIT {limit}'
            ).fetchdf()
            return {col: df[col].tolist() for col in df.columns}
        except Exception:
            return {}

    def execute_query(self, sql: str) -> tuple[pd.DataFrame | None, str]:
        """
        Execute a SQL query and return results.
        Returns (DataFrame, error_message). On success, error_message is empty.
        """
        try:
            result = self.conn.execute(sql).fetchdf()
            return result, ""
        except duckdb.CatalogException as e:
            return None, f"Table or column not found: {str(e)}"
        except duckdb.ParserException as e:
            return None, f"SQL syntax error: {str(e)}"
        except duckdb.BinderException as e:
            return None, f"Column reference error: {str(e)}"
        except Exception as e:
            return None, f"Query execution error: {str(e)}"

    def get_row_count(self, table_name: str) -> int:
        """Get row count for a table."""
        try:
            result = self.conn.execute(
                f'SELECT COUNT(*) FROM "{table_name}"'
            ).fetchone()
            return result[0] if result else 0
        except Exception:
            return 0

    def create_combined_view(self, view_name: str, table_names: list[str]) -> tuple[bool, str]:
        """Create a UNION ALL view of compatible tables."""
        if not table_names:
            return False, "No tables specified."

        try:
            self.conn.execute(f'DROP VIEW IF EXISTS "{view_name}"')
            unions = [f'SELECT *, \'{tn}\' AS _source_table FROM "{tn}"' for tn in table_names]
            sql = f'CREATE VIEW "{view_name}" AS ' + " UNION ALL ".join(unions)
            self.conn.execute(sql)
            return True, ""
        except Exception as e:
            return False, f"Failed to create combined view: {str(e)}"

    def drop_table(self, table_name: str) -> None:
        """Remove a table."""
        try:
            self.conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
            self.conn.execute(f'DROP VIEW IF EXISTS "{table_name}"')
            self._tables.pop(table_name, None)
        except Exception:
            pass

    def close(self) -> None:
        """Close the connection."""
        try:
            self.conn.close()
        except Exception:
            pass
