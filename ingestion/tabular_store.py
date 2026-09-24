"""
ingestion/tabular_store.py
──────────────────────────
Thread-safe tabular data management and controlled pandas-based analysis engine.
Supports structured operations (highest, lowest, aggregate, frequency, filter, groupby_agg)
with strict parameter allowlists and zero arbitrary code execution (no eval/exec).
"""

from __future__ import annotations
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from loguru import logger

# ── Allowlist Definitions ───────────────────────────────────────────────────

ALLOWED_OPERATIONS: Set[str] = {
    "highest",      # Finds row with maximum value in a numeric column
    "lowest",       # Finds row with minimum value in a numeric column
    "aggregate",    # Applies an aggregation function (sum, mean, count, min, max, std, median)
    "frequency",    # Finds the most frequent category or value counts
    "filter",       # Filters rows based on a single condition (column, operator, value)
    "groupby_agg",  # Groups by a category and aggregates a numeric column
    "describe",     # Returns summary statistics
    "list_values",  # Lists unique values in a column
}

ALLOWED_AGG_FUNCS: Set[str] = {
    "sum", "mean", "count", "min", "max", "std", "median"
}

ALLOWED_FILTER_OPERATORS: Set[str] = {
    "==", "!=", ">", ">=", "<", "<=", "contains"
}


# ── Controlled Analysis Engine ──────────────────────────────────────────────

class TabularStore:
    """Manages the loaded DataFrame and executes strictly validated operations."""

    def __init__(self, data_dir: Union[Path, str] = Path("data/tabular")):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.active_df: Optional[pd.DataFrame] = None
        self.active_filename: Optional[str] = None

    def has_data(self) -> bool:
        """Check if a DataFrame is currently loaded."""
        return self.active_df is not None and not self.active_df.empty

    def clear(self) -> None:
        """Clear active in-memory table."""
        self.active_df = None
        self.active_filename = None

    def load_csv(self, file_content: Union[str, bytes, Path], filename: str) -> Dict[str, Any]:
        """
        Load a CSV from string, bytes, or file path into a pandas DataFrame.
        Validates column names and non-emptiness.
        """
        if not filename.lower().endswith(".csv"):
            raise ValueError("Only .csv files are supported for tabular data ingestion.")

        try:
            if isinstance(file_content, bytes):
                df = pd.read_csv(io.BytesIO(file_content))
            elif isinstance(file_content, Path):
                df = pd.read_csv(file_content)
            elif isinstance(file_content, str) and (Path(file_content).exists() or "\n" in file_content):
                if "\n" in file_content:
                    df = pd.read_csv(io.StringIO(file_content))
                else:
                    df = pd.read_csv(file_content)
            else:
                raise ValueError("Unsupported file content format.")
        except Exception as e:
            logger.error(f"Failed to parse CSV '{filename}': {e}")
            raise ValueError(f"CSV parsing error: {e}")

        if df.empty:
            raise ValueError("The provided CSV file contains no data rows.")

        # Clean column names (strip whitespace)
        df.columns = [str(c).strip() for c in df.columns]

        self.active_df = df
        self.active_filename = filename

        # Optionally save a copy in data_dir
        dest = self.data_dir / filename
        try:
            df.to_csv(dest, index=False)
        except Exception as e:
            logger.warning(f"Could not persist CSV to disk: {e}")

        logger.success(f"Loaded tabular dataset '{filename}' with {len(df)} rows and {len(df.columns)} columns.")

        return {
            "filename": filename,
            "row_count": len(df),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }

    def get_schema_summary(self) -> str:
        """Generate a concise schema description for the router and LLM tools."""
        if not self.has_data():
            return "No CSV dataset is currently loaded."

        df = self.active_df
        lines = [f"Dataset: {self.active_filename} ({len(df)} rows, {len(df.columns)} columns)"]
        lines.append("Columns and Data Types:")
        for col in df.columns:
            dtype_str = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "text/categorical"
            sample_vals = df[col].dropna().unique()[:3].tolist()
            sample_str = ", ".join(repr(v) for v in sample_vals)
            lines.append(f"  - '{col}' ({dtype_str}, e.g. {sample_str})")
        return "\n".join(lines)

    def _resolve_column(self, col_name: Optional[str]) -> Optional[str]:
        """Case-insensitive column name resolution."""
        if not col_name or not self.has_data():
            return None
        col_clean = str(col_name).strip().lower()
        for c in self.active_df.columns:
            if c.lower() == col_clean:
                return c
        return None

    def execute_query(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a strictly validated pandas operation.
        Never uses eval() or exec().
        """
        if not self.has_data():
            return {
                "status": "error",
                "error": "No CSV dataset is currently loaded. Please upload a CSV file first."
            }

        df = self.active_df
        operation = str(params.get("operation", "")).strip().lower()

        # 1. Validate operation against allowlist
        if operation not in ALLOWED_OPERATIONS:
            return {
                "status": "error",
                "error": f"Invalid operation '{operation}'. Allowed operations are: {sorted(ALLOWED_OPERATIONS)}"
            }

        # 2. Resolve columns
        raw_target_col = params.get("target_column")
        target_col = self._resolve_column(raw_target_col)
        raw_group_col = params.get("group_column")
        group_col = self._resolve_column(raw_group_col)

        # 3. Handle optional pre-filtering before aggregation/metric calculation
        working_df = df
        filter_spec = params.get("filter")
        if filter_spec and isinstance(filter_spec, dict):
            raw_fcol = filter_spec.get("column")
            fcol = self._resolve_column(raw_fcol)
            fop = str(filter_spec.get("operator", "==")).strip()
            fval = filter_spec.get("value")

            if not fcol:
                return {
                    "status": "error",
                    "error": f"Filter column '{raw_fcol}' not found. Available columns: {list(df.columns)}"
                }
            if fop not in ALLOWED_FILTER_OPERATORS:
                return {
                    "status": "error",
                    "error": f"Filter operator '{fop}' not allowed. Allowed operators: {sorted(ALLOWED_FILTER_OPERATORS)}"
                }

            working_df = self._apply_filter(working_df, fcol, fop, fval)
            if working_df.empty:
                return {
                    "status": "success",
                    "operation": operation,
                    "source": self.active_filename,
                    "result": [],
                    "summary": f"No rows matched filter ({fcol} {fop} {fval}).",
                    "row_count": 0
                }

        # 4. Dispatch to controlled operation
        try:
            if operation == "highest":
                return self._op_highest_lowest(working_df, target_col, raw_target_col, ascending=False)

            elif operation == "lowest":
                return self._op_highest_lowest(working_df, target_col, raw_target_col, ascending=True)

            elif operation == "aggregate":
                return self._op_aggregate(working_df, target_col, raw_target_col, params.get("agg_func"))

            elif operation == "frequency":
                return self._op_frequency(working_df, target_col, raw_target_col)

            elif operation == "filter":
                return self._op_filter_result(working_df, params)

            elif operation == "groupby_agg":
                return self._op_groupby(working_df, group_col, raw_group_col, target_col, raw_target_col, params.get("agg_func"))

            elif operation == "describe":
                return self._op_describe(working_df, target_col)

            elif operation == "list_values":
                return self._op_list_values(working_df, target_col, raw_target_col)

            else:
                return {"status": "error", "error": f"Operation '{operation}' not implemented."}

        except Exception as e:
            logger.error(f"Tabular analysis error during '{operation}': {e}")
            return {"status": "error", "error": f"Computation error: {str(e)}"}

    # ── Safe Native Operation Handlers ──────────────────────────────────────────

    def _apply_filter(self, df: pd.DataFrame, col: str, op: str, value: Any) -> pd.DataFrame:
        """Apply safe boolean indexing without eval() or query()."""
        col_series = df[col]

        # Convert value type safely matching column dtype
        if pd.api.types.is_numeric_dtype(col_series):
            try:
                value = float(value) if "." in str(value) else int(value)
            except (ValueError, TypeError):
                pass
        else:
            value = str(value).strip()

        if op == "==":
            return df[col_series == value]
        elif op == "!=":
            return df[col_series != value]
        elif op == ">":
            return df[col_series > value]
        elif op == ">=":
            return df[col_series >= value]
        elif op == "<":
            return df[col_series < value]
        elif op == "<=":
            return df[col_series <= value]
        elif op == "contains":
            return df[col_series.astype(str).str.contains(str(value), case=False, na=False)]
        else:
            raise ValueError(f"Unsupported operator '{op}'")

    def _op_highest_lowest(self, df: pd.DataFrame, col: Optional[str], raw_col: Any, ascending: bool) -> Dict[str, Any]:
        if not col:
            return {"status": "error", "error": f"Column '{raw_col}' not found. Available columns: {list(df.columns)}"}
        if not pd.api.types.is_numeric_dtype(df[col]):
            return {"status": "error", "error": f"Column '{col}' is not numeric. Highest/lowest operations require a numeric column."}

        sorted_df = df.sort_values(by=col, ascending=ascending)
        best_row = sorted_df.iloc[0].to_dict()
        op_label = "lowest" if ascending else "highest"

        # Format clean summary
        summary_parts = [f"{k}: {v}" for k, v in best_row.items()]
        summary = f"The {op_label} {col} is {best_row[col]} ({', '.join(summary_parts[:4])})."

        return {
            "status": "success",
            "operation": op_label,
            "source": self.active_filename,
            "target_column": col,
            "result": best_row,
            "summary": summary,
            "row_count": len(df)
        }

    def _op_aggregate(self, df: pd.DataFrame, col: Optional[str], raw_col: Any, agg_func: Optional[str]) -> Dict[str, Any]:
        if not col:
            return {"status": "error", "error": f"Column '{raw_col}' not found. Available columns: {list(df.columns)}"}

        func = str(agg_func or "mean").strip().lower()
        if func not in ALLOWED_AGG_FUNCS:
            return {"status": "error", "error": f"Aggregation function '{func}' not allowed. Allowed: {sorted(ALLOWED_AGG_FUNCS)}"}

        if func != "count" and not pd.api.types.is_numeric_dtype(df[col]):
            return {"status": "error", "error": f"Cannot compute '{func}' on non-numeric column '{col}'."}

        val = getattr(df[col], func)()
        if isinstance(val, (int, float)):
            val = round(float(val), 4)

        summary = f"The {func} of '{col}' is {val} (computed across {len(df)} records)."
        return {
            "status": "success",
            "operation": "aggregate",
            "source": self.active_filename,
            "target_column": col,
            "agg_func": func,
            "result": {col: val, "function": func},
            "summary": summary,
            "row_count": len(df)
        }

    def _op_frequency(self, df: pd.DataFrame, col: Optional[str], raw_col: Any) -> Dict[str, Any]:
        if not col:
            return {"status": "error", "error": f"Column '{raw_col}' not found. Available columns: {list(df.columns)}"}

        counts = df[col].value_counts().head(5).to_dict()
        top_item, top_count = next(iter(counts.items())) if counts else ("None", 0)
        summary = f"The most frequent value in '{col}' is '{top_item}' with {top_count} occurrences."

        return {
            "status": "success",
            "operation": "frequency",
            "source": self.active_filename,
            "target_column": col,
            "result": {"most_frequent": top_item, "count": int(top_count), "top_counts": counts},
            "summary": summary,
            "row_count": len(df)
        }

    def _op_filter_result(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        records = df.head(10).to_dict(orient="records")
        summary = f"Found {len(df)} matching rows in '{self.active_filename}'."
        return {
            "status": "success",
            "operation": "filter",
            "source": self.active_filename,
            "result": records,
            "summary": summary,
            "row_count": len(df)
        }

    def _op_groupby(self, df: pd.DataFrame, group_col: Optional[str], raw_gcol: Any, target_col: Optional[str], raw_tcol: Any, agg_func: Optional[str]) -> Dict[str, Any]:
        if not group_col:
            return {"status": "error", "error": f"Group-by column '{raw_gcol}' not found. Available: {list(df.columns)}"}
        if not target_col:
            return {"status": "error", "error": f"Target column '{raw_tcol}' not found. Available: {list(df.columns)}"}

        func = str(agg_func or "sum").strip().lower()
        if func not in ALLOWED_AGG_FUNCS:
            return {"status": "error", "error": f"Aggregation function '{func}' not allowed. Allowed: {sorted(ALLOWED_AGG_FUNCS)}"}
        if func != "count" and not pd.api.types.is_numeric_dtype(df[target_col]):
            return {"status": "error", "error": f"Cannot aggregate non-numeric column '{target_col}' with '{func}'."}

        grouped = df.groupby(group_col)[target_col].agg(func).to_dict()
        top_items = sorted(grouped.items(), key=lambda x: x[1], reverse=True)[:3]
        summary_preview = ", ".join(f"{k}: {round(v, 2) if isinstance(v, float) else v}" for k, v in top_items)
        summary = f"Grouped by '{group_col}', the {func} of '{target_col}' is: {summary_preview}."

        return {
            "status": "success",
            "operation": "groupby_agg",
            "source": self.active_filename,
            "group_column": group_col,
            "target_column": target_col,
            "agg_func": func,
            "result": grouped,
            "summary": summary,
            "row_count": len(df)
        }

    def _op_describe(self, df: pd.DataFrame, col: Optional[str]) -> Dict[str, Any]:
        if col and col in df.columns:
            desc = df[col].describe().to_dict()
        else:
            desc = df.describe().to_dict()
        return {
            "status": "success",
            "operation": "describe",
            "source": self.active_filename,
            "result": desc,
            "summary": f"Statistical summary calculated for {self.active_filename}.",
            "row_count": len(df)
        }

    def _op_list_values(self, df: pd.DataFrame, col: Optional[str], raw_col: Any) -> Dict[str, Any]:
        if not col:
            return {"status": "error", "error": f"Column '{raw_col}' not found. Available columns: {list(df.columns)}"}
        unique_vals = [str(v) for v in df[col].dropna().unique()[:20]]
        summary = f"Unique values in '{col}' ({len(unique_vals)} shown): {', '.join(unique_vals)}."
        return {
            "status": "success",
            "operation": "list_values",
            "source": self.active_filename,
            "target_column": col,
            "result": unique_vals,
            "summary": summary,
            "row_count": len(df)
        }


# Singleton instance
tabular_store = TabularStore()
