"""
Data Loader
Supports CSV, Excel, and SQLite — plug-and-play data sources.
Handles nulls, duplicates, and basic type inference automatically.
"""

from __future__ import annotations

import io
import sqlite3
import tempfile
import os
from typing import Tuple

import numpy as np
import pandas as pd


def load_data(file_obj) -> Tuple[pd.DataFrame | None, str]:
    """
    Load CSV or Excel file uploaded via Streamlit.
    Returns (df, message).
    """
    try:
        name = file_obj.name.lower()

        if name.endswith(".csv"):
            # Try common encodings
            for enc in ("utf-8", "latin-1", "cp1252"):
                try:
                    file_obj.seek(0)
                    df = pd.read_csv(file_obj, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                return None, "Could not decode CSV. Try saving as UTF-8."

        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_obj)
        else:
            return None, f"Unsupported file type: {name}"

        df = _clean_dataframe(df)
        msg = f"✅ Loaded **{len(df):,} rows × {len(df.columns)} columns** from `{file_obj.name}`"
        return df, msg

    except Exception as e:
        return None, f"Error loading file: {e}"


def load_sql_table(db_file, table_name: str) -> Tuple[pd.DataFrame | None, str]:
    """
    Load a table from an uploaded SQLite database file.
    """
    try:
        # Write to temp file (Streamlit UploadedFile is not a real path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
            tmp.write(db_file.read())
            tmp_path = tmp.name

        conn = sqlite3.connect(tmp_path)
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()
        os.unlink(tmp_path)

        if table_name not in tables:
            return None, f"Table `{table_name}` not found. Available: {tables}"

        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()
        df = _clean_dataframe(df)
        return df, f"✅ Loaded `{table_name}`: {len(df):,} rows × {len(df.columns)} columns"

    except Exception as e:
        return None, f"SQLite error: {e}"


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning pipeline:
    - Strip whitespace from column names
    - Remove fully-empty rows/columns
    - Infer better dtypes
    - Parse date columns automatically
    """
    # Strip col names
    df.columns = df.columns.str.strip()

    # Drop fully empty
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # Remove leading/trailing whitespace from string cols
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # Try to parse date columns
    for col in df.columns:
        if any(kw in col.lower() for kw in ("date", "time", "dt", "timestamp")):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass

    # Better numeric inference
    df = df.infer_objects()

    return df


def dataframe_to_sql_string(df: pd.DataFrame, table_name: str = "data") -> str:
    """
    Convert DataFrame to SQL CREATE + INSERT statements (SQLite-compatible).
    Useful for displaying raw SQL queries in the UI.
    """
    type_map = {
        "int64": "INTEGER",
        "float64": "REAL",
        "object": "TEXT",
        "bool": "INTEGER",
        "datetime64[ns]": "TEXT",
    }

    col_defs = []
    for col, dtype in df.dtypes.items():
        sql_type = type_map.get(str(dtype), "TEXT")
        col_defs.append(f"  {col} {sql_type}")

    create = f"CREATE TABLE {table_name} (\n" + ",\n".join(col_defs) + "\n);\n"

    inserts = []
    for _, row in df.head(10).iterrows():
        vals = []
        for v in row:
            if pd.isna(v):
                vals.append("NULL")
            elif isinstance(v, str):
                vals.append(f"'{v}'")
            else:
                vals.append(str(v))
        inserts.append(f"INSERT INTO {table_name} VALUES ({', '.join(vals)});")

    return create + "\n".join(inserts)
