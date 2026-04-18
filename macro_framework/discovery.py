from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from .db import get_engine


def list_schemas() -> pd.DataFrame:
    sql = text(
        "SELECT nspname AS schema FROM pg_namespace "
        "WHERE nspname NOT LIKE 'pg_%' AND nspname <> 'information_schema' "
        "ORDER BY nspname"
    )
    with get_engine().connect() as conn:
        return pd.read_sql(sql, conn)


def list_tables(schema: str | None = None) -> pd.DataFrame:
    sql = """
        SELECT table_schema, table_name, table_type
        FROM information_schema.tables
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
    """
    params: dict[str, str] = {}
    if schema is not None:
        sql += " AND table_schema = :schema"
        params["schema"] = schema
    sql += " ORDER BY table_schema, table_name"
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def find_etf_tables(keywords: tuple[str, ...] = ("etf", "fund", "price", "security", "instrument", "ticker")) -> pd.DataFrame:
    tables = list_tables()
    pat = "|".join(keywords)
    mask = tables["table_name"].str.contains(pat, case=False, regex=True)
    return tables.loc[mask].reset_index(drop=True)


def table_columns(table: str, schema: str = "public") -> pd.DataFrame:
    sql = text(
        """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
        ORDER BY ordinal_position
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(sql, conn, params={"schema": schema, "table": table})


def table_preview(table: str, schema: str = "public", limit: int = 5) -> pd.DataFrame:
    qualified = f'"{schema}"."{table}"'
    sql = text(f"SELECT * FROM {qualified} LIMIT :n")
    with get_engine().connect() as conn:
        return pd.read_sql(sql, conn, params={"n": limit})
