from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sqlalchemy import text

from .db import get_engine


def load_universe() -> pd.DataFrame:
    sql = text("SELECT symbol, name, isin, category, notes FROM etf_meta ORDER BY category, symbol")
    with get_engine().connect() as conn:
        return pd.read_sql(sql, conn)


def coverage_summary() -> pd.DataFrame:
    """Per-ETF price coverage: first/last date, row count, trading-year span."""
    sql = text(
        """
        SELECT p.symbol,
               m.name,
               m.category,
               MIN(p.date)                                  AS first_date,
               MAX(p.date)                                  AS last_date,
               COUNT(*)                                     AS n_rows,
               ROUND((MAX(p.date) - MIN(p.date))::numeric / 365.25, 2) AS years
        FROM etf_prices p
        LEFT JOIN etf_meta m USING (symbol)
        GROUP BY p.symbol, m.name, m.category
        ORDER BY first_date, p.symbol
        """
    )
    with get_engine().connect() as conn:
        return pd.read_sql(sql, conn)


def get_prices(
    symbols: str | Iterable[str],
    start: str | None = None,
    end: str | None = None,
    field: str = "close",
) -> pd.DataFrame:
    """Wide-format price frame: index=date, columns=symbol."""
    if field not in {"open", "high", "low", "close", "volume"}:
        raise ValueError(f"invalid field: {field}")
    syms = [symbols] if isinstance(symbols, str) else list(symbols)
    if not syms:
        raise ValueError("at least one symbol required")

    sql = f"SELECT symbol, date, {field} FROM etf_prices WHERE symbol = ANY(:syms)"
    params: dict[str, object] = {"syms": syms}
    if start:
        sql += " AND date >= :start"
        params["start"] = start
    if end:
        sql += " AND date <= :end"
        params["end"] = end
    sql += " ORDER BY date, symbol"

    with get_engine().connect() as conn:
        long = pd.read_sql(text(sql), conn, params=params, parse_dates=["date"])
    return long.pivot(index="date", columns="symbol", values=field).sort_index()


def universe_with_history(min_years: float = 5.0) -> pd.DataFrame:
    """ETFs whose price history spans at least `min_years` years — candidates for long-horizon sims."""
    cov = coverage_summary()
    return cov.loc[cov["years"] >= min_years].reset_index(drop=True)
