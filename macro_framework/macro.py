"""FRED macro-state extraction and z-score normalization.

Three series form the default regime panel:
  - CPIAUCSL       → CPI YoY (inflation velocity)
  - T10Y2Y         → 10Y-2Y Treasury spread (growth expectations / yield curve)
  - BAMLH0A0HYM2   → ICE BofA US HY Option-Adjusted Spread (credit stress)
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from .db import get_engine

DEFAULT_SERIES: dict[str, str] = {
    "cpi_yoy": "CPIAUCSL",
    "t10y2y":  "T10Y2Y",
    "hy_oas":  "BAMLH0A0HYM2",
}


def load_fred_series(series_id: str) -> pd.Series:
    """Return a date-indexed Series for a FRED series_id."""
    sql = text("SELECT date, value FROM fred_series WHERE series_id = :s ORDER BY date")
    with get_engine().connect() as conn:
        df = pd.read_sql(sql, conn, params={"s": series_id}, parse_dates=["date"])
    if df.empty:
        raise ValueError(f"no rows in fred_series for series_id={series_id!r}")
    return df.set_index("date")["value"].rename(series_id)


def cpi_yoy(cpi_level: pd.Series) -> pd.Series:
    """Convert a CPI level series (monthly) to year-over-year percent change."""
    return (cpi_level.pct_change(12) * 100.0).rename("cpi_yoy")


def rolling_zscore(s: pd.Series, window: int = 60) -> pd.Series:
    """Rolling z-score over `window` observations. Use for out-of-sample / real-time work."""
    mu = s.rolling(window, min_periods=max(12, window // 2)).mean()
    sd = s.rolling(window, min_periods=max(12, window // 2)).std(ddof=1)
    return (s - mu) / sd


def build_macro_panel(
    series_map: dict[str, str] | None = None,
    freq: str = "ME",
    zscore_window: int = 60,
) -> pd.DataFrame:
    """Return a monthly panel with raw and z-scored macro variables.

    Columns: cpi_yoy, t10y2y, hy_oas (raw) + cpi_yoy_z, t10y2y_z, hy_oas_z.
    `zscore_window` is in *monthly* observations (default 60 = 5y rolling).
    """
    series_map = series_map or DEFAULT_SERIES
    cpi = load_fred_series(series_map["cpi_yoy"])
    t10y2y = load_fred_series(series_map["t10y2y"]).rename("t10y2y")
    hy = load_fred_series(series_map["hy_oas"]).rename("hy_oas")

    inflation = cpi_yoy(cpi)  # monthly already
    t10y2y_m = t10y2y.resample(freq).last()
    hy_m = hy.resample(freq).last()
    inflation_m = inflation.resample(freq).last()

    raw = pd.concat([inflation_m, t10y2y_m, hy_m], axis=1).dropna(how="all")
    z = raw.apply(lambda col: rolling_zscore(col, window=zscore_window))
    z.columns = [f"{c}_z" for c in raw.columns]
    return raw.join(z)
