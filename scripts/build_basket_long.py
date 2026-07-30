"""Long total-return close panel: the 4-ETF basket + the SPY regression benchmark.

The offline cache ``data/etf_prices_wide_2013_2026.parquet`` starts 2013-01-02, so the
max-timeframe frontier has no opportunity set before then. This persists the same
basket ``scripts/build_static_bh_long.py`` trades — SWDA.L / XLK / IAU / BIL — back to
SWDA.L's inception, from the identical source (yfinance auto-adjusted Close), so the
frontier and the buy-and-hold line it is plotted against share one price basis.

Window end is PINNED: yfinance keeps extending, and an unpinned end would silently move
every published figure on the next run.

Reproducible: ``uv run python scripts/build_basket_long.py``. Gitignored, shipped via
the GH data release beside its ``spy_close_2009_2026`` sibling.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

REPO = Path(__file__).resolve().parent.parent
BASKET = ["SWDA.L", "XLK", "IAU", "BIL"]
BENCH = "SPY"
#: SPY rides along because the notebooks' regression benchmark must be total-return too.
#: ``data/etf_prices_wide_2013_2026.parquet`` is PRICE-ONLY: regressing a strategy on an
#: unadjusted SPY understates the benchmark's return by the dividend yield (~1.8%/yr), which
#: lands entirely in the intercept and inflates every "alpha" and appraisal ratio built on it.
#: The release scripts (build_tear_sheet.py, extend_stream_2026.py) already pull
#: auto_adjust=True; this closes the same gap for the offline notebook path.
COLUMNS = BASKET + [BENCH]
START, END = "2009-09-01", "2026-05-30"  # SWDA.L lists 2009-09-25; end pinned
OUT = REPO / "data" / "basket_close_2009_2026.parquet"


# --- Frozen acquisition contract (remediation task 5.1) ---------------------------
# Everything the immutable market snapshot may fetch is pinned HERE, before any
# network access: identity, coverage, universe, total-return semantics, quote
# units, FX direction, and the single ALFRED vintage. Acquisition (5.2) must
# refuse to start on an incomplete or ambiguous contract.

SNAPSHOT_ID = "market_total_return_fx_2026-06-30_v1"
SNAPSHOT_REQUESTED_START = "2009-09-01"  # SWDA.L lists 2009-09-25; matches legacy START
SNAPSHOT_REQUESTED_END = "2026-06-30"  # pinned by the snapshot identity, never open-ended
SNAPSHOT_SYMBOLS = ("SWDA.L", "XLK", "IAU", "BIL", "SPY")
SNAPSHOT_QUOTES = {
    # SWDA.L is quoted on the LSE in pence sterling; GBP major = GBp * 0.01
    "SWDA.L": {"quote_currency": "GBP", "quote_unit": "GBp", "scale_to_major": 0.01},
    "XLK": {"quote_currency": "USD", "quote_unit": "USD", "scale_to_major": 1.0},
    "IAU": {"quote_currency": "USD", "quote_unit": "USD", "scale_to_major": 1.0},
    "BIL": {"quote_currency": "USD", "quote_unit": "USD", "scale_to_major": 1.0},
    "SPY": {"quote_currency": "USD", "quote_unit": "USD", "scale_to_major": 1.0},
}
FX_SERIES_ID = "DEXUSUK"  # Federal Reserve H.10 via FRED/ALFRED
FX_FIELD = "USD_per_GBP"  # quoted USD per GBP: GBP-major level * FX; never inverted
FX_CONVERSION = "multiply"
# "Vintage" is ALFRED's term for an AS-OF PUBLICATION DATE: the series exactly as
# it stood on that day. H.10 FX fixings are historical observations that are
# rarely revised — pinning the as-of date buys byte-reproducible retrieval (the
# same request returns the same bytes forever), not different data.
TOTAL_RETURN_FIELD = (
    "yfinance auto_adjust=True Close (adjusted total-return level, dividends reinvested)"
)

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def _iso_date(name: str, value: object) -> date:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ISO date, got {value!r}") from exc


@dataclass(frozen=True)
class AcquisitionContract:
    """The frozen snapshot acquisition configuration (task 5.1).

    Construction refuses every incomplete or ambiguous source, unit, coverage,
    or vintage configuration; ``require_ready_for_acquisition`` additionally
    refuses to start network acquisition until the raw-response hashes are
    declared (filled in by the one-time capture, task 12.1, via
    ``dataclasses.replace``)."""

    snapshot_id: str
    requested_start: str  # ISO date, inclusive
    requested_end: str  # ISO date, inclusive; pinned, never open-ended
    symbols: tuple[str, ...]
    quotes: dict[str, dict[str, object]]  # exactly one entry per symbol
    cash_symbol: str
    benchmark_symbol: str
    total_return_field: str
    etf_source_id: str
    fx_source_id: str
    fx_series_id: str
    fx_field: str  # must be exactly "USD_per_GBP"
    fx_conversion: str  # must be exactly "multiply"
    vintage_date: str  # as-of publication date (ALFRED "vintage"): the series as it stood this day
    realtime_start: str  # must equal vintage_date (ALFRED realtime window collapses to one day)
    realtime_end: str  # must equal vintage_date
    etf_raw_response_sha256: str | None = None
    fx_raw_response_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbols", tuple(self.symbols))  # freeze for real
        for name in (
            "snapshot_id",
            "cash_symbol",
            "benchmark_symbol",
            "total_return_field",
            "etf_source_id",
            "fx_source_id",
            "fx_series_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.total_return_field != TOTAL_RETURN_FIELD:
            # provenance string may never drift from what the derived request does
            raise ValueError(
                f"total_return_field must be exactly {TOTAL_RETURN_FIELD!r}"
            )
        start = _iso_date("requested_start", self.requested_start)
        end = _iso_date("requested_end", self.requested_end)
        if start >= end:
            raise ValueError(
                f"requested_start {self.requested_start} must be strictly before "
                f"requested_end {self.requested_end}"
            )
        if not self.symbols:
            raise ValueError("symbols must not be empty")
        for symbol in self.symbols:
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError(f"symbols contains a blank symbol: {symbol!r}")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError(f"symbols contains duplicates: {self.symbols!r}")
        if not isinstance(self.quotes, dict):
            raise ValueError("quotes must be a dict of per-symbol quote metadata")
        if set(self.quotes) != set(self.symbols):
            raise ValueError(
                f"quotes must cover exactly the symbol universe; "
                f"missing {sorted(set(self.symbols) - set(self.quotes))}, "
                f"extra {sorted(set(self.quotes) - set(self.symbols))}"
            )
        for symbol, quote in self.quotes.items():
            if set(quote) != {"quote_currency", "quote_unit", "scale_to_major"}:
                raise ValueError(
                    f"quote for {symbol} must define exactly "
                    f"quote_currency/quote_unit/scale_to_major, got {sorted(quote)}"
                )
            currency, unit, scale = (
                quote["quote_currency"],
                quote["quote_unit"],
                quote["scale_to_major"],
            )
            if currency not in ("USD", "GBP"):
                raise ValueError(f"{symbol}: quote_currency must be USD or GBP, got {currency!r}")
            if unit not in ("USD", "GBp"):
                raise ValueError(f"{symbol}: quote_unit must be USD or GBp, got {unit!r}")
            if unit == "GBp" and currency != "GBP":
                raise ValueError(f"{symbol}: quote_unit GBp requires quote_currency GBP")
            if unit == "USD" and currency != "USD":
                raise ValueError(f"{symbol}: quote_unit USD requires quote_currency USD")
            if type(scale) is not float:  # rejects bool/int/Decimal masquerading as 1.0
                raise ValueError(f"{symbol}: scale_to_major must be a float, got {scale!r}")
            if unit == "GBp" and scale != 0.01:
                raise ValueError(f"{symbol}: GBp scales to GBP by 0.01, got {scale!r}")
            if unit == "USD" and scale != 1.0:
                raise ValueError(f"{symbol}: USD quote_unit requires scale_to_major 1.0, got {scale!r}")
        for name in ("cash_symbol", "benchmark_symbol"):
            if getattr(self, name) not in self.symbols:
                raise ValueError(f"{name} {getattr(self, name)!r} is not in the symbol universe")
        if self.fx_field != "USD_per_GBP":
            # the anti-inversion pin: GBP majors are MULTIPLIED by USD_per_GBP
            raise ValueError(f"fx_field must be exactly 'USD_per_GBP', got {self.fx_field!r}")
        if self.fx_conversion != "multiply":
            raise ValueError(f"fx_conversion must be 'multiply', got {self.fx_conversion!r}")
        vintage = _iso_date("vintage_date", self.vintage_date)
        if vintage < end:
            raise ValueError(
                f"vintage_date {self.vintage_date} must not precede requested_end "
                f"{self.requested_end}: data published before the coverage end cannot "
                "contain the full requested history"
            )
        for name in ("realtime_start", "realtime_end"):
            if _iso_date(name, getattr(self, name)) != vintage:
                raise ValueError(
                    f"{name} {getattr(self, name)} must equal vintage_date "
                    f"{self.vintage_date}: FX retrieval is pinned to ONE ALFRED vintage "
                    "(one as-of publication date, for byte-reproducible refetches)"
                )

    def etf_request_params(self) -> dict[str, object]:
        """Derived, never stored — the pinned fields ARE the request."""
        end_exclusive = (date.fromisoformat(self.requested_end) + timedelta(days=1)).isoformat()
        return {
            "tickers": list(self.symbols),
            "start": self.requested_start,
            "end": end_exclusive,  # yfinance end is exclusive; requested_end is inclusive
            "auto_adjust": True,  # adjusted total-return levels, dividends reinvested
            "progress": False,
            "threads": False,
        }

    def fx_request_params(self) -> dict[str, object]:
        return {
            "series_id": self.fx_series_id,
            "realtime_start": self.realtime_start,
            "realtime_end": self.realtime_end,
            "observation_start": self.requested_start,
            "observation_end": self.requested_end,
        }

    def require_ready_for_acquisition(self) -> None:
        """Acquisition-start gate: raw-response hash declarations are mandatory."""
        for name in ("etf_raw_response_sha256", "fx_raw_response_sha256"):
            value = getattr(self, name)
            if value is None or not isinstance(value, str) or not value.strip():
                raise ValueError(f"acquisition cannot start: {name} is required")
            if not _SHA256_HEX.fullmatch(value):
                raise ValueError(f"{name} must be 64 lowercase hex chars, got {value!r}")

    def fingerprint(self) -> str:
        payload = json.dumps(
            dataclasses.asdict(self), sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_snapshot_contract(
    *,
    vintage_date: str,
    etf_raw_response_sha256: str | None = None,
    fx_raw_response_sha256: str | None = None,
) -> AcquisitionContract:
    """The pinned contract; only the ALFRED vintage (owned by task 12.1) varies."""
    return AcquisitionContract(
        snapshot_id=SNAPSHOT_ID,
        requested_start=SNAPSHOT_REQUESTED_START,
        requested_end=SNAPSHOT_REQUESTED_END,
        symbols=SNAPSHOT_SYMBOLS,
        quotes={symbol: dict(quote) for symbol, quote in SNAPSHOT_QUOTES.items()},
        cash_symbol="BIL",
        benchmark_symbol="SPY",
        total_return_field=TOTAL_RETURN_FIELD,
        etf_source_id="yfinance",
        fx_source_id="fred_alfred_h10",
        fx_series_id=FX_SERIES_ID,
        fx_field=FX_FIELD,
        fx_conversion=FX_CONVERSION,
        vintage_date=vintage_date,
        realtime_start=vintage_date,
        realtime_end=vintage_date,
        etf_raw_response_sha256=etf_raw_response_sha256,
        fx_raw_response_sha256=fx_raw_response_sha256,
    )


def main() -> None:
    raw = yf.download(COLUMNS, start=START, end=END, auto_adjust=True, progress=False)["Close"]
    first = {s: raw[s].first_valid_index() for s in COLUMNS}
    # Do NOT dropna across all columns. SWDA.L is LSE-listed and does not trade on UK bank
    # holidays, so a global dropna would put SPY on the LSE∩NYSE intersection and silently
    # delete 61 US sessions (Easter Monday, early May, August BH, Boxing Day) from any
    # regression using it. Keep the union from the last inception; consumers restrict:
    #   benchmark -> px["SPY"].dropna()            (full NYSE calendar)
    #   frontier  -> px[BASKET].dropna(how="any")  (rectangular, as the optimizer needs)
    px = raw[COLUMNS].loc[max(first.values()):]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    px.index.name = "Date"

    binding = max(first, key=lambda s: first[s])
    assert px.index.min() == max(first.values()), "panel must start at the last inception"
    assert px[BASKET].dropna(how="any").shape[0] < px["SPY"].dropna().shape[0], \
        "SPY must keep more sessions than the LSE-constrained basket"

    px.to_parquet(OUT)
    print(f"{OUT.name}: {px.index.min():%Y-%m-%d} -> {px.index.max():%Y-%m-%d} "
          f"({(px.index.max() - px.index.min()).days / 365.25:.2f}y, {len(px)} rows)")
    print(f"binding constraint: {binding} inception {first[binding]:%Y-%m-%d}")
    for s in COLUMNS:
        print(f"  {s:8s} first {first[s]:%Y-%m-%d}")


if __name__ == "__main__":
    main()


# --- Acquisition and normalization (task 5.2) --------------------------------------
# Pure parse/normalize functions with injectable source frames; the live fetchers
# are thin defaults consumed only by the one-time credentialed capture (task 12.1).

BASKET_LOCAL_SYMBOLS = ("SWDA.L", "XLK", "IAU")  # -> basket_adjusted_close_local
CASH_MARKET_SYMBOLS = ("BIL", "SPY")  # -> cash_market_total_return


@dataclass(frozen=True, eq=False)
class NormalizedSnapshotData:
    """The three normalized snapshot tables plus coverage disclosure for 5.3."""

    basket_local: pd.DataFrame  # SWDA.L in local GBp + XLK/IAU USD, union calendar
    cash_market: pd.DataFrame  # BIL + SPY adjusted total-return levels
    fx: pd.DataFrame  # single column USD_per_GBP on SOURCE observation dates
    coverage: dict[str, dict[str, object]]  # per-symbol actual coverage + fx parse counts


def _validate_snapshot_index(index: pd.Index, name: str) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"{name} must be a pandas DatetimeIndex")
    if index.empty:
        raise ValueError(f"{name} must not be empty")
    if index.tz is not None:
        raise ValueError(f"{name} must be timezone-naive")
    if index.hasnans:
        raise ValueError(f"{name} must not contain NaT labels")
    if not index.is_unique:
        raise ValueError(f"{name} must contain unique labels")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{name} must be strictly increasing")
    if bool((index != index.normalize()).any()):
        # intraday labels would let two observations share one declared DATE
        raise ValueError(f"{name} must contain date-granular (midnight) labels only")
    return index


def _require_within_bounds(index: pd.DatetimeIndex, contract: AcquisitionContract, name: str) -> None:
    start = pd.Timestamp(contract.requested_start)
    end = pd.Timestamp(contract.requested_end)
    if index.min() < start or index.max() > end:
        raise ValueError(
            f"{name} observations {index.min().date()}..{index.max().date()} fall outside "
            f"the declared bounds {contract.requested_start}..{contract.requested_end}"
        )


def _coverage_entry(observed: pd.Series) -> dict[str, object]:
    return {
        "first": observed.index[0].date().isoformat(),
        "last": observed.index[-1].date().isoformat(),
        "rows": int(len(observed)),
    }


def parse_fredgraph_csv(
    text: str, series_id: str, *, column: str | None = None
) -> tuple[pd.Series, dict[str, int]]:
    """Parse a fredgraph/alfredgraph CSV payload.

    ONLY the literal '.' sentinel — a date on which H.10 published NOTHING —
    may be dropped, and always WITH disclosure counts. Any other non-numeric
    token is a defective source observation and is rejected, never repaired
    (tasks.md 5.2). ``column`` reads a vintage-suffixed ALFRED column
    (e.g. ``DEXUSUK_20260703``) while the returned series keeps ``series_id``.
    """
    column = column or series_id
    frame = pd.read_csv(io.StringIO(text))
    if column not in frame.columns:
        raise ValueError(f"FRED payload does not contain series {column!r}")
    date_col = frame.columns[0]
    raw = frame[column]
    numeric = pd.to_numeric(raw, errors="coerce")
    bad = numeric.isna()
    if bool(bad.any()):
        tokens = raw[bad].astype(str).str.strip()
        garbage = tokens[tokens != "."]
        if len(garbage):
            raise ValueError(
                f"non-numeric FRED value {garbage.iloc[0]!r} at "
                f"{frame[date_col][garbage.index[0]]}: only the '.' unpublished "
                "sentinel may be dropped"
            )
    series = pd.Series(
        numeric.values,
        index=pd.DatetimeIndex(pd.to_datetime(frame[date_col])),
        name=series_id,
    )
    kept = series.dropna()
    if kept.empty:
        raise ValueError(f"empty FRED series {series_id!r}")
    return kept, {
        "raw_rows": int(len(series)),
        "non_numeric_dropped": int(len(series) - len(kept)),
    }


def fetch_etf_close(contract: AcquisitionContract) -> pd.DataFrame:
    """Live yfinance fetch from the DERIVED request params (12.1 capture path)."""
    params = contract.etf_request_params()
    raw = yf.download(
        params["tickers"],
        start=params["start"],
        end=params["end"],
        auto_adjust=params["auto_adjust"],
        progress=params["progress"],
        threads=params["threads"],
    )
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw["Close"]
    return raw


def fetch_fx_observations(contract: AcquisitionContract) -> tuple[pd.Series, dict[str, int]]:
    """Live ALFRED fetch of the series AS PUBLISHED ON the pinned as-of date
    (12.1 capture path).

    fredgraph.csv silently IGNORES ``vintage_date`` (verified against the live
    service), so this uses alfredgraph.csv, which honors it and returns the
    series under an as-of-suffixed column (``DEXUSUK_YYYYMMDD``) — the suffix
    doubles as capture-time proof that the requested publication date was
    actually applied rather than today's copy of the history."""
    params = contract.fx_request_params()
    vintage = str(params["realtime_start"])
    url = (
        "https://alfred.stlouisfed.org/graph/alfredgraph.csv"
        f"?id={params['series_id']}&cosd={params['observation_start']}"
        f"&coed={params['observation_end']}&vintage_date={vintage}"
    )
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    expected_column = f"{params['series_id']}_{vintage.replace('-', '')}"
    try:
        series, disclosure = parse_fredgraph_csv(
            resp.text, str(params["series_id"]), column=expected_column
        )
    except ValueError as exc:
        if "does not contain series" in str(exc):
            raise ValueError(
                f"ALFRED did not honor vintage {vintage}: expected column "
                f"{expected_column!r} is absent from the response"
            ) from exc
        raise
    # the transport may ignore cosd/coed for some vintages; selecting the PINNED
    # observation window here is executing the request, not repairing data
    within = series.loc[str(params["observation_start"]) : str(params["observation_end"])]
    disclosure["outside_requested_window_dropped"] = int(len(series) - len(within))
    if within.empty:
        raise ValueError(
            f"ALFRED vintage {vintage} returned no observations inside "
            f"{params['observation_start']}..{params['observation_end']}"
        )
    return within, disclosure


def normalize_etf_levels(
    contract: AcquisitionContract, raw_close: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, object]]]:
    """Normalize the union-calendar total-return panel WITHOUT a global
    complete-case filter: a NaN cell where a symbol simply had no session
    (LSE vs NYSE holidays) is legitimate absence, while every value a symbol
    DOES report must be a finite positive level. SWDA.L stays in local GBp."""
    if not isinstance(raw_close, pd.DataFrame):
        raise ValueError("raw_close must be a pandas DataFrame")
    if isinstance(raw_close.columns, pd.MultiIndex):
        raise ValueError("raw_close must have single-level symbol columns, not a MultiIndex")
    if not raw_close.columns.is_unique:
        raise ValueError("raw_close contains duplicate column labels")
    missing = [s for s in contract.symbols if s not in raw_close.columns]
    if missing:
        raise ValueError(f"raw_close is missing symbol column(s) {missing}")
    frame = raw_close[list(contract.symbols)].copy()
    index = _validate_snapshot_index(frame.index, "raw_close.index")
    _require_within_bounds(index, contract, "ETF")

    all_nan = frame.isna().all(axis=1)
    frame = frame.loc[~all_nan]  # a row NO symbol observed is not a session at all
    if frame.empty:
        raise ValueError("raw_close contains no observations")

    coverage: dict[str, dict[str, object]] = {}
    for symbol in contract.symbols:
        column = frame[symbol]
        if (
            not pd.api.types.is_numeric_dtype(column.dtype)
            or pd.api.types.is_bool_dtype(column.dtype)
            or pd.api.types.is_complex_dtype(column.dtype)
        ):
            raise ValueError(f"{symbol}: levels must be real numeric values")
        observed = column.dropna()
        if observed.empty:
            raise ValueError(f"{symbol}: no observations within the declared bounds")
        values = observed.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            label = observed.index[int(np.flatnonzero(~np.isfinite(values))[0])]
            raise ValueError(f"{symbol}: non-finite level at {label.date()}")
        if (values <= 0).any():
            label = observed.index[int(np.flatnonzero(values <= 0)[0])]
            raise ValueError(f"{symbol}: non-positive level at {label.date()}")
        coverage[symbol] = _coverage_entry(observed)

    frame.index = pd.DatetimeIndex(frame.index)
    frame.index.name = "Date"
    basket_local = frame[list(BASKET_LOCAL_SYMBOLS)].copy()
    cash_market = frame[list(CASH_MARKET_SYMBOLS)].copy()
    return basket_local, cash_market, coverage


def normalize_fx_observations(
    contract: AcquisitionContract,
    observations: pd.Series,
    parse_disclosure: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Normalize DEXUSUK onto its OWN source observation dates as USD_per_GBP.

    No reindexing onto the ETF calendar, no filling: consumers resolve gaps
    with the backward-as-of staleness rule at the Markowitz boundary."""
    if not isinstance(observations, pd.Series):
        raise ValueError("fx observations must be a pandas Series")
    index = _validate_snapshot_index(observations.index, "fx.index")
    _require_within_bounds(index, contract, "FX")
    if (
        not pd.api.types.is_numeric_dtype(observations.dtype)
        or pd.api.types.is_bool_dtype(observations.dtype)
        or pd.api.types.is_complex_dtype(observations.dtype)
    ):
        raise ValueError("fx observations must be real numeric values")
    values = observations.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        label = index[int(np.flatnonzero(~np.isfinite(values))[0])]
        raise ValueError(f"fx observation at {label.date()} is not finite")
    if (values <= 0).any():
        label = index[int(np.flatnonzero(values <= 0)[0])]
        raise ValueError(f"fx observation at {label.date()} is not positive")

    fx = observations.rename(contract.fx_field).to_frame()
    fx.index = pd.DatetimeIndex(fx.index)
    fx.index.name = "Date"
    disclosure: dict[str, object] = dict(parse_disclosure or {})
    disclosure.update(_coverage_entry(observations))
    return fx, disclosure


def acquire_snapshot_data(
    contract: AcquisitionContract,
    *,
    etf_close: pd.DataFrame | None = None,
    fx_observations: pd.Series | None = None,
    fx_parse_disclosure: dict[str, int] | None = None,
) -> NormalizedSnapshotData:
    """Acquire (or accept injected) source data and normalize it to the pinned
    contract. Refuses to start until the raw-response hashes are declared."""
    contract.require_ready_for_acquisition()
    if etf_close is None:
        etf_close = fetch_etf_close(contract)
    if fx_observations is None:
        fx_observations, fx_parse_disclosure = fetch_fx_observations(contract)
    elif fx_parse_disclosure is not None:
        retained = (
            int(fx_parse_disclosure.get("raw_rows", 0))
            - int(fx_parse_disclosure.get("non_numeric_dropped", 0))
            - int(fx_parse_disclosure.get("outside_requested_window_dropped", 0))
        )
        if retained != len(fx_observations):
            raise ValueError(
                f"fx_parse_disclosure claims {retained} retained observations but "
                f"{len(fx_observations)} were supplied"
            )
    basket_local, cash_market, coverage = normalize_etf_levels(contract, etf_close)
    fx, fx_disclosure = normalize_fx_observations(
        contract, fx_observations, fx_parse_disclosure
    )
    coverage[contract.fx_field] = fx_disclosure
    return NormalizedSnapshotData(
        basket_local=basket_local, cash_market=cash_market, fx=fx, coverage=coverage
    )


# --- Append-only snapshot persistence and validation (task 5.3) --------------------

SNAPSHOT_MANIFEST_SCHEMA = "market_snapshot.v1"
_SNAPSHOT_FILES = (
    "basket_adjusted_close_local.parquet",
    "cash_market_total_return.parquet",
    "fx_usd_per_gbp.parquet",
)


def _file_inventory(path: Path, frame: pd.DataFrame) -> dict[str, object]:
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": int(path.stat().st_size),
        "rows": int(len(frame)),
        "start": frame.index.min().date().isoformat(),
        "end": frame.index.max().date().isoformat(),
        "schema_id": f"{SNAPSHOT_MANIFEST_SCHEMA}/{path.name.removesuffix('.parquet')}",
    }


def _overlap_revisions(current: pd.DataFrame, preceding: pd.DataFrame) -> dict[str, int]:
    """Quantify vendor revisions on the shared index/columns of two snapshots."""
    shared_index = current.index.intersection(preceding.index)
    shared_columns = current.columns.intersection(preceding.columns)
    if shared_index.empty or shared_columns.empty:
        return {"overlap_rows": 0, "changed_cells": 0}
    left = current.loc[shared_index, shared_columns]
    right = preceding.loc[shared_index, shared_columns]
    same = (left == right) | (left.isna() & right.isna())
    return {
        "overlap_rows": int(len(shared_index)),
        "changed_cells": int((~same).to_numpy().sum()),
    }


def _find_preceding_snapshot(output_root: Path, snapshot_id: str) -> Path | None:
    """Latest COMPLETED compatible snapshot other than the one being built."""
    candidates = []
    if not output_root.is_dir():
        return None
    for entry in sorted(output_root.iterdir()):
        if entry.name == snapshot_id or not entry.is_dir():
            continue
        manifest_path = entry / "manifest.json"
        if not (entry / "COMPLETED").is_file() or not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("schema") == SNAPSHOT_MANIFEST_SCHEMA:
            try:  # compare PARSED timestamps: raw strings mis-sort across offsets
                built = pd.Timestamp(str(manifest.get("build_time", "")))
            except (TypeError, ValueError):
                continue
            if built.tz is None:
                continue
            candidates.append((built, entry))
    if not candidates:
        return None
    return max(candidates)[1]


def build_market_snapshot(
    *,
    snapshot_id: str,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
    output_root: Path,
    contract: AcquisitionContract | None = None,
    data: NormalizedSnapshotData | None = None,
    build_time: str | None = None,
) -> Path:
    """Write one append-only market snapshot; COMPLETED is written LAST.

    ``contract``/``data`` are injectable for offline builds from persisted raw
    sources (tasks 5.4/12.1); the live path acquires via the pinned contract.
    A failed build leaves the staging directory dirty WITHOUT ``COMPLETED``;
    recovery is delete-and-rebuild, never in-place repair.
    """
    if contract is None:
        raise ValueError("build_market_snapshot requires the frozen AcquisitionContract")
    if snapshot_id != contract.snapshot_id:
        raise ValueError(
            f"snapshot_id {snapshot_id!r} does not match the contract's "
            f"{contract.snapshot_id!r}"
        )
    if (
        pd.Timestamp(requested_start) != pd.Timestamp(contract.requested_start)
        or pd.Timestamp(requested_end) != pd.Timestamp(contract.requested_end)
    ):
        raise ValueError("requested coverage does not match the frozen contract")

    snapshot_dir = Path(output_root) / snapshot_id
    if snapshot_dir.exists():
        if (snapshot_dir / "COMPLETED").exists():
            raise ValueError(
                f"snapshot identity {snapshot_id!r} is COMPLETED and immutable; "
                "snapshots are append-only"
            )
        if any(snapshot_dir.iterdir()):
            raise ValueError(
                f"refusing to write into non-empty staging directory {snapshot_dir}"
            )
    if build_time is None:
        build_time = pd.Timestamp.now("UTC").isoformat()
    else:
        try:
            parsed = pd.Timestamp(build_time)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"build_time must be an ISO-8601 timestamp, got {build_time!r}") from exc
        if parsed.tz is None:
            raise ValueError(f"build_time must be timezone-aware, got {build_time!r}")
        build_time = parsed.tz_convert("UTC").isoformat()
    if data is None:
        data = acquire_snapshot_data(contract)

    preceding_dir = _find_preceding_snapshot(Path(output_root), snapshot_id)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "basket_adjusted_close_local.parquet": data.basket_local,
        "cash_market_total_return.parquet": data.cash_market,
        "fx_usd_per_gbp.parquet": data.fx,
    }
    files: dict[str, dict[str, object]] = {}
    for name, frame in tables.items():
        path = snapshot_dir / name
        frame.to_parquet(path)
        files[name] = _file_inventory(path, frame)

    overlap: dict[str, object] = {"preceding_snapshot": None}
    if preceding_dir is not None:
        overlap["preceding_snapshot"] = preceding_dir.name
        for name, frame in tables.items():
            prior_path = preceding_dir / name
            if prior_path.is_file():
                overlap[name] = _overlap_revisions(frame, pd.read_parquet(prior_path))

    manifest = {
        "schema": SNAPSHOT_MANIFEST_SCHEMA,
        "snapshot_id": snapshot_id,
        "build_time": build_time,
        "cash_symbol": contract.cash_symbol,
        "benchmark_symbol": contract.benchmark_symbol,
        "requested_coverage": {
            "start": contract.requested_start,
            "end": contract.requested_end,
        },
        "actual_coverage": data.coverage,
        "source_identifiers": {
            "etf": contract.etf_source_id,
            "fx": contract.fx_source_id,
            "fx_series": contract.fx_series_id,
        },
        "quotes": contract.quotes,
        "total_return_field": contract.total_return_field,
        "fx_field": contract.fx_field,
        "fx_conversion": contract.fx_conversion,
        "fx_vintage_date": contract.vintage_date,
        "contract_fingerprint": contract.fingerprint(),
        "raw_response_sha256": {
            "etf": contract.etf_raw_response_sha256,
            "fx": contract.fx_raw_response_sha256,
        },
        "files": files,
        "overlap_revisions": overlap,
        "completed": True,
    }
    (snapshot_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    # every validation must pass BEFORE the completion marker exists
    validate_market_snapshot(snapshot_dir, require_completed=False)
    manifest_sha = hashlib.sha256((snapshot_dir / "manifest.json").read_bytes()).hexdigest()
    (snapshot_dir / "COMPLETED").write_text(
        f"{build_time}\nmanifest_sha256={manifest_sha}\n"
    )
    return snapshot_dir


def validate_market_snapshot(
    snapshot_dir: Path, *, require_completed: bool = True
) -> dict[str, object]:
    """Validate one snapshot directory byte-for-byte against its manifest."""
    snapshot_dir = Path(snapshot_dir)
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{snapshot_dir}: manifest.json is missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SNAPSHOT_MANIFEST_SCHEMA:
        raise ValueError(f"{snapshot_dir}: unknown manifest schema {manifest.get('schema')!r}")
    if require_completed and not (snapshot_dir / "COMPLETED").is_file():
        raise ValueError(f"{snapshot_dir}: COMPLETED marker is absent; snapshot is incomplete")
    if "overlap_revisions" not in manifest:
        raise ValueError(f"{snapshot_dir}: overlap revision disclosure is absent")

    start = pd.Timestamp(manifest["requested_coverage"]["start"])
    end = pd.Timestamp(manifest["requested_coverage"]["end"])
    report: dict[str, object] = {
        "snapshot_id": manifest.get("snapshot_id"),
        "schema": manifest["schema"],
        "files": {},
    }
    for name in _SNAPSHOT_FILES:
        path = snapshot_dir / name
        recorded = manifest.get("files", {}).get(name)
        if recorded is None or not path.is_file():
            raise ValueError(f"{snapshot_dir}: {name} is missing from disk or manifest")
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != recorded["sha256"]:
            raise ValueError(
                f"{snapshot_dir}: {name} bytes were mutated after inventory "
                f"(sha256 {actual_sha[:12]}... != recorded {recorded['sha256'][:12]}...)"
            )
        frame = pd.read_parquet(path)
        if len(frame) != recorded["rows"]:
            raise ValueError(f"{snapshot_dir}: {name} row count changed")
        _validate_snapshot_index(frame.index, f"{name} index")
        if frame.index.min() < start or frame.index.max() > end:
            raise ValueError(f"{snapshot_dir}: {name} observations fall outside requested coverage")
        for column in frame.columns:
            observed = frame[column].dropna()
            values = observed.to_numpy(dtype=float)
            if len(values) and not np.isfinite(values).all():
                raise ValueError(f"{snapshot_dir}: {name} column {column!r} has a non-finite value")
        report["files"][name] = {"rows": int(len(frame)), "sha256": actual_sha}
    report["completed"] = (snapshot_dir / "COMPLETED").is_file()
    report["overlap_revisions"] = manifest["overlap_revisions"]
    return report
