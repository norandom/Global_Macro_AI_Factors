"""Report-row schema identities and the pre-emission provenance gate.

One schema identity per metric family (R1.2, R1.6, R7.3): reader-facing
elapsed-time/252 rows, explicit vectorbt/365 legacy rows, differential rows,
attribution records, crisis records, and monthly-return tables. Row VALUES are
produced by the later reporting tasks (4.2-4.5); this module only decides
whether a row may be emitted at all: an unknown or mixed schema identity,
missing measurement provenance, or a CAPM/Jensen label on the current
mixed-local-currency histories fails before a row exists (R3.9, R7.6).

Each schema carries a CLOSED field vocabulary derived from the shared finance
contracts (``metric_block`` keys, ``SSRInference``, ``MarketAttribution``,
``CrisisMetrics``), so a row claiming one family cannot smuggle in a field
owned by another. Later reporting tasks extend the vocabularies here as they
build rows; this module stays the single authority.
"""

from __future__ import annotations

import calendar
import dataclasses
import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from macro_framework.evaluation import CALENDAR_DAYS, CrisisMetrics, cagr, max_drawdown
from macro_framework.skill_metric import (
    MarketAttribution,
    differential_returns,
    portfolio_excess_returns,
)
from macro_framework.ssr import TRADING_DAYS, SSRInference, SSRResult

RowKind = Literal["full", "performance_only"]
CurrencyBasis = Literal["USD", "legacy_mixed_local_quotes"]
_CURRENCY_BASES = ("USD", "legacy_mixed_local_quotes")

# The three identities pinned verbatim by the approved design; the remaining
# three are required to be distinct/versioned and are made canonical here.
READER_SCHEMA = "portfolio_metrics.reader.v2"
LEGACY_SCHEMA = "portfolio_metrics.vectorbt365.v1"
DIFFERENTIAL_SCHEMA = "portfolio_metrics.differential.v2"
ATTRIBUTION_SCHEMA = "attribution.raw_market_model.v1"
CRISIS_SCHEMA = "crisis_metrics.boundary_anchored.v1"
MONTHLY_SCHEMA = "monthly_returns.reader.v1"

#: Measurement provenance required on every emitted row (task 4.1): portfolio
#: identity, return basis, window, count, annualization, cash benchmark,
#: currency basis, and source lineage.
REQUIRED_PROVENANCE = (
    "schema",
    "portfolio_id",
    "return_basis",
    "window_label",
    "start",
    "end",
    "n_obs",
    "periods_per_year",
    "cash_benchmark_id",
    "currency_basis",
    "source",
)

# Field vocabularies, described from the existing shared authorities rather than
# re-invented: ``evaluation.metric_block`` names the two annualization families;
# SSR/attribution/crisis fields project the frozen result dataclasses.
_BASIS_FREE_FIELDS = frozenset({"total_return", "maxdd", "downside_rms"})
_READER_METRIC_FIELDS = _BASIS_FREE_FIELDS | {"cagr", "ann_vol", "sharpe", "sortino", "calmar"}
_LEGACY_METRIC_FIELDS = _BASIS_FREE_FIELDS | {
    "cagr_rows",
    "ann_vol_cal",
    "sharpe_cal",
    "sortino_cal",
    "calmar_rows",
}
_SSR_FIELDS = frozenset(
    "ssr_" + f.name for f in dataclasses.fields(SSRResult)
) | frozenset(
    "ssr_" + f.name for f in dataclasses.fields(SSRInference) if f.name != "result"
)
_ATTRIBUTION_FIELDS = frozenset(
    "raw_market_model_" + f.name for f in dataclasses.fields(MarketAttribution)
)
_CRISIS_FIELDS = frozenset(f.name for f in dataclasses.fields(CrisisMetrics))
_MONTHLY_FIELDS = frozenset({"year", "month", "monthly_return"})
_PROVENANCE_FIELDS = frozenset(REQUIRED_PROVENANCE)


@dataclass(frozen=True)
class ReportSchema:
    schema_id: str
    family: Literal["reader", "legacy", "differential", "attribution", "crisis", "monthly"]
    #: annualization pinned by the schema identity itself; None = carried per row
    periods_per_year: int | None
    #: closed vocabulary — a row may carry these keys and nothing else
    fields: frozenset[str]


REPORT_SCHEMAS: Mapping[str, ReportSchema] = {
    s.schema_id: s
    for s in (
        ReportSchema(
            READER_SCHEMA,
            "reader",
            252,
            _PROVENANCE_FIELDS | {"row_kind"} | _READER_METRIC_FIELDS
            | _SSR_FIELDS | _ATTRIBUTION_FIELDS,
        ),
        ReportSchema(LEGACY_SCHEMA, "legacy", 365, _PROVENANCE_FIELDS | _LEGACY_METRIC_FIELDS),
        ReportSchema(
            DIFFERENTIAL_SCHEMA,
            "differential",
            252,
            _PROVENANCE_FIELDS | _READER_METRIC_FIELDS | _SSR_FIELDS
            | {"endpoint_total_return_difference"},
        ),
        ReportSchema(
            ATTRIBUTION_SCHEMA, "attribution", None, _PROVENANCE_FIELDS | _ATTRIBUTION_FIELDS
        ),
        ReportSchema(CRISIS_SCHEMA, "crisis", None, _PROVENANCE_FIELDS | _CRISIS_FIELDS),
        ReportSchema(MONTHLY_SCHEMA, "monthly", 12, _PROVENANCE_FIELDS | _MONTHLY_FIELDS),
    )
}


@dataclass(frozen=True)
class LineMetadata:
    """Identity of one reported portfolio line (design: Reporting Contracts)."""

    portfolio_id: str
    label: str
    window_label: str
    currency_basis: CurrencyBasis
    total_return_basis: str
    cash_benchmark_id: str

    def __post_init__(self) -> None:
        for name in (
            "portfolio_id",
            "label",
            "window_label",
            "total_return_basis",
            "cash_benchmark_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.currency_basis not in _CURRENCY_BASES:
            raise ValueError(
                f"currency_basis must be one of {_CURRENCY_BASES}, got {self.currency_basis!r}"
            )


def _timestamp(row: Mapping[str, object], key: str) -> pd.Timestamp:
    try:
        value = pd.Timestamp(row[key])  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a valid timestamp") from exc
    if pd.isna(value):
        raise ValueError(f"{key} must be a valid timestamp, not NaT")
    return value


def _positive_int(row: Mapping[str, object], key: str) -> int:
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return int(value)


def validate_report_row(row: Mapping[str, object]) -> dict[str, object]:
    """Gate one report row before emission; returns a plain-dict copy.

    Every reporting builder (tasks 4.2-4.5) must route its finished row through
    this gate so that mixed schema families, missing provenance, and prohibited
    attribution labels fail before any artifact sees the row.
    """
    if not isinstance(row, Mapping):
        raise TypeError("report row must be a mapping")
    schema_id = row.get("schema")
    schema = REPORT_SCHEMAS.get(schema_id) if isinstance(schema_id, str) else None
    if schema is None:
        raise ValueError(
            f"unknown report schema {schema_id!r}; known: {sorted(REPORT_SCHEMAS)}"
        )

    # keys only, never values — provenance prose may legitimately mention CAPM
    ambiguous = sorted(k for k in row if "capm" in k.lower() or "jensen" in k.lower())
    if ambiguous:
        raise ValueError(
            f"prohibited attribution label(s) {ambiguous}: current mixed-local-currency "
            "histories publish raw market-model intercepts, never CAPM/Jensen alpha"
        )

    # None and NaN are both "absent" — NaN is the normal pandas/CSV missing form
    missing = [
        key
        for key in REQUIRED_PROVENANCE
        if row.get(key) is None
        or (isinstance(row.get(key), float) and math.isnan(row.get(key)))  # type: ignore[arg-type]
    ]
    if missing:
        raise ValueError(f"row is missing required provenance: {', '.join(missing)}")
    for key in ("portfolio_id", "return_basis", "window_label", "cash_benchmark_id", "source"):
        value = row[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty string")

    if _timestamp(row, "start") > _timestamp(row, "end"):
        raise ValueError("start must be on or before end")
    _positive_int(row, "n_obs")
    periods = _positive_int(row, "periods_per_year")
    if schema.periods_per_year is not None and periods != schema.periods_per_year:
        raise ValueError(
            f"{schema.family} rows annualize on {schema.periods_per_year} periods/year, "
            f"got {periods}"
        )
    if row["currency_basis"] not in _CURRENCY_BASES:
        raise ValueError(
            f"currency_basis must be one of {_CURRENCY_BASES}, got {row['currency_basis']!r}"
        )
    if schema.family == "reader" and row.get("row_kind") not in ("full", "performance_only"):
        raise ValueError("reader rows require row_kind 'full' or 'performance_only'")
    if schema.family == "monthly":
        # task 4.5: a monthly row's labels are bound to its own window, so an
        # in-row mixed-window mutation fails deterministically (R8.6)
        for key in ("year", "month", "monthly_return"):
            if key not in row:
                raise ValueError(f"monthly rows require {key}")
        year, month = row["year"], row["month"]
        if isinstance(year, bool) or not isinstance(year, Integral):
            raise ValueError("year must be an integer")
        if isinstance(month, bool) or not isinstance(month, Integral) or not 1 <= int(month) <= 12:
            raise ValueError("month must be an integer in 1..12")
        monthly_return = row["monthly_return"]
        if (
            isinstance(monthly_return, bool)
            or not isinstance(monthly_return, Real)
            or not math.isfinite(float(monthly_return))  # type: ignore[arg-type]
        ):
            raise ValueError("monthly_return must be a finite real number")
        month_start, month_end = _timestamp(row, "start"), _timestamp(row, "end")
        labeled = (int(year), int(month))
        if (month_start.year, month_start.month) != labeled or (
            month_end.year,
            month_end.month,
        ) != labeled:
            raise ValueError(
                f"monthly row labeled {labeled[0]}-{labeled[1]:02d} but its window is "
                f"{month_start.date()}..{month_end.date()}"
            )
        if int(row["n_obs"]) > calendar.monthrange(*labeled)[1]:  # type: ignore[arg-type]
            raise ValueError(
                f"monthly n_obs {row['n_obs']} exceeds the days in {labeled[0]}-{labeled[1]:02d}"
            )

    unknown = sorted(set(row) - schema.fields)
    if unknown:
        raise ValueError(
            f"field(s) {unknown} are not part of {schema.schema_id}; a row may not mix "
            "schema families or invent undeclared fields"
        )
    if schema.family == "reader" and any(k.startswith("raw_market_model_") for k in row):
        # task 4.4: attribution may ride in a reader row ONLY as a full row whose
        # attribution window is exactly the performance window; anything shorter
        # is emitted separately as its own attribution record (R3.6, R3.7)
        if row.get("row_kind") != "full":
            raise ValueError(
                "attribution fields require row_kind 'full'; a performance_only row "
                "emits its shortened attribution as a separate record"
            )
        binding = ("raw_market_model_start", "raw_market_model_end", "raw_market_model_n_obs")
        if any(key not in row for key in binding):
            raise ValueError(
                "full rows must carry raw_market_model_start/end/n_obs to prove the "
                "attribution window equals the performance window"
            )
        if tuple(row[k] for k in binding) != (row["start"], row["end"], row["n_obs"]):
            raise ValueError(
                "mixed windows: full-row attribution start/end/n_obs "
                f"{tuple(row[k] for k in binding)!r} differ from performance coverage "
                f"{(row['start'], row['end'], row['n_obs'])!r}"
            )
    return dict(row)


# --- Row builders (task 4.2): reader-facing and explicit legacy rows ------------

_READER_METRIC_KEYS = ("returns", "total_return", "maxdd", "downside_rms", "cagr", "ann_vol", "calmar")
_LEGACY_METRIC_KEYS = (
    "returns",
    "total_return",
    "maxdd",
    "downside_rms",
    "cagr_rows",
    "ann_vol_cal",
    "sharpe_cal",
    "sortino_cal",
    "calmar_rows",
)


def _metric_block_input(metrics: Mapping[str, object], required: tuple[str, ...]) -> pd.Series:
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be the evaluation.metric_block(...) mapping")
    missing = [key for key in required if key not in metrics]
    if missing:
        raise ValueError(
            f"metrics must be the evaluation.metric_block(...) mapping; missing: {', '.join(missing)}"
        )
    returns = metrics["returns"]
    if not isinstance(returns, pd.Series) or returns.empty:
        raise ValueError("metrics['returns'] must be a non-empty pandas Series")
    return returns


def _row_provenance(
    meta: LineMetadata,
    returns: pd.Series,
    *,
    schema: str,
    periods_per_year: int,
    source: str,
) -> dict[str, object]:
    if not isinstance(meta, LineMetadata):
        raise TypeError("meta must be a LineMetadata")
    return {
        "schema": schema,
        "portfolio_id": meta.portfolio_id,
        "return_basis": meta.total_return_basis,
        "window_label": meta.window_label,
        "start": returns.index[0],
        "end": returns.index[-1],
        "n_obs": len(returns),
        "periods_per_year": periods_per_year,
        "cash_benchmark_id": meta.cash_benchmark_id,
        "currency_basis": meta.currency_basis,
        "source": source,
    }


def _stream_sharpe_sortino(stream: pd.Series) -> tuple[float, float, float]:
    """Sharpe, Sortino, and downside RMS of one daily stream under
    metric_block's exact conventions."""
    mean = float(stream.mean())
    std = float(stream.std(ddof=1))
    downside = np.minimum(stream.to_numpy(dtype=float), 0.0)
    rms = float(np.sqrt(np.mean(downside**2))) if len(downside) else float("nan")
    root = math.sqrt(TRADING_DAYS)
    return (
        mean / std * root if std > 0 else float("nan"),
        mean / rms * root if rms > 0 else float("nan"),
        rms,
    )


def _require_ssr_on_stream(
    ssr: SSRInference, stream_sharpe: float, n_obs: int, *, label: str
) -> None:
    """Stream-identity gate (R2.1, R1.4): on the honest path the SSR input's
    full-sample Sharpe equals the stream Sharpe recomputed by the builder (same
    formula, same data), so an SSR built on a DIFFERENT stream — e.g. the raw
    returns, the historical defect class — fails here instead of emitting a
    self-contradictory row. Carve-out: the documented insufficient-inference
    result keeps sr_full NaN (design ~L472)."""
    if not isinstance(ssr, SSRInference):
        raise TypeError("ssr must be an SSRInference")
    if ssr.result.n_obs != n_obs:
        raise ValueError(
            f"ssr was computed on {ssr.result.n_obs} observations but the {label} has "
            f"{n_obs}; SSR must come from the same validated {label}"
        )
    if ssr.periods_per_year != TRADING_DAYS:
        raise ValueError("report rows require SSR on the 252-day convention")
    if ssr.result.n_rolling >= 10:
        sr_full = float(ssr.result.sr_full)
        coherent = (math.isnan(stream_sharpe) and math.isnan(sr_full)) or math.isclose(
            stream_sharpe, sr_full, rel_tol=1e-9, abs_tol=1e-12
        )
        if not coherent:
            raise ValueError(
                f"ssr was not computed on this {label}: its full-sample Sharpe "
                f"{sr_full!r} does not match the {label} Sharpe {stream_sharpe!r} (R2.1, R1.4)"
            )


def _ssr_fields(ssr: SSRInference) -> dict[str, object]:
    fields = {"ssr_" + f.name: getattr(ssr.result, f.name) for f in dataclasses.fields(SSRResult)}
    fields.update(
        {
            "ssr_" + f.name: getattr(ssr, f.name)
            for f in dataclasses.fields(SSRInference)
            if f.name != "result"
        }
    )
    return fields


def build_reader_metric_row(
    meta: LineMetadata,
    metrics: Mapping[str, object],
    cash_returns: pd.Series,
    ssr: SSRInference,
    *,
    source: str,
    attribution: MarketAttribution | None = None,
) -> dict[str, object]:
    """Reader-facing row: elapsed CAGR + 252-day risk scaling from ``metric_block``,
    with Sharpe/Sortino/SSR built from ONE validated cash-excess return stream
    (R1.1, R1.3, R2.1). The excess stream is derived HERE from the row's own
    performance returns and the supplied cash series, so one portfolio definition
    and one return stream hold by construction. Attribution joins only when it
    covers the exact performance window; otherwise the row is ``performance_only``
    and the shortened attribution is emitted separately by task 4.4 (R3.6).
    """
    returns = _metric_block_input(metrics, _READER_METRIC_KEYS)
    # exact-index, finite-value contract: rejects misaligned or NaN-bearing cash
    excess_returns = portfolio_excess_returns(returns, cash_returns)
    # Sharpe/Sortino mirror metric_block's exact conventions on the excess stream —
    # metric_block's own sharpe/sortino are raw-return and must not be copied.
    sharpe, sortino, _ = _stream_sharpe_sortino(excess_returns)
    _require_ssr_on_stream(ssr, sharpe, len(excess_returns), label="cash-excess stream")

    row = _row_provenance(
        meta, returns, schema=READER_SCHEMA, periods_per_year=TRADING_DAYS, source=source
    )
    row.update(
        {key: metrics[key] for key in ("total_return", "maxdd", "downside_rms", "cagr", "ann_vol", "calmar")}
    )
    row["sharpe"] = sharpe
    row["sortino"] = sortino
    row.update(_ssr_fields(ssr))
    row["row_kind"] = "performance_only"
    if attribution is not None:
        if not isinstance(attribution, MarketAttribution):
            raise TypeError("attribution must be a MarketAttribution")
        if attribution.periods_per_year != TRADING_DAYS:
            raise ValueError("reader rows require attribution on the 252-day convention")
        if (attribution.start, attribution.end, attribution.n_obs) == (
            row["start"],
            row["end"],
            row["n_obs"],
        ):
            row["row_kind"] = "full"
            row.update(
                {
                    "raw_market_model_" + f.name: getattr(attribution, f.name)
                    for f in dataclasses.fields(MarketAttribution)
                }
            )
        # else: shorter coverage stays performance_only (task 4.4 emits the
        # shortened attribution as its own record)
    return validate_report_row(row)


def build_differential_metric_row(
    meta: LineMetadata,
    comparison_returns: pd.Series,
    reference_returns: pd.Series,
    ssr: SSRInference,
    *,
    source: str,
) -> dict[str, object]:
    """One coherent differential row (R1.4): every portfolio statistic derives
    from the exact daily comparison-minus-reference spread built HERE, so
    changing either source line changes every statistic through this one
    producer. The endpoint wealth gap survives only as the separately named
    descriptive ``endpoint_total_return_difference`` (R1.5) — it is never a
    statistic of the spread portfolio.
    """
    spread = differential_returns(comparison_returns, reference_returns)
    if bool((spread.to_numpy(dtype=float) <= -1.0).any()):
        raise ValueError(
            "differential spread contains a return <= -100%; the compounded spread "
            "curve is undefined"
        )
    sharpe, sortino, downside_rms = _stream_sharpe_sortino(spread)
    _require_ssr_on_stream(ssr, sharpe, len(spread), label="differential spread")

    # ponytail: base-anchor the spread curve one session before the first return
    # so total return, CAGR, and drawdown share one wealth definition
    base = spread.index[0] - pd.offsets.BDay()
    curve = pd.concat([pd.Series([1.0], index=pd.DatetimeIndex([base])), (1.0 + spread).cumprod()])
    mdd = max_drawdown(curve)
    growth = cagr(curve)
    std = float(spread.std(ddof=1))

    row = _row_provenance(
        meta, spread, schema=DIFFERENTIAL_SCHEMA, periods_per_year=TRADING_DAYS, source=source
    )
    row.update(
        {
            "total_return": float(curve.iloc[-1] - 1.0),
            "maxdd": float(mdd),
            "downside_rms": downside_rms,
            "cagr": float(growth),
            "ann_vol": std * math.sqrt(TRADING_DAYS),
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": float(growth / abs(mdd)) if mdd else float("nan"),
            "endpoint_total_return_difference": float(
                (1.0 + comparison_returns).prod() - (1.0 + reference_returns).prod()
            ),
        }
    )
    row.update(_ssr_fields(ssr))
    return validate_report_row(row)


def build_legacy_metric_row(
    meta: LineMetadata,
    metrics: Mapping[str, object],
    *,
    source: str,
) -> dict[str, object]:
    """Explicit vectorbt/365 legacy row under separately named fields (R1.2, R1.6)."""
    returns = _metric_block_input(metrics, _LEGACY_METRIC_KEYS)
    row = _row_provenance(
        meta, returns, schema=LEGACY_SCHEMA, periods_per_year=CALENDAR_DAYS, source=source
    )
    row.update(
        {
            key: metrics[key]
            for key in (
                "total_return",
                "maxdd",
                "downside_rms",
                "cagr_rows",
                "ann_vol_cal",
                "sharpe_cal",
                "sortino_cal",
                "calmar_rows",
            )
        }
    )
    return validate_report_row(row)


# --- Separate window records (task 4.4) -------------------------------------------


def build_attribution_record(
    meta: LineMetadata,
    attribution: MarketAttribution,
    *,
    source: str,
) -> dict[str, object]:
    """Shortened (or standalone) attribution as its OWN record: actual start,
    end, count, annualization, and model identity — never disguised as a full
    reader row (R3.7)."""
    if not isinstance(meta, LineMetadata):
        raise TypeError("meta must be a LineMetadata")
    if not isinstance(attribution, MarketAttribution):
        raise TypeError("attribution must be a MarketAttribution")
    row = {
        "schema": ATTRIBUTION_SCHEMA,
        "portfolio_id": meta.portfolio_id,
        "return_basis": meta.total_return_basis,
        "window_label": meta.window_label,
        "start": attribution.start,
        "end": attribution.end,
        "n_obs": attribution.n_obs,
        "periods_per_year": attribution.periods_per_year,
        "cash_benchmark_id": meta.cash_benchmark_id,
        "currency_basis": meta.currency_basis,
        "source": source,
    }
    row.update(
        {
            "raw_market_model_" + f.name: getattr(attribution, f.name)
            for f in dataclasses.fields(MarketAttribution)
        }
    )
    return validate_report_row(row)


def build_crisis_record(
    meta: LineMetadata,
    crisis: CrisisMetrics,
    *,
    source: str,
) -> dict[str, object]:
    """Project the shared typed crisis result into a report row VERBATIM — no
    boundary recalculation (R4.5). Provenance window = anchor..actual_end."""
    if not isinstance(meta, LineMetadata):
        raise TypeError("meta must be a LineMetadata")
    if not isinstance(crisis, CrisisMetrics):
        raise TypeError("crisis must be a CrisisMetrics")
    row = {
        "schema": CRISIS_SCHEMA,
        "portfolio_id": meta.portfolio_id,
        "return_basis": meta.total_return_basis,
        "window_label": meta.window_label,
        "start": crisis.anchor,
        "end": crisis.actual_end,
        "n_obs": crisis.n_returns,
        "periods_per_year": crisis.periods_per_year,
        "cash_benchmark_id": meta.cash_benchmark_id,
        "currency_basis": meta.currency_basis,
        "source": source,
    }
    row.update({f.name: getattr(crisis, f.name) for f in dataclasses.fields(CrisisMetrics)})
    return validate_report_row(row)


# --- Table assembly and monthly-return semantics (task 4.5) -----------------------


def build_monthly_return_rows(
    meta: LineMetadata,
    metrics: Mapping[str, object],
    *,
    source: str,
) -> list[dict[str, object]]:
    """Canonical monthly-return rows compounded from the SAME validated strategy
    return stream as the corresponding performance row (task 4.5)."""
    returns = _metric_block_input(metrics, ("returns", "total_return"))
    rows: list[dict[str, object]] = []
    for (year, month), chunk in returns.groupby([returns.index.year, returns.index.month]):
        row = _row_provenance(
            meta, chunk, schema=MONTHLY_SCHEMA, periods_per_year=12, source=source
        )
        row.update(
            {
                "year": int(year),
                "month": int(month),
                "monthly_return": float((1.0 + chunk).prod() - 1.0),
            }
        )
        rows.append(validate_report_row(row))
    return rows


def report_table(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Validate and assemble emitted rows into one table (task 4.5).

    Row-level: every row re-passes the pre-emission gate. Table-level: rows
    sharing (portfolio_id, schema, window_label) must agree on window metadata
    (mixed windows / mixed portfolio definitions), and monthly rows must
    recompound to their reader row's total_return (stale generated values).
    """
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("rows must be a sequence of report-row mappings")
    if not rows:
        raise ValueError("report table must contain at least one row")
    validated = [validate_report_row(row) for row in rows]

    coherence: dict[tuple, tuple] = {}
    for row in validated:
        if row["schema"] == MONTHLY_SCHEMA:
            continue  # monthly rows carry per-month windows by design
        key = (row["portfolio_id"], row["schema"], row["window_label"])
        window = (row["start"], row["end"], row["n_obs"], row["periods_per_year"])
        if key in coherence and coherence[key] != window:
            raise ValueError(
                f"mixed windows for portfolio/schema/window {key!r}: "
                f"{coherence[key]!r} vs {window!r}"
            )
        coherence[key] = window

    monthly: dict[tuple, list[dict[str, object]]] = {}
    seen_months: set[tuple] = set()
    for row in validated:
        if row["schema"] == MONTHLY_SCHEMA:
            key = (
                row["portfolio_id"],
                row["window_label"],
                int(row["year"]),  # type: ignore[arg-type]
                int(row["month"]),  # type: ignore[arg-type]
            )
            if key in seen_months:
                raise ValueError(
                    f"duplicate month {key[2]}-{key[3]:02d} for ({key[0]!r}, {key[1]!r})"
                )
            seen_months.add(key)
            monthly.setdefault((row["portfolio_id"], row["window_label"]), []).append(row)
    for row in validated:
        if row["schema"] != READER_SCHEMA:
            continue
        months = monthly.get((row["portfolio_id"], row["window_label"]))
        if not months:
            continue
        if "total_return" not in row:
            raise ValueError(
                "stale-value check requires total_return on the reader row for "
                f"({row['portfolio_id']!r}, {row['window_label']!r})"
            )
        compounded = 1.0
        for month_row in months:
            compounded *= 1.0 + float(month_row["monthly_return"])  # type: ignore[arg-type]
        compounded -= 1.0
        if not math.isclose(
            compounded, float(row["total_return"]), rel_tol=1e-9, abs_tol=1e-12  # type: ignore[arg-type]
        ):
            raise ValueError(
                f"stale generated values: monthly returns compound to {compounded!r} but "
                f"the performance row reports total_return {row['total_return']!r} for "
                f"({row['portfolio_id']!r}, {row['window_label']!r})"
            )
    return pd.DataFrame(validated)
