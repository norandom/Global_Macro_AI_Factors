"""Tear-sheet data pack for every equity line, as Excel-ready CSVs.

Per line (static B&H both windows, the factor PIT/non-PIT/v2 lines, and the
2019-2024 tracks): return/risk metrics under the repo's published conventions
(365-day annualization, day-0 zero return; mirrors head_to_head_report via
vectorbt), tail statistics, the Sharpe Stability Ratio block (Newey-West HAC,
Andrews bandwidth; not reproducible in native Excel, hence precomputed), and
a two-level risk decomposition:

- CAPM vs SPY: beta, annualized alpha, R², the systematic share relative
  to the broad equity market; residual vol is "idiosyncratic-to-the-market"
  (for a multi-asset book much of it is OTHER systematic factors: gold, rates).
- Basket (4-ETF) regression: R² against the portfolio's own asset-class
  factors (SWDA.L/XLK/IAU/BIL daily returns). For a static basket this is ~1:
  single-name idiosyncratic risk is already diversified inside the ETF
  wrappers, and what remains for dynamic lines is allocation-timing residual,
  not stock-picking risk.

Outputs (data/tear_sheet/, gitignored; upload to the data release):
- tear_sheet.csv            — one row per line, all metrics as columns
- risk_decomposition.csv    — CAPM + basket regression per line
- monthly_returns_<key>.csv — month x year matrices for the headline lines

Prices for SPY + the 4 ETFs via yfinance (documented DB substitution).
Reproducible: ``uv run python scripts/build_tear_sheet.py``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "workbook"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

from factor_workbook.rederive import equity_metrics  # noqa: E402
from macro_framework.evaluation import crisis_metrics, metric_block  # noqa: E402
from macro_framework.reporting import (  # noqa: E402
    ATTRIBUTION_SCHEMA,
    CRISIS_SCHEMA,
    DIFFERENTIAL_SCHEMA,
    LEGACY_SCHEMA,
    MONTHLY_SCHEMA,
    READER_SCHEMA,
    REQUIRED_PROVENANCE,
    LineMetadata,
    build_attribution_record,
    build_crisis_record,
    build_monthly_return_rows,
    build_reader_metric_row,
    report_table,
    validate_report_row,
)
from macro_framework.skill_metric import (  # noqa: E402
    portfolio_excess_returns,
    raw_market_model_attribution,
)
from macro_framework.ssr import ssr_inference  # noqa: E402

OUT = REPO / "data" / "tear_sheet"
BASKET = ["SWDA.L", "XLK", "IAU", "BIL"]

LINES = {  # key -> (equity parquet, label)
    "static_bh_2016_2026": ("static_bh_equity_2016_2026.parquet", "Static B&H 25% EW (nb04 10y, in-sample)"),
    "static_bh_2014_2024": ("static_bh_equity_2014_2024.parquet", "Static B&H 25% EW (walk-forward window)"),
    "factor_pit_v1": ("factor_equity_v1.parquet", "PIT recall-guarded factor (deployable)"),
    "factor_nonpit_diag": ("factor_nonpit_diagnostic_equity_v1.parquet", "Non-PIT recall-enabled (DIAGNOSTIC)"),
    "factor_pit_v2": ("factor_equity_v2.parquet", "PIT factor, rejected prompt v2"),
    "baseline": ("baseline_equity_2019_2024.parquet", "Baseline HRP+momentum"),
    "track_a_llm": ("track_a_equity_2019_2024.parquet", "Track A (LLM directional)"),
    "track_a_steered": ("track_a_steered_equity_2019_2024.parquet", "Track A memory-guarded"),
    "track_b": ("track_b_equity_2019_2024.parquet", "Track B (MC/Nash)"),
}
MONTHLY_KEYS = ["static_bh_2016_2026", "factor_pit_v1", "factor_nonpit_diag"]

ANNUAL = 365  # repo convention (vectorbt calendar-year basis)


def _active_value(value: pd.Series) -> pd.Series:
    """The line from the last flat day before it first moves (skips pre-start stubs).

    Several 2019-start tracks are stored on 2014-anchored frames with a flat
    stub; including the stub dilutes CAGR/vol/Sharpe (the published
    contrast-summary metrics embed that full-frame convention — the tear sheet
    deliberately reports the ACTIVE span instead, disclosed in the window
    columns, matching the luck-vs-skill table's slice convention).
    """
    moving = value[value.ne(value.iloc[0])]
    if moving.empty:
        return value
    first_move = moving.index.min()
    prior = value.index[value.index < first_move]
    start = prior.max() if len(prior) else first_move
    return value.loc[start:]


def _ols(y: pd.Series, x: pd.DataFrame) -> tuple[np.ndarray, float, pd.Series]:
    x_ = np.column_stack([np.ones(len(x)), x.to_numpy()])
    coef, *_ = np.linalg.lstsq(x_, y.to_numpy(), rcond=None)
    fitted = x_ @ coef
    resid = y - fitted
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coef, r2, resid


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    px = yf.download(BASKET + ["SPY"], start="2013-12-01", end="2026-02-01",
                     auto_adjust=True, progress=False)["Close"]
    factor_returns = px.pct_change().dropna(how="all")

    tear_rows, risk_rows = [], []
    for key, (fname, label) in LINES.items():
        path = REPO / "data" / fname
        if not path.exists():
            print(f"  SKIP {key} (missing {fname})")
            continue
        value = _active_value(pd.read_parquet(path)["value"])
        r = value.pct_change().dropna()
        m = equity_metrics(value)
        ssr_inf = ssr_inference(r)
        ssr = ssr_inf.result

        neg = r[r < 0]
        dd = value / value.cummax() - 1
        dd_end = dd.idxmin()
        dd_start = value.loc[:dd_end].idxmax()
        recovery = dd.loc[dd_end:][dd.loc[dd_end:] >= -1e-12]
        monthly = (1 + r).resample("ME").prod() - 1

        tear_rows.append({
            "line": key, "label": label,
            "start": r.index.min().date(), "end": r.index.max().date(),
            "n_days": len(r),
            "total_return": m.total_return, "cagr": m.annualized_return,
            "ann_vol": m.annualized_vol, "sharpe": m.sharpe,
            "sortino": m.sortino, "calmar": m.calmar,
            "max_drawdown": m.max_drawdown,
            "max_dd_peak": dd_start.date(), "max_dd_trough": dd_end.date(),
            "max_dd_recovered": recovery.index.min().date() if len(recovery) else None,
            "skew": float(r.skew()), "excess_kurtosis": float(r.kurtosis()),
            "var_95_daily": float(r.quantile(0.05)),
            "cvar_95_daily": float(r[r <= r.quantile(0.05)].mean()),
            "best_day": float(r.max()), "worst_day": float(r.min()),
            "positive_day_rate": float((r > 0).mean()),
            "best_month": float(monthly.max()), "worst_month": float(monthly.min()),
            "crisis_2022_return": m.crisis_return,
            "crisis_2022_max_dd": m.crisis_max_drawdown,
            "ssr": ssr.ssr, "mean_rolling_sharpe": ssr.mean_rolling_sr,
            "nw_sigma_hac": ssr.sigma_hac, "nw_bandwidth_L": ssr.L_hac,
            # No standalone MBB p column: SSR is the effect size, and the verdict
            # sentence already carries the p-value in prose. A bare p next to a bare
            # SSR invites thresholding one against the other.
            "ssr_verdict": ssr_inf.verdict(),
        })

        spy = factor_returns["SPY"].reindex(r.index).dropna()
        y = r.reindex(spy.index)
        coef, r2_capm, resid = _ols(y, spy.to_frame())
        basket = factor_returns[BASKET].reindex(r.index).dropna()
        yb = r.reindex(basket.index)
        _, r2_basket, resid_b = _ols(yb, basket)
        risk_rows.append({
            "line": key, "label": label,
            "beta_spy": float(coef[1]),
            "alpha_ann_vs_spy": float(coef[0] * ANNUAL),
            "r2_capm": r2_capm,
            "corr_spy": float(y.corr(spy)),
            "systematic_share_capm": r2_capm,
            "idio_vol_ann_capm": float(resid.std(ddof=1) * np.sqrt(ANNUAL)),
            "r2_basket_4etf": r2_basket,
            "residual_vol_ann_basket": float(resid_b.std(ddof=1) * np.sqrt(ANNUAL)),
            "note": ("basket R2 ~ 1: single-name idiosyncratic risk is diversified inside the "
                      "ETF wrappers; residual on dynamic lines is allocation-timing, not stock-picking"),
        })

        if key in MONTHLY_KEYS:
            mt = monthly.to_frame("ret")
            mt["year"], mt["month"] = mt.index.year, mt.index.month
            (mt.pivot_table(index="year", columns="month", values="ret")
               .to_csv(OUT / f"monthly_returns_{key}.csv", float_format="%.6f"))

        print(f"  {key}: sharpe {m.sharpe:.2f}  ssr {ssr.ssr:.3f}  beta {coef[1]:.2f}  "
              f"r2_capm {r2_capm:.2f}  r2_basket {r2_basket:.4f}")

    pd.DataFrame(tear_rows).to_csv(OUT / "tear_sheet.csv", index=False, float_format="%.8f")
    pd.DataFrame(risk_rows).to_csv(OUT / "risk_decomposition.csv", index=False, float_format="%.8f")
    print(f"[done] -> {OUT.relative_to(REPO)}")


# --------------------------------------------------------------------------- #
# Canonical report producer (tasks 9.1/9.2)                                    #
#                                                                              #
# This module is the ONE explicit owner of assembled canonical report tables   #
# (R7.1, R7.2). Strategy producers (scripts/extend_stream_2026.py,             #
# scripts/build_sjm_crowding.py, scripts/build_basket_long.py) stay limited    #
# to immutable strategy outputs and run-local records; this producer reads     #
# them ONLY through completed manifests with verified hashes and projects the  #
# validated run-local records into publication tables without recalculating    #
# any financial metric (R7.4, R7.6).                                           #
# --------------------------------------------------------------------------- #

#: The single owner of every assembled canonical report table (task 9.1).
REPORT_TABLE_OWNER = "scripts/build_tear_sheet.py"

#: Schema identity of the Factor bundle's manifest-owned run-local records
#: (task 6.6/6.7); pinned here so a schema-incompatible payload fails loudly.
FACTOR_METRIC_RECORDS_SCHEMA = "factor_run.metric_records.v1"

FACTOR_PIT_PORTFOLIO = "factor_pit_ext2026"
FACTOR_NONPIT_PORTFOLIO = "factor_nonpit_diagnostic_ext2026"
FACTOR_DIFFERENTIAL_PORTFOLIO = "factor_nonpit_minus_pit_ext2026"

#: The complete approved (portfolio, schema) catalog of Factor run-local
#: records, in the producer's deterministic order. The assembler accepts
#: exactly this catalog — never a second, independently recalculated family.
FACTOR_RECORD_CATALOG: tuple[tuple[str, str], ...] = (
    (FACTOR_PIT_PORTFOLIO, READER_SCHEMA),
    (FACTOR_PIT_PORTFOLIO, LEGACY_SCHEMA),
    (FACTOR_NONPIT_PORTFOLIO, READER_SCHEMA),
    (FACTOR_NONPIT_PORTFOLIO, LEGACY_SCHEMA),
    (FACTOR_DIFFERENTIAL_PORTFOLIO, DIFFERENTIAL_SCHEMA),
    (FACTOR_PIT_PORTFOLIO, ATTRIBUTION_SCHEMA),
    (FACTOR_NONPIT_PORTFOLIO, ATTRIBUTION_SCHEMA),
    (FACTOR_PIT_PORTFOLIO, CRISIS_SCHEMA),
    (FACTOR_NONPIT_PORTFOLIO, CRISIS_SCHEMA),
)

#: Canonical Factor/AI-variant publication tables (task 9.2) and their table
#: schema identities, matching the frozen data-v4 asset catalog (task 10.1).
FACTOR_REPORT_TABLE_SCHEMAS: Mapping[str, str] = {
    "portfolio_metrics_reader_ext2026": READER_SCHEMA,
    "portfolio_metrics_vectorbt365_ext2026": LEGACY_SCHEMA,
    "portfolio_metrics_differential_ext2026": DIFFERENTIAL_SCHEMA,
    "attribution_raw_market_model_ext2026": ATTRIBUTION_SCHEMA,
    "crisis_metrics_ext2026": CRISIS_SCHEMA,
    "tear_sheet_ai_variants_ext2026": "tear_sheet.ai_variants.v1",
}

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class VerifiedReportInput:
    """One canonical report input verified against its completed manifest."""

    family: str  # "sjm_run" | "market_snapshot" | "markowitz_inputs"
    run_dir: Path
    identity: str
    manifest_sha256: str
    manifest: Mapping[str, object]


@dataclass(frozen=True)
class VerifiedFactorRun:
    """The completed Factor bundle plus its manifest-owned run-local records."""

    run_dir: Path
    run_id: str
    manifest_sha256: str
    manifest: Mapping[str, object]
    metric_records: Mapping[str, object]


def _factor_run_module():
    """scripts/extend_stream_2026 lazily: keeps macro_framework.factor_scoring
    out of sys.modules at test collection time (repo convention)."""
    try:
        from scripts import extend_stream_2026 as ext
    except ImportError:  # scripts/ itself on sys.path (test convention)
        import extend_stream_2026 as ext
    return ext


def _sjm_module():
    try:
        from scripts import build_sjm_crowding as sjm
    except ImportError:
        import build_sjm_crowding as sjm
    return sjm


def _snapshot_module():
    try:
        from scripts import build_basket_long as producer
    except ImportError:
        import build_basket_long as producer
    return producer


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pinned_identity(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty pinned identity string")
    return value


def _pinned_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a pinned 64-hex sha256 digest")
    return value


def _require_run_directory(path: Path | str, name: str) -> Path:
    path = Path(path)
    if not path.is_dir() or not (path / "manifest.json").is_file():
        raise ValueError(
            f"{name} must be a completed, manifest-carrying run directory; a "
            f"loose artifact is never a canonical report input: {path}"
        )
    return path


def _require_pinned_manifest(
    run_dir: Path,
    *,
    identity_key: str,
    identity: str,
    manifest_sha256: str,
    label: str,
) -> dict[str, object]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    declared = manifest.get(identity_key)
    if declared != identity:
        raise ValueError(
            f"{label} identity mismatch: expected {identity!r}, the completed "
            f"manifest declares {declared!r}"
        )
    actual = _sha256_file(run_dir / "manifest.json")
    if actual != manifest_sha256:
        raise ValueError(
            f"{label} manifest sha256 mismatch: the pinned digest is stale or "
            "the manifest was rewritten after completion"
        )
    return manifest


def load_factor_report_input(
    run_dir: Path | str, *, run_id: str, manifest_sha256: str
) -> VerifiedFactorRun:
    """Load the completed Factor bundle for canonical reporting (task 9.1).

    Full bundle validation (task 6.9: completion marker, byte inventory,
    record-level revalidation) runs first, then the caller's pinned identity
    and manifest digest must hold, then the manifest-owned run-local metric
    records are read back under their inventoried hash. Tampered, loose,
    incomplete, stale, or schema-incompatible inputs all fail here — before
    any table assembly exists.
    """
    _pinned_identity("run_id", run_id)
    _pinned_sha256("manifest_sha256", manifest_sha256)
    run_dir = _require_run_directory(run_dir, "factor run input")
    ext = _factor_run_module()
    ext.load_completed_factor_run(run_dir)
    manifest = _require_pinned_manifest(
        run_dir,
        identity_key="run_id",
        identity=run_id,
        manifest_sha256=manifest_sha256,
        label="factor run",
    )
    entry = manifest["files"]["metric_records"]
    metric_path = run_dir / entry["file"]
    metric_bytes = metric_path.read_bytes()
    if hashlib.sha256(metric_bytes).hexdigest() != entry["sha256"]:
        raise ValueError(
            f"{run_dir}: {entry['file']} bytes were mutated after inventory"
        )
    metric_records = json.loads(metric_bytes.decode("utf-8"))
    if metric_records.get("schema") != FACTOR_METRIC_RECORDS_SCHEMA:
        raise ValueError(
            f"{run_dir}: run-local records declare schema "
            f"{metric_records.get('schema')!r}, not {FACTOR_METRIC_RECORDS_SCHEMA!r}"
        )
    return VerifiedFactorRun(
        run_dir=run_dir,
        run_id=run_id,
        manifest_sha256=manifest_sha256,
        manifest=manifest,
        metric_records=metric_records,
    )


def load_sjm_report_input(
    run_dir: Path | str, *, run_id: str, manifest_sha256: str
) -> VerifiedReportInput:
    """Load the completed SJM v3 run for canonical reporting (task 9.1)."""
    _pinned_identity("run_id", run_id)
    _pinned_sha256("manifest_sha256", manifest_sha256)
    run_dir = _require_run_directory(run_dir, "SJM run input")
    _sjm_module().validate_sjm_run(run_dir)
    manifest = _require_pinned_manifest(
        run_dir,
        identity_key="run_id",
        identity=run_id,
        manifest_sha256=manifest_sha256,
        label="SJM run",
    )
    return VerifiedReportInput(
        family="sjm_run",
        run_dir=run_dir,
        identity=run_id,
        manifest_sha256=manifest_sha256,
        manifest=manifest,
    )


def _load_snapshot_input(
    snapshot_dir: Path | str,
    *,
    snapshot_id: str,
    manifest_sha256: str,
    family: str,
) -> VerifiedReportInput:
    _pinned_identity("snapshot_id", snapshot_id)
    _pinned_sha256("manifest_sha256", manifest_sha256)
    snapshot_dir = _require_run_directory(snapshot_dir, f"{family} input")
    _snapshot_module().validate_market_snapshot(snapshot_dir)
    manifest = _require_pinned_manifest(
        snapshot_dir,
        identity_key="snapshot_id",
        identity=snapshot_id,
        manifest_sha256=manifest_sha256,
        label="market snapshot",
    )
    return VerifiedReportInput(
        family=family,
        run_dir=snapshot_dir,
        identity=snapshot_id,
        manifest_sha256=manifest_sha256,
        manifest=manifest,
    )


def load_market_report_input(
    snapshot_dir: Path | str, *, snapshot_id: str, manifest_sha256: str
) -> VerifiedReportInput:
    """Load the completed market snapshot for canonical reporting (task 9.1)."""
    return _load_snapshot_input(
        snapshot_dir,
        snapshot_id=snapshot_id,
        manifest_sha256=manifest_sha256,
        family="market_snapshot",
    )


def load_markowitz_report_input(
    snapshot_dir: Path | str, *, snapshot_id: str, manifest_sha256: str
) -> VerifiedReportInput:
    """Load the Markowitz input family (task 9.1): the SAME completed market
    snapshot, whose three validated frames (local ETF closes, USD-per-GBP FX,
    and cash/market total-return levels) feed the weekly USD valuations."""
    return _load_snapshot_input(
        snapshot_dir,
        snapshot_id=snapshot_id,
        manifest_sha256=manifest_sha256,
        family="markowitz_inputs",
    )


# --- Task 9.2: canonical Factor and AI-variant tables ------------------------- #


@dataclass(frozen=True)
class FactorReportTables:
    """Assembled canonical Factor tables plus their verified producer lineage."""

    owner: str
    lineage: Mapping[str, object]
    tables: Mapping[str, pd.DataFrame]


def _record_window(row: Mapping[str, object], label: str) -> tuple:
    try:
        start = pd.Timestamp(row["start"])  # type: ignore[arg-type]
        end = pd.Timestamp(row["end"])  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} record window is invalid") from exc
    return start, end, int(row["n_obs"])  # type: ignore[arg-type]


def _stream_window(stream: Mapping[str, object], label: str) -> tuple:
    return _record_window(
        {"start": stream["start"], "end": stream["end"], "n_obs": stream["n_obs"]},
        label,
    )


def _require_stream_bound_records(
    bundle: Mapping[str, object],
) -> dict[tuple[str, str], Mapping[str, object]]:
    """Validate every run-local record against its declared source stream.

    The approved catalog must appear exactly once each — a duplicated or extra
    record (an independently recalculated second Factor row family) is not a
    canonical input. Reader/legacy/differential rows must equal their declared
    stream windows; attribution records must be an exact or shortened suffix
    coherent with the reader's row_kind; crisis records must not exceed their
    stream (R1.x, R3.6, R3.7, R7.2).
    """
    records = bundle["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("run-local records must be a non-empty list")
    keys = [(row["portfolio_id"], row["schema"]) for row in records]
    if keys != list(FACTOR_RECORD_CATALOG):
        raise ValueError(
            "factor run-local records must contain each approved portfolio/schema "
            f"exactly once in producer order; got {keys!r}. A second independently "
            "produced Factor row family is not a canonical input"
        )
    by_key = dict(zip(keys, records, strict=True))
    streams = bundle["source_streams"]

    for portfolio_id in (FACTOR_PIT_PORTFOLIO, FACTOR_NONPIT_PORTFOLIO):
        declared = _stream_window(
            streams[portfolio_id], f"source_streams.{portfolio_id}"
        )
        for schema in (READER_SCHEMA, LEGACY_SCHEMA):
            row = by_key[(portfolio_id, schema)]
            if _record_window(row, f"{portfolio_id} {schema}") != declared:
                raise ValueError(
                    f"{portfolio_id} {schema} record diverges from its declared "
                    "source stream"
                )
        attribution = by_key[(portfolio_id, ATTRIBUTION_SCHEMA)]
        a_start, a_end, a_n = _record_window(
            attribution, f"{portfolio_id} attribution"
        )
        if a_end != declared[1] or a_start < declared[0] or a_n > declared[2]:
            raise ValueError(
                f"{portfolio_id} attribution record exceeds its declared source "
                "stream: attribution must be an exact or shortened suffix"
            )
        reader = by_key[(portfolio_id, READER_SCHEMA)]
        if reader["row_kind"] == "full":
            if (a_start, a_end, a_n) != declared:
                raise ValueError(
                    f"{portfolio_id} full reader requires attribution over the "
                    "exact performance window of its declared source stream"
                )
        elif not (a_start > declared[0] and a_n < declared[2]):
            raise ValueError(
                f"{portfolio_id} performance_only reader requires a separately "
                "disclosed SHORTENED attribution record"
            )
        crisis = by_key[(portfolio_id, CRISIS_SCHEMA)]
        c_start, c_end, c_n = _record_window(crisis, f"{portfolio_id} crisis")
        if c_end > declared[1] or c_n > declared[2] or c_start > c_end:
            raise ValueError(
                f"{portfolio_id} crisis record diverges from its declared source "
                "stream"
            )

    differential_stream = streams[FACTOR_DIFFERENTIAL_PORTFOLIO]
    if (
        differential_stream["comparison"],
        differential_stream["reference"],
    ) != (FACTOR_NONPIT_PORTFOLIO, FACTOR_PIT_PORTFOLIO):
        raise ValueError(
            "differential source stream must declare comparison=non-PIT and "
            "reference=PIT"
        )
    declared = _stream_window(
        differential_stream, f"source_streams.{FACTOR_DIFFERENTIAL_PORTFOLIO}"
    )
    differential = by_key[(FACTOR_DIFFERENTIAL_PORTFOLIO, DIFFERENTIAL_SCHEMA)]
    if _record_window(differential, "differential") != declared:
        raise ValueError(
            "differential record diverges from its declared source stream"
        )
    return by_key


def assemble_factor_report_tables(
    metric_records: Mapping[str, object], *, lineage: Mapping[str, object]
) -> FactorReportTables:
    """Project validated Factor run-local records into the canonical tables.

    Pure projection (task 9.2): every published row IS a validated run-local
    record routed through the ``macro_framework.reporting`` emission gate — no
    financial metric is recalculated here. ``lineage`` must carry the verified
    ``factor_run`` identity and the ``market_snapshot`` identity the records
    were built on; a diverging snapshot lineage fails before any table exists.
    """
    if not isinstance(metric_records, Mapping):
        raise TypeError("metric_records must be the Factor run-local record bundle")
    if metric_records.get("schema") != FACTOR_METRIC_RECORDS_SCHEMA:
        raise ValueError(
            f"run-local records declare schema {metric_records.get('schema')!r}, "
            f"not {FACTOR_METRIC_RECORDS_SCHEMA!r}"
        )
    factor_lineage = lineage.get("factor_run")
    if not isinstance(factor_lineage, Mapping):
        raise ValueError("lineage.factor_run must identify the verified Factor run")
    _pinned_identity("lineage.factor_run.run_id", factor_lineage.get("run_id"))
    _pinned_sha256(
        "lineage.factor_run.manifest_sha256",
        factor_lineage.get("manifest_sha256"),
    )
    snapshot = metric_records["market_snapshot"]
    pinned_snapshot = lineage.get("market_snapshot")
    if not isinstance(pinned_snapshot, Mapping) or (
        pinned_snapshot.get("snapshot_id"),
        pinned_snapshot.get("manifest_sha256"),
    ) != (snapshot["snapshot_id"], snapshot["manifest_sha256"]):
        raise ValueError(
            "market-snapshot lineage diverges: canonical rows must trace to the "
            "verified snapshot the run-local records were built on"
        )

    by_key = _require_stream_bound_records(metric_records)
    reader_rows = [
        by_key[(portfolio_id, READER_SCHEMA)]
        for portfolio_id in (FACTOR_PIT_PORTFOLIO, FACTOR_NONPIT_PORTFOLIO)
    ]
    legacy_rows = [
        by_key[(portfolio_id, LEGACY_SCHEMA)]
        for portfolio_id in (FACTOR_PIT_PORTFOLIO, FACTOR_NONPIT_PORTFOLIO)
    ]
    attribution_rows = [
        by_key[(portfolio_id, ATTRIBUTION_SCHEMA)]
        for portfolio_id in (FACTOR_PIT_PORTFOLIO, FACTOR_NONPIT_PORTFOLIO)
    ]
    crisis_rows = [
        by_key[(portfolio_id, CRISIS_SCHEMA)]
        for portfolio_id in (FACTOR_PIT_PORTFOLIO, FACTOR_NONPIT_PORTFOLIO)
    ]
    differential_row = by_key[(FACTOR_DIFFERENTIAL_PORTFOLIO, DIFFERENTIAL_SCHEMA)]

    tables = {
        "portfolio_metrics_reader_ext2026": report_table(reader_rows),
        "portfolio_metrics_vectorbt365_ext2026": report_table(legacy_rows),
        "portfolio_metrics_differential_ext2026": report_table([differential_row]),
        "attribution_raw_market_model_ext2026": report_table(attribution_rows),
        "crisis_metrics_ext2026": report_table(crisis_rows),
        "tear_sheet_ai_variants_ext2026": report_table(
            [*reader_rows, differential_row]
        ),
    }
    return FactorReportTables(
        owner=REPORT_TABLE_OWNER,
        lineage={
            "factor_run": dict(factor_lineage),
            "market_snapshot": dict(pinned_snapshot),
        },
        tables=tables,
    )


def build_factor_report_tables(factor_input: VerifiedFactorRun) -> FactorReportTables:
    """Canonical Factor/AI-variant tables from ONE verified completed bundle."""
    if not isinstance(factor_input, VerifiedFactorRun):
        raise TypeError("factor_input must come from load_factor_report_input")
    manifest = factor_input.manifest
    lineage = {
        "factor_run": {
            "run_id": factor_input.run_id,
            "manifest_sha256": factor_input.manifest_sha256,
            "source_commit": manifest["source_commit"],
        },
        "market_snapshot": dict(manifest["input_manifests"]["market_snapshot"]),
    }
    return assemble_factor_report_tables(factor_input.metric_records, lineage=lineage)


# --- Task 9.6: canonical risk, attribution, crisis, and monthly-return tables --- #


#: Producer-owned schema of the corrected risk-decomposition table. It replaces
#: the retired CAPM-labeled decomposition: every field keeps the raw
#: market-model vocabulary (R3.9) and every record carries its SOURCE schema
#: plus full portfolio/window identity.
RISK_DECOMPOSITION_SCHEMA = "risk_decomposition.v1"

#: Canonical auxiliary tables (task 9.6), pinned to the frozen data-v4 catalog.
#: The attribution, crisis, and differential secondary tables stay owned by the
#: Factor family (task 9.2); one auxiliary build revalidates and carries them.
AUXILIARY_REPORT_TABLE_SCHEMAS: Mapping[str, str] = MappingProxyType(
    {
        "monthly_returns_ext2026": MONTHLY_SCHEMA,
        "risk_decomposition_ext2026": RISK_DECOMPOSITION_SCHEMA,
    }
)

#: Attribution-record fields a risk row projects verbatim — never relabeled.
RISK_PROJECTED_ATTRIBUTION_FIELDS = (
    "raw_market_model_kind",
    "raw_market_model_intercept_native_period",
    "raw_market_model_intercept_ann_arithmetic",
    "raw_market_model_intercept_se_hac",
    "raw_market_model_intercept_t_hac",
    "raw_market_model_beta",
    "raw_market_model_r2",
    "raw_market_model_hac_maxlags",
)


def risk_decomposition_columns() -> tuple[str, ...]:
    """The canonical (and locale-mirror source) schema of the risk table."""
    return (
        *REQUIRED_PROVENANCE,
        "source_schema",
        *RISK_PROJECTED_ATTRIBUTION_FIELDS,
        "systematic_variance_share",
        "idiosyncratic_variance_share",
    )


@dataclass(frozen=True)
class AuxiliaryReportTables:
    """Canonical monthly-return and risk tables plus the reconciled family."""

    owner: str
    lineage: Mapping[str, object]
    factor_tables: FactorReportTables
    rows: Mapping[str, tuple[Mapping[str, object], ...]]
    tables: Mapping[str, pd.DataFrame]


def _factor_inventory_entry(
    manifest: Mapping[str, object], artifact: str
) -> Mapping[str, object]:
    for entry in manifest["files"].values():
        if entry.get("file") == artifact:
            return entry
    raise ValueError(
        f"declared stream artifact {artifact!r} is not in the manifest inventory; "
        "canonical secondary tables build only from inventoried streams"
    )


def _require_attribution_reconciles(
    attribution: Mapping[str, object],
    reader: Mapping[str, object],
    portfolio_id: str,
) -> None:
    """The attribution record and the reader row's embedded attribution must be
    one shared result (task 9.6): secondary tables reconcile to their
    corresponding canonical portfolio rows."""
    for key in sorted(k for k in reader if k.startswith("raw_market_model_")):
        if not _values_equal(reader[key], attribution.get(key)):
            raise ValueError(
                f"{portfolio_id}: attribution record does not reconcile to the "
                f"canonical reader row on {key}; secondary tables must reconcile "
                "to their corresponding canonical portfolio rows"
            )


def _validated_stream_metrics(
    factor_input: VerifiedFactorRun, portfolio_id: str
) -> tuple[Mapping[str, object], str, str]:
    """metric_block over the manifest-inventoried persisted stream, proven to
    BE the declared source stream of the portfolio rows (same dates, count)."""
    stream = factor_input.metric_records["source_streams"][portfolio_id]
    artifact = str(stream["artifact"])
    entry = _factor_inventory_entry(factor_input.manifest, artifact)
    data = _read_inventoried_bytes(factor_input.run_dir / artifact, entry)
    value = pd.read_parquet(io.BytesIO(data))["value"]
    start, end, n_obs = _stream_window(stream, f"source_streams.{portfolio_id}")
    prior = value.index[value.index < start]
    if not len(prior):
        raise ValueError(
            f"{portfolio_id}: persisted stream lacks the value anchor preceding "
            f"{start.date()}"
        )
    metrics = metric_block(value.loc[prior[-1] : end])
    returns = metrics["returns"]
    if (returns.index[0], returns.index[-1], len(returns)) != (start, end, n_obs):
        raise ValueError(
            f"{portfolio_id}: persisted stream diverges from its declared source "
            "stream; secondary tables must use the SAME validated stream as the "
            "portfolio rows"
        )
    return metrics, artifact, str(entry["sha256"])


def _risk_decomposition_row(attribution: Mapping[str, object]) -> dict[str, object]:
    """Pure projection of one validated attribution record; the (possibly
    shortened) attribution window identity stays on the risk record."""
    validated = validate_report_row(attribution)
    r2 = float(validated["raw_market_model_r2"])
    row: dict[str, object] = {key: validated[key] for key in REQUIRED_PROVENANCE}
    row["start"] = pd.Timestamp(validated["start"])  # type: ignore[arg-type]
    row["end"] = pd.Timestamp(validated["end"])  # type: ignore[arg-type]
    row["n_obs"] = int(validated["n_obs"])  # type: ignore[arg-type]
    row["periods_per_year"] = int(validated["periods_per_year"])  # type: ignore[arg-type]
    row["schema"] = RISK_DECOMPOSITION_SCHEMA
    row["source_schema"] = validated["schema"]
    for key in RISK_PROJECTED_ATTRIBUTION_FIELDS:
        row[key] = validated[key]
    row["systematic_variance_share"] = r2
    row["idiosyncratic_variance_share"] = 1.0 - r2
    return row


def build_auxiliary_report_tables(
    factor_input: VerifiedFactorRun,
) -> AuxiliaryReportTables:
    """Canonical monthly-return and risk-decomposition tables from the SAME
    validated streams as the portfolio rows (task 9.6).

    One call revalidates and carries the whole secondary family: the
    attribution, crisis (boundary-inclusive values untouched), and differential
    tables come from the Factor family build, monthly rows recompound exactly
    to their canonical reader rows via the shared ``report_table`` stale-value
    gate, and risk rows are verbatim raw market-model projections whose
    shortened attribution windows stay separate.
    """
    factor_tables = build_factor_report_tables(factor_input)
    by_key = _require_stream_bound_records(factor_input.metric_records)

    monthly_rows: list[dict[str, object]] = []
    risk_rows: list[dict[str, object]] = []
    for portfolio_id in (FACTOR_PIT_PORTFOLIO, FACTOR_NONPIT_PORTFOLIO):
        reader = by_key[(portfolio_id, READER_SCHEMA)]
        attribution = by_key[(portfolio_id, ATTRIBUTION_SCHEMA)]
        _require_attribution_reconciles(attribution, reader, portfolio_id)
        metrics, artifact, artifact_sha = _validated_stream_metrics(
            factor_input, portfolio_id
        )
        meta = LineMetadata(
            portfolio_id=portfolio_id,
            label=f"Monthly returns ({portfolio_id})",
            window_label=str(reader["window_label"]),
            currency_basis=str(reader["currency_basis"]),
            total_return_basis=str(reader["return_basis"]),
            cash_benchmark_id=str(reader["cash_benchmark_id"]),
        )
        months = build_monthly_return_rows(
            meta,
            metrics,
            source=f"factor_run:{factor_input.run_id}/{artifact}#{artifact_sha}",
        )
        # reconcile: the months must recompound to the canonical reader row
        report_table([reader, *months])
        monthly_rows.extend(months)
        risk_rows.append(_risk_decomposition_row(attribution))

    tables = {
        "monthly_returns_ext2026": _ordered_report_table(monthly_rows),
        "risk_decomposition_ext2026": pd.DataFrame(
            risk_rows, columns=list(risk_decomposition_columns())
        ),
    }
    return AuxiliaryReportTables(
        owner=REPORT_TABLE_OWNER,
        lineage=dict(factor_tables.lineage),
        factor_tables=factor_tables,
        rows={
            "monthly_returns_ext2026": tuple(monthly_rows),
            "risk_decomposition_ext2026": tuple(risk_rows),
        },
        tables=tables,
    )


# --------------------------------------------------------------------------- #
# Tasks 9.3 / 9.4 / 9.5: canonical SJM, trio/static/dashboard, and Markowitz    #
# report tables. Every financial statistic comes from the shared               #
# macro_framework calculators routed through the reporting emission gate; the  #
# producers here select windows, verify lineage, and project — they own no     #
# duplicate finance formulas. The "tail" block of every reader row is the      #
# schema's downside/drawdown vocabulary (downside_rms, sortino, maxdd).       #
# --------------------------------------------------------------------------- #

#: Deterministic SSR inference defaults recorded with every produced table
#: (R7.4: the settings needed to interpret the output).
SSR_REPORT_DEFAULTS: Mapping[str, object] = MappingProxyType(
    {"window": 252, "sr_star": 0.0, "n_boot": 1000, "seed": 0, "alpha": 0.05}
)

#: Boundary-inclusive crisis window shared with the Factor family (task 6.7).
SJM_CRISIS_START = "2022-01-01"
SJM_CRISIS_END = "2022-12-31"

#: Canonical SJM tear-sheet table (task 9.3), pinned to the data-v4 catalog.
SJM_REPORT_TABLE_SCHEMAS: Mapping[str, str] = MappingProxyType(
    {"tear_sheet_sjm_crowding_ext2026": "tear_sheet.sjm.v3"}
)

#: Canonical trio, static-window, and window-dashboard tables (task 9.4).
TRIO_REPORT_TABLE_SCHEMAS: Mapping[str, str] = MappingProxyType(
    {
        "tear_sheet_trio_ext2026": "tear_sheet.trio.v4",
        "tear_sheet_static_bh_windows": "tear_sheet.static_windows.v1",
        "tear_sheet_static_bh_window_dashboard": "tear_sheet.window_dashboard.v1",
    }
)

#: Canonical ten-year and maximum-window Markowitz tables (task 9.5).
MARKOWITZ_REPORT_TABLE_SCHEMAS: Mapping[str, str] = MappingProxyType(
    {
        "markowitz_10y_moments": "markowitz.moments.v1",
        "markowitz_10y_frontier": "markowitz.frontier.v1",
        "markowitz_max_moments": "markowitz.moments.v1",
        "markowitz_max_frontier": "markowitz.frontier.v1",
        "tear_sheet_trio_10y": "tear_sheet.trio_10y.v1",
        "tear_sheet_trio_max": "tear_sheet.trio_max.v1",
    }
)

#: The two canonical Markowitz windows; table stems derive from these names.
MARKOWITZ_WINDOW_NAMES = ("10y", "max")

#: The USD Markowitz opportunity set (asset-only; strategy points are banned).
MARKOWITZ_ASSET_UNIVERSE = ("SWDA.L", "XLK", "IAU", "BIL")

#: The static buy-and-hold basket: 25% each, bought once, never rebalanced.
STATIC_BH_WEIGHT = 0.25
_STATIC_BH_LOCAL = ("SWDA.L", "XLK", "IAU")
STATIC_BH_CASH_SYMBOL = "BIL"

#: Locale projections of every canonical table are generated FROM these specs
#: (task 9.7 consumes them); the German source schema is producer-owned data.
REPORT_CSV_LOCALE_SPECS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "en-US": MappingProxyType(
            {"sep": ",", "decimal": ".", "float_format": "%.8f", "encoding": "utf-8"}
        ),
        "de-DE": MappingProxyType(
            {"sep": ";", "decimal": ",", "float_format": "%.8f", "encoding": "utf-8"}
        ),
    }
)
GERMAN_LOCALE_CSV_SPEC = REPORT_CSV_LOCALE_SPECS["de-DE"]


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_family(report_input: object, family: str) -> VerifiedReportInput:
    if not isinstance(report_input, VerifiedReportInput) or (
        report_input.family != family
    ):
        raise ValueError(
            f"canonical report input must be the verified {family!r} family "
            "(load it through the task-9.1 gate)"
        )
    return report_input


def _merged_ssr_settings(ssr_settings: Mapping[str, object] | None) -> dict[str, object]:
    merged = dict(SSR_REPORT_DEFAULTS)
    if ssr_settings is not None:
        unknown = sorted(set(ssr_settings) - set(merged))
        if unknown:
            raise ValueError(f"unknown SSR inference setting(s): {', '.join(unknown)}")
        merged.update(ssr_settings)
    return merged


def _read_inventoried_bytes(path: Path, entry: Mapping[str, object]) -> bytes:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry.get("sha256"):
        raise ValueError(f"{path}: bytes were mutated after inventory")
    return data


def _read_sjm_frame(
    run_dir: Path, manifest: Mapping[str, object], role: str
) -> pd.DataFrame:
    entry = manifest["files"][role]
    data = _read_inventoried_bytes(run_dir / entry["file"], entry)
    return pd.read_parquet(io.BytesIO(data))


def _read_snapshot_frame(
    snapshot_dir: Path, manifest: Mapping[str, object], filename: str
) -> pd.DataFrame:
    entry = manifest["files"][filename]
    data = _read_inventoried_bytes(snapshot_dir / filename, entry)
    return pd.read_parquet(io.BytesIO(data))


def _anchored_curve(returns: pd.Series, anchor: pd.Timestamp) -> pd.Series:
    """Value curve starting at exactly 1.0 on the anchor session (the SJM run
    validator's own reconstruction; mechanical, not a financial metric)."""
    return pd.Series(
        np.r_[1.0, np.cumprod(1.0 + returns.to_numpy(dtype=float))],
        index=returns.index.insert(0, anchor),
    )


def report_row_table_columns(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    """Deterministic canonical column order: provenance first, then sorted
    metric fields — the source schema both locale mirrors project."""
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    lead = [key for key in (*REQUIRED_PROVENANCE, "row_kind") if key in keys]
    rest = sorted(key for key in keys if key not in lead)
    return tuple(lead + rest)


def _ordered_report_table(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    table = report_table(list(rows))
    return table.reindex(columns=list(report_row_table_columns(list(rows))))


def parquet_safe_report_table(table: pd.DataFrame) -> pd.DataFrame:
    """Normalize mixed-schema date objects before a canonical Parquet write.

    Report tables intentionally combine reader, attribution, and crisis rows.
    Their sparse date fields therefore become object columns containing both
    ``Timestamp`` values and nulls. PyArrow cannot infer that mixed physical
    representation, so canonical persistence normalizes every date-bearing
    column to pandas ``datetime64[ns]`` without changing row values or order.
    """
    if not isinstance(table, pd.DataFrame):
        raise TypeError("canonical report table must be a pandas DataFrame")
    normalized = table.copy()
    for column in normalized.columns:
        if column in {"start", "end", "actual_end", "anchor", "first_return_date", "requested_start", "requested_end", "raw_market_model_start", "raw_market_model_end"} or column.endswith("_date"):
            normalized[column] = pd.to_datetime(normalized[column], errors="raise")
    return normalized


def _values_equal(left: object, right: object) -> bool:
    left_nan = isinstance(left, float) and math.isnan(left)
    right_nan = isinstance(right, float) and math.isnan(right)
    if left_nan or right_nan:
        return left_nan and right_nan
    return bool(left == right)


def _require_repeated_rows_agree(
    row_groups: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    """Repeated portfolio/window rows must agree on dates, count, cash
    benchmark, currency, annualization, and values (task 9.4, R7.2)."""
    seen: dict[tuple, tuple[str, Mapping[str, object]]] = {}
    for table_name, rows in row_groups.items():
        for row in rows:
            key = (row["portfolio_id"], row["schema"], row["window_label"])
            prior = seen.get(key)
            if prior is None:
                seen[key] = (table_name, row)
                continue
            prior_name, prior_row = prior
            if prior_row is row:
                continue
            if set(prior_row) != set(row) or any(
                not _values_equal(prior_row[field], row[field]) for field in prior_row
            ):
                raise ValueError(
                    f"repeated portfolio/window row {key!r} disagrees between "
                    f"{prior_name} and {table_name}"
                )


# --- Task 9.3: canonical SJM report tables ------------------------------------- #


def sjm_report_windows(
    coverage_start: pd.Timestamp,
    coverage_end: pd.Timestamp,
    *,
    dev_end: pd.Timestamp,
    holdout_start: pd.Timestamp | None = None,
) -> dict[str, dict[str, object]]:
    """Split one coverage span on the frozen protocol boundary.

    ``full`` always exists; ``development`` and ``holdout`` exist only where
    the boundary actually splits the sample. A window that equals the full
    span is marked ``{"coincides_with": "full"}`` and a window outside
    coverage is labeled absent with a reason — never fabricated (task 9.3).
    """
    start, end = pd.Timestamp(coverage_start), pd.Timestamp(coverage_end)
    if start > end:
        raise ValueError("coverage_start must be on or before coverage_end")
    boundary = pd.Timestamp(dev_end)
    cut = (
        pd.Timestamp(holdout_start)
        if holdout_start is not None
        else boundary + pd.Timedelta(days=1)
    )
    windows: dict[str, dict[str, object]] = {"full": {"start": start, "end": end}}
    if start > boundary:
        windows["development"] = {
            "available": False,
            "reason": f"coverage begins {start.date()} after dev_end {boundary.date()}",
        }
    elif end <= boundary:
        windows["development"] = {"coincides_with": "full"}
    else:
        windows["development"] = {"start": start, "end": boundary}
    if end < cut:
        windows["holdout"] = {
            "available": False,
            "reason": (
                f"coverage ends {end.date()} before holdout start {cut.date()}"
            ),
        }
    elif start >= cut:
        windows["holdout"] = {"coincides_with": "full"}
    else:
        windows["holdout"] = {"start": cut, "end": end}
    return windows


@dataclass(frozen=True)
class SJMReportTables:
    """Canonical SJM tear-sheet rows plus their complete provenance."""

    owner: str
    lineage: Mapping[str, object]
    protocol: Mapping[str, object]
    selected_config_sha256: str
    cash_benchmark: Mapping[str, object]
    coverage: Mapping[str, object]
    windows: Mapping[str, object]
    portfolios: Mapping[str, str]
    inference_settings: Mapping[str, object]
    rows: tuple[Mapping[str, object], ...]
    tables: Mapping[str, pd.DataFrame]


def _sjm_line_rows(
    *,
    portfolio_id: str,
    label: str,
    value: pd.Series,
    cash: pd.Series,
    total_return_basis: str,
    cash_benchmark_id: str,
    source: str,
    settings: Mapping[str, object],
    window_slices: Mapping[str, pd.Series],
    attribution,
    attribution_source: str,
    crisis_start: str,
    crisis_end: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Reader rows per window plus attribution and crisis records for one line."""
    rows: list[dict[str, object]] = []
    emitted: dict[str, object] = {}
    full_meta: LineMetadata | None = None
    for role, value_slice in window_slices.items():
        metrics = metric_block(value_slice)
        returns = metrics["returns"]
        cash_slice = cash.loc[returns.index]
        excess = portfolio_excess_returns(returns, cash_slice)
        ssr = ssr_inference(excess, **settings)
        window_label = (
            f"{role} {returns.index[0].date()}..{returns.index[-1].date()}"
        )
        meta = LineMetadata(
            portfolio_id=portfolio_id,
            label=label,
            window_label=window_label,
            currency_basis="legacy_mixed_local_quotes",
            total_return_basis=total_return_basis,
            cash_benchmark_id=cash_benchmark_id,
        )
        rows.append(
            build_reader_metric_row(
                meta,
                metrics,
                cash_slice,
                ssr,
                source=source,
                attribution=attribution if role == "full" else None,
            )
        )
        emitted[role] = {
            "label": window_label,
            "start": returns.index[0],
            "end": returns.index[-1],
            "n_obs": len(returns),
        }
        if role == "full":
            full_meta = meta
    assert full_meta is not None  # "full" is always in window_slices

    attribution_record = build_attribution_record(
        dataclasses.replace(
            full_meta,
            window_label=(
                f"attribution {attribution.start.date()}..{attribution.end.date()}"
            ),
        ),
        attribution,
        source=attribution_source,
    )
    crisis = crisis_metrics(value, crisis_start, crisis_end)
    if crisis is None:
        raise ValueError(
            f"{portfolio_id}: required crisis window {crisis_start}..{crisis_end} "
            "has no preceding anchor or no observations in this run"
        )
    crisis_record = build_crisis_record(
        dataclasses.replace(
            full_meta,
            window_label=f"crisis {crisis.anchor.date()}..{crisis.actual_end.date()}",
        ),
        crisis,
        source=source,
    )
    return rows, {
        "windows": emitted,
        "attribution_record": attribution_record,
        "crisis_record": crisis_record,
    }


def _sjm_window_slices(
    value: pd.Series, windows: Mapping[str, Mapping[str, object]], dev_end: pd.Timestamp
) -> dict[str, pd.Series]:
    """Anchor-preserving value slices for every emitted window role."""
    slices: dict[str, pd.Series] = {"full": value}
    development = windows["development"]
    if "start" in development:
        slices["development"] = value.loc[: development["end"]]
    holdout = windows["holdout"]
    if "start" in holdout:
        anchor = value.index[value.index <= dev_end][-1]
        slices["holdout"] = value.loc[anchor:]
    return slices


def build_sjm_report_tables(
    sjm_input: VerifiedReportInput,
    market_input: VerifiedReportInput,
    *,
    performance_start: str | pd.Timestamp | None = None,
    ssr_settings: Mapping[str, object] | None = None,
    crisis_start: str | pd.Timestamp = SJM_CRISIS_START,
    crisis_end: str | pd.Timestamp = SJM_CRISIS_END,
) -> SJMReportTables:
    """Canonical SJM performance, holdout, tail, SSR, crisis, and raw
    market-model rows from ONE completed SJM v3 run (task 9.3).

    Both inputs come through the task-9.1 manifest gate; the run's recorded
    market-snapshot lineage must be the very snapshot supplied here. Every
    persisted stream is re-read under its manifest inventory hash, and every
    displayed field reproduces from the shared macro_framework calculators —
    no notebook-local finance formulas. ``performance_start`` optionally
    selects one explicit later reporting window while preserving the last
    prior run level as its return anchor; it is used when a consumer requires
    a common Factor/SJM/static comparison window.
    """
    _require_family(sjm_input, "sjm_run")
    _require_family(market_input, "market_snapshot")
    settings = _merged_ssr_settings(ssr_settings)
    manifest = sjm_input.manifest
    lineage_snapshot = manifest["input_manifests"]["market_snapshot"]
    if (
        lineage_snapshot.get("snapshot_id"),
        lineage_snapshot.get("manifest_sha256"),
    ) != (market_input.identity, market_input.manifest_sha256):
        raise ValueError(
            "market-snapshot lineage diverges: the SJM run was built on "
            f"{lineage_snapshot.get('snapshot_id')!r} "
            f"({str(lineage_snapshot.get('manifest_sha256'))[:12]}...), refusing "
            "an unrelated snapshot as the canonical report input"
        )

    run_dir = sjm_input.run_dir
    returns_frame = _read_sjm_frame(run_dir, manifest, "daily_returns")
    control_frame = _read_sjm_frame(run_dir, manifest, "control_returns")
    equity_frame = _read_sjm_frame(run_dir, manifest, "equity")
    equity = equity_frame["value"]
    cash = returns_frame["cash_return"]
    control_value = _anchored_curve(
        control_frame["control_return"], equity.index[0]
    )
    if performance_start is not None:
        requested_start = pd.Timestamp(performance_start)
        eligible = equity.index[equity.index < requested_start]
        if eligible.empty or requested_start not in equity.index:
            raise ValueError(
                f"performance_start {requested_start.date()} requires a preceding "
                "SJM anchor and an exact persisted run observation"
            )
        anchor = eligible[-1]
        equity = equity.loc[anchor:]
        control_value = control_value.loc[anchor:]
        cash = cash.loc[equity.index[1:]]

    run_id = str(manifest["run_id"])
    snapshot_id = market_input.identity
    cash_benchmark_id = f"BIL@{snapshot_id}"
    dev_end = pd.Timestamp(manifest["protocol"]["dev_end"])
    windows = sjm_report_windows(
        equity.index[1],
        equity.index[-1],
        dev_end=dev_end,
    )

    ext = _factor_run_module()
    market_returns, market_lineage = ext.load_completed_snapshot_market_returns(
        market_input.run_dir, equity.index[1:], value_index=equity.index
    )
    market_source = (
        f"market_snapshot:{snapshot_id}/{market_lineage['benchmark_file']}"
        f"#SPY@{market_lineage['benchmark_file_sha256']}"
    )

    files = manifest["files"]
    overlay_source = (
        f"sjm_run:{run_id}/{files['equity']['file']}#{files['equity']['sha256']}"
    )
    control_source = (
        f"sjm_run:{run_id}/{files['control_returns']['file']}"
        f"#{files['control_returns']['sha256']}"
    )
    portfolios = {"overlay": run_id, "control": f"{run_id}_control"}

    rows: list[dict[str, object]] = []
    trailing: list[dict[str, object]] = []
    emitted_windows: Mapping[str, object] = {}
    for line_role, (value, basis, source, label) in {
        "overlay": (
            equity,
            "sjm_v3_overlay_anchored_equity",
            overlay_source,
            f"SJM v3 crowding de-risk overlay ({run_id})",
        ),
        "control": (
            control_value,
            "sjm_v3_control_anchored_equity",
            control_source,
            f"Correlation-overlay control ({run_id})",
        ),
    }.items():
        returns_full = metric_block(value)["returns"]
        attribution = raw_market_model_attribution(
            returns_full.loc[market_returns.index], market_returns
        )
        line_rows, extras = _sjm_line_rows(
            portfolio_id=portfolios[line_role],
            label=label,
            value=value,
            cash=cash,
            total_return_basis=basis,
            cash_benchmark_id=cash_benchmark_id,
            source=f"{source}|{market_source}",
            settings=settings,
            window_slices=_sjm_window_slices(value, windows, dev_end),
            attribution=attribution,
            attribution_source=f"{source}|{market_source}",
            crisis_start=crisis_start,
            crisis_end=crisis_end,
        )
        rows.extend(line_rows)
        trailing.extend([extras["attribution_record"], extras["crisis_record"]])
        if line_role == "overlay":
            emitted_windows = extras["windows"]
    rows.extend(trailing)

    result_windows: dict[str, object] = {}
    for role in ("full", "development", "holdout"):
        spec = windows[role]
        if "start" in spec and role in emitted_windows:
            result_windows[role] = emitted_windows[role]
        else:
            result_windows[role] = dict(spec)

    tables = {
        "tear_sheet_sjm_crowding_ext2026": _ordered_report_table(rows),
    }
    return SJMReportTables(
        owner=REPORT_TABLE_OWNER,
        lineage={
            "sjm_run": {
                "run_id": sjm_input.identity,
                "manifest_sha256": sjm_input.manifest_sha256,
            },
            "factor_run": dict(manifest["input_manifests"]["factor_run"]),
            "market_snapshot": dict(lineage_snapshot),
        },
        protocol=dict(manifest["protocol"]),
        selected_config_sha256=_canonical_json_sha256(manifest["selected_config"]),
        cash_benchmark=dict(manifest["cash_benchmark"]),
        coverage=dict(manifest["coverage"]),
        windows=result_windows,
        portfolios=portfolios,
        inference_settings={**settings, "periods_per_year": 252},
        rows=tuple(rows),
        tables=tables,
    )


# --- Task 9.4: canonical trio, static-window, and dashboard tables --------------- #


@dataclass(frozen=True)
class StaticWindowSpec:
    """One fresh-buy static buy-and-hold window (exact window identity)."""

    label: str
    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("static window label must be a non-empty string")
        object.__setattr__(self, "start", pd.Timestamp(self.start))
        object.__setattr__(self, "end", pd.Timestamp(self.end))
        if self.start > self.end:
            raise ValueError(f"static window {self.label!r}: start is after end")


def build_static_bh_rows(
    market_input: VerifiedReportInput,
    window: StaticWindowSpec,
    *,
    ssr_settings: Mapping[str, object] | None = None,
    attribution: bool = False,
) -> tuple[dict[str, object], dict[str, object] | None]:
    """One canonical fresh-buy static B&H reader row (plus its raw
    market-model record when requested) from the validated snapshot alone.

    Buy-and-hold holds SHARES: 25% of wealth buys each of SWDA.L/XLK/IAU/BIL
    at the first common trading day of the window and is never rebalanced.
    All statistics flow through ``metric_block`` and the shared row builders;
    Sharpe/Sortino/SSR use the corrected cash-excess convention (R1.1, R2.1).
    """
    _require_family(market_input, "market_snapshot")
    if not isinstance(window, StaticWindowSpec):
        raise TypeError("window must be a StaticWindowSpec")
    settings = _merged_ssr_settings(ssr_settings)
    manifest = market_input.manifest
    snapshot_dir = market_input.run_dir
    basket = _read_snapshot_frame(
        snapshot_dir, manifest, "basket_adjusted_close_local.parquet"
    )
    cash_market = _read_snapshot_frame(
        snapshot_dir, manifest, "cash_market_total_return.parquet"
    )
    missing = [c for c in _STATIC_BH_LOCAL if c not in basket.columns]
    if missing or STATIC_BH_CASH_SYMBOL not in cash_market.columns:
        raise ValueError(
            f"{snapshot_dir}: static basket requires columns "
            f"{[*_STATIC_BH_LOCAL, STATIC_BH_CASH_SYMBOL]}"
        )
    common = pd.concat(
        [basket[list(_STATIC_BH_LOCAL)], cash_market[[STATIC_BH_CASH_SYMBOL]]],
        axis=1,
    ).dropna()
    sliced = common.loc[window.start : window.end]
    if len(sliced) < 3:
        raise ValueError(
            f"static window {window.label!r}: no eligible common trading days "
            f"within snapshot coverage for {window.start.date()}.."
            f"{window.end.date()}"
        )
    value = STATIC_BH_WEIGHT * (sliced / sliced.iloc[0]).sum(axis=1)
    value.name = "value"

    metrics = metric_block(value)
    returns = metrics["returns"]
    cash_returns = sliced[STATIC_BH_CASH_SYMBOL].pct_change().dropna()
    excess = portfolio_excess_returns(returns, cash_returns)
    ssr = ssr_inference(excess, **settings)

    snapshot_id = market_input.identity
    basket_sha = manifest["files"]["basket_adjusted_close_local.parquet"]["sha256"]
    cash_sha = manifest["files"]["cash_market_total_return.parquet"]["sha256"]
    source = (
        f"{REPORT_TABLE_OWNER}:static_bh_25pct"
        f"|market_snapshot:{snapshot_id}/basket_adjusted_close_local.parquet"
        f"#{basket_sha}"
        f"|market_snapshot:{snapshot_id}/cash_market_total_return.parquet"
        f"#BIL@{cash_sha}"
    )
    meta = LineMetadata(
        portfolio_id=f"static_bh_25pct_{value.index[0].date()}",
        label=f"Static B&H 25% each ({window.label})",
        window_label=window.label,
        currency_basis="legacy_mixed_local_quotes",
        total_return_basis="adjusted_total_return_local_quotes_static_25pct",
        cash_benchmark_id=f"BIL@{snapshot_id}",
    )

    attribution_result = None
    attribution_record = None
    if attribution:
        ext = _factor_run_module()
        market_returns, market_lineage = ext.load_completed_snapshot_market_returns(
            snapshot_dir, returns.index, value_index=value.index
        )
        attribution_result = raw_market_model_attribution(
            returns.loc[market_returns.index], market_returns
        )
        attribution_record = build_attribution_record(
            meta,
            attribution_result,
            source=(
                f"{source}|market_snapshot:{snapshot_id}/"
                f"{market_lineage['benchmark_file']}"
                f"#SPY@{market_lineage['benchmark_file_sha256']}"
            ),
        )
    row = build_reader_metric_row(
        meta, metrics, cash_returns, ssr, source=source, attribution=attribution_result
    )
    return row, attribution_record


@dataclass(frozen=True)
class TrioReportTables:
    """Canonical trio, static-window, and dashboard tables plus lineage."""

    owner: str
    lineage: Mapping[str, object]
    windows: Mapping[str, object]
    inference_settings: Mapping[str, object]
    rows: Mapping[str, tuple[Mapping[str, object], ...]]
    tables: Mapping[str, pd.DataFrame]


def _require_component_rows_match_tables(
    rows: Sequence[Mapping[str, object]], tables: Mapping[str, pd.DataFrame]
) -> None:
    """A component's row list and its published tables must be the same data —
    a doctored component cannot smuggle divergent values into the trio."""
    for name, stored in tables.items():
        rebuilt = _ordered_report_table(rows)
        if not rebuilt.equals(stored):
            raise ValueError(
                f"component rows do not match its published table {name!r}"
            )


def build_trio_report_tables(
    factor_input: VerifiedFactorRun,
    sjm_reports: SJMReportTables,
    market_input: VerifiedReportInput,
    *,
    static_windows: Sequence[StaticWindowSpec],
    trio_static_window: StaticWindowSpec,
    ssr_settings: Mapping[str, object] | None = None,
) -> TrioReportTables:
    """Canonical trio, static buy-and-hold window, and window-dashboard tables
    with exact portfolio and window identity (task 9.4).

    The trio table's Factor and SJM rows ARE the validated producer rows —
    never a second recalculated family; the static rungs are fresh buys built
    here from the one lineage-bound snapshot. Repeated portfolio/window rows
    across every produced table must agree exactly (R7.2).
    """
    if not isinstance(factor_input, VerifiedFactorRun):
        raise TypeError("factor_input must come from load_factor_report_input")
    if not isinstance(sjm_reports, SJMReportTables):
        raise TypeError("sjm_reports must come from build_sjm_report_tables")
    _require_family(market_input, "market_snapshot")
    if not static_windows:
        raise ValueError("static_windows must name at least one ladder rung")

    market_pin = {
        "snapshot_id": market_input.identity,
        "manifest_sha256": market_input.manifest_sha256,
    }
    factor_snapshot = dict(
        factor_input.manifest["input_manifests"]["market_snapshot"]
    )
    if factor_snapshot != market_pin:
        raise ValueError(
            "factor-run market-snapshot lineage diverges from the supplied "
            "canonical snapshot input"
        )
    if dict(sjm_reports.lineage["market_snapshot"]) != market_pin:
        raise ValueError(
            "SJM market-snapshot lineage diverges from the supplied canonical "
            "snapshot input"
        )
    if dict(sjm_reports.lineage["factor_run"]) != {
        "run_id": factor_input.run_id,
        "manifest_sha256": factor_input.manifest_sha256,
    }:
        raise ValueError(
            "SJM factor-run lineage diverges from the supplied Factor bundle"
        )
    _require_component_rows_match_tables(sjm_reports.rows, sjm_reports.tables)

    # revalidates the record catalog and stream binding before any projection
    build_factor_report_tables(factor_input)
    records = factor_input.metric_records["records"]
    factor_reader = next(
        row
        for row in records
        if (row["portfolio_id"], row["schema"])
        == (FACTOR_PIT_PORTFOLIO, READER_SCHEMA)
    )
    overlay_id = sjm_reports.portfolios["overlay"]
    full_label = sjm_reports.windows["full"]["label"]
    overlay_row = next(
        row
        for row in sjm_reports.rows
        if row["schema"] == READER_SCHEMA
        and row["portfolio_id"] == overlay_id
        and row["window_label"] == full_label
    )

    def _spec_key(spec: StaticWindowSpec) -> tuple:
        return (spec.label, spec.start, spec.end)

    built: dict[tuple, tuple[dict[str, object], dict[str, object] | None]] = {}
    for spec in (*static_windows, trio_static_window):
        if not isinstance(spec, StaticWindowSpec):
            raise TypeError("static windows must be StaticWindowSpec instances")
        if _spec_key(spec) not in built:
            built[_spec_key(spec)] = build_static_bh_rows(
                market_input, spec, ssr_settings=ssr_settings, attribution=True
            )
    static_rows = [built[_spec_key(spec)][0] for spec in static_windows]
    attribution_records = [
        built[_spec_key(spec)][1]
        for spec in static_windows
        if built[_spec_key(spec)][1] is not None
    ]
    trio_static_row = built[_spec_key(trio_static_window)][0]

    trio_rows = [factor_reader, overlay_row, trio_static_row]
    if len({row["cash_benchmark_id"] for row in trio_rows}) != 1 or len(
        {row["currency_basis"] for row in trio_rows}
    ) != 1:
        raise ValueError(
            "trio rows must share one cash benchmark and one currency basis"
        )
    performance_signatures = {
        (
            pd.Timestamp(row["start"]),
            pd.Timestamp(row["end"]),
            int(row["n_obs"]),
            int(row["periods_per_year"]),
        )
        for row in trio_rows
    }
    if len(performance_signatures) != 1:
        raise ValueError(
            "trio rows must share one performance start/end/count/annualization "
            f"signature; found {sorted(performance_signatures, key=repr)!r}"
        )

    rows_map = {
        "tear_sheet_trio_ext2026": tuple(trio_rows),
        "tear_sheet_static_bh_windows": tuple(static_rows),
        "tear_sheet_static_bh_window_dashboard": tuple(
            static_rows + attribution_records
        ),
    }
    tables = {name: _ordered_report_table(rows) for name, rows in rows_map.items()}
    _require_repeated_rows_agree(
        {
            "factor_run.metric_records": records,
            "tear_sheet_sjm_crowding_ext2026": sjm_reports.rows,
            **{name: rows for name, rows in rows_map.items()},
        }
    )
    return TrioReportTables(
        owner=REPORT_TABLE_OWNER,
        lineage={
            "factor_run": {
                "run_id": factor_input.run_id,
                "manifest_sha256": factor_input.manifest_sha256,
            },
            "sjm_run": dict(sjm_reports.lineage["sjm_run"]),
            "market_snapshot": market_pin,
        },
        windows={
            "static_windows": tuple(
                {"label": spec.label, "start": spec.start, "end": spec.end}
                for spec in static_windows
            ),
            "trio_static": {
                "label": trio_static_window.label,
                "start": trio_static_window.start,
                "end": trio_static_window.end,
            },
        },
        inference_settings={**_merged_ssr_settings(ssr_settings), "periods_per_year": 252},
        rows=rows_map,
        tables=tables,
    )


# --- Task 9.4a: completed canonical trio report bundle ----------------------------- #


CANONICAL_REPORTS_SCHEMA = "canonical_reports.v1"
TRIO_REPORT_STEM = "tear_sheet_trio_ext2026"


@dataclass(frozen=True)
class CanonicalTrioReportBundle:
    """One immutable, completed canonical report directory for the trio table.

    The bundle is deliberately narrow: the report producer builds and inventories
    the canonical trio Parquet table; presentation notebooks only read it after
    checking the manifest, completion marker, table bytes, row count, and row
    contract.  This does not replay SJM selection or manufacture a new strategy
    run.
    """

    root: Path
    manifest: Mapping[str, object]
    table: pd.DataFrame


def _factor_pit_reader_row(factor_input: VerifiedFactorRun) -> Mapping[str, object]:
    """Return the one validated PIT reader record that defines trio timing."""
    factor_reports = build_factor_report_tables(factor_input)
    reader = factor_reports.tables["portfolio_metrics_reader_ext2026"]
    rows = reader[
        (reader["portfolio_id"] == FACTOR_PIT_PORTFOLIO)
        & (reader["schema"] == READER_SCHEMA)
    ]
    if len(rows) != 1:
        raise ValueError("completed Factor bundle must contain exactly one PIT reader row")
    return validate_report_row(rows.iloc[0].to_dict())


def _factor_window_static_spec(
    market_input: VerifiedReportInput,
    factor_row: Mapping[str, object],
) -> StaticWindowSpec:
    """Choose the prior common static level as the anchor for Factor timing.

    This selects an existing snapshot observation only; it leaves all portfolio
    construction and financial metrics to ``build_static_bh_rows``.
    """
    start = pd.Timestamp(factor_row["start"])
    end = pd.Timestamp(factor_row["end"])
    manifest = market_input.manifest
    basket = _read_snapshot_frame(
        market_input.run_dir, manifest, "basket_adjusted_close_local.parquet"
    )
    cash_market = _read_snapshot_frame(
        market_input.run_dir, manifest, "cash_market_total_return.parquet"
    )
    common = pd.concat(
        [basket[list(_STATIC_BH_LOCAL)], cash_market[[STATIC_BH_CASH_SYMBOL]]],
        axis=1,
    ).dropna()
    anchors = common.index[common.index < start]
    if start not in common.index or not len(anchors):
        raise ValueError(
            "Factor performance start cannot be represented by a static-basket "
            "common-calendar anchor in the verified market snapshot"
        )
    if end not in common.index:
        raise ValueError(
            "Factor performance end is absent from the static-basket common "
            "calendar in the verified market snapshot"
        )
    return StaticWindowSpec(
        label=f"Factor performance window (buy {anchors[-1].date()})",
        start=anchors[-1],
        end=end,
    )


def _canonical_trio_manifest(
    *,
    factor_input: VerifiedFactorRun,
    sjm_input: VerifiedReportInput,
    market_input: VerifiedReportInput,
    table_path: Path,
    table_relative_path: Path,
    table: pd.DataFrame,
    static_window: StaticWindowSpec,
) -> dict[str, object]:
    """Build the compact immutable inventory consumed by presentation notebooks."""
    return {
        "schema": CANONICAL_REPORTS_SCHEMA,
        "completed": True,
        "producer": REPORT_TABLE_OWNER,
        "input_manifests": {
            "factor_run": {
                "run_id": factor_input.run_id,
                "manifest_sha256": factor_input.manifest_sha256,
            },
            "sjm_run": {
                "run_id": sjm_input.identity,
                "manifest_sha256": sjm_input.manifest_sha256,
            },
            "market_snapshot": {
                "snapshot_id": market_input.identity,
                "manifest_sha256": market_input.manifest_sha256,
            },
        },
        "trio_static_window": {
            "label": static_window.label,
            "start": static_window.start.date().isoformat(),
            "end": static_window.end.date().isoformat(),
        },
        "tables": {
            TRIO_REPORT_STEM: {
                "file": str(table_relative_path.as_posix()),
                "schema": TRIO_REPORT_TABLE_SCHEMAS[TRIO_REPORT_STEM],
                "rows": int(len(table)),
                "sha256": _sha256_file(table_path),
            }
        },
    }


def materialize_canonical_trio_report_bundle(
    factor_input: VerifiedFactorRun,
    sjm_input: VerifiedReportInput,
    market_input: VerifiedReportInput,
    *,
    destination: Path | str,
    trio_static_window: StaticWindowSpec | None = None,
    ssr_settings: Mapping[str, object] | None = None,
) -> CanonicalTrioReportBundle:
    """Materialize a fresh, manifest-inventoried completed trio report bundle.

    All inputs must have passed their completed-manifest gates.  The Factor PIT
    reader record controls the common performance start; SJM is explicitly
    reanchored to that start before trio assembly.  The function writes only a
    new/empty report destination, inventories the persisted Parquet bytes, and
    writes ``COMPLETED`` last.  It never calls the SJM selection/build command.
    """
    if not isinstance(factor_input, VerifiedFactorRun):
        raise TypeError("factor_input must come from load_factor_report_input")
    _require_family(sjm_input, "sjm_run")
    _require_family(market_input, "market_snapshot")
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty canonical report root: {destination}")
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"canonical report destination is not a directory: {destination}")

    factor_row = _factor_pit_reader_row(factor_input)
    performance_start = pd.Timestamp(factor_row["start"])
    static_window = trio_static_window or _factor_window_static_spec(
        market_input, factor_row
    )
    sjm_reports = build_sjm_report_tables(
        sjm_input,
        market_input,
        performance_start=performance_start,
        ssr_settings=ssr_settings,
    )
    trio_reports = build_trio_report_tables(
        factor_input,
        sjm_reports,
        market_input,
        static_windows=(static_window,),
        trio_static_window=static_window,
        ssr_settings=ssr_settings,
    )
    table = parquet_safe_report_table(
        trio_reports.tables[TRIO_REPORT_STEM].copy()
    )
    rows = [validate_report_row(row) for row in table.to_dict(orient="records")]
    signatures = {
        (row["start"], row["end"], row["n_obs"], row["periods_per_year"])
        for row in rows
    }
    if len(rows) != 3 or len(signatures) != 1:
        raise ValueError("canonical trio output must contain three rows on one performance signature")
    if signatures.pop()[0] != performance_start:
        raise ValueError("canonical trio did not retain the Factor performance start")

    destination.mkdir(parents=True, exist_ok=True)
    table_path = destination / "tables" / f"{TRIO_REPORT_STEM}.parquet"
    table_path.parent.mkdir()
    table.to_parquet(table_path, index=False)
    manifest = _canonical_trio_manifest(
        factor_input=factor_input,
        sjm_input=sjm_input,
        market_input=market_input,
        table_path=table_path,
        table_relative_path=table_path.relative_to(destination),
        table=table,
        static_window=static_window,
    )
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (destination / "COMPLETED").write_text(
        f"manifest_sha256={_sha256_file(manifest_path)}\n"
    )
    return CanonicalTrioReportBundle(destination, manifest, table)


# --- Task 9.5: canonical ten-year and maximum-window Markowitz tables ------------ #


def _markowitz_module():
    """macro_framework.markowitz lazily (SciPy import at call time only)."""
    from macro_framework import markowitz

    return markowitz


def markowitz_quote_specs() -> dict[str, object]:
    """The frozen snapshot quote metadata for the USD opportunity set."""
    producer = _snapshot_module()
    markowitz = _markowitz_module()
    return {
        asset: markowitz.QuoteSpec(**producer.SNAPSHOT_QUOTES[asset])
        for asset in MARKOWITZ_ASSET_UNIVERSE
    }


_MARKOWITZ_IDENTITY_COLUMNS = (
    "window",
    "snapshot_id",
    "base_currency",
    "valuation_rule",
    "requested_start",
    "requested_end",
    "actual_start",
    "actual_end",
    "n_obs",
    "periods_per_year",
    "source_dates_sha256",
)


def markowitz_moments_columns(universe: Sequence[str]) -> tuple[str, ...]:
    """The canonical (and German-locale source) schema of a moments table."""
    return (
        *_MARKOWITZ_IDENTITY_COLUMNS,
        "asset",
        "quote_currency",
        "quote_unit",
        "mean_ann_arithmetic",
        "vol_ann",
        *(f"cov_{asset}" for asset in universe),
    )


def markowitz_frontier_columns(universe: Sequence[str]) -> tuple[str, ...]:
    """The canonical (and German-locale source) schema of a frontier table."""
    return (
        *_MARKOWITZ_IDENTITY_COLUMNS,
        "residual_tolerance",
        "target_return_ann",
        "success",
        "status",
        "message",
        "iterations",
        "objective",
        "budget_residual",
        "target_residual",
        "bound_violation",
        "return_ann",
        "volatility_ann",
        "feasible",
        *(f"weight_{asset}" for asset in universe),
    )


def _source_dates_sha256(valuations) -> str:
    """Seal the per-cutoff source-date provenance of one weekly window."""
    observed = {
        str(asset): [
            pd.Timestamp(value).date().isoformat()
            for value in valuations.observed_dates[asset]
        ]
        for asset in valuations.observed_dates.columns
    }
    fx_dates = [
        None if pd.isna(value) else pd.Timestamp(value).date().isoformat()
        for value in valuations.fx_observed_dates
    ]
    return _canonical_json_sha256(
        {
            "schema": "markowitz.source_dates.v1",
            "snapshot_id": valuations.snapshot_id,
            "observed_dates": observed,
            "fx_observed_dates": fx_dates,
        }
    )


def markowitz_max_supported_start(
    markowitz_input: VerifiedReportInput,
    *,
    requested_end: str | pd.Timestamp,
    quote_specs: Mapping[str, object] | None = None,
) -> pd.Timestamp:
    """First Friday for which every opportunity-set asset (and required FX)
    has an eligible observation — the maximum supported window start.

    Eligibility authority stays with ``weekly_usd_valuations``: the candidate
    derived from the youngest asset's first observation is verified by the
    shared validator itself, never by a private staleness rule.
    """
    _require_family(markowitz_input, "markowitz_inputs")
    specs = dict(quote_specs or markowitz_quote_specs())
    markowitz = _markowitz_module()
    manifest = markowitz_input.manifest
    snapshot_dir = markowitz_input.run_dir
    basket = _read_snapshot_frame(
        snapshot_dir, manifest, "basket_adjusted_close_local.parquet"
    )
    cash_market = _read_snapshot_frame(
        snapshot_dir, manifest, "cash_market_total_return.parquet"
    )
    fx = _read_snapshot_frame(snapshot_dir, manifest, "fx_usd_per_gbp.parquet")

    firsts: list[pd.Timestamp] = []
    for asset in specs:
        frame = basket if asset in basket.columns else cash_market
        if asset not in frame.columns:
            raise ValueError(f"{asset}: source column is absent from the snapshot")
        first = frame[asset].first_valid_index()
        if first is None:
            raise ValueError(f"{asset}: source column has no observations")
        firsts.append(pd.Timestamp(first))
    if any(spec.quote_currency == "GBP" for spec in specs.values()):
        first_fx = fx["USD_per_GBP"].first_valid_index()
        if first_fx is None:
            raise ValueError("USD_per_GBP: source column has no observations")
        firsts.append(pd.Timestamp(first_fx))

    candidate = max(firsts)
    friday = candidate + pd.Timedelta(days=(4 - candidate.dayofweek) % 7)
    end = pd.Timestamp(requested_end)
    for _ in range(8):
        try:
            markowitz.weekly_usd_valuations(
                snapshot_dir,
                quote_specs=specs,
                requested_start=friday,
                requested_end=end,
            )
        except ValueError:
            friday += pd.Timedelta(days=7)
            continue
        return friday
    raise ValueError(
        "no supported Friday start within 8 weeks of the youngest asset's "
        "first observation"
    )


@dataclass(frozen=True)
class MarkowitzReportTables:
    """Canonical USD weekly moment, frontier, and trio panel tables."""

    owner: str
    lineage: Mapping[str, object]
    base_currency: str
    valuation_rule: str
    windows: Mapping[str, object]
    rows: Mapping[str, tuple[Mapping[str, object], ...]]
    tables: Mapping[str, pd.DataFrame]


def build_markowitz_report_tables(
    markowitz_input: VerifiedReportInput,
    *,
    requested_windows: Mapping[str, tuple],
    trio_rows: Mapping[str, Sequence[Mapping[str, object]]],
    quote_specs: Mapping[str, object] | None = None,
    n_points: int = 60,
) -> MarkowitzReportTables:
    """Canonical ten-year and maximum-window Markowitz tables (task 9.5).

    Moments and frontiers come exclusively from ``macro_framework.markowitz``
    on the completed snapshot: one disclosed base currency (USD), requested
    AND actual windows, weekly counts, the exact 365.2425/7 annualization,
    sealed source-date hashes, per-asset moments, complete weight vectors,
    and full solver diagnostics — failed targets included. The USD frontier
    carries NO strategy points: trio panel rows are separate, validated,
    mixed-local report rows whose shorter coverage is labeled, never blended
    into the frontier tables.
    """
    _require_family(markowitz_input, "markowitz_inputs")
    if not isinstance(requested_windows, Mapping) or not requested_windows:
        raise ValueError("requested_windows must name at least one window")
    unknown = sorted(set(requested_windows) - set(MARKOWITZ_WINDOW_NAMES))
    if unknown:
        raise ValueError(
            f"unknown Markowitz window name(s) {unknown}; the canonical windows "
            f"are {MARKOWITZ_WINDOW_NAMES!r}"
        )
    if not isinstance(trio_rows, Mapping) or set(trio_rows) != set(requested_windows):
        raise ValueError(
            "trio_rows must supply trio panel rows for exactly the requested windows"
        )
    specs = dict(quote_specs or markowitz_quote_specs())
    universe = tuple(specs)
    markowitz = _markowitz_module()

    tables: dict[str, pd.DataFrame] = {}
    rows_map: dict[str, tuple] = {}
    windows_meta: dict[str, object] = {}
    for name, bounds in requested_windows.items():
        start, end = (pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1]))
        valuations = markowitz.weekly_usd_valuations(
            markowitz_input.run_dir,
            quote_specs=specs,
            requested_start=start,
            requested_end=end,
        )
        moments = markowitz.annualized_moments(valuations)
        frontier = markowitz.efficient_frontier(moments, n_points=n_points)
        source_hash = _source_dates_sha256(valuations)
        identity = {
            "window": name,
            "snapshot_id": moments.snapshot_id,
            "base_currency": "USD",
            "valuation_rule": moments.valuation_rule,
            "requested_start": start,
            "requested_end": end,
            "actual_start": moments.start,
            "actual_end": moments.end,
            "n_obs": moments.n_obs,
            "periods_per_year": moments.periods_per_year,
            "source_dates_sha256": source_hash,
        }

        moment_rows = []
        for asset in universe:
            row = dict(identity)
            row["asset"] = asset
            row["quote_currency"] = specs[asset].quote_currency
            row["quote_unit"] = specs[asset].quote_unit
            row["mean_ann_arithmetic"] = float(moments.mean_ann_arithmetic[asset])
            row["vol_ann"] = float(
                np.sqrt(moments.covariance_ann.loc[asset, asset])
            )
            for other in universe:
                row[f"cov_{other}"] = float(moments.covariance_ann.loc[asset, other])
            moment_rows.append(row)
        tables[f"markowitz_{name}_moments"] = pd.DataFrame(
            moment_rows, columns=list(markowitz_moments_columns(universe))
        )

        frontier_rows = []
        for point in frontier.points:
            row = dict(identity)
            row["residual_tolerance"] = frontier.residual_tolerance
            for field in (
                "target_return_ann",
                "success",
                "status",
                "message",
                "iterations",
                "objective",
                "budget_residual",
                "target_residual",
                "bound_violation",
                "return_ann",
                "volatility_ann",
                "feasible",
            ):
                row[field] = getattr(point, field)
            for asset in universe:
                row[f"weight_{asset}"] = float(point.weights[asset])
            frontier_rows.append(row)
        tables[f"markowitz_{name}_frontier"] = pd.DataFrame(
            frontier_rows, columns=list(markowitz_frontier_columns(universe))
        )

        panel_rows = [validate_report_row(row) for row in trio_rows[name]]
        if not panel_rows:
            raise ValueError(f"window {name!r}: trio panel rows must be non-empty")
        coverage: dict[str, object] = {}
        for row in panel_rows:
            if row["currency_basis"] != "legacy_mixed_local_quotes":
                raise ValueError(
                    "the USD frontier carries no strategy points: trio panel "
                    "rows stay on the mixed-local basis "
                    "('legacy_mixed_local_quotes') until strategies are rebuilt "
                    "in USD; a row claiming another basis cannot join the panel"
                )
            row_start = pd.Timestamp(row["start"])
            row_end = pd.Timestamp(row["end"])
            if row_start < start or row_end > end:
                raise ValueError(
                    f"trio panel row {row['portfolio_id']!r} claims "
                    f"{row_start.date()}..{row_end.date()} beyond the requested "
                    f"window {start.date()}..{end.date()}"
                )
            coverage[str(row["portfolio_id"])] = {
                "start": row_start,
                "end": row_end,
                "n_obs": int(row["n_obs"]),
                "coverage": (
                    "spans_requested_start"
                    if row_start - start <= pd.Timedelta(days=7)
                    else "shorter_than_requested"
                ),
            }
        tables[f"tear_sheet_trio_{name}"] = _ordered_report_table(panel_rows)
        rows_map[f"tear_sheet_trio_{name}"] = tuple(panel_rows)
        windows_meta[name] = {
            **identity,
            "cutoff_start": valuations.start,
            "cutoff_end": valuations.end,
            "n_feasible": frontier.n_feasible,
            "n_targets": frontier.n_targets,
            "trio_coverage": coverage,
        }

    return MarkowitzReportTables(
        owner=REPORT_TABLE_OWNER,
        lineage={
            "market_snapshot": {
                "snapshot_id": markowitz_input.identity,
                "manifest_sha256": markowitz_input.manifest_sha256,
            }
        },
        base_currency="USD",
        valuation_rule=str(next(iter(windows_meta.values()))["valuation_rule"]),
        windows=windows_meta,
        rows=rows_map,
        tables=tables,
    )


# --- Complete immutable canonical report bundles ---------------------------------- #
#
# ``materialize_canonical_trio_report_bundle`` above was deliberately introduced as
# a narrow migration bridge.  The presentation notebooks need a *family* of tables,
# however: publishing only its trio leaves the Factor, SJM, static-window, dashboard,
# and Markowitz consumers to fall back to stale local files.  This boundary owns the
# complete family as one immutable report root.  It is intentionally not the
# data-v4 publisher: that later stage flattens a validated report bundle into the
# frozen release catalog.


CANONICAL_REPORT_TABLE_SCHEMAS: Mapping[str, str] = MappingProxyType(
    {
        **FACTOR_REPORT_TABLE_SCHEMAS,
        **AUXILIARY_REPORT_TABLE_SCHEMAS,
        **SJM_REPORT_TABLE_SCHEMAS,
        **TRIO_REPORT_TABLE_SCHEMAS,
        **MARKOWITZ_REPORT_TABLE_SCHEMAS,
    }
)

# These are the explicit static-window rungs consumed by notebooks 15.2, 15.3, and
# 18.2.  The common trio static comparison is deliberately selected separately from
# this ladder so it exactly shares the Factor performance window.
DEFAULT_STATIC_WINDOW_SPECS: tuple[StaticWindowSpec, ...] = (
    StaticWindowSpec(
        "Full 16.7y (buy 2009)", pd.Timestamp("2009-09-25"), pd.Timestamp("2026-05-29")
    ),
    StaticWindowSpec(
        "16.4y (buy 2009)", pd.Timestamp("2009-09-25"), pd.Timestamp("2026-01-30")
    ),
    StaticWindowSpec(
        "10.0y (buy 2016)", pd.Timestamp("2016-02-01"), pd.Timestamp("2026-01-30")
    ),
    StaticWindowSpec(
        "7.1y (buy 2019)", pd.Timestamp("2019-01-02"), pd.Timestamp("2026-01-30")
    ),
)
MARKOWITZ_10Y_START = pd.Timestamp("2016-02-01")
MARKOWITZ_REPORT_END = pd.Timestamp("2026-01-30")


@dataclass(frozen=True)
class CanonicalReportBundle:
    """One complete, immutable, manifest-verified canonical report family."""

    root: Path
    manifest: Mapping[str, object]
    tables: Mapping[str, pd.DataFrame]


def _canonical_report_input_manifests(
    factor_input: VerifiedFactorRun,
    sjm_input: VerifiedReportInput,
    market_input: VerifiedReportInput,
) -> dict[str, dict[str, str]]:
    """The three exact completed producer identities a report bundle binds."""
    return {
        "factor_run": {
            "run_id": factor_input.run_id,
            "manifest_sha256": factor_input.manifest_sha256,
        },
        "sjm_run": {
            "run_id": sjm_input.identity,
            "manifest_sha256": sjm_input.manifest_sha256,
        },
        "market_snapshot": {
            "snapshot_id": market_input.identity,
            "manifest_sha256": market_input.manifest_sha256,
        },
    }


def _safe_report_relative_path(value: object, *, directory: str) -> Path:
    """Validate one manifest-owned path below a fixed report subdirectory."""
    if not isinstance(value, str) or not value:
        raise ValueError("canonical report inventory file must be a non-empty relative path")
    path = Path(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or len(path.parts) != 2
        or path.parts[0] != directory
        or path.name in {"", ".", ".."}
    ):
        raise ValueError(
            f"canonical report inventory file must be a safe {directory}/ relative path: {value!r}"
        )
    return path


def _require_new_report_destination(destination: Path) -> None:
    """Allow an absent/empty directory only; incomplete and completed roots stay immutable."""
    if not destination.exists():
        return
    if not destination.is_dir():
        raise ValueError(f"canonical report destination is not a directory: {destination}")
    if any(destination.iterdir()):
        raise ValueError(
            f"refusing to overwrite non-empty canonical report root: {destination}"
        )


def _report_window_value_slice(
    value: pd.Series,
    *,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
    earliest_start: pd.Timestamp | None = None,
    label: str,
) -> pd.Series:
    """Select a strictly anchored, honest strategy slice for a panel table.

    A strategy which began after a requested Markowitz window is represented by
    its actual shorter coverage; it is never back-filled.  Conversely, a stale
    endpoint is not silently extended.  The selected value series retains one
    actual preceding anchor for ``metric_block`` and strict cash alignment.
    """
    if not isinstance(value, pd.Series) or value.empty:
        raise ValueError(f"{label}: value stream must be non-empty")
    value = value.copy()
    value.index = pd.DatetimeIndex(value.index)
    if (
        value.index.has_duplicates
        or not value.index.is_monotonic_increasing
        or value.index.tz is not None
        or not np.isfinite(value.to_numpy(dtype=float)).all()
    ):
        raise ValueError(f"{label}: persisted value stream is not finite, unique, ordered, and timezone-naive")
    if requested_start > requested_end:
        raise ValueError(f"{label}: requested start is after requested end")
    effective_start = max(
        pd.Timestamp(requested_start),
        pd.Timestamp(earliest_start) if earliest_start is not None else value.index[1],
    )
    return_dates = value.index[(value.index >= effective_start) & (value.index <= requested_end)]
    if len(return_dates) < 2:
        raise ValueError(
            f"{label}: fewer than two persisted return observations are available "
            f"for {effective_start.date()}..{pd.Timestamp(requested_end).date()}"
        )
    first_return, last_return = return_dates[0], return_dates[-1]
    anchors = value.index[value.index < first_return]
    if not len(anchors):
        raise ValueError(f"{label}: no persisted value anchor precedes {first_return.date()}")
    return value.loc[anchors[-1] : last_return]


def _windowed_reader_row(
    *,
    value: pd.Series,
    market_input: VerifiedReportInput,
    portfolio_id: str,
    label: str,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
    earliest_start: pd.Timestamp | None,
    total_return_basis: str,
    source: str,
    ssr_settings: Mapping[str, object] | None,
) -> dict[str, object]:
    """Reader row for a disclosed shorter strategy panel, from verified bytes.

    This is solely for the 10-year and maximum-window *panel* rows.  The primary
    Factor and SJM tables remain projections of their run-local records.  Each
    panel row is reconstructed through the shared metric, BIL-excess, and raw
    market-model contracts from a manifest-inventoried curve and snapshot.
    """
    panel_value = _report_window_value_slice(
        value,
        requested_start=requested_start,
        requested_end=requested_end,
        earliest_start=earliest_start,
        label=portfolio_id,
    )
    metrics = metric_block(panel_value)
    returns = metrics["returns"]
    ext = _factor_run_module()
    cash, cash_lineage = ext.load_completed_snapshot_bil_returns(
        market_input.run_dir, returns.index, anchor=panel_value.index[0]
    )
    market, market_lineage = ext.load_completed_snapshot_market_returns(
        market_input.run_dir, returns.index, value_index=panel_value.index
    )
    attribution = raw_market_model_attribution(returns.loc[market.index], market)
    ssr = ssr_inference(
        portfolio_excess_returns(returns, cash), **_merged_ssr_settings(ssr_settings)
    )
    window_label = (
        f"Markowitz panel requested {pd.Timestamp(requested_start).date()}.."
        f"{pd.Timestamp(requested_end).date()}; strategy "
        f"{returns.index[0].date()}..{returns.index[-1].date()}"
    )
    meta = LineMetadata(
        portfolio_id=portfolio_id,
        label=label,
        window_label=window_label,
        currency_basis="legacy_mixed_local_quotes",
        total_return_basis=total_return_basis,
        cash_benchmark_id=f"BIL@{market_input.identity}",
    )
    lineage = (
        f"{source}|market_snapshot:{market_input.identity}/{cash_lineage['cash_file']}"
        f"#BIL@{cash_lineage['cash_file_sha256']}"
        f"|market_snapshot:{market_input.identity}/{market_lineage['benchmark_file']}"
        f"#SPY@{market_lineage['benchmark_file_sha256']}"
    )
    return build_reader_metric_row(
        meta, metrics, cash, ssr, source=lineage, attribution=attribution
    )


def _factor_panel_reader_row(
    factor_input: VerifiedFactorRun,
    market_input: VerifiedReportInput,
    *,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
    ssr_settings: Mapping[str, object] | None,
) -> dict[str, object]:
    """One shorter disclosed Factor PIT panel row from its inventoried equity."""
    stream = factor_input.metric_records["source_streams"][FACTOR_PIT_PORTFOLIO]
    artifact = str(stream["artifact"])
    entry = _factor_inventory_entry(factor_input.manifest, artifact)
    value = pd.read_parquet(
        io.BytesIO(_read_inventoried_bytes(factor_input.run_dir / artifact, entry))
    )["value"]
    return _windowed_reader_row(
        value=value,
        market_input=market_input,
        portfolio_id=FACTOR_PIT_PORTFOLIO,
        label="AI macro-factor (PIT) Markowitz panel",
        requested_start=requested_start,
        requested_end=requested_end,
        earliest_start=pd.Timestamp(stream["start"]),
        total_return_basis="factor_pit_anchored_equity",
        source=(
            f"factor_run:{factor_input.run_id}/{artifact}#{entry['sha256']}"
        ),
        ssr_settings=ssr_settings,
    )


def _sjm_panel_reader_row(
    sjm_input: VerifiedReportInput,
    market_input: VerifiedReportInput,
    *,
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
    ssr_settings: Mapping[str, object] | None,
) -> dict[str, object]:
    """One shorter disclosed SJM overlay panel row from its inventoried equity."""
    manifest = sjm_input.manifest
    equity_entry = manifest["files"]["equity"]
    value = _read_sjm_frame(sjm_input.run_dir, manifest, "equity")["value"]
    return _windowed_reader_row(
        value=value,
        market_input=market_input,
        portfolio_id=sjm_input.identity,
        label=f"SJM v3 crowding de-risk overlay ({sjm_input.identity}) Markowitz panel",
        requested_start=requested_start,
        requested_end=requested_end,
        earliest_start=pd.Timestamp(manifest["coverage"]["start"]),
        total_return_basis="sjm_v3_overlay_anchored_equity",
        source=(
            f"sjm_run:{sjm_input.identity}/{equity_entry['file']}#{equity_entry['sha256']}"
        ),
        ssr_settings=ssr_settings,
    )


def _panel_rows_for_markowitz_window(
    factor_input: VerifiedFactorRun,
    sjm_input: VerifiedReportInput,
    market_input: VerifiedReportInput,
    *,
    name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    ssr_settings: Mapping[str, object] | None,
) -> tuple[dict[str, object], ...]:
    """Static, Factor, and SJM rows for one asset-only Markowitz companion panel."""
    static, _ = build_static_bh_rows(
        market_input,
        StaticWindowSpec(f"{name} Markowitz static panel", start, end),
        ssr_settings=ssr_settings,
        attribution=False,
    )
    factor = _factor_panel_reader_row(
        factor_input,
        market_input,
        requested_start=start,
        requested_end=end,
        ssr_settings=ssr_settings,
    )
    sjm = _sjm_panel_reader_row(
        sjm_input,
        market_input,
        requested_start=start,
        requested_end=end,
        ssr_settings=ssr_settings,
    )
    return (static, factor, sjm)


def _report_manifest_entry(
    *, path: Path, relative: Path, schema: str, rows: int
) -> dict[str, object]:
    return {
        "file": relative.as_posix(),
        "schema": schema,
        "rows": int(rows),
        "sha256": _sha256_file(path),
    }


def _canonical_report_manifest(
    *,
    report_id: str,
    inputs: Mapping[str, Mapping[str, str]],
    tables: Mapping[str, pd.DataFrame],
    root: Path,
    configuration: Mapping[str, object],
) -> dict[str, object]:
    """Build the final finite JSON inventory after all bytes are on disk."""
    table_inventory: dict[str, dict[str, object]] = {}
    mirror_inventory: dict[str, dict[str, object]] = {}
    for stem in sorted(CANONICAL_REPORT_TABLE_SCHEMAS):
        table = tables[stem]
        table_rel = Path("tables") / f"{stem}.parquet"
        table_inventory[stem] = _report_manifest_entry(
            path=root / table_rel,
            relative=table_rel,
            schema=CANONICAL_REPORT_TABLE_SCHEMAS[stem],
            rows=len(table),
        )
        for locale, suffix in (("en-US", ".csv"), ("de-DE", "_de.csv")):
            mirror_rel = Path("mirrors") / f"{stem}{suffix}"
            mirror_inventory[mirror_rel.name] = {
                **_report_manifest_entry(
                    path=root / mirror_rel,
                    relative=mirror_rel,
                    schema=CANONICAL_REPORT_TABLE_SCHEMAS[stem],
                    rows=len(table),
                ),
                "locale": locale,
                "source_table": stem,
            }
    return {
        "schema": CANONICAL_REPORTS_SCHEMA,
        "completed": True,
        "producer": REPORT_TABLE_OWNER,
        "report_id": report_id,
        "input_manifests": {key: dict(value) for key, value in sorted(inputs.items())},
        "configuration": dict(configuration),
        "tables": table_inventory,
        "mirrors": mirror_inventory,
    }


def _validate_report_manifest_inputs(
    manifest: Mapping[str, object],
    *,
    factor_input: VerifiedFactorRun | None,
    sjm_input: VerifiedReportInput | None,
    market_input: VerifiedReportInput | None,
) -> None:
    inputs = manifest.get("input_manifests")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "factor_run",
        "sjm_run",
        "market_snapshot",
    }:
        raise ValueError("canonical report manifest must pin Factor, SJM, and market inputs")
    expected = None
    provided = (factor_input, sjm_input, market_input)
    if any(item is not None for item in provided):
        if not all(item is not None for item in provided):
            raise ValueError("canonical report input validation requires Factor, SJM, and market inputs together")
        assert factor_input is not None and sjm_input is not None and market_input is not None
        expected = _canonical_report_input_manifests(
            factor_input, sjm_input, market_input
        )
    for role, identity_key in (
        ("factor_run", "run_id"),
        ("sjm_run", "run_id"),
        ("market_snapshot", "snapshot_id"),
    ):
        entry = inputs[role]
        if not isinstance(entry, Mapping):
            raise ValueError(f"canonical report input {role!r} must be an object")
        _pinned_identity(f"input_manifests.{role}.{identity_key}", entry.get(identity_key))
        _pinned_sha256(
            f"input_manifests.{role}.manifest_sha256", entry.get("manifest_sha256")
        )
        if expected is not None and dict(entry) != expected[role]:
            raise ValueError(
                f"canonical report input {role!r} diverges from the validated producer manifest"
            )


def _validate_report_table_rows(table: pd.DataFrame, *, stem: str) -> None:
    """Run the row gate for every report-schema row persisted in a table."""
    if table.empty or not isinstance(table.index, pd.RangeIndex):
        raise ValueError(f"{stem}: canonical table must be non-empty and flat")
    if "schema" not in table.columns:
        return  # Markowitz moments/frontiers carry their own fixed tabular schemas.
    known_report_schemas = {
        READER_SCHEMA,
        LEGACY_SCHEMA,
        DIFFERENTIAL_SCHEMA,
        ATTRIBUTION_SCHEMA,
        CRISIS_SCHEMA,
        MONTHLY_SCHEMA,
    }
    for row in table.to_dict(orient="records"):
        if row.get("schema") in known_report_schemas:
            # Mixed report-schema DataFrames use NaN padding for fields that do
            # not belong to a given row schema.  The row contract is expressed
            # over the row's present fields, not over DataFrame storage padding.
            # Keep zero/False/empty text intact — only scalar missing values are
            # projections to omit before re-running the emission gate.
            unpadded = {
                key: value
                for key, value in row.items()
                if value is not None
                and not (
                    not isinstance(value, (str, bytes))
                    and bool(pd.isna(value))
                )
            }
            validate_report_row(unpadded)


def validate_canonical_report_bundle(
    root: Path | str,
    *,
    factor_input: VerifiedFactorRun | None = None,
    sjm_input: VerifiedReportInput | None = None,
    market_input: VerifiedReportInput | None = None,
) -> Mapping[str, object]:
    """Read-only integrity verification for a complete report family.

    The verifier rejects incomplete roots, stale/mutated tables or mirrors,
    traversal-shaped inventory paths, a marker that does not bind the final
    manifest, and any extra output file.  Passing the three validated producer
    inputs additionally proves the report's exact Factor/SJM/market lineage.
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"canonical report root is absent or not a directory: {root}")
    manifest_path = root / "manifest.json"
    marker_path = root / "COMPLETED"
    if not manifest_path.is_file() or not marker_path.is_file():
        raise ValueError("canonical report root is incomplete: manifest.json and COMPLETED are required")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest_path}: canonical report manifest is invalid JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("canonical report manifest must be a JSON object")
    if manifest.get("schema") != CANONICAL_REPORTS_SCHEMA:
        raise ValueError(
            f"canonical report manifest schema must be {CANONICAL_REPORTS_SCHEMA!r}"
        )
    if manifest.get("completed") is not True:
        raise ValueError("canonical report manifest must declare completed=true")
    if manifest.get("producer") != REPORT_TABLE_OWNER:
        raise ValueError("canonical report manifest producer is not the report-table owner")
    _pinned_identity("report_id", manifest.get("report_id"))
    _validate_report_manifest_inputs(
        manifest,
        factor_input=factor_input,
        sjm_input=sjm_input,
        market_input=market_input,
    )

    expected_stems = set(CANONICAL_REPORT_TABLE_SCHEMAS)
    tables = manifest.get("tables")
    if not isinstance(tables, Mapping) or set(tables) != expected_stems:
        raise ValueError(
            "canonical report manifest table inventory must contain the complete required report family"
        )
    mirrors = manifest.get("mirrors")
    expected_mirror_names = {
        f"{stem}{suffix}"
        for stem in expected_stems
        for suffix in (".csv", "_de.csv")
    }
    if not isinstance(mirrors, Mapping) or set(mirrors) != expected_mirror_names:
        raise ValueError(
            "canonical report manifest mirror inventory must contain both locale files for every table"
        )

    mirror_exporter = None
    loaded_tables: dict[str, pd.DataFrame] = {}
    expected_table_files: set[Path] = set()
    expected_mirror_files: set[Path] = set()
    for stem in sorted(expected_stems):
        entry = tables[stem]
        if not isinstance(entry, Mapping):
            raise ValueError(f"{stem}: table inventory entry must be an object")
        relative = _safe_report_relative_path(entry.get("file"), directory="tables")
        expected_relative = Path("tables") / f"{stem}.parquet"
        if relative != expected_relative:
            raise ValueError(f"{stem}: table inventory path must be {expected_relative.as_posix()!r}")
        if entry.get("schema") != CANONICAL_REPORT_TABLE_SCHEMAS[stem]:
            raise ValueError(f"{stem}: table inventory schema diverges from the canonical contract")
        rows = entry.get("rows")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
            raise ValueError(f"{stem}: table inventory rows must be a positive integer")
        digest = entry.get("sha256")
        _pinned_sha256(f"tables.{stem}.sha256", digest)
        path = root / relative
        if not path.is_file() or _sha256_file(path) != digest:
            raise ValueError(f"{stem}: canonical table is missing or mutated after inventory")
        table = pd.read_parquet(path)
        if len(table) != rows:
            raise ValueError(f"{stem}: table row count diverges from its inventory")
        _validate_report_table_rows(table, stem=stem)
        loaded_tables[stem] = table
        expected_table_files.add(relative)

        for locale, suffix in (("en-US", ".csv"), ("de-DE", "_de.csv")):
            name = f"{stem}{suffix}"
            mirror = mirrors[name]
            if not isinstance(mirror, Mapping):
                raise ValueError(f"{name}: mirror inventory entry must be an object")
            relative = _safe_report_relative_path(mirror.get("file"), directory="mirrors")
            expected_relative = Path("mirrors") / name
            if relative != expected_relative:
                raise ValueError(f"{name}: mirror inventory path is not canonical")
            if (
                mirror.get("schema") != CANONICAL_REPORT_TABLE_SCHEMAS[stem]
                or mirror.get("source_table") != stem
                or mirror.get("locale") != locale
                or mirror.get("rows") != rows
            ):
                raise ValueError(f"{name}: mirror inventory metadata diverges from its canonical table")
            digest = mirror.get("sha256")
            _pinned_sha256(f"mirrors.{name}.sha256", digest)
            path = root / relative
            if not path.is_file() or _sha256_file(path) != digest:
                raise ValueError(f"{name}: locale mirror is missing or mutated after inventory")
            if mirror_exporter is None:
                try:
                    from scripts import export_csv_mirrors as mirror_exporter
                except ImportError:
                    import export_csv_mirrors as mirror_exporter
            mirror_exporter.verify_mirror_round_trip(
                table, path, locale=locale
            )
            expected_mirror_files.add(relative)

    table_files = {
        Path("tables") / path.name
        for path in (root / "tables").iterdir()
        if path.is_file()
    } if (root / "tables").is_dir() else set()
    mirror_files = {
        Path("mirrors") / path.name
        for path in (root / "mirrors").iterdir()
        if path.is_file()
    } if (root / "mirrors").is_dir() else set()
    if table_files != expected_table_files or mirror_files != expected_mirror_files:
        raise ValueError("canonical report root contains missing or unmanifested table/mirror output")
    expected_root_names = {"manifest.json", "COMPLETED", "tables", "mirrors"}
    actual_root_names = {path.name for path in root.iterdir()}
    if actual_root_names != expected_root_names:
        raise ValueError("canonical report root contains an unmanifested or missing top-level output")
    marker_lines = marker_path.read_text().splitlines()
    expected_marker = f"manifest_sha256={_sha256_file(manifest_path)}"
    if marker_lines != [expected_marker]:
        raise ValueError("canonical report COMPLETED marker does not bind the final manifest")
    return manifest


def load_completed_canonical_report_bundle(
    root: Path | str,
    *,
    factor_input: VerifiedFactorRun | None = None,
    sjm_input: VerifiedReportInput | None = None,
    market_input: VerifiedReportInput | None = None,
) -> CanonicalReportBundle:
    """Validate and load every table in a completed canonical report root."""
    root = Path(root)
    manifest = validate_canonical_report_bundle(
        root,
        factor_input=factor_input,
        sjm_input=sjm_input,
        market_input=market_input,
    )
    tables = {
        stem: pd.read_parquet(root / str(entry["file"]))
        for stem, entry in sorted(manifest["tables"].items())
    }
    return CanonicalReportBundle(root=root, manifest=manifest, tables=tables)


def materialize_canonical_report_bundle(
    factor_input: VerifiedFactorRun,
    sjm_input: VerifiedReportInput,
    market_input: VerifiedReportInput,
    *,
    destination: Path | str,
    static_windows: Sequence[StaticWindowSpec] = DEFAULT_STATIC_WINDOW_SPECS,
    trio_static_window: StaticWindowSpec | None = None,
    markowitz_10y_start: str | pd.Timestamp = MARKOWITZ_10Y_START,
    markowitz_end: str | pd.Timestamp = MARKOWITZ_REPORT_END,
    markowitz_n_points: int = 60,
    ssr_settings: Mapping[str, object] | None = None,
) -> CanonicalReportBundle:
    """Materialize the complete current report family from validated inputs only.

    No old report table is read or copied and no network acquisition occurs.
    The destination must be absent or empty.  All canonical Parquet tables and
    locale mirrors are fully written and verified before the final manifest is
    emitted; ``COMPLETED`` is the final filesystem mutation.  Failure therefore
    leaves a diagnosable but non-consumable root, and any retry needs a new
    destination rather than overwriting it.
    """
    if not isinstance(factor_input, VerifiedFactorRun):
        raise TypeError("factor_input must come from load_factor_report_input")
    _require_family(sjm_input, "sjm_run")
    _require_family(market_input, "market_snapshot")
    destination = Path(destination)
    _require_new_report_destination(destination)
    if not static_windows:
        raise ValueError("static_windows must contain at least one explicit window")
    static_windows = tuple(static_windows)
    if not all(isinstance(window, StaticWindowSpec) for window in static_windows):
        raise TypeError("static_windows must contain StaticWindowSpec values")
    if len({(window.label, window.start, window.end) for window in static_windows}) != len(static_windows):
        raise ValueError("static_windows must not contain duplicate window identities")
    markowitz_start = pd.Timestamp(markowitz_10y_start)
    markowitz_end = pd.Timestamp(markowitz_end)
    if markowitz_start > markowitz_end:
        raise ValueError("markowitz_10y_start must be on or before markowitz_end")
    if isinstance(markowitz_n_points, bool) or not isinstance(markowitz_n_points, int) or markowitz_n_points < 2:
        raise ValueError("markowitz_n_points must be an integer of at least two")

    # Re-run the completed-manifest gates before there is any output directory.
    # They verify the current bytes, marker, source inventory, and lineage; typed
    # instances alone are deliberately insufficient after a caller has loaded them.
    factor_input = load_factor_report_input(
        factor_input.run_dir,
        run_id=factor_input.run_id,
        manifest_sha256=factor_input.manifest_sha256,
    )
    sjm_input = load_sjm_report_input(
        sjm_input.run_dir,
        run_id=sjm_input.identity,
        manifest_sha256=sjm_input.manifest_sha256,
    )
    market_input = load_market_report_input(
        market_input.run_dir,
        snapshot_id=market_input.identity,
        manifest_sha256=market_input.manifest_sha256,
    )
    markowitz_input = load_markowitz_report_input(
        market_input.run_dir,
        snapshot_id=market_input.identity,
        manifest_sha256=market_input.manifest_sha256,
    )

    inputs = _canonical_report_input_manifests(factor_input, sjm_input, market_input)
    factor_reports = build_factor_report_tables(factor_input)
    auxiliary_reports = build_auxiliary_report_tables(factor_input)
    factor_row = _factor_pit_reader_row(factor_input)
    common_start = pd.Timestamp(factor_row["start"])
    common_static = trio_static_window or _factor_window_static_spec(
        market_input, factor_row
    )
    sjm_reports = build_sjm_report_tables(
        sjm_input,
        market_input,
        performance_start=common_start,
        ssr_settings=ssr_settings,
    )
    trio_reports = build_trio_report_tables(
        factor_input,
        sjm_reports,
        market_input,
        static_windows=static_windows,
        trio_static_window=common_static,
        ssr_settings=ssr_settings,
    )
    max_start = markowitz_max_supported_start(
        markowitz_input, requested_end=markowitz_end
    )
    markowitz_reports = build_markowitz_report_tables(
        markowitz_input,
        requested_windows={
            "10y": (markowitz_start, markowitz_end),
            "max": (max_start, markowitz_end),
        },
        trio_rows={
            "10y": _panel_rows_for_markowitz_window(
                factor_input,
                sjm_input,
                market_input,
                name="10y",
                start=markowitz_start,
                end=markowitz_end,
                ssr_settings=ssr_settings,
            ),
            "max": _panel_rows_for_markowitz_window(
                factor_input,
                sjm_input,
                market_input,
                name="max",
                start=max_start,
                end=markowitz_end,
                ssr_settings=ssr_settings,
            ),
        },
        n_points=markowitz_n_points,
    )
    tables = (
        dict(factor_reports.tables)
        | dict(auxiliary_reports.tables)
        | dict(sjm_reports.tables)
        | dict(trio_reports.tables)
        | dict(markowitz_reports.tables)
    )
    if set(tables) != set(CANONICAL_REPORT_TABLE_SCHEMAS):
        raise ValueError(
            "complete canonical report assembly did not produce the required report-table catalog"
        )
    canonical_tables = {
        stem: parquet_safe_report_table(tables[stem].reset_index(drop=True))
        for stem in sorted(tables)
    }
    for stem, table in canonical_tables.items():
        _validate_report_table_rows(table, stem=stem)

    # Write every data table first, then pure locale projections, then the final
    # manifest, and finally the completion marker.  No operation follows marker
    # creation except the caller receiving the fully validated result.
    destination.mkdir(parents=True, exist_ok=True)
    tables_dir = destination / "tables"
    tables_dir.mkdir()
    for stem, table in canonical_tables.items():
        table.to_parquet(tables_dir / f"{stem}.parquet", index=False)
    try:
        from scripts import export_csv_mirrors as mirror_exporter
    except ImportError:
        import export_csv_mirrors as mirror_exporter
    mirror_exporter.write_locale_mirrors(canonical_tables, destination / "mirrors")
    configuration = {
        "static_windows": [
            {
                "label": window.label,
                "start": window.start.date().isoformat(),
                "end": window.end.date().isoformat(),
            }
            for window in static_windows
        ],
        "trio_static_window": {
            "label": common_static.label,
            "start": common_static.start.date().isoformat(),
            "end": common_static.end.date().isoformat(),
        },
        "markowitz": {
            "10y_requested_start": markowitz_start.date().isoformat(),
            "requested_end": markowitz_end.date().isoformat(),
            "max_requested_start": max_start.date().isoformat(),
            "frontier_points": markowitz_n_points,
        },
        "ssr_settings": {
            **_merged_ssr_settings(ssr_settings),
            "periods_per_year": 252,
        },
    }
    manifest = _canonical_report_manifest(
        report_id=destination.name,
        inputs=inputs,
        tables=canonical_tables,
        root=destination,
        configuration=configuration,
    )
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (destination / "COMPLETED").write_text(
        f"manifest_sha256={_sha256_file(manifest_path)}\n"
    )
    verified = validate_canonical_report_bundle(
        destination,
        factor_input=factor_input,
        sjm_input=sjm_input,
        market_input=market_input,
    )
    return CanonicalReportBundle(destination, verified, canonical_tables)


# --- Guard-ablation diagnostic canonical reports ----------------------------------- #
#
# This is intentionally a separate report family.  ``canonical_reports.v1`` and the
# data-v4 catalog are frozen publication contracts; the four-cell diagnostic is an
# Appendix-only input and therefore has its own manifest, inventory, and loader.

FACTOR_GUARD_ABLATION_RUN_SCHEMA = "factor_guard_ablation_run.v1"
FACTOR_GUARD_ABLATION_METRIC_RECORDS_SCHEMA = "factor_guard_ablation.metric_records.v1"
FACTOR_GUARD_ABLATION_PANEL_SCHEMA = "factor_guard_ablation.panel.v1"
CANONICAL_GUARD_ABLATION_REPORTS_SCHEMA = "canonical_guard_ablation_reports.v1"

GUARD_ABLATION_CONFIG_ORDER = (
    "factor_pit_ext2026",
    "factor_pit_unguarded_diagnostic_ext2026",
    "factor_nonpit_diagnostic_ext2026",
    "factor_nonpit_unguarded_diagnostic_ext2026",
)
GUARD_ABLATION_WINDOW_ORDER = ("full", "pre_cutoff", "post_cutoff")
GUARD_ABLATION_COMPARISON_ORDER = (
    "pit_unguarded_minus_guarded",
    "nonpit_unguarded_minus_guarded",
    "nonpit_unguarded_minus_pit_guarded_combined_stress",
)
_GUARD_ABLATION_COMPARISONS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {
        "pit_unguarded_minus_guarded": (
            "factor_pit_unguarded_diagnostic_ext2026",
            "factor_pit_ext2026",
        ),
        "nonpit_unguarded_minus_guarded": (
            "factor_nonpit_unguarded_diagnostic_ext2026",
            "factor_nonpit_diagnostic_ext2026",
        ),
        "nonpit_unguarded_minus_pit_guarded_combined_stress": (
            "factor_nonpit_unguarded_diagnostic_ext2026",
            "factor_pit_ext2026",
        ),
    }
)
_GUARD_ABLATION_COMPARISON_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "factor_pit_unguarded_minus_pit_guarded_ext2026": "pit_unguarded_minus_guarded",
        "pit_unguarded_minus_pit_guarded": "pit_unguarded_minus_guarded",
        "factor_nonpit_unguarded_minus_nonpit_guarded_ext2026": "nonpit_unguarded_minus_guarded",
        "nonpit_unguarded_minus_nonpit_guarded": "nonpit_unguarded_minus_guarded",
        "factor_nonpit_unguarded_minus_pit_guarded_ext2026": "nonpit_unguarded_minus_pit_guarded_combined_stress",
        "nonpit_unguarded_minus_pit_guarded": "nonpit_unguarded_minus_pit_guarded_combined_stress",
    }
)

GUARD_ABLATION_REPORT_TABLE_SCHEMAS: Mapping[str, str] = MappingProxyType(
    {
        "tear_sheet_factor_guard_ablation_ext2026": "tear_sheet.factor_guard_ablation.v1",
        "factor_guard_ablation_equity_ext2026": "factor_guard_ablation.equity.v1",
        "factor_guard_ablation_panel_ext2026": FACTOR_GUARD_ABLATION_PANEL_SCHEMA,
    }
)

_GUARD_ABLATION_UNGUARDED_FILES = (
    "factor_pit_unguarded_diagnostic_equity_ext2026.parquet",
    "factor_pit_unguarded_diagnostic_targets_ext2026.parquet",
    "factor_pit_unguarded_diagnostic_decision_log_ext2026.json",
    "factor_nonpit_unguarded_diagnostic_equity_ext2026.parquet",
    "factor_nonpit_unguarded_diagnostic_targets_ext2026.parquet",
    "factor_nonpit_unguarded_diagnostic_decision_log_ext2026.json",
)
_GUARD_ABLATION_EQUITY_COLUMNS = (
    "date",
    "configuration",
    "normalized_wealth",
    "drawdown",
    "relative_wealth",
    "relative_wealth_kind",
)


@dataclass(frozen=True)
class VerifiedGuardAblationRun:
    """A completed, byte-inventoried four-cell diagnostic producer run.

    The report layer validates the producer manifest itself rather than importing
    the producer module.  That preserves a narrow, stable consumer boundary while
    allowing the producer to retain ownership of replay and financial mechanics.
    """

    run_dir: Path
    run_id: str
    manifest_sha256: str
    manifest: Mapping[str, object]
    metric_records: Mapping[str, object]
    panel: pd.DataFrame
    files: Mapping[str, Mapping[str, object]]
    metric_entry: Mapping[str, object]
    panel_entry: Mapping[str, object]
    curve_entry: Mapping[str, object] | None
    curve_table: pd.DataFrame | None


def _guard_ablation_safe_relative_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("guard-ablation inventory file must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or path.name in {"", ".", ".."}:
        raise ValueError(f"guard-ablation inventory path is unsafe: {value!r}")
    return path


def _guard_ablation_manifest_files(
    run_dir: Path, manifest: Mapping[str, object]
) -> dict[str, Mapping[str, object]]:
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("guard-ablation manifest must contain a non-empty files inventory")
    resolved: dict[str, Mapping[str, object]] = {}
    declared: set[Path] = set()
    for role, raw_entry in files.items():
        if not isinstance(role, str) or not role or not isinstance(raw_entry, Mapping):
            raise ValueError("guard-ablation manifest files inventory is malformed")
        relative = _guard_ablation_safe_relative_path(raw_entry.get("file"))
        if relative in declared:
            raise ValueError("guard-ablation manifest inventories one artifact more than once")
        declared.add(relative)
        _pinned_sha256(f"files.{role}.sha256", raw_entry.get("sha256"))
        path = run_dir / relative
        if not path.is_file():
            raise ValueError(f"guard-ablation artifact is absent: {relative.as_posix()}")
        if _sha256_file(path) != raw_entry["sha256"]:
            raise ValueError(
                f"guard-ablation artifact was mutated after inventory: {relative.as_posix()}"
            )
        size = raw_entry.get("size")
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size != path.stat().st_size):
            raise ValueError(f"guard-ablation artifact size diverges from inventory: {relative.as_posix()}")
        resolved[role] = raw_entry

    actual = {
        path.relative_to(run_dir)
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    expected = declared | {Path("manifest.json"), Path("COMPLETED")}
    if actual != expected:
        raise ValueError("guard-ablation run contains unmanifested or missing output")
    return resolved


def _guard_ablation_inventory_entry(
    files: Mapping[str, Mapping[str, object]],
    *,
    filenames: Sequence[str],
    roles: Sequence[str] = (),
    required: bool = True,
) -> Mapping[str, object] | None:
    candidates: list[Mapping[str, object]] = []
    accepted_roles = set(roles)
    accepted_names = set(filenames)
    for role, entry in files.items():
        if role in accepted_roles or Path(str(entry["file"])).name in accepted_names:
            candidates.append(entry)
    # A role and its filename can point at the same entry; several different
    # entries are an ambiguous producer contract rather than a tie to resolve.
    unique = {str(entry["file"]): entry for entry in candidates}
    if not unique:
        if required:
            raise ValueError(
                "guard-ablation manifest is missing required artifact: "
                + ", ".join(filenames)
            )
        return None
    if len(unique) != 1:
        raise ValueError(
            "guard-ablation manifest names multiple artifacts for one required role: "
            + ", ".join(sorted(unique))
        )
    return next(iter(unique.values()))


def _guard_ablation_read_bytes(
    run: VerifiedGuardAblationRun | Path,
    entry: Mapping[str, object],
) -> bytes:
    run_dir = run.run_dir if isinstance(run, VerifiedGuardAblationRun) else run
    path = run_dir / _guard_ablation_safe_relative_path(entry["file"])
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"{path}: guard-ablation artifact was mutated after inventory")
    return data


def _canonical_guard_ablation_comparison(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("guard-ablation differential comparison_id must be a string")
    return _GUARD_ABLATION_COMPARISON_ALIASES.get(value, value)


def _guard_ablation_record_wrappers(
    metric_records: Mapping[str, object],
) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    """Validate and return producer-owned reader/differential wrappers.

    A wrapper carries diagnostic-only cell/window identity outside the immutable
    standard report row.  The embedded row is passed verbatim through the shared
    report-row gate, preventing an Appendix table from becoming a second metric
    calculator or a parallel schema family.
    """
    if metric_records.get("schema") != FACTOR_GUARD_ABLATION_METRIC_RECORDS_SCHEMA:
        raise ValueError(
            "guard-ablation metric records declare an incompatible schema "
            f"{metric_records.get('schema')!r}"
        )
    records = metric_records.get("records")
    if not isinstance(records, list):
        raise ValueError("guard-ablation metric records must contain an ordered records list")

    readers: list[Mapping[str, object]] = []
    differentials: list[Mapping[str, object]] = []
    expected_reader = [
        (configuration, window)
        for configuration in GUARD_ABLATION_CONFIG_ORDER
        for window in GUARD_ABLATION_WINDOW_ORDER
    ]
    expected_differential = [
        (comparison, window)
        for comparison in GUARD_ABLATION_COMPARISON_ORDER
        for window in GUARD_ABLATION_WINDOW_ORDER
    ]
    actual_reader: list[tuple[str, str]] = []
    actual_differential: list[tuple[str, str]] = []
    for position, wrapped in enumerate(records):
        if not isinstance(wrapped, Mapping):
            raise ValueError(f"guard-ablation metric record {position} is not an object")
        kind = wrapped.get("record_kind")
        window = wrapped.get("window")
        record = wrapped.get("record")
        if kind not in {"reader", "differential"} or window not in GUARD_ABLATION_WINDOW_ORDER:
            raise ValueError(
                f"guard-ablation metric record {position} must declare a supported record_kind and window"
            )
        if not isinstance(record, Mapping):
            raise ValueError(f"guard-ablation metric record {position} has no embedded producer row")
        validated = validate_report_row(dict(record))
        if kind == "reader":
            configuration = wrapped.get("configuration")
            if configuration not in GUARD_ABLATION_CONFIG_ORDER:
                raise ValueError(f"guard-ablation reader record {position} has an unknown configuration")
            if validated["schema"] != READER_SCHEMA or validated["portfolio_id"] != configuration:
                raise ValueError(
                    "guard-ablation reader wrapper must bind its configuration to one "
                    "producer-owned reader row"
                )
            actual_reader.append((str(configuration), str(window)))
            readers.append(
                {
                    "record_kind": "reader",
                    "configuration": configuration,
                    "comparison_id": None,
                    "window": window,
                    "record": validated,
                }
            )
        else:
            comparison = _canonical_guard_ablation_comparison(wrapped.get("comparison_id"))
            if comparison not in GUARD_ABLATION_COMPARISON_ORDER:
                raise ValueError(f"guard-ablation differential record {position} has an unknown comparison")
            if validated["schema"] != DIFFERENTIAL_SCHEMA:
                raise ValueError("guard-ablation differential wrapper must contain a differential row")
            if "endpoint_total_return_difference" not in validated:
                raise ValueError(
                    "guard-ablation differential rows must preserve the separately named endpoint gap"
                )
            actual_differential.append((comparison, str(window)))
            differentials.append(
                {
                    "record_kind": "differential",
                    "configuration": None,
                    "comparison_id": comparison,
                    "window": window,
                    "record": validated,
                }
            )
    if actual_reader != expected_reader or actual_differential != expected_differential:
        raise ValueError(
            "guard-ablation metric records must contain the exact ordered four-cell "
            "reader and three-comparison differential matrix"
        )
    if len(records) != 21:
        raise ValueError("guard-ablation metric record matrix must contain exactly 21 records")
    return tuple(readers), tuple(differentials)


def _guard_ablation_panel_contract(
    entry: Mapping[str, object], manifest: Mapping[str, object]
) -> tuple[str, ...]:
    """Get the producer-declared, exact v1 panel columns.

    The panel intentionally carries mechanism fields owned by the diagnostic
    producer.  Its exact column list is sealed in the producer inventory so a
    report consumer neither drops a diagnostic nor invents a local replacement.
    """
    schema = entry.get("schema")
    if schema != FACTOR_GUARD_ABLATION_PANEL_SCHEMA:
        raise ValueError("guard-ablation panel inventory declares an incompatible schema")
    columns = entry.get("columns")
    if columns is None:
        panel_contract = manifest.get("panel_schema")
        columns = panel_contract.get("columns") if isinstance(panel_contract, Mapping) else None
    if not isinstance(columns, list) or not columns or not all(
        isinstance(column, str) and column for column in columns
    ) or len(set(columns)) != len(columns):
        raise ValueError("guard-ablation panel inventory must declare unique exact columns")
    return tuple(columns)


def _validate_guard_ablation_panel(
    panel: pd.DataFrame, *, columns: Sequence[str]
) -> pd.DataFrame:
    if not isinstance(panel, pd.DataFrame) or not panel.index.equals(pd.RangeIndex(len(panel))):
        raise ValueError("guard-ablation panel must be a flat DataFrame")
    if tuple(panel.columns) != tuple(columns):
        raise ValueError("guard-ablation panel columns diverge from its strict producer contract")
    date_column = "rebalance_date" if "rebalance_date" in panel.columns else "date"
    required = {date_column, "configuration", "evidence_id", "prompt_mode", "guard_enabled", "p_memorized"}
    if not required.issubset(panel.columns):
        raise ValueError(
            "guard-ablation panel must retain date, configuration, evidence, prompt, guard, and memorization fields"
        )
    if len(panel) != 360:
        raise ValueError("guard-ablation panel must contain exactly 360 rebalance/configuration rows")
    if panel["configuration"].isna().any() or panel["evidence_id"].isna().any() or panel["prompt_mode"].isna().any():
        raise ValueError("guard-ablation panel contains missing mechanism identities")
    parsed_dates = pd.to_datetime(panel[date_column], errors="raise")
    if parsed_dates.dt.tz is not None:
        raise ValueError("guard-ablation panel rebalance dates must be timezone-naive")
    dates_by_configuration: list[tuple[pd.Timestamp, ...]] = []
    expected_guard = {
        "factor_pit_ext2026": True,
        "factor_pit_unguarded_diagnostic_ext2026": False,
        "factor_nonpit_diagnostic_ext2026": True,
        "factor_nonpit_unguarded_diagnostic_ext2026": False,
    }
    offset = 0
    for configuration in GUARD_ABLATION_CONFIG_ORDER:
        block = panel.iloc[offset : offset + 90]
        block_dates = parsed_dates.iloc[offset : offset + 90]
        if len(block) != 90 or tuple(block["configuration"]) != (configuration,) * 90:
            raise ValueError("guard-ablation panel configuration blocks are not in stable required order")
        if block_dates.duplicated().any() or not block_dates.is_monotonic_increasing:
            raise ValueError("guard-ablation panel dates must be unique and increasing within every configuration")
        if not all(isinstance(value, (bool, np.bool_)) for value in block["guard_enabled"]):
            raise ValueError("guard-ablation panel guard_enabled values must be booleans")
        if set(bool(value) for value in block["guard_enabled"]) != {expected_guard[configuration]}:
            raise ValueError("guard-ablation panel guard flags do not match the declared configuration")
        memorized = pd.to_numeric(block["p_memorized"], errors="raise")
        observed = memorized.dropna()
        if not np.isfinite(observed.to_numpy(dtype=float)).all() or not ((observed >= 0.0) & (observed <= 1.0)).all():
            raise ValueError("guard-ablation panel non-null p_memorized values must be finite probabilities")
        if memorized.isna().any():
            if not {"parse_ok", "steered"}.issubset(block.columns):
                raise ValueError("guard-ablation panel score-null rows require parse/steering state")
            invalid_null = memorized.isna() & block["steered"].astype(bool)
            if invalid_null.any():
                raise ValueError("guard-ablation panel cannot mark a missing memorization score as steered")
        dates_by_configuration.append(tuple(pd.Timestamp(value) for value in block_dates))
        offset += 90
    if any(dates != dates_by_configuration[0] for dates in dates_by_configuration[1:]):
        raise ValueError("guard-ablation panel configurations must share one rebalance calendar")
    return panel.copy()


def _validate_guard_ablation_curve_table(table: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(table, pd.DataFrame) or not table.index.equals(pd.RangeIndex(len(table))):
        raise ValueError("guard-ablation equity table must be a flat DataFrame")
    if tuple(table.columns) != _GUARD_ABLATION_EQUITY_COLUMNS:
        raise ValueError("guard-ablation equity table columns diverge from the canonical contract")
    if table.empty:
        raise ValueError("guard-ablation equity table must not be empty")
    dates = pd.to_datetime(table["date"], errors="raise")
    if dates.dt.tz is not None:
        raise ValueError("guard-ablation equity dates must be timezone-naive")
    date_blocks: list[tuple[pd.Timestamp, ...]] = []
    offset = 0
    for configuration in GUARD_ABLATION_CONFIG_ORDER:
        block = table[table["configuration"] == configuration]
        if block.empty or tuple(table.iloc[offset : offset + len(block)]["configuration"]) != (configuration,) * len(block):
            raise ValueError("guard-ablation equity configurations are not in stable required order")
        block_dates = pd.to_datetime(block["date"], errors="raise")
        if block_dates.duplicated().any() or not block_dates.is_monotonic_increasing:
            raise ValueError("guard-ablation equity dates must be unique and increasing")
        wealth = pd.to_numeric(block["normalized_wealth"], errors="raise").to_numpy(dtype=float)
        drawdown = pd.to_numeric(block["drawdown"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(wealth).all() or not np.isfinite(drawdown).all() or (wealth <= 0.0).any() or (drawdown > 1e-12).any():
            raise ValueError("guard-ablation equity wealth/drawdown values are invalid")
        if not np.isclose(wealth[0], 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("every guard-ablation curve must be normalized to 1.0 at the common start")
        expected_kind = {
            "factor_pit_ext2026": None,
            "factor_pit_unguarded_diagnostic_ext2026": "controlled_pit_unguarded_vs_guarded",
            "factor_nonpit_diagnostic_ext2026": None,
            "factor_nonpit_unguarded_diagnostic_ext2026": "combined_nonpit_unguarded_vs_pit_guarded_stress",
        }[configuration]
        kind = block["relative_wealth_kind"]
        relative = pd.to_numeric(block["relative_wealth"], errors="coerce")
        if expected_kind is None:
            if kind.notna().any() or relative.notna().any():
                raise ValueError("guarded/reference curves must not claim a relative-wealth comparison")
        else:
            if set(kind.astype(str)) != {expected_kind} or relative.isna().any() or not np.isfinite(relative.to_numpy(dtype=float)).all() or (relative <= 0.0).any():
                raise ValueError("guard-ablation relative-wealth semantics are invalid")
        date_blocks.append(tuple(pd.Timestamp(value) for value in block_dates))
        offset += len(block)
    if offset != len(table) or any(dates_ != date_blocks[0] for dates_ in date_blocks[1:]):
        raise ValueError("guard-ablation equity table must contain four common-window curves")
    return table.copy()


def load_completed_guard_ablation_run(
    run_dir: Path | str, *, run_id: str, manifest_sha256: str
) -> VerifiedGuardAblationRun:
    """Load a completed producer-owned guard-ablation run without importing it."""
    _pinned_identity("run_id", run_id)
    _pinned_sha256("manifest_sha256", manifest_sha256)
    run_dir = _require_run_directory(run_dir, "guard-ablation run input")
    manifest_path = run_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest_path}: guard-ablation manifest is invalid JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != FACTOR_GUARD_ABLATION_RUN_SCHEMA:
        raise ValueError(
            f"guard-ablation manifest schema must be {FACTOR_GUARD_ABLATION_RUN_SCHEMA!r}"
        )
    if manifest.get("completed") is not True:
        raise ValueError("guard-ablation manifest must declare completed=true")
    if manifest.get("run_id") != run_id:
        raise ValueError("guard-ablation run identity mismatch")
    if _sha256_file(manifest_path) != manifest_sha256:
        raise ValueError("guard-ablation manifest sha256 mismatch")
    if not isinstance(manifest.get("source_commit"), str) or not manifest["source_commit"]:
        raise ValueError("guard-ablation manifest must pin a source_commit")
    inputs = manifest.get("input_manifests")
    if not isinstance(inputs, Mapping) or set(("factor_run", "market_snapshot")) - set(inputs):
        raise ValueError("guard-ablation manifest must pin parent Factor and market manifests")
    for role, identity in (("factor_run", "run_id"), ("market_snapshot", "snapshot_id")):
        entry = inputs[role]
        if not isinstance(entry, Mapping):
            raise ValueError(f"guard-ablation input manifest {role!r} must be an object")
        _pinned_identity(f"input_manifests.{role}.{identity}", entry.get(identity))
        _pinned_sha256(f"input_manifests.{role}.manifest_sha256", entry.get("manifest_sha256"))
    marker = (run_dir / "COMPLETED").read_text().splitlines()
    expected_marker = f"manifest_sha256={manifest_sha256}"
    # Strategy producers conventionally retain their build time above the hash;
    # accept that established two-line form (and the one-line canonical form),
    # while requiring the final line to bind the current manifest exactly.
    if len(marker) not in {1, 2} or marker[-1] != expected_marker:
        raise ValueError("guard-ablation COMPLETED marker does not bind the final manifest")
    if len(marker) == 2:
        try:
            pd.Timestamp(marker[0])
        except (TypeError, ValueError) as exc:
            raise ValueError("guard-ablation COMPLETED marker has an invalid build time") from exc

    files = _guard_ablation_manifest_files(run_dir, manifest)
    for filename in _GUARD_ABLATION_UNGUARDED_FILES:
        _guard_ablation_inventory_entry(files, filenames=(filename,))
    metric_entry = _guard_ablation_inventory_entry(
        files,
        filenames=("factor_guard_ablation_metric_records_ext2026.json",),
        roles=("metric_records",),
    )
    panel_entry = _guard_ablation_inventory_entry(
        files,
        filenames=("factor_guard_ablation_panel_ext2026.parquet",),
        roles=("panel",),
    )
    assert metric_entry is not None and panel_entry is not None
    try:
        metric_records = json.loads(_guard_ablation_read_bytes(run_dir, metric_entry).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("guard-ablation metric-record artifact is not valid UTF-8 JSON") from exc
    if not isinstance(metric_records, Mapping):
        raise ValueError("guard-ablation metric-record artifact must be a JSON object")
    _guard_ablation_record_wrappers(metric_records)
    panel_columns = _guard_ablation_panel_contract(panel_entry, manifest)
    panel = pd.read_parquet(io.BytesIO(_guard_ablation_read_bytes(run_dir, panel_entry)))
    panel = _validate_guard_ablation_panel(panel, columns=panel_columns)

    curve_entry = _guard_ablation_inventory_entry(
        files,
        filenames=(
            "factor_guard_ablation_curves_ext2026.parquet",
            "factor_guard_ablation_curve_ext2026.parquet",
        ),
        roles=("curves", "curve_table"),
        required=False,
    )
    curve_table = None
    if curve_entry is not None:
        curve_table = _validate_guard_ablation_curve_table(
            pd.read_parquet(io.BytesIO(_guard_ablation_read_bytes(run_dir, curve_entry)))
        )
    return VerifiedGuardAblationRun(
        run_dir=run_dir,
        run_id=run_id,
        manifest_sha256=manifest_sha256,
        manifest=manifest,
        metric_records=metric_records,
        panel=panel,
        files=MappingProxyType(dict(files)),
        metric_entry=metric_entry,
        panel_entry=panel_entry,
        curve_entry=curve_entry,
        curve_table=curve_table,
    )


# A descriptive alias retains the naming symmetry with ``load_factor_report_input``.
load_guard_ablation_report_input = load_completed_guard_ablation_run


@dataclass(frozen=True)
class GuardAblationReportTables:
    """Pure diagnostic table projections and their verified source lineage."""

    owner: str
    lineage: Mapping[str, object]
    rows: Mapping[str, tuple[Mapping[str, object], ...]]
    tables: Mapping[str, pd.DataFrame]


def project_guard_ablation_metric_records(
    metric_records: Mapping[str, object],
) -> tuple[tuple[Mapping[str, object], ...], pd.DataFrame]:
    """Project the sealed producer record matrix, without recalculating a metric."""
    readers, differentials = _guard_ablation_record_wrappers(metric_records)
    rows: list[dict[str, object]] = []
    for wrapped in (*readers, *differentials):
        record = dict(wrapped["record"])
        row = {
            "row_order": len(rows),
            "record_kind": wrapped["record_kind"],
            "configuration": wrapped["configuration"],
            "comparison_id": wrapped["comparison_id"],
            "window": wrapped["window"],
            "metric_semantics": (
                "portfolio_return_metrics"
                if wrapped["record_kind"] == "reader"
                else "daily_comparison_minus_reference_return_metrics; "
                "endpoint_total_return_difference_is_descriptive_endpoint_gap"
            ),
            **record,
        }
        rows.append(row)
    columns = [
        "row_order", "record_kind", "configuration", "comparison_id", "window", "metric_semantics"
    ]
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    return tuple(rows), pd.DataFrame(rows, columns=columns)


def _guard_ablation_equity_filename(configuration: str) -> str:
    return {
        "factor_pit_ext2026": "factor_equity_ext2026.parquet",
        "factor_pit_unguarded_diagnostic_ext2026": "factor_pit_unguarded_diagnostic_equity_ext2026.parquet",
        "factor_nonpit_diagnostic_ext2026": "factor_nonpit_diagnostic_equity_ext2026.parquet",
        "factor_nonpit_unguarded_diagnostic_ext2026": "factor_nonpit_unguarded_diagnostic_equity_ext2026.parquet",
    }[configuration]


def _guard_ablation_value_series(data: bytes, *, label: str) -> pd.Series:
    frame = pd.read_parquet(io.BytesIO(data))
    if "value" not in frame.columns:
        raise ValueError(f"{label}: equity artifact must have a value column")
    value = frame["value"].copy()
    value.index = pd.DatetimeIndex(value.index)
    values = value.to_numpy(dtype=float)
    if (
        len(value) < 3
        or value.index.has_duplicates
        or not value.index.is_monotonic_increasing
        or value.index.tz is not None
        or not np.isfinite(values).all()
        or (values <= 0.0).any()
    ):
        raise ValueError(f"{label}: equity curve must be finite, positive, unique, ordered, and timezone-naive")
    return value


def _guard_ablation_parent_equity(
    factor_input: VerifiedFactorRun, configuration: str
) -> pd.Series:
    stream = factor_input.metric_records.get("source_streams", {}).get(configuration)
    if not isinstance(stream, Mapping) or not isinstance(stream.get("artifact"), str):
        raise ValueError(f"parent Factor run lacks the declared {configuration!r} equity stream")
    artifact = str(stream["artifact"])
    entry = _factor_inventory_entry(factor_input.manifest, artifact)
    return _guard_ablation_value_series(
        _read_inventoried_bytes(factor_input.run_dir / artifact, entry),
        label=f"parent Factor {configuration}",
    )


def _guard_ablation_equity_from_inputs(
    guard_input: VerifiedGuardAblationRun,
    *,
    factor_input: VerifiedFactorRun | None,
) -> pd.DataFrame:
    if factor_input is not None:
        factor_input = load_factor_report_input(
            factor_input.run_dir,
            run_id=factor_input.run_id,
            manifest_sha256=factor_input.manifest_sha256,
        )
        lineage = guard_input.manifest["input_manifests"]["factor_run"]
        if (
            lineage.get("run_id"), lineage.get("manifest_sha256")
        ) != (factor_input.run_id, factor_input.manifest_sha256):
            raise ValueError("guard-ablation parent Factor lineage diverges from the supplied completed Factor run")
    values: dict[str, pd.Series] = {}
    for configuration in GUARD_ABLATION_CONFIG_ORDER:
        entry = _guard_ablation_inventory_entry(
            guard_input.files,
            filenames=(_guard_ablation_equity_filename(configuration),),
            required=False,
        )
        if entry is not None:
            values[configuration] = _guard_ablation_value_series(
                _guard_ablation_read_bytes(guard_input, entry), label=configuration
            )
        elif configuration in {"factor_pit_ext2026", "factor_nonpit_diagnostic_ext2026"} and factor_input is not None:
            values[configuration] = _guard_ablation_parent_equity(factor_input, configuration)
        else:
            raise ValueError(
                f"guard-ablation run has no validated curve table or equity stream for {configuration}; "
                "supply the verified parent Factor run for guarded curves"
            )
    common = values[GUARD_ABLATION_CONFIG_ORDER[0]].index
    for series in values.values():
        common = common.intersection(series.index)
    common = common.sort_values()
    if len(common) < 2:
        raise ValueError("guard-ablation equity inputs have no viable common window")
    normalized = {
        configuration: values[configuration].loc[common] / float(values[configuration].loc[common].iloc[0])
        for configuration in GUARD_ABLATION_CONFIG_ORDER
    }
    controlled = normalized["factor_pit_unguarded_diagnostic_ext2026"] / normalized["factor_pit_ext2026"]
    stress = normalized["factor_nonpit_unguarded_diagnostic_ext2026"] / normalized["factor_pit_ext2026"]
    rows: list[dict[str, object]] = []
    for configuration in GUARD_ABLATION_CONFIG_ORDER:
        curve = normalized[configuration]
        relative_kind = {
            "factor_pit_ext2026": None,
            "factor_pit_unguarded_diagnostic_ext2026": "controlled_pit_unguarded_vs_guarded",
            "factor_nonpit_diagnostic_ext2026": None,
            "factor_nonpit_unguarded_diagnostic_ext2026": "combined_nonpit_unguarded_vs_pit_guarded_stress",
        }[configuration]
        relative = (
            controlled
            if configuration == "factor_pit_unguarded_diagnostic_ext2026"
            else stress
            if configuration == "factor_nonpit_unguarded_diagnostic_ext2026"
            else None
        )
        drawdown = curve / curve.cummax() - 1.0
        for date, wealth in curve.items():
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "configuration": configuration,
                    "normalized_wealth": float(wealth),
                    "drawdown": float(drawdown.loc[date]),
                    "relative_wealth": None if relative is None else float(relative.loc[date]),
                    "relative_wealth_kind": relative_kind,
                }
            )
    return _validate_guard_ablation_curve_table(
        pd.DataFrame(rows, columns=list(_GUARD_ABLATION_EQUITY_COLUMNS))
    )


def build_guard_ablation_report_tables(
    guard_input: VerifiedGuardAblationRun,
    *,
    factor_input: VerifiedFactorRun | None = None,
) -> GuardAblationReportTables:
    """Build the isolated four-cell diagnostic tables from verified inputs only."""
    if not isinstance(guard_input, VerifiedGuardAblationRun):
        raise TypeError("guard_input must come from load_completed_guard_ablation_run")
    if factor_input is not None:
        if not isinstance(factor_input, VerifiedFactorRun):
            raise TypeError("factor_input must come from load_factor_report_input")
        parent = guard_input.manifest["input_manifests"]["factor_run"]
        if (parent.get("run_id"), parent.get("manifest_sha256")) != (
            factor_input.run_id,
            factor_input.manifest_sha256,
        ):
            raise ValueError(
                "guard-ablation parent Factor lineage diverges from the supplied completed Factor run"
            )
    metric_rows, metrics = project_guard_ablation_metric_records(guard_input.metric_records)
    equity = (
        _validate_guard_ablation_curve_table(guard_input.curve_table)
        if guard_input.curve_table is not None
        else _guard_ablation_equity_from_inputs(guard_input, factor_input=factor_input)
    )
    panel = _validate_guard_ablation_panel(
        guard_input.panel,
        columns=_guard_ablation_panel_contract(guard_input.panel_entry, guard_input.manifest),
    )
    return GuardAblationReportTables(
        owner=REPORT_TABLE_OWNER,
        lineage={
            "factor_guard_ablation_run": {
                "run_id": guard_input.run_id,
                "manifest_sha256": guard_input.manifest_sha256,
            },
            "parent_factor_run": dict(guard_input.manifest["input_manifests"]["factor_run"]),
            "market_snapshot": dict(guard_input.manifest["input_manifests"]["market_snapshot"]),
        },
        rows={"tear_sheet_factor_guard_ablation_ext2026": metric_rows},
        tables={
            "tear_sheet_factor_guard_ablation_ext2026": metrics,
            "factor_guard_ablation_equity_ext2026": equity,
            "factor_guard_ablation_panel_ext2026": panel,
        },
    )


@dataclass(frozen=True)
class CanonicalGuardAblationReportBundle:
    """One immutable Appendix diagnostic report bundle, outside data-v4."""

    root: Path
    manifest: Mapping[str, object]
    tables: Mapping[str, pd.DataFrame]


def _guard_ablation_report_manifest(
    *,
    report_id: str,
    guard_input: VerifiedGuardAblationRun,
    tables: Mapping[str, pd.DataFrame],
    root: Path,
) -> dict[str, object]:
    source_artifacts = {
        "metric_records": {
            "file": guard_input.metric_entry["file"],
            "sha256": guard_input.metric_entry["sha256"],
        },
        "panel": {
            "file": guard_input.panel_entry["file"],
            "sha256": guard_input.panel_entry["sha256"],
        },
    }
    if guard_input.curve_entry is not None:
        source_artifacts["curves"] = {
            "file": guard_input.curve_entry["file"],
            "sha256": guard_input.curve_entry["sha256"],
        }
    table_inventory: dict[str, dict[str, object]] = {}
    mirror_inventory: dict[str, dict[str, object]] = {}
    for stem in sorted(GUARD_ABLATION_REPORT_TABLE_SCHEMAS):
        table = tables[stem]
        table_path = root / "tables" / f"{stem}.parquet"
        table_inventory[stem] = _report_manifest_entry(
            path=table_path,
            relative=table_path.relative_to(root),
            schema=GUARD_ABLATION_REPORT_TABLE_SCHEMAS[stem],
            rows=len(table),
        )
        if stem == "factor_guard_ablation_panel_ext2026":
            table_inventory[stem]["columns"] = list(table.columns)
        for locale, suffix in (("en-US", ".csv"), ("de-DE", "_de.csv")):
            mirror_path = root / "mirrors" / f"{stem}{suffix}"
            mirror_inventory[mirror_path.name] = {
                **_report_manifest_entry(
                    path=mirror_path,
                    relative=mirror_path.relative_to(root),
                    schema=GUARD_ABLATION_REPORT_TABLE_SCHEMAS[stem],
                    rows=len(table),
                ),
                "locale": locale,
                "source_table": stem,
            }
    return {
        "schema": CANONICAL_GUARD_ABLATION_REPORTS_SCHEMA,
        "completed": True,
        "producer": REPORT_TABLE_OWNER,
        "report_id": report_id,
        "input_manifests": {
            "factor_guard_ablation_run": {
                "run_id": guard_input.run_id,
                "manifest_sha256": guard_input.manifest_sha256,
            }
        },
        "source_artifacts": source_artifacts,
        "tables": table_inventory,
        "mirrors": mirror_inventory,
    }


def _validate_guard_ablation_tear_table(table: pd.DataFrame) -> None:
    if not table.index.equals(pd.RangeIndex(len(table))) or len(table) != 21:
        raise ValueError("guard-ablation tear sheet must be a flat exact 21-row table")
    required = {"row_order", "record_kind", "configuration", "comparison_id", "window", "metric_semantics", "schema"}
    if not required.issubset(table.columns) or list(table["row_order"]) != list(range(21)):
        raise ValueError("guard-ablation tear sheet has an invalid row identity/order contract")
    reader = table.iloc[:12]
    differential = table.iloc[12:]
    expected_readers = [
        (configuration, window)
        for configuration in GUARD_ABLATION_CONFIG_ORDER
        for window in GUARD_ABLATION_WINDOW_ORDER
    ]
    expected_differentials = [
        (comparison, window)
        for comparison in GUARD_ABLATION_COMPARISON_ORDER
        for window in GUARD_ABLATION_WINDOW_ORDER
    ]
    if list(zip(reader["configuration"], reader["window"], strict=True)) != expected_readers or set(reader["record_kind"]) != {"reader"}:
        raise ValueError("guard-ablation tear sheet reader rows diverge from the four-cell order")
    if list(zip(differential["comparison_id"], differential["window"], strict=True)) != expected_differentials or set(differential["record_kind"]) != {"differential"}:
        raise ValueError("guard-ablation tear sheet differential rows diverge from the comparison order")
    for row in table.to_dict(orient="records"):
        record = {
            key: value
            for key, value in row.items()
            if key not in {"row_order", "record_kind", "configuration", "comparison_id", "window", "metric_semantics"}
            and value is not None
            and not (not isinstance(value, (str, bytes)) and bool(pd.isna(value)))
        }
        validate_report_row(record)
        if row["record_kind"] == "differential":
            if row["schema"] != DIFFERENTIAL_SCHEMA or "endpoint_total_return_difference" not in record or "endpoint_gap" not in str(row["metric_semantics"]):
                raise ValueError("guard-ablation differential rows lost their return-versus-endpoint semantics")
        elif row["schema"] != READER_SCHEMA:
            raise ValueError("guard-ablation reader rows must retain the reader schema")


def _validate_guard_ablation_report_tables(tables: Mapping[str, pd.DataFrame]) -> None:
    if set(tables) != set(GUARD_ABLATION_REPORT_TABLE_SCHEMAS):
        raise ValueError("guard-ablation bundle must contain exactly its three diagnostic tables")
    _validate_guard_ablation_tear_table(tables["tear_sheet_factor_guard_ablation_ext2026"])
    _validate_guard_ablation_curve_table(tables["factor_guard_ablation_equity_ext2026"])
    panel = tables["factor_guard_ablation_panel_ext2026"]
    # The panel column order is run-specific but is already sealed by its input
    # inventory.  Here we preserve it and recheck its universal 360-row mechanics.
    _validate_guard_ablation_panel(panel, columns=tuple(panel.columns))


def materialize_canonical_guard_ablation_report_bundle(
    guard_input: VerifiedGuardAblationRun,
    *,
    destination: Path | str,
    factor_input: VerifiedFactorRun | None = None,
) -> CanonicalGuardAblationReportBundle:
    """Materialize the standalone immutable diagnostic canonical bundle.

    Existing roots (including incomplete roots) are never reused.  All tables
    and deterministic locale projections are finalized before the manifest; the
    completion marker is the last filesystem mutation.
    """
    if not isinstance(guard_input, VerifiedGuardAblationRun):
        raise TypeError("guard_input must come from load_completed_guard_ablation_run")
    destination = Path(destination)
    _require_new_report_destination(destination)
    guard_input = load_completed_guard_ablation_run(
        guard_input.run_dir,
        run_id=guard_input.run_id,
        manifest_sha256=guard_input.manifest_sha256,
    )
    if factor_input is not None:
        factor_input = load_factor_report_input(
            factor_input.run_dir,
            run_id=factor_input.run_id,
            manifest_sha256=factor_input.manifest_sha256,
        )
    reports = build_guard_ablation_report_tables(guard_input, factor_input=factor_input)
    tables = {
        stem: parquet_safe_report_table(table.reset_index(drop=True))
        for stem, table in reports.tables.items()
    }
    _validate_guard_ablation_report_tables(tables)
    destination.mkdir(parents=True, exist_ok=True)
    tables_dir = destination / "tables"
    tables_dir.mkdir()
    for stem in sorted(tables):
        tables[stem].to_parquet(tables_dir / f"{stem}.parquet", index=False)
    try:
        from scripts import export_csv_mirrors as mirror_exporter
    except ImportError:
        import export_csv_mirrors as mirror_exporter
    # Do not call require_catalog_mirror_coverage: this diagnostic namespace is
    # deliberately outside frozen data-v4 and is not a catalog extension.
    mirror_exporter.write_locale_mirrors(tables, destination / "mirrors")
    manifest = _guard_ablation_report_manifest(
        report_id=destination.name,
        guard_input=guard_input,
        tables=tables,
        root=destination,
    )
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (destination / "COMPLETED").write_text(
        f"manifest_sha256={_sha256_file(manifest_path)}\n"
    )
    verified = validate_canonical_guard_ablation_report_bundle(
        destination, guard_input=guard_input, factor_input=factor_input
    )
    return CanonicalGuardAblationReportBundle(destination, verified, tables)


def validate_canonical_guard_ablation_report_bundle(
    root: Path | str,
    *,
    guard_input: VerifiedGuardAblationRun | None = None,
    factor_input: VerifiedFactorRun | None = None,
) -> Mapping[str, object]:
    """Read-only validation for the separate guard-ablation canonical family."""
    root = Path(root)
    manifest_path, marker_path = root / "manifest.json", root / "COMPLETED"
    if not root.is_dir() or not manifest_path.is_file() or not marker_path.is_file():
        raise ValueError("canonical guard-ablation report root is incomplete")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError("canonical guard-ablation report manifest is invalid JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != CANONICAL_GUARD_ABLATION_REPORTS_SCHEMA:
        raise ValueError("canonical guard-ablation report manifest has an incompatible schema")
    if manifest.get("completed") is not True or manifest.get("producer") != REPORT_TABLE_OWNER:
        raise ValueError("canonical guard-ablation report manifest is not completed by the report owner")
    _pinned_identity("report_id", manifest.get("report_id"))
    inputs = manifest.get("input_manifests")
    if not isinstance(inputs, Mapping) or set(inputs) != {"factor_guard_ablation_run"}:
        raise ValueError("canonical guard-ablation report must pin exactly its diagnostic source run")
    source = inputs["factor_guard_ablation_run"]
    if not isinstance(source, Mapping):
        raise ValueError("canonical guard-ablation source manifest identity is malformed")
    _pinned_identity("input_manifests.factor_guard_ablation_run.run_id", source.get("run_id"))
    _pinned_sha256("input_manifests.factor_guard_ablation_run.manifest_sha256", source.get("manifest_sha256"))

    expected_stems = set(GUARD_ABLATION_REPORT_TABLE_SCHEMAS)
    tables_inventory = manifest.get("tables")
    expected_mirrors = {
        f"{stem}{suffix}" for stem in expected_stems for suffix in (".csv", "_de.csv")
    }
    mirrors = manifest.get("mirrors")
    if not isinstance(tables_inventory, Mapping) or set(tables_inventory) != expected_stems:
        raise ValueError("canonical guard-ablation report table inventory is incomplete")
    if not isinstance(mirrors, Mapping) or set(mirrors) != expected_mirrors:
        raise ValueError("canonical guard-ablation report mirror inventory is incomplete")

    loaded: dict[str, pd.DataFrame] = {}
    table_files: set[Path] = set()
    mirror_files: set[Path] = set()
    try:
        from scripts import export_csv_mirrors as mirror_exporter
    except ImportError:
        import export_csv_mirrors as mirror_exporter
    for stem in sorted(expected_stems):
        entry = tables_inventory[stem]
        if not isinstance(entry, Mapping):
            raise ValueError(f"{stem}: canonical guard-ablation table entry is malformed")
        expected_path = Path("tables") / f"{stem}.parquet"
        path = _safe_report_relative_path(entry.get("file"), directory="tables")
        if path != expected_path or entry.get("schema") != GUARD_ABLATION_REPORT_TABLE_SCHEMAS[stem]:
            raise ValueError(f"{stem}: canonical guard-ablation table inventory diverges")
        if isinstance(entry.get("rows"), bool) or not isinstance(entry.get("rows"), int) or entry["rows"] < 1:
            raise ValueError(f"{stem}: canonical guard-ablation row count is invalid")
        _pinned_sha256(f"tables.{stem}.sha256", entry.get("sha256"))
        disk_path = root / path
        if not disk_path.is_file() or _sha256_file(disk_path) != entry["sha256"]:
            raise ValueError(f"{stem}: canonical guard-ablation table is missing or mutated")
        table = pd.read_parquet(disk_path)
        if len(table) != entry["rows"]:
            raise ValueError(f"{stem}: canonical guard-ablation table row count diverges")
        if stem == "factor_guard_ablation_panel_ext2026" and entry.get("columns") != list(table.columns):
            raise ValueError("guard-ablation panel columns diverge from the canonical report inventory")
        loaded[stem] = table
        table_files.add(path)
        for locale, suffix in (("en-US", ".csv"), ("de-DE", "_de.csv")):
            name = f"{stem}{suffix}"
            mirror = mirrors[name]
            if not isinstance(mirror, Mapping):
                raise ValueError(f"{name}: canonical guard-ablation mirror entry is malformed")
            mirror_path = _safe_report_relative_path(mirror.get("file"), directory="mirrors")
            if mirror_path != Path("mirrors") / name or (
                mirror.get("schema"), mirror.get("source_table"), mirror.get("locale"), mirror.get("rows")
            ) != (GUARD_ABLATION_REPORT_TABLE_SCHEMAS[stem], stem, locale, entry["rows"]):
                raise ValueError(f"{name}: canonical guard-ablation mirror inventory diverges")
            _pinned_sha256(f"mirrors.{name}.sha256", mirror.get("sha256"))
            disk_mirror = root / mirror_path
            if not disk_mirror.is_file() or _sha256_file(disk_mirror) != mirror["sha256"]:
                raise ValueError(f"{name}: canonical guard-ablation mirror is missing or mutated")
            mirror_exporter.verify_mirror_round_trip(table, disk_mirror, locale=locale)
            mirror_files.add(mirror_path)
    _validate_guard_ablation_report_tables(loaded)
    actual_tables = {
        Path("tables") / path.name for path in (root / "tables").iterdir() if path.is_file()
    } if (root / "tables").is_dir() else set()
    actual_mirrors = {
        Path("mirrors") / path.name for path in (root / "mirrors").iterdir() if path.is_file()
    } if (root / "mirrors").is_dir() else set()
    if actual_tables != table_files or actual_mirrors != mirror_files or {path.name for path in root.iterdir()} != {"manifest.json", "COMPLETED", "tables", "mirrors"}:
        raise ValueError("canonical guard-ablation report root contains unmanifested output")
    if marker_path.read_text().splitlines() != [f"manifest_sha256={_sha256_file(manifest_path)}"]:
        raise ValueError("canonical guard-ablation COMPLETED marker does not bind the final manifest")

    source_artifacts = manifest.get("source_artifacts")
    if not isinstance(source_artifacts, Mapping) or not {"metric_records", "panel"}.issubset(source_artifacts):
        raise ValueError("canonical guard-ablation report must pin metric-record and panel source hashes")
    if guard_input is not None:
        if not isinstance(guard_input, VerifiedGuardAblationRun):
            raise TypeError("guard_input must come from load_completed_guard_ablation_run")
        current = load_completed_guard_ablation_run(
            guard_input.run_dir,
            run_id=guard_input.run_id,
            manifest_sha256=guard_input.manifest_sha256,
        )
        if dict(source) != {"run_id": current.run_id, "manifest_sha256": current.manifest_sha256}:
            raise ValueError("canonical guard-ablation report source lineage diverges from the validated run")
        expected_sources = {
            "metric_records": {"file": current.metric_entry["file"], "sha256": current.metric_entry["sha256"]},
            "panel": {"file": current.panel_entry["file"], "sha256": current.panel_entry["sha256"]},
        }
        if current.curve_entry is not None:
            expected_sources["curves"] = {"file": current.curve_entry["file"], "sha256": current.curve_entry["sha256"]}
        if {key: dict(value) for key, value in source_artifacts.items() if isinstance(value, Mapping)} != expected_sources:
            raise ValueError("canonical guard-ablation report source table hashes diverge from the validated run")
        if factor_input is not None:
            factor_input = load_factor_report_input(
                factor_input.run_dir,
                run_id=factor_input.run_id,
                manifest_sha256=factor_input.manifest_sha256,
            )
        expected_reports = build_guard_ablation_report_tables(current, factor_input=factor_input)
        for stem, expected in expected_reports.tables.items():
            # Parquet normalizes Timestamp frequency metadata.  Compare the
            # canonical values while treating both date-bearing projections as
            # the same persisted timestamps rather than requiring an in-memory
            # DatetimeIndex frequency that Parquet cannot retain.
            expected = expected.reset_index(drop=True).copy()
            actual = loaded[stem].copy()
            for column in expected.columns:
                if column in actual.columns and (
                    column in {"date", "start", "end"} or column.endswith("_date")
                ):
                    expected[column] = pd.to_datetime(expected[column], errors="raise")
                    actual[column] = pd.to_datetime(actual[column], errors="raise")
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_freq=False)
    return manifest


def load_completed_canonical_guard_ablation_report_bundle(
    root: Path | str,
    *,
    guard_input: VerifiedGuardAblationRun | None = None,
    factor_input: VerifiedFactorRun | None = None,
) -> CanonicalGuardAblationReportBundle:
    """Validate and load all three tables from a completed diagnostic bundle."""
    root = Path(root)
    manifest = validate_canonical_guard_ablation_report_bundle(
        root, guard_input=guard_input, factor_input=factor_input
    )
    return CanonicalGuardAblationReportBundle(
        root=root,
        manifest=manifest,
        tables={
            stem: pd.read_parquet(root / str(entry["file"]))
            for stem, entry in sorted(manifest["tables"].items())
        },
    )


if __name__ == "__main__":
    main()
