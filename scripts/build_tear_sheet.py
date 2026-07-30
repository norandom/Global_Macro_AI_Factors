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


if __name__ == "__main__":
    main()
