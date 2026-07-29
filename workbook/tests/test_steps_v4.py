"""Fixture-driven S0-S5/V4 and historical-isolation tests (tasks 10.5 + 10.6).

The V4 step consumes the corrected canonical ``data-v4`` tables verbatim
(reader, legacy, differential, attribution, crisis, Factor, SJM, Markowitz)
and re-derives representative rows through the vendored root implementations:
cash-excess SSR from portfolio returns minus the aligned BIL total returns,
boundary-anchored crisis values, and the Factor differential spread. Every
fixture here is a compact schema-true construction — published rows are
computed from the same released series the step re-derives from, so agreement
is exact. Missing cash, shortened attribution, incomplete windows, schema
errors, and manifest failures must surface as VISIBLE check rows, and the
historical S0-S5 audit must stay isolated from the corrected schemas.
"""

import hashlib
import io
import json
import math

import numpy as np
import pandas as pd
import pytest

from factor_workbook import release, steps
from factor_workbook.contract import (
    ATTRIBUTION_SCHEMA,
    CRISIS_SCHEMA,
    DIFFERENTIAL_SCHEMA,
    MONTHLY_SCHEMA,
    READER_SCHEMA,
    RISK_DECOMPOSITION_SCHEMA,
    SSR_REPORT_DEFAULTS,
    SchemaError,
)
from factor_workbook.release import FetchError, Provenance, ReleaseError
from factor_workbook.steps import V4_FRAMING, build_s0, build_s5, build_v4
from factor_workbook.vendored_ssr import ssr_inference

# --------------------------------------------------------------------------- #
# deterministic synthetic release (compact, schema-true, value-consistent)     #
# --------------------------------------------------------------------------- #

IDX = pd.bdate_range("2022-01-03", periods=263, name="Date")
_RNG = np.random.default_rng(7)

PIT_ID = "factor_pit_ext2026"
NONPIT_ID = "factor_nonpit_diagnostic_ext2026"
DIFFERENTIAL_ID = "factor_nonpit_minus_pit_ext2026"
OVERLAY_BASIS = "sjm_v3_overlay_anchored_equity"
CONTROL_BASIS = "sjm_v3_control_anchored_equity"
SJM_RUN = "sjm_run_fixture"
CASH_ID = "BIL@snapshot_fixture"
CRISIS_START, CRISIS_END = pd.Timestamp("2022-06-01"), pd.Timestamp("2022-08-31")
ETFS = ("SWDA.L", "XLK", "IAU", "BIL")
MARKOWITZ_PPY = 365.2425 / 7

_SETTINGS = dict(SSR_REPORT_DEFAULTS)


def _curve(returns: np.ndarray) -> pd.Series:
    """Anchored value curve: 1.0 on the first session, cumprod afterwards."""
    return pd.Series(np.r_[1.0, np.cumprod(1.0 + returns)], index=IDX, name="value")


PIT_CURVE = _curve(_RNG.normal(0.0006, 0.010, len(IDX) - 1))
NONPIT_CURVE = _curve(
    PIT_CURVE.pct_change().dropna().to_numpy() + _RNG.normal(0.0002, 0.003, len(IDX) - 1)
)
SJM_CURVE = _curve(_RNG.normal(0.0004, 0.008, len(IDX) - 1))
CASH = pd.Series(_RNG.uniform(0.00008, 0.00016, len(IDX) - 1), index=IDX[1:], name="cash_return")
CONTROL_RETURNS = pd.Series(_RNG.normal(0.0003, 0.009, len(IDX) - 1), index=IDX[1:], name="control_return")
CONTROL_CURVE = pd.concat(
    [pd.Series([1.0], index=pd.DatetimeIndex([IDX[0]])), (1.0 + CONTROL_RETURNS).cumprod()]
).rename("value")
CONTROL_CURVE.index.name = "Date"

#: Pinned-but-arbitrary SSR block for rows the step does NOT re-derive.
STATIC_SSR = {
    "ssr_n_obs": len(IDX) - 1, "ssr_n_rolling": 11, "ssr_sr_full": 0.5,
    "ssr_mean_rolling_sr": 0.4, "ssr_sigma_hac": 0.2, "ssr_L_hac": 1,
    "ssr_ssr": 2.0, "ssr_sr_star": 0.0, "ssr_p_value": 0.2, "ssr_block_len": 2,
    "ssr_n_boot": 1000, "ssr_seed": 0, "ssr_alpha": 0.05,
    "ssr_p_value_lower": 0.8, "ssr_window": 252, "ssr_periods_per_year": 252,
}


def _ssr_block(inference) -> dict:
    result = inference.result
    return {
        "ssr_n_obs": result.n_obs, "ssr_n_rolling": result.n_rolling,
        "ssr_sr_full": result.sr_full, "ssr_mean_rolling_sr": result.mean_rolling_sr,
        "ssr_sigma_hac": result.sigma_hac, "ssr_L_hac": result.L_hac,
        "ssr_ssr": result.ssr, "ssr_sr_star": inference.sr_star,
        "ssr_p_value": inference.p_value, "ssr_block_len": inference.block_len,
        "ssr_n_boot": inference.n_boot, "ssr_seed": inference.seed,
        "ssr_alpha": inference.alpha, "ssr_p_value_lower": inference.p_value_lower,
        "ssr_window": inference.window, "ssr_periods_per_year": inference.periods_per_year,
    }


def _provenance(schema: str, pid: str, basis: str, label: str,
                start, end, n_obs: int, ppy=252) -> dict:
    return {
        "schema": schema, "portfolio_id": pid, "return_basis": basis,
        "window_label": label, "start": start, "end": end, "n_obs": n_obs,
        "periods_per_year": ppy, "cash_benchmark_id": CASH_ID,
        "currency_basis": "legacy_mixed_local_quotes", "source": f"fixture:{pid}",
    }


_METRIC_FILL = {"maxdd": -0.1, "downside_rms": 0.005, "cagr": 0.05,
                "ann_vol": 0.1, "sharpe": 0.5, "sortino": 0.7, "calmar": 0.5}


def _attribution_fields(start, end, n_obs: int) -> dict:
    return {
        "raw_market_model_kind": "raw_market_model",
        "raw_market_model_intercept_native_period": 0.0001,
        "raw_market_model_intercept_ann_arithmetic": 0.0252,
        "raw_market_model_intercept_se_hac": 0.001,
        "raw_market_model_intercept_t_hac": 0.1,
        "raw_market_model_beta": 0.9, "raw_market_model_r2": 0.8,
        "raw_market_model_n_obs": n_obs, "raw_market_model_start": start,
        "raw_market_model_end": end, "raw_market_model_periods_per_year": 252,
        "raw_market_model_hac_maxlags": 1,
    }


def _reader_row(pid: str, basis: str, curve: pd.Series, *, row_kind: str,
                ssr: dict) -> dict:
    returns = curve.pct_change().dropna()
    start, end = returns.index[0], returns.index[-1]
    row = _provenance(
        READER_SCHEMA, pid, basis, f"full {start.date()}..{end.date()}",
        start, end, len(returns),
    )
    row["row_kind"] = row_kind
    row["total_return"] = float(curve.iloc[-1] / curve.iloc[0] - 1.0)
    row.update(_METRIC_FILL)
    row.update(ssr)
    if row_kind == "full":
        row.update(_attribution_fields(start, end, len(returns)))
    return row


def _attribution_record(pid: str, basis: str, *, shortened: bool) -> dict:
    returns_index = IDX[1:]
    start = returns_index[20] if shortened else returns_index[0]
    end, n_obs = returns_index[-1], len(returns_index) - (20 if shortened else 0)
    row = _provenance(
        ATTRIBUTION_SCHEMA, pid, basis,
        f"attribution {start.date()}..{end.date()}", start, end, n_obs,
    )
    row.update(_attribution_fields(start, end, n_obs))
    return row


def _crisis_row(pid: str, basis: str, curve: pd.Series) -> dict:
    """Boundary-anchored crisis row via the canonical math, written longhand."""
    anchors = curve.loc[curve.index < CRISIS_START]
    window = curve.loc[(curve.index >= CRISIS_START) & (curve.index <= CRISIS_END)]
    episode = pd.concat([anchors.iloc[[-1]], window])
    returns = episode.pct_change().iloc[1:]
    row = _provenance(
        CRISIS_SCHEMA, pid, basis,
        f"crisis {anchors.index[-1].date()}..{window.index[-1].date()}",
        anchors.index[-1], window.index[-1], len(returns),
    )
    row.update(
        {
            "requested_start": CRISIS_START, "requested_end": CRISIS_END,
            "anchor": anchors.index[-1], "first_return_date": window.index[0],
            "actual_end": window.index[-1],
            "episode_return": float(episode.iloc[-1] / episode.iloc[0] - 1.0),
            "boundary_anchored_max_drawdown": float((episode / episode.cummax() - 1.0).min()),
            "volatility_ann": float(returns.std(ddof=1)) * math.sqrt(252),
            "n_returns": len(returns),
        }
    )
    return row


def _differential_row() -> dict:
    comparison = NONPIT_CURVE.pct_change().dropna()
    reference = PIT_CURVE.pct_change().dropna()
    spread = comparison - reference
    inference = ssr_inference(spread, **_SETTINGS)
    start, end = spread.index[0], spread.index[-1]
    row = _provenance(
        DIFFERENTIAL_SCHEMA, DIFFERENTIAL_ID, "differential_return_spread",
        f"full {start.date()}..{end.date()}", start, end, len(spread),
    )
    row["total_return"] = float((1.0 + spread).prod() - 1.0)
    row.update(_METRIC_FILL)
    row.update(_ssr_block(inference))
    row["endpoint_total_return_difference"] = float(
        (1.0 + comparison).prod() - (1.0 + reference).prod()
    )
    return row


def _sjm_reader_row(basis: str, curve: pd.Series, *, row_kind: str) -> dict:
    returns = curve.pct_change().dropna()
    excess = returns - CASH.loc[returns.index]
    pid = SJM_RUN if basis == OVERLAY_BASIS else f"{SJM_RUN}_control"
    return _reader_row(
        pid, basis, curve, row_kind=row_kind,
        ssr=_ssr_block(ssr_inference(excess, **_SETTINGS)),
    )


def _markowitz_identity(window: str) -> dict:
    return {
        "window": window, "snapshot_id": "snapshot_fixture",
        "base_currency": "USD", "valuation_rule": "friday_close_last_observation",
        "requested_start": pd.Timestamp("2016-01-01"),
        "requested_end": pd.Timestamp("2026-01-01"),
        "actual_start": pd.Timestamp("2016-01-08"),
        "actual_end": pd.Timestamp("2025-12-26"),
        "n_obs": 520, "periods_per_year": MARKOWITZ_PPY,
        "source_dates_sha256": "b" * 64,
    }


def _moments_rows(window: str) -> list[dict]:
    rows = []
    for asset in ETFS:
        row = _markowitz_identity(window)
        row.update({"asset": asset, "quote_currency": "USD", "quote_unit": "major",
                    "mean_ann_arithmetic": 0.08, "vol_ann": 0.15})
        row.update({f"cov_{name}": 0.01 for name in ETFS})
        rows.append(row)
    return rows


def _frontier_rows(window: str) -> list[dict]:
    row = _markowitz_identity(window)
    row.update({"residual_tolerance": 1e-8, "target_return_ann": 0.06,
                "success": True, "status": 0, "message": "ok", "iterations": 7,
                "objective": 0.012, "budget_residual": 0.0, "target_residual": 0.0,
                "bound_violation": 0.0, "return_ann": 0.06,
                "volatility_ann": 0.11, "feasible": True})
    row.update({f"weight_{name}": 0.25 for name in ETFS})
    return [row]


def _monthly_rows() -> list[dict]:
    row = _provenance(MONTHLY_SCHEMA, PIT_ID, "portfolio_value_curve",
                      "monthly 2022-01", IDX[1], IDX[21], 21, ppy=12)
    row.update({"year": 2022, "month": 1, "monthly_return": 0.01})
    return [row]


def _risk_rows() -> list[dict]:
    row = _provenance(RISK_DECOMPOSITION_SCHEMA, PIT_ID, "portfolio_value_curve",
                      "risk full", IDX[1], IDX[-1], len(IDX) - 1)
    row["source_schema"] = ATTRIBUTION_SCHEMA
    for name, value in _attribution_fields(IDX[1], IDX[-1], len(IDX) - 1).items():
        if name not in ("raw_market_model_n_obs", "raw_market_model_start",
                        "raw_market_model_end", "raw_market_model_periods_per_year"):
            row[name] = value
    row.update({"systematic_variance_share": 0.7, "idiosyncratic_variance_share": 0.3})
    return [row]


def _legacy_rows() -> list[dict]:
    row = _provenance(
        "portfolio_metrics.vectorbt365.v1", PIT_ID, "portfolio_value_curve",
        f"full {IDX[1].date()}..{IDX[-1].date()}", IDX[1], IDX[-1],
        len(IDX) - 1, ppy=365,
    )
    row.update({metric: 0.01 for metric in (
        "total_return", "maxdd", "downside_rms", "cagr_rows", "ann_vol_cal",
        "sharpe_cal", "sortino_cal", "calmar_rows")})
    return [row]


def _parquet(frame: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    frame.to_parquet(buf)
    return buf.getvalue()


def _series_frame(series: pd.Series, column: str) -> pd.DataFrame:
    frame = series.rename(column).to_frame()
    frame.index.name = "Date"
    return frame


def _weights_frame() -> pd.DataFrame:
    return pd.DataFrame({name: 0.25 for name in ETFS}, index=IDX)


def build_payloads() -> dict[str, bytes]:
    pit_reader = _reader_row(PIT_ID, "portfolio_value_curve", PIT_CURVE,
                             row_kind="full", ssr=STATIC_SSR)
    nonpit_reader = _reader_row(NONPIT_ID, "portfolio_value_curve", NONPIT_CURVE,
                                row_kind="performance_only", ssr=STATIC_SSR)
    differential = _differential_row()
    overlay_reader = _sjm_reader_row(OVERLAY_BASIS, SJM_CURVE, row_kind="full")
    control_reader = _sjm_reader_row(CONTROL_BASIS, CONTROL_CURVE,
                                     row_kind="performance_only")
    overlay_crisis = _crisis_row(SJM_RUN, OVERLAY_BASIS, SJM_CURVE)
    control_attribution = _attribution_record(
        f"{SJM_RUN}_control", CONTROL_BASIS, shortened=True
    )
    return {
        "portfolio_metrics_reader_ext2026.parquet": _parquet(
            pd.DataFrame([pit_reader, nonpit_reader])
        ),
        "portfolio_metrics_vectorbt365_ext2026.parquet": _parquet(
            pd.DataFrame(_legacy_rows())
        ),
        "portfolio_metrics_differential_ext2026.parquet": _parquet(
            pd.DataFrame([differential])
        ),
        "attribution_raw_market_model_ext2026.parquet": _parquet(
            pd.DataFrame(
                [
                    _attribution_record(PIT_ID, "portfolio_value_curve", shortened=False),
                    _attribution_record(NONPIT_ID, "portfolio_value_curve", shortened=True),
                ]
            )
        ),
        "crisis_metrics_ext2026.parquet": _parquet(
            pd.DataFrame(
                [
                    _crisis_row(PIT_ID, "portfolio_value_curve", PIT_CURVE),
                    _crisis_row(NONPIT_ID, "portfolio_value_curve", NONPIT_CURVE),
                ]
            )
        ),
        "tear_sheet_ai_variants_ext2026.parquet": _parquet(
            pd.DataFrame([pit_reader, nonpit_reader, differential])
        ),
        "tear_sheet_sjm_crowding_ext2026.parquet": _parquet(
            pd.DataFrame(
                [overlay_reader, control_reader, control_attribution, overlay_crisis]
            )
        ),
        "tear_sheet_trio_ext2026.parquet": _parquet(
            pd.DataFrame([pit_reader, overlay_reader])
        ),
        "monthly_returns_ext2026.parquet": _parquet(pd.DataFrame(_monthly_rows())),
        "risk_decomposition_ext2026.parquet": _parquet(pd.DataFrame(_risk_rows())),
        "markowitz_10y_moments.parquet": _parquet(pd.DataFrame(_moments_rows("10y"))),
        "markowitz_10y_frontier.parquet": _parquet(pd.DataFrame(_frontier_rows("10y"))),
        "markowitz_max_moments.parquet": _parquet(pd.DataFrame(_moments_rows("max"))),
        "markowitz_max_frontier.parquet": _parquet(pd.DataFrame(_frontier_rows("max"))),
        "factor_equity_ext2026.parquet": _parquet(_series_frame(PIT_CURVE, "value")),
        "factor_targets_ext2026.parquet": _parquet(_weights_frame()),
        "factor_nonpit_diagnostic_equity_ext2026.parquet": _parquet(
            _series_frame(NONPIT_CURVE, "value")
        ),
        "factor_nonpit_diagnostic_targets_ext2026.parquet": _parquet(_weights_frame()),
        "sjm_crowding_v3_total_return_bil_equity_ext2026.parquet": _parquet(
            _series_frame(SJM_CURVE, "value")
        ),
        "sjm_crowding_v3_total_return_bil_targets_ext2026.parquet": _parquet(
            _series_frame(pd.Series(0.5, index=IDX), "target_exposure")
        ),
        "sjm_crowding_v3_total_return_bil_daily_returns_ext2026.parquet": _parquet(
            pd.DataFrame(
                {
                    "daily_return": SJM_CURVE.pct_change().dropna(),
                    "factor_return": _RNG.normal(0.0004, 0.008, len(IDX) - 1),
                    "cash_return": CASH,
                }
            )
        ),
        "sjm_crowding_v3_total_return_bil_control_returns_ext2026.parquet": _parquet(
            _series_frame(CONTROL_RETURNS, "control_return")
        ),
    }


PAYLOADS = build_payloads()


class FakeV4Client:
    """data-v4 client stand-in with manifest-verified provenance records."""

    def __init__(self, payloads: dict[str, bytes], tag: str = "data-v4"):
        self.tag = tag
        self._payloads = payloads
        self._provenance: list[Provenance] = []

    def fetch(self, asset: str) -> tuple[bytes, Provenance]:
        data = self._payloads.get(asset)
        if data is None:
            raise ReleaseError(FetchError(asset, "missing", f"HTTP 404 for {asset}"))
        provenance = Provenance(
            tag=self.tag, asset=asset, url=f"fixture://{asset}",
            fetched_at="2026-07-29T00:00:00+00:00",
            sha256=hashlib.sha256(data).hexdigest(), from_cache=False,
            expected_sha256=hashlib.sha256(data).hexdigest(), verified=True,
            verification="publication_manifest_sha256",
        )
        self._provenance.append(provenance)
        return data, provenance

    def provenance_table(self) -> list[Provenance]:
        return list(self._provenance)


def _checks(view, fragment: str):
    return [check for check in view.checks if fragment in check.name]


@pytest.fixture(scope="module")
def v4_view():
    return build_v4(FakeV4Client(dict(PAYLOADS)))


# --------------------------------------------------------------------------- #
# canonical consumption: tables verbatim, no local alternative calculations    #
# --------------------------------------------------------------------------- #


def test_v4_step_loads_all_canonical_tables_verbatim(v4_view):
    expected = {
        "reader", "legacy", "differential", "attribution", "crisis",
        "tear_sheet_ai_variants", "tear_sheet_sjm", "tear_sheet_trio",
        "monthly_returns", "risk_decomposition",
        "markowitz_10y_moments", "markowitz_10y_frontier",
        "markowitz_max_moments", "markowitz_max_frontier",
        "factor_equity", "factor_targets", "factor_nonpit_equity",
        "factor_nonpit_targets", "sjm_equity", "sjm_targets",
        "sjm_daily_returns", "sjm_control_returns",
        "inference", "attribution_coverage", "window_coverage",
    }
    assert set(v4_view.tables) == expected
    assert v4_view.framing == V4_FRAMING
    assert v4_view.title.startswith("V4")
    # verbatim: the published reader values are displayed, never recomputed
    reader = v4_view.tables["reader"]
    published = pd.read_parquet(
        io.BytesIO(PAYLOADS["portfolio_metrics_reader_ext2026.parquet"])
    )
    pd.testing.assert_frame_equal(reader, published)
    load_checks = _checks(v4_view, "V4 load ")
    assert len(load_checks) == 22 and all(check.ok for check in load_checks)
    manifest_checks = _checks(v4_view, "manifest verification")
    assert len(manifest_checks) == 1 and manifest_checks[0].ok


def test_v4_happy_fixture_has_every_check_green(v4_view):
    failed = [check.name for check in v4_view.checks if not check.ok]
    assert failed == []


# --------------------------------------------------------------------------- #
# cash-excess SSR: portfolio returns minus aligned BIL total returns           #
# --------------------------------------------------------------------------- #


def test_v4_cash_excess_ssr_agrees_and_surfaces_metadata(v4_view):
    for line in ("sjm overlay", "sjm control"):
        ssr_checks = [
            check for check in _checks(v4_view, f"V4 {line} ssr_") if "vs published" in check.name
        ]
        assert len(ssr_checks) == 16, line
        assert all(check.ok for check in ssr_checks), line
        coverage = _checks(v4_view, f"V4 {line} cash coverage")
        assert len(coverage) == 1 and coverage[0].ok
        total = _checks(v4_view, f"V4 {line} total_return vs published")
        assert len(total) == 1 and total[0].ok
    inference = v4_view.tables["inference"]
    assert set(inference["line"]) == {"sjm overlay", "sjm control", "factor differential"}
    row = inference[inference["line"] == "sjm overlay"].iloc[0]
    # all deterministic inference metadata surfaced with the row
    assert (row["ssr_window"], row["ssr_n_boot"], row["ssr_seed"]) == (252, 1000, 0)
    assert (row["ssr_sr_star"], row["ssr_alpha"]) == (0.0, 0.05)
    assert row["ssr_block_len"] >= 1
    assert math.isfinite(row["ssr_p_value"]) and math.isfinite(row["ssr_p_value_lower"])
    assert isinstance(row["verdict"], str) and "SSR=" in row["verdict"]


def test_v4_missing_cash_is_a_visible_check_not_an_exception():
    truncated = CASH.iloc[:-40]
    payloads = dict(PAYLOADS)
    payloads["sjm_crowding_v3_total_return_bil_daily_returns_ext2026.parquet"] = _parquet(
        pd.DataFrame(
            {
                "daily_return": SJM_CURVE.pct_change().dropna().iloc[:-40],
                "factor_return": 0.0,
                "cash_return": truncated,
            }
        )
    )
    view = build_v4(FakeV4Client(payloads))
    coverage = _checks(view, "V4 sjm overlay cash coverage")
    assert len(coverage) == 1 and not coverage[0].ok
    assert "40" in coverage[0].message
    # without aligned cash the SSR is never constructed from substitute data
    assert _checks(view, "V4 sjm overlay ssr_") == []
    assert "sjm overlay" not in set(view.tables["inference"]["line"])


# --------------------------------------------------------------------------- #
# crisis values and boundary identity                                          #
# --------------------------------------------------------------------------- #


def test_v4_crisis_values_and_boundaries_agree(v4_view):
    for line in ("sjm overlay", f"factor {PIT_ID}", f"factor {NONPIT_ID}"):
        for field in ("episode_return", "boundary_anchored_max_drawdown", "volatility_ann"):
            matches = _checks(v4_view, f"V4 {line} crisis {field}")
            assert len(matches) == 1 and matches[0].ok, (line, field)
        boundary = _checks(v4_view, f"V4 {line} crisis boundary (1 =")
        assert len(boundary) == 1 and boundary[0].ok, line


def test_v4_incomplete_windows_are_disclosed(v4_view):
    coverage = v4_view.tables["window_coverage"]
    kinds = set(coverage["kind"])
    assert kinds == {"crisis", "markowitz"}
    markowitz = coverage[coverage["kind"] == "markowitz"]
    assert bool(markowitz["incomplete"].all())  # actual sits inside requested
    crisis = coverage[coverage["kind"] == "crisis"]
    assert not bool(crisis["incomplete"].any())  # series covers the request


def test_v4_wrongly_shortened_crisis_window_fails_the_boundary_check():
    """A published crisis row claiming an earlier end than the released series
    supports is an incomplete-window defect and must go red."""
    row = _crisis_row(SJM_RUN, OVERLAY_BASIS, SJM_CURVE)
    early = SJM_CURVE.index[SJM_CURVE.index <= CRISIS_END][-5]
    row.update({"actual_end": early, "end": early})
    payloads = dict(PAYLOADS)
    payloads["tear_sheet_sjm_crowding_ext2026.parquet"] = _parquet(
        pd.DataFrame(
            [
                _sjm_reader_row(OVERLAY_BASIS, SJM_CURVE, row_kind="full"),
                _sjm_reader_row(CONTROL_BASIS, CONTROL_CURVE, row_kind="performance_only"),
                _attribution_record(f"{SJM_RUN}_control", CONTROL_BASIS, shortened=True),
                row,
            ]
        )
    )
    view = build_v4(FakeV4Client(payloads))
    boundary = _checks(view, "V4 sjm overlay crisis boundary (1 =")
    assert len(boundary) == 1 and not boundary[0].ok
    assert "actual_end" in boundary[0].message


# --------------------------------------------------------------------------- #
# Factor and differential views                                                #
# --------------------------------------------------------------------------- #


def test_v4_factor_and_differential_views_agree(v4_view):
    for pid in (PIT_ID, NONPIT_ID):
        for field in ("n_obs", "total_return"):
            matches = _checks(v4_view, f"V4 factor {pid} {field} vs released series")
            assert len(matches) == 1 and matches[0].ok, (pid, field)
    diff_ssr = [
        check
        for check in _checks(v4_view, "V4 factor differential ssr_")
        if "vs published" in check.name
    ]
    assert len(diff_ssr) == 16 and all(check.ok for check in diff_ssr)
    for name in ("endpoint_total_return_difference", "total_return"):
        matches = _checks(v4_view, f"V4 factor differential {name} vs published")
        assert len(matches) == 1 and matches[0].ok, name
    alignment = _checks(v4_view, "V4 factor differential session alignment")
    assert len(alignment) == 1 and alignment[0].ok


# --------------------------------------------------------------------------- #
# shortened attribution                                                        #
# --------------------------------------------------------------------------- #


def test_v4_shortened_attribution_is_surfaced_and_missing_records_flagged():
    view = build_v4(FakeV4Client(dict(PAYLOADS)))
    coverage_checks = _checks(view, "V4 attribution coverage")
    assert len(coverage_checks) == 1 and coverage_checks[0].ok
    table = view.tables["attribution_coverage"]
    nonpit = table[table["portfolio_id"] == NONPIT_ID]
    assert (nonpit["row_kind"] == "performance_only").all()
    assert (nonpit["attribution"] == "separate_record").all()
    pit = table[(table["portfolio_id"] == PIT_ID) & (table["table"] == "reader")]
    assert (pit["attribution"] == "inline_full_window").all()

    # drop the nonpit attribution record everywhere -> the check goes red
    payloads = dict(PAYLOADS)
    payloads["attribution_raw_market_model_ext2026.parquet"] = _parquet(
        pd.DataFrame([_attribution_record(PIT_ID, "portfolio_value_curve", shortened=False)])
    )
    flagged = build_v4(FakeV4Client(payloads))
    coverage_checks = _checks(flagged, "V4 attribution coverage")
    assert len(coverage_checks) == 1 and not coverage_checks[0].ok
    assert NONPIT_ID in coverage_checks[0].message


# --------------------------------------------------------------------------- #
# schema errors, missing assets, cross-tag isolation                           #
# --------------------------------------------------------------------------- #


def test_v4_schema_and_missing_asset_failures_are_visible_checks():
    payloads = dict(PAYLOADS)
    corrupt = pd.read_parquet(
        io.BytesIO(payloads["portfolio_metrics_reader_ext2026.parquet"])
    ).assign(periods_per_year=365)
    payloads["portfolio_metrics_reader_ext2026.parquet"] = _parquet(corrupt)
    del payloads["monthly_returns_ext2026.parquet"]

    view = build_v4(FakeV4Client(payloads))
    reader_check = _checks(view, "V4 load portfolio_metrics_reader_ext2026")
    assert len(reader_check) == 1 and not reader_check[0].ok
    assert "periods_per_year" in reader_check[0].message
    monthly_check = _checks(view, "V4 load monthly_returns_ext2026")
    assert len(monthly_check) == 1 and not monthly_check[0].ok
    assert "missing" in monthly_check[0].message
    assert "reader" not in view.tables and "monthly_returns" not in view.tables
    # dependent factor re-derivations degrade to a visible inputs check
    inputs = _checks(view, "V4 factor re-derivation inputs")
    assert len(inputs) == 1 and not inputs[0].ok


def test_v4_cross_tag_isolation_in_both_directions():
    with pytest.raises(SchemaError, match="data-v2"):
        build_v4(FakeV4Client(dict(PAYLOADS), tag="data-v2"))
    v4_client = FakeV4Client(dict(PAYLOADS))
    with pytest.raises(SchemaError, match="data-v4"):
        build_s0(v4_client)
    with pytest.raises(SchemaError, match="data-v4"):
        build_s5(v4_client)


# --------------------------------------------------------------------------- #
# real release client: manifest success, stale bytes, unmanifested assets      #
# --------------------------------------------------------------------------- #


def _url(asset: str) -> str:
    return (
        "https://github.com/norandom/Global_Macro_AI_Factors/releases/download/"
        f"data-v4/{asset}"
    )


def _manifest_bytes(assets: dict[str, bytes]) -> bytes:
    return json.dumps(
        {
            "schema_id": "publication_manifest.v1",
            "release_tag": "data-v4",
            "completed": True,
            "artifacts": [
                {"path": name, "sha256": hashlib.sha256(data).hexdigest(),
                 "size": len(data)}
                for name, data in sorted(assets.items())
            ],
        }
    ).encode("utf-8")


def _install_transport(monkeypatch, served: dict[str, bytes]):
    class FakeResponse:
        def __init__(self, status_code: int, content: bytes = b""):
            self.status_code = status_code
            self.content = content

    def fake_get(url, headers=None, timeout=None, **kwargs):
        asset = url.rsplit("/", 1)[1]
        if url == _url(asset) and asset in served:
            return FakeResponse(200, served[asset])
        return FakeResponse(404)

    monkeypatch.setattr(release.requests, "get", fake_get)


def test_v4_real_client_manifest_success_stale_and_unmanifested(monkeypatch, tmp_path):
    manifested = {
        name: data for name, data in PAYLOADS.items()
        if name != "monthly_returns_ext2026.parquet"  # unmanifested asset
    }
    served = dict(PAYLOADS)
    served["publication_manifest.json"] = _manifest_bytes(manifested)
    # stale bytes: served frontier disagrees with its manifest hash
    served["markowitz_10y_frontier.parquet"] = served["markowitz_10y_frontier.parquet"] + b"x"
    _install_transport(monkeypatch, served)

    client = release.ReleaseClient(
        "data-v4", cache_dir=tmp_path, token_provider=lambda: None
    )
    view = build_v4(client)

    monthly = _checks(view, "V4 load monthly_returns_ext2026")
    assert len(monthly) == 1 and not monthly[0].ok
    assert "not listed" in monthly[0].message
    stale = _checks(view, "V4 load markowitz_10y_frontier")
    assert len(stale) == 1 and not stale[0].ok
    assert "integrity" in stale[0].message or "sha256" in stale[0].message
    assert not (tmp_path / "data-v4" / "markowitz_10y_frontier.parquet").exists()
    # everything manifest-verified loads and every value re-derivation agrees
    other_failures = [
        check.name
        for check in view.checks
        if not check.ok
        and "monthly_returns_ext2026" not in check.name
        and "markowitz_10y_frontier" not in check.name
    ]
    assert other_failures == []
    manifest_checks = _checks(view, "manifest verification")
    assert len(manifest_checks) == 1 and manifest_checks[0].ok


def test_v4_manifest_failure_fails_every_load_visibly(monkeypatch, tmp_path):
    _install_transport(monkeypatch, dict(PAYLOADS))  # no manifest served
    client = release.ReleaseClient(
        "data-v4", cache_dir=tmp_path, token_provider=lambda: None
    )
    view = build_v4(client)
    load_checks = _checks(view, "V4 load ")
    assert len(load_checks) == 22
    assert all(not check.ok for check in load_checks)
    assert all("publication_manifest" in check.message for check in load_checks)
    assert view.tables.get("reader") is None
    manifest_checks = _checks(view, "manifest verification")
    assert len(manifest_checks) == 1 and not manifest_checks[0].ok
