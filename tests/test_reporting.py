"""Report-row schemas, the pre-emission gate (task 4.1), and row builders (task 4.2)."""

from __future__ import annotations

import dataclasses
import functools
import math

import numpy as np
import pandas as pd
import pytest

from macro_framework.evaluation import cagr, cagr_rows, metric_block
from macro_framework.reporting import (
    ATTRIBUTION_SCHEMA,
    CRISIS_SCHEMA,
    DIFFERENTIAL_SCHEMA,
    LEGACY_SCHEMA,
    MONTHLY_SCHEMA,
    READER_SCHEMA,
    REPORT_SCHEMAS,
    REQUIRED_PROVENANCE,
    LineMetadata,
    build_legacy_metric_row,
    build_reader_metric_row,
    validate_report_row,
)
from macro_framework.skill_metric import portfolio_excess_returns, raw_market_model_attribution
from macro_framework.ssr import ssr_inference


def _provenance(schema: str, periods: int) -> dict:
    return {
        "schema": schema,
        "portfolio_id": "factor_v2_ext2026",
        "return_basis": "total_return",
        "window_label": "2016-01-04..2026-06-30",
        "start": "2016-01-04",
        "end": "2026-06-30",
        "n_obs": 2571,
        "periods_per_year": periods,
        "cash_benchmark_id": "BIL",
        "currency_basis": "legacy_mixed_local_quotes",
        "source": "scripts/build_tear_sheet.py",
    }


def _reader_row() -> dict:
    return _provenance(READER_SCHEMA, 252) | {"row_kind": "full", "cagr": 0.11, "sharpe": 1.2}


def _legacy_row() -> dict:
    return _provenance(LEGACY_SCHEMA, 365) | {"cagr_rows": 0.12, "sharpe_cal": 1.4}


def test_six_schema_identities_are_distinct_and_pinned():
    ids = {
        READER_SCHEMA,
        LEGACY_SCHEMA,
        DIFFERENTIAL_SCHEMA,
        ATTRIBUTION_SCHEMA,
        CRISIS_SCHEMA,
        MONTHLY_SCHEMA,
    }
    assert len(ids) == 6
    # the three identities pinned verbatim by the approved design
    assert READER_SCHEMA == "portfolio_metrics.reader.v2"
    assert LEGACY_SCHEMA == "portfolio_metrics.vectorbt365.v1"
    assert DIFFERENTIAL_SCHEMA == "portfolio_metrics.differential.v2"


def test_valid_reader_and_legacy_rows_pass_the_gate():
    assert validate_report_row(_reader_row())["schema"] == READER_SCHEMA
    assert validate_report_row(_legacy_row())["periods_per_year"] == 365


@pytest.mark.parametrize("key", REQUIRED_PROVENANCE)
def test_missing_provenance_fails_before_emission(key):
    row = _reader_row()
    del row[key]
    with pytest.raises(ValueError, match=key):
        validate_report_row(row)


def test_unknown_schema_identity_is_rejected():
    with pytest.raises(ValueError, match="unknown report schema"):
        validate_report_row(_reader_row() | {"schema": "portfolio_metrics.reader.v1"})
    # unhashable schema values fail with the same clean error, not a bare TypeError
    with pytest.raises(ValueError, match="unknown report schema"):
        validate_report_row(_reader_row() | {"schema": ["portfolio_metrics.reader.v2"]})


@pytest.mark.parametrize(
    "key", ["portfolio_id", "return_basis", "window_label", "cash_benchmark_id", "source"]
)
@pytest.mark.parametrize("bad", ["", "   ", float("nan"), 7])
def test_blank_nan_or_non_string_provenance_fails_before_emission(key, bad):
    # "" and NaN are the normal pandas/CSV missing-value forms — they must not
    # count as provenance (R7.3, R7.4)
    with pytest.raises(ValueError, match=key):
        validate_report_row(_reader_row() | {key: bad})


def test_legacy_annualization_basis_is_explicit_and_never_the_reader_convention():
    # the identity pins the basis: reader=252, legacy=365 — a swapped basis is rejected
    with pytest.raises(ValueError, match="365"):
        validate_report_row(_legacy_row() | {"periods_per_year": 252})
    with pytest.raises(ValueError, match="252"):
        validate_report_row(_reader_row() | {"periods_per_year": 365})
    # a legacy-named value cannot ride in a reader row, nor a bare name in legacy
    with pytest.raises(ValueError, match="not part of"):
        validate_report_row(_reader_row() | {"ann_vol_cal": 0.2})
    with pytest.raises(ValueError, match="not part of"):
        validate_report_row(_legacy_row() | {"sharpe": 1.2})


def test_ac_1_2():
    test_legacy_annualization_basis_is_explicit_and_never_the_reader_convention()


def test_annualization_alternatives_stay_separately_named():
    reader = validate_report_row(_reader_row())
    legacy = validate_report_row(_legacy_row())
    assert reader["schema"] != legacy["schema"]
    assert "sharpe" in reader and "sharpe" not in legacy
    assert "sharpe_cal" in legacy and "sharpe_cal" not in reader
    # the basis-named vocabularies are disjoint outside the basis-free fields
    reader_only = REPORT_SCHEMAS[READER_SCHEMA].fields - REPORT_SCHEMAS[LEGACY_SCHEMA].fields
    legacy_only = REPORT_SCHEMAS[LEGACY_SCHEMA].fields - REPORT_SCHEMAS[READER_SCHEMA].fields
    assert {"cagr", "ann_vol", "sharpe", "sortino", "calmar"} <= reader_only
    assert {"cagr_rows", "ann_vol_cal", "sharpe_cal", "sortino_cal", "calmar_rows"} <= legacy_only


def test_ac_1_6():
    test_annualization_alternatives_stay_separately_named()


@pytest.mark.parametrize("label", ["capm_alpha_ann", "jensen_alpha", "r2_capm"])
def test_capm_and_jensen_labels_are_prohibited(label):
    with pytest.raises(ValueError, match="raw market-model"):
        validate_report_row(_reader_row() | {label: 0.01})


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("n_obs", 0, "positive integer"),
        ("n_obs", True, "positive integer"),
        ("periods_per_year", 2.5, "positive integer"),
        ("start", "2027-01-01", "start must be on or before end"),
        ("end", "not-a-date", "timestamp"),
        ("currency_basis", "EUR", "currency_basis"),
    ],
)
def test_malformed_provenance_values_are_rejected(field, value, match):
    with pytest.raises(ValueError, match=match):
        validate_report_row(_reader_row() | {field: value})


def test_undeclared_fields_are_rejected_per_schema():
    with pytest.raises(ValueError, match="not part of"):
        validate_report_row(_reader_row() | {"bogus_metric": 1.0})
    # crisis fields cannot ride in a legacy row
    with pytest.raises(ValueError, match="not part of"):
        validate_report_row(_legacy_row() | {"boundary_anchored_max_drawdown": -0.2})
    # row_kind belongs to reader rows only
    with pytest.raises(ValueError, match="not part of"):
        validate_report_row(_legacy_row() | {"row_kind": "full"})


def test_schema_vocabularies_project_the_shared_finance_contracts():
    assert "raw_market_model_intercept_ann_arithmetic" in REPORT_SCHEMAS[ATTRIBUTION_SCHEMA].fields
    assert "raw_market_model_beta" in REPORT_SCHEMAS[READER_SCHEMA].fields
    assert "ssr_p_value" in REPORT_SCHEMAS[READER_SCHEMA].fields
    assert "ssr_p_value" in REPORT_SCHEMAS[DIFFERENTIAL_SCHEMA].fields
    assert "endpoint_total_return_difference" in REPORT_SCHEMAS[DIFFERENTIAL_SCHEMA].fields
    assert "boundary_anchored_max_drawdown" in REPORT_SCHEMAS[CRISIS_SCHEMA].fields
    assert {"year", "month", "monthly_return"} <= REPORT_SCHEMAS[MONTHLY_SCHEMA].fields


def test_reader_rows_require_a_valid_row_kind():
    row = _reader_row()
    del row["row_kind"]
    with pytest.raises(ValueError, match="row_kind"):
        validate_report_row(row)
    with pytest.raises(ValueError, match="row_kind"):
        validate_report_row(_reader_row() | {"row_kind": "partial"})
    ok = validate_report_row(_reader_row() | {"row_kind": "performance_only"})
    assert ok["row_kind"] == "performance_only"


def test_line_metadata_is_frozen_with_validated_fields():
    meta = LineMetadata(
        portfolio_id="factor_v2_ext2026",
        label="Factor (PIT)",
        window_label="2016-01-04..2026-06-30",
        currency_basis="USD",
        total_return_basis="total_return",
        cash_benchmark_id="BIL",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.label = "x"  # type: ignore[misc]
    with pytest.raises(ValueError, match="currency_basis"):
        dataclasses.replace(meta, currency_basis="EUR")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="portfolio_id"):
        dataclasses.replace(meta, portfolio_id=" ")


# --- Task 4.2: reader-facing and explicit legacy row builders ---------------------


@functools.lru_cache(maxsize=None)
def _line(n: int = 320, seed: int = 7):
    """One deterministic portfolio line with every builder input precomputed."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2016-01-04", periods=n)
    value = pd.Series((1 + rng.normal(6e-4, 8e-3, n)).cumprod() * 100.0, index=idx)
    metrics = metric_block(value)
    port = metrics["returns"]
    cash = pd.Series(1e-4, index=port.index)
    excess = portfolio_excess_returns(port, cash)
    ssr = ssr_inference(excess, n_boot=25, seed=0)
    market = pd.Series(rng.normal(4e-4, 9e-3, n - 1), index=port.index)
    attr = raw_market_model_attribution(port, market)
    meta = LineMetadata(
        portfolio_id="factor_v2_ext2026",
        label="Factor (PIT)",
        window_label="2016-01-04..2017-03-27",
        currency_basis="legacy_mixed_local_quotes",
        total_return_basis="total_return",
        cash_benchmark_id="BIL",
    )
    return meta, value, metrics, cash, excess, ssr, attr, market


def test_reader_row_uses_elapsed_cagr_and_252_conventions():
    meta, value, metrics, cash, excess, ssr, attr, _ = _line()
    port = metrics["returns"]

    row = build_reader_metric_row(
        meta, metrics, cash, ssr, source="tests/test_reporting.py", attribution=attr
    )

    assert row["schema"] == READER_SCHEMA
    assert row["periods_per_year"] == 252
    assert (row["start"], row["end"], row["n_obs"]) == (port.index[0], port.index[-1], len(port))
    # elapsed-time CAGR, never row-count growth (R1.1)
    assert row["cagr"] == pytest.approx(cagr(value))
    assert row["cagr"] != pytest.approx(cagr_rows(value))
    assert row["ann_vol"] == pytest.approx(float(port.std(ddof=1)) * math.sqrt(252))
    # Sharpe/Sortino from the ONE validated BIL-excess stream (R2.1), not raw returns
    expected_sharpe = float(excess.mean() / excess.std(ddof=1)) * math.sqrt(252)
    downside = np.minimum(excess.to_numpy(dtype=float), 0.0)
    expected_sortino = float(excess.mean() / np.sqrt(np.mean(downside**2))) * math.sqrt(252)
    assert row["sharpe"] == pytest.approx(expected_sharpe)
    assert row["sortino"] == pytest.approx(expected_sortino)
    assert row["sharpe"] != pytest.approx(metrics["sharpe"])
    # complete deterministic SSR metadata (R2.8)
    assert row["ssr_ssr"] == ssr.result.ssr
    assert row["ssr_p_value"] == ssr.p_value
    assert (row["ssr_n_boot"], row["ssr_seed"], row["ssr_alpha"]) == (25, 0, 0.05)
    assert (row["ssr_window"], row["ssr_periods_per_year"]) == (252, 252)
    assert row["ssr_n_obs"] == len(excess)
    # no legacy names, no non-scalar passthrough
    assert set(row).isdisjoint({"cagr_rows", "ann_vol_cal", "sharpe_cal", "sortino_cal", "calmar_rows"})
    assert "returns" not in row and "dd" not in row


def test_ac_1_1():
    test_reader_row_uses_elapsed_cagr_and_252_conventions()


def test_full_rows_use_one_window_stream_and_observation_set():
    meta, value, metrics, cash, excess, ssr, attr, market = _line()
    port = metrics["returns"]

    full = build_reader_metric_row(
        meta, metrics, cash, ssr, source="tests/test_reporting.py", attribution=attr
    )
    assert full["row_kind"] == "full"
    assert (full["raw_market_model_start"], full["raw_market_model_end"]) == (full["start"], full["end"])
    assert full["raw_market_model_n_obs"] == full["n_obs"]
    assert full["raw_market_model_beta"] == attr.beta
    assert full["raw_market_model_kind"] == "raw_market_model"

    # shorter attribution coverage degrades to performance_only, never a mixed full row
    short_attr = raw_market_model_attribution(port.iloc[:-30], market.iloc[:-30])
    short = build_reader_metric_row(
        meta, metrics, cash, ssr, source="tests/test_reporting.py", attribution=short_attr
    )
    assert short["row_kind"] == "performance_only"
    assert not any(key.startswith("raw_market_model_") for key in short)
    none = build_reader_metric_row(meta, metrics, cash, ssr, source="t", attribution=None)
    assert none["row_kind"] == "performance_only"

    # mismatched streams are rejected, not silently repaired (R1.3)
    with pytest.raises(ValueError, match="identical indexes"):
        build_reader_metric_row(meta, metrics, cash.iloc[:-1], ssr, source="t")
    with pytest.raises(ValueError, match="same validated cash-excess stream"):
        build_reader_metric_row(
            meta, metrics, cash, ssr_inference(excess.iloc[:-5], n_boot=10), source="t"
        )
    with pytest.raises(ValueError, match="252-day convention"):
        build_reader_metric_row(
            meta,
            metrics,
            cash,
            ssr,
            source="t",
            attribution=raw_market_model_attribution(port, market, periods_per_year=365),
        )


def test_ac_1_3():
    test_full_rows_use_one_window_stream_and_observation_set()


def test_reader_row_rejects_ssr_from_the_raw_return_stream():
    # the historical defect class (R2.1): equal-length RAW-stream SSR must fail
    # the stream-identity gate, never emit a self-contradictory row
    meta, value, metrics, cash, excess, ssr, attr, _ = _line()
    raw_ssr = ssr_inference(metrics["returns"], n_boot=10, seed=0)
    with pytest.raises(ValueError, match="cash-excess stream"):
        build_reader_metric_row(meta, metrics, cash, raw_ssr, source="t")


def test_reader_row_rejects_nan_bearing_cash():
    meta, value, metrics, cash, excess, ssr, attr, _ = _line()
    bad_cash = cash.copy()
    bad_cash.iloc[5] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build_reader_metric_row(meta, metrics, bad_cash, ssr, source="t")


def test_reader_row_preserves_insufficient_inference_carveout():
    # a valid short line: SSR keeps the documented insufficient result (NaN
    # sr_full/p-values) and the row still emits with NaN ssr_* metadata
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2016-01-04", periods=100)
    value = pd.Series((1 + rng.normal(6e-4, 8e-3, 100)).cumprod() * 100.0, index=idx)
    metrics = metric_block(value)
    cash = pd.Series(1e-4, index=metrics["returns"].index)
    ssr = ssr_inference(
        portfolio_excess_returns(metrics["returns"], cash), window=95, n_boot=10
    )
    assert ssr.result.n_rolling < 10

    meta, *_ = _line()
    row = build_reader_metric_row(meta, metrics, cash, ssr, source="t")
    assert row["row_kind"] == "performance_only"
    assert math.isnan(row["ssr_ssr"]) and math.isnan(row["ssr_p_value"])


def test_reader_and_legacy_annualization_are_separate():
    # defect 2 shared boundary, fixture class: stale_artifacts
    meta, value, metrics, cash, excess, ssr, attr, _ = _line()

    reader = build_reader_metric_row(
        meta, metrics, cash, ssr, source="tests/test_reporting.py", attribution=attr
    )
    legacy = build_legacy_metric_row(meta, metrics, source="tests/test_reporting.py")

    assert legacy["schema"] == LEGACY_SCHEMA
    assert legacy["periods_per_year"] == 365
    assert legacy["cagr_rows"] == pytest.approx(cagr_rows(value))
    assert legacy["sharpe_cal"] == metrics["sharpe_cal"]
    # the two conventions reproduce their distinct formulas
    assert legacy["cagr_rows"] != pytest.approx(reader["cagr"])
    assert legacy["ann_vol_cal"] == pytest.approx(reader["ann_vol"] * math.sqrt(365 / 252))
    # completion criterion: no shared ambiguous field names beyond provenance + basis-free
    shared = set(reader) & set(legacy)
    assert shared == set(REQUIRED_PROVENANCE) | {"total_return", "maxdd", "downside_rms"}


# --- Task 4.3: one coherent differential-reporting contract -----------------------


@functools.lru_cache(maxsize=None)
def _differential_line(seed: int = 13, n: int = 300):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-02", periods=n)
    comparison = pd.Series(rng.normal(8e-4, 9e-3, n), index=idx, name="comparison")
    reference = pd.Series(rng.normal(4e-4, 8e-3, n), index=idx, name="reference")
    spread = comparison - reference
    ssr = ssr_inference(pd.Series(spread.to_numpy(), index=idx), n_boot=25, seed=0)
    meta, *_ = _line()
    return meta, comparison, reference, spread, ssr


def test_differential_statistics_share_one_daily_spread():
    # defect 9 shared boundary: every statistic derives from ONE daily spread
    from macro_framework.reporting import build_differential_metric_row

    meta, comparison, reference, spread, ssr = _differential_line()
    row = build_differential_metric_row(meta, comparison, reference, ssr, source="t")

    assert row["schema"] == DIFFERENTIAL_SCHEMA
    assert row["periods_per_year"] == 252
    assert (row["start"], row["end"], row["n_obs"]) == (
        spread.index[0], spread.index[-1], len(spread)
    )
    assert row["total_return"] == pytest.approx(float((1 + spread).prod() - 1))
    assert row["sharpe"] == pytest.approx(
        float(spread.mean() / spread.std(ddof=1)) * math.sqrt(252)
    )
    assert row["ann_vol"] == pytest.approx(float(spread.std(ddof=1)) * math.sqrt(252))
    assert row["ssr_ssr"] == ssr.result.ssr and row["ssr_p_value"] == ssr.p_value

    # changing EITHER source line changes every spread statistic via one producer
    bumped_ref = reference * 1.5
    bumped_spread = comparison - bumped_ref
    bumped_ssr = ssr_inference(
        pd.Series(bumped_spread.to_numpy(), index=spread.index), n_boot=25, seed=0
    )
    bumped = build_differential_metric_row(meta, comparison, bumped_ref, bumped_ssr, source="t")
    for key in ("total_return", "cagr", "ann_vol", "sharpe", "maxdd", "ssr_ssr"):
        assert bumped[key] != pytest.approx(row[key]), key

    # an SSR from either RAW source line (equal length) is rejected, not emitted
    with pytest.raises(ValueError, match="differential spread"):
        build_differential_metric_row(
            meta, comparison, reference, ssr_inference(comparison, n_boot=10), source="t"
        )
    # misaligned reference fails the exact-index contract
    with pytest.raises(ValueError, match="identical indexes"):
        build_differential_metric_row(meta, comparison, reference.iloc[:-1], ssr, source="t")
    # a spread return at or below -100% cannot be compounded
    broken_ref = reference.copy()
    broken_ref.iloc[10] = comparison.iloc[10] + 1.5
    with pytest.raises(ValueError, match="-100%"):
        build_differential_metric_row(meta, comparison, broken_ref, ssr, source="t")


def test_ac_1_4():
    test_differential_statistics_share_one_daily_spread()


def test_endpoint_wealth_difference_is_a_separately_named_descriptive_field():
    from macro_framework.reporting import build_differential_metric_row

    meta, comparison, reference, spread, ssr = _differential_line()
    row = build_differential_metric_row(meta, comparison, reference, ssr, source="t")

    expected_gap = float((1 + comparison).prod() - (1 + reference).prod())
    assert row["endpoint_total_return_difference"] == pytest.approx(expected_gap)
    # the endpoint gap is NOT the spread portfolio's total return
    assert row["endpoint_total_return_difference"] != pytest.approx(row["total_return"])
    # and the endpoint field cannot ride in a single-portfolio schema
    with pytest.raises(ValueError, match="not part of"):
        validate_report_row(_reader_row() | {"endpoint_total_return_difference": expected_gap})
    with pytest.raises(ValueError, match="not part of"):
        validate_report_row(_legacy_row() | {"endpoint_total_return_difference": expected_gap})


def test_ac_1_5():
    test_endpoint_wealth_difference_is_a_separately_named_descriptive_field()


# --- Task 4.4: separate window records --------------------------------------------


def test_short_attribution_is_emitted_separately():
    # defect 10 shared boundary: shortened attribution never rides in a full row;
    # it becomes its own record with the actual window and model identity (R3.7)
    from macro_framework.reporting import build_attribution_record

    meta, value, metrics, cash, excess, ssr, attr, market = _line()
    port = metrics["returns"]
    short_attr = raw_market_model_attribution(port.iloc[:-30], market.iloc[:-30])

    reader = build_reader_metric_row(
        meta, metrics, cash, ssr, source="t", attribution=short_attr
    )
    assert reader["row_kind"] == "performance_only"
    assert not any(key.startswith("raw_market_model_") for key in reader)

    record = build_attribution_record(meta, short_attr, source="t")
    assert record["schema"] == ATTRIBUTION_SCHEMA
    assert (record["start"], record["end"], record["n_obs"]) == (
        short_attr.start, short_attr.end, short_attr.n_obs
    )
    assert record["periods_per_year"] == short_attr.periods_per_year
    assert record["raw_market_model_kind"] == "raw_market_model"
    assert record["raw_market_model_beta"] == short_attr.beta
    assert record["end"] != reader["end"]  # the shortened window is explicit


def test_gate_rejects_full_rows_with_mixed_attribution_windows():
    meta, value, metrics, cash, excess, ssr, attr, _ = _line()
    full = build_reader_metric_row(meta, metrics, cash, ssr, source="t", attribution=attr)

    # attribution end differing from performance coverage cannot claim "full"
    with pytest.raises(ValueError, match="mixed windows"):
        validate_report_row(full | {"raw_market_model_end": full["start"]})
    with pytest.raises(ValueError, match="mixed windows"):
        validate_report_row(full | {"raw_market_model_n_obs": full["n_obs"] - 30})
    # attribution fields on a performance_only row are a contradiction
    with pytest.raises(ValueError, match="row_kind 'full'"):
        validate_report_row(full | {"row_kind": "performance_only"})
    # a full row cannot assert coherence without the window-binding triple
    stripped = {k: v for k, v in full.items() if k != "raw_market_model_end"}
    with pytest.raises(ValueError, match="raw_market_model_start/end/n_obs"):
        validate_report_row(stripped)


def test_crisis_record_projects_the_shared_result_verbatim():
    import dataclasses as dc

    from macro_framework.evaluation import crisis_metrics
    from macro_framework.reporting import build_crisis_record

    meta, value, *_ = _line()
    crisis = crisis_metrics(value, "2016-06-01", "2016-08-31")
    record = build_crisis_record(meta, crisis, source="t")

    assert record["schema"] == CRISIS_SCHEMA
    for field in dc.fields(crisis):
        assert record[field.name] == getattr(crisis, field.name), field.name
    assert (record["start"], record["end"], record["n_obs"]) == (
        crisis.anchor, crisis.actual_end, crisis.n_returns
    )


# --- Task 4.5: assembled tables and monthly-return semantics ----------------------


def test_monthly_rows_derive_from_the_performance_return_stream():
    from macro_framework.reporting import build_monthly_return_rows

    meta, value, metrics, cash, excess, ssr, attr, _ = _line()
    rows = build_monthly_return_rows(meta, metrics, source="t")

    assert all(row["schema"] == MONTHLY_SCHEMA for row in rows)
    assert all(row["periods_per_year"] == 12 for row in rows)
    assert sum(row["n_obs"] for row in rows) == len(metrics["returns"])
    compounded = 1.0
    for row in rows:
        compounded *= 1.0 + row["monthly_return"]
    assert compounded - 1.0 == pytest.approx(metrics["total_return"])
    # per-month provenance windows are the actual month boundaries
    first = rows[0]
    chunk = metrics["returns"].loc[str(first["start"].date())[:7]]
    assert first["n_obs"] == len(chunk)


def test_report_table_round_trips_valid_rows_with_stable_semantics():
    from macro_framework.reporting import (
        build_legacy_metric_row,
        build_monthly_return_rows,
        report_table,
    )

    meta, value, metrics, cash, excess, ssr, attr, _ = _line()
    reader = build_reader_metric_row(meta, metrics, cash, ssr, source="t", attribution=attr)
    legacy = build_legacy_metric_row(meta, metrics, source="t")
    monthly = build_monthly_return_rows(meta, metrics, source="t")

    table = report_table([reader, legacy, *monthly])

    assert len(table) == 2 + len(monthly)
    assert set(table["schema"]) == {READER_SCHEMA, LEGACY_SCHEMA, MONTHLY_SCHEMA}
    # stable column meaning: one column per field name, values preserved exactly
    reader_row = table[table["schema"] == READER_SCHEMA].iloc[0]
    assert reader_row["cagr"] == reader["cagr"]
    assert reader_row["sharpe"] == reader["sharpe"]
    legacy_row = table[table["schema"] == LEGACY_SCHEMA].iloc[0]
    assert legacy_row["sharpe_cal"] == legacy["sharpe_cal"]
    # reader-only columns are absent (NaN) on legacy rows, not silently shared
    assert pd.isna(legacy_row["sharpe"])


def test_report_table_rejects_mixed_windows_and_stale_monthlies():
    from macro_framework.reporting import (
        build_monthly_return_rows,
        report_table,
    )

    meta, value, metrics, cash, excess, ssr, attr, _ = _line()
    reader = build_reader_metric_row(meta, metrics, cash, ssr, source="t", attribution=attr)
    monthly = build_monthly_return_rows(meta, metrics, source="t")

    # same portfolio/schema/window label with a different window -> mixed windows
    shifted = dict(reader)
    shifted["end"] = reader["start"]
    shifted["raw_market_model_end"] = reader["start"]
    with pytest.raises(ValueError, match="mixed windows for portfolio"):
        report_table([reader, shifted])

    # a tampered monthly value no longer recompounds to the performance row
    stale = [dict(row) for row in monthly]
    stale[3]["monthly_return"] += 0.05
    with pytest.raises(ValueError, match="stale generated values"):
        report_table([reader, *stale])

    # gate failures propagate through table assembly
    with pytest.raises(ValueError, match="not part of"):
        report_table([reader | {"sharpe_cal": 1.0}])


def test_report_table_rejects_empty_or_non_sequence_input():
    from macro_framework.reporting import report_table

    with pytest.raises(ValueError, match="at least one row"):
        report_table([])
    with pytest.raises(TypeError, match="sequence"):
        report_table("not-rows")


def test_monthly_rows_pin_annualization_and_bind_their_labels():
    from macro_framework.reporting import build_monthly_return_rows

    meta, value, metrics, cash, excess, ssr, attr, _ = _line()
    first = build_monthly_return_rows(meta, metrics, source="t")[0]

    with pytest.raises(ValueError, match="12"):
        validate_report_row(first | {"periods_per_year": 252})
    with pytest.raises(ValueError, match="labeled"):
        validate_report_row(first | {"year": first["year"] + 15})
    with pytest.raises(ValueError, match=r"1\.\.12"):
        validate_report_row(first | {"month": 13})
    with pytest.raises(ValueError, match="exceeds the days"):
        validate_report_row(first | {"n_obs": 999})
    with pytest.raises(ValueError, match="monthly rows require monthly_return"):
        validate_report_row({k: v for k, v in first.items() if k != "monthly_return"})
    with pytest.raises(ValueError, match="finite"):
        validate_report_row(first | {"monthly_return": float("nan")})


def test_stale_check_requires_the_reader_total_return():
    from macro_framework.reporting import build_monthly_return_rows, report_table

    meta, value, metrics, cash, excess, ssr, attr, _ = _line()
    reader = build_reader_metric_row(meta, metrics, cash, ssr, source="t", attribution=attr)
    monthly = build_monthly_return_rows(meta, metrics, source="t")

    stripped = {k: v for k, v in reader.items() if k != "total_return"}
    with pytest.raises(ValueError, match="requires total_return"):
        report_table([stripped, *monthly])


def test_report_table_rejects_duplicate_months():
    from macro_framework.reporting import build_monthly_return_rows, report_table

    meta, value, metrics, cash, excess, ssr, attr, _ = _line()
    monthly = build_monthly_return_rows(meta, metrics, source="t")
    with pytest.raises(ValueError, match="duplicate month"):
        report_table([*monthly, monthly[0]])


# --- Task 4.6: completed shared finance contracts export through the package ------


def test_completed_finance_contracts_export_through_package():
    import macro_framework as mf

    for name in (
        "ssr_inference",
        "SSRInference",
        "SSRResult",
        "crisis_metrics",
        "CrisisMetrics",
        "LineMetadata",
        "REQUIRED_PROVENANCE",
        "READER_SCHEMA",
        "LEGACY_SCHEMA",
        "DIFFERENTIAL_SCHEMA",
        "ATTRIBUTION_SCHEMA",
        "CRISIS_SCHEMA",
        "MONTHLY_SCHEMA",
        "validate_report_row",
        "build_reader_metric_row",
        "build_legacy_metric_row",
        "build_differential_metric_row",
        "build_attribution_record",
        "build_crisis_record",
        "build_monthly_return_rows",
        "report_table",
    ):
        assert name in mf.__all__, name
        assert hasattr(mf, name), name
    # the temporary ambiguous attribution export survives until repository-wide
    # caller migration (task 11.11)
    assert mf.market_attribution is mf.raw_market_model_attribution
