"""Offline Friday USD valuation tests for remediation task 8.1."""

from __future__ import annotations

import dataclasses
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.optimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_basket_long as bbl  # noqa: E402
from macro_framework.markowitz import (  # noqa: E402
    FRONTIER_RESIDUAL_TOLERANCE,
    MOMENT_PSD_TOLERANCE,
    WEEKLY_PERIODS_PER_YEAR,
    QuoteSpec,
    WeeklyValuations,
    _quote_fx_provenance_sha256,
    annualized_moments,
    efficient_frontier,
    validate_frontier_point,
    weekly_usd_valuations,
)


QUOTE_SPECS = {
    "SWDA.L": QuoteSpec("GBP", "GBp", 0.01),
    "XLK": QuoteSpec("USD", "USD", 1.0),
    "IAU": QuoteSpec("USD", "USD", 1.0),
    "BIL": QuoteSpec("USD", "USD", 1.0),
}


def _contract() -> bbl.AcquisitionContract:
    digest = hashlib.sha256(b"offline-fixture").hexdigest()
    return dataclasses.replace(
        bbl.make_snapshot_contract(vintage_date="2026-07-03"),
        etf_raw_response_sha256=digest,
        fx_raw_response_sha256=digest,
    )


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.DatetimeIndex(
        [
            "2024-05-27",
            "2024-05-28",
            "2024-05-29",
            "2024-05-30",
            "2024-05-31",
            "2024-06-04",
            "2024-06-06",
            "2024-06-07",
        ],
        name="Date",
    )
    basket = pd.DataFrame(
        {
            "SWDA.L": [np.nan, np.nan, np.nan, 10_000.0, np.nan, np.nan, np.nan, 10_100.0],
            "XLK": [np.nan, np.nan, np.nan, np.nan, 200.0, np.nan, np.nan, 202.0],
            # Tuesday is exactly three calendar days before Friday and remains eligible.
            "IAU": [np.nan, 45.0, np.nan, np.nan, np.nan, 46.0, np.nan, np.nan],
        },
        index=dates,
    )
    cash_market = pd.DataFrame(
        {
            "BIL": [np.nan, np.nan, np.nan, np.nan, 91.0, np.nan, np.nan, 91.1],
            "SPY": [np.nan, np.nan, np.nan, np.nan, 525.0, np.nan, np.nan, 530.0],
        },
        index=dates,
    )
    fx = pd.DataFrame(
        {"USD_per_GBP": [np.nan, np.nan, 1.25, np.nan, np.nan, np.nan, 1.30, np.nan]},
        index=dates,
    )
    return basket, cash_market, fx


def _snapshot(
    tmp_path: Path,
    *,
    basket: pd.DataFrame | None = None,
    fx: pd.DataFrame | None = None,
) -> Path:
    default_basket, cash_market, default_fx = _frames()
    contract = _contract()
    data = bbl.NormalizedSnapshotData(
        basket_local=default_basket if basket is None else basket,
        cash_market=cash_market,
        fx=default_fx if fx is None else fx,
        coverage={},
    )
    return bbl.build_market_snapshot(
        snapshot_id=contract.snapshot_id,
        requested_start=contract.requested_start,
        requested_end=contract.requested_end,
        output_root=tmp_path,
        contract=contract,
        data=data,
        build_time="2026-07-28T12:00:00+00:00",
    )


def _build(
    tmp_path: Path,
    *,
    basket: pd.DataFrame | None = None,
    fx: pd.DataFrame | None = None,
):
    return weekly_usd_valuations(
        _snapshot(tmp_path, basket=basket, fx=fx),
        quote_specs=QUOTE_SPECS,
        requested_start="2024-05-31",
        requested_end="2024-06-07",
    )


def test_ac_5_1(tmp_path):
    result = _build(tmp_path)

    assert result.base_currency == "USD"
    assert result.snapshot_id == _contract().snapshot_id
    assert result.levels_usd.index.equals(
        pd.DatetimeIndex(
            ["2024-05-31 22:00:00+00:00", "2024-06-07 22:00:00+00:00"],
            name="valuation_cutoff",
        )
    )
    assert result.levels_usd.loc[result.levels_usd.index[0], "SWDA.L"] == pytest.approx(
        10_000.0 / 100.0 * 1.25
    )
    assert result.levels_usd.loc[result.levels_usd.index[1], "SWDA.L"] == pytest.approx(
        10_100.0 / 100.0 * 1.30
    )
    assert result.levels_usd[["XLK", "IAU", "BIL"]].to_numpy().tolist() == [
        [200.0, 45.0, 91.0],
        [202.0, 46.0, 91.1],
    ]


def test_ac_5_2(tmp_path):
    result = _build(tmp_path)
    first, second = result.levels_usd.index

    assert result.observed_dates.loc[first].to_dict() == {
        "SWDA.L": pd.Timestamp("2024-05-30"),
        "XLK": pd.Timestamp("2024-05-31"),
        "IAU": pd.Timestamp("2024-05-28"),
        "BIL": pd.Timestamp("2024-05-31"),
    }
    assert result.observed_dates.loc[second].to_dict() == {
        "SWDA.L": pd.Timestamp("2024-06-07"),
        "XLK": pd.Timestamp("2024-06-07"),
        "IAU": pd.Timestamp("2024-06-04"),
        "BIL": pd.Timestamp("2024-06-07"),
    }
    assert result.fx_observed_dates.to_dict() == {
        first: pd.Timestamp("2024-05-29"),
        second: pd.Timestamp("2024-06-06"),
    }
    cutoff_dates = result.levels_usd.index.tz_convert(None).normalize()
    for cutoff_date, (_, source_dates) in zip(cutoff_dates, result.observed_dates.iterrows()):
        assert bool((source_dates <= cutoff_date).all())
    assert result.start == first and result.end == second
    assert "Friday 22:00 UTC" in result.valuation_rule


def test_ac_5_6(tmp_path):
    basket, _, _ = _frames()
    accepted = _build(tmp_path / "three_days", basket=basket)
    assert accepted.observed_dates.iloc[0]["IAU"] == pd.Timestamp("2024-05-28")

    basket.loc[pd.Timestamp("2024-05-28"), "IAU"] = np.nan
    basket.loc[pd.Timestamp("2024-05-27"), "IAU"] = 45.0
    with pytest.raises(ValueError, match=r"IAU.*2024-05-31.*four|IAU.*2024-05-31.*4"):
        _build(tmp_path / "four_days", basket=basket)

    _, _, fx = _frames()
    fx.loc[pd.Timestamp("2024-05-29"), "USD_per_GBP"] = np.nan
    fx.loc[pd.Timestamp("2024-05-27"), "USD_per_GBP"] = 1.25
    with pytest.raises(ValueError, match=r"USD_per_GBP.*2024-05-31.*4"):
        _build(tmp_path / "stale_fx", fx=fx)


def test_requires_completed_hash_valid_snapshot_and_matching_quote_specs(tmp_path):
    incomplete = _snapshot(tmp_path / "incomplete")
    (incomplete / "COMPLETED").unlink()
    with pytest.raises(ValueError, match="COMPLETED"):
        weekly_usd_valuations(
            incomplete,
            quote_specs=QUOTE_SPECS,
            requested_start="2024-05-31",
            requested_end="2024-06-07",
        )

    tampered = _snapshot(tmp_path / "tampered")
    path = tampered / "basket_adjusted_close_local.parquet"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="sha256|mutated"):
        weekly_usd_valuations(
            tampered,
            quote_specs=QUOTE_SPECS,
            requested_start="2024-05-31",
            requested_end="2024-06-07",
        )

    manifest_tampered = _snapshot(tmp_path / "manifest_tampered")
    manifest_path = manifest_tampered / "manifest.json"
    manifest_path.write_text(manifest_path.read_text().replace('"GBp"', '"USD"', 1))
    with pytest.raises(ValueError, match="manifest sha256"):
        weekly_usd_valuations(
            manifest_tampered,
            quote_specs=QUOTE_SPECS,
            requested_start="2024-05-31",
            requested_end="2024-06-07",
        )

    with pytest.raises(ValueError, match="SWDA.L.*quote"):
        weekly_usd_valuations(
            _snapshot(tmp_path / "quotes"),
            quote_specs={**QUOTE_SPECS, "SWDA.L": QuoteSpec("USD", "USD", 1.0)},
            requested_start="2024-05-31",
            requested_end="2024-06-07",
        )


def test_requires_complete_common_usd_matrix(tmp_path):
    basket, _, _ = _frames()
    basket.loc[basket.index < pd.Timestamp("2024-06-07"), "SWDA.L"] = np.nan
    with pytest.raises(ValueError, match=r"SWDA.L.*(?:eligible|observation).*2024-05-31"):
        _build(tmp_path, basket=basket)


def _moment_provenance(
    asset_quote_specs: tuple[tuple[str, QuoteSpec], ...],
    fx_required_assets: tuple[str, ...],
) -> str:
    return _quote_fx_provenance_sha256(
        snapshot_id="snapshot-fixture-v1",
        base_currency="USD",
        valuation_rule=(
            "Friday 22:00 UTC; latest observation at or before cutoff; "
            "maximum staleness 3 calendar days"
        ),
        asset_quote_specs=asset_quote_specs,
        fx_required_assets=fx_required_assets,
    )


def _moment_valuations() -> WeeklyValuations:
    cutoffs = pd.DatetimeIndex(
        [
            "2024-01-05 22:00:00+00:00",
            "2024-01-12 22:00:00+00:00",
            "2024-01-19 22:00:00+00:00",
            "2024-01-26 22:00:00+00:00",
        ],
        name="valuation_cutoff",
    )
    levels = pd.DataFrame(
        {
            "ASSET_A": [100.0, 110.0, 99.0, 118.8],
            "ASSET_B": [50.0, 50.0, 60.0, 54.0],
        },
        index=cutoffs,
    )
    observed = pd.DataFrame(
        {
            "ASSET_A": cutoffs.tz_convert(None).normalize(),
            "ASSET_B": cutoffs.tz_convert(None).normalize(),
        },
        index=cutoffs,
    )
    asset_quote_specs = (
        ("ASSET_A", QuoteSpec("USD", "USD", 1.0)),
        ("ASSET_B", QuoteSpec("USD", "USD", 1.0)),
    )
    return WeeklyValuations(
        levels_usd=levels,
        observed_dates=observed,
        fx_observed_dates=pd.Series(pd.NaT, index=cutoffs, dtype="datetime64[ns]"),
        asset_quote_specs=asset_quote_specs,
        producer_provenance_sha256=_moment_provenance(asset_quote_specs, ()),
        base_currency="USD",
        valuation_rule=(
            "Friday 22:00 UTC; latest observation at or before cutoff; "
            "maximum staleness 3 calendar days"
        ),
        start=cutoffs[0],
        end=cutoffs[-1],
        snapshot_id="snapshot-fixture-v1",
    )


def _moment_valuations_with_required_fx() -> WeeklyValuations:
    valuations = _moment_valuations()
    source_dates = valuations.levels_usd.index.tz_convert(None).normalize()
    asset_quote_specs = (
        ("ASSET_A", QuoteSpec("GBP", "GBp", 0.01)),
        ("ASSET_B", QuoteSpec("USD", "USD", 1.0)),
    )
    fx_required_assets = ("ASSET_A",)
    return dataclasses.replace(
        valuations,
        fx_observed_dates=pd.Series(
            source_dates,
            index=valuations.levels_usd.index,
            name="USD_per_GBP_observed_date",
        ),
        asset_quote_specs=asset_quote_specs,
        producer_provenance_sha256=_moment_provenance(
            asset_quote_specs, fx_required_assets
        ),
        fx_required_assets=fx_required_assets,
    )


def test_weekly_valuations_identify_assets_requiring_fx(tmp_path):
    result = _build(tmp_path)

    assert result.asset_quote_specs == tuple(QUOTE_SPECS.items())
    assert result.fx_required_assets == ("SWDA.L",)
    assert len(result.producer_provenance_sha256) == 64
    assert result.producer_provenance_sha256 == _quote_fx_provenance_sha256(
        snapshot_id=result.snapshot_id,
        base_currency=result.base_currency,
        valuation_rule=result.valuation_rule,
        asset_quote_specs=result.asset_quote_specs,
        fx_required_assets=result.fx_required_assets,
    )


def test_moments_accept_complete_date_granular_provenance():
    result = annualized_moments(_moment_valuations_with_required_fx())

    assert result.n_obs == 3


def test_moments_reject_in_universe_fx_required_asset_substitution():
    valuations = _moment_valuations_with_required_fx()

    with pytest.raises(ValueError, match="fx_required_assets.*authoritative quote metadata"):
        annualized_moments(
            dataclasses.replace(valuations, fx_required_assets=("ASSET_B",))
        )


def test_moments_reject_complete_fx_requirement_erasure():
    valuations = _moment_valuations_with_required_fx()
    erased_fx_dates = pd.Series(
        pd.NaT,
        index=valuations.levels_usd.index,
        dtype="datetime64[ns]",
    )

    with pytest.raises(ValueError, match="fx_required_assets.*authoritative quote metadata"):
        annualized_moments(
            dataclasses.replace(
                valuations,
                fx_required_assets=(),
                fx_observed_dates=erased_fx_dates,
            )
        )


def test_moments_reject_coherent_fx_lineage_erasure():
    valuations = _moment_valuations_with_required_fx()
    erased_fx_dates = pd.Series(
        pd.NaT,
        index=valuations.levels_usd.index,
        dtype="datetime64[ns]",
    )

    with pytest.raises(ValueError, match="producer-authorized.*provenance"):
        annualized_moments(
            dataclasses.replace(
                valuations,
                asset_quote_specs=(
                    ("ASSET_A", QuoteSpec("USD", "USD", 1.0)),
                    ("ASSET_B", QuoteSpec("USD", "USD", 1.0)),
                ),
                fx_required_assets=(),
                fx_observed_dates=erased_fx_dates,
            )
        )


def test_moments_reject_coherent_fx_lineage_substitution():
    valuations = _moment_valuations_with_required_fx()

    with pytest.raises(ValueError, match="producer-authorized.*provenance"):
        annualized_moments(
            dataclasses.replace(
                valuations,
                asset_quote_specs=(
                    ("ASSET_A", QuoteSpec("USD", "USD", 1.0)),
                    ("ASSET_B", QuoteSpec("GBP", "GBp", 0.01)),
                ),
                fx_required_assets=("ASSET_B",),
            )
        )


@pytest.mark.parametrize(
    ("field", "invalid_value", "requires_fx"),
    [
        ("quote_currency", "EUR", True),
        ("quote_unit", "GBp", False),
        ("scale_to_major", 0.01, False),
    ],
    ids=["unsupported-currency", "unit-currency-mismatch", "scale-unit-mismatch"],
)
def test_moments_revalidate_quote_spec_field_invariants(
    field, invalid_value, requires_fx
):
    valuations = (
        _moment_valuations_with_required_fx() if requires_fx else _moment_valuations()
    )
    first_asset, quote_spec = valuations.asset_quote_specs[0]
    object.__setattr__(quote_spec, field, invalid_value)

    with pytest.raises(ValueError, match="QuoteSpec.*invalid|quote.*requires"):
        annualized_moments(
            dataclasses.replace(
                valuations,
                asset_quote_specs=(
                    (first_asset, quote_spec),
                    valuations.asset_quote_specs[1],
                ),
            )
        )


@pytest.mark.parametrize(
    "asset_quote_specs",
    [
        [("ASSET_A", QuoteSpec("USD", "USD", 1.0))],
        (("ASSET_A", QuoteSpec("USD", "USD", 1.0)),),
        (
            ("ASSET_B", QuoteSpec("USD", "USD", 1.0)),
            ("ASSET_A", QuoteSpec("USD", "USD", 1.0)),
        ),
        (
            ("ASSET_A", object()),
            ("ASSET_B", QuoteSpec("USD", "USD", 1.0)),
        ),
    ],
    ids=["not-tuple", "missing-asset", "wrong-order", "malformed-spec"],
)
def test_moments_reject_malformed_authoritative_quote_metadata(asset_quote_specs):
    with pytest.raises(ValueError, match="asset_quote_specs"):
        annualized_moments(
            dataclasses.replace(
                _moment_valuations(),
                asset_quote_specs=asset_quote_specs,
            )
        )


def test_moments_accept_exactly_three_day_asset_and_fx_provenance():
    valuations = _moment_valuations_with_required_fx()
    first = valuations.levels_usd.index[0]
    three_days_before = first.tz_convert(None).normalize() - pd.Timedelta(days=3)
    observed = valuations.observed_dates.copy()
    observed.loc[first, "ASSET_A"] = three_days_before
    fx_observed = valuations.fx_observed_dates.copy()
    fx_observed.loc[first] = three_days_before

    result = annualized_moments(
        dataclasses.replace(
            valuations,
            observed_dates=observed,
            fx_observed_dates=fx_observed,
        )
    )

    assert result.n_obs == 3


def test_moments_reject_provenance_inconsistent_with_declared_rule():
    valuations = _moment_valuations()

    with pytest.raises(ValueError, match="valuation_rule.*maximum staleness 3"):
        annualized_moments(
            dataclasses.replace(
                valuations,
                valuation_rule=(
                    "Friday 22:00 UTC; latest observation at or before cutoff; "
                    "maximum staleness 4 calendar days"
                ),
            )
        )


@pytest.mark.parametrize("field", ["observed_dates", "fx_observed_dates"])
def test_moments_require_exact_provenance_cutoff_index_metadata(field):
    valuations = _moment_valuations_with_required_fx()
    invalid = getattr(valuations, field).copy()
    invalid.index = invalid.index.rename("wrong_cutoff_name")

    with pytest.raises(ValueError, match=field):
        annualized_moments(dataclasses.replace(valuations, **{field: invalid}))


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "missing-asset",
        "extra-asset",
        "reordered-assets",
        "mismatched-index",
        "missing-date",
        "look-ahead",
        "stale",
        "intraday",
        "timezone-aware",
        "malformed",
    ],
)
def test_moments_reject_invalid_asset_observation_provenance(case):
    valuations = _moment_valuations()
    invalid = valuations.observed_dates.copy()
    first = valuations.levels_usd.index[0]
    first_date = first.tz_convert(None).normalize()

    if case == "empty":
        invalid = invalid.iloc[:0]
    elif case == "missing-asset":
        invalid = invalid.drop(columns="ASSET_B")
    elif case == "extra-asset":
        invalid["EXTRA"] = invalid["ASSET_A"]
    elif case == "reordered-assets":
        invalid = invalid[["ASSET_B", "ASSET_A"]]
    elif case == "mismatched-index":
        invalid.index = invalid.index + pd.Timedelta(days=7)
    elif case == "missing-date":
        invalid.loc[first, "ASSET_A"] = pd.NaT
    elif case == "look-ahead":
        invalid.loc[first, "ASSET_A"] = first_date + pd.Timedelta(days=1)
    elif case == "stale":
        invalid.loc[first, "ASSET_A"] = first_date - pd.Timedelta(days=4)
    elif case == "intraday":
        invalid.loc[first, "ASSET_A"] = first_date + pd.Timedelta(hours=12)
    elif case == "timezone-aware":
        invalid["ASSET_A"] = pd.DatetimeIndex(invalid["ASSET_A"]).tz_localize("UTC")
    elif case == "malformed":
        invalid["ASSET_A"] = invalid["ASSET_A"].dt.strftime("%Y-%m-%d")
    else:  # pragma: no cover - the parameter list is exhaustive
        raise AssertionError(case)

    with pytest.raises(ValueError, match="observed_dates"):
        annualized_moments(dataclasses.replace(valuations, observed_dates=invalid))


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "mismatched-index",
        "missing-date",
        "all-missing",
        "look-ahead",
        "stale",
        "intraday",
        "timezone-aware",
        "malformed",
    ],
)
def test_moments_reject_invalid_required_fx_provenance(case):
    valuations = _moment_valuations_with_required_fx()
    invalid = valuations.fx_observed_dates.copy()
    first = valuations.levels_usd.index[0]
    first_date = first.tz_convert(None).normalize()

    if case == "empty":
        invalid = invalid.iloc[:0]
    elif case == "mismatched-index":
        invalid.index = invalid.index + pd.Timedelta(days=7)
    elif case == "missing-date":
        invalid.loc[first] = pd.NaT
    elif case == "all-missing":
        invalid.loc[:] = pd.NaT
    elif case == "look-ahead":
        invalid.loc[first] = first_date + pd.Timedelta(days=1)
    elif case == "stale":
        invalid.loc[first] = first_date - pd.Timedelta(days=4)
    elif case == "intraday":
        invalid.loc[first] = first_date + pd.Timedelta(hours=12)
    elif case == "timezone-aware":
        invalid = pd.Series(
            pd.DatetimeIndex(invalid).tz_localize("UTC"),
            index=invalid.index,
            name=invalid.name,
        )
    elif case == "malformed":
        invalid = invalid.dt.strftime("%Y-%m-%d")
    else:  # pragma: no cover - the parameter list is exhaustive
        raise AssertionError(case)

    with pytest.raises(ValueError, match="fx_observed_dates"):
        annualized_moments(dataclasses.replace(valuations, fx_observed_dates=invalid))


@pytest.mark.parametrize(
    "required_assets",
    [
        ["ASSET_A"],
        ("",),
        ("UNKNOWN",),
        ("ASSET_A", "ASSET_A"),
        ("ASSET_B", "ASSET_A"),
    ],
    ids=["not-tuple", "blank", "wrong-universe", "duplicate", "wrong-order"],
)
def test_moments_reject_malformed_fx_required_asset_provenance(required_assets):
    valuations = _moment_valuations_with_required_fx()

    with pytest.raises(ValueError, match="fx_required_assets"):
        annualized_moments(
            dataclasses.replace(valuations, fx_required_assets=required_assets)
        )


def test_moments_reject_spurious_fx_dates_when_no_asset_requires_conversion():
    valuations = _moment_valuations()
    source_dates = valuations.levels_usd.index.tz_convert(None).normalize()
    unexpected_fx = pd.Series(source_dates, index=valuations.levels_usd.index)

    with pytest.raises(ValueError, match="fx_observed_dates.*no asset requires"):
        annualized_moments(
            dataclasses.replace(valuations, fx_observed_dates=unexpected_fx)
        )


def test_moments_reject_provenance_before_calculating_returns(monkeypatch):
    valuations = _moment_valuations()
    invalid = valuations.observed_dates.copy()
    invalid.loc[invalid.index[0], "ASSET_A"] = pd.NaT

    def unexpected_calculation(*args, **kwargs):
        raise AssertionError("return calculation reached before provenance validation")

    monkeypatch.setattr(pd.DataFrame, "pct_change", unexpected_calculation)

    with pytest.raises(ValueError, match="observed_dates"):
        annualized_moments(dataclasses.replace(valuations, observed_dates=invalid))


def test_ac_5_3():
    """Weekly returns and moments are exact and annualized by 365.2425/7 (also 5.2)."""
    valuations = _moment_valuations()

    result = annualized_moments(valuations)

    expected_returns = pd.DataFrame(
        {
            "ASSET_A": [0.1, -0.1, 0.2],
            "ASSET_B": [0.0, 0.2, -0.1],
        },
        index=valuations.levels_usd.index[1:],
    )
    pd.testing.assert_frame_equal(result.weekly_returns, expected_returns)
    pd.testing.assert_series_equal(
        result.mean_ann_arithmetic,
        pd.Series(
            [1.0 / 15.0, 1.0 / 30.0],
            index=["ASSET_A", "ASSET_B"],
            name="mean_ann_arithmetic",
        )
        * WEEKLY_PERIODS_PER_YEAR,
    )
    pd.testing.assert_frame_equal(
        result.covariance_ann,
        pd.DataFrame(
            [[7.0 / 300.0, -7.0 / 300.0], [-7.0 / 300.0, 7.0 / 300.0]],
            index=["ASSET_A", "ASSET_B"],
            columns=["ASSET_A", "ASSET_B"],
        )
        * WEEKLY_PERIODS_PER_YEAR,
    )
    assert result.periods_per_year == 365.2425 / 7 == 52.1775


def test_ac_5_4():
    """Moment provenance discloses the actual return window and basis (also 7.4)."""
    valuations = _moment_valuations()

    result = annualized_moments(valuations)

    assert result.return_dates.equals(valuations.levels_usd.index[1:])
    assert result.n_obs == 3
    assert result.start == valuations.levels_usd.index[1]
    assert result.end == valuations.levels_usd.index[-1]
    assert result.snapshot_id == valuations.snapshot_id
    assert result.base_currency == "USD"
    assert result.valuation_rule == valuations.valuation_rule
    assert result.periods_per_year == WEEKLY_PERIODS_PER_YEAR


def test_moments_require_two_returns_for_finite_sample_covariance():
    valuations = _moment_valuations()
    shortened_levels = valuations.levels_usd.iloc[:2]
    shortened = dataclasses.replace(
        valuations,
        levels_usd=shortened_levels,
        observed_dates=valuations.observed_dates.iloc[:2],
        fx_observed_dates=valuations.fx_observed_dates.iloc[:2],
        end=shortened_levels.index[-1],
    )

    with pytest.raises(ValueError, match="at least three.*valuations|two.*returns"):
        annualized_moments(shortened)


def test_moments_reject_incomplete_levels_without_filling():
    valuations = _moment_valuations()
    incomplete = valuations.levels_usd.copy()
    incomplete.loc[incomplete.index[1], "ASSET_A"] = np.nan

    with pytest.raises(ValueError, match="complete.*finite|missing|non-finite"):
        annualized_moments(dataclasses.replace(valuations, levels_usd=incomplete))


@pytest.mark.parametrize(
    "invalid_value",
    [True, 100.0 + 1.0j],
)
def test_moments_reject_non_real_numeric_level_matrices(invalid_value):
    valuations = _moment_valuations()
    invalid = valuations.levels_usd.astype(object)
    invalid.loc[invalid.index[1], "ASSET_A"] = invalid_value

    with pytest.raises(ValueError, match="real numeric"):
        annualized_moments(dataclasses.replace(valuations, levels_usd=invalid))


def test_moments_reject_native_complex128_levels_before_lossy_float_cast():
    valuations = _moment_valuations()
    invalid = valuations.levels_usd.astype(np.complex128)
    invalid.loc[invalid.index[1], "ASSET_A"] += 1.0j

    with pytest.raises(ValueError, match="real numeric"):
        annualized_moments(dataclasses.replace(valuations, levels_usd=invalid))


def test_moments_reject_nonconsecutive_friday_grid():
    valuations = _moment_valuations()
    levels = valuations.levels_usd.drop(valuations.levels_usd.index[1])

    with pytest.raises(ValueError, match="consecutive.*Friday|seven days|7 days"):
        annualized_moments(
            dataclasses.replace(
                valuations,
                levels_usd=levels,
                observed_dates=valuations.observed_dates.loc[levels.index],
                fx_observed_dates=valuations.fx_observed_dates.loc[levels.index],
            )
        )


def test_moments_reject_nonweekly_annualization():
    with pytest.raises(ValueError, match="365.2425 / 7|52.1775|weekly annualization"):
        annualized_moments(_moment_valuations(), periods_per_year=252.0)


def test_ac_8_5_non_finite_annualized_mean_is_rejected(monkeypatch):
    def invalid_mean(self, *args, **kwargs):
        return pd.Series([np.nan, 0.0], index=self.columns)

    monkeypatch.setattr(pd.DataFrame, "mean", invalid_mean)

    with pytest.raises(ValueError, match="arithmetic means.*finite"):
        annualized_moments(_moment_valuations())


def test_ac_8_5_complex_annualized_mean_is_rejected_without_lossy_cast(monkeypatch):
    def complex_mean(self, *args, **kwargs):
        return pd.Series([1.0 + 0.5j, 2.0 + 0.0j], index=self.columns)

    monkeypatch.setattr(pd.DataFrame, "mean", complex_mean)

    with pytest.raises(ValueError, match="arithmetic means.*real"):
        annualized_moments(_moment_valuations())


@pytest.mark.parametrize(
    "mean_index",
    [
        pd.Index(["ASSET_A"]),
        pd.Index(["ASSET_A", "ASSET_B", "EXTRA"]),
    ],
    ids=["missing-asset", "extra-asset"],
)
def test_ac_8_5_mean_labels_must_exactly_match_asset_universe(monkeypatch, mean_index):
    def invalid_mean(self, *args, **kwargs):
        return pd.Series(np.zeros(len(mean_index)), index=mean_index)

    monkeypatch.setattr(pd.DataFrame, "mean", invalid_mean)

    with pytest.raises(ValueError, match="arithmetic mean.*asset labels"):
        annualized_moments(_moment_valuations())


@pytest.mark.parametrize(
    ("invalid_covariance", "row_labels", "column_labels"),
    [
        ([[1.0]], ["ASSET_A"], ["ASSET_A"]),
        (
            np.eye(3).tolist(),
            ["ASSET_A", "ASSET_B", "EXTRA"],
            ["ASSET_A", "ASSET_B", "EXTRA"],
        ),
        (
            [[1.0, 0.0], [0.0, 1.0]],
            ["ASSET_B", "ASSET_A"],
            ["ASSET_A", "ASSET_B"],
        ),
        (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ["ASSET_A", "ASSET_B"],
            ["ASSET_A", "ASSET_B", "EXTRA"],
        ),
    ],
    ids=["missing-asset", "extra-asset", "inconsistent-label-order", "non-square-extra-column"],
)
def test_ac_8_5_covariance_dimensions_and_labels_must_exactly_match_asset_universe(
    monkeypatch, invalid_covariance, row_labels, column_labels
):
    def invalid_cov(self, *args, **kwargs):
        return pd.DataFrame(
            invalid_covariance,
            index=row_labels,
            columns=column_labels,
        )

    monkeypatch.setattr(pd.DataFrame, "cov", invalid_cov)

    with pytest.raises(ValueError, match="covariance.*square.*asset labels"):
        annualized_moments(_moment_valuations())


@pytest.mark.parametrize(
    ("invalid_covariance", "message"),
    [
        ([[np.nan, 0.0], [0.0, 1.0]], "finite"),
        ([[1.0, 0.5], [0.0, 1.0]], "symmetric"),
        ([[1.0, 2.0], [2.0, 1.0]], "positive semidefinite"),
    ],
)
def test_ac_8_5_invalid_covariance_is_rejected(
    monkeypatch, invalid_covariance, message
):
    def invalid_cov(self, *args, **kwargs):
        return pd.DataFrame(invalid_covariance, index=self.columns, columns=self.columns)

    monkeypatch.setattr(pd.DataFrame, "cov", invalid_cov)

    with pytest.raises(ValueError, match=message):
        annualized_moments(_moment_valuations())


def test_ac_8_5_complex_covariance_is_rejected_without_lossy_cast(monkeypatch):
    def complex_cov(self, *args, **kwargs):
        return pd.DataFrame(
            [[1.0 + 0.5j, 0.0], [0.0, 1.0 + 0.0j]],
            index=self.columns,
            columns=self.columns,
        )

    monkeypatch.setattr(pd.DataFrame, "cov", complex_cov)

    with pytest.raises(ValueError, match="covariance.*real"):
        annualized_moments(_moment_valuations())


def test_ac_8_5_psd_check_accepts_roundoff_within_documented_tolerance(monkeypatch):
    weekly_roundoff = -MOMENT_PSD_TOLERANCE / (2.0 * WEEKLY_PERIODS_PER_YEAR)

    def almost_psd(self, *args, **kwargs):
        return pd.DataFrame(
            [[weekly_roundoff, 0.0], [0.0, 1.0]],
            index=self.columns,
            columns=self.columns,
        )

    monkeypatch.setattr(pd.DataFrame, "cov", almost_psd)

    result = annualized_moments(_moment_valuations())

    assert np.linalg.eigvalsh(result.covariance_ann.to_numpy()).min() == pytest.approx(
        -MOMENT_PSD_TOLERANCE / 2.0
    )


def test_ac_8_5_covariance_is_symmetric_and_psd_with_documented_tolerance():
    result = annualized_moments(_moment_valuations())

    assert result.psd_tolerance == MOMENT_PSD_TOLERANCE
    np.testing.assert_allclose(
        result.covariance_ann.to_numpy(),
        result.covariance_ann.to_numpy().T,
        rtol=0.0,
        atol=MOMENT_PSD_TOLERANCE,
    )
    assert np.linalg.eigvalsh(result.covariance_ann.to_numpy()).min() >= (
        -MOMENT_PSD_TOLERANCE
    )


# --- Task 8.4: offline numerical and validation tests ---


def test_gbp_assets_convert_to_usd_before_returns(tmp_path):
    """Defect 4: GBp levels scale by 1/100 and multiply by USD_per_GBP."""
    result = _build(tmp_path)
    first, second = result.levels_usd.index

    assert result.levels_usd.loc[first, "SWDA.L"] == pytest.approx(
        10_000.0 / 100.0 * 1.25
    )
    # Counterexamples: dividing by USD_per_GBP (inverted FX direction) or
    # skipping the GBp/100 scale produce materially different USD levels.
    assert abs(result.levels_usd.loc[first, "SWDA.L"] - 10_000.0 / 100.0 / 1.25) > 1.0
    assert abs(result.levels_usd.loc[first, "SWDA.L"] - 10_000.0 * 1.25) > 1.0
    # The FX move (1.25 -> 1.30) makes the direction observable in the weekly
    # USD return: multiplication appreciates it, inversion flips the sign.
    usd_return = (
        result.levels_usd.loc[second, "SWDA.L"]
        / result.levels_usd.loc[first, "SWDA.L"]
        - 1.0
    )
    correct = (10_100.0 * 1.30) / (10_000.0 * 1.25) - 1.0
    inverted = (10_100.0 / 1.30) / (10_000.0 / 1.25) - 1.0
    assert usd_return == pytest.approx(correct)
    assert usd_return > 0.0 > inverted
    assert abs(usd_return - inverted) > 0.05


def test_weekly_grid_uses_matching_annualization():
    """Defect 11: the weekly common grid annualizes by 365.2425/7, never 252."""
    moments = annualized_moments(_moment_valuations())

    assert moments.periods_per_year == WEEKLY_PERIODS_PER_YEAR == 52.1775
    weekly_mean = moments.weekly_returns.mean()
    np.testing.assert_allclose(
        moments.mean_ann_arithmetic.to_numpy(),
        (weekly_mean * WEEKLY_PERIODS_PER_YEAR).to_numpy(),
    )
    assert not np.allclose(
        moments.mean_ann_arithmetic.to_numpy(), (weekly_mean * 252.0).to_numpy()
    )
    np.testing.assert_allclose(
        moments.covariance_ann.to_numpy(),
        (moments.weekly_returns.cov() * WEEKLY_PERIODS_PER_YEAR).to_numpy(),
    )
    with pytest.raises(ValueError, match="365.2425 / 7"):
        annualized_moments(_moment_valuations(), periods_per_year=252.0)

    frontier = efficient_frontier(moments, n_points=3)
    assert frontier.periods_per_year == WEEKLY_PERIODS_PER_YEAR
    with pytest.raises(ValueError, match="365.2425 / 7"):
        efficient_frontier(dataclasses.replace(moments, periods_per_year=252.0))


def test_valuations_use_friday_source_dates_without_look_ahead(tmp_path):
    """A post-cutoff Saturday print is never selected for a Friday valuation."""
    basket, _, _ = _frames()
    saturday = pd.Timestamp("2024-06-08")
    basket.loc[saturday, :] = np.nan
    basket.loc[saturday, "XLK"] = 999.0

    result = _build(tmp_path, basket=basket)

    second = result.levels_usd.index[1]
    assert result.levels_usd.loc[second, "XLK"] == 202.0
    assert result.observed_dates.loc[second, "XLK"] == pd.Timestamp("2024-06-07")


def _mixed_calendar_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three common Fridays with mixed currencies and cross-exchange holidays."""
    dates = pd.DatetimeIndex(
        [
            "2024-05-28",
            "2024-05-29",
            "2024-05-30",
            "2024-05-31",
            "2024-06-04",
            "2024-06-06",
            "2024-06-07",
            "2024-06-13",
            "2024-06-14",
        ],
        name="Date",
    )
    nan = np.nan
    basket = pd.DataFrame(
        {
            # LSE last prints Thursday 2024-05-30 and is on holiday on Friday
            # 2024-06-14, so the third common Friday uses the Thursday print.
            "SWDA.L": [nan, nan, 10_000.0, nan, nan, nan, 10_100.0, 10_201.0, nan],
            "XLK": [nan, nan, nan, 200.0, nan, nan, 202.0, nan, 204.02],
            "IAU": [45.0, nan, nan, nan, 46.0, nan, nan, 46.92, nan],
        },
        index=dates,
    )
    cash_market = pd.DataFrame(
        {
            "BIL": [nan, nan, nan, 91.0, nan, nan, 91.1, nan, 91.2],
            "SPY": [nan, nan, nan, 525.0, nan, nan, 530.0, nan, 535.0],
        },
        index=dates,
    )
    fx = pd.DataFrame(
        {"USD_per_GBP": [nan, 1.25, nan, nan, nan, 1.30, nan, nan, 1.28]},
        index=dates,
    )
    return basket, cash_market, fx


def test_ac_8_5(tmp_path):
    """Mixed currencies, cross-exchange calendars, and matching annualization."""
    basket, cash_market, fx = _mixed_calendar_frames()
    contract = _contract()
    snapshot = bbl.build_market_snapshot(
        snapshot_id=contract.snapshot_id,
        requested_start=contract.requested_start,
        requested_end=contract.requested_end,
        output_root=tmp_path,
        contract=contract,
        data=bbl.NormalizedSnapshotData(
            basket_local=basket, cash_market=cash_market, fx=fx, coverage={}
        ),
        build_time="2026-07-28T12:00:00+00:00",
    )

    valuations = weekly_usd_valuations(
        snapshot,
        quote_specs=QUOTE_SPECS,
        requested_start="2024-05-31",
        requested_end="2024-06-14",
    )

    # Mixed currencies: the GBp asset is valued in USD before any return math.
    assert valuations.levels_usd["SWDA.L"].tolist() == pytest.approx(
        [100.0 * 1.25, 101.0 * 1.30, 102.01 * 1.28]
    )
    # Cross-exchange calendars: the Friday LSE holiday keeps the common Friday
    # grid and records the Thursday source date without look-ahead.
    third = valuations.levels_usd.index[2]
    assert valuations.observed_dates.loc[third, "SWDA.L"] == pd.Timestamp("2024-06-13")
    assert valuations.observed_dates.loc[third, "XLK"] == pd.Timestamp("2024-06-14")

    moments = annualized_moments(valuations)

    # Every return spans consecutive common Fridays: the effective observation
    # frequency is weekly, and the annualization factor matches it exactly.
    assert moments.return_dates.equals(valuations.levels_usd.index[1:])
    assert moments.n_obs == 2
    assert moments.periods_per_year == WEEKLY_PERIODS_PER_YEAR == 365.2425 / 7
    pd.testing.assert_series_equal(
        moments.mean_ann_arithmetic,
        moments.weekly_returns.mean() * WEEKLY_PERIODS_PER_YEAR,
        check_names=False,
    )
    # Finite, symmetric, positive-semidefinite annualized moments.
    covariance = moments.covariance_ann.to_numpy()
    assert np.isfinite(moments.mean_ann_arithmetic.to_numpy()).all()
    assert np.isfinite(covariance).all()
    np.testing.assert_allclose(
        covariance, covariance.T, rtol=0.0, atol=MOMENT_PSD_TOLERANCE
    )
    assert np.linalg.eigvalsh(covariance).min() >= -MOMENT_PSD_TOLERANCE


# --- Task 8.3: feasible long-only frontiers with diagnostics ---


def _frontier_moments():
    return annualized_moments(_moment_valuations())


def test_frontier_retains_full_diagnostics_and_weights_for_every_target():
    moments = _frontier_moments()

    result = efficient_frontier(moments, n_points=5)

    mu = moments.mean_ann_arithmetic
    expected_targets = np.linspace(float(mu.min()), float(mu.max()), 5)
    assert result.n_targets == 5
    assert len(result.points) == 5
    assert result.targets_ann == tuple(expected_targets)
    for point, target in zip(result.points, expected_targets):
        assert point.target_return_ann == target
        assert point.success is True
        assert point.feasible is True
        assert isinstance(point.status, int)
        assert isinstance(point.message, str) and point.message
        assert isinstance(point.iterations, int) and point.iterations >= 1
        assert list(point.weights.index) == ["ASSET_A", "ASSET_B"]
        assert abs(point.budget_residual) <= result.residual_tolerance
        assert abs(point.target_residual) <= result.residual_tolerance
        assert 0.0 <= point.bound_violation <= result.residual_tolerance
        assert point.objective >= -MOMENT_PSD_TOLERANCE
        assert point.volatility_ann == float(np.sqrt(max(point.objective, 0.0)))
    # Hand-check: perfectly negatively correlated pair hedges to zero variance
    # at the 50/50 midpoint target, and the endpoint targets are the corners.
    midpoint = result.points[2]
    np.testing.assert_allclose(midpoint.weights.to_numpy(), [0.5, 0.5], atol=1e-6)
    assert midpoint.objective == pytest.approx(0.0, abs=1e-10)
    np.testing.assert_allclose(result.points[0].weights.to_numpy(), [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(result.points[-1].weights.to_numpy(), [1.0, 0.0], atol=1e-6)
    # Determinism: a rerun reproduces every stored weight vector exactly.
    repeat = efficient_frontier(moments, n_points=5)
    assert repeat.targets_ann == result.targets_ann
    for first, second in zip(result.points, repeat.points):
        assert np.array_equal(first.weights.to_numpy(), second.weights.to_numpy())
        assert first.objective == second.objective


def test_frontier_validates_publishable_points_against_stored_weights_and_moments():
    moments = _frontier_moments()

    result = efficient_frontier(moments, n_points=4)

    mu = moments.mean_ann_arithmetic.to_numpy(dtype=float)
    sigma = moments.covariance_ann.to_numpy(dtype=float)
    publishable = result.publishable_points()
    assert result.n_feasible == 4
    assert len(publishable) == 4
    assert all(first is second for first, second in zip(publishable, result.points))
    for point in publishable:
        weights = point.weights.to_numpy(dtype=float)
        assert point.budget_residual == float(weights.sum() - 1.0)
        assert point.target_residual == float(weights @ mu - point.target_return_ann)
        assert point.bound_violation == float(
            max(0.0, float((-weights).max()), float((weights - 1.0).max()))
        )
        assert point.objective == float(weights @ sigma @ weights)
        assert point.return_ann == float(weights @ mu)
        assert point.volatility_ann == float(np.sqrt(max(point.objective, 0.0)))
        validate_frontier_point(point, moments)

    tampered = dataclasses.replace(
        publishable[0], weights=publishable[0].weights + 0.25
    )
    with pytest.raises(ValueError, match="stored weights"):
        validate_frontier_point(tampered, moments)


def test_frontier_surfaces_induced_solver_failure_without_dropping_targets(monkeypatch):
    moments = _frontier_moments()
    real_minimize = scipy.optimize.minimize
    calls = {"count": 0}

    def induced(*args, **kwargs):
        calls["count"] += 1
        solved = real_minimize(*args, **kwargs)
        if calls["count"] == 2:
            solved.success = False
            solved.status = 9
            solved.message = "induced iteration limit"
        return solved

    monkeypatch.setattr(scipy.optimize, "minimize", induced)

    result = efficient_frontier(moments, n_points=3)

    assert len(result.points) == 3
    failed = result.points[1]
    assert failed.success is False
    assert failed.feasible is False
    assert failed.status == 9
    assert failed.message == "induced iteration limit"
    assert isinstance(failed.weights, pd.Series)
    assert result.n_feasible == 2
    publishable = result.publishable_points()
    assert len(publishable) == 2
    assert all(point.feasible for point in publishable)


def test_frontier_rejects_success_claims_that_violate_residual_tolerances(monkeypatch):
    moments = _frontier_moments()
    real_minimize = scipy.optimize.minimize
    calls = {"count": 0}

    def dishonest(*args, **kwargs):
        calls["count"] += 1
        solved = real_minimize(*args, **kwargs)
        if calls["count"] == 1:
            solved.x = np.array([1.0, 1.0])
            solved.success = True
        return solved

    monkeypatch.setattr(scipy.optimize, "minimize", dishonest)

    result = efficient_frontier(moments, n_points=3)

    dishonest_point = result.points[0]
    assert dishonest_point.success is True
    assert dishonest_point.feasible is False
    assert dishonest_point.budget_residual == pytest.approx(1.0)
    assert result.n_feasible == 2
    assert len(result.publishable_points()) == 2


def test_frontier_discloses_window_snapshot_and_annualization():
    moments = _frontier_moments()

    result = efficient_frontier(moments, n_points=3)

    assert result.base_currency == "USD"
    assert result.snapshot_id == moments.snapshot_id
    assert result.valuation_rule == moments.valuation_rule
    assert result.periods_per_year == WEEKLY_PERIODS_PER_YEAR
    assert result.start == moments.start
    assert result.end == moments.end
    assert result.n_obs == moments.n_obs == 3
    assert result.residual_tolerance == FRONTIER_RESIDUAL_TOLERANCE


@pytest.mark.parametrize(
    "n_points",
    [1, 0, True, 2.5, "60"],
    ids=["too-few", "zero", "bool", "float", "string"],
)
def test_frontier_rejects_invalid_n_points(n_points):
    with pytest.raises(ValueError, match="n_points"):
        efficient_frontier(_frontier_moments(), n_points=n_points)


def test_frontier_rejects_tampered_or_incoherent_moments():
    moments = _frontier_moments()

    with pytest.raises(ValueError, match="AnnualizedMoments"):
        efficient_frontier(object())

    non_psd = moments.covariance_ann.copy()
    non_psd.iloc[0, 1] = 5.0
    non_psd.iloc[1, 0] = 5.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        efficient_frontier(dataclasses.replace(moments, covariance_ann=non_psd))

    asymmetric = moments.covariance_ann.copy()
    asymmetric.iloc[0, 1] += 1.0
    with pytest.raises(ValueError, match="symmetric"):
        efficient_frontier(dataclasses.replace(moments, covariance_ann=asymmetric))

    non_finite_mean = moments.mean_ann_arithmetic.copy()
    non_finite_mean.iloc[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        efficient_frontier(
            dataclasses.replace(moments, mean_ann_arithmetic=non_finite_mean)
        )

    relabeled = moments.covariance_ann.copy()
    relabeled.index = ["ASSET_B", "ASSET_A"]
    with pytest.raises(ValueError, match="asset labels"):
        efficient_frontier(dataclasses.replace(moments, covariance_ann=relabeled))

    with pytest.raises(ValueError, match="365.2425"):
        efficient_frontier(dataclasses.replace(moments, periods_per_year=252.0))
