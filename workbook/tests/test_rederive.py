"""Offline tests for the pure re-derivation formulas (task 3.1).

Each formula is checked against hand-computed values on synthetic inputs AND
against the published values carried by the checked-in fixtures. The parquet
fixtures are ROW SUBSETS of the real data-v1 assets, so full-stream statistics
cannot be reproduced from them; the published-value checks therefore use
row-local identities (e.g. ``guarded_tilt == raw * (1 - p)`` per fixture row)
and the internal identities of the row-complete published JSON summaries
(mean/median/metric deltas of ``factor_contrast_summary_v1.json``, the
``mean_std``/``mean_mac`` aggregates of ``factor_stability_v*.json``), which
the formulas must satisfy exactly within display precision (R2.6, R3.2, R4.3,
R4.4, R5.3, R6.1).
"""

import io
import json
import math
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factor_workbook.rederive import (
    ClassStats,
    EquityMetrics,
    PremiumResult,
    contamination_premium,
    equity_metrics,
    evidence_class_stats,
    guarded_tilt,
    loading_stability,
    paired_cohens_d,
    wilson_ci,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load_evidence(model: str = "openai_gpt-oss-20b") -> pd.DataFrame:
    """Extract one model's raw evidence records from the evidence tarball."""
    with tarfile.open(FIXTURES / "norecall_screen_evidence.tar.gz", "r:*") as tar:
        member = tar.extractfile(f"evidence/{model}/evidence.parquet")
        assert member is not None
        return pd.read_parquet(io.BytesIO(member.read()))


# --------------------------------------------------------------------------- #
# wilson_ci (R3.2)                                                             #
# --------------------------------------------------------------------------- #


class TestWilsonCI:
    def test_hand_computed_symmetric(self):
        # p=0.5, n=20, z=1.96 (nb13 form): denom = 1 + z^2/n = 1.19208,
        # center = (0.5 + z^2/40)/denom = 0.5 exactly; the bounds below are an
        # independent high-precision (decimal.Decimal, 30 digits) evaluation.
        lo, hi = wilson_ci(10, 20)
        assert lo == pytest.approx(0.29929491442981992, rel=1e-12)
        assert hi == pytest.approx(0.70070508557018008, rel=1e-12)
        assert lo + hi == pytest.approx(1.0, rel=1e-12)  # symmetric around 0.5

    def test_hand_computed_asymmetric(self):
        # Independent re-computation of the nb13 closed form for 3/10.
        p, n, z = 0.3, 10, 1.96
        denom = 1.0 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        hw = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        lo, hi = wilson_ci(3, 10)
        assert lo == pytest.approx(center - hw, rel=1e-12)
        assert hi == pytest.approx(center + hw, rel=1e-12)

    def test_extremes_stay_in_unit_interval(self):
        lo0, hi0 = wilson_ci(0, 10)
        lo1, hi1 = wilson_ci(10, 10)
        assert lo0 == pytest.approx(0.0)
        assert 0.0 < hi0 < 1.0
        assert 0.0 < lo1 < 1.0
        assert hi1 == pytest.approx(1.0)

    def test_zero_n_degrades_to_full_interval(self):
        assert wilson_ci(0, 0) == (0.0, 1.0)

    def test_fixture_accuracy_inside_interval(self):
        # Re-derived from the published per-call records (R3.2): the observed
        # accuracy must sit inside its own Wilson interval.
        eval_df = pd.read_parquet(
            FIXTURES / "naive_directional_eval_openai_gpt-oss-20b.parquet"
        )
        k = int(eval_df["correct"].sum())
        n = len(eval_df)
        lo, hi = wilson_ci(k, n)
        assert lo <= k / n <= hi
        # Cross-check against an independent inline evaluation of the form.
        p, z = k / n, 1.96
        denom = 1.0 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        hw = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        assert lo == pytest.approx(center - hw, rel=1e-12)
        assert hi == pytest.approx(center + hw, rel=1e-12)


# --------------------------------------------------------------------------- #
# guarded_tilt (R4.3)                                                          #
# --------------------------------------------------------------------------- #


class TestGuardedTilt:
    def test_hand_computed(self):
        raw = pd.Series([0.5, -0.4, 1.0])
        p = pd.Series([0.2, 0.5, 0.0])
        expected = pd.Series([0.4, -0.2, 1.0])
        pd.testing.assert_series_equal(guarded_tilt(raw, p), expected)

    def test_score_clipped_to_unit_interval(self):
        raw = pd.Series([1.0, -1.0])
        p = pd.Series([1.5, -0.5])  # clip -> 1.0 and 0.0
        expected = pd.Series([0.0, -1.0])
        pd.testing.assert_series_equal(guarded_tilt(raw, p), expected)

    def test_matches_published_fixture_values(self):
        # Row-local identity on the published release rows (R4.3): the
        # published guarded_tilt column IS raw * (1 - p_memorized).
        views = pd.read_parquet(FIXTURES / "factor_views_v1.parquet")
        rederived = guarded_tilt(views["raw_tilt"], views["p_memorized"])
        np.testing.assert_allclose(
            rederived.to_numpy(), views["guarded_tilt"].to_numpy(), rtol=1e-12
        )


# --------------------------------------------------------------------------- #
# paired_cohens_d + contamination_premium (R6.1)                               #
# --------------------------------------------------------------------------- #


class TestPairedCohensD:
    def test_hand_computed(self):
        # deltas [1,2,3,4]: mean 2.5, population std sqrt(1.25) -> d = sqrt(5).
        assert paired_cohens_d(pd.Series([1.0, 2.0, 3.0, 4.0])) == pytest.approx(
            math.sqrt(5.0), rel=1e-12
        )

    def test_degenerate_inputs_degrade_to_zero(self):
        assert paired_cohens_d(pd.Series([], dtype=float)) == 0.0
        assert paired_cohens_d(pd.Series([0.7])) == 0.0
        assert paired_cohens_d(pd.Series([0.3, 0.3, 0.3])) == 0.0  # zero variance


class TestContaminationPremium:
    def test_hand_computed(self):
        pit = pd.Series([0.1, 0.2, 0.3, 0.4])
        nonpit = pd.Series([0.6, 0.5, 0.9, 0.8])
        result = contamination_premium(pit, nonpit)
        assert isinstance(result, PremiumResult)
        assert result.n_pairs == 4
        assert result.mean_delta == pytest.approx(0.45, rel=1e-12)
        # median(nonpit)=0.7, median(pit)=0.25
        assert result.median_delta == pytest.approx(0.45, rel=1e-12)
        # deltas [0.5,0.3,0.6,0.4]: mean 0.45, pop std sqrt(0.0125)
        assert result.paired_d == pytest.approx(0.45 / math.sqrt(0.0125), rel=1e-12)

    def test_empty_degrades_to_zero(self):
        empty = pd.Series([], dtype=float)
        result = contamination_premium(empty, empty)
        assert result.n_pairs == 0
        assert result.mean_delta == 0.0
        assert result.median_delta == 0.0
        assert result.paired_d == 0.0

    def test_contrast_fixture_row_identity_and_self_consistency(self):
        # The published per-date contrast rows carry delta = nonpit - pit;
        # the premium re-derived from the rows must be internally consistent
        # with those very rows (the fixture is a row subset of the full
        # 72-pair stream, so the full-data figures are checked structurally
        # via the summary identities below).
        contrast = pd.read_parquet(FIXTURES / "factor_contrast_v1.parquet")
        np.testing.assert_allclose(
            contrast["delta"].to_numpy(),
            (contrast["nonpit_p"] - contrast["pit_p"]).to_numpy(),
            rtol=1e-9,
        )
        result = contamination_premium(contrast["pit_p"], contrast["nonpit_p"])
        assert result.n_pairs == len(contrast)
        assert result.mean_delta == pytest.approx(
            float(contrast["delta"].mean()), rel=1e-12
        )
        assert result.paired_d == pytest.approx(
            paired_cohens_d(contrast["delta"]), rel=1e-12
        )

    def test_published_summary_delta_identities(self):
        # The row-complete published summary must satisfy the premium's own
        # semantics exactly within display precision (R6.1): every published
        # delta is nonpit - pit of the published per-variant figures.
        summary = json.loads(
            (FIXTURES / "factor_contrast_summary_v1.json").read_text()
        )
        premium = summary["contamination_premium"]
        assert premium["p_memorized_mean_delta"] == pytest.approx(
            summary["nonpit_p_memorized"]["mean"]
            - summary["pit_p_memorized"]["mean"],
            rel=1e-9,
        )
        assert premium["p_memorized_median_delta"] == pytest.approx(
            summary["nonpit_p_memorized"]["median"]
            - summary["pit_p_memorized"]["median"],
            rel=1e-9,
        )
        for metric, pit_value in summary["pit_metrics"].items():
            assert premium[f"{metric}_delta"] == pytest.approx(
                summary["nonpit_metrics"][metric] - pit_value, abs=1e-11
            ), metric


# --------------------------------------------------------------------------- #
# loading_stability (R4.4)                                                     #
# --------------------------------------------------------------------------- #


class TestLoadingStability:
    def test_hand_computed(self):
        # Two axes, four dates, one parse_ok=False row (skipped), deliberately
        # unsorted index (must be date-sorted before the mean-abs-change).
        idx = pd.to_datetime(["2020-03-01", "2020-01-01", "2020-02-01", "2020-04-01"])
        loadings = pd.DataFrame(
            {"growth": [0.5, 0.1, 0.3, 0.9], "policy": [-0.2, 0.0, -0.4, 0.6]},
            index=idx,
        )
        parse_ok = pd.Series([True, True, True, False], index=idx)
        result = loading_stability(loadings, parse_ok)
        # Date-sorted parsed growth: [0.1, 0.3, 0.5]
        assert result["growth_std"] == pytest.approx(
            float(np.std([0.1, 0.3, 0.5])), rel=1e-12
        )
        assert result["growth_mac"] == pytest.approx(0.2, rel=1e-12)
        # Date-sorted parsed policy: [0.0, -0.4, -0.2]
        assert result["policy_std"] == pytest.approx(
            float(np.std([0.0, -0.4, -0.2])), rel=1e-12
        )
        assert result["policy_mac"] == pytest.approx(0.3, rel=1e-12)
        assert result["mean_std"] == pytest.approx(
            (result["growth_std"] + result["policy_std"]) / 2, rel=1e-12
        )
        assert result["mean_mac"] == pytest.approx(
            (result["growth_mac"] + result["policy_mac"]) / 2, rel=1e-12
        )

    def test_nan_axis_values_contribute_nothing(self):
        idx = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])
        loadings = pd.DataFrame({"growth": [0.1, np.nan, 0.5]}, index=idx)
        parse_ok = pd.Series([True, True, True], index=idx)
        result = loading_stability(loadings, parse_ok)
        # Series is [0.1, 0.5]: std over 2 points, one transition of 0.4.
        assert result["growth_std"] == pytest.approx(0.2, rel=1e-12)
        assert result["growth_mac"] == pytest.approx(0.4, rel=1e-12)

    def test_degenerate_streams_are_all_zero(self):
        idx = pd.to_datetime(["2020-01-01", "2020-02-01"])
        loadings = pd.DataFrame({"growth": [0.1, 0.2]}, index=idx)
        none_parsed = pd.Series([False, False], index=idx)
        result = loading_stability(loadings, none_parsed)
        assert result == {"growth_std": 0.0, "growth_mac": 0.0,
                          "mean_std": 0.0, "mean_mac": 0.0}
        empty = loading_stability(loadings.iloc[:0], none_parsed.iloc[:0])
        assert empty == {"growth_std": 0.0, "growth_mac": 0.0,
                         "mean_std": 0.0, "mean_mac": 0.0}

    def test_fixture_self_consistency(self):
        # Re-derive over the published fixture rows and check against an
        # independent numpy computation on the parse_ok subset.
        loadings = pd.read_parquet(FIXTURES / "factor_loadings_v1.parquet")
        parse_ok = loadings.pop("parse_ok")
        result = loading_stability(loadings, parse_ok)
        parsed = loadings[parse_ok.astype(bool)].sort_index()
        for axis in loadings.columns:
            vals = parsed[axis].dropna().to_numpy()
            assert result[f"{axis}_std"] == pytest.approx(
                float(np.std(vals)) if len(vals) >= 2 else 0.0, rel=1e-12
            )
            assert result[f"{axis}_mac"] == pytest.approx(
                float(np.mean(np.abs(np.diff(vals)))) if len(vals) >= 2 else 0.0,
                rel=1e-12,
            )

    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_published_stability_aggregate_identity(self, version):
        # The published full-stream stability summaries must satisfy the
        # aggregate formula exactly within display precision (R4.4):
        # mean_std / mean_mac are the means of the published per-axis values.
        published = json.loads(
            (FIXTURES / f"factor_stability_{version}.json").read_text()
        )
        stds = [v for k, v in published.items()
                if k.endswith("_std") and k != "mean_std"]
        macs = [v for k, v in published.items()
                if k.endswith("_mac") and k != "mean_mac"]
        assert published["mean_std"] == pytest.approx(
            sum(stds) / len(stds), rel=1e-9
        )
        assert published["mean_mac"] == pytest.approx(
            sum(macs) / len(macs), rel=1e-9
        )


# --------------------------------------------------------------------------- #
# equity_metrics (R5.3)                                                        #
# --------------------------------------------------------------------------- #


class TestEquityMetrics:
    def test_hand_computed_including_crisis_window(self):
        # Four daily values spanning the 2022 crisis window boundary.
        idx = pd.to_datetime(
            ["2021-12-30", "2021-12-31", "2022-01-03", "2022-01-04", "2022-01-05"]
        )
        value = pd.Series([100.0, 110.0, 99.0, 105.93, 103.0], index=idx)
        m = equity_metrics(value)
        assert isinstance(m, EquityMetrics)

        returns = value.pct_change().dropna().to_numpy()
        total = 103.0 / 100.0 - 1.0
        ann_ret = (1.0 + total) ** (252 / len(returns)) - 1.0
        ann_vol = float(np.std(returns, ddof=1)) * math.sqrt(252)
        assert m.total_return == pytest.approx(total, rel=1e-12)
        assert m.annualized_return == pytest.approx(ann_ret, rel=1e-12)
        assert m.annualized_vol == pytest.approx(ann_vol, rel=1e-12)
        assert m.sharpe == pytest.approx(ann_ret / ann_vol, rel=1e-12)
        downside = returns[returns < 0]
        dstd = float(np.std(downside)) * math.sqrt(252)
        assert m.sortino == pytest.approx(ann_ret / dstd, rel=1e-12)

        # Max drawdown from the running max of the value line: peak 110 ->
        # trough 99.
        max_dd = 99.0 / 110.0 - 1.0
        assert m.max_drawdown == pytest.approx(max_dd, rel=1e-12)
        assert m.calmar == pytest.approx(ann_ret / abs(max_dd), rel=1e-12)

        # Crisis window is the fixed 2022 slice: [99.0, 105.93, 103.0].
        window = value.loc["2022-01-01":"2022-12-31"]
        assert m.crisis_return == pytest.approx(103.0 / 99.0 - 1.0, rel=1e-12)
        crisis_dd = (window / window.cummax() - 1.0).min()
        assert m.crisis_max_drawdown == pytest.approx(float(crisis_dd), rel=1e-12)
        crisis_vol = float(window.pct_change().std(ddof=1)) * math.sqrt(252)
        assert m.crisis_vol_ann == pytest.approx(crisis_vol, rel=1e-12)

    def test_constant_series_degrades_to_zero_ratios(self):
        idx = pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05"])
        m = equity_metrics(pd.Series([100.0, 100.0, 100.0], index=idx))
        assert m.total_return == 0.0
        assert m.annualized_vol == 0.0
        assert m.sharpe == 0.0
        assert m.sortino == 0.0
        assert m.max_drawdown == 0.0
        assert m.calmar == 0.0

    def test_fixture_outside_crisis_window_yields_nan_crisis_fields(self):
        # The equity fixture rows are 2014 dates: no 2022 crisis overlap.
        value = pd.read_parquet(FIXTURES / "factor_equity_v1.parquet")["value"]
        m = equity_metrics(value)
        assert m.total_return == pytest.approx(
            float(value.iloc[-1] / value.iloc[0] - 1.0), rel=1e-12
        )
        dd = (value / value.cummax() - 1.0).min()
        assert m.max_drawdown == pytest.approx(float(dd), abs=1e-12)
        assert math.isnan(m.crisis_return)
        assert math.isnan(m.crisis_max_drawdown)
        assert math.isnan(m.crisis_vol_ann)


# --------------------------------------------------------------------------- #
# evidence_class_stats (R2.6)                                                  #
# --------------------------------------------------------------------------- #


class TestEvidenceClassStats:
    def test_hand_computed_two_arms(self):
        evidence = pd.DataFrame(
            {
                "arm": ["identifying", "identifying", "control"],
                "std_loss": [1.0, 3.0, 5.0],
                "std_min_k": [0.5, 0.5, 2.0],
                "raw_loss": [9.0, 9.0, 9.0],  # non-std_* column: ignored
            }
        )
        stats = evidence_class_stats(evidence)
        assert isinstance(stats, ClassStats)
        assert stats.arm_counts == {"control": 1, "identifying": 2}
        fs = stats.feature_stats
        assert set(fs.columns) == {"std_loss_mean", "std_loss_std",
                                   "std_min_k_mean", "std_min_k_std"}
        assert fs.loc["identifying", "std_loss_mean"] == pytest.approx(2.0)
        assert fs.loc["identifying", "std_loss_std"] == pytest.approx(
            math.sqrt(2.0), rel=1e-12
        )  # sample std of [1, 3]
        assert fs.loc["control", "std_min_k_mean"] == pytest.approx(2.0)

    def test_fixture_evidence_records(self):
        # Re-derived class counts + std_* summaries from the published raw
        # evidence records (R2.6), checked against an independent pandas
        # computation over the same rows.
        evidence = _load_evidence()
        stats = evidence_class_stats(evidence)
        expected_counts = evidence["arm"].value_counts().to_dict()
        assert stats.arm_counts == expected_counts
        std_cols = [c for c in evidence.columns if c.startswith("std_")]
        assert std_cols, "fixture must carry std_* feature columns"
        for arm, group in evidence.groupby("arm"):
            for col in std_cols:
                assert stats.feature_stats.loc[arm, f"{col}_mean"] == pytest.approx(
                    float(group[col].mean()), rel=1e-12
                )
