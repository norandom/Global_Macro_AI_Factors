"""Integration test: real ContrastResult.contamination_premium() → recall gate,
plus OOS-disjointness and look-ahead rejection (Task 8.2; R2.4, R3.1, R3.3).

Unlike the 6.2 unit tests (which fed the recall premium as a raw injected float),
this wires the actual cross-module contract: a real
``factor_scoring.ContrastResult`` produces the premium via
``.contamination_premium()``, and THAT value drives
``skill_metric.evaluate_gates``. Constructs ContrastResult directly (no NIM/scorer).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from macro_framework.skill_metric import (
    BasketResidual,
    GateConfig,
    evaluate_gates,
)
from macro_framework.ssr import SSRResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import factor_loop as fl  # noqa: E402


# --- Otherwise-passing skill / stability / risk-shape inputs so ONLY the recall
#     gate is under test (mirrors tests/test_skill_metric.py). --------------------
def _residual() -> BasketResidual:
    return BasketResidual(
        alpha_ann=0.1, t_alpha_hac=4.0, r2=0.9, idio_vol_ann=0.11,
        appraisal=0.9, n_obs=500, hac_maxlags=5,
    )


def _ssr() -> SSRResult:
    return SSRResult(
        n_obs=500, n_rolling=250, sr_full=1.0, mean_rolling_sr=1.0,
        sigma_hac=0.4, L_hac=5, ssr=2.5,
    )


_PASS = dict(oos_calmar=1.2, baseline_calmar=1.0, oos_maxdd=-0.10, baseline_maxdd=-0.15)


def _contrast(pit: list[float], nonpit: list[float]):
    # Lazy import (repo convention): keeps factor_scoring out of sys.modules at
    # collection time so test_factor_scoring's foundation import-guard stays green.
    from macro_framework.factor_scoring import ContrastResult

    return ContrastResult(
        pit_p_memorized=pit,
        nonpit_p_memorized=nonpit,
        pit_metrics={},
        nonpit_metrics={},
        n_pairs=len(pit),
    )


# --- R2.4: a high-contamination-premium non-PIT rung FAILS the recall gate. -------
def test_high_premium_contrast_fails_recall_gate():
    # non-PIT recall (real tickers+dates) memorizes hard; PIT (dateless) does not.
    contrast = _contrast(pit=[0.10, 0.12, 0.11], nonpit=[0.55, 0.60, 0.58])
    premium = contrast.contamination_premium()["p_memorized_mean_delta"]
    assert premium > GateConfig().recall_premium_max  # ~0.47 >> 0.05 tolerance

    verdict = evaluate_gates(_residual(), _ssr(), premium, **_PASS)

    assert verdict.recall_pass is False
    assert verdict.passed is False
    assert verdict.first_failure is not None and verdict.first_failure.startswith("recall:")


# --- Control: a ~0-premium PIT-clean contrast PASSES the recall gate. --------------
def test_zero_premium_contrast_passes_recall_gate():
    contrast = _contrast(pit=[0.30, 0.31, 0.29], nonpit=[0.30, 0.31, 0.29])
    premium = contrast.contamination_premium()["p_memorized_mean_delta"]
    assert abs(premium) <= GateConfig().recall_premium_max

    verdict = evaluate_gates(_residual(), _ssr(), premium, **_PASS)

    assert verdict.recall_pass is True
    assert verdict.passed is True
    assert verdict.first_failure is None


# --- R3.1: OOS window must be disjoint from the tuning/cutoff window. --------------
def test_oos_disjoint_rejects_overlap_and_accepts_disjoint():
    with pytest.raises(fl.ConfigurationError):
        fl.assert_oos_disjoint(("2024-07-01", "2025-12-31"), ("2016-01-01", "2024-12-31"))
    # disjoint: OOS strictly after the cutoff — must not raise.
    fl.assert_oos_disjoint(("2025-01-01", "2025-12-31"), ("2016-01-01", "2024-12-31"))


# --- R3.3: a future-requiring mutation is rejected; a cache-reusing one is not. ----
def test_lookahead_rejected_for_future_mutation_and_allowed_for_normal():
    future = fl.Mutation(
        kind="overlay", param="overlay", value={"requires_future": True}, rescoring=True
    )
    reason = fl.check_lookahead(future)
    assert reason is not None

    normal = fl.Mutation(kind="tau", param="tau", value=0.10, rescoring=False)
    assert fl.check_lookahead(normal) is None
