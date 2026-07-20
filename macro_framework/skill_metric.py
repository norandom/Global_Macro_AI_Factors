"""Own-basket appraisal ratio (HAC) and single-factor market attribution.

Isolates *timing skill* from static own-factor exposure: the strategy's daily
returns are regressed on the four own-factor ETF daily returns with a **constant
beta** over the window, so allocation-timing shows up in the residual rather than
in the fitted factor loadings. This mirrors the own-4-ETF-basket regression in
``scripts/build_tear_sheet.py`` (``_ols`` / ``BASKET`` / ``residual_vol_ann_basket``)
but adds a Newey-West (HAC) t-statistic on the intercept.

Annualization basis (reused from ``macro_framework.ssr.TRADING_DAYS``):
- mean/alpha are scaled by ``periods_per_year`` (×252),
- volatility is scaled by ``sqrt(periods_per_year)`` (√252).

Pure and deterministic; no IO. Equal inputs → equal output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm

from macro_framework.ssr import TRADING_DAYS, SSRResult

IDIO_FLOOR: float = 1e-4  # annualized residual-vol floor below which appraisal is undefined


@dataclass(frozen=True)
class BasketResidual:
    alpha_ann: float
    t_alpha_hac: float
    r2: float
    idio_vol_ann: float
    appraisal: float | None  # alpha_ann / idio_vol_ann, or None if idio < IDIO_FLOOR
    n_obs: int
    hac_maxlags: int


@dataclass(frozen=True)
class MarketAttribution:
    alpha_ann: float
    beta: float
    r2: float


def _align(y: pd.Series, x: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Inner-join on the shared daily index and drop any NaN rows (LSE/US holiday mismatch)."""
    joined = pd.concat([y.rename("__y__"), x], axis=1, join="inner").dropna()
    return joined["__y__"], joined.drop(columns="__y__")


def basket_residual(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,  # own 4-ETF daily returns
    *,
    periods_per_year: int = TRADING_DAYS,
    hac_maxlags: int = 5,
) -> BasketResidual:
    """Constant-beta regression of strategy returns on the own factor basket.

    Reports annualized alpha, its Newey-West HAC t-statistic, R², annualized
    residual (idiosyncratic) volatility, and the appraisal ratio
    (alpha_ann / idio_vol_ann), returned as ``None`` when residual vol < IDIO_FLOOR.
    """
    y, x = _align(strategy_returns, factor_returns)
    res = sm.OLS(y.to_numpy(), sm.add_constant(x.to_numpy())).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_maxlags}
    )
    alpha_ann = float(res.params[0] * periods_per_year)
    t_alpha_hac = float(res.tvalues[0])
    r2 = float(res.rsquared)
    idio_vol_ann = float(np.asarray(res.resid).std(ddof=1) * np.sqrt(periods_per_year))
    appraisal = None if idio_vol_ann < IDIO_FLOOR else alpha_ann / idio_vol_ann
    return BasketResidual(
        alpha_ann=alpha_ann,
        t_alpha_hac=t_alpha_hac,
        r2=r2,
        idio_vol_ann=idio_vol_ann,
        appraisal=appraisal,
        n_obs=int(len(y)),
        hac_maxlags=hac_maxlags,
    )


def market_attribution(
    strategy_returns: pd.Series,
    market_returns: pd.Series,
    *,
    periods_per_year: int = TRADING_DAYS,
    hac_maxlags: int = 5,
) -> MarketAttribution:
    """Single-factor market attribution: annualized alpha, beta, and R²."""
    y, x = _align(strategy_returns, market_returns.to_frame("__mkt__"))
    res = sm.OLS(y.to_numpy(), sm.add_constant(x.to_numpy())).fit(
        cov_type="HAC", cov_kwds={"maxlags": hac_maxlags}
    )
    return MarketAttribution(
        alpha_ann=float(res.params[0] * periods_per_year),
        beta=float(res.params[1]),
        r2=float(res.rsquared),
    )


# --- Composite acceptance gates (Requirement 2) ----------------------------------


@dataclass(frozen=True)
class GateConfig:
    """Thresholds for the four keep/discard gates. Defaults encode the requirement.

    ``mode``:
    - ``"absolute"`` (default): skill gate passes iff ``t_alpha_hac > skill_t_min`` (2.2).
    - ``"relative_improvement"``: documented escape hatch for the structurally-small
      residual (design Open Questions 1-2). The absolute ``t>2`` bar is unreachable on
      a ~1.5y OOS window, so the skill gate instead passes on positive improvement +
      significance: ``t_alpha_hac > 0`` AND a defined, positive appraisal ratio. The
      other three gates are identical in both modes.
    """

    skill_t_min: float = 2.0
    ssr_min: float = 1.96
    recall_premium_max: float = 0.05  # |PIT vs non-PIT p_memorized delta| tolerance (~0)
    calmar_tolerance: float = 0.0  # OOS Calmar must be >= baseline - tolerance
    maxdd_tolerance: float = 0.0  # OOS |maxDD| must be <= baseline |maxDD| + tolerance
    mode: Literal["absolute", "relative_improvement"] = "absolute"


@dataclass(frozen=True)
class GateVerdict:
    passed: bool
    skill_pass: bool
    stability_pass: bool
    recall_pass: bool
    risk_shape_pass: bool
    first_failure: str | None  # e.g. "stability: SSR=0.14 < 1.96"
    values: dict[str, float]


def _skill_gate(residual: BasketResidual, config: GateConfig) -> tuple[bool, str | None]:
    t = residual.t_alpha_hac
    if not math.isfinite(t) or residual.appraisal is None:
        return False, f"skill: t={t} / appraisal={residual.appraisal} undefined"
    if config.mode == "relative_improvement":
        ok = t > 0.0 and residual.appraisal > 0.0
        msg = None if ok else f"skill(relative): t={t:.4g} or appraisal={residual.appraisal:.4g} not > 0"
        return ok, msg
    ok = t > config.skill_t_min
    return ok, None if ok else f"skill: t={t:.4g} <= {config.skill_t_min}"


def _stability_gate(ssr: SSRResult, config: GateConfig) -> tuple[bool, str | None]:
    val = ssr.ssr
    ok = math.isfinite(val) and val >= config.ssr_min
    return ok, None if ok else f"stability: SSR={val:.4g} < {config.ssr_min}"


def _recall_gate(recall_premium: float, config: GateConfig) -> tuple[bool, str | None]:
    ok = math.isfinite(recall_premium) and abs(recall_premium) <= config.recall_premium_max
    return ok, None if ok else f"recall: |premium|={abs(recall_premium):.4g} > {config.recall_premium_max}"


def _risk_shape_gate(
    oos_calmar: float,
    baseline_calmar: float,
    oos_maxdd: float,
    baseline_maxdd: float,
    config: GateConfig,
) -> tuple[bool, str | None]:
    finite = all(math.isfinite(x) for x in (oos_calmar, baseline_calmar, oos_maxdd, baseline_maxdd))
    if not finite:
        return False, "risk_shape: Calmar/maxDD undefined (non-finite input)"
    calmar_ok = oos_calmar >= baseline_calmar - config.calmar_tolerance
    dd_ok = abs(oos_maxdd) <= abs(baseline_maxdd) + config.maxdd_tolerance
    if not calmar_ok:
        return False, f"risk_shape: Calmar={oos_calmar:.4g} < baseline={baseline_calmar:.4g}"
    if not dd_ok:
        return False, f"risk_shape: |maxDD|={abs(oos_maxdd):.4g} > baseline={abs(baseline_maxdd):.4g}"
    return True, None


def evaluate_gates(
    residual: BasketResidual,
    ssr: SSRResult,
    recall_premium: float,  # from ContrastResult.contamination_premium()
    oos_calmar: float,
    baseline_calmar: float,
    oos_maxdd: float,
    baseline_maxdd: float,
    *,
    config: GateConfig = GateConfig(),
) -> GateVerdict:
    """Compose the four keep/discard gates into one PASS/FAIL verdict (2.1-2.6).

    ``passed`` is True iff every sub-gate passes under ``config.mode``. On failure
    ``first_failure`` names the first failing gate (skill -> stability -> recall ->
    risk_shape) and the value that missed its threshold. Degenerate inputs (NaN t /
    NaN SSR / None appraisal / non-finite Calmar-maxDD) fail their gate explicitly,
    never a silent pass.
    """
    skill_pass, skill_msg = _skill_gate(residual, config)
    stability_pass, stab_msg = _stability_gate(ssr, config)
    recall_pass, recall_msg = _recall_gate(recall_premium, config)
    risk_pass, risk_msg = _risk_shape_gate(
        oos_calmar, baseline_calmar, oos_maxdd, baseline_maxdd, config
    )

    first_failure = None
    for passed, msg in (
        (skill_pass, skill_msg),
        (stability_pass, stab_msg),
        (recall_pass, recall_msg),
        (risk_pass, risk_msg),
    ):
        if not passed:
            first_failure = msg
            break

    values = {
        "skill_t": float(residual.t_alpha_hac),
        "appraisal": float("nan") if residual.appraisal is None else float(residual.appraisal),
        "ssr": float(ssr.ssr),
        "recall_premium": float(recall_premium),
        "oos_calmar": float(oos_calmar),
        "baseline_calmar": float(baseline_calmar),
        "oos_maxdd": float(oos_maxdd),
        "baseline_maxdd": float(baseline_maxdd),
    }
    return GateVerdict(
        passed=skill_pass and stability_pass and recall_pass and risk_pass,
        skill_pass=skill_pass,
        stability_pass=stability_pass,
        recall_pass=recall_pass,
        risk_shape_pass=risk_pass,
        first_failure=first_failure,
        values=values,
    )
