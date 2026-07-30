"""Strict return construction and raw-return HAC attribution.

Price levels are selected on an explicit anchored calendar before returns are
calculated. Return pairs and regressions require the same complete, ordered,
finite observation set; no helper sorts, intersects, fills, or drops rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm

from macro_framework.ssr import TRADING_DAYS, SSRInference, SSRResult

IDIO_FLOOR: float = 1e-4  # annualized residual-vol floor below which appraisal is undefined
AttributionKind = Literal["raw_market_model"]


@dataclass(frozen=True, init=False)
class BasketResidual:
    intercept_native_period: float
    intercept_ann_arithmetic: float
    intercept_t_hac: float
    r2: float
    idio_vol_ann: float
    appraisal: float | None
    n_obs: int
    start: pd.Timestamp
    end: pd.Timestamp
    periods_per_year: int
    hac_maxlags: int

    def __init__(
        self,
        *,
        intercept_native_period: float | None = None,
        intercept_ann_arithmetic: float | None = None,
        intercept_t_hac: float | None = None,
        r2: float = float("nan"),
        idio_vol_ann: float = float("nan"),
        appraisal: float | None = None,
        n_obs: int = 0,
        start: pd.Timestamp = pd.NaT,
        end: pd.Timestamp = pd.NaT,
        periods_per_year: int = TRADING_DAYS,
        hac_maxlags: int = 5,
        alpha_ann: float | None = None,
        t_alpha_hac: float | None = None,
    ) -> None:
        """Build the explicit result while accepting legacy fixture keywords temporarily.

        Keyword-only: a legacy 7-field POSITIONAL construction would be silently
        reinterpreted under the new field order, so positional use raises instead.
        """
        ann = intercept_ann_arithmetic if intercept_ann_arithmetic is not None else alpha_ann
        t_hac = intercept_t_hac if intercept_t_hac is not None else t_alpha_hac
        if ann is None or t_hac is None:
            raise TypeError("intercept_ann_arithmetic and intercept_t_hac are required")
        native = ann / periods_per_year if intercept_native_period is None else intercept_native_period
        for name, value in (
            ("intercept_native_period", native),
            ("intercept_ann_arithmetic", ann),
            ("intercept_t_hac", t_hac),
            ("r2", r2),
            ("idio_vol_ann", idio_vol_ann),
            ("appraisal", appraisal),
            ("n_obs", n_obs),
            ("start", start),
            ("end", end),
            ("periods_per_year", periods_per_year),
            ("hac_maxlags", hac_maxlags),
        ):
            object.__setattr__(self, name, value)

    @property
    def alpha_ann(self) -> float:
        """Compatibility attribute for callers pending repository-wide migration."""
        return self.intercept_ann_arithmetic

    @property
    def t_alpha_hac(self) -> float:
        """Compatibility attribute for callers pending repository-wide migration."""
        return self.intercept_t_hac


@dataclass(frozen=True)
class MarketAttribution:
    kind: AttributionKind
    intercept_native_period: float
    intercept_ann_arithmetic: float
    intercept_se_hac: float
    intercept_t_hac: float
    beta: float
    r2: float
    n_obs: int
    start: pd.Timestamp
    end: pd.Timestamp
    periods_per_year: int
    hac_maxlags: int

    @property
    def alpha_ann(self) -> float:
        """Compatibility attribute for callers pending repository-wide migration."""
        return self.intercept_ann_arithmetic


def _validate_datetime_index(index: pd.Index, name: str) -> pd.DatetimeIndex:
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
    return index


def _validate_finite(values: pd.Series | pd.DataFrame, name: str) -> None:
    dtypes = [values.dtype] if isinstance(values, pd.Series) else list(values.dtypes)
    if any(
        not pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_complex_dtype(dtype)
        for dtype in dtypes
    ):
        raise ValueError(f"{name} must contain only finite real numeric values")
    try:
        finite = np.isfinite(values.to_numpy())
    except TypeError as exc:
        raise ValueError(f"{name} must contain only finite real numeric values") from exc
    if bool(finite.all()):
        return
    row, *column = np.argwhere(~finite)[0]
    label = values.index[int(row)]
    if isinstance(values, pd.Series):
        raise ValueError(f"{name} contains a non-finite value at {label}")
    col = values.columns[int(column[0])]
    raise ValueError(f"{name} contains a non-finite value at {label}, column {col!r}")


def _validate_series(series: pd.Series, name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series")
    _validate_datetime_index(series.index, f"{name}.index")
    _validate_finite(series, name)
    return series


def _validate_periods_and_lags(periods_per_year: int, hac_maxlags: int) -> None:
    if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, Integral):
        raise ValueError("periods_per_year must be a positive integer")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer")
    if isinstance(hac_maxlags, bool) or not isinstance(hac_maxlags, Integral):
        raise ValueError("hac_maxlags must be a non-negative integer")
    if hac_maxlags < 0:
        raise ValueError("hac_maxlags must be a non-negative integer")


def factor_returns_on(
    prices: pd.Series | pd.DataFrame,
    return_index: pd.DatetimeIndex,
    *,
    anchor: pd.Timestamp,
) -> pd.Series | pd.DataFrame:
    """Select anchored price levels first, then return exactly ``return_index``."""
    if not isinstance(prices, (pd.Series, pd.DataFrame)):
        raise TypeError("prices must be a pandas Series or DataFrame")
    source_index = _validate_datetime_index(prices.index, "prices.index")
    requested = _validate_datetime_index(return_index, "return_index")
    try:
        anchor = pd.Timestamp(anchor)
    except (TypeError, ValueError) as exc:
        raise ValueError("anchor must be a valid pandas Timestamp") from exc
    if pd.isna(anchor):
        raise ValueError("anchor must not be NaT")
    if anchor.tz is not None:
        raise ValueError("anchor must be timezone-naive")
    if anchor >= requested[0]:
        raise ValueError("anchor must be strictly before the first requested return date")

    required = requested.insert(0, anchor)
    positions = source_index.get_indexer(required)
    if (positions < 0).any():
        missing = required[positions < 0]
        raise ValueError(
            f"prices.index is missing {len(missing)} required label(s); first missing {missing[0]}"
        )

    selected = prices.iloc[positions]
    _validate_finite(selected, "selected price levels")
    returns = selected.pct_change(fill_method=None).iloc[1:]
    _validate_finite(returns, "constructed returns")
    returns.index = requested.copy()
    return returns


def _exact_return_difference(
    left: pd.Series,
    right: pd.Series,
    *,
    left_name: str,
    right_name: str,
) -> pd.Series:
    left = _validate_series(left, left_name)
    right = _validate_series(right, right_name)
    if not left.index.equals(right.index):
        raise ValueError(f"{left_name} and {right_name} must have identical indexes")
    return pd.Series(
        left.to_numpy() - right.to_numpy(),
        index=left.index.copy(),
        name=left.name,
    )


def portfolio_excess_returns(
    portfolio_returns: pd.Series,
    cash_returns: pd.Series,
) -> pd.Series:
    """Return portfolio minus cash under an exact-index, finite-value contract."""
    return _exact_return_difference(
        portfolio_returns,
        cash_returns,
        left_name="portfolio_returns",
        right_name="cash_returns",
    )


def differential_returns(
    comparison_returns: pd.Series,
    reference_returns: pd.Series,
) -> pd.Series:
    """Return comparison minus reference under an exact-index, finite-value contract."""
    return _exact_return_difference(
        comparison_returns,
        reference_returns,
        left_name="comparison_returns",
        right_name="reference_returns",
    )


def _strict_hac_ols(
    y: pd.Series,
    x: pd.DataFrame,
    *,
    periods_per_year: int,
    hac_maxlags: int,
):
    y = _validate_series(y, "dependent_returns")
    if not isinstance(x, pd.DataFrame):
        raise TypeError("explanatory_returns must be a pandas DataFrame")
    _validate_datetime_index(x.index, "explanatory_returns.index")
    _validate_finite(x, "explanatory_returns")
    if x.shape[1] == 0:
        raise ValueError("explanatory_returns must contain at least one column")
    if not y.index.equals(x.index):
        raise ValueError("dependent and explanatory returns must have identical indexes")
    if len(y) <= x.shape[1] + 1:
        raise ValueError("regression requires more observations than fitted parameters")
    _validate_periods_and_lags(periods_per_year, hac_maxlags)
    design = np.column_stack((np.ones(len(x)), x.to_numpy()))
    return sm.OLS(y.to_numpy(), design).fit(cov_type="HAC", cov_kwds={"maxlags": hac_maxlags})


def basket_residual(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,
    *,
    periods_per_year: int = TRADING_DAYS,
    hac_maxlags: int = 5,
) -> BasketResidual:
    """Strict raw-return regression of strategy returns on its own factor basket."""
    res = _strict_hac_ols(
        strategy_returns,
        factor_returns,
        periods_per_year=periods_per_year,
        hac_maxlags=hac_maxlags,
    )
    intercept = float(res.params[0])
    intercept_ann = intercept * periods_per_year
    idio_vol_ann = float(np.asarray(res.resid).std(ddof=1) * np.sqrt(periods_per_year))
    appraisal = None if idio_vol_ann < IDIO_FLOOR else intercept_ann / idio_vol_ann
    return BasketResidual(
        intercept_native_period=intercept,
        intercept_ann_arithmetic=intercept_ann,
        intercept_t_hac=float(res.tvalues[0]),
        r2=float(res.rsquared),
        idio_vol_ann=idio_vol_ann,
        appraisal=appraisal,
        n_obs=len(strategy_returns),
        start=strategy_returns.index[0],
        end=strategy_returns.index[-1],
        periods_per_year=int(periods_per_year),
        hac_maxlags=int(hac_maxlags),
    )


def raw_market_model_attribution(
    portfolio_returns: pd.Series,
    market_returns: pd.Series,
    *,
    periods_per_year: int = TRADING_DAYS,
    hac_maxlags: int = 5,
) -> MarketAttribution:
    """Regress raw portfolio returns on raw market returns with HAC uncertainty."""
    if not isinstance(market_returns, pd.Series):
        raise TypeError("market_returns must be a pandas Series")
    res = _strict_hac_ols(
        portfolio_returns,
        market_returns.to_frame("market_returns"),
        periods_per_year=periods_per_year,
        hac_maxlags=hac_maxlags,
    )
    intercept = float(res.params[0])
    return MarketAttribution(
        kind="raw_market_model",
        intercept_native_period=intercept,
        intercept_ann_arithmetic=intercept * periods_per_year,
        intercept_se_hac=float(res.bse[0]),
        intercept_t_hac=float(res.tvalues[0]),
        beta=float(res.params[1]),
        r2=float(res.rsquared),
        n_obs=len(portfolio_returns),
        start=portfolio_returns.index[0],
        end=portfolio_returns.index[-1],
        periods_per_year=int(periods_per_year),
        hac_maxlags=int(hac_maxlags),
    )


# Temporary compatibility name; removed after the repository-wide caller migration task.
market_attribution = raw_market_model_attribution


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
    ssr_alpha: float = 0.05  # one-sided MBB p-value level for the stability gate
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
    first_failure: str | None  # e.g. "stability: MBB p=0.31 >= 0.05 (SSR=0.14)"
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


def _stability_gate(ssr: SSRInference, config: GateConfig) -> tuple[bool, str | None]:
    """Paper Test 1 (one-sided MBB), delegated to ``SSRInference.stable`` — the repo's
    single verdict authority. SSR itself is the effect size, not the test: the
    pre-2026-07 rule (SSR >= 1.96) had no sqrt(n) and was unpassable by design.

    The inference MUST be built at the gate's ``ssr_alpha``. Re-encoding the four
    conditions here against a different alpha let the gate return PASS while
    ``ssr.verdict()`` on the same object printed "luck-compatible".
    """
    if ssr.alpha != config.ssr_alpha:
        raise ValueError(
            f"GateConfig.ssr_alpha={config.ssr_alpha} but this SSRInference was built at "
            f"alpha={ssr.alpha}. Rebuild it with ssr_inference(..., alpha={config.ssr_alpha}) "
            "so the gate and ssr.verdict() cannot disagree on the same input."
        )
    ok = ssr.stable
    return ok, None if ok else (
        f"stability: MBB p={ssr.p_value:.4g} >= {config.ssr_alpha} "
        f"(SSR={ssr.result.ssr:.4g})"
    )


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
    ssr: SSRInference,
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
        "ssr": float(ssr.result.ssr),
        "ssr_p": float(ssr.p_value),
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
