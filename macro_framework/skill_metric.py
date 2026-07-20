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

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from macro_framework.ssr import TRADING_DAYS

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
