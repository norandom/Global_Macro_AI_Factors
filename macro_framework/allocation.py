"""Portfolio weight construction: HRP with CVaR + Black-Litterman tilt.

Riskfolio-Lib is imported lazily — keeps `import macro_framework` cheap.
"""

from __future__ import annotations

import contextlib
import io
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    import riskfolio as rp


def hrp_cvar_weights(
    returns: pd.DataFrame,
    alpha: float = 0.05,
    linkage: str = "single",
    codependence: str = "pearson",
) -> pd.Series:
    """Hierarchical Risk Parity with CVaR as the risk measure."""
    import riskfolio as rp

    hcp = rp.HCPortfolio(returns=returns, alpha=alpha)
    w = hcp.optimization(
        model="HRP",
        codependence=codependence,
        rm="CVaR",
        rf=0.0,
        linkage=linkage,
        max_k=10,
        leaf_order=True,
    )
    return w["weights"].reindex(returns.columns).fillna(0.0)


def hrp_cvar_weights_with_fixed(
    returns: pd.DataFrame,
    fixed_weights: dict[str, float],
    alpha: float = 0.05,
) -> pd.Series:
    """Pin `fixed_weights` (e.g. a cash slot), run HRP-CVaR on the remaining assets,
    scaled so the total sums to 1. Useful when a near-zero-vol asset would otherwise
    dominate pure HRP.
    """
    fixed = pd.Series(fixed_weights)
    if fixed.sum() >= 1.0 or fixed.sum() < 0.0:
        raise ValueError(f"fixed weights must sum to [0, 1); got {fixed.sum():.3f}")
    risky = [c for c in returns.columns if c not in fixed.index]
    if not risky:
        return fixed.reindex(returns.columns).fillna(0.0)
    w_risky = hrp_cvar_weights(returns[risky], alpha=alpha) * (1.0 - fixed.sum())
    return pd.concat([w_risky, fixed]).reindex(returns.columns).fillna(0.0)


def bl_mv_weights(
    returns: pd.DataFrame,
    prior_weights: pd.Series,
    P: np.ndarray | pd.DataFrame,
    Q: np.ndarray | pd.DataFrame,
    tau: float = 0.05,
    delta: float | None = None,
    rf_daily: float = 0.0,
    obj: str = "Utility",
) -> pd.Series:
    """Mean-variance weights under Black-Litterman posterior.

    `prior_weights` are the equilibrium portfolio (we use HRP as prior).
    `P` / `Q` encode views; Q is expressed at the **returns frequency** (daily here).
    `obj`: 'Utility' (max utility — stable default), 'Sharpe' (tangency), 'MinRisk'.
    """
    import riskfolio as rp

    P_df = P if isinstance(P, pd.DataFrame) else pd.DataFrame(P, columns=returns.columns)
    Q_df = Q if isinstance(Q, pd.DataFrame) else pd.DataFrame(np.asarray(Q).reshape(-1, 1))
    w_prior = prior_weights.reindex(returns.columns).to_frame()

    with contextlib.redirect_stdout(io.StringIO()):
        port = rp.Portfolio(returns=returns)
        port.assets_stats(method_mu="hist", method_cov="ledoit")
        port.blacklitterman_stats(P=P_df, Q=Q_df, rf=rf_daily, w=w_prior, delta=delta, eq=True)
        w = port.optimization(model="BL", rm="MV", obj=obj, rf=rf_daily, hist=False)
    return w["weights"].reindex(returns.columns).fillna(0.0)


def hrp_bl_blend(
    returns: pd.DataFrame,
    views_P: np.ndarray | pd.DataFrame | None = None,
    views_Q: np.ndarray | pd.DataFrame | None = None,
    tilt: float = 0.3,
    tau: float = 0.05,
    alpha: float = 0.05,
) -> pd.Series:
    """HRP-CVaR base weights optionally tilted toward Black-Litterman MV weights.

    tilt ∈ [0,1]: 0 = pure HRP, 1 = pure BL-MV. Default 0.3 keeps HRP as the risk
    backbone and lets views nudge the allocation.
    """
    w_hrp = hrp_cvar_weights(returns, alpha=alpha)
    if views_P is None or views_Q is None or tilt <= 0.0:
        return w_hrp
    w_bl = bl_mv_weights(returns, prior_weights=w_hrp, P=views_P, Q=views_Q, tau=tau)
    w = (1.0 - tilt) * w_hrp + tilt * w_bl
    return w / w.sum()
