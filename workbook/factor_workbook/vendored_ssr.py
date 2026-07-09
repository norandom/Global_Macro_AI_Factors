# Vendored VERBATIM from macro_framework/ssr.py (R6.2).
# Source: /home/mc/projects/Global_Macro_AI_Factors/macro_framework/ssr.py
#         (repo-relative: macro_framework/ssr.py)
# Commit: c8b03e7efafc42a6567dbd23566a5531a3e1276d
# Do not edit by hand — re-sync from the source module and re-run
# workbook/tests/test_parity_root_env.py if the original diverges.
"""Sharpe Stability Ratio — Bajo Traver & Rodríguez Domínguez (2026).

Z_t     = rolling Sharpe (window τ) of excess returns, annualized
SSR     = (mean(Z) - SR*) / sigma_HAC(Z)
sigma_HAC uses Newey-West Bartlett kernel with Andrews (1991) automatic bandwidth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def rolling_sharpe(returns: pd.Series, window: int = TRADING_DAYS) -> pd.Series:
    mu = returns.rolling(window).mean()
    sigma = returns.rolling(window).std(ddof=1)
    sr = (mu / sigma) * np.sqrt(TRADING_DAYS)
    return sr.dropna()


def andrews_bandwidth(z: np.ndarray) -> int:
    """Andrews (1991) data-dependent bandwidth for Bartlett kernel (AR(1) plug-in)."""
    n = len(z)
    if n < 4:
        return 1
    zc = z - z.mean()
    gamma0 = float((zc * zc).mean())
    if gamma0 <= 0:
        return 1
    gamma1 = float((zc[1:] * zc[:-1]).mean())
    rho = float(np.clip(gamma1 / gamma0, -0.97, 0.97))
    alpha = (4.0 * rho**2) / ((1.0 - rho) ** 2 * (1.0 + rho) ** 2)
    L = 1.1447 * (alpha * n) ** (1.0 / 3.0)
    return max(1, min(int(np.floor(L)), n // 4))


def newey_west_var(z: np.ndarray, L: int | None = None) -> tuple[float, int]:
    """Newey-West HAC long-run variance with Bartlett kernel."""
    z = np.asarray(z, dtype=float)
    if L is None:
        L = andrews_bandwidth(z)
    zc = z - z.mean()
    total = float((zc * zc).mean())
    for k in range(1, L + 1):
        gk = float((zc[k:] * zc[:-k]).mean())
        total += 2.0 * (1.0 - k / (L + 1)) * gk
    return max(total, 0.0), L


@dataclass(frozen=True)
class SSRResult:
    n_obs: int
    n_rolling: int
    sr_full: float
    mean_rolling_sr: float
    sigma_hac: float
    L_hac: int
    ssr: float


def compute_ssr(returns: pd.Series, window: int = TRADING_DAYS, sr_star: float = 0.0) -> SSRResult:
    r = returns.dropna()
    rolling = rolling_sharpe(r, window=window)
    if len(rolling) < 10:
        return SSRResult(len(r), len(rolling), np.nan, np.nan, np.nan, 0, np.nan)
    z = rolling.to_numpy()
    z_bar = float(z.mean())
    sigma2, L = newey_west_var(z)
    sigma_hac = float(np.sqrt(sigma2)) if sigma2 > 0 else np.nan
    ssr = (z_bar - sr_star) / sigma_hac if sigma_hac and sigma_hac > 0 else np.nan
    sr_full = float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return SSRResult(
        n_obs=int(len(r)),
        n_rolling=int(len(rolling)),
        sr_full=sr_full,
        mean_rolling_sr=z_bar,
        sigma_hac=sigma_hac,
        L_hac=int(L),
        ssr=float(ssr) if np.isfinite(ssr) else np.nan,
    )
