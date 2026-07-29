"""Coherent USD inputs for Markowitz analysis."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import pandas as pd
from scipy import optimize

from scripts.build_basket_long import validate_market_snapshot

QuoteCurrency = Literal["USD", "GBP"]
QuoteUnit = Literal["USD", "GBp"]

_SNAPSHOT_FILES = (
    "basket_adjusted_close_local.parquet",
    "cash_market_total_return.parquet",
    "fx_usd_per_gbp.parquet",
)
_MAX_STALENESS_DAYS = 3
_VALUATION_RULE = (
    "Friday 22:00 UTC; latest observation at or before cutoff; "
    "maximum staleness 3 calendar days"
)
WEEKLY_PERIODS_PER_YEAR = 365.2425 / 7
MOMENT_PSD_TOLERANCE = 1e-12
#: A frontier point is publishable only when its budget residual, target
#: residual, and bound violation - all recomputed from the stored weights -
#: are within this absolute tolerance.
FRONTIER_RESIDUAL_TOLERANCE = 1e-8
_QUOTE_FX_PROVENANCE_SCHEMA = "weekly-quote-fx-provenance-v1"


@dataclass(frozen=True)
class QuoteSpec:
    quote_currency: QuoteCurrency
    quote_unit: QuoteUnit
    scale_to_major: float

    def __post_init__(self) -> None:
        if self.quote_currency not in ("USD", "GBP"):
            raise ValueError(f"quote_currency must be USD or GBP, got {self.quote_currency!r}")
        if self.quote_unit not in ("USD", "GBp"):
            raise ValueError(f"quote_unit must be USD or GBp, got {self.quote_unit!r}")
        if type(self.scale_to_major) is not float:
            raise ValueError("scale_to_major must be a float")
        expected = ("GBP", 0.01) if self.quote_unit == "GBp" else ("USD", 1.0)
        if (self.quote_currency, self.scale_to_major) != expected:
            raise ValueError(
                f"{self.quote_unit} requires quote_currency={expected[0]} and "
                f"scale_to_major={expected[1]}"
            )


@dataclass(frozen=True)
class WeeklyValuations:
    levels_usd: pd.DataFrame
    observed_dates: pd.DataFrame
    fx_observed_dates: pd.Series
    asset_quote_specs: tuple[tuple[str, QuoteSpec], ...]
    producer_provenance_sha256: str
    base_currency: Literal["USD"]
    valuation_rule: str
    start: pd.Timestamp
    end: pd.Timestamp
    snapshot_id: str = ""
    fx_required_assets: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnnualizedMoments:
    weekly_returns: pd.DataFrame
    mean_ann_arithmetic: pd.Series
    covariance_ann: pd.DataFrame
    return_dates: pd.DatetimeIndex
    n_obs: int
    snapshot_id: str
    base_currency: Literal["USD"]
    valuation_rule: str
    periods_per_year: float
    start: pd.Timestamp
    end: pd.Timestamp
    psd_tolerance: float


@dataclass(frozen=True)
class FrontierPoint:
    """One long-only target solve with its complete solver diagnostics.

    Every diagnostic is recomputed from ``weights`` and the input moments, so
    the stored numbers always describe the stored weight vector. ``feasible``
    is True only when the solver reported success and every residual is within
    the frontier residual tolerance.
    """

    target_return_ann: float
    success: bool
    status: int
    message: str
    iterations: int
    objective: float
    budget_residual: float
    target_residual: float
    bound_violation: float
    weights: pd.Series
    return_ann: float
    volatility_ann: float
    feasible: bool


@dataclass(frozen=True)
class FrontierResult:
    """Deterministic long-only frontier retaining every attempted target."""

    points: tuple[FrontierPoint, ...]
    targets_ann: tuple[float, ...]
    n_targets: int
    n_feasible: int
    residual_tolerance: float
    snapshot_id: str
    base_currency: Literal["USD"]
    valuation_rule: str
    periods_per_year: float
    start: pd.Timestamp
    end: pd.Timestamp
    n_obs: int

    def publishable_points(self) -> tuple[FrontierPoint, ...]:
        """Return only feasible points; failed targets stay in ``points``."""
        return tuple(point for point in self.points if point.feasible)


def _quote_fx_provenance_sha256(
    *,
    snapshot_id: str,
    base_currency: str,
    valuation_rule: str,
    asset_quote_specs: tuple[tuple[str, QuoteSpec], ...],
    fx_required_assets: tuple[str, ...],
) -> str:
    """Seal producer-owned quote and FX lineage against coherent re-signing."""
    payload = {
        "schema": _QUOTE_FX_PROVENANCE_SCHEMA,
        "snapshot_id": snapshot_id,
        "base_currency": base_currency,
        "valuation_rule": valuation_rule,
        "asset_quote_specs": [
            {
                "asset": asset,
                "quote_currency": spec.quote_currency,
                "quote_unit": spec.quote_unit,
                "scale_to_major": spec.scale_to_major,
            }
            for asset, spec in asset_quote_specs
        ],
        "fx_required_assets": list(fx_required_assets),
    }
    canonical = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_observation_date_series(
    series: pd.Series,
    *,
    cutoffs: pd.DatetimeIndex,
    field_name: str,
) -> None:
    if not pd.api.types.is_datetime64_any_dtype(series.dtype):
        raise ValueError(f"{field_name} must contain date-granular datetime values")
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"{field_name} values must be timezone-naive source dates")
    if bool(series.isna().any()):
        first_missing = int(np.flatnonzero(series.isna().to_numpy())[0])
        raise ValueError(
            f"{field_name} is missing at cutoff {cutoffs[first_missing].isoformat()}"
        )

    observed = pd.DatetimeIndex(series.array)
    normalized = observed.normalize()
    if bool((observed != normalized).any()):
        first_intraday = int(np.flatnonzero(np.asarray(observed != normalized))[0])
        raise ValueError(
            f"{field_name} must be date-granular at cutoff "
            f"{cutoffs[first_intraday].isoformat()}"
        )

    cutoff_dates = cutoffs.tz_convert(None).normalize()
    look_ahead = observed > cutoff_dates
    if bool(look_ahead.any()):
        position = int(np.flatnonzero(np.asarray(look_ahead))[0])
        raise ValueError(
            f"{field_name} look-ahead date {observed[position].date()} exceeds cutoff "
            f"{cutoff_dates[position].date()}"
        )

    ages = cutoff_dates - observed
    stale = ages > pd.Timedelta(days=_MAX_STALENESS_DAYS)
    if bool(stale.any()):
        position = int(np.flatnonzero(np.asarray(stale))[0])
        raise ValueError(
            f"{field_name} date {observed[position].date()} for cutoff "
            f"{cutoff_dates[position].date()} is {ages[position].days} calendar days stale "
            f"(maximum {_MAX_STALENESS_DAYS})"
        )


def _validate_valuation_provenance(
    valuations: WeeklyValuations,
    *,
    levels: pd.DataFrame,
) -> None:
    cutoffs = levels.index
    expected_assets = levels.columns
    observed_dates = valuations.observed_dates
    if not isinstance(observed_dates, pd.DataFrame) or observed_dates.empty:
        raise ValueError("observed_dates must be a non-empty asset provenance matrix")
    if not observed_dates.index.identical(cutoffs):
        raise ValueError("observed_dates index must exactly match levels_usd valuation cutoffs")
    if not observed_dates.columns.identical(expected_assets):
        raise ValueError("observed_dates columns must exactly match levels_usd asset columns")
    for asset in expected_assets:
        _validate_observation_date_series(
            observed_dates[asset],
            cutoffs=cutoffs,
            field_name=f"observed_dates[{asset!r}]",
        )

    asset_quote_specs = valuations.asset_quote_specs
    if type(asset_quote_specs) is not tuple:
        raise ValueError("asset_quote_specs must be an immutable tuple")
    quote_assets: list[str] = []
    authoritative_required_assets: list[str] = []
    for position, entry in enumerate(asset_quote_specs):
        if type(entry) is not tuple or len(entry) != 2:
            raise ValueError(
                "asset_quote_specs entries must be (asset, QuoteSpec) tuples; "
                f"invalid entry at position {position}"
            )
        asset, spec = entry
        if not isinstance(asset, str) or not asset.strip() or type(spec) is not QuoteSpec:
            raise ValueError(
                "asset_quote_specs entries must contain a non-blank asset and QuoteSpec; "
                f"invalid entry at position {position}"
            )
        try:
            QuoteSpec(spec.quote_currency, spec.quote_unit, spec.scale_to_major)
        except (AttributeError, ValueError) as exc:
            raise ValueError(
                f"asset_quote_specs[{asset!r}] QuoteSpec is invalid: {exc}"
            ) from exc
        quote_assets.append(asset)
        if spec.quote_currency != valuations.base_currency:
            authoritative_required_assets.append(asset)
    if tuple(quote_assets) != tuple(expected_assets):
        raise ValueError(
            "asset_quote_specs assets must exactly match levels_usd asset columns in order"
        )

    required_assets = valuations.fx_required_assets
    if type(required_assets) is not tuple:
        raise ValueError("fx_required_assets must be a tuple of asset labels")
    if any(not isinstance(asset, str) or not asset.strip() for asset in required_assets):
        raise ValueError("fx_required_assets must contain non-blank string asset labels")
    if len(required_assets) != len(set(required_assets)):
        raise ValueError("fx_required_assets must contain unique asset labels")
    if required_assets != tuple(authoritative_required_assets):
        raise ValueError(
            "fx_required_assets must exactly match the authoritative quote metadata "
            "carried by weekly_usd_valuations"
        )

    producer_provenance = valuations.producer_provenance_sha256
    if (
        not isinstance(producer_provenance, str)
        or len(producer_provenance) != 64
        or any(character not in "0123456789abcdef" for character in producer_provenance)
    ):
        raise ValueError(
            "producer_provenance_sha256 must be a lowercase SHA-256 digest"
        )
    expected_provenance = _quote_fx_provenance_sha256(
        snapshot_id=valuations.snapshot_id,
        base_currency=valuations.base_currency,
        valuation_rule=valuations.valuation_rule,
        asset_quote_specs=asset_quote_specs,
        fx_required_assets=required_assets,
    )
    if producer_provenance != expected_provenance:
        raise ValueError(
            "producer-authorized quote/FX provenance does not match WeeklyValuations metadata"
        )

    fx_observed_dates = valuations.fx_observed_dates
    if not isinstance(fx_observed_dates, pd.Series) or fx_observed_dates.empty:
        raise ValueError("fx_observed_dates must be a non-empty provenance series")
    if not fx_observed_dates.index.identical(cutoffs):
        raise ValueError("fx_observed_dates index must exactly match levels_usd valuation cutoffs")
    if not pd.api.types.is_datetime64_any_dtype(fx_observed_dates.dtype):
        raise ValueError("fx_observed_dates must contain date-granular datetime values")
    if isinstance(fx_observed_dates.dtype, pd.DatetimeTZDtype):
        raise ValueError("fx_observed_dates values must be timezone-naive source dates")

    if required_assets:
        _validate_observation_date_series(
            fx_observed_dates,
            cutoffs=cutoffs,
            field_name="fx_observed_dates",
        )
    elif bool(fx_observed_dates.notna().any()):
        raise ValueError(
            "fx_observed_dates must be entirely missing when no asset requires FX conversion"
        )


def annualized_moments(
    valuations: WeeklyValuations,
    *,
    periods_per_year: float = WEEKLY_PERIODS_PER_YEAR,
) -> AnnualizedMoments:
    """Compute sample moments from consecutive common Friday valuations.

    Arithmetic weekly means and sample covariance are scaled by exactly
    ``365.2425 / 7``. Covariance is accepted as positive semidefinite when its
    smallest eigenvalue is at least ``-MOMENT_PSD_TOLERANCE``.
    """
    if not isinstance(valuations, WeeklyValuations):
        raise ValueError("valuations must be a WeeklyValuations instance")
    if isinstance(periods_per_year, (bool, np.bool_)):
        raise ValueError("weekly annualization must be exactly 365.2425 / 7")
    try:
        annualization = float(periods_per_year)
    except (TypeError, ValueError) as exc:
        raise ValueError("weekly annualization must be exactly 365.2425 / 7") from exc
    if not np.isfinite(annualization) or annualization != WEEKLY_PERIODS_PER_YEAR:
        raise ValueError("weekly annualization must be exactly 365.2425 / 7 (52.1775)")

    levels = valuations.levels_usd
    if not isinstance(levels, pd.DataFrame) or levels.empty or levels.shape[1] == 0:
        raise ValueError("levels_usd must be a non-empty asset matrix")
    if len(levels) < 3:
        raise ValueError(
            "levels_usd requires at least three consecutive Friday valuations "
            "to produce two returns"
        )
    if not levels.columns.is_unique:
        raise ValueError("levels_usd asset columns must be unique")
    if any(not isinstance(column, str) or not column.strip() for column in levels.columns):
        raise ValueError("levels_usd asset columns must be non-blank strings")

    index = levels.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("levels_usd index must be a DatetimeIndex")
    if index.tz is None or str(index.tz) != "UTC":
        raise ValueError("levels_usd valuation cutoffs must use the UTC timezone")
    if index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError(
            "levels_usd valuation cutoffs must be non-missing, unique, and strictly increasing"
        )
    utc_index = index.tz_convert("UTC")
    expected_cutoffs = utc_index.normalize() + pd.Timedelta(hours=22)
    if bool((utc_index != expected_cutoffs).any()) or bool((utc_index.dayofweek != 4).any()):
        raise ValueError("levels_usd valuation cutoffs must be Fridays at 22:00 UTC")
    if bool((utc_index[1:] - utc_index[:-1] != pd.Timedelta(days=7)).any()):
        raise ValueError("levels_usd requires consecutive Friday valuations seven days apart")
    if pd.Timestamp(valuations.start) != index[0] or pd.Timestamp(valuations.end) != index[-1]:
        raise ValueError("valuation start and end must match the common Friday level matrix")
    if valuations.base_currency != "USD":
        raise ValueError("Markowitz moments require base_currency='USD'")
    if not isinstance(valuations.snapshot_id, str) or not valuations.snapshot_id.strip():
        raise ValueError("snapshot_id must be a non-blank string")
    if valuations.valuation_rule != _VALUATION_RULE:
        raise ValueError(
            "valuation_rule must declare Friday 22:00 UTC as-of valuations with "
            "maximum staleness 3 calendar days"
        )
    if any(
        not pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_complex_dtype(dtype)
        for dtype in levels.dtypes
    ):
        raise ValueError("levels_usd must contain only real numeric asset columns")

    try:
        level_values = levels.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("levels_usd must contain a complete finite numeric matrix") from exc
    if not np.isfinite(level_values).all():
        raise ValueError("levels_usd must contain a complete finite matrix; filling is disabled")
    if bool((level_values <= 0.0).any()):
        raise ValueError("levels_usd values must be strictly positive")

    _validate_valuation_provenance(valuations, levels=levels)

    weekly_returns = levels.astype(float).pct_change(fill_method=None).iloc[1:]
    return_values = weekly_returns.to_numpy(dtype=float)
    if not np.isfinite(return_values).all():
        raise ValueError("weekly returns must be complete and finite; filling is disabled")

    expected_assets = levels.columns
    mean_ann = weekly_returns.mean(axis=0) * annualization
    if not isinstance(mean_ann, pd.Series) or not mean_ann.index.equals(expected_assets):
        raise ValueError(
            "annualized arithmetic mean must be a pandas Series with exact asset labels"
        )
    if (
        not pd.api.types.is_numeric_dtype(mean_ann.dtype)
        or pd.api.types.is_bool_dtype(mean_ann.dtype)
        or pd.api.types.is_complex_dtype(mean_ann.dtype)
    ):
        raise ValueError("annualized arithmetic means must contain only real numeric values")
    mean_ann.name = "mean_ann_arithmetic"

    covariance_ann = weekly_returns.cov() * annualization
    expected_shape = (len(expected_assets), len(expected_assets))
    if (
        not isinstance(covariance_ann, pd.DataFrame)
        or covariance_ann.shape != expected_shape
        or not covariance_ann.index.equals(expected_assets)
        or not covariance_ann.columns.equals(expected_assets)
    ):
        raise ValueError(
            "annualized covariance must be a square pandas DataFrame with exact asset labels "
            "on rows and columns"
        )
    if any(
        not pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_complex_dtype(dtype)
        for dtype in covariance_ann.dtypes
    ):
        raise ValueError("annualized covariance must contain only real numeric values")
    mean_values = mean_ann.to_numpy(dtype=float)
    if not np.isfinite(mean_values).all():
        raise ValueError("annualized arithmetic means must be finite")
    covariance_values = covariance_ann.to_numpy(dtype=float)
    if not np.isfinite(covariance_values).all():
        raise ValueError("annualized covariance must be finite")
    if not np.allclose(
        covariance_values,
        covariance_values.T,
        rtol=0.0,
        atol=MOMENT_PSD_TOLERANCE,
    ):
        raise ValueError(
            "annualized covariance must be symmetric within tolerance "
            f"{MOMENT_PSD_TOLERANCE:g}"
        )
    minimum_eigenvalue = float(np.linalg.eigvalsh(covariance_values).min())
    if minimum_eigenvalue < -MOMENT_PSD_TOLERANCE:
        raise ValueError(
            "annualized covariance must be positive semidefinite within tolerance "
            f"{MOMENT_PSD_TOLERANCE:g}; minimum eigenvalue={minimum_eigenvalue:.12g}"
        )

    return_dates = pd.DatetimeIndex(weekly_returns.index.copy(), name=levels.index.name)
    return AnnualizedMoments(
        weekly_returns=weekly_returns.copy(),
        mean_ann_arithmetic=mean_ann.copy(),
        covariance_ann=covariance_ann.copy(),
        return_dates=return_dates,
        n_obs=len(return_dates),
        snapshot_id=valuations.snapshot_id,
        base_currency="USD",
        valuation_rule=valuations.valuation_rule,
        periods_per_year=annualization,
        start=return_dates[0],
        end=return_dates[-1],
        psd_tolerance=MOMENT_PSD_TOLERANCE,
    )


def _frontier_inputs(moments: AnnualizedMoments) -> tuple[pd.Index, np.ndarray, np.ndarray]:
    """Validate frontier moments and return (assets, mean, covariance) arrays."""
    if not isinstance(moments, AnnualizedMoments):
        raise ValueError("moments must be an AnnualizedMoments instance")
    if moments.base_currency != "USD":
        raise ValueError("frontier requires base_currency='USD'")
    if moments.periods_per_year != WEEKLY_PERIODS_PER_YEAR:
        raise ValueError(
            "frontier requires weekly annualization of exactly 365.2425 / 7"
        )
    if not isinstance(moments.snapshot_id, str) or not moments.snapshot_id.strip():
        raise ValueError("frontier requires a non-blank snapshot_id")

    mean = moments.mean_ann_arithmetic
    covariance = moments.covariance_ann
    if not isinstance(mean, pd.Series) or mean.empty or not mean.index.is_unique:
        raise ValueError("mean_ann_arithmetic must be a non-empty uniquely labeled Series")
    assets = mean.index
    if (
        not isinstance(covariance, pd.DataFrame)
        or not covariance.index.equals(assets)
        or not covariance.columns.equals(assets)
    ):
        raise ValueError(
            "covariance_ann must carry exactly the mean asset labels on rows and columns"
        )
    try:
        mu = mean.to_numpy(dtype=float)
        sigma = covariance.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("frontier moments must be real numeric") from exc
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all():
        raise ValueError("frontier moments must be finite")
    if not np.allclose(sigma, sigma.T, rtol=0.0, atol=MOMENT_PSD_TOLERANCE):
        raise ValueError(
            f"covariance_ann must be symmetric within tolerance {MOMENT_PSD_TOLERANCE:g}"
        )
    minimum_eigenvalue = float(np.linalg.eigvalsh(sigma).min())
    if minimum_eigenvalue < -MOMENT_PSD_TOLERANCE:
        raise ValueError(
            "covariance_ann must be positive semidefinite within tolerance "
            f"{MOMENT_PSD_TOLERANCE:g}; minimum eigenvalue={minimum_eigenvalue:.12g}"
        )
    return assets, mu, sigma


def _point_diagnostics(
    weights: np.ndarray,
    *,
    target: float,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, float]:
    """Recompute every published diagnostic from one stored weight vector."""
    objective = float(weights @ sigma @ weights)
    return {
        "budget_residual": float(weights.sum() - 1.0),
        "target_residual": float(weights @ mu - target),
        "bound_violation": float(
            max(0.0, float((-weights).max()), float((weights - 1.0).max()))
        ),
        "objective": objective,
        "return_ann": float(weights @ mu),
        "volatility_ann": float(np.sqrt(max(objective, 0.0))),
    }


def validate_frontier_point(
    point: FrontierPoint,
    moments: AnnualizedMoments,
    *,
    residual_tolerance: float = FRONTIER_RESIDUAL_TOLERANCE,
) -> None:
    """Re-derive one publishable point from its stored weights and moments.

    Raises ValueError when any stored diagnostic disagrees with the stored
    weights or any residual exceeds ``residual_tolerance``.
    """
    if not isinstance(point, FrontierPoint):
        raise ValueError("point must be a FrontierPoint")
    assets, mu, sigma = _frontier_inputs(moments)
    weights = point.weights
    if not isinstance(weights, pd.Series) or not weights.index.equals(assets):
        raise ValueError(
            "stored weights must be a Series labeled by the moment asset universe"
        )
    weight_values = weights.to_numpy(dtype=float)
    if not np.isfinite(weight_values).all():
        raise ValueError("stored weights must be finite")
    recomputed = _point_diagnostics(
        weight_values, target=point.target_return_ann, mu=mu, sigma=sigma
    )
    for field_name, expected in recomputed.items():
        stored = float(getattr(point, field_name))
        if not np.isfinite(stored) or abs(stored - expected) > residual_tolerance:
            raise ValueError(
                f"frontier point {field_name} does not match its stored weights: "
                f"stored {stored!r}, recomputed {expected!r}"
            )
    for residual_name in ("budget_residual", "target_residual", "bound_violation"):
        if abs(recomputed[residual_name]) > residual_tolerance:
            raise ValueError(
                f"frontier point {residual_name} {recomputed[residual_name]!r} "
                f"exceeds residual tolerance {residual_tolerance:g}"
            )


def efficient_frontier(
    moments: AnnualizedMoments,
    *,
    n_points: int = 60,
) -> FrontierResult:
    """Solve deterministic fully-invested long-only minimum-variance targets.

    Targets span the attainable annualized-return range
    ``[min(mean), max(mean)]`` on an evenly spaced deterministic grid. Every
    target keeps its solver diagnostics and weight vector; failed or
    infeasible targets remain visible in ``points``. A point is publishable
    only when the solver reports success and the residuals recomputed from
    its stored weights are within ``FRONTIER_RESIDUAL_TOLERANCE``.
    """
    assets, mu, sigma = _frontier_inputs(moments)
    if isinstance(n_points, (bool, np.bool_)) or not isinstance(
        n_points, (int, np.integer)
    ):
        raise ValueError("n_points must be an integer of at least 2")
    if n_points < 2:
        raise ValueError("n_points must be an integer of at least 2")

    n_assets = len(assets)
    targets = np.linspace(float(mu.min()), float(mu.max()), int(n_points))
    initial_weights = np.full(n_assets, 1.0 / n_assets)
    bounds = [(0.0, 1.0)] * n_assets

    points: list[FrontierPoint] = []
    for target in targets:
        solved = optimize.minimize(
            lambda w: float(w @ sigma @ w),
            initial_weights,
            jac=lambda w: 2.0 * (sigma @ w),
            method="SLSQP",
            bounds=bounds,
            constraints=(
                {
                    "type": "eq",
                    "fun": lambda w: float(w.sum() - 1.0),
                    "jac": lambda w: np.ones_like(w),
                },
                {
                    "type": "eq",
                    "fun": lambda w, t=float(target): float(w @ mu - t),
                    "jac": lambda w: mu,
                },
            ),
            options={"maxiter": 200, "ftol": 1e-12},
        )
        weight_values = np.asarray(solved.x, dtype=float)
        diagnostics = _point_diagnostics(
            weight_values, target=float(target), mu=mu, sigma=sigma
        )
        feasible = (
            bool(solved.success)
            and bool(np.isfinite(weight_values).all())
            and np.isfinite(diagnostics["objective"])
            and abs(diagnostics["budget_residual"]) <= FRONTIER_RESIDUAL_TOLERANCE
            and abs(diagnostics["target_residual"]) <= FRONTIER_RESIDUAL_TOLERANCE
            and diagnostics["bound_violation"] <= FRONTIER_RESIDUAL_TOLERANCE
        )
        points.append(
            FrontierPoint(
                target_return_ann=float(target),
                success=bool(solved.success),
                status=int(solved.status),
                message=str(solved.message),
                iterations=int(solved.nit),
                objective=diagnostics["objective"],
                budget_residual=diagnostics["budget_residual"],
                target_residual=diagnostics["target_residual"],
                bound_violation=diagnostics["bound_violation"],
                weights=pd.Series(weight_values, index=assets.copy(), name="weight"),
                return_ann=diagnostics["return_ann"],
                volatility_ann=diagnostics["volatility_ann"],
                feasible=feasible,
            )
        )

    for point in points:
        if point.feasible:
            validate_frontier_point(point, moments)

    return FrontierResult(
        points=tuple(points),
        targets_ann=tuple(float(target) for target in targets),
        n_targets=int(n_points),
        n_feasible=sum(1 for point in points if point.feasible),
        residual_tolerance=FRONTIER_RESIDUAL_TOLERANCE,
        snapshot_id=moments.snapshot_id,
        base_currency="USD",
        valuation_rule=moments.valuation_rule,
        periods_per_year=moments.periods_per_year,
        start=moments.start,
        end=moments.end,
        n_obs=moments.n_obs,
    )


def _load_snapshot(snapshot_dir: Path) -> tuple[dict[str, object], dict[str, pd.DataFrame]]:
    snapshot_dir = Path(snapshot_dir)
    validate_market_snapshot(snapshot_dir)
    manifest_path = snapshot_dir / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    marker_hash = next(
        (
            line.removeprefix("manifest_sha256=")
            for line in (snapshot_dir / "COMPLETED").read_text().splitlines()
            if line.startswith("manifest_sha256=")
        ),
        None,
    )
    if marker_hash != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError(f"{snapshot_dir}: manifest sha256 does not match COMPLETED")
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{snapshot_dir}: manifest.json is not valid JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("completed") is not True:
        raise ValueError(f"{snapshot_dir}: manifest does not declare completed=true")
    return manifest, {
        name: pd.read_parquet(snapshot_dir / name) for name in _SNAPSHOT_FILES
    }


def _source_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{column}: source index must be a DatetimeIndex")
    index = frame.index
    if index.tz is not None or index.hasnans:
        raise ValueError(f"{column}: source dates must be timezone-naive and non-missing")
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError(f"{column}: source dates must be unique and strictly increasing")
    if bool((index != index.normalize()).any()):
        raise ValueError(f"{column}: source dates must be date-granular")

    series = frame[column].dropna()
    if series.empty:
        raise ValueError(f"{column}: source contains no observations")
    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all() or bool((values <= 0).any()):
        raise ValueError(f"{column}: source observations must be finite and positive")
    return series.astype(float)


def _as_of(series: pd.Series, cutoff_date: pd.Timestamp, name: str) -> tuple[float, pd.Timestamp]:
    position = int(series.index.searchsorted(cutoff_date, side="right")) - 1
    if position < 0:
        raise ValueError(
            f"{name} has no eligible observation at or before cutoff {cutoff_date.date()}"
        )
    observed_date = pd.Timestamp(series.index[position])
    if observed_date > cutoff_date:
        raise ValueError(
            f"{name} look-ahead observation {observed_date.date()} exceeds cutoff "
            f"{cutoff_date.date()}"
        )
    age_days = int((cutoff_date - observed_date).days)
    if age_days > _MAX_STALENESS_DAYS:
        raise ValueError(
            f"{name} observation for cutoff {cutoff_date.date()} is {age_days} calendar "
            f"days stale (maximum {_MAX_STALENESS_DAYS})"
        )
    return float(series.iloc[position]), observed_date


def _requested_date(value: str | pd.Timestamp, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid date") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{name} must not be NaT")
    if timestamp.tz is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.normalize()


def weekly_usd_valuations(
    snapshot_dir: Path,
    *,
    quote_specs: Mapping[str, QuoteSpec],
    requested_start: str | pd.Timestamp,
    requested_end: str | pd.Timestamp,
) -> WeeklyValuations:
    """Build complete Friday 22:00 UTC as-of levels in US dollars.

    Source observations remain on their actual snapshot dates. An observation is
    eligible only when it is no later than the Friday cutoff and no more than
    three calendar days old.
    """
    if not isinstance(quote_specs, Mapping) or not quote_specs:
        raise ValueError("quote_specs must be a non-empty mapping")
    quote_items = tuple(quote_specs.items())
    for symbol, spec in quote_items:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError(f"quote_specs contains an invalid symbol {symbol!r}")
        if not isinstance(spec, QuoteSpec):
            raise ValueError(f"{symbol}: quote specification must be a QuoteSpec")

    start_date = _requested_date(requested_start, "requested_start")
    end_date = _requested_date(requested_end, "requested_end")
    if start_date > end_date:
        raise ValueError("requested_start must be on or before requested_end")
    friday_dates = pd.date_range(start_date, end_date, freq="W-FRI")
    if friday_dates.empty:
        raise ValueError(
            f"requested window {start_date.date()}..{end_date.date()} contains no Friday"
        )
    cutoffs = pd.DatetimeIndex(friday_dates.tz_localize("UTC") + pd.Timedelta(hours=22))
    cutoffs.name = "valuation_cutoff"

    manifest, tables = _load_snapshot(Path(snapshot_dir))
    manifest_quotes = manifest.get("quotes")
    if not isinstance(manifest_quotes, dict):
        raise ValueError(f"{snapshot_dir}: explicit quote metadata is absent from manifest")

    available: dict[str, pd.Series] = {}
    for frame in tables.values():
        if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
            raise ValueError(f"{snapshot_dir}: snapshot tables require unique columns")
        for column in frame.columns:
            if column in available:
                raise ValueError(f"{snapshot_dir}: duplicate source column {column!r}")
            available[str(column)] = _source_series(frame, str(column))

    for symbol, spec in quote_items:
        if symbol not in available:
            raise ValueError(f"{symbol}: source column is absent from the snapshot")
        recorded = manifest_quotes.get(symbol)
        expected = {
            "quote_currency": spec.quote_currency,
            "quote_unit": spec.quote_unit,
            "scale_to_major": spec.scale_to_major,
        }
        if recorded != expected:
            raise ValueError(
                f"{symbol} quote specification does not match snapshot metadata: "
                f"expected {expected!r}, found {recorded!r}"
            )

    needs_fx = any(spec.quote_currency == "GBP" for _, spec in quote_items)
    fx_series = available.get("USD_per_GBP")
    if needs_fx and fx_series is None:
        raise ValueError("USD_per_GBP source column is absent from the snapshot")

    levels_rows: list[dict[str, float]] = []
    observed_rows: list[dict[str, pd.Timestamp]] = []
    fx_dates: list[pd.Timestamp] = []
    for cutoff in cutoffs:
        cutoff_date = cutoff.tz_localize(None).normalize()
        fx_value: float | None = None
        fx_date = pd.NaT
        if needs_fx:
            assert fx_series is not None
            fx_value, fx_date = _as_of(fx_series, cutoff_date, "USD_per_GBP")

        level_row: dict[str, float] = {}
        date_row: dict[str, pd.Timestamp] = {}
        for symbol, spec in quote_items:
            level, observed_date = _as_of(available[symbol], cutoff_date, symbol)
            if spec.quote_currency == "GBP":
                assert fx_value is not None
                level = level * spec.scale_to_major * fx_value
            level_row[symbol] = level
            date_row[symbol] = observed_date
        levels_rows.append(level_row)
        observed_rows.append(date_row)
        fx_dates.append(fx_date)

    levels_usd = pd.DataFrame(levels_rows, index=cutoffs, columns=list(quote_specs))
    observed_dates = pd.DataFrame(observed_rows, index=cutoffs, columns=list(quote_specs))
    fx_observed_dates = pd.Series(
        fx_dates, index=cutoffs, name="USD_per_GBP_observed_date", dtype="datetime64[ns]"
    )
    if levels_usd.isna().any().any() or not np.isfinite(levels_usd.to_numpy()).all():
        raise ValueError("requested window does not produce a complete finite common USD matrix")

    snapshot_id = str(manifest["snapshot_id"])
    fx_required_assets = tuple(
        symbol for symbol, spec in quote_items if spec.quote_currency == "GBP"
    )
    return WeeklyValuations(
        levels_usd=levels_usd,
        observed_dates=observed_dates,
        fx_observed_dates=fx_observed_dates,
        asset_quote_specs=quote_items,
        producer_provenance_sha256=_quote_fx_provenance_sha256(
            snapshot_id=snapshot_id,
            base_currency="USD",
            valuation_rule=_VALUATION_RULE,
            asset_quote_specs=quote_items,
            fx_required_assets=fx_required_assets,
        ),
        base_currency="USD",
        valuation_rule=_VALUATION_RULE,
        start=cutoffs[0],
        end=cutoffs[-1],
        snapshot_id=snapshot_id,
        fx_required_assets=fx_required_assets,
    )
