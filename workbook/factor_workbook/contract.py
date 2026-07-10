"""Schema-contract registry (ASSET_SPECS) and typed loaders with fail-fast
asset+column-specific validation (R1.4, R7.2 base).

Every consumed release asset — the ``data-v1`` set plus the ``data-v2``
static buy-and-hold additions — has one :class:`AssetSpec` recording the
captured schema (columns, dtypes, index, minimum rows). The loaders
:func:`load_frame` / :func:`load_json` fetch through a
:class:`~factor_workbook.release.ReleaseClient`, validate against the spec,
and return the validated table plus its provenance. Any mismatch raises
:class:`SchemaError` naming the asset, the offending column, and the expected
dtype — the foundation of the discrepancy detector.

Dtype comparison is deliberately tolerant on datetime resolution: the release
parquets mix ``datetime64[ms]`` and ``datetime64[ns]``, so datetime columns
are compared unit-insensitively. String columns are compared as ``str``
whether pandas surfaces them as ``str`` or all-``None`` ``object`` columns
(``fail_reason`` / ``dropped_reason`` are legitimately all-null), and an
entirely-null column satisfies any contract dtype: parquet writers persist
all-null columns as ``object`` or ``float64`` depending on origin
(``raw_ref_delta`` differs across the published evidence members).

JSON specs use dotted key paths as "columns" (e.g. ``meta.nim_model``) with
JSON type names as "dtypes"; the decision logs share a common per-date shape
while their ``meta`` blocks differ per variant (v1 carries cutoff/holdout
fields, v2 the prompt version, nonpit the ``variant`` marker).
"""

import io
import json
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from factor_workbook.release import Provenance, ReleaseClient

_ETF_WEIGHTS = {"SWDA.L": "float64", "XLK": "float64", "IAU": "float64", "BIL": "float64"}
_AXES = {
    "inflation": "float64",
    "growth": "float64",
    "credit_stress": "float64",
    "policy": "float64",
    "risk_appetite": "float64",
}
_DECISION_LOG_COMMON = {
    "meta": "dict",
    "meta.nim_model": "str",
    "meta.n_rebalances": "int",
    "meta.n_recall_guarded": "int",
    "meta.line": "str",
    "p_memorized": "dict",
    "parse_ok": "dict",
    "steered": "dict",
    "conviction": "dict",
    "loadings": "dict",
    "views": "dict",
}
_STABILITY_KEYS = {
    f"{axis}_{measure}": "float"
    for axis in ("credit_stress", "growth", "inflation", "mean", "policy", "risk_appetite")
    for measure in ("mac", "std")
}
_METRIC_KEYS = {
    "controlled_auc": "float",
    "controlled_ci_low": "float",
    "controlled_ci_high": "float",
    "controlled_perm_p": "float",
    "positive_control_auc": "float",
    "positive_control_perm_p": "float",
    "parse_rate": "float",
    "n_per_class": "int",
    "verdict": "str",
}


def _static_window_keys(window: str) -> dict[str, str]:
    """Contract key paths of one published static-B&H window block (task 7.1)."""
    keys: dict[str, str] = {
        window: "dict",
        f"{window}.window": "list",
        f"{window}.weights_at_inception": "dict",
        f"{window}.static_bh": "dict",
        f"{window}.static_bh_ssr": "dict",
        f"{window}.spy_bh": "dict",
        f"{window}.crisis_episodes": "dict",
    }
    keys |= {
        f"{window}.static_bh.{metric}": "float"
        for metric in (
            "total_return",
            "annualized_return",
            "annualized_vol",
            "sharpe",
            "sortino",
            "calmar",
            "max_drawdown",
        )
    }
    keys |= {
        f"{window}.static_bh_ssr.ssr": "float",
        f"{window}.static_bh_ssr.mean_rolling_sr": "float",
        f"{window}.static_bh_ssr.sigma_hac": "float",
        f"{window}.static_bh_ssr.L_hac": "int",
        f"{window}.static_bh_ssr.n_rolling": "int",
    }
    keys |= {
        f"{window}.spy_bh.{metric}": "float"
        for metric in ("total_return", "annualized_return", "sharpe", "max_drawdown")
    }
    keys |= {
        f"{window}.crisis_episodes.{episode}": "dict"
        for episode in ("covid_2020", "inflation_2022")
    }
    return keys


class SchemaError(Exception):
    """A release asset violated its captured schema contract (R1.4)."""


@dataclass(frozen=True)
class AssetSpec:
    """Captured schema contract for one consumed release asset.

    Attributes:
        asset: Release asset name (differs from repo paths for some assets).
        kind: Physical format; ``tar_*`` kinds live inside a ``.tar.gz``.
        index: Expected index name (frames only; None for a RangeIndex).
        columns: Frames: column -> dtype. JSON: dotted key path -> JSON type
            name in {str, int, float, bool, dict, list}.
        min_rows: Minimum rows enforced on load — deliberately conservative
            so schema-true fixture subsets validate too.
        expected_rows: Full production row count of the ``data-v1`` asset;
            metadata only, never enforced on load.
        member: Tar member path template (``{model}`` = model slug) for
            ``tar_*`` kinds.
        row_container: JSON only — dotted path of the container whose length
            is checked against ``min_rows`` (e.g. the per-date dict).
    """

    asset: str
    kind: Literal["parquet", "json", "tar_parquet", "tar_json"]
    index: str | None
    columns: dict[str, str]
    min_rows: int
    expected_rows: int = 0
    member: str | None = None
    row_container: str | None = None


def _monthly(asset: str, columns: dict[str, str]) -> AssetSpec:
    return AssetSpec(asset, "parquet", "date", columns, min_rows=3, expected_rows=72)


def _daily(asset: str, columns: dict[str, str], expected_rows: int) -> AssetSpec:
    return AssetSpec(asset, "parquet", "Date", columns, min_rows=3, expected_rows=expected_rows)


def _decision_log(asset: str, meta_extra: dict[str, str]) -> AssetSpec:
    return AssetSpec(
        asset,
        "json",
        None,
        {**_DECISION_LOG_COMMON, **meta_extra},
        min_rows=3,
        expected_rows=72,
        row_container="p_memorized",
    )


ASSET_SPECS: dict[str, AssetSpec] = {
    # -- monthly factor streams (72 rows, date-indexed, datetime64[ms]) -----
    "factor_loadings_v1": _monthly(
        "factor_loadings_v1.parquet", {"parse_ok": "bool", **_AXES}
    ),
    "factor_loadings_v2": _monthly(
        "factor_loadings_v2.parquet", {"parse_ok": "bool", "prompt_version": "str", **_AXES}
    ),
    "factor_scores_v1": _monthly(
        "factor_scores_v1.parquet", {"p_memorized": "float64", "fail_reason": "str"}
    ),
    "factor_scores_v2": _monthly(
        "factor_scores_v2.parquet",
        {"p_memorized": "float64", "fail_reason": "str", "prompt_version": "str"},
    ),
    "factor_contrast_v1": _monthly(
        "factor_contrast_v1.parquet",
        {"pit_p": "float64", "nonpit_p": "float64", "delta": "float64"},
    ),
    "factor_nonpit_diagnostic_loadings_v1": _monthly(
        "factor_nonpit_diagnostic_loadings_v1.parquet",
        {"parse_ok": "bool", "variant": "str", **_AXES},
    ),
    "factor_nonpit_diagnostic_scores_v1": _monthly(
        "factor_nonpit_diagnostic_scores_v1.parquet",
        {"p_memorized": "float64", "fail_reason": "str", "variant": "str"},
    ),
    "macro_panel_monthly": AssetSpec(
        "macro_panel_monthly.parquet",
        "parquet",
        "date",
        {
            "cpi_yoy": "float64",
            "t10y2y": "float64",
            "hy_oas": "float64",
            "cpi_yoy_z": "float64",
            "t10y2y_z": "float64",
            "hy_oas_z": "float64",
        },
        min_rows=3,
        expected_rows=196,
    ),
    # -- per-view / per-call tables (RangeIndex) -----------------------------
    "factor_views_v1": AssetSpec(
        "factor_views_v1.parquet",
        "parquet",
        None,
        {
            "date": "datetime64",
            "asset": "str",
            "raw_tilt": "float64",
            "p_memorized": "float64",
            "guarded_tilt": "float64",
            "conviction": "float64",
        },
        min_rows=3,
        expected_rows=284,
    ),
    "naive_directional_eval": AssetSpec(
        "naive_directional_eval_openai_gpt-oss-20b.parquet",
        "parquet",
        None,
        {
            "date": "datetime64",
            "prompt": "str",
            "reply": "str",
            "predicted_direction": "int64",
            "confidence": "float64",
            "realized_direction": "int64",
            "correct": "bool",
        },
        min_rows=3,
        expected_rows=72,
    ),
    # -- daily simulation tables ---------------------------------------------
    "factor_targets_v1": _daily("factor_targets_v1.parquet", dict(_ETF_WEIGHTS), 2828),
    "factor_targets_v2": _daily("factor_targets_v2.parquet", dict(_ETF_WEIGHTS), 2828),
    "factor_nonpit_diagnostic_targets_v1": _daily(
        "factor_nonpit_diagnostic_targets_v1.parquet", dict(_ETF_WEIGHTS), 2828
    ),
    "factor_equity_v1": _daily("factor_equity_v1.parquet", {"value": "float64"}, 2717),
    "factor_equity_v2": _daily("factor_equity_v2.parquet", {"value": "float64"}, 2717),
    "factor_nonpit_diagnostic_equity_v1": _daily(
        "factor_nonpit_diagnostic_equity_v1.parquet", {"value": "float64"}, 2717
    ),
    # -- luck vs skill --------------------------------------------------------
    "factor_luck_vs_skill_v1": AssetSpec(
        "factor_luck_vs_skill_v1.parquet",
        "parquet",
        "line",
        {
            "n_obs": "int64",
            "n_rolling": "int64",
            "total_return": "float64",
            "sharpe": "float64",
            "mean_rolling_sr": "float64",
            "ssr": "float64",
            "nw_long_run_var": "float64",
            "nw_sigma_hac": "float64",
            "nw_bandwidth_L": "int64",
            "verdict": "str",
        },
        min_rows=3,
        expected_rows=3,
    ),
    # -- JSON summaries --------------------------------------------------------
    "factor_stability_v1": AssetSpec(
        "factor_stability_v1.json", "json", None, dict(_STABILITY_KEYS), min_rows=0
    ),
    "factor_stability_v2": AssetSpec(
        "factor_stability_v2.json", "json", None, dict(_STABILITY_KEYS), min_rows=0
    ),
    "factor_contrast_summary_v1": AssetSpec(
        "factor_contrast_summary_v1.json",
        "json",
        None,
        {
            "contamination_premium": "dict",
            "contamination_premium.p_memorized_paired_d": "float",
            "contamination_premium.sharpe_delta": "float",
            "framing": "str",
            "n_pairs": "int",
            "nim_model": "str",
            "pit_metrics": "dict",
            "nonpit_metrics": "dict",
            "pit_p_memorized": "dict",
            "nonpit_p_memorized": "dict",
        },
        min_rows=0,
    ),
    "prompt_version_gate_v1": AssetSpec(
        "prompt_version_gate_v1.json",
        "json",
        None,
        {
            "adopted_version": "str",
            "checks": "dict",
            "decision": "str",
            "head_to_head_deltas_v2_minus_v1": "dict",
            "n_rebalances": "int",
            "nim_model": "str",
            "parse_rates": "dict",
            "prior_versions_preserved": "str",
            "prompt_v2_suffix": "str",
        },
        min_rows=0,
    ),
    # -- decision logs: common per-date shape, per-variant metas ---------------
    "factor_decision_log_v1": _decision_log(
        "factor_decision_log_v1.json",
        {"meta.cutoff_date": "str", "meta.holdout_auc": "float", "meta.is_weak": "bool"},
    ),
    "factor_decision_log_v2": _decision_log(
        "factor_decision_log_v2.json",
        {"meta.prompt_version": "str", "meta.prompt_v2_suffix": "str"},
    ),
    "factor_nonpit_diagnostic_decision_log_v1": _decision_log(
        "factor_nonpit_diagnostic_decision_log_v1.json", {"meta.variant": "str"}
    ),
    # -- no-recall screen (release asset names differ from repo paths) ---------
    "norecall_screen_results": AssetSpec(
        "norecall_screen_results.json",
        "json",
        None,
        {
            "screen": "str",
            "cutoff_date": "str",
            "n_per_class": "int",
            "parse_sample": "int",
            "candidates": "list",
            "built_at": "str",
            "results": "list",
        },
        min_rows=3,
        expected_rows=5,
        row_container="results",
    ),
    "norecall_screen_evidence": AssetSpec(
        "norecall_screen_evidence.tar.gz",
        "tar_parquet",
        None,
        {
            "arm": "str",
            "row_index": "int64",
            "as_of": "str",
            "prompt": "str",
            "reply": "str",
            "n_tokens": "float64",
            "included": "bool",
            "dropped_reason": "str",
            "raw_loss": "float64",
            "raw_min_k": "float64",
            "raw_min_k_pp": "float64",
            "raw_zlib_ratio": "float64",
            "raw_ref_delta": "str",
            "std_loss": "float64",
            "std_min_k": "float64",
            "std_min_k_pp": "float64",
            "std_zlib_ratio": "float64",
        },
        min_rows=3,
        expected_rows=521,
        member="evidence/{model}/evidence.parquet",
    ),
    "norecall_screen_evidence_baseline": AssetSpec(
        "norecall_screen_evidence.tar.gz",
        "tar_json",
        None,
        {
            "model": "str",
            "n_valid": "int",
            "min_valid": "int",
            "is_calibrated": "bool",
            "feature_means": "dict",
            "feature_stds": "dict",
        },
        min_rows=0,
        member="evidence/{model}/baseline.json",
    ),
    "norecall_screen_evidence_summary": AssetSpec(
        "norecall_screen_evidence.tar.gz",
        "tar_json",
        None,
        {"model": "str", "cutoff_date": "str", **_METRIC_KEYS},
        min_rows=0,
        member="evidence/{model}/summary.json",
    ),
    # -- data-v2 static buy-and-hold line (task 7.1; absent on data-v1) --------
    "static_bh_equity_2014_2024": _daily(
        "static_bh_equity_2014_2024.parquet", {"value": "float64"}, 2717
    ),
    "static_bh_equity_2016_2026": _daily(
        "static_bh_equity_2016_2026.parquet", {"value": "float64"}, 2469
    ),
    "static_bh_targets_2014_2024": _daily(
        "static_bh_targets_2014_2024.parquet", dict(_ETF_WEIGHTS), 2717
    ),
    "static_bh_stats": AssetSpec(
        "static_bh_stats.json",
        "json",
        None,
        {
            **_static_window_keys("2014_2024"),
            **_static_window_keys("2016_2026"),
            # published as null for 2016_2026 — contracted for 2014_2024 only
            "2014_2024.weight_drift_final": "dict",
            "caveat": "str",
            "source": "str",
            "built_at": "str",
        },
        min_rows=0,
    ),
}


_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "str": str,
    "int": int,
    "float": (int, float),  # JSON serializes 1.0 as 1
    "bool": bool,
    "dict": dict,
    "list": list,
}


def _normalize_dtype(dtype: Any) -> str:
    """Collapse a pandas dtype to the contract's tolerant dtype vocabulary."""
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime64"  # unit-insensitive: [ms] and [ns] both occur
    if pd.api.types.is_bool_dtype(dtype):
        return "bool"
    if pd.api.types.is_integer_dtype(dtype):
        return "int64"
    if pd.api.types.is_float_dtype(dtype):
        return "float64"
    # str dtype and all-null object columns both count as str
    return "str" if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype) else str(dtype)


def _spec_and_bytes(
    client: ReleaseClient, key: str, model: str | None
) -> tuple[AssetSpec, bytes, Provenance]:
    spec = ASSET_SPECS[key]
    if spec.member is None:
        data, provenance = client.fetch(spec.asset)
        return spec, data, provenance
    if "{model}" in spec.member and model is None:
        raise ValueError(f"{key}: a model slug is required for {spec.member!r}")
    member = spec.member.format(model=model)
    data, provenance = client.fetch_tar_member(spec.asset, member)
    return spec, data, provenance


def _validate_frame(spec: AssetSpec, df: pd.DataFrame) -> None:
    if df.index.name != spec.index:
        raise SchemaError(
            f"asset {spec.asset}: expected index {spec.index!r}, got {df.index.name!r}"
        )
    for column, expected in spec.columns.items():
        if column not in df.columns:
            raise SchemaError(
                f"asset {spec.asset}: missing column {column!r} (expected dtype {expected})"
            )
        actual = _normalize_dtype(df[column].dtype)
        # An entirely-null column carries no dtype signal: parquet writers
        # persist it as object OR float64 depending on origin (raw_ref_delta
        # is all-null str-typed in 20b/phi-4 but all-null float64 in 120b),
        # so all-null satisfies the contract dtype.
        if actual != expected and not df[column].isna().all():
            raise SchemaError(
                f"asset {spec.asset}: column {column!r}: expected dtype {expected}, "
                f"got {df[column].dtype}"
            )
    for column in df.columns:
        if column not in spec.columns:
            raise SchemaError(f"asset {spec.asset}: unexpected column {column!r} not in contract")
    if len(df) < spec.min_rows:
        raise SchemaError(
            f"asset {spec.asset}: expected at least {spec.min_rows} rows, got {len(df)}"
        )


def _dig(obj: dict, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(obj, dict) or part not in obj:
            raise KeyError(path)
        obj = obj[part]
    return obj


def _validate_json(spec: AssetSpec, obj: Any) -> None:
    if not isinstance(obj, dict):
        raise SchemaError(f"asset {spec.asset}: expected a JSON object, got {type(obj).__name__}")
    for path, expected in spec.columns.items():
        try:
            value = _dig(obj, path)
        except KeyError:
            raise SchemaError(
                f"asset {spec.asset}: missing column {path!r} (expected dtype {expected})"
            ) from None
        expected_type = _JSON_TYPES[expected]
        if not isinstance(value, expected_type) or (
            expected != "bool" and isinstance(value, bool)
        ):
            raise SchemaError(
                f"asset {spec.asset}: column {path!r}: expected dtype {expected}, "
                f"got {type(value).__name__}"
            )
    if spec.row_container is not None and len(_dig(obj, spec.row_container)) < spec.min_rows:
        raise SchemaError(
            f"asset {spec.asset}: expected at least {spec.min_rows} rows in "
            f"{spec.row_container!r}, got {len(_dig(obj, spec.row_container))}"
        )


def load_frame(
    client: ReleaseClient, key: str, *, model: str | None = None
) -> tuple[pd.DataFrame, Provenance]:
    """Load and validate a tabular release asset by its registry key.

    Args:
        client: Release client to fetch through (owns tag + provenance).
        key: Logical registry key, e.g. ``"factor_loadings_v1"``.
        model: Model slug for tar members parameterized by model, e.g.
            ``"openai_gpt-oss-20b"`` for ``norecall_screen_evidence``.

    Returns:
        The validated DataFrame and the retrieval provenance.

    Raises:
        KeyError: Unknown registry key.
        ValueError: The key is not a tabular asset, or a required model
            slug is missing.
        SchemaError: Any contract mismatch — names asset, column, dtype.
    """
    spec = ASSET_SPECS[key]
    if spec.kind not in ("parquet", "tar_parquet"):
        raise ValueError(f"{key}: kind {spec.kind!r} is not tabular; use load_json")
    spec, data, provenance = _spec_and_bytes(client, key, model)
    df = pd.read_parquet(io.BytesIO(data))
    _validate_frame(spec, df)
    return df, provenance


def load_json(
    client: ReleaseClient, key: str, *, model: str | None = None
) -> tuple[dict, Provenance]:
    """Load and validate a JSON release asset by its registry key.

    Args:
        client: Release client to fetch through (owns tag + provenance).
        key: Logical registry key, e.g. ``"factor_decision_log_v1"``.
        model: Model slug for tar members parameterized by model.

    Returns:
        The validated JSON object and the retrieval provenance.

    Raises:
        KeyError: Unknown registry key.
        ValueError: The key is not a JSON asset, or a required model slug
            is missing.
        SchemaError: Any contract mismatch — names asset, key path, type.
    """
    spec = ASSET_SPECS[key]
    if spec.kind not in ("json", "tar_json"):
        raise ValueError(f"{key}: kind {spec.kind!r} is not JSON; use load_frame")
    spec, data, provenance = _spec_and_bytes(client, key, model)
    obj = json.loads(data)
    _validate_json(spec, obj)
    return obj, provenance
