"""Schema-contract registry (ASSET_SPECS) and typed loaders with fail-fast
asset+column-specific validation (R1.4, R7.2 base).

Every consumed release asset (the ``data-v1`` set plus the ``data-v2``
static buy-and-hold additions) has one :class:`AssetSpec` recording the
captured schema (columns, dtypes, index, minimum rows). The loaders
:func:`load_frame` / :func:`load_json` fetch through a
:class:`~factor_workbook.release.ReleaseClient`, validate against the spec,
and return the validated table plus its provenance. Any mismatch raises
:class:`SchemaError` naming the asset, the offending column, and the expected
dtype. This is the foundation of the discrepancy detector.

Dtype comparison is deliberately tolerant on datetime resolution: the release
parquets mix ``datetime64[ms]`` and ``datetime64[ns]``, so datetime columns
are compared unit-insensitively. String columns are compared as ``str``
whether pandas surfaces them as ``str`` or all-``None`` ``object`` columns
(``fail_reason`` / ``dropped_reason`` are legitimately all-null), and an
entirely-null column satisfies any contract dtype: parquet writers persist
all-null columns as ``object`` or ``float64`` depending on origin
(``raw_ref_delta`` differs across the published evidence members).

Columns listed in ``AssetSpec.optional`` are the one exception to fail-fast on
absence: a release tag is immutable, so a column added after a tag was cut can
never appear in it. Those columns validate exactly like any other when present
and are simply absent otherwise, which is what keeps the shipped default
``data-v2`` (HAC-only luck-vs-skill table, no MBB columns) loadable.

The corrected canonical ``data-v4`` tables have their own tag-bound registry
(``DATA_V4_ASSET_SPECS`` + ``load_v4_frame``/``load_v4_json``, task 10.3) at
the bottom of this module; the historical registry above stays byte-unchanged
and refuses a ``data-v4``-tagged client, so cross-tag substitution fails in
both directions.

JSON specs use dotted key paths as "columns" (e.g. ``meta.nim_model``) with
JSON type names as "dtypes"; the decision logs share a common per-date shape
while their ``meta`` blocks differ per variant (v1 carries cutoff/holdout
fields, v2 the prompt version, nonpit the ``variant`` marker).
"""

import dataclasses
import io
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping

import pandas as pd

from factor_workbook import vendored_ssr
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
        optional: Subset of ``columns`` that later releases added: absence is
            tolerated (an older, immutable release predates them), presence is
            validated exactly like any other contracted column.
    """

    asset: str
    kind: Literal["parquet", "json", "tar_parquet", "tar_json"]
    index: str | None
    columns: dict[str, str]
    min_rows: int
    expected_rows: int = 0
    member: str | None = None
    row_container: str | None = None
    optional: frozenset[str] = frozenset()


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
            "mbb_p": "float64",
            "mbb_block": "int64",
            "verdict": "str",
        },
        min_rows=3,
        expected_rows=3,
        # the MBB inference columns post-date the immutable data-v2 release,
        # which ships the HAC-only table — tolerated absent, validated present
        optional=frozenset({"mbb_p", "mbb_block"}),
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


def _dtype_matches(actual: str, expected: str) -> bool:
    """``number64`` (data-v4 mixed-row report tables only) accepts int64 OR
    float64: an integer field carried by only one row family of a mixed table
    is NaN-diluted to float64 by pandas, while a single-family table keeps
    int64. Historical contracts never use the token, so their comparisons are
    byte-identical to the pre-data-v4 behavior."""
    if expected == "number64":
        return actual in ("int64", "float64")
    return actual == expected


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
    if getattr(client, "tag", None) == DATA_V4_TAG:
        raise SchemaError(
            f"registry key {key!r} belongs to the historical "
            f"{'/'.join(HISTORICAL_TAGS)} contracts; the client is bound to tag "
            f"{DATA_V4_TAG!r} — cross-tag substitution is prohibited (use the "
            "data-v4 registry via load_v4_frame/load_v4_json)"
        )
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
            if column in spec.optional:
                continue  # added by a later release; older tags predate it
            raise SchemaError(
                f"asset {spec.asset}: missing column {column!r} (expected dtype {expected})"
            )
        actual = _normalize_dtype(df[column].dtype)
        # An entirely-null column carries no dtype signal: parquet writers
        # persist it as object OR float64 depending on origin (raw_ref_delta
        # is all-null str-typed in 20b/phi-4 but all-null float64 in 120b),
        # so all-null satisfies the contract dtype.
        if not _dtype_matches(actual, expected) and not df[column].isna().all():
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


# --------------------------------------------------------------------------- #
# data-v4: tag-aware canonical schemas (task 10.3)                             #
#                                                                              #
# The corrected canonical tables live under the immutable ``data-v4`` release  #
# tag with their own registry. Every shape below MIRRORS the frozen producer   #
# contracts — scripts/build_tear_sheet.py table schemas, the reporting row     #
# vocabularies of macro_framework.reporting, and the frozen data-v4 asset      #
# catalog of scripts/publish_finance_remediation.py (task 10.1) — never an     #
# invented shape. The workbook project is dependency-isolated, so the          #
# vocabulary is mirrored as data; workbook/tests/test_parity_root_env.py       #
# proves the mirror equal to the root authorities in the root environment.     #
#                                                                              #
# Tag binding: ``load_v4_frame``/``load_v4_json`` accept ONLY a client bound   #
# to ``data-v4``, and the historical loaders refuse a data-v4 client, so       #
# cross-tag substitution fails in both directions while the data-v1..v3        #
# contracts above remain byte-unchanged.                                       #
# --------------------------------------------------------------------------- #

DATA_V4_TAG = "data-v4"
HISTORICAL_TAGS = ("data-v1", "data-v2", "data-v3")

# Row-schema identities (macro_framework.reporting) and producer table schemas
# (scripts/build_tear_sheet.py / the frozen catalog), mirrored verbatim.
READER_SCHEMA = "portfolio_metrics.reader.v2"
LEGACY_SCHEMA = "portfolio_metrics.vectorbt365.v1"
DIFFERENTIAL_SCHEMA = "portfolio_metrics.differential.v2"
ATTRIBUTION_SCHEMA = "attribution.raw_market_model.v1"
CRISIS_SCHEMA = "crisis_metrics.boundary_anchored.v1"
MONTHLY_SCHEMA = "monthly_returns.reader.v1"
RISK_DECOMPOSITION_SCHEMA = "risk_decomposition.v1"
AI_VARIANTS_TEAR_SHEET_SCHEMA = "tear_sheet.ai_variants.v1"
SJM_TEAR_SHEET_SCHEMA = "tear_sheet.sjm.v3"
TRIO_TEAR_SHEET_SCHEMA = "tear_sheet.trio.v4"
MARKOWITZ_MOMENTS_SCHEMA = "markowitz.moments.v1"
MARKOWITZ_FRONTIER_SCHEMA = "markowitz.frontier.v1"
PUBLICATION_MANIFEST_SCHEMA = "publication_manifest.v1"

#: Deterministic SSR inference settings recorded on every produced table
#: (mirror of the report producer's SSR_REPORT_DEFAULTS, R7.4).
SSR_REPORT_DEFAULTS: Mapping[str, object] = MappingProxyType(
    {"window": 252, "sr_star": 0.0, "n_boot": 1000, "seed": 0, "alpha": 0.05}
)

#: The two disclosed currency bases of canonical report rows.
V4_CURRENCY_BASES = ("USD", "legacy_mixed_local_quotes")

#: The USD Markowitz opportunity set and its exact weekly annualization.
MARKOWITZ_ASSET_UNIVERSE = ("SWDA.L", "XLK", "IAU", "BIL")
MARKOWITZ_WEEKLY_PERIODS_PER_YEAR = 365.2425 / 7

#: Locale projections of every canonical table derive FROM these specs
#: (mirror of the report producer's REPORT_CSV_LOCALE_SPECS).
REPORT_CSV_LOCALE_SPECS: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "en-US": MappingProxyType(
            {"sep": ",", "decimal": ".", "float_format": "%.8f", "encoding": "utf-8"}
        ),
        "de-DE": MappingProxyType(
            {"sep": ";", "decimal": ",", "float_format": "%.8f", "encoding": "utf-8"}
        ),
    }
)

#: Public alias -> canonical target basenames frozen by the data-v4 catalog.
DATA_V4_COMPATIBILITY_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "nb16_ai_variants_tearsheet.csv": "tear_sheet_ai_variants_ext2026.csv",
        "nb16_ai_variants_tearsheet_de.csv": "tear_sheet_ai_variants_ext2026_de.csv",
        "nb17_sjm_crowding_tearsheet.csv": "tear_sheet_sjm_crowding_ext2026.csv",
        "nb17_sjm_crowding_tearsheet_de.csv": "tear_sheet_sjm_crowding_ext2026_de.csv",
        "sjm_crowding_derisk_equity_ext2026.parquet": (
            "sjm_crowding_v3_total_return_bil_equity_ext2026.parquet"
        ),
        "sjm_crowding_derisk_equity_ext2026.csv": (
            "sjm_crowding_v3_total_return_bil_equity_ext2026.csv"
        ),
        "sjm_crowding_derisk_equity_ext2026_de.csv": (
            "sjm_crowding_v3_total_return_bil_equity_ext2026_de.csv"
        ),
        "tear_sheet_ext2026.csv": "portfolio_metrics_reader_ext2026.csv",
        "tear_sheet_ext2026_de.csv": "portfolio_metrics_reader_ext2026_de.csv",
    }
)

#: Measurement provenance required on every canonical report row (task 4.1).
V4_PROVENANCE_COLUMNS: dict[str, str] = {
    "schema": "str",
    "portfolio_id": "str",
    "return_basis": "str",
    "window_label": "str",
    "start": "datetime64",
    "end": "datetime64",
    "n_obs": "int64",
    "periods_per_year": "int64",
    "cash_benchmark_id": "str",
    "currency_basis": "str",
    "source": "str",
}

_V4_READER_METRICS = {
    metric: "float64"
    for metric in (
        "total_return",
        "maxdd",
        "downside_rms",
        "cagr",
        "ann_vol",
        "sharpe",
        "sortino",
        "calmar",
    )
}
_V4_LEGACY_METRICS = {
    metric: "float64"
    for metric in (
        "total_return",
        "maxdd",
        "downside_rms",
        "cagr_rows",
        "ann_vol_cal",
        "sharpe_cal",
        "sortino_cal",
        "calmar_rows",
    )
}

_SSR_INT_FIELDS = frozenset(
    {"n_obs", "n_rolling", "L_hac", "block_len", "n_boot", "seed", "window", "periods_per_year"}
)


def _ssr_columns(int_dtype: str) -> dict[str, str]:
    """``ssr_*`` row fields projected from the vendored (byte-identical to
    root) SSRResult/SSRInference dataclasses — the one shared authority."""
    names = [field.name for field in dataclasses.fields(vendored_ssr.SSRResult)]
    names += [
        field.name
        for field in dataclasses.fields(vendored_ssr.SSRInference)
        if field.name != "result"
    ]
    return {
        f"ssr_{name}": (int_dtype if name in _SSR_INT_FIELDS else "float64")
        for name in names
    }


_ATTRIBUTION_FIELD_DTYPES: dict[str, str] = {
    "kind": "str",
    "intercept_native_period": "float64",
    "intercept_ann_arithmetic": "float64",
    "intercept_se_hac": "float64",
    "intercept_t_hac": "float64",
    "beta": "float64",
    "r2": "float64",
    "n_obs": "int64",
    "start": "datetime64",
    "end": "datetime64",
    "periods_per_year": "int64",
    "hac_maxlags": "int64",
}


def _attribution_columns(int_dtype: str) -> dict[str, str]:
    return {
        f"raw_market_model_{name}": (int_dtype if dtype == "int64" else dtype)
        for name, dtype in _ATTRIBUTION_FIELD_DTYPES.items()
    }


def _crisis_columns(int_dtype: str) -> dict[str, str]:
    columns = {
        name: "datetime64"
        for name in (
            "requested_start",
            "requested_end",
            "anchor",
            "first_return_date",
            "actual_end",
        )
    }
    columns |= {
        name: "float64"
        for name in ("episode_return", "boundary_anchored_max_drawdown", "volatility_ann")
    }
    columns["n_returns"] = int_dtype
    return columns


#: Attribution fields a risk row projects verbatim (window identity stays on
#: the provenance columns) — mirror of RISK_PROJECTED_ATTRIBUTION_FIELDS.
_RISK_PROJECTED_ATTRIBUTION = (
    "kind",
    "intercept_native_period",
    "intercept_ann_arithmetic",
    "intercept_se_hac",
    "intercept_t_hac",
    "beta",
    "r2",
    "hac_maxlags",
)

_MARKOWITZ_IDENTITY_COLUMNS: dict[str, str] = {
    "window": "str",
    "snapshot_id": "str",
    "base_currency": "str",
    "valuation_rule": "str",
    "requested_start": "datetime64",
    "requested_end": "datetime64",
    "actual_start": "datetime64",
    "actual_end": "datetime64",
    "n_obs": "int64",
    "periods_per_year": "float64",  # exactly 365.2425/7 weekly cutoffs
    "source_dates_sha256": "str",
}

_MARKOWITZ_MOMENTS_COLUMNS = {
    **_MARKOWITZ_IDENTITY_COLUMNS,
    "asset": "str",
    "quote_currency": "str",
    "quote_unit": "str",
    "mean_ann_arithmetic": "float64",
    "vol_ann": "float64",
    **{f"cov_{asset}": "float64" for asset in MARKOWITZ_ASSET_UNIVERSE},
}
_MARKOWITZ_FRONTIER_COLUMNS = {
    **_MARKOWITZ_IDENTITY_COLUMNS,
    "residual_tolerance": "float64",
    "target_return_ann": "float64",
    "success": "bool",
    "status": "int64",
    "message": "str",
    "iterations": "int64",
    "objective": "float64",
    "budget_residual": "float64",
    "target_residual": "float64",
    "bound_violation": "float64",
    "return_ann": "float64",
    "volatility_ann": "float64",
    "feasible": "bool",
    **{f"weight_{asset}": "float64" for asset in MARKOWITZ_ASSET_UNIVERSE},
}

_V4_ETF_WEIGHTS = {"SWDA.L": "float64", "XLK": "float64", "IAU": "float64", "BIL": "float64"}


def _v4_report_table(stem: str, columns: dict[str, str], **kwargs: Any) -> AssetSpec:
    """Report tables persist ``report_table`` frames: RangeIndex, no name."""
    return AssetSpec(f"{stem}.parquet", "parquet", None, columns, min_rows=1, **kwargs)


def _v4_daily(stem: str, columns: dict[str, str]) -> AssetSpec:
    return AssetSpec(f"{stem}.parquet", "parquet", "Date", columns, min_rows=3)


_V4_ATTRIBUTION_OPTIONAL = frozenset(_attribution_columns("number64"))

DATA_V4_ASSET_SPECS: dict[str, AssetSpec] = {
    # -- canonical reader / legacy / differential / attribution / crisis ------
    # attribution columns join a reader row only when attribution covers the
    # exact performance window ("full"); a performance_only-only table omits
    # them, and a mixed table NaN-dilutes the integer fields -> number64.
    "portfolio_metrics_reader_ext2026": _v4_report_table(
        "portfolio_metrics_reader_ext2026",
        {
            **V4_PROVENANCE_COLUMNS,
            "row_kind": "str",
            **_V4_READER_METRICS,
            **_ssr_columns("int64"),
            **_attribution_columns("number64"),
        },
        optional=_V4_ATTRIBUTION_OPTIONAL,
    ),
    "portfolio_metrics_vectorbt365_ext2026": _v4_report_table(
        "portfolio_metrics_vectorbt365_ext2026",
        {**V4_PROVENANCE_COLUMNS, **_V4_LEGACY_METRICS},
    ),
    "portfolio_metrics_differential_ext2026": _v4_report_table(
        "portfolio_metrics_differential_ext2026",
        {
            **V4_PROVENANCE_COLUMNS,
            **_V4_READER_METRICS,
            **_ssr_columns("int64"),
            "endpoint_total_return_difference": "float64",
        },
    ),
    "attribution_raw_market_model_ext2026": _v4_report_table(
        "attribution_raw_market_model_ext2026",
        {**V4_PROVENANCE_COLUMNS, **_attribution_columns("int64")},
    ),
    "crisis_metrics_ext2026": _v4_report_table(
        "crisis_metrics_ext2026",
        {**V4_PROVENANCE_COLUMNS, **_crisis_columns("int64")},
    ),
    # -- assembled tear sheets (mixed row families) ----------------------------
    "tear_sheet_ai_variants_ext2026": _v4_report_table(
        "tear_sheet_ai_variants_ext2026",
        {
            **V4_PROVENANCE_COLUMNS,
            "row_kind": "str",
            **_V4_READER_METRICS,
            **_ssr_columns("int64"),
            **_attribution_columns("number64"),
            "endpoint_total_return_difference": "float64",
        },
        optional=_V4_ATTRIBUTION_OPTIONAL,
    ),
    "tear_sheet_sjm_crowding_ext2026": _v4_report_table(
        "tear_sheet_sjm_crowding_ext2026",
        {
            **V4_PROVENANCE_COLUMNS,
            "row_kind": "str",
            **_V4_READER_METRICS,
            **_ssr_columns("number64"),
            **_attribution_columns("number64"),
            **_crisis_columns("number64"),
        },
    ),
    "tear_sheet_trio_ext2026": _v4_report_table(
        "tear_sheet_trio_ext2026",
        {
            **V4_PROVENANCE_COLUMNS,
            "row_kind": "str",
            **_V4_READER_METRICS,
            **_ssr_columns("int64"),
            **_attribution_columns("number64"),
        },
        optional=_V4_ATTRIBUTION_OPTIONAL,
    ),
    # -- auxiliary monthly-return and risk-decomposition tables ----------------
    "monthly_returns_ext2026": _v4_report_table(
        "monthly_returns_ext2026",
        {
            **V4_PROVENANCE_COLUMNS,
            "year": "int64",
            "month": "int64",
            "monthly_return": "float64",
        },
    ),
    "risk_decomposition_ext2026": _v4_report_table(
        "risk_decomposition_ext2026",
        {
            **V4_PROVENANCE_COLUMNS,
            "source_schema": "str",
            **{
                f"raw_market_model_{name}": _ATTRIBUTION_FIELD_DTYPES[name]
                for name in _RISK_PROJECTED_ATTRIBUTION
            },
            "systematic_variance_share": "float64",
            "idiosyncratic_variance_share": "float64",
        },
    ),
    # -- Markowitz moment and frontier tables (both canonical windows) ---------
    "markowitz_10y_moments": _v4_report_table(
        "markowitz_10y_moments", dict(_MARKOWITZ_MOMENTS_COLUMNS)
    ),
    "markowitz_10y_frontier": _v4_report_table(
        "markowitz_10y_frontier", dict(_MARKOWITZ_FRONTIER_COLUMNS)
    ),
    "markowitz_max_moments": _v4_report_table(
        "markowitz_max_moments", dict(_MARKOWITZ_MOMENTS_COLUMNS)
    ),
    "markowitz_max_frontier": _v4_report_table(
        "markowitz_max_frontier", dict(_MARKOWITZ_FRONTIER_COLUMNS)
    ),
    # -- Factor strategy series -------------------------------------------------
    "factor_equity_ext2026": _v4_daily("factor_equity_ext2026", {"value": "float64"}),
    "factor_targets_ext2026": _v4_daily("factor_targets_ext2026", dict(_V4_ETF_WEIGHTS)),
    "factor_nonpit_diagnostic_equity_ext2026": _v4_daily(
        "factor_nonpit_diagnostic_equity_ext2026", {"value": "float64"}
    ),
    "factor_nonpit_diagnostic_targets_ext2026": _v4_daily(
        "factor_nonpit_diagnostic_targets_ext2026", dict(_V4_ETF_WEIGHTS)
    ),
    # -- SJM v3 strategy series -------------------------------------------------
    "sjm_crowding_v3_total_return_bil_equity_ext2026": _v4_daily(
        "sjm_crowding_v3_total_return_bil_equity_ext2026", {"value": "float64"}
    ),
    "sjm_crowding_v3_total_return_bil_targets_ext2026": _v4_daily(
        "sjm_crowding_v3_total_return_bil_targets_ext2026",
        {"target_exposure": "float64"},
    ),
    "sjm_crowding_v3_total_return_bil_daily_returns_ext2026": _v4_daily(
        "sjm_crowding_v3_total_return_bil_daily_returns_ext2026",
        {"daily_return": "float64", "factor_return": "float64", "cash_return": "float64"},
    ),
    "sjm_crowding_v3_total_return_bil_control_returns_ext2026": _v4_daily(
        "sjm_crowding_v3_total_return_bil_control_returns_ext2026",
        {"control_return": "float64"},
    ),
    # -- publication manifest ---------------------------------------------------
    "publication_manifest": AssetSpec(
        "publication_manifest.json",
        "json",
        None,
        {
            "schema": "str",
            "schema_id": "str",
            "release_tag": "str",
            "publication_id": "str",
            "build_time": "str",
            "catalog_sha256": "str",
            "artifacts": "list",
            "assets": "dict",
            "input_manifests": "dict",
            "compatibility_paths": "list",
            "completed": "bool",
        },
        min_rows=1,
        row_container="artifacts",
    ),
}
# The tabular compatibility alias validates under its target's exact schema.
DATA_V4_ASSET_SPECS["sjm_crowding_derisk_equity_ext2026"] = dataclasses.replace(
    DATA_V4_ASSET_SPECS["sjm_crowding_v3_total_return_bil_equity_ext2026"],
    asset="sjm_crowding_derisk_equity_ext2026.parquet",
)

#: Registry key -> catalog schema identity (mirror of the frozen data-v4 set).
DATA_V4_TABLE_SCHEMAS: dict[str, str] = {
    "portfolio_metrics_reader_ext2026": READER_SCHEMA,
    "portfolio_metrics_vectorbt365_ext2026": LEGACY_SCHEMA,
    "portfolio_metrics_differential_ext2026": DIFFERENTIAL_SCHEMA,
    "attribution_raw_market_model_ext2026": ATTRIBUTION_SCHEMA,
    "crisis_metrics_ext2026": CRISIS_SCHEMA,
    "tear_sheet_ai_variants_ext2026": AI_VARIANTS_TEAR_SHEET_SCHEMA,
    "tear_sheet_sjm_crowding_ext2026": SJM_TEAR_SHEET_SCHEMA,
    "tear_sheet_trio_ext2026": TRIO_TEAR_SHEET_SCHEMA,
    "monthly_returns_ext2026": MONTHLY_SCHEMA,
    "risk_decomposition_ext2026": RISK_DECOMPOSITION_SCHEMA,
    "markowitz_10y_moments": MARKOWITZ_MOMENTS_SCHEMA,
    "markowitz_10y_frontier": MARKOWITZ_FRONTIER_SCHEMA,
    "markowitz_max_moments": MARKOWITZ_MOMENTS_SCHEMA,
    "markowitz_max_frontier": MARKOWITZ_FRONTIER_SCHEMA,
    "factor_equity_ext2026": "factor.equity.v1",
    "factor_targets_ext2026": "factor.targets.v1",
    "factor_nonpit_diagnostic_equity_ext2026": "factor.equity.v1",
    "factor_nonpit_diagnostic_targets_ext2026": "factor.targets.v1",
    "sjm_crowding_v3_total_return_bil_equity_ext2026": "sjm.equity.v3",
    "sjm_crowding_v3_total_return_bil_targets_ext2026": "sjm.targets.v3",
    "sjm_crowding_v3_total_return_bil_daily_returns_ext2026": "sjm.daily_returns.v3",
    "sjm_crowding_v3_total_return_bil_control_returns_ext2026": "sjm.control_returns.v3",
    "sjm_crowding_derisk_equity_ext2026": "sjm.equity.v3",
    "publication_manifest": PUBLICATION_MANIFEST_SCHEMA,
}

#: Table -> allowed per-row schema identities (mixed tear sheets carry several).
DATA_V4_ROW_SCHEMAS: dict[str, tuple[str, ...]] = {
    "portfolio_metrics_reader_ext2026": (READER_SCHEMA,),
    "portfolio_metrics_vectorbt365_ext2026": (LEGACY_SCHEMA,),
    "portfolio_metrics_differential_ext2026": (DIFFERENTIAL_SCHEMA,),
    "attribution_raw_market_model_ext2026": (ATTRIBUTION_SCHEMA,),
    "crisis_metrics_ext2026": (CRISIS_SCHEMA,),
    "tear_sheet_ai_variants_ext2026": (READER_SCHEMA, DIFFERENTIAL_SCHEMA),
    "tear_sheet_sjm_crowding_ext2026": (READER_SCHEMA, ATTRIBUTION_SCHEMA, CRISIS_SCHEMA),
    "tear_sheet_trio_ext2026": (READER_SCHEMA,),
    "monthly_returns_ext2026": (MONTHLY_SCHEMA,),
    "risk_decomposition_ext2026": (RISK_DECOMPOSITION_SCHEMA,),
}

#: Row-schema-pinned annualization (mirror of REPORT_SCHEMAS.periods_per_year);
#: None = carried per row (attribution windows, crisis windows, risk rows).
_V4_ROW_ANNUALIZATION: dict[str, int | None] = {
    READER_SCHEMA: 252,
    LEGACY_SCHEMA: 365,
    DIFFERENTIAL_SCHEMA: 252,
    MONTHLY_SCHEMA: 12,
    ATTRIBUTION_SCHEMA: None,
    CRISIS_SCHEMA: None,
    RISK_DECOMPOSITION_SCHEMA: None,
}

#: Row families that record the deterministic SSR inference settings.
_V4_SSR_SETTING_SCHEMAS = frozenset({READER_SCHEMA, DIFFERENTIAL_SCHEMA})

#: Markowitz table -> its one canonical window name.
_V4_MARKOWITZ_WINDOWS: dict[str, str] = {
    "markowitz_10y_moments": "10y",
    "markowitz_10y_frontier": "10y",
    "markowitz_max_moments": "max",
    "markowitz_max_frontier": "max",
}


def _require_v4_client(key: str, client: ReleaseClient) -> None:
    tag = getattr(client, "tag", None)
    if tag != DATA_V4_TAG:
        raise SchemaError(
            f"registry key {key!r} is bound to release tag {DATA_V4_TAG!r}; the "
            f"client is bound to {tag!r} — cross-tag substitution is prohibited"
        )


def _v4_row_error(spec: AssetSpec, index: int, row: Mapping[str, object], column: str, detail: str) -> SchemaError:
    identity = row.get("portfolio_id", row.get("asset", "<unidentified>"))
    return SchemaError(
        f"asset {spec.asset}: row {index} ({identity!r}): column {column!r}: {detail}"
    )


def _validate_v4_report_frame(key: str, spec: AssetSpec, df: pd.DataFrame) -> None:
    """Row-level semantics of one canonical report table: schema identity,
    window, count, annualization, cash benchmark, currency basis, and the
    deterministic SSR settings — each failure names its exact field."""
    allowed = DATA_V4_ROW_SCHEMAS[key]
    for index, row in enumerate(df.to_dict("records")):
        schema = row["schema"]
        if schema not in allowed:
            raise _v4_row_error(
                spec, index, row, "schema",
                f"row schema {schema!r} is not among this table's canonical row "
                f"schemas {allowed!r}",
            )
        pinned = _V4_ROW_ANNUALIZATION[schema]
        ppy = row["periods_per_year"]
        if pinned is not None and int(ppy) != pinned:
            raise _v4_row_error(
                spec, index, row, "periods_per_year",
                f"schema {schema!r} pins annualization {pinned}, got {int(ppy)}",
            )
        if pinned is None and int(ppy) < 1:
            raise _v4_row_error(
                spec, index, row, "periods_per_year",
                f"must be a positive per-row annualization, got {ppy!r}",
            )
        start, end = row["start"], row["end"]
        if pd.isna(start) or pd.isna(end) or start > end:
            raise _v4_row_error(
                spec, index, row, "start",
                f"window {start!r}..{end!r} is not a valid start<=end window",
            )
        if pd.isna(row["n_obs"]) or int(row["n_obs"]) < 1:
            raise _v4_row_error(
                spec, index, row, "n_obs",
                f"must be a positive observation count, got {row['n_obs']!r}",
            )
        cash = row["cash_benchmark_id"]
        if not isinstance(cash, str) or not cash.strip():
            raise _v4_row_error(
                spec, index, row, "cash_benchmark_id",
                f"must identify the cash benchmark, got {cash!r}",
            )
        basis = row["currency_basis"]
        if basis not in V4_CURRENCY_BASES:
            raise _v4_row_error(
                spec, index, row, "currency_basis",
                f"must be one of {V4_CURRENCY_BASES}, got {basis!r}",
            )
        if schema in _V4_SSR_SETTING_SCHEMAS:
            for setting, pinned_value in SSR_REPORT_DEFAULTS.items():
                column = f"ssr_{setting}"
                value = row[column]
                if pd.isna(value) or float(value) != float(pinned_value):  # type: ignore[arg-type]
                    raise _v4_row_error(
                        spec, index, row, column,
                        f"deterministic SSR setting must be {pinned_value!r}, "
                        f"got {value!r}",
                    )
        if "source_schema" in spec.columns and row["source_schema"] != ATTRIBUTION_SCHEMA:
            raise _v4_row_error(
                spec, index, row, "source_schema",
                f"risk rows project attribution records ({ATTRIBUTION_SCHEMA!r}), "
                f"got {row['source_schema']!r}",
            )


def _validate_v4_markowitz_frame(key: str, spec: AssetSpec, df: pd.DataFrame) -> None:
    """Markowitz identity semantics: one window per table, USD base, the exact
    weekly annualization, positive counts, and actual-within-requested dates."""
    window = _V4_MARKOWITZ_WINDOWS[key]
    for index, row in enumerate(df.to_dict("records")):
        if row["window"] != window:
            raise _v4_row_error(
                spec, index, row, "window",
                f"table {key!r} carries only the {window!r} window, got "
                f"{row['window']!r}",
            )
        if row["base_currency"] != "USD":
            raise _v4_row_error(
                spec, index, row, "base_currency",
                f"the Markowitz opportunity set is valued in USD, got "
                f"{row['base_currency']!r}",
            )
        if float(row["periods_per_year"]) != MARKOWITZ_WEEKLY_PERIODS_PER_YEAR:  # type: ignore[arg-type]
            raise _v4_row_error(
                spec, index, row, "periods_per_year",
                "weekly annualization must be exactly 365.2425/7 "
                f"({MARKOWITZ_WEEKLY_PERIODS_PER_YEAR!r}), got "
                f"{row['periods_per_year']!r}",
            )
        if pd.isna(row["n_obs"]) or int(row["n_obs"]) < 1:
            raise _v4_row_error(
                spec, index, row, "n_obs",
                f"must be a positive weekly cutoff count, got {row['n_obs']!r}",
            )
        requested_start, requested_end = row["requested_start"], row["requested_end"]
        actual_start, actual_end = row["actual_start"], row["actual_end"]
        if pd.isna(requested_start) or pd.isna(requested_end) or requested_start > requested_end:
            raise _v4_row_error(
                spec, index, row, "requested_start",
                f"requested window {requested_start!r}..{requested_end!r} is invalid",
            )
        if (
            pd.isna(actual_start)
            or pd.isna(actual_end)
            or actual_start > actual_end
            or actual_start < requested_start
            or actual_end > requested_end
        ):
            raise _v4_row_error(
                spec, index, row, "actual_start",
                f"actual window {actual_start!r}..{actual_end!r} must sit inside "
                f"the requested window {requested_start!r}..{requested_end!r}",
            )


def _validate_v4_manifest(spec: AssetSpec, obj: Mapping[str, object]) -> None:
    for column, expected in (
        ("schema_id", PUBLICATION_MANIFEST_SCHEMA),
        ("release_tag", DATA_V4_TAG),
    ):
        if obj.get(column) != expected:
            raise SchemaError(
                f"asset {spec.asset}: column {column!r}: expected {expected!r}, "
                f"got {obj.get(column)!r}"
            )
    if obj.get("completed") is not True:
        raise SchemaError(
            f"asset {spec.asset}: column 'completed': the publication manifest "
            "must declare completed=true"
        )


def load_v4_frame(client: ReleaseClient, key: str) -> tuple[pd.DataFrame, Provenance]:
    """Load and validate one tabular ``data-v4`` asset by its registry key.

    Args:
        client: Release client BOUND to the ``data-v4`` tag.
        key: Logical data-v4 registry key, e.g.
            ``"portfolio_metrics_reader_ext2026"``.

    Returns:
        The validated DataFrame and the retrieval provenance.

    Raises:
        KeyError: Unknown data-v4 registry key.
        ValueError: The key is not a tabular asset.
        SchemaError: Cross-tag substitution, or any structural or semantic
            contract mismatch — always naming asset, row, and field.
    """
    spec = DATA_V4_ASSET_SPECS[key]
    if spec.kind != "parquet":
        raise ValueError(f"{key}: kind {spec.kind!r} is not tabular; use load_v4_json")
    _require_v4_client(key, client)
    data, provenance = client.fetch(spec.asset)
    df = pd.read_parquet(io.BytesIO(data))
    _validate_frame(spec, df)
    if key in DATA_V4_ROW_SCHEMAS:
        _validate_v4_report_frame(key, spec, df)
    elif key in _V4_MARKOWITZ_WINDOWS:
        _validate_v4_markowitz_frame(key, spec, df)
    return df, provenance


def load_v4_json(client: ReleaseClient, key: str) -> tuple[dict, Provenance]:
    """Load and validate one JSON ``data-v4`` asset by its registry key.

    Args:
        client: Release client BOUND to the ``data-v4`` tag.
        key: Logical data-v4 registry key, e.g. ``"publication_manifest"``.

    Returns:
        The validated JSON object and the retrieval provenance.

    Raises:
        KeyError: Unknown data-v4 registry key.
        ValueError: The key is not a JSON asset.
        SchemaError: Cross-tag substitution or any contract mismatch.
    """
    spec = DATA_V4_ASSET_SPECS[key]
    if spec.kind != "json":
        raise ValueError(f"{key}: kind {spec.kind!r} is not JSON; use load_v4_frame")
    _require_v4_client(key, client)
    data, provenance = client.fetch(spec.asset)
    obj = json.loads(data)
    _validate_json(spec, obj)
    if key == "publication_manifest":
        _validate_v4_manifest(spec, obj)
    return obj, provenance
