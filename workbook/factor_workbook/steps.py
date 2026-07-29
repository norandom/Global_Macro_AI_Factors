"""Storyboard step assembly: typed S0-S5 view models with mandated framing
text and explicit gap markers (R2-R6, R7.3). Implemented in tasks 4.1-4.5;
S0 (the data-v2 static buy-and-hold opener) added by the task 7.1 amendment.

S1 (task 4.1): the certification step. Known upstream gaps are view-model
states, never exceptions (R2.4, R2.5): a verdict row whose raw-evidence
bundle member is absent renders ``pending_evidence`` (data-driven — keyed
off the release content, not off a hardcoded model), and the unservable
candidate (verdict ``screen_failed``) renders ``unscreenable`` — not
exonerated — with its recorded error.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import pandas as pd

from factor_workbook import certification
from factor_workbook.contract import (
    ATTRIBUTION_SCHEMA,
    CRISIS_SCHEMA,
    DATA_V4_TAG,
    READER_SCHEMA,
    SSR_REPORT_DEFAULTS,
    SchemaError,
    load_frame,
    load_json,
    load_v4_frame,
)
from factor_workbook.rederive import (
    EquityMetrics,
    contamination_premium,
    equity_metrics,
    evidence_class_stats,
    guarded_tilt,
    loading_stability,
    wilson_ci,
)
from factor_workbook.release import ReleaseClient, ReleaseError
from factor_workbook.vendored_ssr import SSRInference, SSRResult, ssr_inference
from factor_workbook.verify import Check, compare

_N_SPLITS = 5  # certification_stats default; each arm needs >= this many rows

#: Honest outcome framing for S1 (R2.2, R7.3) — displayed with the sheet.
S1_FRAMING = (
    "Honest outcome: the certified set is empty — every screenable candidate "
    "recalls the identified macro history with statistical certainty. "
    "openai/gpt-oss-20b is a documented, user-selected fallback that runs "
    "recall-guarded; it was never certified, and no candidate was."
)

_CERT_COLUMNS = [
    "model",
    "controlled_auc",
    "controlled_ci_low",
    "controlled_ci_high",
    "controlled_perm_p",
    "positive_control_auc",
    "positive_control_perm_p",
    "parse_rate",
    "n_per_class",
    "verdict",
    "status",
    "note",
]


@dataclass(frozen=True)
class StepView:
    """One storyboard step as a typed, sheet-ready view model (R7.3).

    Attributes:
        title: Sheet title, e.g. ``"S1 — Model certification"``.
        framing: The mandated framing language displayed with the step.
        tables: Named tables the add-in expands into the sheet.
        checks: Published-vs-re-derived verification rows for the step (R7.2).
    """

    title: str
    framing: str
    tables: dict[str, pd.DataFrame]
    checks: list[Check]


#: Published static_bh metric fields recomputable from the equity series alone
#: (rederive.equity_metrics is the exact producer; build_static_bh.py used it
#: to WRITE static_bh_stats.json, so full-data agreement is exact).
_S0_METRIC_FIELDS = [
    "total_return",
    "annualized_return",
    "annualized_vol",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
]

#: Published static_bh_ssr fields; the vendored ``SSRResult`` attributes carry
#: the same names (same producer, exact full-data agreement).
_S0_SSR_FIELDS = ["ssr", "mean_rolling_sr", "sigma_hac", "L_hac", "n_rolling"]


def _claim_check(name: str, holds: bool, inference: SSRInference) -> Check:
    """Re-derive one framing sentence's verdict bit as a 1/0 check row (R7.2).

    The framing states a conclusion; ``vendored_ssr.ssr_inference`` is the only
    thing allowed to decide whether it holds — never a threshold re-encoded
    here. A degenerate inference (too few rolling observations, as on the
    fixture subsets) can neither confirm nor deny the claim, so it renders as a
    disagreement rather than passing by default.
    """
    return compare(name, 1.0, float(holds and math.isfinite(inference.p_value)))


def _inference_row(label: str, inference: SSRInference, *, differential: bool = False) -> dict:
    """One re-derived MBB-inference row: p-values, block length, verdict."""
    return {
        "line": label,
        "ssr": inference.result.ssr,
        "mbb_p": inference.p_value,
        "mbb_p_mirror": inference.p_value_lower,
        "mbb_block": inference.block_len,
        "n_boot": inference.n_boot,
        "alpha": inference.alpha,
        "stable": inference.stable,
        "stably_below": inference.stably_below,
        "verdict_rederived": inference.verdict(differential=differential),
    }

#: Per-episode crisis fields re-derived via ``equity_metrics(value, crisis=...)``.
_S0_CRISIS_FIELDS = ["crisis_return", "crisis_max_drawdown", "crisis_vol_ann"]

#: (stats window key, registry key, view table name); 10y opener first.
_S0_WINDOWS = (
    ("2016_2026", "static_bh_equity_2016_2026", "equity_10y"),
    ("2014_2024", "static_bh_equity_2014_2024", "equity"),
)

#: Two-claims framing for S0 (R7.3, task 7.1) — displayed with the sheet. The
#: in-sample caveat is carried verbatim from the published stats["caveat"].
S0_FRAMING = (
    "Step 0 — the static buy-and-hold opener, two claims kept separate: the "
    "crisis-drawdown episodes (COVID 2020, inflation 2022) are REAL "
    "event-level observables — their timing, depth, and co-movement carry no "
    "selection artifact — while the performance LEVEL is hindsight-flattered. "
    "IN-SAMPLE BY CONSTRUCTION: the four ETFs were selected by the Sharpe "
    "Stability Ratio computed over the same window being simulated "
    "(nb02/nb03). This line illustrates how strong a hindsight-selected "
    "static portfolio looks — the lookahead/contamination problem the "
    "recall-guarded pipeline measures. Its performance is a hindsight "
    "artifact, never attainable skill. The line's own SSR of 0.147 tests "
    "stably above zero (one-sided MBB p < 0.05) — but that is temporal "
    "consistency of a hindsight-selected book, beta-compatible and NOT a "
    "skill claim. This is the problem the following steps S1-S5 measure. "
    "No forecast-accuracy claim is made."
)


def build_s0(client: ReleaseClient) -> StepView:
    """Assemble the S0 static buy-and-hold view (task 7.1; R1.4, R7.2, R7.3).

    Per published window — notebook 04's original 2016-2026 decade
    (``equity_10y``) and the walk-forward-aligned 2014-2024 window
    (``equity``) — the loaded equity value series with its drawdown
    re-derived (value over running max, minus one). The ``targets_drift``
    table is the drifting buy-and-hold weights verbatim (shares are held;
    weights drift). ``stats`` flattens the published per-window metrics —
    static line beside the SPY reference — and ``crisis_episodes`` carries
    one row per named macro-crisis episode per window. Every published
    static-line figure is re-derived from the loaded equity series and
    attached as a check: the seven equity metrics, the five SSR fields
    (vendored ``ssr_inference`` over daily returns), and the per-episode
    crisis figures; on fixture-sized subsets the disagreements are rendered
    flags, never exceptions (R7.2). The ``inference`` table carries what the
    published stats do not: the paper's one-sided moving-block-bootstrap
    p-value, its mirror tail, the block length, and the rendered verdict — and
    the framing's "tests stably above zero" sentence is re-derived from it as
    its own check row, so the headline claim is never asserted on trust. On
    ``data-v1`` the assets are absent and
    the loaders raise the typed per-asset :class:`ReleaseError`, which the
    Excel surface renders in-cell (R1.4).

    Args:
        client: Release client for the pinned data version (``data-v2``+).

    Returns:
        The typed S0 view model with the mandated two-claims framing.
    """
    stats, _ = load_json(client, "static_bh_stats")
    tables: dict[str, pd.DataFrame] = {}
    checks: list[Check] = []
    stats_rows: list[dict] = []
    episode_rows: list[dict] = []
    inference_rows: list[dict] = []
    for window, key, table_name in _S0_WINDOWS:
        values, _ = load_frame(client, key)
        value = values["value"]
        tables[table_name] = pd.DataFrame(
            {"value": value, "drawdown": value / value.cummax() - 1.0}
        )
        published = stats[window]
        metrics = equity_metrics(value)
        for field in _S0_METRIC_FIELDS:
            stats_rows.append(
                {
                    "window": window,
                    "metric": field,
                    "static_bh": published["static_bh"][field],
                    "spy_bh": published["spy_bh"].get(field),
                }
            )
            checks.append(
                compare(
                    f"S0 {window} {field} vs published",
                    published["static_bh"][field],
                    getattr(metrics, field),
                )
            )
        inference = ssr_inference(value.pct_change().dropna())
        ssr = inference.result
        inference_rows.append(_inference_row(window, inference))
        checks.append(
            _claim_check(
                f"S0 {window} framing claim 'SSR tests stably above zero' (1 = holds)",
                inference.stable,
                inference,
            )
        )
        for field in _S0_SSR_FIELDS:
            stats_rows.append(
                {
                    "window": window,
                    "metric": field,
                    "static_bh": published["static_bh_ssr"][field],
                    "spy_bh": None,
                }
            )
            checks.append(
                compare(
                    f"S0 {window} {field} vs published",
                    published["static_bh_ssr"][field],
                    getattr(ssr, field),
                )
            )
        for episode, record in sorted(published["crisis_episodes"].items()):
            start, end = record["window"]
            crisis = equity_metrics(value, crisis=(start, end))
            row: dict = {"window": window, "episode": episode, "start": start, "end": end}
            for field in _S0_CRISIS_FIELDS:
                row[f"static_{field}"] = record["static_bh"][field]
                row[f"spy_{field}"] = record["spy_bh"][field]
                checks.append(
                    compare(
                        f"S0 {window} {episode} {field} vs published",
                        record["static_bh"][field],
                        getattr(crisis, field),
                    )
                )
            episode_rows.append(row)
    targets, _ = load_frame(client, "static_bh_targets_2014_2024")
    tables["targets_drift"] = targets
    tables["stats"] = pd.DataFrame(stats_rows)
    tables["crisis_episodes"] = pd.DataFrame(episode_rows)
    tables["inference"] = pd.DataFrame(inference_rows)
    return StepView(
        title="S0 — Static buy-and-hold line (hindsight-selected, in-sample)",
        framing=S0_FRAMING,
        tables=tables,
        checks=checks,
    )


def _slug(model: str) -> str:
    """Release evidence-directory slug for a candidate model name."""
    return model.replace("/", "_")


def _evidence_consistency(
    client: ReleaseClient, model: str, slug: str, evidence: pd.DataFrame
) -> tuple[pd.DataFrame, list[Check]]:
    """Class-count/feature-statistics consistency vs the published summary (R2.6).

    Always: per MAIN evidence arm (identifying / anonymized / prose_confounded),
    the TOTAL gathered row count — included and dropped alike — is compared
    against the published ``n_per_class``: the screen gathers exactly
    ``n_per_class`` prompts per main arm and records failures as dropped rows,
    so gathered-total is the published invariant while included counts vary
    with parse/timeout attrition (validation 2026-07-09: comparing included
    rows false-alarmed on the pristine release). The ``parse_sample`` arm has
    its own fixed sample size, unpublished — excluded from the count check.
    The ``std_*`` feature mean/std summary rows are re-derived from INCLUDED
    rows (the rows that fed the statistics). Deeper (statistics extra only):
    where both main arms carry at least ``_N_SPLITS`` included rows, the point
    AUC is re-derived and compared against the published ``controlled_auc``;
    fixture-sized subsets skip this gracefully — the full-data path runs live.

    Returns:
        The per-arm feature-statistics table and the comparison checks.
    """
    summary, _ = load_json(client, "norecall_screen_evidence_summary", model=slug)
    included = evidence[evidence["included"]]
    stats = evidence_class_stats(included)
    gathered = evidence[evidence["arm"] != "parse_sample"]["arm"].value_counts()
    checks = [
        compare(
            f"S1 {model} [{arm}] gathered rows vs published n_per_class",
            summary["n_per_class"],
            count,
        )
        for arm, count in sorted(gathered.items())
    ]

    if certification.available():
        baseline, _ = load_json(client, "norecall_screen_evidence_baseline", model=slug)
        columns = [
            f"std_{name}"
            for name, mean in baseline["feature_means"].items()
            if mean is not None
        ]
        x_is = included[included["arm"] == "identifying"][columns].to_numpy()
        x_oos = included[included["arm"] == "anonymized"][columns].to_numpy()
        if min(len(x_is), len(x_oos)) >= _N_SPLITS:
            # ponytail: point AUC only — auc_obs is independent of n_boot /
            # n_perm, so both are minimized; CI/perm-p stay published-only.
            auc = certification.certification_stats(
                x_is, x_oos, n_boot=1, n_perm=1, seed=0
            )[0]
            checks.append(
                compare(
                    f"S1 {model} controlled AUC re-derived from raw evidence",
                    summary["controlled_auc"],
                    auc,
                )
            )
    return stats.feature_stats, checks


def build_s1(client: ReleaseClient) -> StepView:
    """Assemble the S1 certification view with evidence drill-down (R2.1-2.6).

    One certification-table row per screened candidate — published controlled
    separation with CI and permutation p, positive control, parse rate, sample
    size, verdict — plus an explicit gap-status column: ``evidence_available``,
    ``pending_evidence`` (R2.4), or ``unscreenable`` — not exonerated (R2.5).
    Per evidence-bearing candidate the raw per-prompt records are attached as
    ``evidence:<slug>`` (R2.3) and the re-derived class statistics as
    ``class_stats:<slug>`` with consistency checks (R2.6). Never raises on
    the known upstream gaps.

    Args:
        client: Release client for the pinned data version.

    Returns:
        The typed S1 view model with the mandated framing.
    """
    results, _ = load_json(client, "norecall_screen_results")
    tables: dict[str, pd.DataFrame] = {}
    checks: list[Check] = []
    rows: list[dict] = []
    for record in results["results"]:
        model = record["model"]
        slug = _slug(model)
        if record.get("verdict") == "screen_failed":
            status = "unscreenable"
            note = (
                "unscreenable — not exonerated: "
                f"{record.get('error', 'no recorded reason')}"
            )
        else:
            try:
                evidence, _ = load_frame(client, "norecall_screen_evidence", model=slug)
            except ReleaseError:
                # known upstream gap: verdict row without raw evidence (R2.4)
                status, note = "pending_evidence", "raw evidence pending"
            else:
                status, note = "evidence_available", ""
                tables[f"evidence:{slug}"] = evidence
                stats_table, model_checks = _evidence_consistency(
                    client, model, slug, evidence
                )
                tables[f"class_stats:{slug}"] = stats_table
                checks.extend(model_checks)
        rows.append(
            {column: record.get(column) for column in _CERT_COLUMNS}
            | {"model": model, "status": status, "note": note}
        )
    tables["certification"] = pd.DataFrame(rows, columns=_CERT_COLUMNS)
    return StepView(
        title="S1 — Model certification (no-recall screen)",
        framing=S1_FRAMING,
        tables=tables,
        checks=checks,
    )


#: Published full-data naive directional accuracy (28/72, nb13, data-v1).
_S2_PUBLISHED_ACCURACY = 28 / 72

#: No-alpha outcome framing for S2 (R3.3, R7.3) — displayed with the sheet.
S2_FRAMING = (
    "Coin-flip outcome, by design: the accuracy shown is the expected, "
    "correct result of an honesty measurement, not a shortfall. "
    "openai/gpt-oss-20b was selected despite maximal recall precisely to "
    "demonstrate guarding. On the full published data the naive directional "
    "accuracy is 0.389 (28/72) with a Wilson 95% interval of [0.285, 0.504] "
    "— an interval that contains the coin-flip level 0.5. The accuracy "
    "figure is never a performance target to be improved, and no "
    "forecast-accuracy claim is made."
)


def build_s2(client: ReleaseClient) -> StepView:
    """Assemble the S2 coin-flip naive-evaluation view (R3.1-3.3).

    The per-call directional records — date, prompt, model reply, predicted
    direction, confidence, realized direction, correctness — verbatim as
    ``naive_eval`` (R3.1), plus a one-row ``summary`` re-derived from those
    records: n, successes, accuracy, the Wilson 95% interval, and whether the
    interval contains the coin-flip level 0.5 (R3.2). The re-derived accuracy
    is compared against the published full-data figure as a rendered check
    row (R7.2) — on a row-subset load the disagreement is a flag, never an
    exception.

    Args:
        client: Release client for the pinned data version.

    Returns:
        The typed S2 view model with the mandated no-alpha framing.
    """
    records, _ = load_frame(client, "naive_directional_eval")
    n = len(records)
    successes = int(records["correct"].sum())
    accuracy = successes / n
    ci_low, ci_high = wilson_ci(successes, n)
    summary = pd.DataFrame(
        [
            {
                "n": n,
                "successes": successes,
                "accuracy": accuracy,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "contains_half": ci_low <= 0.5 <= ci_high,
            }
        ]
    )
    checks = [
        compare("S2 accuracy vs published (full data)", _S2_PUBLISHED_ACCURACY, accuracy)
    ]
    return StepView(
        title="S2 — Coin-flip naive prediction",
        framing=S2_FRAMING,
        tables={"naive_eval": records, "summary": summary},
        checks=checks,
    )


#: The five macro axes carried by every loadings table.
_AXES = ["inflation", "growth", "credit_stress", "policy", "risk_appetite"]

#: Factor-development framing for S3 (R4.3, R4.5, R7.3) — displayed with the sheet.
S3_FRAMING = (
    "Factor development on the numbers: the per-rebalance loadings are "
    "continuous exposures on the five macro axes, built from the published "
    "figures. The recall guard discounts every raw tilt by the measured "
    "memorization score — guarded = raw * (1 - p_memorized) — re-derived "
    "here and checked against the published guarded values. The prompt-v2 "
    "refinement was adopted by the accept-gate on contamination (mean "
    "p_memorized 0.2384 vs 0.3080) and every performance check passed. "
    "The same gate rejected v2 on the previous run of the same notebook, so "
    "the decision is one draw from a non-deterministic generator, not a "
    "settled ranking; both prompt versions' data are preserved as "
    "alternatives. No forecast-accuracy claim is made."
)


def _score_summary_row(version: str, scores: pd.DataFrame) -> dict:
    """One distribution-summary row for a version's loaded p_memorized scores."""
    p = scores["p_memorized"].dropna()
    return {
        "version": version,
        "n": len(p),
        "mean": float(p.mean()),
        "median": float(p.median()),
        "p90": float(p.quantile(0.9)),
        "min": float(p.min()),
        "max": float(p.max()),
    }


def _gate_row(name: str, payload: dict, adopted: str, decision: str) -> dict:
    """Flatten one accept-gate check: its v1/v2 inputs, tolerance, and verdict."""
    return {
        "check": name,
        "v1": next(v for k, v in payload.items() if k.endswith("_v1")),
        "v2": next(v for k, v in payload.items() if k.endswith("_v2")),
        "tolerance": payload.get("tolerance"),
        "pass": payload["pass"],
        "adopted_version": adopted,
        "decision": decision,
    }


def build_s3(client: ReleaseClient) -> StepView:
    """Assemble the S3 factor-development view with the guard re-derived (R4.1-4.5).

    Per prompt version the loadings-with-parse-status (``loadings_v*``, R4.1)
    and memorization scores (``scores_v*``) plus a one-row-per-version
    distribution summary re-derived from the loaded scores (``score_summary``,
    R4.2). The raw-vs-guarded views table carries the guard formula re-derived
    as its own column, with a per-row identity check against the published
    guarded values — raw times one minus score, rtol 1e-9 (R4.3). Per-version
    stability shows the re-derived ``mean_std``/``mean_mac`` next to the
    published figures with comparison checks; on fixture-sized row subsets the
    disagreement is a rendered flag, never an exception (R4.4, R7.2). The
    ``gate`` table flattens the recorded accept-gate: one row per check with
    its v1/v2 inputs, tolerance, and pass flag, plus the adopted version and
    the rejection decision — both versions' data preserved (R4.5).

    Args:
        client: Release client for the pinned data version.

    Returns:
        The typed S3 view model with the mandated framing.
    """
    tables: dict[str, pd.DataFrame] = {}
    checks: list[Check] = []
    summary_rows: list[dict] = []
    stability_rows: list[dict] = []
    for version in ("v1", "v2"):
        loadings, _ = load_frame(client, f"factor_loadings_{version}")
        scores, _ = load_frame(client, f"factor_scores_{version}")
        tables[f"loadings_{version}"] = loadings
        tables[f"scores_{version}"] = scores
        summary_rows.append(_score_summary_row(version, scores))
        rederived = loading_stability(loadings[_AXES], loadings["parse_ok"])
        published, _ = load_json(client, f"factor_stability_{version}")
        stability_rows.append(
            {
                "version": version,
                "mean_std_rederived": rederived["mean_std"],
                "mean_std_published": published["mean_std"],
                "mean_mac_rederived": rederived["mean_mac"],
                "mean_mac_published": published["mean_mac"],
            }
        )
        for measure in ("mean_std", "mean_mac"):
            checks.append(
                compare(
                    f"S3 stability {measure} {version} vs published",
                    published[measure],
                    rederived[measure],
                )
            )
    tables["score_summary"] = pd.DataFrame(summary_rows)
    tables["stability"] = pd.DataFrame(stability_rows)

    views, _ = load_frame(client, "factor_views_v1")
    views = views.assign(
        guarded_tilt_rederived=guarded_tilt(views["raw_tilt"], views["p_memorized"])
    )
    tables["views_v1"] = views
    # per-row identity of the published table — passes on any row subset
    max_deviation = float(
        (views["guarded_tilt"] - views["guarded_tilt_rederived"]).abs().max()
    )
    checks.append(
        compare("S3 guarded_tilt equals raw*(1-p)", 0.0, max_deviation, tol=1e-9)
    )

    gate, _ = load_json(client, "prompt_version_gate_v1")
    tables["gate"] = pd.DataFrame(
        [
            _gate_row(name, payload, gate["adopted_version"], gate["decision"])
            for name, payload in sorted(gate["checks"].items())
        ]
    )
    return StepView(
        title="S3 — AI macro-factor development (recall-guarded)",
        framing=S3_FRAMING,
        tables=tables,
        checks=checks,
    )


#: Line labels (R5.2): the PIT line is the deployable portfolio, exactly so;
#: the recall-enabled line is a diagnostic control and never deployable.
_S4_PIT_LABEL = "PIT recall-guarded (deployable)"
_S4_NONPIT_LABEL = (
    "non-PIT recall-enabled DIAGNOSTIC CONTROL — never the deployable portfolio"
)

#: Published-metric fields recomputable from the equity value series alone
#: (avg_turnover needs the trade stream and stays published-only context).
_S4_METRIC_FIELDS = [
    "total_return",
    "annualized_return",
    "annualized_vol",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "crisis_return",
    "crisis_max_drawdown",
]

#: Relative tolerance of the S4 published-vs-re-derived metric checks. On the
#: full 2014-2024 series every metric reproduces at rel err <= 1.7e-6 (the
#: loosest being the pit crisis_return, whose published figure came from a
#: separate pipeline run than the released equity parquet); 1e-5 covers that
#: float-level slack while staying far inside display precision.
_S4_METRIC_TOL = 1e-5

#: Two-line framing for S4 (R5.2, R7.3) — displayed with the sheet.
S4_FRAMING = (
    "Two lines over the same rebalance stream: the PIT recall-guarded line "
    "is the deployable portfolio; the recall-enabled line is a DIAGNOSTIC "
    "CONTROL that exists to measure recall, not to deploy — it is never the "
    "deployable or recommended portfolio. Near-coincident equity curves are "
    "the expected honest outcome of guarding, not a shortfall to close. "
    "No forecast-accuracy claim is made."
)


def _decision_detail(log: dict) -> pd.DataFrame:
    """Per-rebalance-date guard detail joined from one decision log (R5.4)."""
    detail = pd.DataFrame(
        {field: log[field] for field in ("p_memorized", "steered", "parse_ok", "conviction")}
    )
    detail.index = pd.to_datetime(detail.index)
    detail.index.name = "date"
    return detail.sort_index()


def _metrics_row(line: str, label: str, metrics: EquityMetrics, published: dict) -> dict:
    """One head-to-head row: re-derived metrics beside the published track (R5.3)."""
    row: dict = {"line": line, "label": label}
    for field in _S4_METRIC_FIELDS:
        row[f"{field}_rederived"] = getattr(metrics, field)
        row[f"{field}_published"] = published[field]
    row["avg_turnover_published"] = published["avg_turnover"]
    return row


def build_s4(client: ReleaseClient) -> StepView:
    """Assemble the S4 two-line walk-forward simulation view (R5.1-5.4).

    Both lines — the PIT recall-guarded deployable line and the recall-enabled
    diagnostic control — over the same stream: the joined equity curves
    (``equity``), per-line target weights (``targets_*``), and per-date guard
    detail joined from the decision logs — memorization score, steered flag,
    parse status, conviction (``detail_*``, R5.4). The ``metrics`` table
    carries one row per line with the metrics recomputed from the loaded
    equity series beside the published reference-track figures (R5.3), each
    equity-derivable figure attached as a comparison check at the documented
    ``_S4_METRIC_TOL`` — on the full release every check passes (the metrics
    mirror the producing vectorbt convention, see ``rederive.equity_metrics``);
    on a row-subset load the disagreement is a rendered flag, never an
    exception (R7.2). The
    labeling rule is hard (R5.2): the PIT line is labeled deployable and the
    diagnostic marker never appears on it.

    Args:
        client: Release client for the pinned data version.

    Returns:
        The typed S4 view model with the mandated framing.
    """
    summary, _ = load_json(client, "factor_contrast_summary_v1")
    tables: dict[str, pd.DataFrame] = {}
    checks: list[Check] = []
    equity: dict[str, pd.Series] = {}
    metrics_rows: list[dict] = []
    for line, prefix, label, published in (
        ("pit", "factor", _S4_PIT_LABEL, summary["pit_metrics"]),
        ("nonpit", "factor_nonpit_diagnostic", _S4_NONPIT_LABEL, summary["nonpit_metrics"]),
    ):
        values, _ = load_frame(client, f"{prefix}_equity_v1")
        targets, _ = load_frame(client, f"{prefix}_targets_v1")
        log, _ = load_json(client, f"{prefix}_decision_log_v1")
        equity[f"value_{line}"] = values["value"]
        tables[f"targets_{line}"] = targets
        tables[f"detail_{line}"] = _decision_detail(log)
        rederived = equity_metrics(values["value"])
        metrics_rows.append(_metrics_row(line, label, rederived, published))
        checks.extend(
            compare(
                f"S4 {line} {field} vs published",
                published[field],
                getattr(rederived, field),
                tol=_S4_METRIC_TOL,
            )
            for field in _S4_METRIC_FIELDS
        )
    tables["equity"] = pd.DataFrame(equity)
    tables["metrics"] = pd.DataFrame(metrics_rows)
    return StepView(
        title="S4 — Two-line walk-forward simulation",
        framing=S4_FRAMING,
        tables=tables,
        checks=checks,
    )


#: Published->re-derived field mapping of the S5 Sharpe-stability table (R6.2):
#: ``factor_luck_vs_skill_v1`` columns to vendored ``SSRResult`` attributes
#: (``ssr_inference(...).result``).
_S5_SSR_FIELDS = {
    "n_obs": "n_obs",
    "n_rolling": "n_rolling",
    "sharpe": "sr_full",
    "mean_rolling_sr": "mean_rolling_sr",
    "ssr": "ssr",
    "nw_sigma_hac": "sigma_hac",
    "nw_bandwidth_L": "L_hac",
}

#: Relative tolerance of the S5 differential-row checks. The producer (nb14)
#: built the differential from live portfolio values; the released PIT equity
#: parquet reproduces those figures at ~1e-7 relative — and the differential
#: is a near-zero difference of two almost-identical lines, so that
#: float-level slack amplifies (catastrophic cancellation) to ~1.5e-4
#: relative on the differential SSR figures. 1e-3 covers it while staying far
#: inside the 4-decimal display precision; the pit/nonpit rows reproduce at
#: <= 2.1e-7 and keep the default 1e-6.
_S5_DIFF_TOL = 1e-3

#: Luck-vs-skill conclusion framing for S5 (R6.3, R7.3) — the recorded terms.
S5_FRAMING = (
    "Luck versus skill: the contamination premium is +0.400 (paired Cohen's "
    "d = 1.37) in MEMORY, while the head-to-head premium is statistically "
    "~0 in P&L (total-return differential 0.174 over the sample, not "
    "separable from zero). The "
    "return differential's SSR = "
    "0.11 is NOT distinguishable from zero under the paper's one-sided "
    "moving-block-bootstrap test (p = 0.056, marginally above the 0.05 "
    "threshold — the P&L reading is weak evidence, not a clean null): "
    "the recall premium is "
    "LUCK-COMPATIBLE, not skill. Any excess of the recall-enabled line is "
    "LOOKAHEAD/RECALL BIAS, never attainable skill, and the diagnostic line "
    "is never deployable. No forecast-accuracy claim is made."
)


def _total_return(value: pd.Series) -> float:
    """Final over initial value minus one; NaN on an empty series."""
    return float(value.iloc[-1] / value.iloc[0] - 1.0) if len(value) else float("nan")


def build_s5(client: ReleaseClient) -> StepView:
    """Assemble the S5 luck-versus-skill view (R6.1-6.3).

    The paired per-date memorization contrast verbatim (``contrast``) with the
    contamination premium and paired effect size re-derived from those records
    beside the published summary (``premium``, R6.1). The Sharpe-stability
    table (``ssr``) carries the three published ``factor_luck_vs_skill_v1``
    rows — deployable line, diagnostic line, return differential, including
    the Newey-West long-run variance treatment — next to the vendored
    ``ssr_inference`` re-derivation over the loaded equity series, sliced at
    the first rebalance date exactly as the producer built them, with the
    differential as the date-aligned non-PIT-minus-PIT daily returns (R6.2).
    Per-line SSR and total-return checks compare re-derived against published
    at the documented tolerances; on fixture-sized subsets the disagreement
    is a rendered flag, never an exception (R7.2). SSR is only the effect
    size, so the verdict comes from the paper's one-sided moving-block
    bootstrap: the MBB p-value, its block length, and the rendered verdict
    string sit beside the published ones, the p-value is checked against the
    published ``mbb_p`` where the release carries it (data-v2 predates those
    columns), and the framing's "not distinguishable from zero" sentence is
    re-derived as its own check row. The PIT-vs-non-PIT
    loading-stability comparison (``loading_stability``, research.md §5) is
    re-derived from the loaded loadings tables. The framing states the
    recorded conclusion: luck-compatible, lookahead/recall bias, never
    attainable skill (R6.3).

    Args:
        client: Release client for the pinned data version.

    Returns:
        The typed S5 view model with the mandated conclusion wording.
    """
    contrast, _ = load_frame(client, "factor_contrast_v1")
    summary, _ = load_json(client, "factor_contrast_summary_v1")
    published_ssr, _ = load_frame(client, "factor_luck_vs_skill_v1")
    tables: dict[str, pd.DataFrame] = {"contrast": contrast}
    checks: list[Check] = []

    premium = contamination_premium(contrast["pit_p"], contrast["nonpit_p"])
    published_premium = summary["contamination_premium"]
    tables["premium"] = pd.DataFrame(
        [
            {
                "n_pairs_published": summary["n_pairs"],
                "n_pairs_rederived": premium.n_pairs,
                "mean_delta_published": published_premium["p_memorized_mean_delta"],
                "mean_delta_rederived": premium.mean_delta,
                "median_delta_published": published_premium["p_memorized_median_delta"],
                "median_delta_rederived": premium.median_delta,
                "paired_d_published": published_premium["p_memorized_paired_d"],
                "paired_d_rederived": premium.paired_d,
            }
        ]
    )
    checks.append(
        compare("S5 premium n_pairs vs published", summary["n_pairs"], premium.n_pairs)
    )
    for field, published_value in (
        ("mean_delta", published_premium["p_memorized_mean_delta"]),
        ("median_delta", published_premium["p_memorized_median_delta"]),
        ("paired_d", published_premium["p_memorized_paired_d"]),
    ):
        checks.append(
            compare(
                f"S5 premium {field} vs published",
                published_value,
                getattr(premium, field),
            )
        )

    # nb14 sliced the equity at the first rebalance date before pct_change
    # (the equity parquets carry a flat pre-simulation stub from 2014); the
    # contrast index carries exactly those rebalance dates.
    sim_start = contrast.index.min()
    returns: dict[str, pd.Series] = {}
    total_return: dict[str, float] = {}
    for line, prefix in (("pit", "factor"), ("nonpit", "factor_nonpit_diagnostic")):
        values, _ = load_frame(client, f"{prefix}_equity_v1")
        returns[line] = values["value"].loc[sim_start:].pct_change().dropna()
        total_return[line] = _total_return(values["value"])
    # date-aligned subtraction == nb14's intersection-indexed construction
    returns["differential"] = (returns["nonpit"] - returns["pit"]).dropna()
    total_return["differential"] = total_return["nonpit"] - total_return["pit"]

    ssr_rows: list[dict] = []
    has_published_mbb = "mbb_p" in published_ssr.columns  # absent on data-v2
    for i, line in enumerate(("pit", "nonpit", "differential")):
        published = published_ssr.iloc[i]
        differential = line == "differential"
        inference = ssr_inference(returns[line])
        result = inference.result
        row: dict = {"line": published_ssr.index[i]}
        for column, attr in _S5_SSR_FIELDS.items():
            row[f"{column}_published"] = published[column]
            row[f"{column}_rederived"] = getattr(result, attr)
        row["nw_long_run_var_published"] = published["nw_long_run_var"]
        row["nw_long_run_var_rederived"] = float(result.sigma_hac) ** 2
        row["total_return_published"] = published["total_return"]
        row["total_return_rederived"] = total_return[line]
        row["mbb_p_published"] = published.get("mbb_p")
        row["mbb_p_rederived"] = inference.p_value
        row["mbb_block_published"] = published.get("mbb_block")
        row["mbb_block_rederived"] = inference.block_len
        row["verdict"] = published["verdict"]
        row["verdict_rederived"] = inference.verdict(differential=differential)
        ssr_rows.append(row)
        tol = _S5_DIFF_TOL if differential else 1e-6
        checks.append(
            compare(f"S5 {line} ssr vs published", published["ssr"], result.ssr, tol=tol)
        )
        checks.append(
            compare(
                f"S5 {line} total_return vs published",
                published["total_return"],
                total_return[line],
                tol=tol,
            )
        )
        if has_published_mbb:
            # a bootstrap p is a count/n_boot: exact reproduction or nothing
            checks.append(
                compare(
                    f"S5 {line} mbb_p vs published",
                    published["mbb_p"],
                    inference.p_value,
                )
            )
        if differential:
            checks.append(
                _claim_check(
                    "S5 framing claim 'differential NOT distinguishable from zero' "
                    "(1 = holds)",
                    not (inference.stable or inference.stably_below),
                    inference,
                )
            )
    tables["ssr"] = pd.DataFrame(ssr_rows)

    stability_rows: list[dict] = []
    for line, asset in (
        ("pit", "factor_loadings_v1"),
        ("nonpit", "factor_nonpit_diagnostic_loadings_v1"),
    ):
        loadings, _ = load_frame(client, asset)
        rederived = loading_stability(loadings[_AXES], loadings["parse_ok"])
        stability_rows.append(
            {
                "line": line,
                "mean_std": rederived["mean_std"],
                "mean_mac": rederived["mean_mac"],
            }
        )
    tables["loading_stability"] = pd.DataFrame(stability_rows)

    return StepView(
        title="S5 — Luck versus skill (contamination premium vs robust inference)",
        framing=S5_FRAMING,
        tables=tables,
        checks=checks,
    )


# --------------------------------------------------------------------------- #
# V4 — corrected canonical data-v4 tables (task 10.5)                          #
#                                                                              #
# The corrected canonical release is consumed VERBATIM: every displayed value  #
# is the published canonical figure, loaded through the tag-bound data-v4      #
# contracts — this step performs no alternative local calculation. The checks  #
# re-derive representative rows through the vendored root implementations      #
# (cash-excess SSR from portfolio returns minus the aligned BIL total          #
# returns, boundary-anchored crisis values, the differential spread) and flag  #
# any disagreement. Missing cash, shortened attribution, incomplete windows,   #
# schema violations, and manifest failures render as visible check rows,       #
# never silent substitutions. Historical tags keep the immutable S0-S5 audit;  #
# a non-data-v4 client is refused here exactly as a data-v4 client is refused  #
# by the historical loaders.                                                   #
# --------------------------------------------------------------------------- #

#: Framing for the V4 canonical-tables step (R7.3) — displayed with the sheet.
V4_FRAMING = (
    "V4 — the corrected canonical data-v4 tables, consumed verbatim under the "
    "publication manifest: reader (252-trading-day, cash-excess Sharpe/SSR), "
    "legacy vectorbt/365, differential, raw market-model attribution, "
    "boundary-anchored crisis, Factor, SJM v3, and Markowitz USD tables. "
    "Every displayed figure is the published canonical value — no local "
    "alternative calculation. The checks re-derive representative rows with "
    "the vendored root implementations: the SSR is reconstructed from "
    "portfolio returns minus the aligned BIL total returns with all "
    "deterministic inference settings surfaced, crisis values are re-derived "
    "boundary-anchored, and the Factor differential is rebuilt from the "
    "released equity series. Missing cash sessions, shortened attribution, "
    "incomplete windows, schema violations, and manifest failures render as "
    "visible check rows. The S0-S5 storyboard remains the immutable data-v2 "
    "historical audit. No forecast-accuracy claim is made."
)

#: View table name -> data-v4 registry key; every canonical family verbatim.
_V4_TABLE_KEYS: dict[str, str] = {
    "reader": "portfolio_metrics_reader_ext2026",
    "legacy": "portfolio_metrics_vectorbt365_ext2026",
    "differential": "portfolio_metrics_differential_ext2026",
    "attribution": "attribution_raw_market_model_ext2026",
    "crisis": "crisis_metrics_ext2026",
    "tear_sheet_ai_variants": "tear_sheet_ai_variants_ext2026",
    "tear_sheet_sjm": "tear_sheet_sjm_crowding_ext2026",
    "tear_sheet_trio": "tear_sheet_trio_ext2026",
    "monthly_returns": "monthly_returns_ext2026",
    "risk_decomposition": "risk_decomposition_ext2026",
    "markowitz_10y_moments": "markowitz_10y_moments",
    "markowitz_10y_frontier": "markowitz_10y_frontier",
    "markowitz_max_moments": "markowitz_max_moments",
    "markowitz_max_frontier": "markowitz_max_frontier",
    "factor_equity": "factor_equity_ext2026",
    "factor_targets": "factor_targets_ext2026",
    "factor_nonpit_equity": "factor_nonpit_diagnostic_equity_ext2026",
    "factor_nonpit_targets": "factor_nonpit_diagnostic_targets_ext2026",
    "sjm_equity": "sjm_crowding_v3_total_return_bil_equity_ext2026",
    "sjm_targets": "sjm_crowding_v3_total_return_bil_targets_ext2026",
    "sjm_daily_returns": "sjm_crowding_v3_total_return_bil_daily_returns_ext2026",
    "sjm_control_returns": "sjm_crowding_v3_total_return_bil_control_returns_ext2026",
}

#: Frozen producer identities (scripts/build_tear_sheet.py, sjm producer).
_V4_FACTOR_PIT = "factor_pit_ext2026"
_V4_FACTOR_NONPIT = "factor_nonpit_diagnostic_ext2026"
_V4_FACTOR_DIFFERENTIAL = "factor_nonpit_minus_pit_ext2026"
_V4_SJM_OVERLAY_BASIS = "sjm_v3_overlay_anchored_equity"
_V4_SJM_CONTROL_BASIS = "sjm_v3_control_anchored_equity"

#: The pinned deterministic inference settings every canonical row records.
_V4_SSR_SETTINGS = dict(SSR_REPORT_DEFAULTS)

#: The full published ``ssr_*`` vocabulary, projected from the vendored
#: (byte-identical to root) SSR dataclasses — the one shared authority.
_V4_SSR_RESULT_FIELDS = frozenset(f.name for f in dataclasses.fields(SSRResult))
_V4_SSR_FIELDS = tuple(f.name for f in dataclasses.fields(SSRResult)) + tuple(
    f.name for f in dataclasses.fields(SSRInference) if f.name != "result"
)

#: Published crisis column -> re-derived ``EquityMetrics`` attribute. The
#: volatility comparison assumes the canonical 252-day crisis convention; a
#: row on any other basis renders as a visible disagreement, by design.
_V4_CRISIS_VALUE_FIELDS = {
    "episode_return": "crisis_return",
    "boundary_anchored_max_drawdown": "crisis_max_drawdown",
    "volatility_ann": "crisis_vol_ann",
}

#: Tables that may carry reader rows (attribution-coverage scan).
_V4_READER_BEARING = ("reader", "tear_sheet_ai_variants", "tear_sheet_sjm", "tear_sheet_trio")


def _v4_status(name: str, ok: bool, detail: str = "") -> Check:
    """A visible pass/fail condition as a check row (1 = holds), never an
    exception: the failure detail travels in the rendered message (R7.2)."""
    ok = bool(ok)
    return Check(name, 1.0, 1.0 if ok else 0.0, 0.0, ok, "" if ok else f"{name}: {detail}")


def _v4_ssr_value(inference: SSRInference, name: str) -> float:
    return getattr(inference.result if name in _V4_SSR_RESULT_FIELDS else inference, name)


def _v4_inference_row(line: str, inference: SSRInference, *, differential: bool = False) -> dict:
    """One re-derived inference row surfacing ALL deterministic metadata under
    the published ``ssr_*`` column vocabulary, plus the canonical verdict."""
    row: dict = {"line": line}
    row.update({f"ssr_{name}": _v4_ssr_value(inference, name) for name in _V4_SSR_FIELDS})
    row["verdict"] = inference.verdict(differential=differential)
    return row


def _v4_ssr_comparisons(label: str, published_row: pd.Series, inference: SSRInference) -> list[Check]:
    """Published ``ssr_*`` fields against the vendored re-derivation, one check
    per field (bootstrap p-values reproduce exactly: same seed, same draws)."""
    return [
        compare(
            f"{label} ssr_{name} vs published",
            published_row[f"ssr_{name}"],
            _v4_ssr_value(inference, name),
        )
        for name in _V4_SSR_FIELDS
    ]


def _v4_excess_checks(
    line: str,
    returns: pd.Series,
    cash: pd.Series,
    published_row: pd.Series,
    inference_rows: list[dict],
) -> list[Check]:
    """Cash-excess SSR for one line: portfolio returns minus the aligned BIL
    total returns over the published window. A session without a cash return
    is a visible failed coverage check and the SSR is NOT constructed from
    substitute data (R2.1, R4.2)."""
    window = returns.loc[published_row["start"] : published_row["end"]]
    cash_slice = cash.reindex(window.index)
    n_missing = int(cash_slice.isna().sum())
    checks = [
        _v4_status(
            f"V4 {line} cash coverage (1 = every session has an aligned BIL total return)",
            n_missing == 0,
            f"{n_missing} of {len(window)} sessions lack an aligned BIL total return",
        )
    ]
    if n_missing:
        return checks
    inference = ssr_inference(window - cash_slice, **_V4_SSR_SETTINGS)
    inference_rows.append(_v4_inference_row(line, inference))
    checks += _v4_ssr_comparisons(f"V4 {line}", published_row, inference)
    checks.append(
        compare(
            f"V4 {line} total_return vs published",
            published_row["total_return"],
            float((1.0 + window).prod() - 1.0),
        )
    )
    return checks


def _v4_crisis_checks(
    line: str, value: pd.Series, rows: pd.DataFrame, *, required: bool = True
) -> list[Check]:
    """Boundary-anchored crisis re-derivation for one published crisis row.

    Values re-derive through ``rederive.equity_metrics`` (task 10.4 parity
    with the root ``crisis_metrics``); the boundary check re-derives anchor,
    first return, actual end, and count from the released series itself, so a
    published crisis window the series does not support — including a wrongly
    shortened (incomplete) one — goes red instead of passing on trust.
    """
    if len(rows) == 0 and not required:
        return []
    if len(rows) != 1:
        return [
            _v4_status(
                f"V4 {line} crisis row (1 = exactly one published row)",
                False,
                f"found {len(rows)} rows",
            )
        ]
    row = rows.iloc[0]
    metrics = equity_metrics(value, crisis=(row["requested_start"], row["requested_end"]))
    checks = [
        compare(f"V4 {line} crisis {column} vs published", row[column], getattr(metrics, attr))
        for column, attr in _V4_CRISIS_VALUE_FIELDS.items()
    ]
    anchors = value.index[value.index < row["requested_start"]]
    window = value.index[
        (value.index >= row["requested_start"]) & (value.index <= row["requested_end"])
    ]
    mismatches: list[str] = []
    if not len(anchors) or not len(window):
        mismatches.append("released series does not span the requested window")
    else:
        for name, published, rederived in (
            ("anchor", row["anchor"], anchors[-1]),
            ("first_return_date", row["first_return_date"], window[0]),
            ("actual_end", row["actual_end"], window[-1]),
            ("n_returns", row["n_returns"], len(window)),
        ):
            if published != rederived:
                mismatches.append(f"{name}: published {published!r} != released-series {rederived!r}")
    checks.append(
        _v4_status(
            f"V4 {line} crisis boundary (1 = anchor, first return, actual end, and "
            "count match the released series)",
            not mismatches,
            "; ".join(mismatches),
        )
    )
    return checks


def _v4_sjm_checks(loaded: dict[str, pd.DataFrame], inference_rows: list[dict]) -> list[Check]:
    """SJM overlay + control re-derivations against the published tear sheet."""
    missing = [
        name
        for name in ("sjm_equity", "sjm_daily_returns", "sjm_control_returns", "tear_sheet_sjm")
        if name not in loaded
    ]
    checks = [
        _v4_status(
            "V4 sjm re-derivation inputs (1 = equity, daily returns, control returns, "
            "and tear sheet loaded)",
            not missing,
            f"missing tables: {', '.join(missing)}",
        )
    ]
    if missing:
        return checks
    value = loaded["sjm_equity"]["value"]
    cash = loaded["sjm_daily_returns"]["cash_return"]
    sheet = loaded["tear_sheet_sjm"]
    # the control line ships as returns; the anchored curve mirrors the SJM
    # producer's own reconstruction (1.0 on the overlay's first session)
    control_value = pd.concat(
        [
            pd.Series([1.0], index=pd.DatetimeIndex([value.index[0]])),
            (1.0 + loaded["sjm_control_returns"]["control_return"]).cumprod(),
        ]
    )
    readers = sheet[sheet["schema"] == READER_SCHEMA]
    for line, basis, curve in (
        ("sjm overlay", _V4_SJM_OVERLAY_BASIS, value),
        ("sjm control", _V4_SJM_CONTROL_BASIS, control_value),
    ):
        rows = readers[
            (readers["return_basis"] == basis)
            & readers["window_label"].astype(str).str.startswith("full")
        ]
        if len(rows) != 1:
            checks.append(
                _v4_status(
                    f"V4 {line} full reader row (1 = exactly one published row)",
                    False,
                    f"found {len(rows)} rows",
                )
            )
            continue
        checks += _v4_excess_checks(
            line, curve.pct_change().dropna(), cash, rows.iloc[0], inference_rows
        )
    checks += _v4_crisis_checks(
        "sjm overlay",
        value,
        sheet[
            (sheet["schema"] == CRISIS_SCHEMA)
            & (sheet["return_basis"] == _V4_SJM_OVERLAY_BASIS)
        ],
    )
    return checks


def _v4_factor_checks(loaded: dict[str, pd.DataFrame], inference_rows: list[dict]) -> list[Check]:
    """Factor reader endpoints plus the differential spread re-derivation."""
    missing = [
        name
        for name in ("reader", "differential", "factor_equity", "factor_nonpit_equity")
        if name not in loaded
    ]
    checks = [
        _v4_status(
            "V4 factor re-derivation inputs (1 = reader, differential, and both "
            "equity series loaded)",
            not missing,
            f"missing tables: {', '.join(missing)}",
        )
    ]
    if missing:
        return checks
    pit_returns = loaded["factor_equity"]["value"].pct_change().dropna()
    nonpit_returns = loaded["factor_nonpit_equity"]["value"].pct_change().dropna()
    reader = loaded["reader"]
    for pid, returns in ((_V4_FACTOR_PIT, pit_returns), (_V4_FACTOR_NONPIT, nonpit_returns)):
        rows = reader[(reader["schema"] == READER_SCHEMA) & (reader["portfolio_id"] == pid)]
        if len(rows) != 1:
            checks.append(
                _v4_status(
                    f"V4 factor {pid} reader row (1 = exactly one published row)",
                    False,
                    f"found {len(rows)} rows",
                )
            )
            continue
        row = rows.iloc[0]
        window = returns.loc[row["start"] : row["end"]]
        checks.append(compare(f"V4 factor {pid} n_obs vs released series", row["n_obs"], len(window)))
        checks.append(
            compare(
                f"V4 factor {pid} total_return vs released series",
                row["total_return"],
                float((1.0 + window).prod() - 1.0),
            )
        )
    rows = loaded["differential"][
        loaded["differential"]["portfolio_id"] == _V4_FACTOR_DIFFERENTIAL
    ]
    if len(rows) != 1:
        checks.append(
            _v4_status(
                "V4 factor differential row (1 = exactly one published row)",
                False,
                f"found {len(rows)} rows",
            )
        )
        return checks
    row = rows.iloc[0]
    comparison = nonpit_returns.loc[row["start"] : row["end"]]
    reference = pit_returns.loc[row["start"] : row["end"]]
    aligned = comparison.index.equals(reference.index)
    checks.append(
        _v4_status(
            "V4 factor differential session alignment (1 = comparison and reference "
            "share every session)",
            aligned,
            "the released equity series disagree on sessions inside the published window",
        )
    )
    if not aligned:
        return checks
    spread = comparison - reference
    inference = ssr_inference(spread, **_V4_SSR_SETTINGS)
    inference_rows.append(_v4_inference_row("factor differential", inference, differential=True))
    checks += _v4_ssr_comparisons("V4 factor differential", row, inference)
    checks.append(
        compare(
            "V4 factor differential total_return vs published",
            row["total_return"],
            float((1.0 + spread).prod() - 1.0),
        )
    )
    checks.append(
        compare(
            "V4 factor differential endpoint_total_return_difference vs published",
            row["endpoint_total_return_difference"],
            float((1.0 + comparison).prod() - (1.0 + reference).prod()),
        )
    )
    return checks


def _v4_factor_crisis_checks(loaded: dict[str, pd.DataFrame]) -> list[Check]:
    """Crisis rows of the standalone table re-derived from the released Factor
    equity series (rows for portfolios without a released series pass through
    verbatim — nothing is recomputed from substitute data)."""
    crisis = loaded.get("crisis")
    if crisis is None:
        return []
    checks: list[Check] = []
    for pid, table in ((_V4_FACTOR_PIT, "factor_equity"), (_V4_FACTOR_NONPIT, "factor_nonpit_equity")):
        if table not in loaded:
            continue
        checks += _v4_crisis_checks(
            f"factor {pid}",
            loaded[table]["value"],
            crisis[crisis["portfolio_id"] == pid],
            required=False,
        )
    return checks


def _v4_attribution_coverage(loaded: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[Check]]:
    """Shortened attribution surfaced (R3.6, R3.7): every ``performance_only``
    reader row must have its attribution emitted as a SEPARATE record; the
    coverage table shows inline/separate/missing per published reader row."""
    separate: set = set()
    for name in (*_V4_READER_BEARING, "attribution"):
        frame = loaded.get(name)
        if frame is not None and "schema" in frame.columns:
            separate |= set(frame.loc[frame["schema"] == ATTRIBUTION_SCHEMA, "portfolio_id"])
    rows: list[dict] = []
    uncovered: list[str] = []
    for name in _V4_READER_BEARING:
        frame = loaded.get(name)
        if frame is None or "row_kind" not in frame.columns:
            continue
        for _, row in frame[frame["schema"] == READER_SCHEMA].iterrows():
            if row["row_kind"] == "performance_only":
                attribution = "separate_record" if row["portfolio_id"] in separate else "missing"
                if attribution == "missing":
                    uncovered.append(f"{name}:{row['portfolio_id']}")
            else:
                attribution = "inline_full_window"
            rows.append(
                {
                    "table": name,
                    "portfolio_id": row["portfolio_id"],
                    "window_label": row["window_label"],
                    "row_kind": row["row_kind"],
                    "attribution": attribution,
                }
            )
    table = pd.DataFrame(
        rows, columns=["table", "portfolio_id", "window_label", "row_kind", "attribution"]
    )
    check = _v4_status(
        "V4 attribution coverage (1 = every performance_only reader row has a "
        "separate attribution record)",
        not uncovered,
        f"no shortened-attribution record for: {', '.join(uncovered)}",
    )
    return table, [check]


_V4_COVERAGE_COLUMNS = [
    "kind", "table", "identity",
    "requested_start", "requested_end", "actual_start", "actual_end", "incomplete",
]


def _v4_window_coverage(loaded: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Requested-vs-actual window disclosure: crisis and Markowitz rows with an
    explicit ``incomplete`` flag wherever the actual coverage falls short of
    the requested window — disclosed, never fabricated."""
    rows: list[dict] = []
    for name in ("crisis", "tear_sheet_sjm"):
        frame = loaded.get(name)
        if frame is None or "requested_end" not in frame.columns:
            continue
        for _, row in frame[frame["schema"] == CRISIS_SCHEMA].iterrows():
            rows.append(
                {
                    "kind": "crisis",
                    "table": name,
                    "identity": row["portfolio_id"],
                    "requested_start": row["requested_start"],
                    "requested_end": row["requested_end"],
                    "actual_start": row["first_return_date"],
                    "actual_end": row["actual_end"],
                    "incomplete": bool(row["actual_end"] < row["requested_end"]),
                }
            )
    for name in (
        "markowitz_10y_moments",
        "markowitz_10y_frontier",
        "markowitz_max_moments",
        "markowitz_max_frontier",
    ):
        frame = loaded.get(name)
        if frame is None:
            continue
        for _, row in frame.iterrows():
            rows.append(
                {
                    "kind": "markowitz",
                    "table": name,
                    "identity": f"{row['window']}:{row.get('asset', 'frontier')}",
                    "requested_start": row["requested_start"],
                    "requested_end": row["requested_end"],
                    "actual_start": row["actual_start"],
                    "actual_end": row["actual_end"],
                    "incomplete": bool(
                        row["actual_start"] > row["requested_start"]
                        or row["actual_end"] < row["requested_end"]
                    ),
                }
            )
    return pd.DataFrame(rows, columns=_V4_COVERAGE_COLUMNS)


def _v4_manifest_check(client: ReleaseClient) -> Check:
    """Manifest status surfaced: every loaded asset must have passed the
    publication-manifest SHA-256 verification recorded in its provenance."""
    name = "V4 manifest verification (1 = every loaded asset is manifest-verified)"
    provenance_table = getattr(client, "provenance_table", None)
    if provenance_table is None:
        return _v4_status(name, False, "client exposes no provenance table")
    records = provenance_table()
    unverified = [
        record.asset
        for record in records
        if not (record.verified and record.verification == "publication_manifest_sha256")
    ]
    detail = (
        "no asset loaded"
        if not records
        else f"unverified assets: {', '.join(unverified)}"
    )
    return _v4_status(name, bool(records) and not unverified, detail)


def build_v4(client: ReleaseClient) -> StepView:
    """Assemble the corrected canonical data-v4 view (task 10.5).

    Loads every canonical table verbatim through the tag-bound data-v4
    contracts — one visible load check per asset, so missing, stale, corrupt,
    cross-tag, unmanifested, or schema-violating assets each surface as a
    named failed check rather than an exception or substitute data (R7.2,
    R7.4, R8.6). On the loaded tables the step re-derives representative
    published rows with the vendored root implementations: the SJM overlay
    and control cash-excess SSR (portfolio returns minus the aligned BIL
    total returns, all deterministic inference metadata surfaced in the
    ``inference`` table), boundary-anchored crisis values and boundaries, the
    Factor reader endpoints, and the Factor differential spread. Shortened
    attribution and requested-vs-actual window coverage are surfaced in their
    own tables with explicit flags. Historical tags are refused: the S0-S5
    storyboard remains the immutable data-v2 audit (R7.3).

    Args:
        client: Release client BOUND to the ``data-v4`` tag.

    Returns:
        The typed V4 view model with the mandated framing.

    Raises:
        SchemaError: The client is bound to a historical tag — cross-tag
            substitution is prohibited in both directions.
    """
    tag = getattr(client, "tag", None)
    if tag != DATA_V4_TAG:
        raise SchemaError(
            f"the V4 step consumes the corrected canonical {DATA_V4_TAG!r} release; "
            f"the client is bound to {tag!r} — historical tags keep the immutable "
            "S0-S5 audit"
        )
    tables: dict[str, pd.DataFrame] = {}
    checks: list[Check] = []
    loaded: dict[str, pd.DataFrame] = {}
    for name, key in _V4_TABLE_KEYS.items():
        try:
            frame, _ = load_v4_frame(client, key)
        except (ReleaseError, SchemaError) as exc:
            checks.append(_v4_status(f"V4 load {key} (1 = loaded and validated)", False, str(exc)))
        else:
            loaded[name] = tables[name] = frame
            checks.append(_v4_status(f"V4 load {key} (1 = loaded and validated)", True))
    checks.append(_v4_manifest_check(client))
    inference_rows: list[dict] = []
    checks += _v4_sjm_checks(loaded, inference_rows)
    checks += _v4_factor_checks(loaded, inference_rows)
    checks += _v4_factor_crisis_checks(loaded)
    coverage_table, coverage_checks = _v4_attribution_coverage(loaded)
    tables["attribution_coverage"] = coverage_table
    checks += coverage_checks
    tables["window_coverage"] = _v4_window_coverage(loaded)
    tables["inference"] = pd.DataFrame(
        inference_rows,
        columns=["line", *(f"ssr_{name}" for name in _V4_SSR_FIELDS), "verdict"],
    )
    return StepView(
        title="V4 — Corrected canonical tables (data-v4)",
        framing=V4_FRAMING,
        tables=tables,
        checks=checks,
    )
