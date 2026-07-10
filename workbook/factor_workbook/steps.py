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

from dataclasses import dataclass

import pandas as pd

from factor_workbook import certification
from factor_workbook.contract import load_frame, load_json
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
from factor_workbook.vendored_ssr import compute_ssr
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
#: (rederive.equity_metrics is the exact producer — build_static_bh.py used it
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

#: Published static_bh_ssr fields — the vendored ``compute_ssr`` attributes
#: carry the same names (same producer, exact full-data agreement).
_S0_SSR_FIELDS = ["ssr", "mean_rolling_sr", "sigma_hac", "L_hac", "n_rolling"]

#: Per-episode crisis fields re-derived via ``equity_metrics(value, crisis=...)``.
_S0_CRISIS_FIELDS = ["crisis_return", "crisis_max_drawdown", "crisis_vol_ann"]

#: (stats window key, registry key, view table name) — 10y opener first.
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
    "artifact, never attainable skill. The line's own SSR of 0.147 is far "
    "below 1.96: LUCK-COMPATIBLE, never attainable skill. This is the "
    "problem the following steps S1-S5 measure. No forecast-accuracy claim "
    "is made."
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
    (vendored ``compute_ssr`` over daily returns), and the per-episode
    crisis figures; on fixture-sized subsets the disagreements are rendered
    flags, never exceptions (R7.2). On ``data-v1`` the assets are absent and
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
        ssr = compute_ssr(value.pct_change().dropna())
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
    "refinement was rejected by the accept-gate on contamination (mean "
    "p_memorized 0.2709 vs 0.2361) even though every performance check "
    "passed; both prompt versions' data are preserved as alternatives. "
    "No forecast-accuracy claim is made."
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
#: ``factor_luck_vs_skill_v1`` columns to vendored ``SSRResult`` attributes.
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
    "Luck versus skill: the contamination premium is +0.528 (paired Cohen's "
    "d = 1.93) in MEMORY, while the head-to-head premium is ~0 in P&L "
    "(total-return differential 0.028). The return differential's SSR = "
    "0.002 under Newey-West HAC inference is statistically indistinguishable "
    "from zero: the recall premium is LUCK-COMPATIBLE, not skill. Any excess "
    "of the recall-enabled line is LOOKAHEAD/RECALL BIAS, never attainable "
    "skill, and the diagnostic line is never deployable. No "
    "forecast-accuracy claim is made."
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
    ``compute_ssr`` re-derivation over the loaded equity series, sliced at
    the first rebalance date exactly as the producer built them, with the
    differential as the date-aligned non-PIT-minus-PIT daily returns (R6.2).
    Per-line SSR and total-return checks compare re-derived against published
    at the documented tolerances; on fixture-sized subsets the disagreement
    is a rendered flag, never an exception (R7.2). The PIT-vs-non-PIT
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
    for i, line in enumerate(("pit", "nonpit", "differential")):
        published = published_ssr.iloc[i]
        result = compute_ssr(returns[line])
        row: dict = {"line": published_ssr.index[i]}
        for column, attr in _S5_SSR_FIELDS.items():
            row[f"{column}_published"] = published[column]
            row[f"{column}_rederived"] = getattr(result, attr)
        row["nw_long_run_var_published"] = published["nw_long_run_var"]
        row["nw_long_run_var_rederived"] = float(result.sigma_hac) ** 2
        row["total_return_published"] = published["total_return"]
        row["total_return_rederived"] = total_return[line]
        row["verdict"] = published["verdict"]
        ssr_rows.append(row)
        tol = _S5_DIFF_TOL if line == "differential" else 1e-6
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
