"""Storyboard step assembly: typed S1-S5 view models with mandated framing
text and explicit gap markers (R2-R6, R7.3). Implemented in tasks 4.1-4.5.

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
    equity_metrics,
    evidence_class_stats,
    guarded_tilt,
    loading_stability,
    wilson_ci,
)
from factor_workbook.release import ReleaseClient, ReleaseError
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


def _slug(model: str) -> str:
    """Release evidence-directory slug for a candidate model name."""
    return model.replace("/", "_")


def _evidence_consistency(
    client: ReleaseClient, model: str, slug: str, evidence: pd.DataFrame
) -> tuple[pd.DataFrame, list[Check]]:
    """Class-count/feature-statistics consistency vs the published summary (R2.6).

    Always: per evidence arm, the included-row count is compared against the
    published ``n_per_class`` and the ``std_*`` feature mean/std summary rows
    are re-derived. Deeper (statistics extra only): where both arms carry at
    least ``_N_SPLITS`` included rows (>= n_splits*2 usable rows total), the
    point AUC is re-derived and compared against the published
    ``controlled_auc``; fixture-sized subsets skip this gracefully — the
    full-data path runs live (task 6.1).

    Returns:
        The per-arm feature-statistics table and the comparison checks.
    """
    summary, _ = load_json(client, "norecall_screen_evidence_summary", model=slug)
    included = evidence[evidence["included"]]
    stats = evidence_class_stats(included)
    checks = [
        compare(
            f"S1 {model} [{arm}] included rows vs published n_per_class",
            summary["n_per_class"],
            count,
        )
        for arm, count in sorted(stats.arm_counts.items())
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
