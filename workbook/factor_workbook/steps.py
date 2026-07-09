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
from factor_workbook.rederive import evidence_class_stats, wilson_ci
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
