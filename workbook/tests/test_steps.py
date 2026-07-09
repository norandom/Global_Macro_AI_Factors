"""Offline tests for the storyboard step views (tasks 4.1-4.5).

Task 4.1 — the S1 certification view: per-candidate certification table
(R2.1), honest outcome framing (R2.2/R7.3), raw-evidence drill-down tables
(R2.3), both known-gap markers as explicit states — never failures (R2.4,
R2.5) — and the class-count/feature-statistics consistency checks against
the published summary, with the deeper AUC re-derivation only where the
statistics extra is present and both arms carry enough rows (R2.6).
"""

import dataclasses
import io
import json
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from factor_workbook import certification
from factor_workbook.rederive import equity_metrics, loading_stability, wilson_ci
from factor_workbook.release import FetchError, Provenance, ReleaseError
from factor_workbook.steps import (
    S2_FRAMING,
    S3_FRAMING,
    S4_FRAMING,
    S5_FRAMING,
    StepView,
    build_s1,
    build_s2,
    build_s3,
    build_s4,
    build_s5,
)
from factor_workbook.verify import Check

FIXTURES = Path(__file__).parent / "fixtures"
EVIDENCE_ASSET = "norecall_screen_evidence.tar.gz"

FALLBACK = "openai/gpt-oss-20b"
BIG = "openai/gpt-oss-120b"
MAVERICK = "meta/llama-4-maverick-17b-128e-instruct"
PHI4 = "microsoft/phi-4-mini-instruct"
UNSERVABLE = "meta/llama-3.3-70b-instruct"
ALL_CANDIDATES = {FALLBACK, BIG, MAVERICK, PHI4, UNSERVABLE}

FALLBACK_SLUG = "openai_gpt-oss-20b"
BIG_SLUG = "openai_gpt-oss-120b"


class FakeClient:
    """ReleaseClient stand-in over the fixtures with the real error taxonomy.

    Unlike the pared-down fake in test_contract.py, a missing tar member
    raises :class:`ReleaseError` (cause ``unpack``) exactly like the real
    client — the S1 gap detection must key off that, not off AssertionError.
    """

    tag = "data-v1"

    def __init__(self, overrides: dict[str, bytes] | None = None):
        self._overrides = overrides or {}

    def fetch(self, asset: str) -> tuple[bytes, Provenance]:
        data = self._overrides.get(asset)
        if data is None:
            data = (FIXTURES / asset).read_bytes()
        provenance = Provenance(
            tag=self.tag,
            asset=asset,
            url=f"fixture://{asset}",
            fetched_at="2026-07-09T00:00:00+00:00",
            sha256="0" * 64,
            from_cache=False,
        )
        return data, provenance

    def fetch_tar_member(self, asset: str, member: str) -> tuple[bytes, Provenance]:
        data, provenance = self.fetch(asset)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
            try:
                extracted = tar.extractfile(member)
            except KeyError:
                extracted = None
            if extracted is None:
                raise ReleaseError(FetchError(asset, "unpack", f"member {member!r}: not found"))
            return extracted.read(), provenance


@pytest.fixture(scope="module")
def s1() -> StepView:
    """One S1 view over the fixtures for all read-only assertions."""
    return build_s1(FakeClient())


def _row(view: StepView, model: str) -> pd.Series:
    table = view.tables["certification"]
    matches = table[table["model"] == model]
    assert len(matches) == 1, f"expected exactly one row for {model}"
    return matches.iloc[0]


def _fixture_evidence(slug: str) -> pd.DataFrame:
    with tarfile.open(FIXTURES / EVIDENCE_ASSET) as tar:
        member = tar.extractfile(f"evidence/{slug}/evidence.parquet")
        assert member is not None
        return pd.read_parquet(io.BytesIO(member.read()))


# --------------------------------------------------------------------------- #
# StepView shape                                                               #
# --------------------------------------------------------------------------- #


def test_step_view_is_a_frozen_dataclass(s1):
    assert dataclasses.is_frozen(StepView) if hasattr(dataclasses, "is_frozen") else True
    fields = {f.name for f in dataclasses.fields(StepView)}
    assert fields == {"title", "framing", "tables", "checks"}
    with pytest.raises(dataclasses.FrozenInstanceError):
        s1.title = "mutated"


def test_s1_typed_contents(s1):
    assert "certification" in s1.tables
    assert all(isinstance(t, pd.DataFrame) for t in s1.tables.values())
    assert all(isinstance(c, Check) for c in s1.checks)
    assert "certification" in s1.title.lower()


# --------------------------------------------------------------------------- #
# Certification table (R2.1)                                                   #
# --------------------------------------------------------------------------- #


def test_certification_table_one_row_per_candidate(s1):
    table = s1.tables["certification"]
    assert set(table["model"]) == ALL_CANDIDATES
    assert len(table) == 5
    for column in (
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
    ):
        assert column in table.columns, column


def test_certification_table_carries_published_statistics(s1):
    row = _row(s1, FALLBACK)
    assert row["controlled_auc"] == pytest.approx(0.9255064009074705)
    assert row["controlled_ci_low"] == pytest.approx(0.8850433479176796)
    assert row["controlled_ci_high"] == pytest.approx(0.9614832820180413)
    assert row["controlled_perm_p"] == pytest.approx(0.001996007984031936)
    assert row["positive_control_auc"] == pytest.approx(0.7990115054286179)
    assert row["parse_rate"] == pytest.approx(1.0)
    assert row["n_per_class"] == 167
    assert row["verdict"] == "recalls"


# --------------------------------------------------------------------------- #
# Honest outcome framing (R2.2, R7.3)                                          #
# --------------------------------------------------------------------------- #


def test_framing_states_empty_certified_set_and_guarded_fallback(s1):
    framing = s1.framing.lower()
    assert "certified set is empty" in framing
    assert "every screenable candidate recalls" in framing
    assert "statistical certainty" in framing
    assert "openai/gpt-oss-20b" in framing
    assert "fallback" in framing
    assert "recall-guarded" in framing
    # never presented as certified (R2.2 / R7.3)
    assert "is certified" not in framing


# --------------------------------------------------------------------------- #
# Known gaps as explicit states, never failures (R2.4, R2.5)                   #
# --------------------------------------------------------------------------- #


def test_maverick_renders_pending_evidence_marker(s1):
    row = _row(s1, MAVERICK)
    assert row["status"] == "pending_evidence"
    assert "raw evidence pending" in row["note"]
    assert row["verdict"] == "recalls"  # summary verdict still displayed
    assert f"evidence:meta_llama-4-maverick-17b-128e-instruct" not in s1.tables


def test_pending_evidence_is_data_driven_not_hardcoded(s1):
    """The fixture bundle only ships 20b/120b — phi-4-mini therefore also
    lacks raw evidence and must render the same pending state (data-driven
    verdict-row-without-evidence-member logic, not a hardcoded slug)."""
    row = _row(s1, PHI4)
    assert row["status"] == "pending_evidence"
    assert "raw evidence pending" in row["note"]


def test_unservable_candidate_is_unscreenable_not_exonerated(s1):
    row = _row(s1, UNSERVABLE)
    assert row["status"] == "unscreenable"
    assert row["verdict"] == "screen_failed"
    assert "not exonerated" in row["note"]
    # the recorded error travels with the row (R2.5)
    assert "n_is=8" in row["note"]


def test_gaps_never_raise(s1):
    """build_s1 completed over a bundle missing three candidates' evidence —
    the known gaps are view-model states, not exceptions."""
    assert isinstance(s1, StepView)
    statuses = set(s1.tables["certification"]["status"])
    assert statuses == {"evidence_available", "pending_evidence", "unscreenable"}


# --------------------------------------------------------------------------- #
# Raw-evidence drill-down (R2.3)                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("slug", [FALLBACK_SLUG, BIG_SLUG])
def test_evidence_drilldown_table_per_available_candidate(s1, slug):
    table = s1.tables[f"evidence:{slug}"]
    for column in (
        "arm",
        "prompt",
        "reply",
        "included",
        "dropped_reason",
        "raw_loss",
        "raw_min_k",
        "raw_min_k_pp",
        "raw_zlib_ratio",
        "std_loss",
        "std_min_k",
        "std_min_k_pp",
        "std_zlib_ratio",
    ):
        assert column in table.columns, column
    expected = _fixture_evidence(slug)
    assert len(table) == len(expected)


# --------------------------------------------------------------------------- #
# Consistency checks vs the published summary (R2.6)                           #
# --------------------------------------------------------------------------- #


def test_class_stats_table_per_available_candidate(s1):
    for slug in (FALLBACK_SLUG, BIG_SLUG):
        stats = s1.tables[f"class_stats:{slug}"]
        assert "identifying" in stats.index
        for column in ("std_loss_mean", "std_loss_std", "std_min_k_mean"):
            assert column in stats.columns, column
    evidence = _fixture_evidence(FALLBACK_SLUG)
    included = evidence[evidence["included"]]
    assert s1.tables[f"class_stats:{FALLBACK_SLUG}"].loc["identifying", "std_loss_mean"] == (
        pytest.approx(included[included["arm"] == "identifying"]["std_loss"].mean())
    )


def test_class_count_checks_attached_and_flag_the_fixture_subset(s1):
    """TOTAL gathered rows per MAIN arm are compared against the published
    n_per_class — the screen's gather invariant; included counts vary with
    recorded attrition, so comparing them false-alarmed on the pristine live
    release (validation 2026-07-09). On the fixture subset the disagreement is
    a rendered flag (R7.2), never an exception."""
    by_name = {c.name: c for c in s1.checks}
    fallback = by_name[f"S1 {FALLBACK} [identifying] gathered rows vs published n_per_class"]
    assert fallback.published == 167
    assert fallback.rederived == 6
    assert fallback.ok is False
    assert fallback.message  # visible, human-readable flag
    big = by_name[f"S1 {BIG} [identifying] gathered rows vs published n_per_class"]
    assert big.published == 167
    assert big.rederived == 6  # gathered total: dropped rows count too
    assert big.ok is False
    assert not [c for c in s1.checks if "parse_sample" in c.name]


def test_class_count_checks_use_gathered_totals_and_skip_parse_sample():
    """Gathered totals (included + dropped) per MAIN arm equal n_per_class on
    consistent data despite attrition; the parse_sample arm (own fixed size,
    unpublished) is never count-checked — the live false-alarm root cause."""
    import numpy as np

    from factor_workbook import steps as steps_mod

    rng = np.random.default_rng(0)
    n = 12
    feature_cols = ("loss", "min_k", "min_k_pp", "zlib_ratio")

    def _arm_frame(arm: str, rows: int, dropped: int) -> pd.DataFrame:
        return pd.DataFrame({
            "arm": arm,
            "row_index": range(rows),
            "as_of": "2020-01-31",
            "prompt": "p",
            "reply": "r",
            "n_tokens": 10.0,
            "included": [True] * (rows - dropped) + [False] * dropped,
            "dropped_reason": [None] * (rows - dropped) + ["timeout"] * dropped,
            **{f"raw_{k}": rng.normal(size=rows) for k in feature_cols},
            **{f"std_{k}": rng.normal(size=rows) for k in feature_cols},
        })

    # included stays below the AUC threshold (_N_SPLITS) so this test isolates
    # the count-check semantics from the optional deeper re-derivation.
    evidence = pd.concat(
        [
            _arm_frame("identifying", n, dropped=8),
            _arm_frame("anonymized", n, dropped=8),
            _arm_frame("prose_confounded", n, dropped=8),
            _arm_frame("parse_sample", 4, dropped=1),
        ],
        ignore_index=True,
    )

    def fake_load_json(_client, key, model=None):
        if "summary" in key:
            return {"n_per_class": n}, None
        return {"feature_means": {}}, None

    orig = steps_mod.load_json
    steps_mod.load_json = fake_load_json
    try:
        _, checks = steps_mod._evidence_consistency(object(), "m/x", "m_x", evidence)
    finally:
        steps_mod.load_json = orig

    assert not [c for c in checks if "parse_sample" in c.name]
    count_checks = [c for c in checks if "gathered rows" in c.name]
    assert len(count_checks) == 3
    assert all(c.ok for c in count_checks)  # totals match despite attrition


def test_auc_rederivation_skipped_gracefully_on_single_arm_subset(s1):
    """The fixture evidence carries only the identifying arm, far below
    n_splits*2 usable rows — the deeper AUC re-derivation is skipped, not
    attempted and not failed."""
    assert not [c for c in s1.checks if "AUC" in c.name]


def _tar_with_anonymized_arm() -> tuple[bytes, pd.DataFrame]:
    """Repack the fixture bundle with a synthetic anonymized arm for 20b."""
    members: dict[str, bytes] = {}
    with tarfile.open(FIXTURES / EVIDENCE_ASSET) as tar:
        for member in tar.getmembers():
            extracted = tar.extractfile(member)
            assert extracted is not None
            members[member.name] = extracted.read()
    key = f"evidence/{FALLBACK_SLUG}/evidence.parquet"
    df = pd.read_parquet(io.BytesIO(members[key]))
    anonymized = df.copy()
    anonymized["arm"] = "anonymized"
    for column in [c for c in df.columns if c.startswith("std_")]:
        anonymized[column] = anonymized[column] - 1.0
    augmented = pd.concat([df, anonymized], ignore_index=True)
    augmented["row_index"] = range(len(augmented))
    buf = io.BytesIO()
    augmented.to_parquet(buf)
    members[key] = buf.getvalue()
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return out.getvalue(), augmented


@pytest.mark.skipif(not certification.available(), reason="stats extra (sklearn) not installed")
def test_auc_rederivation_runs_when_both_arms_have_enough_rows():
    """With the statistics extra present and both arms >= n_splits included
    rows, the point AUC is re-derived and compared to the published
    controlled_auc (R2.6 deepening)."""
    tar_bytes, augmented = _tar_with_anonymized_arm()
    view = build_s1(FakeClient({EVIDENCE_ASSET: tar_bytes}))
    by_name = {c.name: c for c in view.checks}
    check = by_name[f"S1 {FALLBACK} controlled AUC re-derived from raw evidence"]
    assert check.published == pytest.approx(0.9255064009074705)

    # expected point AUC: same matrices, same deterministic seed; auc_obs is
    # independent of n_boot/n_perm so tiny resample counts keep this fast.
    cols = ["std_loss", "std_min_k", "std_min_k_pp", "std_zlib_ratio"]
    included = augmented[augmented["included"]]
    x_is = included[included["arm"] == "identifying"][cols].to_numpy()
    x_oos = included[included["arm"] == "anonymized"][cols].to_numpy()
    expected = certification.certification_stats(x_is, x_oos, n_boot=1, n_perm=1, seed=0)[0]
    assert check.rederived == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# S2 — coin-flip naive evaluation (task 4.2)                                   #
# --------------------------------------------------------------------------- #

NAIVE_EVAL_ASSET = "naive_directional_eval_openai_gpt-oss-20b.parquet"
PUBLISHED_NAIVE_ACCURACY = 0.3888888888888889  # 28/72, nb13 full data
S2_CHECK_NAME = "S2 accuracy vs published (full data)"


@pytest.fixture(scope="module")
def s2() -> StepView:
    """One S2 view over the 5-row fixture subset for read-only assertions."""
    return build_s2(FakeClient())


def test_s2_per_call_table_verbatim(s2):
    """R3.1: the per-call records appear verbatim, prompt/reply included."""
    table = s2.tables["naive_eval"]
    expected = pd.read_parquet(FIXTURES / NAIVE_EVAL_ASSET)
    for column in (
        "date",
        "prompt",
        "reply",
        "predicted_direction",
        "confidence",
        "realized_direction",
        "correct",
    ):
        assert column in table.columns, column
    pd.testing.assert_frame_equal(table, expected)


def test_s2_summary_rederived_from_the_loaded_records(s2):
    """R3.2: accuracy + Wilson interval re-derived from the per-call rows —
    internally consistent with the loaded (fixture-subset) table."""
    records = s2.tables["naive_eval"]
    summary = s2.tables["summary"]
    assert len(summary) == 1
    row = summary.iloc[0]
    n = len(records)
    successes = int(records["correct"].sum())
    assert row["n"] == n == 5
    assert row["successes"] == successes == 1
    assert row["accuracy"] == pytest.approx(records["correct"].mean())
    low, high = wilson_ci(successes, n)
    assert row["ci_low"] == pytest.approx(low)
    assert row["ci_high"] == pytest.approx(high)
    # the interval-contains-half statement, computed on the LOADED data
    assert bool(row["contains_half"]) is (low <= 0.5 <= high) is True


def test_s2_published_accuracy_check_flags_the_fixture_subset(s2):
    """R7.2: on the 5-row fixture the re-derived accuracy disagrees with the
    published full-data figure — rendered as a flag, never an exception."""
    by_name = {c.name: c for c in s2.checks}
    check = by_name[S2_CHECK_NAME]
    assert check.published == pytest.approx(PUBLISHED_NAIVE_ACCURACY)
    assert check.rederived == pytest.approx(0.2)
    assert check.ok is False
    assert check.message  # visible, human-readable flag


def test_s2_published_accuracy_check_passes_on_full_data():
    """Guard for the live path (task 6.1): with the full 72-row table the
    same check agrees with the published 28/72 accuracy."""
    dates = pd.date_range("2019-01-31", periods=72, freq="ME")
    full = pd.DataFrame(
        {
            "date": dates,
            "prompt": [f"prompt {i}" for i in range(72)],
            "reply": [f"reply {i}" for i in range(72)],
            "predicted_direction": [1] * 72,
            "confidence": [0.65] * 72,
            "realized_direction": [1 if i < 28 else -1 for i in range(72)],
            "correct": [i < 28 for i in range(72)],
        }
    )
    buf = io.BytesIO()
    full.to_parquet(buf)
    view = build_s2(FakeClient({NAIVE_EVAL_ASSET: buf.getvalue()}))
    by_name = {c.name: c for c in view.checks}
    check = by_name[S2_CHECK_NAME]
    assert check.rederived == pytest.approx(PUBLISHED_NAIVE_ACCURACY)
    assert check.ok is True
    row = view.tables["summary"].iloc[0]
    assert row["n"] == 72 and row["successes"] == 28
    # published full-data Wilson interval [0.285, 0.504] contains 0.5
    assert row["ci_low"] == pytest.approx(0.285, abs=5e-4)
    assert row["ci_high"] == pytest.approx(0.504, abs=5e-4)
    assert bool(row["contains_half"]) is True


def test_s2_framing_states_expected_correct_no_alpha_outcome(s2):
    """R3.3/R7.3: coin-flip framed as the expected, correct honesty-measurement
    outcome — never a performance target, no forecast-accuracy claim."""
    framing = s2.framing.lower()
    assert "expected, correct" in framing
    assert "honesty measurement" in framing
    assert "despite maximal recall" in framing
    assert "never a performance target" in framing
    assert "no forecast-accuracy claim" in framing
    # phrased around the published full-data result and the coin-flip level
    assert "28/72" in framing
    assert "0.5" in framing
    assert s2.framing == S2_FRAMING
    assert "coin-flip" in s2.title.lower()


# --------------------------------------------------------------------------- #
# S3 — factor development with the guard re-derived (task 4.3)                 #
# --------------------------------------------------------------------------- #

AXES = ["inflation", "growth", "credit_stress", "policy", "risk_appetite"]
GUARD_CHECK_NAME = "S3 guarded_tilt equals raw*(1-p)"


@pytest.fixture(scope="module")
def s3() -> StepView:
    """One S3 view over the fixture subsets for all read-only assertions."""
    return build_s3(FakeClient())


def test_s3_loadings_tables_with_parse_status(s3):
    """R4.1: per-rebalance loadings on the five axes + parse status, both versions."""
    for version in ("v1", "v2"):
        table = s3.tables[f"loadings_{version}"]
        assert "parse_ok" in table.columns
        for axis in AXES:
            assert axis in table.columns, axis
        expected = pd.read_parquet(FIXTURES / f"factor_loadings_{version}.parquet")
        pd.testing.assert_frame_equal(table, expected)
    # v1 carries the not-parsed rebalance verbatim (a state, not a failure)
    assert bool(s3.tables["loadings_v1"]["parse_ok"].iloc[0]) is False


def test_s3_scores_tables_both_versions(s3):
    """R4.2: per-rebalance memorization scores alongside the loadings."""
    for version in ("v1", "v2"):
        table = s3.tables[f"scores_{version}"]
        assert "p_memorized" in table.columns
        expected = pd.read_parquet(FIXTURES / f"factor_scores_{version}.parquet")
        pd.testing.assert_frame_equal(table, expected)


def test_s3_score_summary_distribution_matches(s3):
    """R4.2: the distribution summary is re-derived from the LOADED scores —
    one row per prompt version, and the two versions genuinely differ."""
    summary = s3.tables["score_summary"]
    assert list(summary["version"]) == ["v1", "v2"]
    for version in ("v1", "v2"):
        p = pd.read_parquet(FIXTURES / f"factor_scores_{version}.parquet")["p_memorized"].dropna()
        row = summary[summary["version"] == version].iloc[0]
        assert row["n"] == len(p)
        assert row["mean"] == pytest.approx(p.mean())
        assert row["median"] == pytest.approx(p.median())
        assert row["p90"] == pytest.approx(p.quantile(0.9))
        assert row["min"] == pytest.approx(p.min())
        assert row["max"] == pytest.approx(p.max())
    v1, v2 = summary.iloc[0], summary.iloc[1]
    assert v1["mean"] != v2["mean"]  # the versions differ (R4.5 alternatives)


def test_s3_views_table_raw_vs_guarded_with_rederivation(s3):
    """R4.3: raw vs guarded side by side with the guard formula re-derived
    in front of the reviewer as its own column."""
    table = s3.tables["views_v1"]
    for column in ("date", "asset", "raw_tilt", "p_memorized", "guarded_tilt",
                   "conviction", "guarded_tilt_rederived"):
        assert column in table.columns, column
    expected = table["raw_tilt"] * (1.0 - table["p_memorized"].clip(0.0, 1.0))
    pd.testing.assert_series_equal(
        table["guarded_tilt_rederived"], expected, check_names=False
    )


def test_s3_guard_check_passes_on_the_fixture(s3):
    """R4.3/R7.2: guarded == raw*(1-p) is a per-row identity of the published
    table — the check PASSES even on a row subset (rtol 1e-9)."""
    by_name = {c.name: c for c in s3.checks}
    check = by_name[GUARD_CHECK_NAME]
    assert check.ok is True
    assert check.tolerance == pytest.approx(1e-9)
    assert check.message == ""


def test_s3_stability_table_per_version_with_published_side_by_side(s3):
    """R4.4: per-version stability rows — re-derived mean_std/mean_mac next
    to the published figures."""
    table = s3.tables["stability"]
    assert list(table["version"]) == ["v1", "v2"]
    for version in ("v1", "v2"):
        loadings = pd.read_parquet(FIXTURES / f"factor_loadings_{version}.parquet")
        rederived = loading_stability(loadings[AXES], loadings["parse_ok"])
        published = json.loads((FIXTURES / f"factor_stability_{version}.json").read_text())
        row = table[table["version"] == version].iloc[0]
        assert row["mean_std_rederived"] == pytest.approx(rederived["mean_std"])
        assert row["mean_mac_rederived"] == pytest.approx(rederived["mean_mac"])
        assert row["mean_std_published"] == pytest.approx(published["mean_std"])
        assert row["mean_mac_published"] == pytest.approx(published["mean_mac"])


def test_s3_stability_checks_flag_the_fixture_subset(s3):
    """R7.2: on the 5-row fixture the re-derived stability disagrees with the
    published full-data figures — rendered as flags, never exceptions."""
    by_name = {c.name: c for c in s3.checks}
    for version, published_std in (("v1", 0.5437447608527022), ("v2", 0.5257532419802151)):
        std = by_name[f"S3 stability mean_std {version} vs published"]
        assert std.published == pytest.approx(published_std)
        assert std.ok is False
        assert std.message  # visible, human-readable flag
        mac = by_name[f"S3 stability mean_mac {version} vs published"]
        assert mac.ok is False
        assert mac.message


def test_s3_gate_view_carries_decision_and_inputs(s3):
    """R4.5: the accept-gate view — one row per check with its inputs and
    pass flag, plus the recorded adopted version and rejection decision."""
    gate = s3.tables["gate"]
    assert set(gate["check"]) == {
        "contamination_no_greater",
        "max_dd_not_deeper",
        "sharpe_not_worse",
        "stability_not_worse",
    }
    assert len(gate) == 4
    assert set(gate["adopted_version"]) == {"v1"}
    assert all("rejected" in d for d in gate["decision"])
    contamination = gate[gate["check"] == "contamination_no_greater"].iloc[0]
    assert bool(contamination["pass"]) is False
    assert contamination["v1"] == pytest.approx(0.23608487596219696)
    assert contamination["v2"] == pytest.approx(0.270941888784604)
    # every performance check passed — the rejection is on contamination alone
    performance = gate[gate["check"] != "contamination_no_greater"]
    assert performance["pass"].all()
    sharpe = gate[gate["check"] == "sharpe_not_worse"].iloc[0]
    assert sharpe["v1"] == pytest.approx(1.1999737224508815)
    assert sharpe["v2"] == pytest.approx(1.1955608394838948)
    assert sharpe["tolerance"] == pytest.approx(0.05)


def test_s3_both_prompt_versions_preserved(s3):
    """R4.5: both versions' artifacts appear in the view as preserved
    alternatives — the rejected v2 is data, not discarded."""
    for version in ("v1", "v2"):
        assert f"loadings_{version}" in s3.tables
        assert f"scores_{version}" in s3.tables
    assert set(s3.tables["score_summary"]["version"]) == {"v1", "v2"}
    assert set(s3.tables["stability"]["version"]) == {"v1", "v2"}


def test_s3_framing_guard_and_rejection_no_accuracy_claim(s3):
    """R7.3: the guard discounts by measured memorization; v2 rejected by the
    accept-gate on contamination with every performance check passing; both
    versions preserved; no forecast-accuracy claim."""
    framing = s3.framing.lower()
    assert "raw * (1 - p_memorized)" in framing
    assert "rejected" in framing
    assert "contamination" in framing
    assert "0.2709" in framing and "0.2361" in framing
    assert "every performance check passed" in framing
    assert "preserved" in framing
    assert "no forecast-accuracy claim" in framing
    assert s3.framing == S3_FRAMING
    assert "factor" in s3.title.lower()


# --------------------------------------------------------------------------- #
# S4 — two-line simulation (task 4.4)                                          #
# --------------------------------------------------------------------------- #

S4_METRIC_FIELDS = [
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
PIT_LABEL = "PIT recall-guarded (deployable)"
DETAIL_FIELDS = ["p_memorized", "steered", "parse_ok", "conviction"]


@pytest.fixture(scope="module")
def s4() -> StepView:
    """One S4 view over the fixture subsets for all read-only assertions."""
    return build_s4(FakeClient())


def _decision_log(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def test_s4_equity_table_both_lines_matching_date_coverage(s4):
    """R5.1: both equity curves joined over the same stream — identical Date
    coverage, no gaps on either side."""
    table = s4.tables["equity"]
    assert list(table.columns) == ["value_pit", "value_nonpit"]
    assert table.index.name == "Date"
    assert table.notna().all().all()  # no one-sided dates after the join
    pit = pd.read_parquet(FIXTURES / "factor_equity_v1.parquet")
    nonpit = pd.read_parquet(FIXTURES / "factor_nonpit_diagnostic_equity_v1.parquet")
    assert list(table.index) == list(pit.index) == list(nonpit.index)
    pd.testing.assert_series_equal(table["value_pit"], pit["value"], check_names=False)
    pd.testing.assert_series_equal(table["value_nonpit"], nonpit["value"], check_names=False)


def test_s4_targets_tables_both_lines(s4):
    """R5.1: per-line target weights verbatim, over the same dates."""
    pit = s4.tables["targets_pit"]
    nonpit = s4.tables["targets_nonpit"]
    for table in (pit, nonpit):
        for column in ("SWDA.L", "XLK", "IAU", "BIL"):
            assert column in table.columns, column
    pd.testing.assert_frame_equal(pit, pd.read_parquet(FIXTURES / "factor_targets_v1.parquet"))
    pd.testing.assert_frame_equal(
        nonpit, pd.read_parquet(FIXTURES / "factor_nonpit_diagnostic_targets_v1.parquet")
    )
    assert list(pit.index) == list(nonpit.index)


def test_s4_per_date_detail_tables(s4):
    """R5.4: per rebalance date the memorization score and whether the guard
    steered that date's exposures — for each line, joined from the decision log."""
    for line, log_name in (
        ("pit", "factor_decision_log_v1.json"),
        ("nonpit", "factor_nonpit_diagnostic_decision_log_v1.json"),
    ):
        detail = s4.tables[f"detail_{line}"]
        assert list(detail.columns) == DETAIL_FIELDS
        log = _decision_log(log_name)
        expected_dates = pd.to_datetime(sorted(log["p_memorized"]))
        assert list(detail.index) == list(expected_dates)
        for date_key, score in log["p_memorized"].items():
            row = detail.loc[pd.Timestamp(date_key)]
            if score is None:
                assert pd.isna(row["p_memorized"])
            else:
                assert row["p_memorized"] == pytest.approx(score)
            assert bool(row["steered"]) is log["steered"][date_key]
            assert bool(row["parse_ok"]) is log["parse_ok"][date_key]
    # both lines cover the same rebalance dates (R5.1)
    assert list(s4.tables["detail_pit"].index) == list(s4.tables["detail_nonpit"].index)


def test_s4_pit_detail_carries_guard_states(s4):
    """R5.4: the not-parsed first pit rebalance is a visible state — no score,
    not steered — while the guarded dates carry score + steered=True."""
    detail = s4.tables["detail_pit"]
    first = detail.iloc[0]
    assert pd.isna(first["p_memorized"]) and not first["steered"] and not first["parse_ok"]
    second = detail.loc[pd.Timestamp("2019-02-01")]
    assert second["p_memorized"] == pytest.approx(0.5536914541968635)
    assert bool(second["steered"]) is True


def test_s4_labels_deployable_vs_diagnostic(s4):
    """R5.2 hard rule: the PIT line is labeled exactly as deployable; the
    recall-enabled line carries DIAGNOSTIC CONTROL; DIAGNOSTIC never appears
    on any pit-line label or field."""
    metrics = s4.tables["metrics"]
    assert list(metrics["line"]) == ["pit", "nonpit"]
    pit_row = metrics[metrics["line"] == "pit"].iloc[0]
    nonpit_row = metrics[metrics["line"] == "nonpit"].iloc[0]
    assert pit_row["label"] == PIT_LABEL
    assert "DIAGNOSTIC CONTROL" in nonpit_row["label"]
    assert all("DIAGNOSTIC" not in str(value) for value in pit_row)
    # nor in any pit-line check name or table key
    assert all("DIAGNOSTIC" not in c.name for c in s4.checks)
    for name in ("equity", "targets_pit", "detail_pit"):
        assert "DIAGNOSTIC" not in name and name in s4.tables


def test_s4_metrics_table_rederived_beside_published(s4):
    """R5.3: one row per line — the metrics recomputed from the loaded equity
    series side by side with the published reference-track figures."""
    metrics = s4.tables["metrics"]
    summary = json.loads((FIXTURES / "factor_contrast_summary_v1.json").read_text())
    for line, equity_name, published in (
        ("pit", "factor_equity_v1.parquet", summary["pit_metrics"]),
        ("nonpit", "factor_nonpit_diagnostic_equity_v1.parquet", summary["nonpit_metrics"]),
    ):
        row = metrics[metrics["line"] == line].iloc[0]
        rederived = equity_metrics(pd.read_parquet(FIXTURES / equity_name)["value"])
        for field in S4_METRIC_FIELDS:
            assert row[f"{field}_published"] == pytest.approx(published[field])
            expected = getattr(rederived, field)
            if pd.isna(expected):
                assert pd.isna(row[f"{field}_rederived"])
            else:
                assert row[f"{field}_rederived"] == pytest.approx(expected)
        # reference-track context not derivable from the equity series alone
        assert row["avg_turnover_published"] == pytest.approx(published["avg_turnover"])


def test_s4_metric_checks_flag_the_fixture_subset(s4):
    """R5.3/R7.2: per line, every equity-derivable published figure carries a
    comparison check — flagged (never raised) on the 5-row fixture subset."""
    summary = json.loads((FIXTURES / "factor_contrast_summary_v1.json").read_text())
    by_name = {c.name: c for c in s4.checks}
    assert len(s4.checks) == 2 * len(S4_METRIC_FIELDS)
    for line, published in (("pit", summary["pit_metrics"]), ("nonpit", summary["nonpit_metrics"])):
        for field in S4_METRIC_FIELDS:
            check = by_name[f"S4 {line} {field} vs published"]
            assert check.published == pytest.approx(published[field])
            assert check.ok is False
            assert check.message  # visible, human-readable flag


REAL_DATA = Path(__file__).resolve().parents[2] / "data"
S4_REAL_ASSETS = [
    "factor_contrast_summary_v1.json",
    "factor_equity_v1.parquet",
    "factor_targets_v1.parquet",
    "factor_decision_log_v1.json",
    "factor_nonpit_diagnostic_equity_v1.parquet",
    "factor_nonpit_diagnostic_targets_v1.parquet",
    "factor_nonpit_diagnostic_decision_log_v1.json",
]


@pytest.mark.skipif(
    not all((REAL_DATA / asset).exists() for asset in S4_REAL_ASSETS),
    reason="real release assets not present locally",
)
def test_s4_metric_checks_pass_on_full_data():
    """R5.3 agreement proof (Implementation Note 3.1): on the REAL published
    assets every equity-derivable metric recomputed from the full 2014-2024
    series matches the published pit/nonpit figure within the documented S4
    tolerance — all 18 checks pass. The producer convention is vectorbt's
    (365-day calendar year, day-0 return included, ddof=1, arithmetic
    sharpe, downside-RMS sortino); the loosest reproduction is the pit
    crisis_return at rel err 1.7e-6, covered by tol=1e-5."""
    overrides = {asset: (REAL_DATA / asset).read_bytes() for asset in S4_REAL_ASSETS}
    view = build_s4(FakeClient(overrides))
    assert len(view.checks) == 2 * len(S4_METRIC_FIELDS)
    failing = [check.message for check in view.checks if not check.ok]
    assert failing == []
    for check in view.checks:
        assert check.message == ""
        assert check.tolerance == pytest.approx(1e-5)
    # the full series spans the fixed 2022 crisis window: crisis figures real
    metrics = view.tables["metrics"]
    assert metrics[["crisis_return_rederived", "crisis_return_published"]].notna().all().all()


# --------------------------------------------------------------------------- #
# S5 — luck versus skill (task 4.5)                                            #
# --------------------------------------------------------------------------- #

# Published full-data contamination premium (factor_contrast_summary_v1, nb14).
PUBLISHED_PREMIUM_MEAN = 0.5282818618139323
PUBLISHED_PREMIUM_MEDIAN = 0.5667478504009134
PUBLISHED_PAIRED_D = 1.9252251066725927
S5_LINES = ["pit", "nonpit", "differential"]
S5_SSR_PUBLISHED_COLUMNS = [
    "n_obs",
    "n_rolling",
    "total_return",
    "sharpe",
    "mean_rolling_sr",
    "ssr",
    "nw_long_run_var",
    "nw_sigma_hac",
    "nw_bandwidth_L",
]


@pytest.fixture(scope="module")
def s5() -> StepView:
    """One S5 view over the fixture subsets for all read-only assertions."""
    return build_s5(FakeClient())


def test_s5_contrast_table_verbatim(s5):
    """R6.1: the paired per-date contrast of memorization scores, verbatim."""
    table = s5.tables["contrast"]
    for column in ("pit_p", "nonpit_p", "delta"):
        assert column in table.columns, column
    expected = pd.read_parquet(FIXTURES / "factor_contrast_v1.parquet")
    pd.testing.assert_frame_equal(table, expected)


def test_s5_premium_table_rederived_beside_published(s5):
    """R6.1: premium + paired effect size re-derived from the LOADED per-date
    records, side by side with the published summary figures."""
    from factor_workbook.rederive import contamination_premium

    premium = s5.tables["premium"]
    assert len(premium) == 1
    row = premium.iloc[0]
    contrast = pd.read_parquet(FIXTURES / "factor_contrast_v1.parquet")
    expected = contamination_premium(contrast["pit_p"], contrast["nonpit_p"])
    assert row["n_pairs_rederived"] == expected.n_pairs == 5
    assert row["mean_delta_rederived"] == pytest.approx(expected.mean_delta)
    assert row["median_delta_rederived"] == pytest.approx(expected.median_delta)
    assert row["paired_d_rederived"] == pytest.approx(expected.paired_d)
    assert row["n_pairs_published"] == 72
    assert row["mean_delta_published"] == pytest.approx(PUBLISHED_PREMIUM_MEAN)
    assert row["median_delta_published"] == pytest.approx(PUBLISHED_PREMIUM_MEDIAN)
    assert row["paired_d_published"] == pytest.approx(PUBLISHED_PAIRED_D)


def test_s5_premium_checks_flag_the_fixture_subset(s5):
    """R6.1/R7.2: on the 5-row fixture the re-derived premium disagrees with
    the published full-data figures — rendered as flags, never exceptions."""
    by_name = {c.name: c for c in s5.checks}
    for field, published in (
        ("mean_delta", PUBLISHED_PREMIUM_MEAN),
        ("median_delta", PUBLISHED_PREMIUM_MEDIAN),
        ("paired_d", PUBLISHED_PAIRED_D),
    ):
        check = by_name[f"S5 premium {field} vs published"]
        assert check.published == pytest.approx(published)
        assert check.ok is False
        assert check.message  # visible, human-readable flag
    n_pairs = by_name["S5 premium n_pairs vs published"]
    assert n_pairs.published == 72
    assert n_pairs.rederived == 5
    assert n_pairs.ok is False


def test_s5_ssr_table_three_lines_published_beside_rederived(s5):
    """R6.2: the Sharpe-stability table — deployable line, diagnostic line,
    and their return differential — published fields (incl. the Newey-West
    long-run variance treatment) beside the vendored re-derivation."""
    table = s5.tables["ssr"]
    published = pd.read_parquet(FIXTURES / "factor_luck_vs_skill_v1.parquet")
    assert list(table["line"]) == list(published.index)
    for column in S5_SSR_PUBLISHED_COLUMNS:
        assert f"{column}_published" in table.columns, column
        assert f"{column}_rederived" in table.columns, column
        for i in range(3):
            assert table.iloc[i][f"{column}_published"] == pytest.approx(
                published.iloc[i][column]
            )
    assert list(table["verdict"]) == list(published["verdict"])
    # fixture equity rows predate the sim window: the re-derivation degrades
    # to the vendored NaN result (n_obs 0), never an exception (R7.2)
    assert (table["n_obs_rederived"] == 0).all()
    assert table["ssr_rederived"].isna().all()


def test_s5_ssr_and_total_return_checks_flag_the_fixture_subset(s5):
    """R6.2/R7.2: per line, the re-derived SSR and total return are compared
    against the published rows — flagged (never raised) on the fixture."""
    by_name = {c.name: c for c in s5.checks}
    for line in S5_LINES:
        for field in ("ssr", "total_return"):
            check = by_name[f"S5 {line} {field} vs published"]
            assert check.ok is False
            assert check.message  # visible, human-readable flag
    # the differential rows carry the documented cancellation-slack tolerance
    assert by_name["S5 differential ssr vs published"].tolerance == pytest.approx(1e-3)
    assert by_name["S5 pit ssr vs published"].tolerance == pytest.approx(1e-6)
    assert len(s5.checks) == 4 + 2 * len(S5_LINES)


def test_s5_loading_stability_pit_vs_nonpit(s5):
    """research.md §5: the PIT-vs-non-PIT loading-stability comparison,
    re-derived from the loaded loadings tables."""
    from factor_workbook.rederive import loading_stability

    table = s5.tables["loading_stability"]
    assert list(table["line"]) == ["pit", "nonpit"]
    for line, asset in (
        ("pit", "factor_loadings_v1.parquet"),
        ("nonpit", "factor_nonpit_diagnostic_loadings_v1.parquet"),
    ):
        loadings = pd.read_parquet(FIXTURES / asset)
        expected = loading_stability(loadings[AXES], loadings["parse_ok"])
        row = table[table["line"] == line].iloc[0]
        assert row["mean_std"] == pytest.approx(expected["mean_std"])
        assert row["mean_mac"] == pytest.approx(expected["mean_mac"])


S5_REAL_ASSETS = [
    "factor_contrast_v1.parquet",
    "factor_contrast_summary_v1.json",
    "factor_luck_vs_skill_v1.parquet",
    "factor_equity_v1.parquet",
    "factor_nonpit_diagnostic_equity_v1.parquet",
    "factor_loadings_v1.parquet",
    "factor_nonpit_diagnostic_loadings_v1.parquet",
]


@pytest.mark.skipif(
    not all((REAL_DATA / asset).exists() for asset in S5_REAL_ASSETS),
    reason="real release assets not present locally",
)
def test_s5_checks_pass_on_full_data():
    """R6.1/R6.2 agreement proof: on the REAL published assets the premium,
    effect size, per-line SSR, and total returns re-derived from the full
    series all match the published values within tolerance — every S5 check
    passes — and the loading-stability comparison reproduces the research.md
    §5 figures (PIT 0.5437/0.3878 vs non-PIT 0.6370/0.5103)."""
    overrides = {asset: (REAL_DATA / asset).read_bytes() for asset in S5_REAL_ASSETS}
    view = build_s5(FakeClient(overrides))
    failing = [check.message for check in view.checks if not check.ok]
    assert failing == []
    # nb14's differential construction reproduced: published 0.028024 / 0.001999
    diff = view.tables["ssr"].iloc[2]
    assert diff["total_return_rederived"] == pytest.approx(0.028024, abs=5e-6)
    assert diff["ssr_rederived"] == pytest.approx(0.001999, abs=5e-6)
    stability = view.tables["loading_stability"].set_index("line")
    assert stability.loc["pit", "mean_std"] == pytest.approx(0.5437, abs=1e-4)
    assert stability.loc["pit", "mean_mac"] == pytest.approx(0.3878, abs=1e-4)
    assert stability.loc["nonpit", "mean_std"] == pytest.approx(0.6370, abs=1e-4)
    assert stability.loc["nonpit", "mean_mac"] == pytest.approx(0.5103, abs=1e-4)


def test_s5_framing_mandated_conclusion_wording(s5):
    """R6.3/R7.3: the recorded conclusion verbatim — memory-vs-P&L contrast,
    differential statistically indistinguishable from zero under Newey-West
    HAC, luck-compatible not skill, any recall-line excess is lookahead/recall
    bias and never attainable skill, the diagnostic line never deployable."""
    framing = s5.framing.lower()
    assert "0.528" in framing and "1.93" in framing  # premium in MEMORY
    assert "in memory" in framing
    assert "~0 in p&l" in framing
    assert "ssr" in framing and "0.002" in framing
    assert "newey-west" in framing
    assert "statistically indistinguishable from zero" in framing
    assert "luck-compatible, not skill" in framing
    assert "lookahead/recall bias" in framing
    assert "never attainable skill" in framing
    assert "never deployable" in framing
    assert "no forecast-accuracy claim" in framing
    assert "LUCK-COMPATIBLE" in s5.framing
    assert "LOOKAHEAD/RECALL BIAS" in s5.framing
    assert s5.framing == S5_FRAMING
    assert "luck" in s5.title.lower() and "skill" in s5.title.lower()


def test_s4_framing_two_lines_diagnostic_never_deployable(s4):
    """R5.2/R7.3: two lines over the same stream; the diagnostic line measures
    recall and is never deployable; near-coincident equity is the expected
    honest outcome; no forecast-accuracy claim."""
    framing = s4.framing.lower()
    assert "same rebalance stream" in framing
    assert "measure recall" in framing
    assert "never the deployable" in framing
    assert "near-coincident" in framing
    assert "expected honest outcome" in framing
    assert "no forecast-accuracy claim" in framing
    assert "DIAGNOSTIC CONTROL" in s4.framing
    assert s4.framing == S4_FRAMING
    assert "two-line" in s4.title.lower()
