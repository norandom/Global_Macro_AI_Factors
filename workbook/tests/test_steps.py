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
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from factor_workbook import certification
from factor_workbook.rederive import wilson_ci
from factor_workbook.release import FetchError, Provenance, ReleaseError
from factor_workbook.steps import S2_FRAMING, StepView, build_s1, build_s2
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
    """Included rows per arm are compared against the published n_per_class;
    on the 6-row fixture subset the disagreement is a rendered flag (R7.2),
    never an exception."""
    by_name = {c.name: c for c in s1.checks}
    fallback = by_name[f"S1 {FALLBACK} [identifying] included rows vs published n_per_class"]
    assert fallback.published == 167
    assert fallback.rederived == 6
    assert fallback.ok is False
    assert fallback.message  # visible, human-readable flag
    big = by_name[f"S1 {BIG} [identifying] included rows vs published n_per_class"]
    assert big.published == 167
    assert big.rederived == 5
    assert big.ok is False


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
