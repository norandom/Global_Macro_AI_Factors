"""Offline tests for scripts/extend_stream_2026.py (task 8.1 pure helpers).

No network, no NIM key, no price fetch: only the pure helpers the extension
script composes around the live pipeline (replay-reply synthesis, the
completed-months panel guard, and the in-training vs post-cutoff split table).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

# NOTE: extend_stream_2026 (and macro_framework.factor_scoring, which it pulls
# in) is imported INSIDE each test, and this file is named to sort AFTER
# tests/test_factor_scoring.py: that suite asserts factor_scoring is NOT yet in
# sys.modules when its foundation test runs (module-level imports here would
# break it at collection time).


def _mod():
    import extend_stream_2026 as ext

    return ext


def test_synth_loadings_reply_roundtrips_through_parse_loadings() -> None:
    from macro_framework import factor_scoring as fs

    ext = _mod()
    loadings = {
        "inflation": 0.4,
        "growth": -0.2,
        "credit_stress": 0.6,
        "policy": -0.1,
        "risk_appetite": -0.3,
    }
    reply = ext.synth_loadings_reply(loadings)
    rl = fs.parse_loadings(reply, pd.Timestamp("2020-03-02"))
    assert rl is not None and rl.parse_ok
    assert rl.loadings == loadings


def test_synth_loadings_reply_none_stays_unparsed() -> None:
    from macro_framework import factor_scoring as fs

    ext = _mod()
    reply = ext.synth_loadings_reply(None)
    assert reply == ""
    assert fs.parse_loadings(reply, pd.Timestamp("2020-03-02")) is None


def test_completed_months_only_drops_current_month_row() -> None:
    ext = _mod()
    idx = pd.DatetimeIndex(["2026-05-31", "2026-06-30", "2026-07-31"])
    panel = pd.DataFrame({"cpi_yoy": [1.0, 2.0, 3.0]}, index=idx)
    out = ext.completed_months_only(panel, today=pd.Timestamp("2026-07-17"))
    assert list(out.index) == [pd.Timestamp("2026-05-31"), pd.Timestamp("2026-06-30")]


def test_split_contrast_table_segments_and_stats() -> None:
    ext = _mod()
    idx = pd.DatetimeIndex(["2024-01-02", "2024-05-01", "2024-08-01", "2025-01-02", "2025-02-03"])
    df = pd.DataFrame(
        {
            "pit_p": [0.1, 0.2, 0.3, 0.4, float("nan")],
            "nonpit_p": [0.7, 0.9, 0.35, 0.45, 0.5],
        },
        index=idx,
    )
    table = ext.split_contrast_table(df, cutoff=pd.Timestamp("2024-06-01"))

    it = table["in_training"]
    assert it["n_pairs"] == 2
    assert abs(it["mean_delta"] - 0.65) < 1e-12
    assert abs(it["median_delta"] - 0.65) < 1e-12
    assert set(it) == {"n_pairs", "mean_delta", "median_delta", "paired_d"}

    pc = table["post_cutoff"]
    assert pc["n_pairs"] == 2  # the NaN pit_p pair is dropped
    assert abs(pc["mean_delta"] - 0.05) < 1e-12

    assert isinstance(table["prediction_outcome"], str)
    assert "collapsed" in table["prediction_outcome"]


def test_classify_premium_outcome_either_way() -> None:
    ext = _mod()
    assert "collapsed" in ext.classify_premium_outcome(0.53, 0.02)
    persisted = ext.classify_premium_outcome(0.53, 0.48)
    assert "did NOT collapse" in persisted


# --------------------------------------------------------------------------- #
# Task 6.1 — dated Factor evidence contract (validate_evidence_records)        #
# --------------------------------------------------------------------------- #


def _evidence(ext, *, variant="pit", d=None, prompt="PIT PROMPT", reid=True, **over):
    """Build one valid DatedFactorEvidence; overrides applied, then re-hashed.

    ``reid=False`` skips recomputing evidence_id so hash/id mismatches can be
    injected deliberately.
    """
    import dataclasses
    from datetime import date as _date

    d = d or _date(2025, 10, 1)
    source = prompt if variant == "pit" else prompt + "\nNON-PIT ADDITIONS"
    response = '{"inflation": 0.1, "growth": 0.2, "credit_stress": 0.3, "policy": -0.1, "risk_appetite": 0.0}'
    rec = ext.DatedFactorEvidence(
        variant=variant,
        rebalance_date=d,
        segment="live_ext2026",
        pit_prompt_text=prompt,
        pit_prompt_sha256=ext.sha256_text(prompt),
        source_prompt_text=source,
        source_prompt_sha256=ext.sha256_text(source),
        response_text=response,
        response_sha256=ext.sha256_text(response),
        response_origin="raw_nim",
        score_p_memorized=0.4,
        score_parse_ok=True,
        score_fail_reason=None,
        score_origin="live_nim",
        loading_inflation=0.1,
        loading_growth=0.2,
        loading_credit_stress=0.3,
        loading_policy=-0.1,
        loading_risk_appetite=0.0,
        loadings_parse_ok=True,
        source_artifact=None,
        source_artifact_sha256=None,
        evidence_id="",
    )
    if over:
        rec = dataclasses.replace(rec, **over)
    if reid:
        rec = ext.with_evidence_id(rec)
    return rec


def test_validate_evidence_records_valid_two_variant_mapping() -> None:
    from datetime import date as _date

    ext = _mod()
    d1, d2 = _date(2025, 10, 1), _date(2025, 11, 3)
    recs = [
        _evidence(ext, variant="pit", d=d1),
        _evidence(ext, variant="pit", d=d2),
        _evidence(ext, variant="nonpit_diagnostic", d=d1),
        _evidence(ext, variant="nonpit_diagnostic", d=d2),
    ]
    expected = [(v, d) for v in ("pit", "nonpit_diagnostic") for d in (d1, d2)]
    out = ext.validate_evidence_records(recs, expected)
    assert set(out) == set(expected)
    assert out[("pit", d1)] is recs[0]
    assert out[("nonpit_diagnostic", d2)] is recs[3]
    import pytest

    with pytest.raises(TypeError):
        out[("pit", d1)] = recs[1]  # immutable mapping


def test_validate_evidence_identical_prompt_on_two_dates_is_valid() -> None:
    # R6.1/R8.7 core: duplicate PROMPT TEXT across dates is legitimate (rounded
    # macro states collide, e.g. Oct-2025/Nov-2025); only a duplicate
    # (variant, date) key is an error.
    from datetime import date as _date

    ext = _mod()
    d1, d2 = _date(2025, 10, 1), _date(2025, 11, 3)
    r1 = _evidence(ext, d=d1, prompt="SAME ROUNDED MACRO STATE")
    r2 = _evidence(ext, d=d2, prompt="SAME ROUNDED MACRO STATE",
                   score_p_memorized=0.9,
                   response_text='{"inflation": 0.5, "growth": 0.5, "credit_stress": 0.5, "policy": 0.5, "risk_appetite": 0.5}',
                   reid=False)
    import dataclasses

    r2 = dataclasses.replace(r2, response_sha256=ext.sha256_text(r2.response_text))
    r2 = ext.with_evidence_id(r2)
    out = ext.validate_evidence_records([r1, r2], [("pit", d1), ("pit", d2)])
    assert out[("pit", d1)].pit_prompt_text == out[("pit", d2)].pit_prompt_text
    assert out[("pit", d1)].evidence_id != out[("pit", d2)].evidence_id
    assert out[("pit", d1)].score_p_memorized != out[("pit", d2)].score_p_memorized


def test_validate_evidence_duplicate_key_raises_even_with_distinct_prompts() -> None:
    import pytest

    ext = _mod()
    r1 = _evidence(ext, prompt="PROMPT A")
    r2 = _evidence(ext, prompt="PROMPT B")
    with pytest.raises(ValueError, match="duplicate evidence key"):
        ext.validate_evidence_records([r1, r2], [("pit", r1.rebalance_date)])


def test_validate_evidence_missing_and_extra_expected_keys_raise() -> None:
    from datetime import date as _date

    import pytest

    ext = _mod()
    d1, d2 = _date(2025, 10, 1), _date(2025, 11, 3)
    r1 = _evidence(ext, d=d1)
    with pytest.raises(ValueError, match="missing expected evidence key"):
        ext.validate_evidence_records([r1], [("pit", d1), ("pit", d2)])
    r2 = _evidence(ext, d=d2)
    with pytest.raises(ValueError, match="unexpected evidence key"):
        ext.validate_evidence_records([r1, r2], [("pit", d1)])


def test_validate_evidence_unsorted_dates_raise() -> None:
    from datetime import date as _date

    import pytest

    ext = _mod()
    d1, d2 = _date(2025, 10, 1), _date(2025, 11, 3)
    recs = [_evidence(ext, d=d2), _evidence(ext, d=d1)]
    with pytest.raises(ValueError, match="strictly increasing"):
        ext.validate_evidence_records(recs, [("pit", d1), ("pit", d2)])


def test_validate_evidence_unknown_variant_and_origin_raise() -> None:
    import pytest

    ext = _mod()
    bad_variant = _evidence(ext, variant="live")
    with pytest.raises(ValueError, match="unsupported variant"):
        ext.validate_evidence_records([bad_variant], [("live", bad_variant.rebalance_date)])
    bad_origin = _evidence(ext, response_origin="cached")
    with pytest.raises(ValueError, match="unsupported response_origin"):
        ext.validate_evidence_records([bad_origin], [("pit", bad_origin.rebalance_date)])


def test_validate_evidence_blank_segment_and_score_origin_raise() -> None:
    import pytest

    ext = _mod()
    rec = _evidence(ext, segment=" ")
    with pytest.raises(ValueError, match="blank segment"):
        ext.validate_evidence_records([rec], [("pit", rec.rebalance_date)])
    rec = _evidence(ext, score_origin="")
    with pytest.raises(ValueError, match="blank score_origin"):
        ext.validate_evidence_records([rec], [("pit", rec.rebalance_date)])


def test_validate_evidence_each_hash_mismatch_raises() -> None:
    import pytest

    ext = _mod()
    for field in ("pit_prompt_sha256", "source_prompt_sha256", "response_sha256"):
        rec = _evidence(ext, reid=False, **{field: "0" * 64})
        rec = ext.with_evidence_id(rec)
        with pytest.raises(ValueError, match=f"{field} mismatch"):
            ext.validate_evidence_records([rec], [("pit", rec.rebalance_date)])


def test_validate_evidence_id_mismatch_raises() -> None:
    import pytest

    ext = _mod()
    rec = _evidence(ext, reid=False, evidence_id="f" * 64)
    with pytest.raises(ValueError, match="evidence_id mismatch"):
        ext.validate_evidence_records([rec], [("pit", rec.rebalance_date)])


def test_validate_evidence_score_parse_state_inconsistencies_raise() -> None:
    import pytest

    ext = _mod()
    # parse_ok True but no p_memorized
    rec = _evidence(ext, score_p_memorized=None)
    with pytest.raises(ValueError, match="score parse-state inconsistent"):
        ext.validate_evidence_records([rec], [("pit", rec.rebalance_date)])
    # parse_ok False but no fail_reason
    rec = _evidence(ext, score_p_memorized=None, score_parse_ok=False, score_fail_reason=None)
    with pytest.raises(ValueError, match="score parse-state inconsistent"):
        ext.validate_evidence_records([rec], [("pit", rec.rebalance_date)])


def test_validate_evidence_loadings_parse_state_inconsistencies_raise() -> None:
    import pytest

    ext = _mod()
    rec = _evidence(ext, loading_growth=None)  # parse_ok True with a hole
    with pytest.raises(ValueError, match="loadings parse-state inconsistent"):
        ext.validate_evidence_records([rec], [("pit", rec.rebalance_date)])
    rec = _evidence(ext, loadings_parse_ok=False)  # parse_ok False with values
    with pytest.raises(ValueError, match="loadings parse-state inconsistent"):
        ext.validate_evidence_records([rec], [("pit", rec.rebalance_date)])


def test_validate_evidence_non_finite_values_raise() -> None:
    import pytest

    ext = _mod()
    rec = _evidence(ext, loading_policy=float("nan"))
    with pytest.raises(ValueError, match="non-finite loading_policy"):
        ext.validate_evidence_records([rec], [("pit", rec.rebalance_date)])
    rec = _evidence(ext, score_p_memorized=float("inf"))
    with pytest.raises(ValueError, match="non-finite score_p_memorized"):
        ext.validate_evidence_records([rec], [("pit", rec.rebalance_date)])


def test_validate_evidence_source_artifact_sha_iff_source_artifact() -> None:
    import pytest

    ext = _mod()
    rec = _evidence(ext, source_artifact="data/factor_loadings_v1.parquet")
    with pytest.raises(ValueError, match="source_artifact_sha256"):
        ext.validate_evidence_records([rec], [("pit", rec.rebalance_date)])
    rec = _evidence(ext, source_artifact_sha256="a" * 64)
    with pytest.raises(ValueError, match="source_artifact_sha256"):
        ext.validate_evidence_records([rec], [("pit", rec.rebalance_date)])


def test_replay_validation_error_is_value_error() -> None:
    ext = _mod()
    assert issubclass(ext.ReplayValidationError, ValueError)


# --------------------------------------------------------------------------- #
# Task 6.2 — persist exact per-date PIT and non-PIT evidence                   #
# --------------------------------------------------------------------------- #


def _build(ext, *, variant="pit", d, pit="PIT PROMPT", response, p=0.4,
           loadings=..., segment="live_ext2026", fail=None, src=None, src_sha=None):
    """Build one record through the production builder (dated_prompt_collisions shape)."""
    from macro_framework import factor_scoring as fs

    if loadings is ...:
        loadings = {a: 0.1 for a in fs.MACRO_AXES}
    score = fs.FactorScore(p_memorized=p, parse_ok=p is not None,
                           fail_reason=fail if p is None else None)
    return ext.build_dated_evidence(
        variant=variant, rebalance_date=d, segment=segment,
        pit_prompt=pit,
        source_prompt=pit if variant == "pit" else pit + "\nNON-PIT ADDITIONS",
        response_text=response, score=score, loadings=loadings,
        source_artifact=src, source_artifact_sha256=src_sha)


def _collision_records(ext):
    """dated_prompt_collisions fixture: identical PIT prompt on two dates, both
    variants, with per-date responses/scores/loadings/origins/sources."""
    from datetime import date as _date

    d1, d2 = _date(2024, 11, 1), _date(2025, 10, 1)
    pit = "SAME ROUNDED MACRO STATE PROMPT"
    recs = []
    for variant in ("pit", "nonpit_diagnostic"):
        # d1 replayed from v1 (reconstructed), d2 live — distinct everything.
        recs.append(_build(ext, variant=variant, d=d1, pit=pit,
                           response='{"inflation": 0.1}', p=0.2,
                           loadings={a: 0.1 for a in _axes()},
                           segment="replayed_v1",
                           src="data/factor_loadings_v1.parquet", src_sha="a" * 64))
        recs.append(_build(ext, variant=variant, d=d2, pit=pit,
                           response='{"inflation": 0.9}', p=0.8,
                           loadings={a: 0.9 for a in _axes()}))
    return d1, d2, pit, recs


def _axes():
    from macro_framework import factor_scoring as fs

    return fs.MACRO_AXES


def test_duplicate_prompt_dates_persist_independent_records() -> None:
    # R6.1 (owner 6.2): identical prompt text on two dates keeps a distinct
    # response, score set, loadings, origin, source, and evidence identity per
    # date — for BOTH variants.
    ext = _mod()
    d1, d2, pit, recs = _collision_records(ext)
    expected = [(v, d) for v in ("pit", "nonpit_diagnostic") for d in (d1, d2)]
    # records are per-variant date-ordered: pit block then nonpit block
    ordered = [recs[0], recs[1], recs[2], recs[3]]
    out = ext.validate_evidence_records(ordered, expected)
    for variant in ("pit", "nonpit_diagnostic"):
        r1, r2 = out[(variant, d1)], out[(variant, d2)]
        assert r1.pit_prompt_text == r2.pit_prompt_text == pit
        assert r1.pit_prompt_sha256 == r2.pit_prompt_sha256
        assert r1.response_text != r2.response_text
        assert r1.response_sha256 != r2.response_sha256
        assert r1.score_p_memorized != r2.score_p_memorized
        assert r1.loading_inflation != r2.loading_inflation
        assert r1.response_origin == "reconstructed_from_v1_loadings"
        assert r2.response_origin == "raw_nim"
        assert r1.source_artifact == "data/factor_loadings_v1.parquet"
        assert r2.source_artifact is None
        assert r1.evidence_id != r2.evidence_id
        assert r1.rebalance_date == d1 and r2.rebalance_date == d2


def test_ac_6_1() -> None:
    test_duplicate_prompt_dates_persist_independent_records()


def test_build_dated_evidence_normalizes_timestamp_dates() -> None:
    from datetime import date as _date

    ext = _mod()
    rec = _build(ext, d=pd.Timestamp("2025-10-01"), response="x")
    assert rec.rebalance_date == _date(2025, 10, 1)
    assert type(rec.rebalance_date) is _date  # validator rejects datetime/Timestamp
    ext.validate_evidence_records([rec], [("pit", rec.rebalance_date)])


def test_build_dated_evidence_reconstructed_marks_origin_and_source() -> None:
    from datetime import date as _date

    import pytest

    ext = _mod()
    d = _date(2020, 3, 2)
    # A replayed row is a reconstruction — even the ""-reply v1 parse failure —
    # and must carry its source artifact hash (R7.3): never raw provenance.
    rec = _build(ext, d=d, segment="replayed_v1", response="", loadings=None, p=None,
                 fail="replayed_nan", src="data/factor_loadings_v1.parquet", src_sha="b" * 64)
    assert rec.response_origin == "reconstructed_from_v1_loadings"
    assert rec.source_artifact_sha256 == "b" * 64
    with pytest.raises(ValueError, match="source_artifact"):
        _build(ext, d=d, segment="replayed_v1", response='{"x": 1}')


def test_build_dated_evidence_preserves_response_bytes_exactly() -> None:
    import hashlib
    from datetime import date as _date

    ext = _mod()
    text = '{"inflation": 0.25}\n<think>η → ∞…</think>'
    rec = _build(ext, d=_date(2025, 10, 1), response=text)
    assert rec.response_text == text
    assert rec.response_sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert rec.response_origin == "raw_nim"
    # an unparseable-but-returned live reply stays raw_nim (not generation_failed)
    unparsed = _build(ext, d=_date(2025, 11, 3), response="not json", loadings=None)
    assert unparsed.response_origin == "raw_nim"
    assert unparsed.loadings_parse_ok is False


def test_build_dated_evidence_retains_generation_and_scoring_failures() -> None:
    from datetime import date as _date

    ext = _mod()
    d = _date(2025, 10, 1)
    # generation exception path (_reply_text -> "") is a RETAINED dated record
    failed = _build(ext, d=d, response="", loadings=None, p=None, fail="TimeoutError")
    assert failed.response_origin == "generation_failed"
    assert failed.loadings_parse_ok is False
    assert all(getattr(failed, f"loading_{a}") is None for a in _axes())
    assert failed.score_parse_ok is False
    assert failed.score_fail_reason == "TimeoutError"
    # a None score with no reason still gets an explicit failure reason
    noreason = _build(ext, d=d, response="ok", p=None)
    assert noreason.score_parse_ok is False
    assert noreason.score_fail_reason  # non-empty, explicit
    # both survive full validation — failures are records, never dropped rows
    ext.validate_evidence_records([failed], [("pit", d)])


def test_write_evidence_table_roundtrips_flat_scalar_rows(tmp_path) -> None:
    import dataclasses

    ext = _mod()
    d1, d2, _, recs = _collision_records(ext)
    expected = [(v, d) for v in ("pit", "nonpit_diagnostic") for d in (d1, d2)]
    path = ext.write_evidence_table(recs, expected, tmp_path / "run1")
    assert path.parent == tmp_path / "run1"
    df = pd.read_parquet(path)
    assert len(df) == 4
    field_names = [f.name for f in dataclasses.fields(ext.DatedFactorEvidence)]
    assert list(df.columns) == field_names
    assert sorted(zip(df["variant"], df["rebalance_date"])) == sorted(
        (v, d) for v, d in expected)
    assert set(df["response_text"]) == {'{"inflation": 0.1}', '{"inflation": 0.9}'}


def test_write_evidence_table_validates_before_writing_anything(tmp_path) -> None:
    import pytest

    ext = _mod()
    d1, d2, _, recs = _collision_records(ext)
    dup = [recs[0], recs[0]]  # duplicate (variant, date)
    run = tmp_path / "run_dup"
    with pytest.raises(ValueError, match="duplicate evidence key"):
        ext.write_evidence_table(dup, [("pit", d1)], run)
    assert not run.exists() or not any(run.iterdir())  # nothing was written
    run2 = tmp_path / "run_missing"
    with pytest.raises(ValueError, match="missing expected evidence key"):
        ext.write_evidence_table([recs[0]], [("pit", d1), ("pit", d2)], run2)
    assert not run2.exists() or not any(run2.iterdir())


def test_write_evidence_table_refuses_non_empty_destination(tmp_path) -> None:
    import pytest

    ext = _mod()
    d1, d2, _, recs = _collision_records(ext)
    expected = [(v, d) for v in ("pit", "nonpit_diagnostic") for d in (d1, d2)]
    run = tmp_path / "occupied"
    run.mkdir()
    marker = run / "factor_loadings_v1.parquet"
    marker.write_bytes(b"historical bytes")
    with pytest.raises(ValueError, match="not empty"):
        ext.write_evidence_table(recs, expected, run)
    assert marker.read_bytes() == b"historical bytes"  # prior artifacts untouched (R7.6)


# --------------------------------------------------------------------------- #
# Task 6.3 — exact dated replay resolution (make_dated_replay_weight_fn)       #
# --------------------------------------------------------------------------- #


def _replay_fixture(ext):
    """Two dated records with the IDENTICAL rendered anonymized PIT prompt on
    two dates but distinct response/score/loadings, frozen into the validated
    immutable evidence mapping. Returns the pieces a sim-shaped test needs."""
    from datetime import date as _date

    import macro_framework as mf
    from macro_framework import factor_scoring as fs

    macro_state = {"cpi_yoy_z": 0.5, "t10y2y_z": -0.3, "hy_oas_z": 0.1}
    amap = mf.AssetMap.default()
    snapshot = [{"id": p, "category": c} for p, c in sorted(amap.categories.items())]
    prompt = fs.render_regime_loadings_prompt(macro_state, snapshot)
    d1, d2 = _date(2025, 10, 1), _date(2025, 11, 3)
    r1 = _build(ext, d=d1, pit=prompt, p=0.2,
                response=ext.synth_loadings_reply({a: 0.1 for a in _axes()}),
                loadings={a: 0.1 for a in _axes()})
    r2 = _build(ext, d=d2, pit=prompt, p=0.8,
                response=ext.synth_loadings_reply({a: 0.9 for a in _axes()}),
                loadings={a: 0.9 for a in _axes()})
    evidence = ext.validate_evidence_records([r1, r2], [("pit", d1), ("pit", d2)])
    return macro_state, snapshot, amap, prompt, d1, d2, evidence


def test_dated_replay_identical_prompts_consume_their_own_dated_evidence() -> None:
    # AC 6.2 core: response and score associate with the ORIGINAL rebalance
    # date, not with the (identical) prompt text.
    ext = _mod()
    _, _, _, prompt, d1, d2, evidence = _replay_fixture(ext)
    seen = {}
    for d, want_load, want_p in ((d1, 0.1, 0.2), (d2, 0.9, 0.8)):
        rec = ext.resolve_dated_evidence(evidence, "pit", d)
        gen, scorer = ext.dated_replay_closures(rec)
        reply = gen(prompt)  # identical text on both dates; resolved by DATE
        sc = scorer.score(prompt)
        assert reply == ext.synth_loadings_reply({a: want_load for a in _axes()})
        assert sc.parse_ok and sc.p_memorized == want_p
        # the simulation-rendered prompt hash equals the dated evidence (R6.2)
        assert ext.sha256_text(prompt) == rec.pit_prompt_sha256
        seen[d] = (reply, sc.p_memorized)
    assert seen[d1] != seen[d2]


def test_dated_replay_weight_fn_two_dates_zero_live_calls() -> None:
    # End-to-end through the UNCHANGED fs.make_factor_weight_fn/factor_rebalance:
    # each date's decision reflects its OWN response (loadings x9) AND its OWN
    # score (discount (1-p): 0.25x), with zero live model calls.
    import pytest

    from macro_framework import factor_scoring as fs

    ext = _mod()
    macro_state, snapshot, amap, prompt, d1, d2, evidence = _replay_fixture(ext)

    def _boom(*a, **k):  # live-scoring double: raises if any live call happens
        raise AssertionError("live model call during replay")

    recorded = []

    class _Agent:
        asset_map = amap

        def views_to_bl(self, views, real_symbols):
            recorded.append(views)
            return None, None

    def build_inputs(ctx):
        return macro_state, snapshot, ctx["rebalance_date"], None

    def combine(ctx, P, Q):
        return pd.Series(1.0, index=ctx["prices"].columns)

    prices = pd.DataFrame({"XX": [1.0, 2.0], "YY": [1.0, 2.0]})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fs.FactorScorer, "score", _boom)
        mp.setattr(fs.FactorScorer, "score_many", _boom)
        weight_fn = ext.make_dated_replay_weight_fn(
            variant="pit", evidence=evidence, agent=_Agent(),
            build_inputs=build_inputs, combine=combine)
        for d in (d1, d2):
            w = weight_fn({"rebalance_date": pd.Timestamp(d), "prices": prices})
            assert (w == 1.0).all()

    assert len(recorded) == 2
    t1 = {v.asset_long: v.expected_excess_annualized for v in recorded[0]}
    t2 = {v.asset_long: v.expected_excess_annualized for v in recorded[1]}
    nonzero = [a for a, t in t1.items() if abs(t) > 1e-12]
    assert nonzero
    for a in nonzero:
        # own response and own score: (0.9/0.1) * ((1-0.8)/(1-0.2)) = 2.25
        assert abs(t2[a] / t1[a] - 2.25) < 1e-9


def test_ac_6_2() -> None:
    test_dated_replay_identical_prompts_consume_their_own_dated_evidence()
    test_dated_replay_weight_fn_two_dates_zero_live_calls()


def test_dated_replay_prompt_mismatch_raises_never_degrades() -> None:
    import pytest

    ext = _mod()
    _, _, _, prompt, d1, _, evidence = _replay_fixture(ext)
    rec = ext.resolve_dated_evidence(evidence, "pit", d1)
    gen, scorer = ext.dated_replay_closures(rec)
    with pytest.raises(ValueError, match="does not match dated evidence"):
        gen(prompt + " TAMPERED")
    with pytest.raises(ValueError, match="does not match dated evidence"):
        scorer.score(prompt + " TAMPERED")
    # the true rendered prompt still resolves (identical text elsewhere is fine)
    assert gen(prompt) == rec.response_text


def test_dated_replay_nonpit_variant_matches_anonymized_pit_prompt() -> None:
    # factor_rebalance re-renders only the ANONYMIZED PIT prompt; comparing the
    # sim prompt against source_prompt would spuriously fail every nonpit date.
    from datetime import date as _date

    import pytest

    ext = _mod()
    d = _date(2025, 10, 1)
    rec = _build(ext, variant="nonpit_diagnostic", d=d, pit="RENDERED PIT PROMPT",
                 response='{"x": 1}', p=0.5, loadings=None)
    evidence = ext.validate_evidence_records([rec], [("nonpit_diagnostic", d)])
    gen, scorer = ext.dated_replay_closures(
        ext.resolve_dated_evidence(evidence, "nonpit_diagnostic", d))
    assert gen("RENDERED PIT PROMPT") == rec.response_text
    assert scorer.score("RENDERED PIT PROMPT").p_memorized == 0.5
    with pytest.raises(ValueError, match="does not match dated evidence"):
        gen(rec.source_prompt_text)  # the identifying prompt is never re-rendered


def test_dated_replay_missing_key_raises_immediately() -> None:
    # Bullet 4: no not_pre_scored FactorScore, no "" reply — a missing
    # (variant, date) key raises at once, and the run-local failures list
    # captures it so the caller re-raises after walk_forward's swallow.
    import pytest

    ext = _mod()
    macro_state, snapshot, amap, prompt, d1, _, evidence = _replay_fixture(ext)
    with pytest.raises(ValueError, match="missing dated evidence"):
        ext.resolve_dated_evidence(evidence, "nonpit_diagnostic", d1)

    class _Agent:
        asset_map = amap

        def views_to_bl(self, views, real_symbols):
            return None, None

    failures = []
    weight_fn = ext.make_dated_replay_weight_fn(
        variant="nonpit_diagnostic", evidence=evidence, agent=_Agent(),
        build_inputs=lambda ctx: (macro_state, snapshot, ctx["rebalance_date"], None),
        combine=lambda ctx, P, Q: pd.Series(1.0, index=ctx["prices"].columns),
        failures=failures)
    prices = pd.DataFrame({"XX": [1.0, 2.0]})
    with pytest.raises(ValueError, match="missing dated evidence"):
        weight_fn({"rebalance_date": pd.Timestamp(d1), "prices": prices})
    assert len(failures) == 1
    assert isinstance(failures[0], ext.ReplayValidationError)


def test_dated_replay_weight_fn_records_mismatch_for_post_sim_reraise() -> None:
    # Evidence whose recorded prompt differs from the sim-rendered one must
    # raise AND land in the failures list (walk_forward swallows weight_fn
    # exceptions, so the caller re-raises from this list after the sim).

    import pytest

    ext = _mod()
    macro_state, snapshot, amap, prompt, d1, _, _ = _replay_fixture(ext)
    stale = _build(ext, d=d1, pit="NOT THE RENDERED PROMPT", response='{"x": 1}',
                   p=0.5, loadings=None)
    evidence = ext.validate_evidence_records([stale], [("pit", d1)])

    class _Agent:
        asset_map = amap

        def views_to_bl(self, views, real_symbols):
            return None, None

    failures = []
    weight_fn = ext.make_dated_replay_weight_fn(
        variant="pit", evidence=evidence, agent=_Agent(),
        build_inputs=lambda ctx: (macro_state, snapshot, ctx["rebalance_date"], None),
        combine=lambda ctx, P, Q: pd.Series(1.0, index=ctx["prices"].columns),
        failures=failures)
    prices = pd.DataFrame({"XX": [1.0, 2.0]})
    with pytest.raises(ValueError, match="does not match dated evidence"):
        weight_fn({"rebalance_date": pd.Timestamp(d1), "prices": prices})
    assert len(failures) == 1


# --------------------------------------------------------------------------- #
# Task 6.4 — source-to-consumption replay auditing                             #
# --------------------------------------------------------------------------- #


def _audit_agent(amap):
    class _Agent:
        asset_map = amap

        def views_to_bl(self, views, real_symbols):
            return None, None

    return _Agent()


def _consume_both_passes(ext, evidence, variant, dates, macro_state, snapshot, amap,
                         consumed) -> None:
    """Drive BOTH production consumption passes for every date, exactly as
    run_variant_line does: the walk-forward weight_fn pass, then the
    decision-log pass (resolve + record + factor_rebalance + decision identity)."""
    from macro_framework import factor_scoring as fs

    weight_fn = ext.make_dated_replay_weight_fn(
        variant=variant, evidence=evidence, agent=_audit_agent(amap),
        build_inputs=lambda ctx: (macro_state, snapshot, ctx["rebalance_date"], None),
        combine=lambda ctx, P, Q: pd.Series(1.0, index=ctx["prices"].columns),
        consumed=consumed)
    prices = pd.DataFrame({"XX": [1.0, 2.0]})
    for d in dates:
        weight_fn({"rebalance_date": pd.Timestamp(d), "prices": prices})
    for d in dates:  # second pass: the decision log
        rec = ext.resolve_dated_evidence(evidence, variant, d)
        ext.record_consumption(consumed, (variant, d), rec)
        gen, scorer = ext.dated_replay_closures(rec)
        dec = fs.factor_rebalance(
            generate_loadings=gen, scorer=scorer, agent=_audit_agent(amap),
            macro_state=macro_state, asset_snapshot=snapshot,
            real_symbols=["XX"], as_of=pd.Timestamp(d))
        ext.record_decision_identity(consumed, (variant, d), rec, dec)


def test_replay_audit_clean_run_passes_and_persists_summary(tmp_path) -> None:
    # AC 6.4 core: every simulated (variant, date) gets a consumption
    # fingerprint, source==consumption validates without raising, and the
    # persisted summary reports expected==consumed counts — without mutating
    # the immutable source evidence.
    import json

    ext = _mod()
    macro_state, snapshot, amap, prompt, d1, d2, evidence = _replay_fixture(ext)
    ids_before = {k: r.evidence_id for k, r in evidence.items()}
    consumed = {}
    _consume_both_passes(ext, evidence, "pit", (d1, d2), macro_state, snapshot, amap,
                         consumed)
    expected = [("pit", d1), ("pit", d2)]
    assert set(consumed) == set(expected)
    assert ext.validate_source_to_consumption(evidence, consumed, expected) is None
    path = ext.write_replay_audit_summary(evidence, consumed, expected, tmp_path)
    assert path.name == ext.REPLAY_AUDIT_NAME
    summary = json.loads(path.read_text())
    assert summary["result"] == "pass"
    assert summary["counts"]["expected_keys"] == summary["counts"]["consumed_keys"] == 2
    assert summary["counts"]["decision_identities"] == 2
    assert summary["window"] == {"first_rebalance": d1.isoformat(),
                                 "last_rebalance": d2.isoformat()}
    assert summary["variants"] == ["pit"]
    # immutable source evidence untouched by recording + audit + persistence
    assert {k: r.evidence_id for k, r in evidence.items()} == ids_before


def test_replay_audit_missing_and_stray_consumed_keys_block() -> None:
    # R6.3 "absent": an expected key the simulation never consumed blocks; a
    # consumption outside the expected set (cross-association shape) blocks.
    import pytest

    ext = _mod()
    _, _, _, _, d1, d2, evidence = _replay_fixture(ext)
    consumed = {}
    ext.record_consumption(consumed, ("pit", d1), evidence[("pit", d1)])
    with pytest.raises(ValueError, match="never consumed by the simulation") as exc:
        ext.validate_source_to_consumption(evidence, consumed, [("pit", d1), ("pit", d2)])
    assert "'pit'" in str(exc.value) and "2025, 11, 3" in str(exc.value)
    ext.record_consumption(consumed, ("pit", d2), evidence[("pit", d2)])
    ext.record_consumption(consumed, ("nonpit_diagnostic", d1), evidence[("pit", d1)])
    with pytest.raises(ValueError, match="outside the expected set"):
        ext.validate_source_to_consumption(evidence, consumed, [("pit", d1), ("pit", d2)])


def test_replay_audit_inconsistently_duplicated_consumption_blocks() -> None:
    # The two-pass happy path: a REPEATED IDENTICAL consumption of one key is
    # valid (weight_fn pass + decision-log pass); the same key consumed with a
    # DIFFERENT fingerprint is an inconsistent duplicate and raises (R6.3).
    import pytest

    ext = _mod()
    _, _, _, _, d1, d2, evidence = _replay_fixture(ext)
    r1, r2 = evidence[("pit", d1)], evidence[("pit", d2)]
    consumed = {}
    ext.record_consumption(consumed, ("pit", d1), r1)
    ext.record_consumption(consumed, ("pit", d1), r1)  # second pass: valid
    assert set(consumed) == {("pit", d1)}
    with pytest.raises(ValueError, match="inconsistently duplicated consumption") as exc:
        ext.record_consumption(consumed, ("pit", d1), r2)
    assert "2025-10-01" in str(exc.value)


def test_replay_audit_cross_date_swap_blocks_every_publishable_output(tmp_path) -> None:
    # Task 6.4 completion signal: a deliberate cross-date response/score swap —
    # invisible to prompt matching because the rendered prompt is IDENTICAL on
    # both dates — fails the audit BEFORE targets/equity/decision-log/metrics
    # exist. Nothing is persisted, and in main() the audit call sits after both
    # variant lines and before the first publication write.
    import inspect

    import pytest

    ext = _mod()
    macro_state, snapshot, amap, prompt, d1, d2, evidence = _replay_fixture(ext)
    swapped = {("pit", d1): evidence[("pit", d2)], ("pit", d2): evidence[("pit", d1)]}
    consumed = {}
    _consume_both_passes(ext, swapped, "pit", (d1, d2), macro_state, snapshot, amap,
                         consumed)
    expected = [("pit", d1), ("pit", d2)]
    with pytest.raises(ValueError, match="cross-associated or altered") as exc:
        ext.write_replay_audit_summary(evidence, consumed, expected, tmp_path)
    msg = str(exc.value)
    assert "'pit'" in msg and "2025-10-01" in msg
    assert "response_sha256" in msg and "score_p_memorized" in msg
    assert "evidence_id" in msg
    assert not (tmp_path / ext.REPLAY_AUDIT_NAME).exists()  # nothing persisted
    # ordering in main(): audit after both variant lines, before _dump_line
    # (the first targets/equity/decision-log write) and every metric artifact.
    src = inspect.getsource(ext.main)
    audit_at = src.index("write_replay_audit_summary")
    assert src.index("factor_nonpit_ext2026") < audit_at < src.index('_dump_line("factor"')


def test_replay_audit_altered_value_blocks() -> None:
    # R6.3 "altered": a single tampered consumed field (a score swap that keeps
    # everything else) raises naming the exact field, variant, and date.
    import dataclasses

    import pytest

    ext = _mod()
    _, _, _, _, d1, d2, evidence = _replay_fixture(ext)
    tampered = ext.with_evidence_id(dataclasses.replace(
        evidence[("pit", d1)],
        score_p_memorized=evidence[("pit", d2)].score_p_memorized))
    consumed = {}
    ext.record_consumption(consumed, ("pit", d1), tampered)
    with pytest.raises(ValueError, match="cross-associated or altered") as exc:
        ext.validate_source_to_consumption(evidence, consumed, [("pit", d1)])
    msg = str(exc.value)
    assert "score_p_memorized" in msg and "'pit'" in msg and "2025-10-01" in msg


def test_replay_audit_decision_identity_mismatch_blocks() -> None:
    # R6.4: the RESULTING decision identity must derive from the consumed
    # evidence — a swapped score or swapped loadings in the decision raises.
    from types import SimpleNamespace

    import pytest

    ext = _mod()
    _, _, _, _, d1, d2, evidence = _replay_fixture(ext)
    r1 = evidence[("pit", d1)]
    consumed = {}
    ext.record_consumption(consumed, ("pit", d1), r1)
    good_loadings = SimpleNamespace(loadings={a: 0.1 for a in _axes()})
    swapped_score = SimpleNamespace(parse_ok=True, p_memorized=0.8, steered=True,
                                    loadings=good_loadings)  # d2's score on d1
    with pytest.raises(ValueError, match="decision p_memorized does not match"):
        ext.record_decision_identity(consumed, ("pit", d1), r1, swapped_score)
    wrong_loadings = SimpleNamespace(
        parse_ok=True, p_memorized=0.2, steered=True,
        loadings=SimpleNamespace(loadings={a: 0.9 for a in _axes()}))
    with pytest.raises(ValueError, match="decision loadings do not match"):
        ext.record_decision_identity(consumed, ("pit", d1), r1, wrong_loadings)


def test_ac_6_3(tmp_path) -> None:
    test_replay_audit_missing_and_stray_consumed_keys_block()
    test_replay_audit_inconsistently_duplicated_consumption_blocks()
    test_replay_audit_cross_date_swap_blocks_every_publishable_output(tmp_path / "swap")
    test_replay_audit_altered_value_blocks()
    test_replay_audit_decision_identity_mismatch_blocks()


def test_ac_6_4(tmp_path) -> None:
    test_replay_audit_clean_run_passes_and_persists_summary(tmp_path / "clean")


# --------------------------------------------------------------------------- #
# Task 6.5 — duplicate-prompt and immutable-evidence regressions (defect 1)    #
# --------------------------------------------------------------------------- #


def _dup_prompt_fixture(ext):
    """dated_prompt_collisions: the real Oct/Nov-2025 collision shape — ONE
    rendered anonymized PIT prompt shared by two rebalance dates, for BOTH
    variants, with per-date responses, scores, loadings, origins, sources."""
    from datetime import date as _date

    import macro_framework as mf
    from macro_framework import factor_scoring as fs

    macro_state = {"cpi_yoy_z": 0.5, "t10y2y_z": -0.3, "hy_oas_z": 0.1}
    amap = mf.AssetMap.default()
    snapshot = [{"id": p, "category": c} for p, c in sorted(amap.categories.items())]
    prompt = fs.render_regime_loadings_prompt(macro_state, snapshot)
    d1, d2 = _date(2025, 10, 1), _date(2025, 11, 3)
    recs = []
    for variant in ("pit", "nonpit_diagnostic"):
        # d1 reconstructed from v1, d2 live: distinct response/score/loadings.
        recs.append(_build(ext, variant=variant, d=d1, pit=prompt, p=0.2,
                           response=ext.synth_loadings_reply({a: 0.1 for a in _axes()}),
                           loadings={a: 0.1 for a in _axes()}, segment="replayed_v1",
                           src="data/factor_loadings_v1.parquet", src_sha="a" * 64))
        recs.append(_build(ext, variant=variant, d=d2, pit=prompt, p=0.8,
                           response=ext.synth_loadings_reply({a: 0.9 for a in _axes()}),
                           loadings={a: 0.9 for a in _axes()}))
    expected = [(v, d) for v in ("pit", "nonpit_diagnostic") for d in (d1, d2)]
    return macro_state, snapshot, amap, prompt, d1, d2, recs, expected


def test_duplicate_prompt_dates_keep_distinct_evidence() -> None:
    # defect 1 shared boundary, fixture classes: dated_prompt_collisions,
    # manifest_failures — R6.1/R6.2/R6.3/R6.4/R8.7: identical prompt text on
    # different rebalance dates keeps distinct dated evidence, replay resolves
    # by exact (variant, date), and every integrity mutation produces its OWN
    # publication-blocking failure. Offline: no NIM key, no network, no DB.
    import dataclasses
    import hashlib
    from datetime import date as _date

    import pytest

    from macro_framework import factor_scoring as fs

    ext = _mod()
    macro_state, snapshot, amap, prompt, d1, d2, recs, expected = _dup_prompt_fixture(ext)

    # (a) natural-key uniqueness: one immutable entry per (variant, date);
    # both collision dates survive — duplicate prompt TEXT never collapses.
    evidence = ext.validate_evidence_records(recs, expected)
    assert set(evidence) == set(expected) and len(evidence) == 4
    for variant in ("pit", "nonpit_diagnostic"):
        r1, r2 = evidence[(variant, d1)], evidence[(variant, d2)]
        assert r1.pit_prompt_text == r2.pit_prompt_text == prompt
        assert r1.pit_prompt_sha256 == r2.pit_prompt_sha256
        assert r1.response_text != r2.response_text
        assert (r1.score_p_memorized, r2.score_p_memorized) == (0.2, 0.8)
        assert r1.loading_growth == 0.1 and r2.loading_growth == 0.9
        assert r1.evidence_id != r2.evidence_id

    # (b) exact hashes: every sha256 recomputes from the UTF-8 text fields
    for rec in recs:
        for text, sha in ((rec.pit_prompt_text, rec.pit_prompt_sha256),
                          (rec.source_prompt_text, rec.source_prompt_sha256),
                          (rec.response_text, rec.response_sha256)):
            assert sha == hashlib.sha256(text.encode("utf-8")).hexdigest()

    # (c) reconstructed origin: replayed-v1 rows never claim raw provenance
    assert evidence[("pit", d1)].response_origin == "reconstructed_from_v1_loadings"
    assert evidence[("pit", d1)].source_artifact == "data/factor_loadings_v1.parquet"
    assert evidence[("pit", d1)].source_artifact_sha256 == "a" * 64
    assert evidence[("pit", d2)].response_origin == "raw_nim"
    assert evidence[("pit", d2)].source_artifact is None

    # (d) generation-failure retention: a failed date is a RETAINED record
    d3 = _date(2025, 12, 1)
    failed = _build(ext, d=d3, pit=prompt, response="", loadings=None, p=None,
                    fail="TimeoutError")
    assert failed.response_origin == "generation_failed"
    assert failed.loadings_parse_ok is False
    assert failed.score_fail_reason == "TimeoutError"
    ext.validate_evidence_records([failed], [("pit", d3)])  # not a dropped row

    # (e) variant isolation: same date, two variants -> independent entries
    assert (evidence[("pit", d1)].evidence_id
            != evidence[("nonpit_diagnostic", d1)].evidence_id)
    assert (evidence[("pit", d1)].source_prompt_text
            != evidence[("nonpit_diagnostic", d1)].source_prompt_text)

    # (f) deterministic evidence identities: same inputs -> identical id;
    # any identity-bearing field change -> different id
    twin = _build(ext, d=d2, pit=prompt, p=0.8,
                  response=ext.synth_loadings_reply({a: 0.9 for a in _axes()}),
                  loadings={a: 0.9 for a in _axes()})
    assert twin.evidence_id == evidence[("pit", d2)].evidence_id
    bumped = ext.with_evidence_id(dataclasses.replace(
        evidence[("pit", d2)], score_p_memorized=0.81))
    assert bumped.evidence_id != evidence[("pit", d2)].evidence_id

    # Replay: the production weight_fn resolves by exact (variant, date), so
    # each collision date consumes ITS OWN response/score/loadings with zero
    # live model calls — the pre-fix later-date-wins prompt map (both dates
    # serving d2's values) is impossible.
    recorded = []

    class _Agent:
        asset_map = amap

        def views_to_bl(self, views, real_symbols):
            recorded.append({v.asset_long: v.expected_excess_annualized for v in views})
            return None, None

    def _boom(*a, **k):
        raise AssertionError("live model call during replay")

    prices = pd.DataFrame({"XX": [1.0, 2.0], "YY": [1.0, 2.0]})
    consumed = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fs.FactorScorer, "score", _boom)
        mp.setattr(fs.FactorScorer, "score_many", _boom)
        weight_fn = ext.make_dated_replay_weight_fn(
            variant="pit", evidence=evidence, agent=_Agent(),
            build_inputs=lambda ctx: (macro_state, snapshot, ctx["rebalance_date"], None),
            combine=lambda ctx, P, Q: pd.Series(1.0, index=ctx["prices"].columns),
            consumed=consumed)
        for d in (d1, d2):
            weight_fn({"rebalance_date": pd.Timestamp(d), "prices": prices})
    assert len(recorded) == 2 and recorded[0] != recorded[1]  # distinct decisions
    assert any(abs(t) > 1e-12 for t in recorded[0].values())
    for d in (d1, d2):  # each date consumed exactly its own dated evidence
        assert (consumed[("pit", d)]["evidence"]
                == ext.consumption_fingerprint(evidence[("pit", d)]))
    ext.validate_source_to_consumption(evidence, consumed, [("pit", d1), ("pit", d2)])

    # Negative paths — each integrity mutation raises its OWN
    # publication-blocking ReplayValidationError (a ValueError, R6.3).

    # missing (variant, date): resolution fails immediately, never degrades
    with pytest.raises(ValueError, match="missing dated evidence"):
        ext.resolve_dated_evidence(evidence, "pit", _date(2026, 1, 2))

    # missing expected key at validation time
    with pytest.raises(ValueError, match="missing expected evidence key"):
        ext.validate_evidence_records(recs[:3], expected)

    # duplicated (variant, date) key
    with pytest.raises(ValueError, match="duplicate evidence key"):
        ext.validate_evidence_records([*recs, recs[0]], expected)

    # cross-DATE swap: d1 consumes d2's record — invisible to prompt matching
    # because the rendered prompt is identical; only dated equality catches it
    consumed_swap = {}
    ext.record_consumption(consumed_swap, ("pit", d1), evidence[("pit", d2)])
    ext.record_consumption(consumed_swap, ("pit", d2), evidence[("pit", d1)])
    with pytest.raises(ValueError, match="cross-associated or altered"):
        ext.validate_source_to_consumption(
            evidence, consumed_swap, [("pit", d1), ("pit", d2)])

    # cross-VARIANT swap: a pit record served under nonpit_diagnostic
    consumed_var = {}
    ext.record_consumption(consumed_var, ("nonpit_diagnostic", d1), evidence[("pit", d1)])
    with pytest.raises(ValueError, match="cross-associated or altered"):
        ext.validate_source_to_consumption(
            evidence, consumed_var, [("nonpit_diagnostic", d1)])

    # hash mismatch: text mutated without rehashing never validates
    mangled = dataclasses.replace(
        evidence[("pit", d1)],
        response_text=evidence[("pit", d1)].response_text + " ")
    with pytest.raises(ValueError, match="response_sha256 mismatch"):
        ext.validate_evidence_records([mangled], [("pit", d1)])


def test_ac_8_7() -> None:
    test_duplicate_prompt_dates_keep_distinct_evidence()


# --------------------------------------------------------------------------- #
# Task 6.6 — run-local reader / legacy / differential / SSR metric records    #
# --------------------------------------------------------------------------- #


def _completed_market_snapshot(
    tmp_path,
    index: pd.DatetimeIndex,
    *,
    spy_missing: tuple[pd.Timestamp, ...] = (),
):
    """Small hash-valid completed snapshot carrying BIL/SPY total-return levels."""
    import hashlib
    import json

    import numpy as np

    snapshot = tmp_path / "market_total_return_fx_2026-06-30_v1"
    snapshot.mkdir()
    spy = pd.Series(
        100.0 * np.cumprod(1.0 + np.linspace(-0.0002, 0.0008, len(index))),
        index=index,
    )
    if spy_missing:
        spy.loc[list(spy_missing)] = np.nan
    tables = {
        "basket_adjusted_close_local.parquet": pd.DataFrame(
            {"SWDA.L": np.linspace(100.0, 110.0, len(index)),
             "XLK": np.linspace(200.0, 220.0, len(index)),
             "IAU": np.linspace(40.0, 42.0, len(index))},
            index=index,
        ),
        "cash_market_total_return.parquet": pd.DataFrame(
            {"BIL": 100.0 * np.cumprod(1.0 + np.linspace(0.00005, 0.00015, len(index))),
             "SPY": spy},
            index=index,
        ),
        "fx_usd_per_gbp.parquet": pd.DataFrame(
            {"USD_per_GBP": np.linspace(1.20, 1.30, len(index))}, index=index
        ),
    }
    files = {}
    for name, frame in tables.items():
        path = snapshot / name
        frame.to_parquet(path)
        files[name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(frame),
        }
    manifest = {
        "schema": "market_snapshot.v1",
        "snapshot_id": snapshot.name,
        "cash_symbol": "BIL",
        "benchmark_symbol": "SPY",
        "total_return_field": (
            "yfinance auto_adjust=True Close (adjusted total-return level, dividends reinvested)"
        ),
        "requested_coverage": {
            "start": index.min().date().isoformat(),
            "end": index.max().date().isoformat(),
        },
        "files": files,
        "overlap_revisions": {"preceding_snapshot": None},
        "completed": True,
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    (snapshot / "COMPLETED").write_text("complete\n")
    return snapshot


def test_factor_metric_records_require_exact_bil_portfolio_anchor(tmp_path) -> None:
    import numpy as np
    import pytest

    ext = _mod()
    return_index = pd.bdate_range("2024-01-02", periods=310)
    value_index = return_index.insert(0, pd.Timestamp("2024-01-01"))
    returns = pd.Series(0.0004, index=return_index)
    value = pd.Series(
        10_000.0 * np.r_[1.0, np.cumprod(1.0 + returns.to_numpy())],
        index=value_index,
    )
    snapshot_index = return_index.insert(0, pd.Timestamp("2023-12-29"))
    snapshot = _completed_market_snapshot(tmp_path, snapshot_index)

    with pytest.raises(ValueError, match=r"missing .*2024-01-01"):
        ext.build_factor_metric_records(value, value, snapshot_dir=snapshot, n_boot=25)


def test_factor_metric_records_use_completed_snapshot_cash_and_shared_contracts(
    tmp_path,
) -> None:
    import inspect
    import json
    import math

    import numpy as np
    import pytest

    import macro_framework as mf

    ext = _mod()
    index = pd.bdate_range("2021-01-04", periods=620)
    rng = np.random.default_rng(606)
    pit_returns = pd.Series(rng.normal(0.0004, 0.006, len(index) - 1), index=index[1:])
    nonpit_returns = pd.Series(rng.normal(0.0007, 0.0065, len(index) - 1), index=index[1:])
    pit_value = pd.Series(
        10_000.0 * np.r_[1.0, np.cumprod(1.0 + pit_returns.to_numpy())], index=index
    )
    nonpit_value = pd.Series(
        10_000.0 * np.r_[1.0, np.cumprod(1.0 + nonpit_returns.to_numpy())], index=index
    )
    snapshot = _completed_market_snapshot(tmp_path, index)

    bundle = ext.build_factor_metric_records(
        pit_value,
        nonpit_value,
        snapshot_dir=snapshot,
        n_boot=25,
        seed=17,
        alpha=0.10,
    )
    out = ext.write_factor_metric_records(
        bundle,
        tmp_path / "run",
        pit_value=pit_value,
        nonpit_value=nonpit_value,
        snapshot_dir=snapshot,
    )
    persisted = json.loads(out.read_text())

    assert persisted["schema"] == "factor_run.metric_records.v1"
    assert len(persisted["record_sha256s"]) == len(persisted["records"]) == 9
    assert persisted["content_sha256"] == bundle["content_sha256"]
    assert persisted["market_snapshot"]["snapshot_id"] == snapshot.name
    assert persisted["market_snapshot"]["cash_benchmark_id"] == "BIL"
    assert persisted["market_snapshot"]["cash_semantics"] == "adjusted_total_return"
    assert persisted["ssr_settings"] == {
        "alpha": 0.10,
        "n_boot": 25,
        "periods_per_year": 252,
        "seed": 17,
        "sr_star": 0.0,
        "window": 252,
    }

    rows = bundle["records"]
    assert len(rows) == 9
    assert [row["schema"] for row in rows].count(mf.READER_SCHEMA) == 2
    assert [row["schema"] for row in rows].count(mf.LEGACY_SCHEMA) == 2
    assert [row["schema"] for row in rows].count(mf.DIFFERENTIAL_SCHEMA) == 1
    assert [row["schema"] for row in rows].count(mf.ATTRIBUTION_SCHEMA) == 2
    assert [row["schema"] for row in rows].count(mf.CRISIS_SCHEMA) == 2
    by_key = {(row["portfolio_id"], row["schema"]): row for row in rows}

    cash = ext.load_completed_snapshot_bil_returns(
        snapshot, pit_returns.index, anchor=pit_value.index[0]
    )[0]
    pit_excess = mf.portfolio_excess_returns(pit_returns, cash)
    pit_ssr = mf.ssr_inference(pit_excess, n_boot=25, seed=17, alpha=0.10)
    pit_reader = by_key[("factor_pit_ext2026", mf.READER_SCHEMA)]
    assert pit_reader["ssr_ssr"] == pytest.approx(pit_ssr.result.ssr)
    assert pit_reader["sharpe"] == pytest.approx(
        float(pit_excess.mean() / pit_excess.std(ddof=1)) * math.sqrt(252)
    )
    assert (pit_reader["start"], pit_reader["end"], pit_reader["n_obs"]) == (
        pit_returns.index[0], pit_returns.index[-1], len(pit_returns)
    )
    assert pit_reader["cash_benchmark_id"] == f"BIL@{snapshot.name}"
    assert pit_reader["currency_basis"] == "legacy_mixed_local_quotes"

    spread = mf.differential_returns(nonpit_returns, pit_returns)
    spread_ssr = mf.ssr_inference(spread, n_boot=25, seed=17, alpha=0.10)
    differential = by_key[("factor_nonpit_minus_pit_ext2026", mf.DIFFERENTIAL_SCHEMA)]
    assert differential["ssr_ssr"] == pytest.approx(spread_ssr.result.ssr)
    assert differential["sharpe"] == pytest.approx(
        float(spread.mean() / spread.std(ddof=1)) * math.sqrt(252)
    )
    assert differential["cash_benchmark_id"] == "not_applicable_direct_daily_spread"
    double_subtracted = mf.portfolio_excess_returns(spread, cash)
    assert differential["sharpe"] != pytest.approx(
        float(double_subtracted.mean() / double_subtracted.std(ddof=1)) * math.sqrt(252)
    )

    source = inspect.getsource(ext.main)
    assert "build_factor_metric_records" in source
    assert "write_factor_metric_records" in source
    assert "equity_metrics" not in source
    assert "_luck_row" not in source
    assert "head_to_head_report" not in source


# --------------------------------------------------------------------------- #
# Task 6.7 — strict attribution, crisis, and explicit-window record integration #
# --------------------------------------------------------------------------- #


def _assert_record_projects_typed_result(record, expected, *, prefix="") -> None:
    """Assert every dataclass field survives the producer projection exactly."""
    import dataclasses
    import math

    import pytest

    for field in dataclasses.fields(expected):
        actual = record[f"{prefix}{field.name}"]
        wanted = getattr(expected, field.name)
        if isinstance(wanted, float):
            if math.isnan(wanted):
                assert isinstance(actual, float) and math.isnan(actual), field.name
            else:
                assert actual == pytest.approx(wanted), field.name
        else:
            assert actual == wanted, field.name


def test_factor_records_project_strict_attribution_and_crisis_contracts(tmp_path) -> None:
    import inspect
    import json

    import numpy as np
    import pytest

    import macro_framework as mf

    ext = _mod()
    index = pd.bdate_range("2021-01-04", periods=620)
    rng = np.random.default_rng(607)
    pit_returns = pd.Series(rng.normal(0.0004, 0.006, len(index) - 1), index=index[1:])
    nonpit_returns = pd.Series(
        rng.normal(0.0007, 0.0065, len(index) - 1), index=index[1:]
    )
    pit_value = pd.Series(
        10_000.0 * np.r_[1.0, np.cumprod(1.0 + pit_returns.to_numpy())], index=index
    )
    nonpit_value = pd.Series(
        10_000.0 * np.r_[1.0, np.cumprod(1.0 + nonpit_returns.to_numpy())], index=index
    )
    snapshot = _completed_market_snapshot(tmp_path, index)

    bundle = ext.build_factor_metric_records(
        pit_value, nonpit_value, snapshot_dir=snapshot, n_boot=20, seed=23
    )
    persisted_path = ext.write_factor_metric_records(
        bundle,
        tmp_path / "run",
        pit_value=pit_value,
        nonpit_value=nonpit_value,
        snapshot_dir=snapshot,
    )
    persisted = json.loads(persisted_path.read_text())
    rows = bundle["records"]
    by_key = {(row["portfolio_id"], row["schema"]): row for row in rows}

    market_levels = pd.read_parquet(
        snapshot / "cash_market_total_return.parquet"
    )["SPY"]
    market_returns = mf.factor_returns_on(
        market_levels, pit_returns.index, anchor=pit_value.index[0]
    )
    expected_attribution = {
        "factor_pit_ext2026": mf.raw_market_model_attribution(
            pit_returns, market_returns
        ),
        "factor_nonpit_diagnostic_ext2026": mf.raw_market_model_attribution(
            nonpit_returns, market_returns
        ),
    }
    attr = by_key[("factor_pit_ext2026", mf.ATTRIBUTION_SCHEMA)]
    reader = by_key[("factor_pit_ext2026", mf.READER_SCHEMA)]

    for portfolio_id, expected_attr in expected_attribution.items():
        projected = by_key[(portfolio_id, mf.ATTRIBUTION_SCHEMA)]
        _assert_record_projects_typed_result(
            projected, expected_attr, prefix="raw_market_model_"
        )

    expected_attr = expected_attribution["factor_pit_ext2026"]
    assert reader["row_kind"] == "full"
    assert reader["raw_market_model_kind"] == "raw_market_model"
    assert attr["raw_market_model_kind"] == "raw_market_model"
    assert (attr["start"], attr["end"], attr["n_obs"]) == (
        expected_attr.start,
        expected_attr.end,
        expected_attr.n_obs,
    )
    assert attr["periods_per_year"] == 252
    assert not any("capm" in key.lower() or "jensen" in key.lower() for key in attr)
    assert snapshot.name in attr["source"]
    assert "cash_market_total_return.parquet#SPY" in attr["source"]
    assert bundle["market_snapshot"]["benchmark_id"] == "SPY"
    assert bundle["market_snapshot"]["benchmark_semantics"] == "adjusted_total_return"
    persisted_attr = next(
        row
        for row in persisted["records"]
        if row["portfolio_id"] == "factor_pit_ext2026"
        and row["schema"] == mf.ATTRIBUTION_SCHEMA
    )
    assert persisted_attr["start"] == expected_attr.start.isoformat()
    assert persisted_attr["end"] == expected_attr.end.isoformat()
    assert persisted_attr["n_obs"] == expected_attr.n_obs

    expected_crises = {
        "factor_pit_ext2026": mf.crisis_metrics(
            pit_value, "2022-01-01", "2022-12-31"
        ),
        "factor_nonpit_diagnostic_ext2026": mf.crisis_metrics(
            nonpit_value, "2022-01-01", "2022-12-31"
        ),
    }
    for portfolio_id, expected_crisis in expected_crises.items():
        assert expected_crisis is not None
        crisis = by_key[(portfolio_id, mf.CRISIS_SCHEMA)]
        _assert_record_projects_typed_result(crisis, expected_crisis)
        assert (crisis["start"], crisis["end"], crisis["n_obs"]) == (
            expected_crisis.anchor,
            expected_crisis.actual_end,
            expected_crisis.n_returns,
        )

    for row in rows:
        assert row["window_label"] == (
            f"{pd.Timestamp(row['start']).date()}..{pd.Timestamp(row['end']).date()}"
        )

    source = inspect.getsource(ext.build_factor_metric_records)
    assert "raw_market_model_attribution" in source
    assert "build_attribution_record" in source
    assert "crisis_metrics" in source
    assert "build_crisis_record" in source
    assert "OLS" not in source
    assert "head_to_head_report" not in source


def test_short_snapshot_attribution_emits_performance_only_and_actual_window(
    tmp_path,
) -> None:
    import numpy as np

    import macro_framework as mf

    ext = _mod()
    index = pd.bdate_range("2021-01-04", periods=620)
    rng = np.random.default_rng(608)
    pit_returns = pd.Series(rng.normal(0.0004, 0.006, len(index) - 1), index=index[1:])
    nonpit_returns = pd.Series(
        rng.normal(0.0007, 0.0065, len(index) - 1), index=index[1:]
    )
    pit_value = pd.Series(
        10_000.0 * np.r_[1.0, np.cumprod(1.0 + pit_returns.to_numpy())], index=index
    )
    nonpit_value = pd.Series(
        10_000.0 * np.r_[1.0, np.cumprod(1.0 + nonpit_returns.to_numpy())], index=index
    )
    snapshot = _completed_market_snapshot(
        tmp_path, index, spy_missing=tuple(index[:45])
    )

    rows = ext.build_factor_metric_records(
        pit_value, nonpit_value, snapshot_dir=snapshot, n_boot=20
    )["records"]
    by_key = {(row["portfolio_id"], row["schema"]): row for row in rows}
    market_returns, _ = ext.load_completed_snapshot_market_returns(
        snapshot, pit_returns.index, value_index=pit_value.index
    )
    expected_attribution = {
        "factor_pit_ext2026": mf.raw_market_model_attribution(
            pit_returns.loc[market_returns.index], market_returns
        ),
        "factor_nonpit_diagnostic_ext2026": mf.raw_market_model_attribution(
            nonpit_returns.loc[market_returns.index], market_returns
        ),
    }

    for portfolio_id in ("factor_pit_ext2026", "factor_nonpit_diagnostic_ext2026"):
        reader = by_key[(portfolio_id, mf.READER_SCHEMA)]
        attr = by_key[(portfolio_id, mf.ATTRIBUTION_SCHEMA)]
        assert reader["row_kind"] == "performance_only"
        assert not any(key.startswith("raw_market_model_") for key in reader)
        _assert_record_projects_typed_result(
            attr,
            expected_attribution[portfolio_id],
            prefix="raw_market_model_",
        )
        assert attr["start"] == index[46]
        assert attr["end"] == reader["end"]
        assert attr["n_obs"] < reader["n_obs"]
        assert attr["window_label"] == f"{index[46].date()}..{index[-1].date()}"


def test_build_rejects_mutated_nonpit_shortened_attribution_before_persistence(
    tmp_path, monkeypatch
) -> None:
    import numpy as np
    import pytest

    import macro_framework as mf

    ext = _mod()
    index = pd.bdate_range("2021-01-04", periods=620)
    rng = np.random.default_rng(610)
    pit_returns = pd.Series(rng.normal(0.0004, 0.006, len(index) - 1), index=index[1:])
    nonpit_returns = pd.Series(
        rng.normal(0.0007, 0.0065, len(index) - 1), index=index[1:]
    )
    pit_value = pd.Series(
        10_000.0 * np.r_[1.0, np.cumprod(1.0 + pit_returns.to_numpy())], index=index
    )
    nonpit_value = pd.Series(
        10_000.0 * np.r_[1.0, np.cumprod(1.0 + nonpit_returns.to_numpy())], index=index
    )
    snapshot = _completed_market_snapshot(
        tmp_path, index, spy_missing=tuple(index[:45])
    )
    real_builder = mf.build_attribution_record
    mutations = (
        ("raw_market_model_beta", lambda value: float(value) + 0.25),
        ("raw_market_model_hac_maxlags", lambda value: int(value) + 1),
    )
    for field, replacement in mutations:
        def mutated_builder(meta, attribution, *, source):
            record = real_builder(meta, attribution, source=source)
            if meta.portfolio_id == "factor_nonpit_diagnostic_ext2026":
                record[field] = replacement(record[field])
            return record

        monkeypatch.setattr(mf, "build_attribution_record", mutated_builder)
        with pytest.raises(ValueError, match="shared attribution"):
            ext.build_factor_metric_records(
                pit_value, nonpit_value, snapshot_dir=snapshot, n_boot=10
            )
    assert not (tmp_path / "run" / ext.FACTOR_METRIC_RECORDS_NAME).exists()


def test_build_rejects_mutated_nonpit_crisis_result_before_persistence(
    tmp_path, monkeypatch
) -> None:
    import numpy as np
    import pytest

    import macro_framework as mf

    ext = _mod()
    index = pd.bdate_range("2021-01-04", periods=620)
    rng = np.random.default_rng(611)
    pit_returns = pd.Series(rng.normal(0.0004, 0.006, len(index) - 1), index=index[1:])
    nonpit_returns = pd.Series(
        rng.normal(0.0007, 0.0065, len(index) - 1), index=index[1:]
    )
    pit_value = pd.Series(
        10_000.0 * np.r_[1.0, np.cumprod(1.0 + pit_returns.to_numpy())], index=index
    )
    nonpit_value = pd.Series(
        10_000.0 * np.r_[1.0, np.cumprod(1.0 + nonpit_returns.to_numpy())], index=index
    )
    snapshot = _completed_market_snapshot(tmp_path, index)
    real_builder = mf.build_crisis_record

    def mutated_builder(meta, crisis, *, source):
        record = real_builder(meta, crisis, source=source)
        if meta.portfolio_id == "factor_nonpit_diagnostic_ext2026":
            record["episode_return"] = float(record["episode_return"]) + 0.10
        return record

    monkeypatch.setattr(mf, "build_crisis_record", mutated_builder)
    with pytest.raises(ValueError, match="shared crisis"):
        ext.build_factor_metric_records(
            pit_value, nonpit_value, snapshot_dir=snapshot, n_boot=10
        )
    assert not (tmp_path / "run" / ext.FACTOR_METRIC_RECORDS_NAME).exists()


def test_one_return_crisis_null_roundtrip_is_canonical_and_self_validating(
    tmp_path,
) -> None:
    import json

    import numpy as np

    import macro_framework as mf

    ext = _mod()
    index = pd.bdate_range("2021-01-04", periods=620)
    rng = np.random.default_rng(612)
    pit_returns = pd.Series(rng.normal(0.0004, 0.006, len(index) - 1), index=index[1:])
    nonpit_returns = pd.Series(
        rng.normal(0.0007, 0.0065, len(index) - 1), index=index[1:]
    )
    pit_value = pd.Series(
        10_000.0 * np.r_[1.0, np.cumprod(1.0 + pit_returns.to_numpy())], index=index
    )
    nonpit_value = pd.Series(
        10_000.0 * np.r_[1.0, np.cumprod(1.0 + nonpit_returns.to_numpy())], index=index
    )
    snapshot = _completed_market_snapshot(tmp_path, index)
    only_return_date = index[300]
    bundle = ext.build_factor_metric_records(
        pit_value,
        nonpit_value,
        snapshot_dir=snapshot,
        n_boot=10,
        crisis_start=only_return_date,
        crisis_end=only_return_date,
    )
    serialized_bundle = json.loads(
        json.dumps(ext._json_record_value(bundle), sort_keys=True, allow_nan=False)
    )
    crisis_rows = [
        row for row in serialized_bundle["records"] if row["schema"] == mf.CRISIS_SCHEMA
    ]
    assert len(crisis_rows) == 2
    assert all(row["n_returns"] == 1 for row in crisis_rows)
    assert all(row["volatility_ann"] is None for row in crisis_rows)
    ext._validate_factor_metric_record_bundle(serialized_bundle)

    out = ext.write_factor_metric_records(
        bundle,
        tmp_path / "roundtrip",
        pit_value=pit_value,
        nonpit_value=nonpit_value,
        snapshot_dir=snapshot,
        crisis_start=only_return_date,
        crisis_end=only_return_date,
    )
    persisted = json.loads(out.read_text())
    ext._validate_factor_metric_record_bundle(persisted)


def test_writer_validates_serialized_readback_before_creating_output_path(
    tmp_path, monkeypatch
) -> None:
    import pytest

    ext, bundle, pit_value, nonpit_value, snapshot = _factor_metric_bundle_for_writer_validation(tmp_path)
    real_validator = ext._validate_factor_metric_record_bundle
    validation_forms = []

    def reject_serialized(candidate):
        real_validator(candidate)
        validation_forms.append(type(candidate["records"][0]["start"]))
        if isinstance(candidate["records"][0]["start"], str):
            raise ValueError("injected serialized read-back rejection")

    monkeypatch.setattr(ext, "_validate_factor_metric_record_bundle", reject_serialized)
    out_dir = tmp_path / "serialized_rejection"
    with pytest.raises(ValueError, match="serialized read-back rejection"):
        ext.write_factor_metric_records(
            bundle,
            out_dir,
            pit_value=pit_value,
            nonpit_value=nonpit_value,
            snapshot_dir=snapshot,
        )
    assert validation_forms == [pd.Timestamp, str]
    assert not out_dir.exists()


def test_factor_records_stop_on_invalid_required_spy_coverage(tmp_path) -> None:
    import numpy as np
    import pytest

    ext = _mod()
    index = pd.bdate_range("2021-01-04", periods=620)
    returns = pd.Series(0.0004, index=index[1:])
    value = pd.Series(
        10_000.0 * np.r_[1.0, np.cumprod(1.0 + returns.to_numpy())], index=index
    )

    for case, missing, match in (
        ("internal", (index[300],), "missing .*required label"),
        ("endpoint", (index[-1],), "required performance end"),
    ):
        case_dir = tmp_path / case
        case_dir.mkdir()
        snapshot = _completed_market_snapshot(case_dir, index, spy_missing=missing)
        with pytest.raises(ValueError, match=match):
            ext.build_factor_metric_records(
                value, value, snapshot_dir=snapshot, n_boot=10
            )


def test_write_factor_records_rejects_schema_specific_tampering_before_persistence(
    tmp_path,
) -> None:
    import copy
    import dataclasses

    import numpy as np
    import pytest

    import macro_framework as mf

    ext = _mod()
    index = pd.bdate_range("2021-01-04", periods=620)
    returns = pd.Series(0.0004, index=index[1:])
    value = pd.Series(
        10_000.0 * np.r_[1.0, np.cumprod(1.0 + returns.to_numpy())], index=index
    )
    snapshot = _completed_market_snapshot(tmp_path, index)
    bundle = ext.build_factor_metric_records(
        value, value, snapshot_dir=snapshot, n_boot=10
    )

    binding_mutations = [
        ("attribution_top_start", mf.ATTRIBUTION_SCHEMA, "top_date", "start"),
        (
            "attribution_typed_start",
            mf.ATTRIBUTION_SCHEMA,
            "typed_date",
            "raw_market_model_start",
        ),
        ("attribution_top_end", mf.ATTRIBUTION_SCHEMA, "top_date", "end"),
        (
            "attribution_typed_end",
            mf.ATTRIBUTION_SCHEMA,
            "typed_date",
            "raw_market_model_end",
        ),
        ("attribution_top_count", mf.ATTRIBUTION_SCHEMA, "integer", "n_obs"),
        (
            "attribution_typed_count",
            mf.ATTRIBUTION_SCHEMA,
            "integer",
            "raw_market_model_n_obs",
        ),
        (
            "attribution_top_annualization",
            mf.ATTRIBUTION_SCHEMA,
            "integer",
            "periods_per_year",
        ),
        (
            "attribution_typed_annualization",
            mf.ATTRIBUTION_SCHEMA,
            "integer",
            "raw_market_model_periods_per_year",
        ),
        (
            "attribution_wrong_raw_label",
            mf.ATTRIBUTION_SCHEMA,
            "literal",
            "raw_market_model_kind",
        ),
        ("crisis_top_start", mf.CRISIS_SCHEMA, "top_date", "start"),
        ("crisis_typed_anchor", mf.CRISIS_SCHEMA, "typed_date", "anchor"),
        ("crisis_top_end", mf.CRISIS_SCHEMA, "top_date", "end"),
        ("crisis_typed_actual_end", mf.CRISIS_SCHEMA, "typed_date", "actual_end"),
        ("crisis_top_count", mf.CRISIS_SCHEMA, "integer", "n_obs"),
        ("crisis_typed_count", mf.CRISIS_SCHEMA, "integer", "n_returns"),
        ("crisis_window_label", mf.CRISIS_SCHEMA, "window_label", "window_label"),
    ]
    completeness_mutations = [
        (
            f"attribution_missing_raw_market_model_{field.name}",
            mf.ATTRIBUTION_SCHEMA,
            "delete",
            f"raw_market_model_{field.name}",
        )
        for field in dataclasses.fields(mf.MarketAttribution)
    ] + [
        (
            f"crisis_missing_{field.name}",
            mf.CRISIS_SCHEMA,
            "delete",
            field.name,
        )
        for field in dataclasses.fields(mf.CrisisMetrics)
    ]

    for case, schema, mutation, field in binding_mutations + completeness_mutations:
        tampered = copy.deepcopy(bundle)
        row = next(record for record in tampered["records"] if record["schema"] == schema)
        if mutation in ("top_date", "typed_date"):
            row[field] = pd.Timestamp(row[field]) + pd.Timedelta(days=1)
            if mutation == "top_date":
                row["window_label"] = (
                    f"{pd.Timestamp(row['start']).date()}.."
                    f"{pd.Timestamp(row['end']).date()}"
                )
        elif mutation == "integer":
            row[field] = int(row[field]) + 1
        elif mutation == "window_label":
            row[field] = "2022-01-01..2022-12-31"
        elif mutation == "literal":
            row[field] = "capm"
        else:
            del row[field]
        _refresh_factor_bundle_hash(ext, tampered)

        out_dir = tmp_path / "run" / case
        with pytest.raises(ValueError):
            ext.write_factor_metric_records(
                tampered,
                out_dir,
                pit_value=value,
                nonpit_value=value,
                snapshot_dir=snapshot,
            )
        assert not (out_dir / ext.FACTOR_METRIC_RECORDS_NAME).exists(), case


def _factor_metric_trusted_writer_case(tmp_path, *, shortened=False):
    import numpy as np

    ext = _mod()
    index = pd.bdate_range("2021-01-04", periods=620)
    rng = np.random.default_rng(609)
    pit_returns = pd.Series(rng.normal(0.0004, 0.006, len(index) - 1), index=index[1:])
    nonpit_returns = pd.Series(
        rng.normal(0.0007, 0.0065, len(index) - 1), index=index[1:]
    )
    pit_value = pd.Series(
        10_000.0 * np.r_[1.0, np.cumprod(1.0 + pit_returns.to_numpy())], index=index
    )
    nonpit_value = pd.Series(
        10_000.0 * np.r_[1.0, np.cumprod(1.0 + nonpit_returns.to_numpy())], index=index
    )
    missing = tuple(index[:45]) if shortened else ()
    snapshot = _completed_market_snapshot(tmp_path, index, spy_missing=missing)
    bundle = ext.build_factor_metric_records(
        pit_value, nonpit_value, snapshot_dir=snapshot, n_boot=10, seed=29
    )
    return ext, bundle, pit_value, nonpit_value, snapshot


def _factor_metric_bundle_for_writer_validation(tmp_path, *, shortened=False):
    return _factor_metric_trusted_writer_case(tmp_path, shortened=shortened)


def _refresh_factor_bundle_hash(ext, bundle) -> None:
    """Re-sign lineage only; immutable record digests keep row mutations detectable."""
    bundle["content_sha256"] = ext._factor_content_sha256(bundle)


def _resign_factor_bundle_payload(ext, bundle) -> None:
    """Model a producer mutation that coherently re-signs rows and bundle content."""
    bundle["record_sha256s"] = ext._factor_record_sha256s(bundle["records"])
    bundle["content_sha256"] = ext._factor_content_sha256(bundle)


def test_write_factor_records_rejects_resigned_shortened_attribution_drift(
    tmp_path,
) -> None:
    import copy

    import pytest

    import macro_framework as mf

    ext, bundle, pit_value, nonpit_value, snapshot = _factor_metric_bundle_for_writer_validation(tmp_path, shortened=True)
    for field, replacement in (
        ("raw_market_model_beta", lambda value: float(value) + 0.25),
        ("raw_market_model_hac_maxlags", lambda value: int(value) + 1),
    ):
        tampered = copy.deepcopy(bundle)
        row = next(
            record
            for record in tampered["records"]
            if record["portfolio_id"] == "factor_nonpit_diagnostic_ext2026"
            and record["schema"] == mf.ATTRIBUTION_SCHEMA
        )
        row[field] = replacement(row[field])
        _resign_factor_bundle_payload(ext, tampered)
        out_dir = tmp_path / "resigned_shortened_attribution" / field
        with pytest.raises(ValueError, match="shared-result lineage"):
            ext.write_factor_metric_records(
            tampered,
            out_dir,
            pit_value=pit_value,
            nonpit_value=nonpit_value,
            snapshot_dir=snapshot,
        )
        assert not out_dir.exists(), field


def test_write_factor_records_rejects_resigned_nonpit_crisis_drift(tmp_path) -> None:
    import copy

    import pytest

    import macro_framework as mf

    ext, bundle, pit_value, nonpit_value, snapshot = _factor_metric_bundle_for_writer_validation(tmp_path)
    tampered = copy.deepcopy(bundle)
    row = next(
        record
        for record in tampered["records"]
        if record["portfolio_id"] == "factor_nonpit_diagnostic_ext2026"
        and record["schema"] == mf.CRISIS_SCHEMA
    )
    row["episode_return"] = float(row["episode_return"]) + 0.10
    _resign_factor_bundle_payload(ext, tampered)
    out_dir = tmp_path / "resigned_nonpit_crisis"
    with pytest.raises(ValueError, match="shared-result lineage"):
        ext.write_factor_metric_records(
            tampered,
            out_dir,
            pit_value=pit_value,
            nonpit_value=nonpit_value,
            snapshot_dir=snapshot,
        )
    assert not out_dir.exists()


def test_write_factor_records_rejects_missing_duplicate_extra_and_wrong_portfolio_catalog(
    tmp_path,
) -> None:
    import copy

    import pytest

    import macro_framework as mf

    ext, bundle, pit_value, nonpit_value, snapshot = _factor_metric_bundle_for_writer_validation(tmp_path)

    def first(schema):
        return next(row for row in bundle["records"] if row["schema"] == schema)

    mutations = {
        "missing_all_attribution": lambda candidate: candidate.__setitem__(
            "records",
            [row for row in candidate["records"] if row["schema"] != mf.ATTRIBUTION_SCHEMA],
        ),
        "missing_one_crisis": lambda candidate: candidate["records"].remove(
            next(row for row in candidate["records"] if row["schema"] == mf.CRISIS_SCHEMA)
        ),
        "duplicate_attribution": lambda candidate: candidate["records"].append(
            copy.deepcopy(
                next(row for row in candidate["records"] if row["schema"] == mf.ATTRIBUTION_SCHEMA)
            )
        ),
        "extra_attribution": lambda candidate: candidate["records"].append(
            {
                **copy.deepcopy(first(mf.ATTRIBUTION_SCHEMA)),
                "portfolio_id": "factor_unexpected_ext2026",
            }
        ),
        "wrong_portfolio_attribution": lambda candidate: next(
            row for row in candidate["records"] if row["schema"] == mf.ATTRIBUTION_SCHEMA
        ).__setitem__("portfolio_id", "factor_nonpit_minus_pit_ext2026"),
        "wrong_portfolio_crisis": lambda candidate: next(
            row for row in candidate["records"] if row["schema"] == mf.CRISIS_SCHEMA
        ).__setitem__("portfolio_id", "factor_nonpit_minus_pit_ext2026"),
    }

    for case, mutate in mutations.items():
        tampered = copy.deepcopy(bundle)
        mutate(tampered)
        _refresh_factor_bundle_hash(ext, tampered)
        out_dir = tmp_path / "catalog" / case
        with pytest.raises(ValueError):
            ext.write_factor_metric_records(
            tampered,
            out_dir,
            pit_value=pit_value,
            nonpit_value=nonpit_value,
            snapshot_dir=snapshot,
        )
        assert not (out_dir / ext.FACTOR_METRIC_RECORDS_NAME).exists(), case


def test_write_factor_records_requires_separate_attribution_for_performance_only_reader(
    tmp_path,
) -> None:
    import copy

    import pytest

    import macro_framework as mf

    ext, bundle, pit_value, nonpit_value, snapshot = _factor_metric_bundle_for_writer_validation(tmp_path, shortened=True)
    tampered = copy.deepcopy(bundle)
    pit_reader = next(
        row
        for row in tampered["records"]
        if row["portfolio_id"] == "factor_pit_ext2026" and row["schema"] == mf.READER_SCHEMA
    )
    assert pit_reader["row_kind"] == "performance_only"
    tampered["records"] = [
        row
        for row in tampered["records"]
        if not (
            row["portfolio_id"] == "factor_pit_ext2026"
            and row["schema"] == mf.ATTRIBUTION_SCHEMA
        )
    ]
    _refresh_factor_bundle_hash(ext, tampered)

    out_dir = tmp_path / "performance_only_missing_attribution"
    with pytest.raises(ValueError):
        ext.write_factor_metric_records(
            tampered,
            out_dir,
            pit_value=pit_value,
            nonpit_value=nonpit_value,
            snapshot_dir=snapshot,
        )
    assert not (out_dir / ext.FACTOR_METRIC_RECORDS_NAME).exists()


def test_write_factor_records_rejects_full_reader_attribution_divergence(tmp_path) -> None:
    import copy

    import pytest

    import macro_framework as mf

    ext, bundle, pit_value, nonpit_value, snapshot = _factor_metric_bundle_for_writer_validation(tmp_path)
    tampered = copy.deepcopy(bundle)
    standalone = next(
        row
        for row in tampered["records"]
        if row["portfolio_id"] == "factor_pit_ext2026"
        and row["schema"] == mf.ATTRIBUTION_SCHEMA
    )
    standalone["raw_market_model_beta"] = float(standalone["raw_market_model_beta"]) + 0.25
    _refresh_factor_bundle_hash(ext, tampered)

    out_dir = tmp_path / "attribution_divergence"
    with pytest.raises(ValueError):
        ext.write_factor_metric_records(
            tampered,
            out_dir,
            pit_value=pit_value,
            nonpit_value=nonpit_value,
            snapshot_dir=snapshot,
        )
    assert not (out_dir / ext.FACTOR_METRIC_RECORDS_NAME).exists()


def test_write_factor_records_rejects_tampered_record_digest_catalog(tmp_path) -> None:
    import copy

    import pytest

    ext, bundle, pit_value, nonpit_value, snapshot = _factor_metric_bundle_for_writer_validation(tmp_path)
    for case, mutate in {
        "missing": lambda candidate: candidate["record_sha256s"].pop(),
        "forged": lambda candidate: candidate["record_sha256s"].__setitem__(0, "0" * 64),
    }.items():
        tampered = copy.deepcopy(bundle)
        mutate(tampered)
        _refresh_factor_bundle_hash(ext, tampered)
        out_dir = tmp_path / "record_digests" / case
        with pytest.raises(ValueError):
            ext.write_factor_metric_records(
            tampered,
            out_dir,
            pit_value=pit_value,
            nonpit_value=nonpit_value,
            snapshot_dir=snapshot,
        )
        assert not (out_dir / ext.FACTOR_METRIC_RECORDS_NAME).exists(), case


def test_write_factor_records_rejects_semantically_invalid_typed_payloads(
    tmp_path,
) -> None:
    import copy

    import pytest

    import macro_framework as mf

    ext, bundle, pit_value, nonpit_value, snapshot = _factor_metric_bundle_for_writer_validation(tmp_path)
    mutations = {
        "attribution_negative_hac_lags": (
            mf.ATTRIBUTION_SCHEMA,
            lambda row: row.__setitem__("raw_market_model_hac_maxlags", -1),
        ),
        "crisis_inverted_requested_window": (
            mf.CRISIS_SCHEMA,
            lambda row: (
                row.__setitem__("requested_start", pd.Timestamp(row["requested_end"]) + pd.Timedelta(days=1)),
                row.__setitem__("requested_end", pd.Timestamp(row["requested_start"]) - pd.Timedelta(days=2)),
            ),
        ),
        "crisis_changed_episode_return": (
            mf.CRISIS_SCHEMA,
            lambda row: row.__setitem__("episode_return", float(row["episode_return"]) + 0.10),
        ),
    }

    for case, (schema, mutate) in mutations.items():
        tampered = copy.deepcopy(bundle)
        row = next(record for record in tampered["records"] if record["schema"] == schema)
        mutate(row)
        _refresh_factor_bundle_hash(ext, tampered)
        out_dir = tmp_path / "typed_payload" / case
        with pytest.raises(ValueError):
            ext.write_factor_metric_records(
            tampered,
            out_dir,
            pit_value=pit_value,
            nonpit_value=nonpit_value,
            snapshot_dir=snapshot,
        )
        assert not (out_dir / ext.FACTOR_METRIC_RECORDS_NAME).exists(), case


def test_write_factor_records_rejects_forged_or_deleted_lineage(tmp_path) -> None:
    import copy

    import pytest

    import macro_framework as mf

    ext, bundle, pit_value, nonpit_value, snapshot = _factor_metric_bundle_for_writer_validation(tmp_path)

    def forge_attribution_source(candidate):
        row = next(
            record for record in candidate["records"] if record["schema"] == mf.ATTRIBUTION_SCHEMA
        )
        row["source"] = "forged:portfolio.parquet|forged:SPY"

    def forge_source_stream_window(candidate):
        stream = candidate["source_streams"]["factor_pit_ext2026"]
        stream["start"] = pd.Timestamp(stream["start"]) + pd.Timedelta(days=1)
        stream["n_obs"] = int(stream["n_obs"]) - 1

    mutations = {
        "forged_attribution_source": forge_attribution_source,
        "deleted_market_snapshot": lambda candidate: candidate.__delitem__("market_snapshot"),
        "deleted_snapshot_benchmark_lineage": lambda candidate: candidate["market_snapshot"].__delitem__(
            "benchmark_file_sha256"
        ),
        "deleted_attribution_result_lineage": lambda candidate: candidate["source_streams"][
            "factor_pit_ext2026"
        ].__delitem__("attribution_result_sha256"),
        "forged_crisis_result_lineage": lambda candidate: candidate["source_streams"][
            "factor_nonpit_diagnostic_ext2026"
        ].__setitem__("crisis_result_sha256", "0" * 64),
        "forged_source_stream_window": forge_source_stream_window,
        "forged_source_stream_artifact": lambda candidate: candidate["source_streams"][
            "factor_pit_ext2026"
        ].__setitem__("artifact", "forged_equity.parquet"),
    }

    for case, mutate in mutations.items():
        tampered = copy.deepcopy(bundle)
        mutate(tampered)
        if "market_snapshot" in tampered:
            _refresh_factor_bundle_hash(ext, tampered)
        out_dir = tmp_path / "lineage" / case
        with pytest.raises(ValueError):
            ext.write_factor_metric_records(
            tampered,
            out_dir,
            pit_value=pit_value,
            nonpit_value=nonpit_value,
            snapshot_dir=snapshot,
        )
        assert not (out_dir / ext.FACTOR_METRIC_RECORDS_NAME).exists(), case


# Round-2 coherent-resign regressions. These deliberately update every digest and
# dependent lineage field that the old self-authenticating validator trusted.
def _coherently_resign_factor_bundle(ext, bundle) -> None:
    import macro_framework as mf

    for portfolio_id in ("factor_pit_ext2026", "factor_nonpit_diagnostic_ext2026"):
        attribution = next(
            row
            for row in bundle["records"]
            if row["portfolio_id"] == portfolio_id
            and row["schema"] == mf.ATTRIBUTION_SCHEMA
        )
        crisis = next(
            row
            for row in bundle["records"]
            if row["portfolio_id"] == portfolio_id and row["schema"] == mf.CRISIS_SCHEMA
        )
        stream = bundle["source_streams"][portfolio_id]
        stream["attribution_result_sha256"] = ext._factor_typed_result_sha256(
            ext._factor_attribution_from_record(
                attribution, name=f"{portfolio_id}.coherent_attribution"
            )
        )
        stream["crisis_result_sha256"] = ext._factor_typed_result_sha256(
            ext._factor_crisis_from_record(
                crisis, name=f"{portfolio_id}.coherent_crisis"
            )
        )
    bundle["record_sha256s"] = ext._factor_record_sha256s(bundle["records"])
    bundle["content_sha256"] = ext._factor_content_sha256(bundle)


def _forge_snapshot_hash_lineage(ext, bundle) -> None:
    import macro_framework as mf

    snapshot = bundle["market_snapshot"]
    snapshot["manifest_sha256"] = "1" * 64
    snapshot["cash_file_sha256"] = "2" * 64
    snapshot["benchmark_file_sha256"] = "2" * 64
    snapshot_id = snapshot["snapshot_id"]
    cash_source = (
        f"market_snapshot:{snapshot_id}/{snapshot['cash_file']}"
        f"#BIL@{snapshot['cash_file_sha256']}"
    )
    market_source = (
        f"market_snapshot:{snapshot_id}/{snapshot['benchmark_file']}"
        f"#SPY@{snapshot['benchmark_file_sha256']}"
    )
    for portfolio_id in ("factor_pit_ext2026", "factor_nonpit_diagnostic_ext2026"):
        artifact = bundle["source_streams"][portfolio_id]["artifact"]
        artifact_source = f"scripts/extend_stream_2026.py:{artifact}"
        reader = next(
            row
            for row in bundle["records"]
            if row["portfolio_id"] == portfolio_id and row["schema"] == mf.READER_SCHEMA
        )
        attribution = next(
            row
            for row in bundle["records"]
            if row["portfolio_id"] == portfolio_id
            and row["schema"] == mf.ATTRIBUTION_SCHEMA
        )
        reader["source"] = f"{artifact_source}|{cash_source}|{market_source}"
        attribution["source"] = f"{artifact_source}|{market_source}"
    _coherently_resign_factor_bundle(ext, bundle)


def _shift_source_windows_coherently(ext, bundle, new_start) -> None:
    import macro_framework as mf

    new_start = pd.Timestamp(new_start)
    snapshot = bundle["market_snapshot"]
    snapshot["cash_start"] = new_start
    snapshot["cash_n_obs"] = int(snapshot["cash_n_obs"]) - 1
    for portfolio_id in ("factor_pit_ext2026", "factor_nonpit_diagnostic_ext2026"):
        stream = bundle["source_streams"][portfolio_id]
        stream["start"] = new_start
        stream["n_obs"] = int(stream["n_obs"]) - 1
        for schema in (mf.READER_SCHEMA, mf.LEGACY_SCHEMA):
            row = next(
                record
                for record in bundle["records"]
                if record["portfolio_id"] == portfolio_id and record["schema"] == schema
            )
            row["start"] = new_start
            row["n_obs"] = int(row["n_obs"]) - 1
            row["window_label"] = f"{new_start.date()}..{pd.Timestamp(row['end']).date()}"
    differential_stream = bundle["source_streams"]["factor_nonpit_minus_pit_ext2026"]
    differential_stream["start"] = new_start
    differential_stream["n_obs"] = int(differential_stream["n_obs"]) - 1
    differential = next(
        row for row in bundle["records"] if row["schema"] == mf.DIFFERENTIAL_SCHEMA
    )
    differential["start"] = new_start
    differential["n_obs"] = int(differential["n_obs"]) - 1
    differential["window_label"] = (
        f"{new_start.date()}..{pd.Timestamp(differential['end']).date()}"
    )
    _coherently_resign_factor_bundle(ext, bundle)


def test_writer_rejects_coherently_resigned_shortened_attribution_beta(tmp_path) -> None:
    import copy

    import pytest

    import macro_framework as mf

    ext, bundle, pit_value, nonpit_value, snapshot = (
        _factor_metric_trusted_writer_case(tmp_path, shortened=True)
    )
    tampered = copy.deepcopy(bundle)
    row = next(
        record
        for record in tampered["records"]
        if record["portfolio_id"] == "factor_nonpit_diagnostic_ext2026"
        and record["schema"] == mf.ATTRIBUTION_SCHEMA
    )
    row["raw_market_model_beta"] = float(row["raw_market_model_beta"]) + 0.25
    _coherently_resign_factor_bundle(ext, tampered)
    out_dir = tmp_path / "coherent_resign_beta"
    with pytest.raises(ValueError):
        ext.write_factor_metric_records(
            tampered,
            out_dir,
            pit_value=pit_value,
            nonpit_value=nonpit_value,
            snapshot_dir=snapshot,
        )
    assert not out_dir.exists()


def test_writer_rejects_coherently_resigned_shortened_attribution_hac_lag(
    tmp_path,
) -> None:
    import copy

    import pytest

    import macro_framework as mf

    ext, bundle, pit_value, nonpit_value, snapshot = (
        _factor_metric_trusted_writer_case(tmp_path, shortened=True)
    )
    tampered = copy.deepcopy(bundle)
    row = next(
        record
        for record in tampered["records"]
        if record["portfolio_id"] == "factor_nonpit_diagnostic_ext2026"
        and record["schema"] == mf.ATTRIBUTION_SCHEMA
    )
    row["raw_market_model_hac_maxlags"] = int(row["raw_market_model_hac_maxlags"]) + 1
    _coherently_resign_factor_bundle(ext, tampered)
    out_dir = tmp_path / "coherent_resign_hac_lag"
    with pytest.raises(ValueError):
        ext.write_factor_metric_records(
            tampered,
            out_dir,
            pit_value=pit_value,
            nonpit_value=nonpit_value,
            snapshot_dir=snapshot,
        )
    assert not out_dir.exists()


def test_writer_rejects_coherently_resigned_crisis_episode_return(tmp_path) -> None:
    import copy

    import pytest

    import macro_framework as mf

    ext, bundle, pit_value, nonpit_value, snapshot = (
        _factor_metric_trusted_writer_case(tmp_path)
    )
    tampered = copy.deepcopy(bundle)
    row = next(
        record
        for record in tampered["records"]
        if record["portfolio_id"] == "factor_nonpit_diagnostic_ext2026"
        and record["schema"] == mf.CRISIS_SCHEMA
    )
    row["episode_return"] = float(row["episode_return"]) + 0.10
    _coherently_resign_factor_bundle(ext, tampered)
    out_dir = tmp_path / "coherent_resign_crisis"
    with pytest.raises(ValueError):
        ext.write_factor_metric_records(
            tampered,
            out_dir,
            pit_value=pit_value,
            nonpit_value=nonpit_value,
            snapshot_dir=snapshot,
        )
    assert not out_dir.exists()


def test_writer_rejects_coherently_forged_market_snapshot_hashes(tmp_path) -> None:
    import copy

    import pytest

    ext, bundle, pit_value, nonpit_value, snapshot = (
        _factor_metric_trusted_writer_case(tmp_path)
    )
    tampered = copy.deepcopy(bundle)
    _forge_snapshot_hash_lineage(ext, tampered)
    out_dir = tmp_path / "coherent_resign_snapshot_hashes"
    with pytest.raises(ValueError):
        ext.write_factor_metric_records(
            tampered,
            out_dir,
            pit_value=pit_value,
            nonpit_value=nonpit_value,
            snapshot_dir=snapshot,
        )
    assert not out_dir.exists()


def test_writer_rejects_coherently_shifted_source_windows(tmp_path) -> None:
    import copy

    import pytest

    ext, bundle, pit_value, nonpit_value, snapshot = (
        _factor_metric_trusted_writer_case(tmp_path, shortened=True)
    )
    tampered = copy.deepcopy(bundle)
    _shift_source_windows_coherently(ext, tampered, pit_value.index[2])
    out_dir = tmp_path / "coherent_resign_source_windows"
    with pytest.raises(ValueError):
        ext.write_factor_metric_records(
            tampered,
            out_dir,
            pit_value=pit_value,
            nonpit_value=nonpit_value,
            snapshot_dir=snapshot,
        )
    assert not out_dir.exists()


# --------------------------------------------------------------------------- #
# Task 6.8 — Factor run-local record parity (field-for-field reproduction)    #
# --------------------------------------------------------------------------- #

#: Documented parity tolerances. Provenance strings, dates, counts, labels, and
#: inference settings must reproduce EXACTLY. Deterministic closed-form
#: arithmetic recomputed here from the same float inputs reproduces to 1e-12
#: relative. The regression plane is re-solved independently (numpy lstsq plus
#: a hand-rolled Bartlett/Newey-West sandwich instead of statsmodels), which
#: agrees to 1e-9 relative.
_PARITY_REL_ARITH = 1e-12
_PARITY_REL_REGRESSION = 1e-9
_PARITY_REGRESSION_FIELDS = frozenset(
    "raw_market_model_" + name
    for name in (
        "intercept_native_period",
        "intercept_ann_arithmetic",
        "intercept_se_hac",
        "intercept_t_hac",
        "beta",
        "r2",
    )
)


def _parity_stream_stats(stream, periods_per_year):
    """Mean/std/downside-RMS scalings of one daily stream, test-local formulas."""
    import math

    import numpy as np

    mean = float(stream.mean())
    std = float(stream.std(ddof=1))
    downside = np.minimum(stream.to_numpy(dtype=float), 0.0)
    rms = float(np.sqrt(np.mean(downside**2)))
    root = math.sqrt(periods_per_year)
    return {
        "sharpe": mean / std * root,
        "sortino": mean / rms * root,
        "ann_vol": std * root,
        "downside_rms": rms,
    }


def _parity_market_model(portfolio_returns, market_returns):
    """Independent OLS + Bartlett/Newey-West(5) recompute of the shared
    raw-market-model attribution fields (no statsmodels)."""
    import numpy as np

    y = portfolio_returns.to_numpy(dtype=float)
    design = np.column_stack((np.ones(len(y)), market_returns.to_numpy(dtype=float)))
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    r2 = 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    lags = 5
    xtx_inv = np.linalg.inv(design.T @ design)
    scores = design * resid[:, None]
    meat = scores.T @ scores
    for k in range(1, lags + 1):
        gamma = scores[k:].T @ scores[:-k]
        meat = meat + (1.0 - k / (lags + 1.0)) * (gamma + gamma.T)
    cov = xtx_inv @ meat @ xtx_inv
    se = float(np.sqrt(cov[0, 0]))
    intercept = float(coef[0])
    return {
        "raw_market_model_kind": "raw_market_model",
        "raw_market_model_intercept_native_period": intercept,
        "raw_market_model_intercept_ann_arithmetic": intercept * 252,
        "raw_market_model_intercept_se_hac": se,
        "raw_market_model_intercept_t_hac": intercept / se,
        "raw_market_model_beta": float(coef[1]),
        "raw_market_model_r2": r2,
        "raw_market_model_n_obs": len(portfolio_returns),
        "raw_market_model_start": portfolio_returns.index[0],
        "raw_market_model_end": portfolio_returns.index[-1],
        "raw_market_model_periods_per_year": 252,
        "raw_market_model_hac_maxlags": 5,
    }


def _parity_expected_records(pit_value, nonpit_value, snapshot):
    """Recompute every checked record field from the fixture equity/cash/benchmark
    inputs alone — nothing from the producer bundle participates."""
    import dataclasses
    import hashlib
    import math

    import numpy as np

    import macro_framework as mf

    def window(returns):
        start, end = returns.index[0], returns.index[-1]
        return {
            "start": start,
            "end": end,
            "n_obs": len(returns),
            "window_label": f"{start.date()}..{end.date()}",
        }

    def elapsed_cagr(first_value, last_value, first_date, last_date):
        years = (last_date - first_date).days / 365.25
        return float((last_value / first_value) ** (1.0 / years) - 1.0)

    levels = pd.read_parquet(snapshot / "cash_market_total_return.parquet")
    file_sha = hashlib.sha256(
        (snapshot / "cash_market_total_return.parquet").read_bytes()
    ).hexdigest()
    manifest_sha = hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest()
    cash_source = (
        f"market_snapshot:{snapshot.name}/cash_market_total_return.parquet#BIL@{file_sha}"
    )
    market_source = (
        f"market_snapshot:{snapshot.name}/cash_market_total_return.parquet#SPY@{file_sha}"
    )
    prefix = "scripts/extend_stream_2026.py"
    artifacts = {
        "factor_pit_ext2026": f"{prefix}:factor_equity_ext2026.parquet",
        "factor_nonpit_diagnostic_ext2026": (
            f"{prefix}:factor_nonpit_diagnostic_equity_ext2026.parquet"
        ),
    }
    values = {
        "factor_pit_ext2026": pit_value,
        "factor_nonpit_diagnostic_ext2026": nonpit_value,
    }
    anchor = pit_value.index[0]
    return_index = pit_value.index[1:]
    cash_returns = levels["BIL"].loc[[anchor, *return_index]].pct_change().iloc[1:]
    market_returns = levels["SPY"].loc[[anchor, *return_index]].pct_change().iloc[1:]

    expected = {"market_snapshot": {
        "snapshot_id": snapshot.name,
        "manifest_sha256": manifest_sha,
        "cash_file_sha256": file_sha,
        "benchmark_file_sha256": file_sha,
    }}
    portfolio_returns = {}
    for portfolio_id, value in values.items():
        returns = value.pct_change().dropna()
        portfolio_returns[portfolio_id] = returns
        raw = _parity_stream_stats(returns, 252)
        raw_cal = _parity_stream_stats(returns, 365)
        excess = returns - cash_returns
        excess_stats = _parity_stream_stats(excess, 252)
        ssr = mf.ssr_inference(excess, n_boot=10, seed=29)
        ssr_fields = {
            "ssr_" + f.name: getattr(ssr.result, f.name)
            for f in dataclasses.fields(mf.SSRResult)
        }
        ssr_fields.update(
            ("ssr_" + f.name, getattr(ssr, f.name))
            for f in dataclasses.fields(mf.SSRInference)
            if f.name != "result"
        )
        total_return = float(value.iloc[-1] / value.iloc[0] - 1.0)
        cagr = elapsed_cagr(
            value.iloc[0], value.iloc[-1], value.index[0], value.index[-1]
        )
        maxdd = float((value / value.cummax() - 1.0).min())
        attribution_fields = _parity_market_model(returns, market_returns)
        common = {
            "portfolio_id": portfolio_id,
            "return_basis": "adjusted_total_return_equity",
            "cash_benchmark_id": f"BIL@{snapshot.name}",
            "currency_basis": "legacy_mixed_local_quotes",
        }
        expected[(portfolio_id, mf.READER_SCHEMA)] = {
            **common,
            **window(returns),
            "periods_per_year": 252,
            "source": f"{artifacts[portfolio_id]}|{cash_source}|{market_source}",
            "row_kind": "full",
            "total_return": total_return,
            "cagr": cagr,
            "ann_vol": raw["ann_vol"],
            "maxdd": maxdd,
            "downside_rms": raw["downside_rms"],
            "calmar": cagr / abs(maxdd),
            "sharpe": excess_stats["sharpe"],
            "sortino": excess_stats["sortino"],
            **ssr_fields,
            **attribution_fields,
        }
        expected[(portfolio_id, mf.LEGACY_SCHEMA)] = {
            **common,
            **window(returns),
            "periods_per_year": 365,
            "source": artifacts[portfolio_id],
            "total_return": total_return,
            "maxdd": maxdd,
            "downside_rms": raw["downside_rms"],
            "cagr_rows": float(
                (value.iloc[-1] / value.iloc[0]) ** (365.0 / len(value)) - 1.0
            ),
            "ann_vol_cal": raw_cal["ann_vol"],
            "sharpe_cal": raw_cal["sharpe"],
            "sortino_cal": raw_cal["sortino"],
        }
        expected[(portfolio_id, mf.LEGACY_SCHEMA)]["calmar_rows"] = (
            expected[(portfolio_id, mf.LEGACY_SCHEMA)]["cagr_rows"] / abs(maxdd)
        )
        expected[(portfolio_id, mf.ATTRIBUTION_SCHEMA)] = {
            **common,
            **window(returns),
            "periods_per_year": 252,
            "source": f"{artifacts[portfolio_id]}|{market_source}",
            **attribution_fields,
        }

        crisis_anchors = value.loc[value.index < pd.Timestamp("2022-01-01")]
        crisis_window = value.loc[
            (value.index >= pd.Timestamp("2022-01-01"))
            & (value.index <= pd.Timestamp("2022-12-31"))
        ]
        episode = pd.concat([value.loc[[crisis_anchors.index[-1]]], crisis_window])
        episode_returns = episode.pct_change().iloc[1:]
        expected[(portfolio_id, mf.CRISIS_SCHEMA)] = {
            **common,
            "start": crisis_anchors.index[-1],
            "end": episode_returns.index[-1],
            "n_obs": len(episode_returns),
            "window_label": (
                f"{crisis_anchors.index[-1].date()}..{episode_returns.index[-1].date()}"
            ),
            "periods_per_year": 252,
            "source": artifacts[portfolio_id],
            "requested_start": pd.Timestamp("2022-01-01"),
            "requested_end": pd.Timestamp("2022-12-31"),
            "anchor": crisis_anchors.index[-1],
            "first_return_date": episode_returns.index[0],
            "actual_end": episode_returns.index[-1],
            "episode_return": float(episode.iloc[-1] / episode.iloc[0] - 1.0),
            "boundary_anchored_max_drawdown": float(
                (episode / episode.cummax() - 1.0).min()
            ),
            "volatility_ann": float(episode_returns.std(ddof=1) * math.sqrt(252)),
            "n_returns": len(episode_returns),
        }

    spread = (
        portfolio_returns["factor_nonpit_diagnostic_ext2026"]
        - portfolio_returns["factor_pit_ext2026"]
    )
    spread_stats = _parity_stream_stats(spread, 252)
    spread_ssr = mf.ssr_inference(spread, n_boot=10, seed=29)
    spread_ssr_fields = {
        "ssr_" + f.name: getattr(spread_ssr.result, f.name)
        for f in dataclasses.fields(mf.SSRResult)
    }
    spread_ssr_fields.update(
        ("ssr_" + f.name, getattr(spread_ssr, f.name))
        for f in dataclasses.fields(mf.SSRInference)
        if f.name != "result"
    )
    curve = np.r_[1.0, np.cumprod(1.0 + spread.to_numpy(dtype=float))]
    curve_base = spread.index[0] - pd.offsets.BDay()
    spread_maxdd = float((curve / np.maximum.accumulate(curve) - 1.0).min())
    spread_cagr = elapsed_cagr(1.0, curve[-1], curve_base, spread.index[-1])
    expected[("factor_nonpit_minus_pit_ext2026", mf.DIFFERENTIAL_SCHEMA)] = {
        "portfolio_id": "factor_nonpit_minus_pit_ext2026",
        "return_basis": "direct_daily_return_spread_nonpit_minus_pit",
        "cash_benchmark_id": "not_applicable_direct_daily_spread",
        "currency_basis": "legacy_mixed_local_quotes",
        **window(spread),
        "periods_per_year": 252,
        "source": (
            f"{artifacts['factor_nonpit_diagnostic_ext2026']}"
            f"-{artifacts['factor_pit_ext2026']}"
        ),
        "total_return": float(curve[-1] - 1.0),
        "cagr": spread_cagr,
        "ann_vol": spread_stats["ann_vol"],
        "maxdd": spread_maxdd,
        "downside_rms": spread_stats["downside_rms"],
        "calmar": spread_cagr / abs(spread_maxdd),
        "sharpe": spread_stats["sharpe"],
        "sortino": spread_stats["sortino"],
        "endpoint_total_return_difference": float(
            np.prod(
                1.0 + portfolio_returns["factor_nonpit_diagnostic_ext2026"].to_numpy()
            )
            - np.prod(1.0 + portfolio_returns["factor_pit_ext2026"].to_numpy())
        ),
        **spread_ssr_fields,
    }
    return expected


def _assert_factor_record_parity(bundle, expected) -> None:
    """Field-for-field comparison of producer records against the test-local
    recomputation, within the documented tolerances. Raises AssertionError on
    the first missing-provenance key, mixed window, or diverging value."""
    import math

    import pytest

    import macro_framework as mf

    records = {(row["portfolio_id"], row["schema"]): row for row in bundle["records"]}
    expected_keys = {key for key in expected if isinstance(key, tuple)}
    assert set(records) == expected_keys, "record catalog diverges"
    snapshot_lineage = bundle["market_snapshot"]
    for field, wanted in expected["market_snapshot"].items():
        assert snapshot_lineage.get(field) == wanted, f"market_snapshot.{field}"

    for key, fields in expected.items():
        if not isinstance(key, tuple):
            continue
        record = records[key]
        for provenance in mf.REQUIRED_PROVENANCE:
            value = record.get(provenance)
            assert value is not None and value == value, (
                f"{key}: missing required provenance {provenance}"
            )
            if provenance in ("portfolio_id", "return_basis", "window_label",
                              "cash_benchmark_id", "source"):
                assert isinstance(value, str) and value.strip(), (
                    f"{key}: missing required provenance {provenance}"
                )
        for field, wanted in fields.items():
            assert field in record, f"{key}: record is missing {field}"
            actual = record[field]
            if isinstance(wanted, float) and not (
                isinstance(actual, bool) or not isinstance(actual, (int, float))
            ):
                if math.isnan(wanted):
                    assert math.isnan(float(actual)), f"{key}: {field}"
                    continue
                rel = (
                    _PARITY_REL_REGRESSION
                    if field in _PARITY_REGRESSION_FIELDS
                    else _PARITY_REL_ARITH
                )
                assert actual == pytest.approx(wanted, rel=rel, abs=1e-15), (
                    f"{key}: {field}"
                )
            else:
                assert actual == wanted, f"{key}: {field}"


def test_factor_records_reproduce_field_for_field_from_fixture_inputs(tmp_path) -> None:
    ext, bundle, pit_value, nonpit_value, snapshot = _factor_metric_trusted_writer_case(
        tmp_path
    )
    expected = _parity_expected_records(pit_value, nonpit_value, snapshot)
    _assert_factor_record_parity(bundle, expected)


def test_factor_record_parity_rejects_mixed_window_record(tmp_path) -> None:
    import copy

    import pytest

    import macro_framework as mf

    ext, bundle, pit_value, nonpit_value, snapshot = _factor_metric_trusted_writer_case(
        tmp_path
    )
    expected = _parity_expected_records(pit_value, nonpit_value, snapshot)
    tampered = copy.deepcopy(bundle)
    reader = next(
        row
        for row in tampered["records"]
        if row["portfolio_id"] == "factor_pit_ext2026"
        and row["schema"] == mf.READER_SCHEMA
    )
    crisis = next(
        row
        for row in tampered["records"]
        if row["portfolio_id"] == "factor_pit_ext2026"
        and row["schema"] == mf.CRISIS_SCHEMA
    )
    # one row now claims the crisis window while carrying performance statistics
    reader["start"] = crisis["start"]
    reader["window_label"] = (
        f"{pd.Timestamp(crisis['start']).date()}..{pd.Timestamp(reader['end']).date()}"
    )
    with pytest.raises(AssertionError, match="start|window_label"):
        _assert_factor_record_parity(tampered, expected)


def test_factor_record_parity_rejects_stale_reader_total_return(tmp_path) -> None:
    import copy

    import pytest

    import macro_framework as mf

    ext, bundle, pit_value, nonpit_value, snapshot = _factor_metric_trusted_writer_case(
        tmp_path
    )
    expected = _parity_expected_records(pit_value, nonpit_value, snapshot)
    tampered = copy.deepcopy(bundle)
    reader = next(
        row
        for row in tampered["records"]
        if row["portfolio_id"] == "factor_pit_ext2026"
        and row["schema"] == mf.READER_SCHEMA
    )
    # a stale figure from a month-earlier run of the same stream
    stale = float(pit_value.iloc[-22] / pit_value.iloc[0] - 1.0)
    assert stale != pytest.approx(reader["total_return"], rel=1e-6)
    reader["total_return"] = stale
    with pytest.raises(AssertionError, match="total_return"):
        _assert_factor_record_parity(tampered, expected)


def test_factor_record_parity_rejects_endpoint_substituted_differential(tmp_path) -> None:
    import copy

    import pytest

    import macro_framework as mf

    ext, bundle, pit_value, nonpit_value, snapshot = _factor_metric_trusted_writer_case(
        tmp_path
    )
    expected = _parity_expected_records(pit_value, nonpit_value, snapshot)
    tampered = copy.deepcopy(bundle)
    differential = next(
        row for row in tampered["records"] if row["schema"] == mf.DIFFERENTIAL_SCHEMA
    )
    # historical defect 9: the endpoint wealth gap is NOT the compounded spread
    endpoint = float(differential["endpoint_total_return_difference"])
    assert endpoint != pytest.approx(differential["total_return"], rel=1e-6)
    differential["total_return"] = endpoint
    with pytest.raises(AssertionError, match="total_return"):
        _assert_factor_record_parity(tampered, expected)


def test_factor_record_parity_rejects_missing_provenance(tmp_path) -> None:
    import copy

    import pytest

    import macro_framework as mf

    ext, bundle, pit_value, nonpit_value, snapshot = _factor_metric_trusted_writer_case(
        tmp_path
    )
    expected = _parity_expected_records(pit_value, nonpit_value, snapshot)
    for field in ("source", "cash_benchmark_id"):
        tampered = copy.deepcopy(bundle)
        attribution = next(
            row
            for row in tampered["records"]
            if row["portfolio_id"] == "factor_pit_ext2026"
            and row["schema"] == mf.ATTRIBUTION_SCHEMA
        )
        del attribution[field]
        with pytest.raises(AssertionError, match=f"missing required provenance {field}"):
            _assert_factor_record_parity(tampered, expected)


# --------------------------------------------------------------------------- #
# Task 6.10 — Factor run manifest, immutability, and completion-order tests    #
# --------------------------------------------------------------------------- #


def _staged_factor_run_case(tmp_path):
    """Stage every Factor run artifact for one temporary, fully valid bundle.

    Reuses the metric-record writer fixture (equity + completed snapshot) and
    builds two-date, two-variant dated evidence INSIDE the equity window so the
    bundle-level window and lineage binds hold end-to-end (R6.3, R7.2, R7.4).
    """
    import json
    from datetime import date as _date

    ext, bundle, pit_value, nonpit_value, snapshot = _factor_metric_trusted_writer_case(
        tmp_path
    )
    staging = tmp_path / "staging"
    staging.mkdir()

    d1, d2 = _date(2021, 6, 1), _date(2022, 6, 1)
    dates = (d1, d2)
    recs = []
    for variant in ("pit", "nonpit_diagnostic"):
        recs.append(_build(ext, variant=variant, d=d1, pit="SAME PROMPT",
                           response='{"inflation": 0.1}', p=0.2,
                           loadings={a: 0.1 for a in _axes()},
                           segment="replayed_v1",
                           src="data/factor_loadings_v1.parquet", src_sha="a" * 64))
        recs.append(_build(ext, variant=variant, d=d2, pit="SAME PROMPT",
                           response='{"inflation": 0.9}', p=0.8,
                           loadings={a: 0.9 for a in _axes()}))
    expected_keys = [(v, d) for v in ("pit", "nonpit_diagnostic") for d in dates]
    evidence_path = ext.write_evidence_table(recs, expected_keys, staging / "evidence")
    evidence_map = ext.validate_evidence_records(recs, expected_keys)
    consumed = {}
    for key in expected_keys:
        ext.record_consumption(consumed, key, evidence_map[key])
    audit_path = ext.write_replay_audit_summary(
        evidence_map, consumed, expected_keys, staging
    )
    metric_path = ext.write_factor_metric_records(
        bundle, staging, pit_value=pit_value, nonpit_value=nonpit_value,
        snapshot_dir=snapshot,
    )

    idx = pd.DatetimeIndex([pd.Timestamp(d1), pd.Timestamp(d2)])

    def _loadings_frame(variant):
        frame = pd.DataFrame(
            {"parse_ok": [True, True],
             "segment": ["replayed_v1", "live_ext2026"],
             "variant": [variant, variant]},
            index=idx,
        )
        for a in _axes():
            frame[a] = [0.1, 0.9]
        return frame

    def _scores_frame(variant):
        return pd.DataFrame(
            {"p_memorized": [0.2, 0.8], "fail_reason": [None, None],
             "segment": ["replayed_v1", "live_ext2026"], "variant": [variant, variant]},
            index=idx,
        )

    def _dlog_payload():
        keys = [str(pd.Timestamp(d)) for d in dates]
        return {
            "meta": {"n_rebalances": len(keys)},
            "p_memorized": {k: p for k, p in zip(keys, (0.2, 0.8))},
            "parse_ok": {k: True for k in keys},
            "steered": {k: False for k in keys},
            "conviction": {k: 0.5 for k in keys},
            "loadings": {k: {a: 0.1 for a in _axes()} for k in keys},
            "views": {k: [] for k in keys},
        }

    targets = pd.DataFrame({"AAA": [0.5, 0.6], "BIL": [0.5, 0.4]}, index=idx)
    contrast = pd.DataFrame(
        {"pit_p": [0.2, 0.8], "nonpit_p": [0.2, 0.8], "delta": [0.0, 0.0],
         "segment": ["in_training", "post_cutoff"]},
        index=idx,
    )
    split_payload = {
        "in_training": {"n_pairs": 1}, "post_cutoff": {"n_pairs": 1},
        "full_stream": {"n_pairs": 2}, "cutoff_date": "2024-06-01",
    }

    _loadings_frame("pit").to_parquet(staging / "factor_loadings_ext2026.parquet")
    _loadings_frame("nonpit_diagnostic").to_parquet(
        staging / "factor_nonpit_diagnostic_loadings_ext2026.parquet")
    _scores_frame("pit").to_parquet(staging / "factor_scores_ext2026.parquet")
    _scores_frame("nonpit_diagnostic").to_parquet(
        staging / "factor_nonpit_diagnostic_scores_ext2026.parquet")
    targets.to_parquet(staging / "factor_targets_ext2026.parquet")
    targets.to_parquet(staging / "factor_nonpit_diagnostic_targets_ext2026.parquet")
    pit_value.to_frame("value").to_parquet(staging / "factor_equity_ext2026.parquet")
    nonpit_value.to_frame("value").to_parquet(
        staging / "factor_nonpit_diagnostic_equity_ext2026.parquet")
    (staging / "factor_decision_log_ext2026.json").write_text(
        json.dumps(_dlog_payload(), indent=2))
    (staging / "factor_nonpit_diagnostic_decision_log_ext2026.json").write_text(
        json.dumps(_dlog_payload(), indent=2))
    contrast.to_parquet(staging / "factor_contrast_ext2026.parquet")
    (staging / "factor_contrast_split_ext2026.json").write_text(
        json.dumps(split_payload, indent=2, sort_keys=True))

    artifacts = {
        "evidence": evidence_path,
        "loadings_pit": staging / "factor_loadings_ext2026.parquet",
        "loadings_nonpit": staging / "factor_nonpit_diagnostic_loadings_ext2026.parquet",
        "scores_pit": staging / "factor_scores_ext2026.parquet",
        "scores_nonpit": staging / "factor_nonpit_diagnostic_scores_ext2026.parquet",
        "targets_pit": staging / "factor_targets_ext2026.parquet",
        "targets_nonpit": staging / "factor_nonpit_diagnostic_targets_ext2026.parquet",
        "equity_pit": staging / "factor_equity_ext2026.parquet",
        "equity_nonpit": staging / "factor_nonpit_diagnostic_equity_ext2026.parquet",
        "decision_log_pit": staging / "factor_decision_log_ext2026.json",
        "decision_log_nonpit": staging / "factor_nonpit_diagnostic_decision_log_ext2026.json",
        "contrast": staging / "factor_contrast_ext2026.parquet",
        "contrast_split": staging / "factor_contrast_split_ext2026.json",
        "replay_audit": audit_path,
        "metric_records": metric_path,
    }
    run_kwargs = dict(
        run_id="factor_ext2026_2021-01-01_2023-06-30_v1",
        output_root=tmp_path / "factor_runs",
        artifacts=artifacts,
        config={"sim_start": "2021-01-01", "sim_end": "2023-06-30", "tilt": 0.30},
        source_commit="ab" * 20,
        model={"nim_model": "openai/gpt-oss-20b", "cutoff_date": "2024-06-01"},
        input_manifests={
            "market_snapshot": {
                "snapshot_id": bundle["market_snapshot"]["snapshot_id"],
                "manifest_sha256": bundle["market_snapshot"]["manifest_sha256"],
            },
            "factor_loadings_v1": {
                "path": "data/factor_loadings_v1.parquet", "sha256": "a" * 64,
            },
        },
        expected_evidence={
            "variants": ["pit", "nonpit_diagnostic"],
            "dates": [d1.isoformat(), d2.isoformat()],
        },
    )
    return ext, staging, run_kwargs


def _completed_factor_run(tmp_path):
    ext, staging, run_kwargs = _staged_factor_run_case(tmp_path)
    run_dir = ext.build_factor_run_bundle(**run_kwargs)
    return ext, staging, run_dir, run_kwargs


def _resign_run_manifest(ext, run_dir, manifest) -> None:
    """Model an adversary able to rewrite manifest AND completion marker
    coherently: content validation must still catch the inconsistency."""
    import hashlib
    import json

    path = run_dir / "manifest.json"
    path.write_text(
        json.dumps(ext._json_record_value(manifest), indent=2, sort_keys=True) + "\n"
    )
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    marker = run_dir / "COMPLETED"
    first_line = marker.read_text().splitlines()[0]
    marker.write_text(f"{first_line}\nmanifest_sha256={sha}\n")


def test_factor_run_bundle_end_to_end_valid_and_marker_carries_manifest_sha(
    tmp_path,
) -> None:
    # AC 6.9/6.10 positive: a temporary valid bundle builds, the marker is the
    # LAST artifact and carries the manifest sha256, and the completed bundle
    # validates from its manifest alone (no snapshot dir, no in-memory inputs).
    import hashlib
    import json

    ext, staging, run_dir, run_kwargs = _completed_factor_run(tmp_path)
    assert run_dir == run_kwargs["output_root"] / run_kwargs["run_id"]
    marker = (run_dir / "COMPLETED").read_text()
    manifest_sha = hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest()
    assert f"manifest_sha256={manifest_sha}" in marker

    report = ext.validate_factor_run_bundle(run_dir)
    assert report["run_id"] == run_kwargs["run_id"]
    assert report["completed"] is True

    manifest = ext.load_completed_factor_run(run_dir)
    assert manifest["schema"] == "factor_run.v1"
    assert manifest["run_id"] == run_kwargs["run_id"]
    assert manifest["completed"] is True
    assert manifest["source_commit"] == "ab" * 20
    assert manifest["config"]["sim_start"] == "2021-01-01"
    assert manifest["model"]["nim_model"] == "openai/gpt-oss-20b"
    assert manifest["prompt_renderer"]["id"] == (
        "macro_framework.factor_scoring.render_regime_loadings_prompt"
    )
    assert manifest["input_manifests"]["market_snapshot"]["snapshot_id"]
    assert manifest["expected_evidence"] == {
        "variants": ["nonpit_diagnostic", "pit"],
        "dates": ["2021-06-01", "2022-06-01"],
        "n_dates": 2,
        "n_keys": 4,
    }
    assert manifest["replay_audit"]["result"] == "pass"
    assert manifest["replay_audit"]["counts"]["expected_keys"] == 4

    files = manifest["files"]
    assert set(files) == set(ext.FACTOR_RUN_ARTIFACTS)
    assert files["evidence"]["rows"] == 4
    assert files["evidence"]["start"] == "2021-06-01"
    assert files["evidence"]["end"] == "2022-06-01"
    assert files["metric_records"]["rows"] == 9
    for role, entry in files.items():
        path = run_dir / entry["file"]
        assert path.is_file()
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert entry["schema_id"] == f"factor_run.v1/{role}"
        assert entry["lineage"]


def test_factor_run_validator_verifies_inventory_hashes_and_counts(tmp_path) -> None:
    import json

    import pytest

    ext, staging, run_dir, _ = _completed_factor_run(tmp_path)
    equity = run_dir / "factor_equity_ext2026.parquet"
    original = equity.read_bytes()
    equity.write_bytes(original + b"tampered")
    with pytest.raises(ValueError, match="mutated after inventory"):
        ext.validate_factor_run_bundle(run_dir)
    equity.write_bytes(original)
    ext.validate_factor_run_bundle(run_dir)  # restored bundle is valid again

    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["files"]["scores_pit"]["rows"] = 3
    _resign_run_manifest(ext, run_dir, manifest)
    with pytest.raises(ValueError, match="rows"):
        ext.validate_factor_run_bundle(run_dir)

    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["files"]["scores_pit"]["rows"] = 2
    manifest["replay_audit"]["counts"]["consumed_keys"] = 3
    _resign_run_manifest(ext, run_dir, manifest)
    with pytest.raises(ValueError, match="replay audit"):
        ext.validate_factor_run_bundle(run_dir)


def test_factor_run_refuses_completed_overwrite_and_dirty_staging(tmp_path) -> None:
    import pytest

    ext, staging, run_dir, run_kwargs = _completed_factor_run(tmp_path)
    manifest_before = (run_dir / "manifest.json").read_bytes()
    with pytest.raises(ValueError, match="COMPLETED and immutable"):
        ext.build_factor_run_bundle(**run_kwargs)
    assert (run_dir / "manifest.json").read_bytes() == manifest_before

    dirty_kwargs = dict(run_kwargs)
    dirty_kwargs["run_id"] = "factor_ext2026_dirty_v1"
    dirty = run_kwargs["output_root"] / dirty_kwargs["run_id"]
    dirty.mkdir(parents=True)
    (dirty / "leftover.bin").write_bytes(b"stale")
    with pytest.raises(ValueError, match="non-empty staging"):
        ext.build_factor_run_bundle(**dirty_kwargs)
    assert (dirty / "leftover.bin").read_bytes() == b"stale"


def test_factor_run_failure_leaves_diagnosable_staging_without_marker(tmp_path) -> None:
    import json

    import pytest

    ext, staging, run_kwargs = _staged_factor_run_case(tmp_path)
    metric_path = staging / "factor_metric_records_ext2026.json"
    payload = json.loads(metric_path.read_text())
    payload["content_sha256"] = "0" * 64
    metric_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="content_sha256"):
        ext.build_factor_run_bundle(**run_kwargs)

    run_dir = run_kwargs["output_root"] / run_kwargs["run_id"]
    assert run_dir.is_dir()
    assert not (run_dir / "COMPLETED").exists()  # marker is written LAST
    # the copied artifacts stay behind for diagnosis
    assert (run_dir / "factor_metric_records_ext2026.json").is_file()
    assert (run_dir / "factor_evidence_ext2026.parquet").is_file()
    # downstream readers refuse the incomplete staging output
    with pytest.raises(ValueError, match="COMPLETED"):
        ext.load_completed_factor_run(run_dir)


def test_downstream_reader_rejects_incomplete_and_stale_bundles(tmp_path) -> None:
    import json

    import pytest

    ext, staging, run_dir, _ = _completed_factor_run(tmp_path)
    marker_path = run_dir / "COMPLETED"
    marker_text = marker_path.read_text()

    marker_path.unlink()
    with pytest.raises(ValueError, match="COMPLETED"):
        ext.load_completed_factor_run(run_dir)

    marker_path.write_text("2026-01-01T00:00:00+00:00\nmanifest_sha256=" + "0" * 64 + "\n")
    with pytest.raises(ValueError, match="stale or tampered"):
        ext.load_completed_factor_run(run_dir)

    marker_path.write_text(marker_text)
    ext.load_completed_factor_run(run_dir)  # restored marker validates again

    # a manifest rewritten AFTER completion (marker not updated) is stale
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["config"]["sim_start"] = "1999-01-01"
    (run_dir / "manifest.json").write_text(
        json.dumps(ext._json_record_value(manifest), indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(ValueError, match="stale or tampered"):
        ext.load_completed_factor_run(run_dir)


def test_downstream_reader_rejects_tampered_evidence_despite_resigned_inventory(
    tmp_path,
) -> None:
    # Byte-inventory forgery is not enough: even when the evidence file, its
    # manifest entry, and the completion marker are all coherently re-signed, a
    # cross-date response swap still fails the record-level hash validation.
    import hashlib
    import json

    import pytest

    ext, staging, run_dir, _ = _completed_factor_run(tmp_path)
    evidence_path = run_dir / "factor_evidence_ext2026.parquet"
    df = pd.read_parquet(evidence_path)
    pit_rows = df.index[df["variant"] == "pit"].tolist()
    swapped = df.loc[pit_rows[0], "response_text"]
    df.loc[pit_rows[0], "response_text"] = df.loc[pit_rows[1], "response_text"]
    df.loc[pit_rows[1], "response_text"] = swapped
    df.to_parquet(evidence_path, index=False)

    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["files"]["evidence"]["sha256"] = hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    manifest["files"]["evidence"]["size"] = evidence_path.stat().st_size
    _resign_run_manifest(ext, run_dir, manifest)

    with pytest.raises(ValueError, match="response_sha256 mismatch"):
        ext.load_completed_factor_run(run_dir)


def test_factor_run_rejects_expected_count_and_coverage_mismatch(tmp_path) -> None:
    import json

    import pytest

    ext, staging, run_dir, _ = _completed_factor_run(tmp_path)

    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["expected_evidence"]["dates"] = [
        "2021-06-01", "2022-06-01", "2022-12-01",
    ]
    manifest["expected_evidence"]["n_dates"] = 3
    manifest["expected_evidence"]["n_keys"] = 6
    _resign_run_manifest(ext, run_dir, manifest)
    with pytest.raises(ValueError, match="missing expected evidence key"):
        ext.load_completed_factor_run(run_dir)

    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["expected_evidence"]["dates"] = ["2021-06-01", "2022-06-01"]
    manifest["expected_evidence"]["n_dates"] = 2
    manifest["expected_evidence"]["n_keys"] = 5
    _resign_run_manifest(ext, run_dir, manifest)
    with pytest.raises(ValueError, match="n_keys"):
        ext.load_completed_factor_run(run_dir)


def test_factor_run_rejects_unmanifested_or_missing_artifacts(tmp_path) -> None:
    import pytest

    ext, staging, run_dir, _ = _completed_factor_run(tmp_path)
    stray = run_dir / "extra_unmanifested.parquet"
    stray.write_bytes(b"stray artifact")
    with pytest.raises(ValueError, match="unmanifested"):
        ext.load_completed_factor_run(run_dir)
    stray.unlink()
    ext.load_completed_factor_run(run_dir)

    (run_dir / "factor_nonpit_diagnostic_decision_log_ext2026.json").unlink()
    with pytest.raises(ValueError, match="missing from disk"):
        ext.load_completed_factor_run(run_dir)


def test_factor_run_rejects_forged_input_manifest_lineage(tmp_path) -> None:
    import json

    import pytest

    ext, staging, run_dir, _ = _completed_factor_run(tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["input_manifests"]["market_snapshot"]["manifest_sha256"] = "f" * 64
    _resign_run_manifest(ext, run_dir, manifest)
    with pytest.raises(ValueError, match="market snapshot lineage"):
        ext.load_completed_factor_run(run_dir)
