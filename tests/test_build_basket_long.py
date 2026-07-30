"""Frozen market-snapshot acquisition contract (remediation task 5.1).

Offline only: nothing here touches the network or writes snapshot state.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_basket_long as bbl  # noqa: E402


def _good() -> bbl.AcquisitionContract:
    return bbl.make_snapshot_contract(vintage_date="2026-07-03")


def test_contract_pins_identity_universe_and_quotes():
    contract = _good()
    # these assertions ARE the freeze — moving any value is a spec change
    assert contract.snapshot_id == "market_total_return_fx_2026-06-30_v1"
    assert (contract.requested_start, contract.requested_end) == ("2009-09-01", "2026-06-30")
    assert contract.symbols == ("SWDA.L", "XLK", "IAU", "BIL", "SPY")
    assert contract.quotes["SWDA.L"] == {
        "quote_currency": "GBP",
        "quote_unit": "GBp",
        "scale_to_major": 0.01,
    }
    for symbol in ("XLK", "IAU", "BIL", "SPY"):
        assert contract.quotes[symbol] == {
            "quote_currency": "USD",
            "quote_unit": "USD",
            "scale_to_major": 1.0,
        }
    assert (contract.cash_symbol, contract.benchmark_symbol) == ("BIL", "SPY")
    assert (contract.fx_series_id, contract.fx_field, contract.fx_conversion) == (
        "DEXUSUK",
        "USD_per_GBP",
        "multiply",
    )
    assert "auto_adjust=True" in contract.total_return_field
    with pytest.raises(dataclasses.FrozenInstanceError):
        contract.fx_field = "GBP_per_USD"  # type: ignore[misc]


def test_derived_request_params_pin_auto_adjust_and_single_vintage():
    contract = _good()
    etf = contract.etf_request_params()
    assert etf["auto_adjust"] is True
    assert etf["tickers"] == list(contract.symbols)
    assert etf["start"] == "2009-09-01"
    assert etf["end"] == "2026-07-01"  # yfinance end-exclusive; requested_end inclusive
    fx = contract.fx_request_params()
    assert fx["series_id"] == "DEXUSUK"
    assert fx["realtime_start"] == fx["realtime_end"] == "2026-07-03"
    assert (fx["observation_start"], fx["observation_end"]) == ("2009-09-01", "2026-06-30")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"etf_source_id": " "}, "etf_source_id"),
        ({"fx_source_id": ""}, "fx_source_id"),
        ({"fx_field": "GBP_per_USD"}, "USD_per_GBP"),
        ({"fx_conversion": "divide"}, "multiply"),
        ({"realtime_end": "2026-07-04"}, "ONE ALFRED vintage"),
        ({"realtime_start": "2026-07-02"}, "ONE ALFRED vintage"),
        ({"vintage_date": "not-a-date"}, "vintage_date"),
        ({"requested_end": ""}, "requested_end"),
        ({"requested_end": "2026-13-40"}, "requested_end"),
        ({"requested_start": "2026-06-30"}, "strictly before"),
        ({"symbols": ("SWDA.L", "XLK", "IAU", "BIL", "SPY", "SPY")}, "duplicates"),
        ({"cash_symbol": "SHV"}, "cash_symbol"),
        ({"benchmark_symbol": "QQQ"}, "benchmark_symbol"),
        ({"snapshot_id": ""}, "snapshot_id"),
    ],
)
def test_refuses_incomplete_or_ambiguous_configuration(mutation, match):
    with pytest.raises(ValueError, match=match):
        dataclasses.replace(_good(), **mutation)


def test_refuses_ambiguous_quote_units():
    good = _good()
    swda_usd_scale = {**good.quotes, "SWDA.L": {**good.quotes["SWDA.L"], "scale_to_major": 1.0}}
    with pytest.raises(ValueError, match="0.01"):
        dataclasses.replace(good, quotes=swda_usd_scale)
    mixed_unit = {**good.quotes, "SWDA.L": {**good.quotes["SWDA.L"], "quote_currency": "USD"}}
    with pytest.raises(ValueError, match="GBp requires quote_currency GBP"):
        dataclasses.replace(good, quotes=mixed_unit)
    missing = {k: v for k, v in good.quotes.items() if k != "IAU"}
    with pytest.raises(ValueError, match="IAU"):
        dataclasses.replace(good, quotes=missing)
    extra = {**good.quotes, "QQQ": {"quote_currency": "USD", "quote_unit": "USD", "scale_to_major": 1.0}}
    with pytest.raises(ValueError, match="QQQ"):
        dataclasses.replace(good, quotes=extra)
    incomplete_entry = {**good.quotes, "XLK": {"quote_currency": "USD"}}
    with pytest.raises(ValueError, match="exactly"):
        dataclasses.replace(good, quotes=incomplete_entry)


def test_acquisition_gate_requires_raw_response_hashes():
    bare = _good()
    with pytest.raises(ValueError, match="etf_raw_response_sha256 is required"):
        bare.require_ready_for_acquisition()

    fake = hashlib.sha256(b"raw-bytes").hexdigest()
    ready = dataclasses.replace(
        bare, etf_raw_response_sha256=fake, fx_raw_response_sha256=fake
    )
    ready.require_ready_for_acquisition()  # both declared -> acquisition may start

    with pytest.raises(ValueError, match="64 lowercase hex"):
        dataclasses.replace(bare, etf_raw_response_sha256=fake[:-1], fx_raw_response_sha256=fake
        ).require_ready_for_acquisition()
    with pytest.raises(ValueError, match="64 lowercase hex"):
        dataclasses.replace(bare, etf_raw_response_sha256=fake.upper(), fx_raw_response_sha256=fake
        ).require_ready_for_acquisition()


def test_fingerprint_is_canonical_and_stable():
    a, b = _good(), _good()
    assert a.fingerprint() == b.fingerprint()
    other = bbl.make_snapshot_contract(vintage_date="2026-07-04")
    assert other.fingerprint() != a.fingerprint()
    expected = hashlib.sha256(
        json.dumps(
            dataclasses.asdict(a), sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    ).hexdigest()
    assert a.fingerprint() == expected


def test_module_import_performs_no_acquisition():
    # importing the module must never fetch or write snapshot state
    assert not (Path(bbl.REPO) / "data" / "market_snapshots").exists()
    assert callable(bbl.main)  # legacy panel build stays behind __main__ only


# --- Task 5.1 gate suggestions (post-approval hardening) --------------------------


def test_contract_hardening_refuses_type_and_ordering_tricks():
    good = _good()
    # symbols passed as a list are frozen to a tuple at construction
    listed = dataclasses.replace(good, symbols=list(good.symbols))
    assert isinstance(listed.symbols, tuple)
    # scale_to_major must be a real float, not bool/int lookalikes
    with pytest.raises(ValueError, match="must be a float"):
        dataclasses.replace(
            good, quotes={**good.quotes, "XLK": {**good.quotes["XLK"], "scale_to_major": True}}
        )
    # provenance text is pinned to the derived request semantics
    with pytest.raises(ValueError, match="total_return_field"):
        dataclasses.replace(good, total_return_field="raw close, unadjusted")
    # a vintage before the coverage end can never cover the window
    with pytest.raises(ValueError, match="must not precede requested_end"):
        bbl.make_snapshot_contract(vintage_date="2010-01-01")
    with pytest.raises(ValueError, match="quotes must be a dict"):
        dataclasses.replace(good, quotes=list(good.symbols))


# --- Task 5.2: acquisition and normalization --------------------------------------


def _ready() -> bbl.AcquisitionContract:
    import hashlib as _h

    fake = _h.sha256(b"raw").hexdigest()
    return dataclasses.replace(
        _good(), etf_raw_response_sha256=fake, fx_raw_response_sha256=fake
    )


def _raw_close() -> "pd.DataFrame":
    import numpy as np
    import pandas as pd

    idx = pd.bdate_range("2016-01-04", periods=10)
    frame = pd.DataFrame(
        {
            "SWDA.L": np.linspace(7500.0, 7600.0, 10),  # local GBp levels
            "XLK": np.linspace(43.0, 44.0, 10),
            "IAU": np.linspace(10.0, 10.4, 10),
            "BIL": np.linspace(45.5, 45.6, 10),
            "SPY": np.linspace(200.0, 205.0, 10),
        },
        index=idx,
    )
    frame.loc[idx[3], "SWDA.L"] = float("nan")  # UK bank holiday: LSE closed
    frame.loc[idx[6], ["XLK", "IAU", "BIL", "SPY"]] = float("nan")  # US holiday
    return frame


def _fx_series() -> "pd.Series":
    import pandas as pd

    idx = pd.DatetimeIndex(
        ["2016-01-04", "2016-01-05", "2016-01-07", "2016-01-08"]  # 01-06 unpublished
    )
    return pd.Series([1.47, 1.46, 1.45, 1.44], index=idx, name="DEXUSUK")


def test_normalize_preserves_union_calendar_without_complete_case_filter():
    import pandas as pd

    contract = _ready()
    raw = _raw_close()
    basket, cash_market, coverage = bbl.normalize_etf_levels(contract, raw)

    assert list(basket.columns) == ["SWDA.L", "XLK", "IAU"]
    assert list(cash_market.columns) == ["BIL", "SPY"]
    assert basket.index.name == cash_market.index.name == "Date"
    # both holiday rows survive: no global complete-case filter
    assert len(basket) == len(cash_market) == 10
    assert pd.isna(basket.loc[raw.index[3], "SWDA.L"])
    assert pd.isna(cash_market.loc[raw.index[6], "SPY"])
    # SWDA.L stays in local GBp — never scaled here
    assert basket["SWDA.L"].dropna().iloc[0] == pytest.approx(7500.0)
    # coverage discloses per-symbol observed sessions
    assert coverage["SWDA.L"]["rows"] == 9
    assert coverage["SPY"]["rows"] == 9
    assert coverage["SPY"]["first"] == "2016-01-04"


def test_normalize_rejects_corrupt_observations_instead_of_repairing():
    import pandas as pd

    contract = _ready()

    inf_level = _raw_close()
    inf_level.iloc[2, inf_level.columns.get_loc("XLK")] = float("inf")
    with pytest.raises(ValueError, match="XLK: non-finite level"):
        bbl.normalize_etf_levels(contract, inf_level)

    negative = _raw_close()
    negative.iloc[4, negative.columns.get_loc("IAU")] = -1.0
    with pytest.raises(ValueError, match="IAU: non-positive level"):
        bbl.normalize_etf_levels(contract, negative)

    duplicated = pd.concat([_raw_close(), _raw_close().iloc[[0]]])
    with pytest.raises(ValueError, match="unique"):
        bbl.normalize_etf_levels(contract, duplicated)

    unordered = _raw_close().iloc[::-1]
    with pytest.raises(ValueError, match="strictly increasing"):
        bbl.normalize_etf_levels(contract, unordered)

    tz_aware = _raw_close()
    tz_aware.index = tz_aware.index.tz_localize("UTC")
    with pytest.raises(ValueError, match="timezone-naive"):
        bbl.normalize_etf_levels(contract, tz_aware)

    out_of_bounds = _raw_close()
    out_of_bounds.index = out_of_bounds.index - pd.DateOffset(years=10)
    with pytest.raises(ValueError, match="declared bounds"):
        bbl.normalize_etf_levels(contract, out_of_bounds)

    missing_symbol = _raw_close().drop(columns=["BIL"])
    with pytest.raises(ValueError, match="BIL"):
        bbl.normalize_etf_levels(contract, missing_symbol)


def test_fx_normalization_preserves_source_dates_and_direction():
    contract = _ready()
    fx, disclosure = bbl.normalize_fx_observations(contract, _fx_series())

    assert list(fx.columns) == ["USD_per_GBP"]
    assert fx.index.name == "Date"
    # SOURCE observation dates preserved exactly — the unpublished 01-06 stays absent
    assert [d.date().isoformat() for d in fx.index] == [
        "2016-01-04", "2016-01-05", "2016-01-07", "2016-01-08",
    ]
    # direction: USD per GBP is carried through, never inverted
    assert fx["USD_per_GBP"].iloc[0] == pytest.approx(1.47)
    assert disclosure["rows"] == 4 and disclosure["first"] == "2016-01-04"


def test_fredgraph_parser_drops_dot_sentinels_with_disclosure():
    text = (
        "DATE,DEXUSUK\n"
        "2016-01-04,1.47\n"
        "2016-01-05,1.46\n"
        "2016-01-06,.\n"
        "2016-01-07,1.45\n"
    )
    series, disclosure = bbl.parse_fredgraph_csv(text, "DEXUSUK")
    assert len(series) == 3
    assert disclosure == {"raw_rows": 4, "non_numeric_dropped": 1}
    with pytest.raises(ValueError, match="does not contain series"):
        bbl.parse_fredgraph_csv(text, "DGS10")
    with pytest.raises(ValueError, match="empty FRED series"):
        bbl.parse_fredgraph_csv("DATE,DEXUSUK\n2016-01-04,.\n", "DEXUSUK")


def test_fx_normalization_rejects_defects():
    import pandas as pd

    contract = _ready()
    nan_fx = _fx_series()
    nan_fx.iloc[1] = float("nan")
    with pytest.raises(ValueError, match="not finite"):
        bbl.normalize_fx_observations(contract, nan_fx)
    with pytest.raises(ValueError, match="unique"):
        bbl.normalize_fx_observations(
            contract, pd.concat([_fx_series(), _fx_series().iloc[[0]]])
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        bbl.normalize_fx_observations(contract, _fx_series().iloc[::-1])
    early = _fx_series()
    early.index = early.index - pd.DateOffset(years=10)
    with pytest.raises(ValueError, match="declared bounds"):
        bbl.normalize_fx_observations(contract, early)
    negative = _fx_series()
    negative.iloc[0] = -1.4
    with pytest.raises(ValueError, match="not positive"):
        bbl.normalize_fx_observations(contract, negative)


def test_acquire_requires_hash_gate_and_normalizes_injected_frames():
    # the acquisition gate fires before any fetch or normalization
    with pytest.raises(ValueError, match="acquisition cannot start"):
        bbl.acquire_snapshot_data(
            _good(), etf_close=_raw_close(), fx_observations=_fx_series()
        )

    data = bbl.acquire_snapshot_data(
        _ready(),
        etf_close=_raw_close(),
        fx_observations=_fx_series(),
        fx_parse_disclosure={"raw_rows": 5, "non_numeric_dropped": 1},
    )
    assert isinstance(data, bbl.NormalizedSnapshotData)
    assert list(data.fx.columns) == ["USD_per_GBP"]
    assert set(data.coverage) == {"SWDA.L", "XLK", "IAU", "BIL", "SPY", "USD_per_GBP"}
    assert data.coverage["USD_per_GBP"]["non_numeric_dropped"] == 1
    assert data.coverage["USD_per_GBP"]["rows"] == 4


# --- Task 5.2 re-gate remediations -------------------------------------------------


def test_normalize_rejects_intraday_and_complex_observations():
    import pandas as pd

    contract = _ready()

    intraday = _raw_close()
    idx = intraday.index.to_list()
    idx[2] = idx[2] + pd.Timedelta(hours=15, minutes=30)
    intraday.index = pd.DatetimeIndex(idx)
    with pytest.raises(ValueError, match="midnight"):
        bbl.normalize_etf_levels(contract, intraday)

    complex_etf = _raw_close().astype({"XLK": complex})
    complex_etf.iloc[1, complex_etf.columns.get_loc("XLK")] = 43 + 2j
    with pytest.raises(ValueError, match="real numeric"):
        bbl.normalize_etf_levels(contract, complex_etf)

    fx_intraday = _fx_series()
    fidx = fx_intraday.index.to_list()
    fidx[1] = fidx[1] + pd.Timedelta(hours=9)
    fx_intraday.index = pd.DatetimeIndex(fidx)
    with pytest.raises(ValueError, match="midnight"):
        bbl.normalize_fx_observations(contract, fx_intraday)

    with pytest.raises(ValueError, match="real numeric"):
        bbl.normalize_fx_observations(contract, _fx_series().astype(complex))


def test_parser_rejects_garbage_tokens_but_drops_only_the_dot_sentinel():
    garbage = "DATE,DEXUSUK\n2016-01-04,1.47\n2016-01-05,N/A\n"
    with pytest.raises(ValueError, match="only the '.' unpublished"):
        bbl.parse_fredgraph_csv(garbage, "DEXUSUK")
    clean = "DATE,DEXUSUK\n2016-01-04,1.47\n2016-01-05,.\n"
    series, disclosure = bbl.parse_fredgraph_csv(clean, "DEXUSUK")
    assert len(series) == 1 and disclosure["non_numeric_dropped"] == 1


def test_parser_accepts_vintage_suffixed_alfred_column():
    text = "DATE,DEXUSUK_20260703\n2016-01-04,1.47\n2016-01-05,.\n"
    series, disclosure = bbl.parse_fredgraph_csv(text, "DEXUSUK", column="DEXUSUK_20260703")
    assert series.name == "DEXUSUK"
    assert len(series) == 1 and disclosure["raw_rows"] == 2
    # a payload without the suffixed column is the vintage-not-honored signal
    with pytest.raises(ValueError, match="does not contain series"):
        bbl.parse_fredgraph_csv(text, "DEXUSUK", column="DEXUSUK_20260704")


def test_normalize_rejects_multiindex_and_duplicate_columns():
    import pandas as pd

    contract = _ready()
    dup = _raw_close()
    dup = pd.concat([dup, dup[["SPY"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate column"):
        bbl.normalize_etf_levels(contract, dup)
    multi = _raw_close()
    multi.columns = pd.MultiIndex.from_product([["Close"], list(multi.columns)])
    with pytest.raises(ValueError, match="MultiIndex"):
        bbl.normalize_etf_levels(contract, multi)


def test_acquire_rejects_inconsistent_injected_disclosure():
    with pytest.raises(ValueError, match="retained observations"):
        bbl.acquire_snapshot_data(
            _ready(),
            etf_close=_raw_close(),
            fx_observations=_fx_series(),
            fx_parse_disclosure={"raw_rows": 999, "non_numeric_dropped": 42},
        )


# --- Tasks 5.3/5.4: append-only snapshots, offline boundary tests ------------------


def _data() -> "bbl.NormalizedSnapshotData":
    return bbl.acquire_snapshot_data(
        _ready(), etf_close=_raw_close(), fx_observations=_fx_series()
    )


def _build(tmp_path, contract=None, data=None):
    contract = contract or _ready()
    return bbl.build_market_snapshot(
        snapshot_id=contract.snapshot_id,
        requested_start=contract.requested_start,
        requested_end=contract.requested_end,
        output_root=tmp_path,
        contract=contract,
        data=data or _data(),
        build_time="2026-07-28T12:00:00+00:00",
    )


def test_snapshot_builds_offline_and_completes_last(tmp_path, monkeypatch):
    import json as _json

    # loud proof that no network is touched during an injected-frame build
    monkeypatch.setattr(bbl.requests, "get", lambda *a, **k: pytest.fail("network"))
    monkeypatch.setattr(bbl.yf, "download", lambda *a, **k: pytest.fail("network"))

    snapshot_dir = _build(tmp_path)
    assert (snapshot_dir / "COMPLETED").is_file()
    manifest = _json.loads((snapshot_dir / "manifest.json").read_text())
    assert manifest["schema"] == "market_snapshot.v1"
    assert manifest["quotes"]["SWDA.L"]["quote_unit"] == "GBp"
    assert manifest["fx_field"] == "USD_per_GBP"
    assert manifest["fx_vintage_date"] == "2026-07-03"
    assert manifest["overlap_revisions"]["preceding_snapshot"] is None
    assert set(manifest["files"]) == {
        "basket_adjusted_close_local.parquet",
        "cash_market_total_return.parquet",
        "fx_usd_per_gbp.parquet",
    }
    report = bbl.validate_market_snapshot(snapshot_dir)
    assert report["completed"] is True
    assert report["files"]["fx_usd_per_gbp.parquet"]["rows"] == 4


def test_snapshot_identity_is_immutable(tmp_path):
    _build(tmp_path)
    with pytest.raises(ValueError, match="COMPLETED and immutable"):
        _build(tmp_path)


def test_completion_marker_absent_after_failed_validation(tmp_path):
    import pandas as pd

    good = _data()
    early = good.basket_local.copy()
    early.index = early.index - pd.DateOffset(years=10)  # outside pinned coverage
    broken = bbl.NormalizedSnapshotData(
        basket_local=early,
        cash_market=good.cash_market,
        fx=good.fx,
        coverage=good.coverage,
    )
    with pytest.raises(ValueError, match="outside requested coverage"):
        _build(tmp_path, data=broken)
    staging = tmp_path / bbl.SNAPSHOT_ID
    assert not (staging / "COMPLETED").exists()  # marker never precedes validation
    # the dirty staging directory cannot be silently reused
    with pytest.raises(ValueError, match="non-empty staging"):
        _build(tmp_path)


def test_validate_detects_byte_mutation_and_nonfinite_data(tmp_path):
    import hashlib as _h
    import io as _io
    import json as _json

    import pandas as pd

    snapshot_dir = _build(tmp_path)
    target = snapshot_dir / "cash_market_total_return.parquet"

    # raw byte mutation is caught by the sha256 inventory
    original = target.read_bytes()
    target.write_bytes(original + b"tamper")
    with pytest.raises(ValueError, match="mutated after inventory"):
        bbl.validate_market_snapshot(snapshot_dir)

    # a re-inventoried non-finite value is still rejected by the data check
    frame = pd.read_parquet(_io.BytesIO(original))
    frame.iloc[0, 0] = float("inf")
    frame.to_parquet(target)
    manifest_path = snapshot_dir / "manifest.json"
    manifest = _json.loads(manifest_path.read_text())
    manifest["files"]["cash_market_total_return.parquet"]["sha256"] = _h.sha256(
        target.read_bytes()
    ).hexdigest()
    manifest_path.write_text(_json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="non-finite"):
        bbl.validate_market_snapshot(snapshot_dir)


def test_absent_overlap_disclosure_blocks_validation(tmp_path):
    import json as _json

    snapshot_dir = _build(tmp_path)
    manifest_path = snapshot_dir / "manifest.json"
    manifest = _json.loads(manifest_path.read_text())
    del manifest["overlap_revisions"]
    manifest_path.write_text(_json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="overlap revision disclosure is absent"):
        bbl.validate_market_snapshot(snapshot_dir)


def test_overlap_revisions_quantified_against_preceding_snapshot(tmp_path):
    import json as _json

    _build(tmp_path)  # v1 completes

    v2_contract = dataclasses.replace(_ready(), snapshot_id=bbl.SNAPSHOT_ID + "_r2")
    revised_close = _raw_close()
    revised_close.iloc[2, revised_close.columns.get_loc("XLK")] += 0.5  # vendor revision
    v2_data = bbl.acquire_snapshot_data(
        v2_contract, etf_close=revised_close, fx_observations=_fx_series()
    )
    v2_dir = bbl.build_market_snapshot(
        snapshot_id=v2_contract.snapshot_id,
        requested_start=v2_contract.requested_start,
        requested_end=v2_contract.requested_end,
        output_root=tmp_path,
        contract=v2_contract,
        data=v2_data,
        build_time="2026-07-28T13:00:00+00:00",
    )
    manifest = _json.loads((v2_dir / "manifest.json").read_text())
    overlap = manifest["overlap_revisions"]
    assert overlap["preceding_snapshot"] == bbl.SNAPSHOT_ID
    basket = overlap["basket_adjusted_close_local.parquet"]
    assert basket["overlap_rows"] == 10
    assert basket["changed_cells"] == 1
    assert overlap["fx_usd_per_gbp.parquet"]["changed_cells"] == 0


def test_snapshot_refuses_contract_mismatches(tmp_path):
    contract = _ready()
    with pytest.raises(ValueError, match="does not match the contract"):
        bbl.build_market_snapshot(
            snapshot_id="other_id",
            requested_start=contract.requested_start,
            requested_end=contract.requested_end,
            output_root=tmp_path,
            contract=contract,
            data=_data(),
        )
    with pytest.raises(ValueError, match="requested coverage"):
        bbl.build_market_snapshot(
            snapshot_id=contract.snapshot_id,
            requested_start="2010-01-01",
            requested_end=contract.requested_end,
            output_root=tmp_path,
            contract=contract,
            data=_data(),
        )


def test_build_time_is_validated_and_ordering_uses_parsed_timestamps(tmp_path):
    import json as _json

    contract = _ready()
    with pytest.raises(ValueError, match="ISO-8601"):
        bbl.build_market_snapshot(
            snapshot_id=contract.snapshot_id,
            requested_start=contract.requested_start,
            requested_end=contract.requested_end,
            output_root=tmp_path / "a",
            contract=contract,
            data=_data(),
            build_time="t",
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        bbl.build_market_snapshot(
            snapshot_id=contract.snapshot_id,
            requested_start=contract.requested_start,
            requested_end=contract.requested_end,
            output_root=tmp_path / "b",
            contract=contract,
            data=_data(),
            build_time="2026-07-28T12:00:00",
        )

    # offsets are normalized to UTC so predecessor selection is chronological
    root = tmp_path / "c"
    v1 = bbl.build_market_snapshot(
        snapshot_id=contract.snapshot_id,
        requested_start=contract.requested_start,
        requested_end=contract.requested_end,
        output_root=root,
        contract=contract,
        data=_data(),
        build_time="2026-07-28T20:00:00+12:00",  # 08:00 UTC
    )
    manifest = _json.loads((v1 / "manifest.json").read_text())
    assert manifest["build_time"] == "2026-07-28T08:00:00+00:00"
    assert manifest["cash_symbol"] == "BIL" and manifest["benchmark_symbol"] == "SPY"

    later = dataclasses.replace(contract, snapshot_id=bbl.SNAPSHOT_ID + "_r2")
    bbl.build_market_snapshot(
        snapshot_id=later.snapshot_id,
        requested_start=later.requested_start,
        requested_end=later.requested_end,
        output_root=root,
        contract=later,
        data=bbl.acquire_snapshot_data(
            later, etf_close=_raw_close(), fx_observations=_fx_series()
        ),
        build_time="2026-07-28T12:00:00+00:00",  # chronologically latest
    )
    third = dataclasses.replace(contract, snapshot_id=bbl.SNAPSHOT_ID + "_r3")
    v3 = bbl.build_market_snapshot(
        snapshot_id=third.snapshot_id,
        requested_start=third.requested_start,
        requested_end=third.requested_end,
        output_root=root,
        contract=third,
        data=bbl.acquire_snapshot_data(
            third, etf_close=_raw_close(), fx_observations=_fx_series()
        ),
        build_time="2026-07-28T13:00:00+00:00",
    )
    v3_manifest = _json.loads((v3 / "manifest.json").read_text())
    assert v3_manifest["overlap_revisions"]["preceding_snapshot"] == bbl.SNAPSHOT_ID + "_r2"
    # the completion marker carries the manifest hash for cheap tamper evidence
    assert "manifest_sha256=" in (v3 / "COMPLETED").read_text()
