"""Deterministic US and German locale mirrors of canonical report tables.

Locale mirrors are pure PROJECTIONS (task 9.7): they are generated only from
canonical in-memory tables owned by ``scripts/build_tear_sheet.py`` — never
through an independent financial calculation. This module formats, writes,
parses, and verifies; it owns no finance formula.

Contract:

- every canonical table stem ``X`` mirrors to the matching basenames ``X.csv``
  (en-US: comma separator, dot decimals) and ``X_de.csv`` (de-DE: semicolon
  separator, comma decimals), the locale specs being producer-owned data
  (``build_tear_sheet.REPORT_CSV_LOCALE_SPECS``);
- deterministic ordering (the canonical table's own row/column order, stems
  written in sorted order), ISO ``YYYY-MM-DD`` dates, empty-field nulls, and
  eight-decimal fixed-point numbers;
- every written mirror is immediately re-parsed with the locale parser and
  must reproduce the source values within ``5e-9`` (design tolerance);
- the frozen data-v4 catalog names the required mirrors of every cataloged
  canonical table; ``require_catalog_mirror_coverage`` fails when any is
  missing.

Real canonical outputs are produced by the staged release build (task 12.4);
everything here is parameterized by in-memory tables and output paths.
"""
from __future__ import annotations

import io
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

MIRROR_PRODUCER = "scripts/export_csv_mirrors.py"

#: Design tolerance for eight-decimal locale exports (round-trip parity).
ROUND_TRIP_TOLERANCE = 5e-9

REQUIRED_MIRROR_LOCALES = ("en-US", "de-DE")

_DATE_FORMAT = "%Y-%m-%d"
_NA_REP = ""


def _report_producer():
    """The canonical report producer owns the locale source schemas."""
    try:
        from scripts import build_tear_sheet as producer
    except ImportError:  # scripts/ itself on sys.path (test convention)
        import build_tear_sheet as producer
    return producer


def _publication_catalog():
    try:
        from scripts import publish_finance_remediation as publisher
    except ImportError:
        import publish_finance_remediation as publisher
    return publisher.DATA_V4_CATALOG


def _locale_spec(locale: str) -> Mapping[str, str]:
    specs = _report_producer().REPORT_CSV_LOCALE_SPECS
    if locale not in specs:
        raise ValueError(
            f"unknown mirror locale {locale!r}; the producer-owned locales are "
            f"{sorted(specs)}"
        )
    return specs[locale]


def mirror_basenames(stem: str) -> dict[str, str]:
    """Required matching basenames of one canonical table stem, per locale."""
    _require_stem(stem)
    return {"en-US": f"{stem}.csv", "de-DE": f"{stem}_de.csv"}


def _require_stem(stem: object) -> str:
    if not isinstance(stem, str) or not stem.strip():
        raise ValueError("canonical table stem must be a non-empty string")
    if stem != Path(stem).name or "/" in stem or "\\" in stem or stem in (".", ".."):
        raise ValueError(f"canonical table stem must be a flat basename: {stem!r}")
    if stem.endswith("_de"):
        raise ValueError(
            f"canonical table stem must not end in '_de' ({stem!r}): it would "
            "collide with the German mirror basenames"
        )
    return stem


def _require_canonical_table(stem: str, table: object) -> pd.DataFrame:
    if not isinstance(table, pd.DataFrame):
        raise ValueError(f"{stem}: canonical table must be a pandas DataFrame")
    if len(table) == 0 or len(table.columns) == 0:
        raise ValueError(f"{stem}: canonical table must not be empty")
    if isinstance(table.columns, pd.MultiIndex) or not all(
        isinstance(column, str) and column.strip() for column in table.columns
    ):
        raise ValueError(f"{stem}: canonical table columns must be plain strings")
    if table.columns.duplicated().any():
        raise ValueError(f"{stem}: canonical table columns must be unique")
    if not table.index.equals(pd.RangeIndex(len(table))):
        raise ValueError(
            f"{stem}: canonical tables are flat — reset the index before "
            "mirroring so no data hides in an unwritten index"
        )
    return table


def _mirror_frame(table: pd.DataFrame) -> pd.DataFrame:
    """Deterministic date projection for object columns.

    Mixed-schema tables NaN-pad timestamp fields into object columns, where
    ``to_csv(date_format=...)`` does not apply; project those cells to the same
    ISO date form the datetime64 columns receive.
    """
    projected = table.copy(deep=False)
    for column in projected.columns:
        series = projected[column]
        if series.dtype == object and any(
            isinstance(value, pd.Timestamp) for value in series
        ):
            projected[column] = series.map(
                lambda value: value.strftime(_DATE_FORMAT)
                if isinstance(value, pd.Timestamp)
                else value
            )
    return projected


def render_locale_csv(table: pd.DataFrame, *, locale: str) -> bytes:
    """One canonical table as deterministic locale CSV bytes."""
    spec = _locale_spec(locale)
    buffer = io.StringIO()
    _mirror_frame(table).to_csv(
        buffer,
        index=False,
        sep=spec["sep"],
        decimal=spec["decimal"],
        float_format=spec["float_format"],
        na_rep=_NA_REP,
        date_format=_DATE_FORMAT,
    )
    return buffer.getvalue().encode(spec["encoding"])


def read_locale_mirror(path: Path | str, *, locale: str) -> pd.DataFrame:
    """Parse one mirror file under its declared locale conventions."""
    spec = _locale_spec(locale)
    return pd.read_csv(
        Path(path),
        sep=spec["sep"],
        decimal=spec["decimal"],
        encoding=spec["encoding"],
    )


def _cell_round_trips(wanted: object, actual: object, tolerance: float) -> bool:
    wanted_missing = wanted is None or (
        not isinstance(wanted, (str, bytes)) and pd.isna(wanted)
    )
    actual_missing = actual is None or (
        not isinstance(actual, (str, bytes)) and pd.isna(actual)
    )
    if wanted_missing or actual_missing:
        return wanted_missing and actual_missing
    if isinstance(wanted, pd.Timestamp):
        return str(actual) == wanted.strftime(_DATE_FORMAT)
    if isinstance(wanted, (bool, np.bool_)):
        if isinstance(actual, (bool, np.bool_)):
            return bool(actual) == bool(wanted)
        return str(actual) == str(bool(wanted))
    if isinstance(wanted, Integral):
        try:
            return float(actual) == float(wanted)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
    if isinstance(wanted, Real):
        try:
            parsed = float(actual)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            try:  # a float inside an object column keeps the source decimal dot
                parsed = float(str(actual).replace(",", "."))
            except ValueError:
                return False
        return math.isfinite(parsed) and abs(parsed - float(wanted)) <= tolerance
    return str(actual) == str(wanted)


def verify_mirror_round_trip(
    table: pd.DataFrame,
    path: Path | str,
    *,
    locale: str,
    tolerance: float = ROUND_TRIP_TOLERANCE,
) -> pd.DataFrame:
    """Require the locale parser to reproduce the canonical source values.

    Numeric parity within ``tolerance``; dates, integers, booleans, and text
    exactly; null positions aligned. Returns the parsed frame.
    """
    path = Path(path)
    parsed = read_locale_mirror(path, locale=locale)
    if list(parsed.columns) != [str(column) for column in table.columns]:
        raise ValueError(
            f"{path}: mirror columns diverge from the canonical table "
            f"({list(parsed.columns)!r} != {list(table.columns)!r})"
        )
    if len(parsed) != len(table):
        raise ValueError(
            f"{path}: mirror row count {len(parsed)} diverges from the "
            f"canonical table row count {len(table)}"
        )
    for column in table.columns:
        source_column = table[column]
        parsed_column = parsed[column]
        for position in range(len(table)):
            wanted = source_column.iloc[position]
            actual = parsed_column.iloc[position]
            if not _cell_round_trips(wanted, actual, tolerance):
                raise ValueError(
                    f"{path}: the {locale} locale parser does not reproduce the "
                    f"canonical value within {tolerance!r} at row {position}, "
                    f"column {column!r}: canonical {wanted!r}, parsed {actual!r}"
                )
    return parsed


def write_locale_mirrors(
    tables: Mapping[str, pd.DataFrame],
    out_dir: Path | str,
    *,
    locales: tuple[str, ...] = REQUIRED_MIRROR_LOCALES,
) -> dict[str, dict[str, Path]]:
    """Write the required locale mirrors of canonical in-memory tables.

    Deterministic: stems are written in sorted order and identical inputs
    produce byte-identical files. Every written mirror is verified round-trip
    before this function returns.
    """
    if not isinstance(tables, Mapping) or not tables:
        raise ValueError(
            "tables must map canonical table stems to in-memory tables"
        )
    validated = {
        _require_stem(stem): _require_canonical_table(stem, table)
        for stem, table in tables.items()
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, dict[str, Path]] = {}
    for stem in sorted(validated):
        table = validated[stem]
        names = mirror_basenames(stem)
        written[stem] = {}
        for locale in locales:
            path = out_dir / names[locale]
            path.write_bytes(render_locale_csv(table, locale=locale))
            verify_mirror_round_trip(table, path, locale=locale)
            written[stem][locale] = path
    return written


def catalog_required_mirrors() -> dict[str, tuple[str, ...]]:
    """Required mirror basenames per cataloged canonical table (data-v4)."""
    required = {
        asset.public_basename: tuple(asset.required_projections)
        for asset in _publication_catalog().assets
        if asset.asset_class == "canonical_payload" and asset.required_projections
    }
    if not required:
        raise ValueError("the publication catalog declares no required mirrors")
    return required


def require_catalog_mirror_coverage(produced: Iterable[str]) -> None:
    """Every cataloged canonical table must have its required mirrors."""
    produced_names = {str(name) for name in produced}
    missing = [
        f"{name} (required mirror of {canonical_name})"
        for canonical_name, mirror_names in sorted(catalog_required_mirrors().items())
        for name in mirror_names
        if name not in produced_names
    ]
    if missing:
        raise ValueError(
            "cataloged canonical tables are missing required locale mirrors: "
            + "; ".join(missing)
        )
