# Simulation examples — final ext26 trio (`data-v3`)

A short import guide for the current three-line comparison in the latest release:

- **Static B&H 16-26** — the raw-return benchmark
- **Factor ext26** — the PIT macro-factor line
- **SJM×crowding de-risk v2** — the drawdown-armed overlay

These files live in the GitHub release
[`data-v3`](https://github.com/norandom/Global_Macro_AI_Factors/releases/tag/data-v3).
Use the `_de.csv` variants on German Excel: semicolon-separated, comma decimals,
zero locale transform steps.

## Files

Base URL prefix:

```text
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v3/
```

### Equity curves (`Date; value; daily_return; drawdown`)

```text
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v3/static_bh_equity_2016_2026_de.csv
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v3/factor_equity_ext2026_de.csv
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v3/sjm_crowding_derisk_v2_equity_ext2026_de.csv
```

### Trio tear sheet (one row per line, one column per metric)

```text
https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v3/tear_sheet_trio_ext2026_de.csv
```

The trio tear sheet is the **common-window** comparison used in the offline
notebook: all three lines are re-evaluated on the same overlap window so the
numbers are directly comparable.

## Fastest path in Excel

### Option 1 — import one curve directly

**Daten → Daten abrufen → Aus dem Web**, paste one of the `_de.csv` URLs above,
then **Laden**.

Sanity anchors:
- `value` starts at **10000**
- `daily_return` and `drawdown` are fractions (`-0,008` = `-0,8 %`)
- `drawdown` is already precomputed, so you can chart equity and drawdown with no formulas

### Option 2 — reusable Power Query (swap only the filename)

Create **Data → Get Data → From Other Sources → Blank Query**, then paste this in
the formula bar:

```powerquery
let
    Source = Csv.Document(
        Web.Contents("https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v3/factor_equity_ext2026_de.csv"),
        [Delimiter=";", Encoding=65001, QuoteStyle=QuoteStyle.None]
    ),
    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    Typed = Table.TransformColumnTypes(
        PromotedHeaders,
        {
            {"Date", type date},
            {"value", type number},
            {"daily_return", type number},
            {"drawdown", type number}
        },
        "de-DE"
    )
in
    Typed
```

To reuse the same query for the other two curves, change only the filename:

- `static_bh_equity_2016_2026_de.csv`
- `factor_equity_ext2026_de.csv`
- `sjm_crowding_derisk_v2_equity_ext2026_de.csv`

That keeps the same sheet, charts, and formulas while swapping the underlying line.

## Power Query for the trio tear sheet

```powerquery
let
    Source = Csv.Document(
        Web.Contents("https://github.com/norandom/Global_Macro_AI_Factors/releases/download/data-v3/tear_sheet_trio_ext2026_de.csv"),
        // QuoteStyle.Csv, not .None: the "Model note (REF)" column contains a
        // semicolon and is written quoted. QuoteStyle.None would split that row
        // into extra columns.
        [Delimiter=";", Encoding=65001, QuoteStyle=QuoteStyle.Csv]
    ),
    PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true])
in
    PromotedHeaders
```

This file is a reference table, not a reusable time-series sheet: load it next to
your chart sheet and read across the three rows.

## What the three lines mean

- **Static B&H 16-26**: the hindsight-flattered benchmark used to show how good
  a naive in-sample ETF selection can look.
- **Factor ext26**: the recall-guarded factor line extended through 2026.
- **SJM×crowding de-risk v2**: the same factor seat with the drawdown-armed
  de-risk overlay from nb17; it is a risk-control variant, not a separate alpha claim.

For the broader S0→S5 walkthrough, use [`ASSESSMENT.md`](ASSESSMENT.md).
For the static-only import note, use [`S0.md`](S0.md).
For per-run provenance and windows, use [`SIMULATIONS.md`](SIMULATIONS.md).
