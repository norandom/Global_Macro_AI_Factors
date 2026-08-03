# data-v5 — full-period (2016-2026) trio release

Built 2026-08-02T09:06:06.351588+00:00. All files are de-DE locale CSVs (semicolon separator,
comma decimal) — they open directly in German-locale Excel.

## tear_sheets/
- `tear_sheet_trio_10y_de.csv` — canonical trio metrics, common window 2016-02-01..2026-01-30
- `tear_sheet_trio_max_de.csv` — canonical trio metrics, each strategy's maximum window
- `tear_sheet_trio_ext2026_de.csv` — canonical trio metrics, aligned window 2016-01-04..2026-06-30
- `nb18_3_tear_sheet_10y_de.csv` — notebook 18.3 §4 extended tear sheet (34 rows: distribution,
  CAPM/market-model, active-vs-SPY, and 4-ETF-basket blocks)
- `nb18_3_risk_decomposition_10y_de.csv` — market-model + basket decomposition table

## equity_curves/ (2016-01-01..2026-06-30)
- `factor_pit_equity_2016_2026_de.csv` — AI macro-factor (PIT) daily equity from the walk-forward
  simulation (first rebalance 2016-01-04)
- `sjm_overlay_equity_2016_2026_de.csv` — SJM v3 crowding de-risk overlay daily equity
  (unit-anchored: 1.0 at the 2016-01-04 start of trading)
- `static_bh_25pct_equity_2016_2026_de.csv` — static Buy & hold 25% x (SWDA.L, XLK, IAU, BIL),
  bought at the first common trading day on/after 2016-02-01

## macro/
- `macro_panel_monthly_2016_2026_de.csv` — monthly macro panel (z-scores + raw levels) driving the
  factor prompts. Includes the 2015-12-31 row: month-end stamped and consumed point-in-time, it is
  the source of the first 2016-01-04 rebalance decision
- `factor_loadings_pit_2016_2026_de.csv` — recorded monthly LLM factor loadings (PIT variant)
- `factor_scores_pit_2016_2026_de.csv` — recorded monthly p_memorized recall-guard scores (PIT variant)

## Provenance
Factor run `factor_ext2026_2016-01-01_2026-06-30_v1` (walk-forward simulation, evidence replay, zero
provider calls), SJM run `sjm_crowding_v3_total_return_bil_provisional_full2016_20260801T100439Z`, market snapshot
`provisional_market_total_return_fx_2026-06-30_v1`. Full sha256 inventory in `manifest.json`.
