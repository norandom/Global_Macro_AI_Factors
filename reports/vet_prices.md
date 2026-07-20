# Price data QC — FMP `etf_prices`

Generated 2026-07-20 08:40 by `scripts/vet_prices.py`.
Data span: 2010-01-04 .. 2026-05-29. Universe: SWDA.L, XLK, IAU, BIL, SPY.

## Error rate, % of rows flagged

| symbol   |   n_rows |   structural |   fat_finger |   jitter |   level_shift |   stale |   ohlc_quirk |   any |
|:---------|---------:|-------------:|-------------:|---------:|--------------:|--------:|-------------:|------:|
| BIL      |     4125 |        0     |            0 |    0     |             0 |   0     |        0     | 0     |
| IAU      |     4125 |        0     |            0 |    0     |             0 |   0     |        0     | 0     |
| SPY      |     4125 |        0     |            0 |    0.048 |             0 |   0     |        0     | 0.048 |
| SWDA.L   |     4097 |        0.439 |            0 |    0.024 |             0 |   0.317 |       17.916 | 0.781 |
| XLK      |     4125 |        0     |            0 |    0.024 |             0 |   0     |        0     | 0.024 |
| TOTAL    |    20597 |        0.087 |            0 |    0.019 |             0 |   0.063 |        3.564 | 0.17  |

Buckets: `structural` OHLC impossible (high<low, close/open >1% outside range, price<=0); `fat_finger` idiosyncratic reversed spike >=5%; `jitter` same but smaller (bad ticks); `level_shift` big non-reversed jump (missed split — back-adjust, never delete); `stale` >=5 flat closes on zero volume; `ohlc_quirk` close/open <=1% outside range with single-print bars (benign thin-LSE auction pattern — close usable, range not). `any` counts errors only, quirks excluded.

Detector: rolling 252d median/MAD z > 6 with >= 60% next-bar reversal, 0.2% absolute floor; market-wide days (>= 50% of the full 112-symbol cross-section at |z|>3) are never flagged.

## Worst flagged rows (by |z|)

| symbol   | date                | structural   | fat_finger   | jitter   | level_shift   | stale   | ohlc_quirk   |           z |
|:---------|:--------------------|:-------------|:-------------|:---------|:--------------|:--------|:-------------|------------:|
| SPY      | 2018-03-26 00:00:00 | False        | False        | True     | False         | False   | False        |   8.27152   |
| SPY      | 2018-03-23 00:00:00 | False        | False        | True     | False         | False   | False        |  -6.93457   |
| SWDA.L   | 2018-02-06 00:00:00 | False        | False        | True     | False         | False   | False        |  -6.43869   |
| XLK      | 2018-03-26 00:00:00 | False        | False        | True     | False         | False   | False        |   6.41128   |
| SWDA.L   | 2011-08-18 00:00:00 | True         | False        | False    | False         | False   | False        |  -6.06595   |
| SWDA.L   | 2011-03-14 00:00:00 | True         | False        | False    | False         | False   | False        |  -4.64792   |
| SWDA.L   | 2011-08-04 00:00:00 | True         | False        | False    | False         | False   | False        |  -3.6395    |
| SWDA.L   | 2011-08-24 00:00:00 | True         | False        | False    | False         | False   | False        |   2.66631   |
| SWDA.L   | 2011-10-10 00:00:00 | True         | False        | False    | False         | False   | False        |   2.50446   |
| SWDA.L   | 2011-07-08 00:00:00 | True         | False        | False    | False         | False   | False        |  -2.27776   |
| SWDA.L   | 2012-05-04 00:00:00 | True         | False        | False    | False         | False   | False        |  -1.9171    |
| SWDA.L   | 2012-08-02 00:00:00 | True         | False        | False    | False         | False   | False        |  -1.28869   |
| SWDA.L   | 2012-06-11 00:00:00 | True         | False        | False    | False         | False   | False        |  -0.421413  |
| SWDA.L   | 2012-11-13 00:00:00 | True         | False        | False    | False         | False   | False        |   0.354001  |
| SWDA.L   | 2011-10-17 00:00:00 | True         | False        | False    | False         | False   | False        |  -0.28334   |
| SWDA.L   | 2011-08-26 00:00:00 | True         | False        | False    | False         | False   | False        |   0.0419152 |
| SWDA.L   | 2010-06-24 00:00:00 | False        | False        | False    | False         | True    | False        |   0         |
| SWDA.L   | 2010-06-28 00:00:00 | False        | False        | False    | False         | True    | False        |   0         |
| SWDA.L   | 2010-06-25 00:00:00 | False        | False        | False    | False         | True    | False        |   0         |
| SWDA.L   | 2010-01-25 00:00:00 | True         | False        | False    | False         | False   | False        | nan         |
