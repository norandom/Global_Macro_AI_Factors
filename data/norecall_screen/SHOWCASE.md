# Model selection: certified no-recall screen

Screen: R8 certified no-recall model selection · cutoff 2023-12-01 · run 2026-07-03 · NVIDIA NIM endpoints.
Source: `results.json` in this directory; per-prompt audit trail in `evidence/<model>/evidence.parquet`
and `data/csv_mirrors/norecall_evidence_<model>.csv`. Raw input: `data/macro_panel_monthly.parquet`
(pre-cutoff rows, rendered identifying vs anonymized vs prose-confounded).

## Results

| Model | Controlled AUC (95% CI) | Perm. p | Pos. control AUC | Perm. p | Parse rate | n per class | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| openai/gpt-oss-20b | **0.926** (0.885–0.961) | 0.002 | 0.799 | 0.002 | 1.00 | 167 | recalls |
| microsoft/phi-4-mini-instruct | 0.695 (0.646–0.781) | 0.002 | 0.885 | 0.002 | 1.00 | 120 | recalls |
| meta/llama-4-maverick-17b-128e | 0.637 (0.561–0.721) | 0.008 | 0.881 | 0.002 | 0.00 | 120 | recalls |
| openai/gpt-oss-120b | 0.633 (0.598–0.731) | 0.004 | 0.729 | 0.002 | 0.40 | 167 | recalls |
| meta/llama-3.3-70b-instruct | — | — | — | — | — | — | screen failed (not exonerated) |

Every screenable model shows statistically significant recall of the pre-cutoff macro history;
the certified-no-recall set is empty. gpt-oss-20b — the strongest recaller — was selected
deliberately, to be run *recall-guarded* rather than assumed clean.

## Metric explanations

| Metric | What it measures | How to read it |
| --- | --- | --- |
| Controlled AUC | Cross-validated AUC of a classifier on the model's MIA features (log-probability signatures), separating two framings of the *same* pre-cutoff macro states: identifying (real asset names/dates, recall possible) vs anonymized (recall impossible). | 0.5 = the model treats both framings identically → no recall signal. The further above 0.5, the more the model's behavior betrays having seen the identified history. |
| 95% CI | Confidence interval on the controlled AUC. | If the interval sits clear of 0.5, the recall signal is not sampling noise. |
| Perm. p (controlled) | Permutation test: probability of an AUC this large under shuffled class labels. | Verdict rule: `recalls` when p < α and AUC > 0.5. All screened models: p ≤ 0.008. |
| Positive control AUC | Same detector on a prose-confounded rendering vs the anonymized class — a framing where the detector *must* fire if it works at all. | Validates the instrument, not the model. If this failed, the verdict degrades to "detector unvalidated" instead of exonerating the model. |
| Perm. p (control) | Significance of the positive-control AUC. | All 0.002 → the detector is confirmed working for every screened model. |
| Parse rate | Fraction of a 20-reply sample that parses into a usable answer. | A capability check only — the AUC is computed from logprobs, not parsed text, so maverick gets an AUC despite parse rate 0. |
| n per class | Prompts per class, capped by available pre-cutoff panel rows. | Sample size behind the CI (120 or 167 here). |
| Verdict | `recalls` / `certified no-recall` / `inconclusive` / `screen failed`. | "Screen failed" (llama-3.3-70b: too few surviving feature rows for stratified CV) means *not exonerated*, not passed. |
