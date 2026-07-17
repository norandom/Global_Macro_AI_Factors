# Global Macro AI Factors

A research repository that builds AI-scored macro factor portfolios and then audits
its own results for lookahead. The question it keeps asking: was the LLM predicting
the decade, or remembering it? Everything ships as versioned data releases so the
answer can be checked in Excel, without trusting this repo's code.

## Use the data

The latest release is
[data-v3](https://github.com/norandom/Global_Macro_AI_Factors/releases/tag/data-v3)
(the post-cutoff extension; `data-v2` and `data-v1` stay immutable underneath it).
Each tabular asset ships as parquet plus a CSV mirror with the same basename.
Files ending in `_de.csv` are semicolon-separated with comma decimals and load in
German Excel with zero transform steps; plain `.csv` files use `.` decimals and
need a locale step on German systems.

Two guides cover the Excel side:

- [workbook/ASSESSMENT.md](workbook/ASSESSMENT.md) walks the whole storyboard
  (S0 static problem through S5 luck vs skill) in one reusable Power Query sheet,
  with the equations, Excel formulas, and the questions a PM would ask of each number.
- [workbook/S0.md](workbook/S0.md) shows three ways to get the static
  buy-and-hold line into Excel, easiest first, including the German-locale trap.

## Use the workbook

The generated Excel workbook re-derives every published headline number from the
raw release data in front of the reviewer. Install the package from the repo root:

```sh
pip install ./workbook
```

The Excel surface needs PyXLL, a commercial Windows add-in (30-day trial
available). Setup, the manual verification checklist, and the token rules are in
[workbook/README.md](workbook/README.md). The data and computation layer below
Excel runs and tests on any platform.

## Develop

```sh
uv sync
uv run pytest -q   # 418 tests, no network needed
```

The memorization scoring and guarding come from
[recall-guard](https://github.com/norandom/memguard_alpha) (pinned `@v0.1.0` in
`pyproject.toml`).

| Path | Contents |
|---|---|
| `macro_framework/` | Shared library: SSR, allocation, backtesting, factor scoring |
| `notebooks/` | 01–15, the research narrative in order |
| `workbook/` | Installable Excel audit package, its tests, and the Excel docs |
| `scripts/` | Builders for the release data packs and calibrations |
| `.kiro/` | Specs and steering for the spec-driven workflow |
| `data/` | Local artifacts; the releases are the public copy |

Nothing here claims predictive alpha. The scoring model's directional accuracy
is 0.389 with a Wilson interval that contains 0.5, and that coin-flip result is
expected and correct: it was never supposed to forecast. The deployable
portfolio line is recall-guarded, meaning LLM tilts are discounted by a measured
memorization probability; the unguarded line exists only as a diagnostic control.
Luck vs skill is judged with the SSR cited below, and every headline line so far
is luck-compatible (SSR far below 1.96). The 4-ETF universe was itself selected
in-sample, which is deliberate: that hindsight-flattered static line is the
problem exhibit the rest of the pipeline measures.

## Citation

The stability metric used throughout (`macro_framework/ssr.py`, vendored into the
workbook package) is the Sharpe Stability Ratio:

Bajo Traver, Mario, and Alejandro Rodríguez Domínguez (2026). "The Sharpe
Stability Ratio: Temporal Consistency of Risk-Adjusted Performance." SSRN
working paper 6344658, posted January 15, 2026.
<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6344658>

```bibtex
@misc{bajotraver2026ssr,
  author       = {Bajo Traver, Mario and Rodr{\'i}guez Dom{\'i}nguez, Alejandro},
  title        = {The Sharpe Stability Ratio: Temporal Consistency of
                  Risk-Adjusted Performance},
  year         = {2026},
  howpublished = {SSRN Working Paper 6344658},
  url          = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6344658}
}
```
