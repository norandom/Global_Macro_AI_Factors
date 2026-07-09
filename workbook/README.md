# Factor Workbook — Excel-native audit of the factor-scoring assessment

## 1. What this is

A five-step storyboard audit workbook. Each sheet (S1–S5) walks a reviewer through the
factor-scoring assessment, and every headline number is **re-derived from the raw
`data-v1` GitHub Release data in front of the reviewer** — the workbook performs no
inference and no scoring runs. Where a re-derived value disagrees with the published
value beyond display precision, a check flag says so instead of silently preferring
either number.

## 2. Runtime prerequisite (read this first)

The Excel surface uses **PyXLL 5.12.x, a COMMERCIAL Windows-Excel add-in**: it requires
a **per-user annual license** (a **30-day trial** is available from
[pyxll.com](https://www.pyxll.com)) and runs only in Excel on Windows with Python up to
3.12.

If your environment lacks the commercial add-in license or has no Excel at all, that is
fine for verification: **the data/computation layer runs and tests WITHOUT Excel on any
platform.** Everything below the Excel presentation layer — release fetch, schema
contract, re-derivation math, verification flags, step views — is exercised by the
automated test suite on Linux (see section 8). Only the behaviors in the manual
checklist (section 7) need a real Excel.

## 3. Install on Windows

1. Install Python **3.12** (PyXLL supports up to 3.12; this package requires
   `>=3.12,<3.13`).
2. Install the lean package from this directory (or from a built wheel):

   ```bat
   pip install <path-to-repo>\workbook
   ```

   Optional extras:

   ```bat
   pip install "<path-to-repo>\workbook[token]"   :: keyring, for a future private repo
   pip install "<path-to-repo>\workbook[stats]"   :: scikit-learn, deeper certification re-derivation
   ```

3. Install and activate the PyXLL add-in (license or trial key required):

   ```bat
   pip install pyxll
   pyxll install
   pyxll activate
   ```

4. Copy `pyxll.cfg.example` from this directory to your PyXLL config location as
   `pyxll.cfg` and set the placeholder paths (Python executable, `pythonpath` pointing
   at your installed environment). The only module PyXLL needs to load is
   `factor_workbook.addin`.

## 4. Generate or obtain the workbook skeleton

The workbook file is generated, not hand-built. From this directory:

```bat
python build_workbook.py
```

This emits `factor_workbook.xlsx`: a navigation `Index` sheet plus the five step sheets,
with all `=FW_*` formulas pre-wired. A shipped copy of the file works identically — the
skeleton contains only formulas and framing text, never data or credentials.

## 5. Using it

- **One cell drives everything**: the release tag in `Index!B1` (named range
  `RELEASE_TAG`, pre-filled with `data-v1`). `FW_LOAD(Index!$B$1)` in `Index!B2`
  produces the client handle every sheet derives from.
- **Changing the tag re-loads**: editing `Index!B1` is the explicit version switch —
  every dataset reloads from the newly named release and the displayed version changes.
  Nothing switches versions silently.
- **Provenance**: `FW_PROVENANCE` on the Index sheet lists the source URL, release tag,
  and checksum of every loaded asset; `FW_VERSION` shows the active release version.
- **Checks columns**: each step sheet's `FW_CHECKS` table shows the re-derivation flags —
  published value vs. re-derived value, with any beyond-tolerance discrepancy flagged.
- **Errors**: a failed asset fetch renders in-cell as
  `#ERROR <asset>: <cause> — <detail>`; the workbook never substitutes stale data.

## 6. Token setup for a future private repo

The repository is currently **public** — no token is needed and the token path stays
dormant. The client consults a token only if the public download URL is refused. When
the repo goes private, supply a GitHub token via **keyring** (install the `[token]`
extra):

```python
import keyring
keyring.set_password("factor-workbook", "github", "<your-token>")
```

Fallback: the `GITHUB_TOKEN` environment variable.

**NEVER put a token in a cell, a formula, or the workbook file itself.** No worksheet
function accepts a token argument, and the token is never written into any shared
artifact — keyring and the environment variable are the only supported channels.

## 7. Manual Excel verification checklist

The automated suite runs against the PyPI PyXLL stub on Linux; the following behaviors
exist only inside a licensed Windows Excel and are **excluded from automated tests by
design**. Verify them by hand after installation:

- [ ] The add-in loads via `pyxll.cfg` and the `FW_*` functions appear in Excel
      (`modules = factor_workbook.addin`).
- [ ] Async `FW_LOAD(Index!$B$1)` fills its cell **without freezing the Excel UI**
      while the release downloads.
- [ ] Dataframe results spill and auto-size correctly; no table collides with the next
      anchor (**40-row** gap ceiling between table anchors on each sheet).
- [ ] Object-cache handles (`FW_LOAD` / `FW_STEP` results) survive **recalc** and are
      reusable across dependent formulas.
- [ ] The workbook operates fully **ribbon**-less — plain worksheet functions only, no
      custom ribbon or macros required.
- [ ] `#ERROR <asset>: <cause> — <detail>` strings render in-cell when an asset fetch
      fails (e.g. temporarily set a bogus tag in `Index!B1`).
- [ ] The provenance table (`FW_PROVENANCE`) and version cell (`FW_VERSION`) populate
      after a successful load.
- [ ] Editing `Index!B1` to another tag triggers a visible full re-load and the
      displayed version changes.

## 8. Development on Linux (no Excel)

Everything except the checklist above is covered by the automated tests. From the
repository root:

```sh
uv run pytest workbook/tests -q   # workbook suite
uv run pytest -q                  # whole repository
```

Import smoke test without installing:

```sh
PYTHONPATH=workbook uv run python -c 'import factor_workbook'
```
