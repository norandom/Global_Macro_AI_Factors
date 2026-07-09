# Vendored from macro_framework/factor_scoring.py::certification_stats (R2.6
# deepening).
# Source: macro_framework/factor_scoring.py (this repository)
# Commit: c85c2ed73eb0aba52b6bb937e9ef99a71272b4a6
# Verbatim copy of the function EXCEPT the availability guard at the top of
# the body (marked "vendoring deviation"), which turns a missing optional
# scikit-learn into a clear RuntimeError instead of an ImportError.
# Do not edit by hand — re-sync from the source module and re-run
# workbook/tests/test_certification.py if the original diverges.
"""Optional certification-statistics re-derivation behind the scikit-learn
``stats`` extra; reports itself unavailable (never fails at import) when the
extra is not installed (R2.6 deepening).

The full separation statistics (cross-validated AUC point estimate, per-class
bootstrap CI, permutation p) are the deeper, optional half of R2.6 — the
always-available half (class counts + feature summary stats) lives in
``rederive.evidence_class_stats``. Install ``factor-workbook[stats]`` to
enable :func:`certification_stats`; :func:`available` reports whether it is.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def available() -> bool:
    """Whether the optional ``stats`` extra (scikit-learn) is installed.

    Checked lazily (no sklearn import) so the lean surface imports this
    module without the extra and can report the re-derivation unavailable.
    """
    try:
        return importlib.util.find_spec("sklearn") is not None
    except (ImportError, ValueError):  # e.g. sys.modules["sklearn"] = None
        return False


def certification_stats(
    x_is: Sequence[Sequence[float]],
    x_oos: Sequence[Sequence[float]],
    *,
    n_boot: int = 200,
    n_perm: int = 500,
    n_splits: int = 5,
    seed: int = 0,
) -> tuple[float, float, float, float]:
    """Offline separation statistics on gathered standardized features (R8.2).

    Pure statistics — NO live inference. Operates on the two standardized MIA
    feature matrices gathered once per candidate (identifying vs anonymized, or
    prose-confounded vs anonymized) and reports how separable the two classes
    are, with statistical-certainty measures:

    - **Point estimate** — the mean fold AUC of a stratified ``n_splits``-fold
      cross-validated ``LogisticRegression(class_weight='balanced',
      solver='liblinear', random_state=seed)`` (the same classifier family the
      MCS calibrator uses, so the screen measures the separation the deployed
      calibrator could exploit).
    - **CI** — a bootstrap 2.5/97.5 percentile interval: rows are resampled
      with replacement PER CLASS (preserving both class sizes) ``n_boot``
      times and the CV AUC recomputed each time. Known ceiling (review
      2026-07-03): with-replacement duplicates span CV fold boundaries, so the
      bootstrap distribution is upward-biased (~+0.1 CI-midpoint on pure noise
      at n=20/class, shrinking with n). The bias is strictly CONSERVATIVE for
      the R8.4 gate — certification requires the CI to CONTAIN 0.5, so
      inflation can only block a certification, never grant a false one.
      Upgrade path if it ever blocks a true no-recall candidate: bootstrap the
      held-out fold predictions instead of the rows.
    - **Permutation p** — the class labels are shuffled ``n_perm`` times, the
      CV AUC recomputed each time, and the two-sided p-value reported as
      ``(1 + #{|auc_perm − 0.5| ≥ |auc_obs − 0.5|}) / (n_perm + 1)`` (the
      add-one permutation estimator; never exactly 0).

    Deterministic given ``seed``: one ``numpy`` Generator seeded from ``seed``
    drives all resampling in a fixed order, and the CV splitter / classifier
    share the same ``seed``.

    Args:
        x_is: the recall-class standardized feature rows (label 1).
        x_oos: the anonymized-class standardized feature rows (label 0).
        n_boot: bootstrap resamples for the CI.
        n_perm: label permutations for the p-value.
        n_splits: stratified CV folds.
        seed: the deterministic seed for resampling, splitting, and fitting.

    Returns:
        ``(auc, ci_low, ci_high, perm_p)``.

    Raises:
        RuntimeError: when the optional ``stats`` extra is not installed
            (vendoring deviation — the original raised ImportError here).
        ValueError: when either class has fewer than ``n_splits`` rows —
            stratified ``n_splits``-fold CV cannot guarantee both classes in
            every fold on such degenerate input.
    """
    # vendoring deviation: lean-surface guard (the ONLY change vs the source).
    if not available():
        raise RuntimeError(
            "certification re-derivation needs scikit-learn: "
            "install factor-workbook[stats]"
        )
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    xi = np.asarray(x_is, dtype=np.float64)
    xo = np.asarray(x_oos, dtype=np.float64)
    if xi.ndim != 2 or xo.ndim != 2 or len(xi) < n_splits or len(xo) < n_splits:
        raise ValueError(
            "certification_stats: each class needs at least n_splits "
            f"(={n_splits}) feature rows of equal width for stratified CV; "
            f"got n_is={len(xi) if xi.ndim else 0}, n_oos={len(xo) if xo.ndim else 0}."
        )

    x = np.vstack([xi, xo])
    y = np.concatenate(
        [np.ones(len(xi), dtype=np.int64), np.zeros(len(xo), dtype=np.int64)]
    )

    def _cv_auc(xm: np.ndarray, ym: np.ndarray) -> float:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_aucs: list[float] = []
        for train_idx, test_idx in skf.split(xm, ym):
            clf = LogisticRegression(
                class_weight="balanced", solver="liblinear", random_state=seed
            )
            clf.fit(xm[train_idx], ym[train_idx])
            fold_aucs.append(
                float(
                    roc_auc_score(
                        ym[test_idx], clf.predict_proba(xm[test_idx])[:, 1]
                    )
                )
            )
        return float(np.mean(fold_aucs))

    auc_obs = _cv_auc(x, y)

    rng = np.random.default_rng(seed)

    # Bootstrap CI: resample rows with replacement PER CLASS so both class
    # sizes are preserved (stratified CV stays valid on every resample).
    n_i, n_o = len(xi), len(xo)
    boot_aucs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx_i = rng.integers(0, n_i, n_i)
        idx_o = rng.integers(0, n_o, n_o)
        boot_aucs[b] = _cv_auc(np.vstack([xi[idx_i], xo[idx_o]]), y)
    ci_low = float(np.percentile(boot_aucs, 2.5))
    ci_high = float(np.percentile(boot_aucs, 97.5))

    # Two-sided permutation p against chance separation (AUC = 0.5).
    obs_dev = abs(auc_obs - 0.5)
    hits = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        if abs(_cv_auc(x, y_perm) - 0.5) >= obs_dev:
            hits += 1
    perm_p = (1 + hits) / (n_perm + 1)

    return auc_obs, ci_low, ci_high, float(perm_p)
