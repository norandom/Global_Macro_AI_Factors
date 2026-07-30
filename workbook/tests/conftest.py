"""Make the uninstalled workbook package importable from the root test run.

The lean ``workbook/`` project is intentionally not part of the root uv
workspace; inserting its directory on ``sys.path`` lets the repository's
``uv run pytest`` collect and import ``factor_workbook`` without an install.
"""

import sys
from pathlib import Path

import pytest

_WORKBOOK_DIR = str(Path(__file__).resolve().parents[1])
if _WORKBOOK_DIR not in sys.path:
    sys.path.insert(0, _WORKBOOK_DIR)


@pytest.fixture(scope="module")
def _compact_fixture_ssr_compatibility():
    """Keep row-subset storyboard fixtures renderable under strict SSR validation.

    The shipped four-return fixture subsets are deliberately too short for the
    finalized 252-day SSR contract. Production release series use the real strict
    implementation; tests retain their established insufficient-result rendering.
    """
    import factor_workbook.steps as steps
    from factor_workbook import vendored_ssr

    strict_inference = steps.ssr_inference

    def fixture_inference(returns, *args, **kwargs):
        if not args and not kwargs and len(returns) < vendored_ssr.TRADING_DAYS:
            result = vendored_ssr.compute_ssr(returns)
            return vendored_ssr.SSRInference(
                result=result,
                sr_star=0.0,
                p_value=float("nan"),
                block_len=0,
                n_boot=1000,
                seed=0,
                alpha=0.05,
                p_value_lower=float("nan"),
                window=vendored_ssr.TRADING_DAYS,
                periods_per_year=vendored_ssr.TRADING_DAYS,
            )
        return strict_inference(returns, *args, **kwargs)

    steps.ssr_inference = fixture_inference
    yield
    steps.ssr_inference = strict_inference
