"""factor_workbook — Excel-native audit view of the factor-scoring assessment.

Lean package consumed two ways from one source tree: collected by the
repository's root ``uv run pytest`` on Linux, and pip-installed from
``workbook/`` on the Windows Excel machine (PyXLL). No heavy work happens
at import time.

The package re-exports its modules (not symbols), so each later task owns
exactly one module file with no ``__init__`` churn.
"""

import factor_workbook.addin as addin
import factor_workbook.certification as certification
import factor_workbook.contract as contract
import factor_workbook.rederive as rederive
import factor_workbook.release as release
import factor_workbook.steps as steps
import factor_workbook.vendored_ssr as vendored_ssr
import factor_workbook.verify as verify

__version__ = "0.1.0"

__all__ = [
    "release",
    "contract",
    "rederive",
    "vendored_ssr",
    "certification",
    "verify",
    "steps",
    "addin",
]
