"""Make the uninstalled workbook package importable from the root test run.

The lean ``workbook/`` project is intentionally not part of the root uv
workspace; inserting its directory on ``sys.path`` lets the repository's
``uv run pytest`` collect and import ``factor_workbook`` without an install.
"""

import sys
from pathlib import Path

_WORKBOOK_DIR = str(Path(__file__).resolve().parents[1])
if _WORKBOOK_DIR not in sys.path:
    sys.path.insert(0, _WORKBOOK_DIR)
