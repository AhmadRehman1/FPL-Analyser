"""research/ml — Phase 0 leakage-free residual ML research engine for FPL-Analyser.

This package is a RESEARCH layer. It does NOT touch production recommendations, the live
squad optimiser, the UI, or any existing Quant model output. It reads the existing DuckDB
(asof-safe, via fpl_quant.backtest.asof_scope) to ask one scientific question:

    Can machine learning identify systematic patterns in the errors made by the existing
    FPL Quant model and use those patterns to improve future predictions?

A negative result is a successful research result. The existing Quant model is the baseline
to beat; ML is never assumed to help.

Path bootstrap: importing this package inserts the repo's `src/` directory onto sys.path so
that `from fpl_quant import db, backtest` resolves whether this is imported as a package
(`python -m research.ml.experiment`) or run as a script
(`python research/ml/experiment.py`). This mirrors the convention tests/conftest.py already
establishes for the test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

# research/ml/__init__.py -> parents[2] is the repo root (research/ml -> research -> repo).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

__all__: list[str] = []
