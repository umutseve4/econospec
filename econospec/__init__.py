"""EconoSpec: cross language conformance tests for reproducible econometrics.

An independent student project. Not an official product of Bursa Uludag
University or its Department of Econometrics.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import compare, csvio, dgp, distributions, estimators, runner, specs

__all__ = [
    "__version__",
    "compare",
    "csvio",
    "dgp",
    "distributions",
    "estimators",
    "runner",
    "specs",
]
