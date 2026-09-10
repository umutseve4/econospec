"""EconoSpec: cross language conformance tests for reproducible econometrics.

An independent student project. Not an official product of Bursa Uludag
University or its Department of Econometrics.

Importing this package pins the linear algebra backend to a single thread
before numpy is loaded. This is not a performance decision. A multithreaded
OpenBLAS splits a reduction across threads, and the split it chooses depends on
how busy the machine is at that moment, so the same dot product can come back
with a different last bit on the same runner minutes apart. The suite asserts
that the reference kernel reproduces the committed golden manifest bit for bit,
and that claim is only defensible when the summation order is fixed. Measured
case, recorded in docs/determinism.md: adf-001 params.const.estimate returned
-0.010046320349888796 where the manifest holds -0.010046320349888798, two units
in the last place apart, on an Intel Xeon Platinum 8370C.

The variables are set with setdefault, so an operator who deliberately exports
OMP_NUM_THREADS keeps control. tests/test_determinism.py then fails loudly
rather than letting the bit for bit assertion become intermittent again.
"""

from __future__ import annotations

import os as _os
import sys as _sys

__version__ = "0.1.0"

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

#: True when numpy had already been imported by the time this package was
#: imported, in which case the pinning below arrived too late to bind.
NUMPY_IMPORTED_BEFORE_PINNING = "numpy" in _sys.modules

for _variable in THREAD_VARIABLES:
    _os.environ.setdefault(_variable, "1")


def thread_pinning_report() -> dict:
    """Describe the threading environment the numerical results were produced in."""
    return {
        "numpy_imported_before_pinning": NUMPY_IMPORTED_BEFORE_PINNING,
        "variables": {name: _os.environ.get(name) for name in THREAD_VARIABLES},
        "single_threaded": all(
            _os.environ.get(name) == "1" for name in THREAD_VARIABLES
        )
        and not NUMPY_IMPORTED_BEFORE_PINNING,
    }


from . import compare, csvio, dgp, distributions, estimators, runner, specs  # noqa: E402

__all__ = [
    "__version__",
    "THREAD_VARIABLES",
    "NUMPY_IMPORTED_BEFORE_PINNING",
    "thread_pinning_report",
    "compare",
    "csvio",
    "dgp",
    "distributions",
    "estimators",
    "runner",
    "specs",
]
