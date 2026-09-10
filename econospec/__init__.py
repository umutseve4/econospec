"""EconoSpec: cross language conformance tests for reproducible econometrics.

An independent student project. Not an official product of Bursa Uludag
University or its Department of Econometrics.

Importing this package pins the linear algebra backend to a single thread
before numpy is loaded. An earlier version of this docstring said that pinning
is what makes the bit for bit claim in tests/test_conformance.py defensible.
The determinism hunt on this branch measured that claim and it did not hold,
so here is what pinning does and does not do.

What it does: removes one source of variation. A multithreaded reduction is
blocked across threads, and the blocking changes the summation order.

What it does not do: make the kernel reproducible. Thirty samples on ten
runners, pinned and unpinned, produced two distinct values for adf-001
params.const.estimate: -0.010046320349888796 on an Intel Xeon Platinum 8573C
and on an AMD EPYC 9V45, both pinned to one thread, against
-0.010046320349888798 on an AMD EPYC 7763 pinned to one thread and on every
runner at four threads. One unit in the last place, a relative difference of
1.7267252243215682e-16. Two jobs pinned to a single thread disagreeing with
each other is the whole answer: the kernel is also selected from the
instruction set the CPU reports, and no environment variable reaches that.

The fix is :mod:`econospec.linalg`, which sums exactly and never calls a
dispatched reduction. Pinning stays as hygiene, and tests/test_determinism.py
keeps asserting it, so the environment behind the numbers is a stated fact
rather than an accident. See docs/determinism.md for the full measurement.

The variables are set with setdefault, so an operator who deliberately exports
OMP_NUM_THREADS keeps control.
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


from . import (  # noqa: E402
    compare,
    csvio,
    dgp,
    distributions,
    estimators,
    linalg,
    runner,
    specs,
)

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
    "linalg",
    "runner",
    "specs",
]
