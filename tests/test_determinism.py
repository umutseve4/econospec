"""The bit for bit claim is only true under a fixed summation order.

tests/test_conformance.py asserts that a fresh reference run reproduces the
committed golden manifest with exact float equality. That assertion was
intermittently red before this file existed: two of thirty samples failed on
adf-001 params.const.estimate by two units in the last place, because a
multithreaded OpenBLAS reduces in an order that depends on machine load.

Importing econospec pins the backend to one thread before numpy loads. These
tests fail loudly if that pinning ever stops working, so the failure mode is a
named, deterministic error instead of a rare red run nobody can reproduce.
"""

from __future__ import annotations

import os

import econospec
from econospec.runner import flatten, run_spec
from econospec.specs import load_all_specs

SPECS = load_all_specs(".")


def test_the_linear_algebra_backend_is_pinned_to_one_thread():
    report = econospec.thread_pinning_report()
    assert not report["numpy_imported_before_pinning"], (
        "numpy was already imported when econospec was imported, so the thread "
        "pinning arrived too late to bind and bit for bit equality against the "
        "golden manifest is no longer a claim this suite can make"
    )
    for name in econospec.THREAD_VARIABLES:
        assert os.environ.get(name) == "1", (
            "%s is %r rather than '1'. A multithreaded reduction changes the "
            "last bits of a dot product depending on machine load, which makes "
            "the golden manifest comparison intermittent. See "
            "docs/determinism.md." % (name, os.environ.get(name))
        )
    assert report["single_threaded"] is True


def test_the_reference_kernel_repeats_itself_across_many_runs():
    """Ten consecutive runs, not two, because the observed failure rate was low.

    The intermittency showed up in roughly one run in fifteen. A test that
    computes a spec twice would have passed through the whole episode without
    noticing, so this one runs the smallest and the most ill conditioned specs
    ten times each and requires every digit to be identical.
    """
    interesting = [s for s in SPECS if s["id"] in ("adf-001", "ols-003")]
    assert len(interesting) == 2, [s["id"] for s in SPECS]
    for spec in interesting:
        first = flatten(run_spec(spec, "."))
        for _ in range(9):
            again = flatten(run_spec(spec, "."))
            assert set(again) == set(first), spec["id"]
            for key in first:
                assert again[key] == first[key], (spec["id"], key, again[key], first[key])
