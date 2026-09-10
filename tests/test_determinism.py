"""The bit for bit claim is only true when nothing on the path is dispatched.

tests/test_conformance.py asserts that a fresh reference run reproduces the
committed golden manifest with exact float equality. That assertion was
intermittently red before this file existed, and the first diagnosis written
here was wrong. It said two units in the last place, load dependent, closed by
pinning the backend to one thread.

The thirty sample hunt measured all three claims and refuted them. The gap is
one unit in the last place. It is deterministic per machine, not load
dependent: three consecutive runs on the same runner failed three times out of
three. And pinning does not close it, because two jobs pinned to a single
thread returned different values from each other on different CPU families.

The fix is econospec.linalg, and tests/test_linalg.py guards it. These tests
keep guarding the pinning, which is still worth having as one fewer moving
part, and they keep the environment behind the numbers a stated fact rather
than an accident. See docs/determinism.md.
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
        "pinning arrived too late to bind. This no longer breaks the bit for "
        "bit claim, which econospec.linalg carries, but it does mean the "
        "environment is not the one the envelopes say it is"
    )
    for name in econospec.THREAD_VARIABLES:
        assert os.environ.get(name) == "1", (
            "%s is %r rather than '1'. A multithreaded reduction changes the "
            "last bits of a dot product depending on how it is blocked. That "
            "is no longer the only such hazard, and it is no longer the one "
            "that produced the observed failure, but it is the one this "
            "process can control. See docs/determinism.md."
            % (name, os.environ.get(name))
        )
    assert report["single_threaded"] is True


def test_the_reference_kernel_repeats_itself_across_many_runs():
    """Ten consecutive runs, not two, because the observed failure rate was low.

    This test could never have caught the real bug, and saying so is the point.
    The divergence was between machines, not between runs on one machine, so
    ten repetitions in one process would have passed on every runner that ever
    failed. It is kept because it is cheap and it does catch state that leaks
    between fits. The cross machine claim is measured by the determinism
    workflow and guarded by tests/test_linalg.py.
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
