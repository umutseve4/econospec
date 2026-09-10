"""Tests for the comparator itself.

A conformance suite that cannot fail is decoration. These tests mutate known
good results and assert that the comparator turns red, so any change that
loosens the comparison has to break this file first, in the diff, where a
reviewer can see it.
"""

from __future__ import annotations

import json
import math
import os

from econospec.compare import (
    compare_flat,
    excluded_reason,
    is_green,
    summarise,
    tolerance_for,
)
from econospec.runner import flatten, run_spec
from econospec.specs import load_all_specs

SPECS = load_all_specs(".")
MUTATION_SIZE = 1e-3


def _golden(spec_id: str) -> dict:
    with open(os.path.join("fixtures", "golden", "%s.json" % spec_id), "r", encoding="utf-8") as h:
        return json.load(h)


def test_reference_run_is_reproducible_within_a_process():
    for spec in SPECS:
        a = flatten(run_spec(spec, "."))
        b = flatten(run_spec(spec, "."))
        assert a == b, spec["id"]


def test_reference_run_reproduces_the_committed_golden_manifest_bit_for_bit():
    for spec in SPECS:
        fresh = flatten(run_spec(spec, "."))
        golden = flatten(_golden(spec["id"]))
        assert set(fresh) == set(golden), spec["id"]
        for key in fresh:
            assert fresh[key] == golden[key], (spec["id"], key, fresh[key], golden[key])


def test_the_manifest_is_not_empty_enough_to_be_meaningless():
    total = sum(len(flatten(_golden(s["id"]))) for s in SPECS)
    assert total >= 200, total
    for spec in SPECS:
        assert len(flatten(_golden(spec["id"]))) >= 10, spec["id"]


def test_no_tolerance_is_wide_enough_to_tie_with_the_mutation():
    """Every relative budget must be strictly tighter than the mutation size.

    A tolerance equal to MUTATION_SIZE puts the comparator on a knife edge: the
    perturbation and the allowance are the same number, so whether the mutation
    is caught depends on the last bit of a floating point comparison and can
    change between machines. It did, once, on params.const.pvalue in ols-003.
    The invariant is asserted here so the next such tolerance is rejected in
    review rather than discovered by an intermittently red run.
    """
    seen = 0
    for spec in SPECS:
        tolerance = spec["comparison"]["tolerance"]
        budgets = [("default", float(tolerance["rtol"]))]
        for override in tolerance.get("key_overrides", []):
            budgets.append((override["pattern"], float(override["rtol"])))
        for source, rtol in budgets:
            assert rtol < MUTATION_SIZE, (
                "%s in %s allows rtol=%g, which is not strictly tighter than "
                "the mutation size %g" % (source, spec["id"], rtol, MUTATION_SIZE)
            )
            assert rtol * 10.0 <= MUTATION_SIZE, (
                "%s in %s allows rtol=%g, leaving less than a factor of ten "
                "of margin under the mutation size %g"
                % (source, spec["id"], rtol, MUTATION_SIZE)
            )
            seen += 1
    assert seen >= len(SPECS), seen


def test_every_single_compared_value_is_load_bearing():
    """Perturb one value at a time; the comparator must catch every one."""
    checked = 0
    for spec in SPECS:
        golden = flatten(_golden(spec["id"]))
        for key in golden:
            if excluded_reason(key, spec["comparison"]) is not None:
                continue
            mutated = dict(golden)
            base = golden[key]
            delta = abs(base) * MUTATION_SIZE if base != 0.0 else MUTATION_SIZE
            mutated[key] = base + delta
            records = compare_flat(mutated, golden, spec, "mutant", "golden")
            assert not is_green(records), (
                "mutating %s in %s by %g slipped past the comparator"
                % (key, spec["id"], delta)
            )
            failed = [r for r in records if r["status"] == "fail"]
            assert [r["key"] for r in failed] == [key], (spec["id"], key)
            checked += 1
    assert checked >= 200, checked


def test_identical_results_are_green():
    for spec in SPECS:
        golden = flatten(_golden(spec["id"]))
        records = compare_flat(dict(golden), golden, spec, "a", "b")
        assert is_green(records), spec["id"]
        assert summarise(records)["fail"] == 0


def test_a_missing_key_is_a_failure_and_never_a_skip():
    spec = SPECS[0]
    golden = flatten(_golden(spec["id"]))
    key = sorted(golden)[0]
    short = {k: v for k, v in golden.items() if k != key}
    records = compare_flat(short, golden, spec, "short", "golden")
    assert not is_green(records)
    missing = [r for r in records if r["status"] == "missing"]
    assert len(missing) == 1 and missing[0]["key"] == key


def test_an_extra_key_is_also_a_failure():
    spec = SPECS[0]
    golden = flatten(_golden(spec["id"]))
    extended = dict(golden)
    extended["scalars.invented_statistic"] = 1.0
    records = compare_flat(extended, golden, spec, "extended", "golden")
    assert not is_green(records)


def test_nan_only_matches_nan():
    spec = SPECS[0]
    golden = flatten(_golden(spec["id"]))
    key = sorted(golden)[0]
    poisoned = dict(golden)
    poisoned[key] = float("nan")
    assert not is_green(compare_flat(poisoned, golden, spec, "poisoned", "golden"))
    both = dict(golden)
    both[key] = float("nan")
    other = dict(golden)
    other[key] = float("nan")
    records = compare_flat(both, other, spec, "a", "b")
    assert is_green(records)


def test_tolerance_overrides_resolve_by_pattern():
    spec = [s for s in SPECS if s["id"] == "ols-001"][0]
    tol = spec["comparison"]["tolerance"]
    rtol_default, atol_default, source_default = tolerance_for("scalars.r2", tol)
    rtol_p, atol_p, source_p = tolerance_for("params.x1.pvalue", tol)
    assert source_default == "default"
    assert source_p == "params.*.pvalue"
    assert atol_p < atol_default
    assert rtol_p > rtol_default


def test_the_pvalue_override_still_fails_a_thousandth_perturbation():
    """The looser p-value budget must not become a licence to disagree."""
    spec = [s for s in SPECS if s["id"] == "ols-001"][0]
    golden = flatten(_golden("ols-001"))
    pkeys = [k for k in golden if k.endswith(".pvalue")]
    assert pkeys
    for key in pkeys:
        mutated = dict(golden)
        mutated[key] = golden[key] * (1.0 + MUTATION_SIZE)
        assert not is_green(compare_flat(mutated, golden, spec, "m", "g")), key


def test_the_pvalue_override_tolerates_realistic_floating_point_noise():
    spec = [s for s in SPECS if s["id"] == "ols-001"][0]
    golden = flatten(_golden("ols-001"))
    jittered = {
        k: (v * (1.0 + 1e-11) if k.endswith(".pvalue") else v) for k, v in golden.items()
    }
    assert is_green(compare_flat(jittered, golden, spec, "jitter", "golden"))


def test_tiny_pvalues_keep_their_resolution():
    """With a fixed absolute floor this test would pass vacuously."""
    spec = [s for s in SPECS if s["id"] == "ols-001"][0]
    golden = flatten(_golden("ols-001"))
    key = [k for k in golden if k.endswith(".pvalue")][0]
    left = dict(golden)
    right = dict(golden)
    left[key] = 1e-40
    right[key] = 2e-40  # a factor of two apart, and both far below 1e-12
    assert not is_green(compare_flat(left, right, spec, "left", "right"))


def test_exclusions_apply_only_to_declared_patterns():
    spec = json.loads(json.dumps(SPECS[0]))
    spec["comparison"]["exclude"] = [
        {"key": "scalars.*", "reason": "declared purely to exercise the matcher in tests"}
    ]
    golden = flatten(_golden(spec["id"]))
    mutated = {k: (v * 2.0 if k.startswith("scalars.") else v) for k, v in golden.items()}
    records = compare_flat(mutated, golden, spec, "m", "g")
    assert is_green(records)
    assert summarise(records)["excluded"] > 0
    # A parameter is still compared, so the suite has not gone blind.
    mutated2 = dict(golden)
    pkey = [k for k in golden if k.startswith("params.")][0]
    mutated2[pkey] = golden[pkey] + 1.0
    assert not is_green(compare_flat(mutated2, golden, spec, "m", "g"))


def test_adf_specs_expose_a_statistic_and_no_pvalue():
    for spec in [s for s in SPECS if s["family"] == "adf"]:
        golden = flatten(_golden(spec["id"]))
        assert any(k == "scalars.adf_stat" for k in golden), spec["id"]
        assert not [k for k in golden if k.endswith(".pvalue")], spec["id"]


def test_golden_values_are_finite():
    for spec in SPECS:
        for key, value in flatten(_golden(spec["id"])).items():
            assert math.isfinite(value), (spec["id"], key)
