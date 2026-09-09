#!/usr/bin/env python3
"""One time patcher, deleted by the workflow that runs it.

The sandbox that authored this repository has no git and no route to push a
large file safely, so two edits to files of fourteen and eight kilobytes are
applied here instead of retyping those files from memory. Every replacement
refuses to proceed unless it matches exactly once, so a partial or duplicated
application fails loudly rather than producing a plausible looking tree.
"""

import io
import sys

MAKE_SPECS_OLD = '''            "pattern": "params.*.pvalue",
            "rtol": 1e-3,
            "atol": 1e-300,
            "rationale": (
                "The tail integral amplifies the already amplified coefficient "
                "error on an ill conditioned design, so the p-value budget is "
                "widened in proportion and the observed divergence is recorded "
                "in docs/divergences.md rather than hidden."
            ),
'''

MAKE_SPECS_NEW = '''            "pattern": "params.*.pvalue",
            "rtol": 1e-4,
            "atol": 1e-300,
            "rationale": (
                "The tail integral amplifies the already amplified coefficient "
                "error on an ill conditioned design, so the p-value budget is "
                "widened in proportion. It stops at 1e-4 because a budget of "
                "1e-3 would exactly equal the mutation size and let a real one "
                "part in a thousand change tie with the tolerance, and any "
                "divergence beyond this is recorded in docs/divergences.md "
                "with the measured number rather than absorbed silently."
            ),
'''

CONFORMANCE_ANCHOR = '''def test_every_single_compared_value_is_load_bearing():
'''

CONFORMANCE_NEW = '''def test_no_tolerance_is_wide_enough_to_tie_with_the_mutation():
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
'''

WORKFLOW_ANCHOR = '''      - name: Unit and mutation tests
'''

WORKFLOW_NEW = '''      - name: Specifications must match their generator
        run: |
          python tools/make_specs.py
          git diff --exit-code specs

      - name: Unit and mutation tests
'''

EDITS = [
    ("tools/make_specs.py", MAKE_SPECS_OLD, MAKE_SPECS_NEW),
    ("tests/test_conformance.py", CONFORMANCE_ANCHOR, CONFORMANCE_NEW),
    (".github/workflows/conformance.yml", WORKFLOW_ANCHOR, WORKFLOW_NEW),
]


def main() -> int:
    for path, old, new in EDITS:
        with io.open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        hits = source.count(old)
        if hits != 1:
            print("REFUSING %s: anchor found %d times, expected exactly 1" % (path, hits))
            return 1
        if new in source:
            print("REFUSING %s: replacement text is already present" % path)
            return 1
        with io.open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(source.replace(old, new))
        print("patched %s" % path)

    with io.open("tools/make_specs.py", "r", encoding="utf-8") as handle:
        text = handle.read()
    if '"rtol": 1e-3,' in text:
        print("REFUSING: make_specs.py still carries a 1e-3 budget")
        return 1
    with io.open("tests/test_conformance.py", "r", encoding="utf-8") as handle:
        text = handle.read()
    if text.count("def test_no_tolerance_is_wide_enough_to_tie_with_the_mutation") != 1:
        print("REFUSING: the invariant test did not land exactly once")
        return 1
    print("all three edits applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
