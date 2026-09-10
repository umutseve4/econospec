# Determinism: why the reference path does its own arithmetic

This file exists because a claim in this repository was not true, and because a
later version of this file was also not true. Both were corrected by
measurement rather than by relaxing the claim.

## The claim

`tests/test_conformance.py::test_reference_run_reproduces_the_committed_golden_manifest_bit_for_bit`
asserts exact float equality between a fresh reference run and the committed
golden manifest:

```python
assert fresh[key] == golden[key]
```

No tolerance. That is deliberate. The manifest is the anchor every other
implementation is compared against, so if the anchor can move, the comparison
means nothing.

## What was measured

Commit `d05ec173b5c0be0a52b1cbdfebe2340dff9615d8` was red once and green once at
`pytest -q tests` with an unchanged tree. Issue #3 refused to close that with a
rerun. `.github/workflows/determinism.yml` took thirty samples on ten separately
allocated runners, run [34469432895](https://github.com/umutseve4/econospec/actions/runs/34469432895):

| Result | Count |
| --- | --- |
| green | 28 of 30 |
| red | 2 of 30 |

Both failures named the same pair of values:

```
AssertionError: ('adf-001', 'params.const.estimate',
                 -0.010046320349888796, -0.010046320349888798)
1 failed, 78 passed
```

The distance between those two floats is **one unit in the last place**, not
two. Their relative difference is **1.7267252243215682e-16**, which is
`1 * 2**-53 * 1.55...`, and not the 2.8e-14 an earlier version of this file
claimed. Both failing samples ran on **AMD EPYC** hardware, 9V74 and 9V45, not
on the Intel Xeon Platinum 8370C an earlier version of this file named.

## Why thread pinning was not the answer

The first fix set `OMP_NUM_THREADS` and its four siblings to `1` at import
time, on the theory that a multithreaded reduction splits differently under
load. `.github/workflows/divergence.yml` tested that theory directly by taking
300 samples per arm with pinning on and off, and it fails on both counts.

First, nothing is load dependent. Every arm reported `distinct 1` for every
recorded stage across all 300 samples. A given machine returns the same bits
every single time. What changes is the machine.

Second, pinning does not close the gap, because **two jobs pinned to a single
thread on different CPU families still disagree with each other**. Grouped by
processor rather than by thread count, the raw numpy path splits cleanly:

| CPU | stage `4_q` fingerprint | `adf-001` const estimate |
| --- | --- | --- |
| AMD EPYC 9V74 80-Core | `2e5efdaa7e7b543e` | -0.010046320349888798 |
| AMD EPYC 7763 64-Core | `2e5efdaa7e7b543e` | -0.010046320349888798 |
| Intel Xeon Platinum 8573C | `d0ba89fc0249ace6` | -0.010046320349888796 |
| AMD EPYC 9V45 96-Core | `d0ba89fc0249ace6` | -0.010046320349888796 |

Stage `5_r`, the upper triangular factor, is `87c90d6edb2b0cb6` on all four
families and under both thread settings. So `R` agrees while `Q` does not: two
LAPACK entry points inside the same factorisation disagree about the same
inputs, and no environment variable reaches that. A dispatched kernel is
selected from the instruction set of the host, and the host is not ours to fix.

## What the defect cost, measured

Pull request #5 changed exactly one file, a YAML workflow, and no Python at
all. Its conformance run [34481628602](https://github.com/umutseve4/econospec/actions/runs/34481628602)
failed at step 8, `pytest -q tests`, which skipped step 9 and failed step 14 as
a consequence. Twelve minutes later the forensics workflow ran the identical
tree on `main` and reported `77 passed in 0.58s`, `PYTEST_EXIT=0`. Same source,
opposite verdict. An unrelated change was blocked by a float in an ADF constant.
That is the argument for fixing the arithmetic rather than rerunning the job.

## The fix

`econospec/linalg.py` removes every dispatched BLAS reduction from the
reference path. `math.fsum` is exactly rounded, and multiplication,
subtraction, division and `math.sqrt` are correctly rounded under IEEE 754, so
a composition of them in a fixed order is a pure function of its inputs on any
conforming platform. Householder QR, the cross products, the residual sums, the
triangular solves and the group reductions in `panel_fe` are all written that
way, and `econospec/estimators.py` routes through them.

This is a property of the reduction, not of the algorithm. Householder QR is
legitimately order dependent, so there is no permutation invariance test on the
estimator, and there should not be one. What is asserted is that the same
inputs in the same order produce the same bits everywhere.

`tests/test_linalg.py` is the standing guard. It checks the primitives against
exact rational arithmetic with `fractions.Fraction`, parses both kernel files
with `ast` to prove no `np.linalg`, `np.dot`, `np.einsum` or `@` reaches the
reference path, checks that the guard itself fires when given a violation, and
re-fits under a monkeypatched numpy with eight `np.linalg` entry points removed.

## Evidence that it holds

The golden manifest was regenerated by CI, not by hand, through
`.github/workflows/regolden.yml`, which runs `tools/run_reference.py --golden`
and commits only `fixtures/golden`. Twelve files changed, 126 lines added and
126 removed.

Before that regeneration, on head `09e05070`, the hunt still failed, but it
failed in the only way that matters: every failing sample produced the same
fresh value, `-0.010046320349888822`, on AMD EPYC 9V74 and on AMD EPYC 7763
alike, against a stale golden of `-0.010046320349888798`. The disagreement
between machines was already gone. Only the anchor was old.

After regeneration, on head `f83a91d3e11d3a15eff4e19cf6fd9401a54fd550`:

| Check | Result |
| --- | --- |
| determinism, [34483191740](https://github.com/umutseve4/econospec/actions/runs/34483191740) | 30 of 30 green |
| conformance, [34483191725](https://github.com/umutseve4/econospec/actions/runs/34483191725) | green |
| divergence, [34483191705](https://github.com/umutseve4/econospec/actions/runs/34483191705) | green |

The suite grew from 78 to 92 passing tests over the same work, because the
guard tests came with the fix.

## What was not done

- **No tolerance was widened.** The comparison is still exact.
- **No test was skipped, marked flaky or retried.** A retry turns a measurement
  into a coin flip that is allowed to be tossed twice.
- **No claim was kept because it was already written down.** Four statements in
  the previous version of this file were falsified by this branch's own CI and
  are corrected above, with the run that falsified each one.

## Honest residue

Two boundaries are known and not closed.

`econospec/distributions.py` calls `math.lgamma`, `math.erfc`, `math.exp` and
`math.log`, which are platform libm and which glibc may dispatch by CPU feature
through IFUNC. Those functions are not correctly rounded by any standard, so a
runner with a different libm could in principle move a p-value. This was never
the observed divergence, and it has not been measured. It is stated here as an
open boundary rather than left for a reader to discover.

`.github/workflows/divergence.yml` instruments the raw numpy path on purpose,
not `econospec.estimators`. Its arms therefore still print the two CPU
dependent values in the table above, and that is the intended behaviour: it is
the experiment that proved the diagnosis, so it must keep measuring the thing
that was broken. It is not a post-fix check, and it must not be cited as one.
