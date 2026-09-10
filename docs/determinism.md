# Determinism: why the linear algebra backend is pinned to one thread

This file exists because a claim in this repository was not true and was made
true rather than quietly relaxed.

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

## The measurement

Commit `d05ec173b5c0be0a52b1cbdfebe2340dff9615d8` was red once and green once at
`pytest -q tests` with an unchanged tree, which is intermittency rather than a
failure. Issue #3 refused to close that with a rerun. `.github/workflows/determinism.yml`
took thirty samples on ten separately allocated runners:

| Result | Count |
| --- | --- |
| green | 28 of 30 |
| red | 2 of 30 |

Both failures named the same value:

```
AssertionError: ('adf-001', 'params.const.estimate',
                 -0.010046320349888796, -0.010046320349888798)
1 failed, 76 passed in 0.36s
```

The two numbers differ by two units in the last place, a relative difference of
about 2.8e-14 of the value itself and of the order of 1e-16 in relative float
terms. Both failing samples ran on an Intel Xeon Platinum 8370C with four cores,
`avx512f` present, numpy 2.1.3 against scipy-openblas 0.3.27. Both failed on the
third of three consecutive runs in the same job, which points at machine load
rather than at the model: a multithreaded OpenBLAS chooses how to split a
reduction at call time, and a different split sums the same numbers in a
different order.

Run links: [34469432895](https://github.com/umutseve4/econospec/actions/runs/34469432895),
and the original episode in [34463609778](https://github.com/umutseve4/econospec/actions/runs/34463609778)
against [34468393536](https://github.com/umutseve4/econospec/actions/runs/34468393536).

## The fix, and what it is not

Importing `econospec` sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS` and `VECLIB_MAXIMUM_THREADS` to `1`
before any submodule imports numpy. A single threaded reduction has one
summation order, so the result is a function of the inputs alone.

What was not done, and why:

- **No tolerance was widened.** The comparison is still exact. Widening it would
  have hidden the next real regression behind the same allowance.
- **No test was skipped, marked flaky or retried.** A retry turns a measurement
  into a coin flip that is allowed to be tossed twice.
- **The pinning is not hidden in CI configuration.** It lives in the package, so
  a reader who runs `pytest` on a laptop gets the same guarantee as the runner.
  `tests/test_determinism.py` fails with a named message if the pinning ever
  stops binding, for instance if numpy is imported before `econospec`.

The variables are set with `setdefault`, so exporting `OMP_NUM_THREADS=8`
remains possible. In that case the suite does not silently become intermittent:
it fails and says why.

## Honest residue

Single threading fixes the reduction order on one machine and across machines
with the same kernel selection. It does not make float arithmetic portable in
principle. A future runner with a different instruction set could still pick a
different kernel for the same call. If that happens, the failure is now
reproducible and named, and the answer will be a measured ulp bound written
here with the number beside it, not a tolerance chosen to make a run green.
