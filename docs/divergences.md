# Known divergences and open reproducibility gaps

This file is the honest half of the project. A conformance suite that only
publishes agreement is marketing. Anything below is either a real difference
between implementations, or a limit on how reproducible the evidence currently
is.

Entries are added when CI observes them, not when they are guessed. If an entry
says "observed", a run produced it. If it says "expected", it has not yet been
confirmed by a run and must not be cited as a finding.

## 1. The ADF trend regressor is parameterised differently

**Status: by design, documented, not a failure.**

The augmented Dickey Fuller regression in this repository runs over
observations `t = lags + 2 ... T`, and the deterministic trend regressor takes
the **original** time index of each retained observation. `statsmodels.tsa`
and `urca::ur.df` restart the trend at 1 within the effective sample.

Consequence: the constant and trend coefficients differ between our
parameterisation and theirs. The coefficient and t ratio on the lagged level,
which is the entire point of the test, do **not** differ, because a shifted
trend spans the same column space once a constant is present.

Both adapters therefore cross check their own ADF statistic against
`adfuller(..., autolag=None)` and `ur.df(...)` at a tolerance of 1e-8 and stop
if they disagree, while the compared envelope carries our parameterisation.

## 2. No p-value is reported for the ADF test

**Status: by design.**

Under the null of a unit root the statistic does not follow a Student t
distribution. Published p-values come from MacKinnon response surface
interpolations whose tables and interpolation rules differ between packages, so
a cross implementation comparison of ADF p-values would measure the tables
rather than the estimation. The parameter tables for ADF specifications carry
`estimate`, `se` and `tstat`, and no `pvalue` at all.

## 3. Inference uses the t distribution even under robust covariance

**Status: convention, fixed in the contract.**

With HC1 or HC3 covariance there is no exact finite sample distribution. We use
Student t with `nobs - rank(X)` degrees of freedom, which is what
`sandwich::coeftest` with a fitted `lm` and statsmodels with `use_t=True`
produce. A suite that compared t based standard errors against normal based
p-values would report a false divergence.

## 4. R package versions are not pinned

**Status: open gap.**

Python dependencies are pinned in `requirements.txt`. The R packages
`jsonlite`, `sandwich`, `AER`, `plm` and `urca` are installed from the public
package manager at run time, so a run reproduces the *method* but not
necessarily the exact binaries. Mitigation: the R adapter writes
`environment.json` next to its envelopes with the resolved versions, and the
observatory renders them, so any past run states what produced it.

Planned fix: a date pinned package manager snapshot, once one has been
confirmed to resolve all five packages on the runner image.

## 5. The 2SLS residual is computed from observed regressors

**Status: convention, fixed in the contract, guarded by a test.**

The 2SLS residual is `y - X b`, never `y - Xhat b`. Using the first stage
fitted values gives a different residual sum of squares and therefore a
different `sigma2` and every standard error derived from it. The two differ
materially. The direction of the difference is not universal, so the test
asserts a material relative gap and that our implementation matches the
correct definition, rather than asserting a sign.

## 6. Degrees of freedom in panel fixed effects

**Status: convention, fixed in the contract.**

`df_resid = nobs - rank(X) - n_entities`, which matches least squares dummy
variable estimation. Packages that report the within estimator without
charging for the absorbed entity means will disagree on every standard error,
and that disagreement is a definitional one.

## 7. Tolerances on ill conditioned designs

**Status: provisional.**

`ols-003` uses a near collinear design with a relative tolerance of 1e-6
instead of 1e-9, because different factorisation orders diverge more when the
design matrix is ill conditioned. The exact observed divergence across
implementations will be recorded here after CI has run, and the tolerance will
be tightened to the smallest value the observed data supports.

## 8. The reference path was CPU dependent by 1 ULP, and no longer is

**Status: observed, closed, guarded.**

A numpy backed reference run of `adf-001` returned
`params.const.estimate = -0.010046320349888796` on an Intel Xeon Platinum 8573C
and on an AMD EPYC 9V45, and `-0.010046320349888798` on an AMD EPYC 9V74 and an
AMD EPYC 7763. One unit in the last place, a relative difference of
1.7267252243215682e-16. Each machine was perfectly repeatable, 300 of 300
samples per arm, so this was never load dependence, and setting the thread
count to one did not close it: two arms pinned to a single thread on different
CPU families still disagreed. Instrumented by stage, the upper triangular
factor `R` was identical everywhere at `87c90d6edb2b0cb6` while `Q` split into
`2e5efdaa7e7b543e` and `d0ba89fc0249ace6`, which places the divergence inside
LAPACK kernel selection rather than in the model.

The fix was to stop dispatching. `econospec/linalg.py` performs every reduction
on the reference path with `math.fsum` and correctly rounded scalar operations
in a fixed order, so the result is a function of the inputs on any IEEE 754
platform. The golden manifest was regenerated by CI against the new kernel, the
thirty sample hunt is 30 of 30 green on head `f83a91d`, and
`tests/test_linalg.py` fails the build if any `np.linalg`, `np.dot`,
`np.einsum` or `@` returns to `econospec/linalg.py` or `econospec/estimators.py`.
Full write up with run links in `docs/determinism.md`.

Remaining boundary, not measured: `econospec/distributions.py` still calls
`math.lgamma`, `math.erfc`, `math.exp` and `math.log`, which are platform libm
and are not correctly rounded by any standard. No divergence has been observed
there. It is listed so that nobody reads the entry above as a stronger
guarantee than it is.
