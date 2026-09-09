# How the reference kernel is verified

The reference kernel imports nothing but numpy. That is a deliberate cost: it
means no library can be blamed, and it means the kernel has to be checked
against something other than itself.

## Layers of checking

### 1. Closed forms

`tests/test_distributions.py` checks the Student t cumulative distribution
against exact closed forms where they exist.

- df = 1, the Cauchy case: `F(t) = 0.5 + arctan(t) / pi`.
- df = 2: `F(t) = 0.5 + t / (2 sqrt(2 + t^2))`.
- df = 4: with `x = t / sqrt(1 + t^2 / 4)`, `F(t) = 0.5 + 0.375 x - x^3 / 32`.

The df = 4 identity is where a real bug was caught. The first version of the
test carried `x^3 / 8` and the suite went red. Re deriving from the density
`f(t) = (3/8) (1 + t^2/4)^{-5/2}` under the substitution `t = 2 tan(theta)`
gives `x^3 / 32`, and the check that `x -> 2` implies `F -> 1` settles it. The
kernel was right and the test was wrong, which is the only reason to write
independent closed forms in the first place.

Symmetry, monotonicity and the tails are checked separately, along with the
normal distribution against published values.

### 2. Independent linear algebra

Each estimator is checked against a different route to the same quantity.

- OLS coefficients against `numpy.linalg.lstsq` and against the pseudo inverse.
- HC1 and HC3 sandwich covariances against a hand assembled meat matrix and,
  for HC3, against the leverage values from the hat matrix.
- R squared against the squared correlation of fitted values with the response.
- Panel fixed effects, computed by the within transform, against least squares
  dummy variable estimation with an explicit dummy for every entity, including
  the degrees of freedom.
- 2SLS against the textbook two stage computation done by hand, in both the
  exactly identified and the over identified case.
- The ADF statistic against an explicitly constructed regression whose t ratio
  is computed directly.

### 3. Traps that are asserted, not avoided

- The 2SLS residual must use observed regressors. A test computes both the
  correct and the naive variance, asserts they differ materially, and asserts
  the kernel matches the correct one.
- A rank deficient design must raise `RankDeficientError` rather than return a
  number produced by a silent pseudo inverse.
- Tiny p-values must keep their resolution. A test shows that 1e-40 against
  2e-40 fails under the p-value tolerance override, so the override cannot be
  mistaken for a licence to ignore small numbers.

### 4. The mutation gate

`tests/test_conformance.py` walks every compared value in the golden manifest,
perturbs it by one part in a thousand, and requires that exactly that key
fails. It is the test that makes every other green result mean something: if
the comparator is ever loosened, this file goes red before anything else does.

### 5. Fixture and contract integrity

- Fixtures are regenerated into a temporary directory and their sha256 sums
  compared against the committed manifest, so committed data cannot drift away
  from the code that claims to produce it.
- The spec validator is tested against deliberately broken specs: a thin
  tolerance rationale, a missing fixture, a model kind that contradicts its
  family, an exclusion with no reason.

### 6. The report cannot overstate the run

`tests/test_report.py` requires that every specification appears in the
generated page, that a specification from an unrecognised family raises instead
of being silently dropped, and that the status word follows the comparator
rather than the author.

## What this does not prove

Agreement means the computation is reproducible. It does not mean a model is
appropriate for a real question, and no fixture here is real data.
