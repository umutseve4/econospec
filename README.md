<div align="center">

# EconoSpec

### Econometric Specification and Verification Lab

**The same model, estimated by three independent implementations, compared value by value inside tolerances that each carry a written justification.**

[![conformance](https://img.shields.io/github/actions/workflow/status/umutseve4/econospec/conformance.yml?branch=main&label=conformance&style=for-the-badge&color=e11d48)](https://github.com/umutseve4/econospec/actions/workflows/conformance.yml)
[![observatory](https://img.shields.io/badge/observatory-live-0ea5e9?style=for-the-badge)](https://umutseve4.github.io/econospec/)
[![specifications](https://img.shields.io/badge/specifications-12-8b5cf6?style=for-the-badge)](specs/)
[![implementations](https://img.shields.io/badge/implementations-3-f59e0b?style=for-the-badge)](adapters/)
[![license](https://img.shields.io/badge/license-MIT-22c55e?style=for-the-badge)](LICENSE)

</div>

> **Independence notice.** This is an independent student project. It is not an
> official product of Bursa Uludag University or its Department of Econometrics,
> and it does not speak for them. See [POSITIONING.md](POSITIONING.md).

---

## The claim being tested

An econometric result that only one program can reproduce is a result about
that program. EconoSpec turns that sentence into something a machine can check.

Every specification is estimated three times by code that shares no
implementation:

| Implementation | Stack | What it is |
| --- | --- | --- |
| `reference-numpy` | numpy only | A from scratch kernel. Linear algebra, sandwich covariances, the t distribution by continued fraction. No statistics library. |
| `python-statsmodels` | statsmodels, linearmodels, scipy | The mainstream Python path. Does not import the reference kernel. |
| `r-base-sandwich-aer-plm` | lm, sandwich, AER, plm, urca | The mainstream R path. Never translated from the Python code. |

The three must agree with a committed golden manifest, key by key, or the run
is red.

## What is covered

| Family | Nominal | Assumption violated | Edge |
| --- | --- | --- | --- |
| Ordinary least squares | classical covariance | heteroskedasticity, HC1 and HC3 | near collinear design |
| Instrumental variables, 2SLS | strong instruments | weak instruments | exactly identified |
| Panel fixed effects | balanced panel | confounded, pooled OLS is biased | unbalanced panel |
| Augmented Dickey Fuller | stationary series | random walk | trend stationary |

Twelve specifications, twelve deterministic fixtures, one checksum manifest,
243 individually compared values.

## Why this is not another regression example

- **The fixtures are data, not code.** They are generated once in Python and
  committed as CSV, so R never draws a random number. Cross language random
  number generation is removed from the question entirely.
- **Conventions are written down before they are tested.** Residual degrees of
  freedom, whether the 2SLS residual uses observed regressors or first stage
  fitted values, whether inference uses t or normal, which R squared is
  centred. Most cross language disagreement is definitional, not numerical.
- **Every tolerance has a reason in prose.** The spec validator rejects a
  rationale shorter than eight words.
- **Every compared value is load bearing.** A mutation test perturbs each of
  the 243 golden values by one part in a thousand and requires exactly that key
  to fail. Loosening the comparator breaks that test first.
- **A skipped adapter is a failure.** Keys are compared on the union of both
  sides, so a missing key can never be silently counted as agreement.
- **No p-value is published for the ADF test.** Under a unit root the statistic
  is not Student t, and the tables are interpolations that differ by package.
  Reporting one would be a number without a meaning.

## Run it

```bash
git clone https://github.com/umutseve4/econospec.git
cd econospec
pip install -r requirements.txt

python tools/gen_fixtures.py --check     # fixtures match their generators
pytest -q tests                          # unit and mutation tests
python tools/run_reference.py            # reference kernel
python adapters/python/run_statsmodels.py --out results/python-statsmodels
Rscript adapters/r/run_r.R --out results/r
python tools/compare_results.py --require reference python-statsmodels r
python tools/build_report.py             # site/index.html
```

The reference kernel alone needs nothing but numpy.

## Layout

```
econospec/     reference kernel: estimators, distributions, comparator
specs/         the contract: model, fixture, tolerances, rationale
fixtures/      committed CSV data, sha256 manifest, golden results
adapters/      python and r implementations, independent by construction
tools/         generate, run, compare, report
tests/         including the mutation gate over every compared value
site/          the Conformance Observatory, generated
```

## Honest limits

Read [docs/divergences.md](docs/divergences.md) before citing anything here. It
records where implementations genuinely differ and why, including the ADF trend
parameterisation and the fact that R package versions are resolved at run time
rather than pinned. Findings are educational, not a statistical service, and
none of this is statistical or financial advice.

The Conformance Observatory is regenerated on every run and uploaded as the
`conformance-evidence` artifact, which is where you can read it today. GitHub
Pages is not enabled for this repository, so there is no published address yet.
That expectation is written down in `.github/pages-policy` as
`PAGES_REQUIRED=false`, and two jobs hold this page to it. One compares the
README against the policy on every push and every pull request, with no network
access, and fails if the page badges or links an address the policy does not
promise. The other asks the Pages API on each push to main: it skips publishing
when the answer is 404 and the policy expects that, and it fails the run when
the policy promises a site the API cannot find, so a deployment that quietly
disappears can never look like a pass.

---

<div align="center">
<sub>Independent student project inspired by econometrics education at Bursa Uludag University. MIT licensed.</sub>
</div>
