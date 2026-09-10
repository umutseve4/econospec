#!/usr/bin/env python3
"""Author the twelve specifications from a single source of truth.

Specs are data, but they are also the documentation a reader trusts, so they
are generated here rather than hand copied twelve times. Checksums are read
from the fixture manifest, which means a fixture change without a spec
regeneration is caught by tests/test_specs.py.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TIGHT = {
    "rtol": 1e-9,
    "atol": 1e-12,
    "rationale": (
        "Both sides read identical CSV bytes and solve the same normal "
        "equations in IEEE 754 double precision, so the only admissible "
        "difference is accumulated rounding from a different factorisation "
        "order. That is bounded well below 1e-9 relative on a well conditioned "
        "design; anything larger is a definitional disagreement, not noise."
    ),
    "key_overrides": [
        {
            "pattern": "params.*.pvalue",
            "rtol": 1e-5,
            "atol": 1e-300,
            "rationale": (
                "P-values span thirty orders of magnitude, so a fixed absolute "
                "floor would make small ones untestable. A purely relative "
                "budget of 1e-5 absorbs the amplification of coefficient "
                "rounding through the tail integral while still failing a one "
                "part in a thousand perturbation. The 1e-300 floor only "
                "handles underflow to zero."
            ),
        }
    ],
}

LOOSE = {
    "rtol": 1e-6,
    "atol": 1e-12,
    "rationale": (
        "This design is deliberately ill conditioned, so the relative error of "
        "the solved coefficients scales with the condition number. A 1e-6 "
        "budget reflects that amplification and is still three orders of "
        "magnitude tighter than any economically meaningful difference."
    ),
    "key_overrides": [
        {
            "pattern": "params.*.pvalue",
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
        }
    ],
}


def spec(
    sid,
    family,
    title,
    case_type,
    narrative,
    generator,
    fixture,
    model,
    tolerance,
    exclude=None,
):
    return {
        "id": sid,
        "family": family,
        "title": title,
        "case_type": case_type,
        "narrative": narrative,
        "fixture": {"generator": generator, "path": "fixtures/%s" % fixture, "sha256": ""},
        "model": model,
        "comparison": {"tolerance": tolerance, "exclude": exclude or []},
    }


SPECS = [
    spec(
        "ols-001",
        "ols",
        "OLS with classical standard errors",
        "nominal",
        "The baseline case. A homoskedastic linear model with two regressors "
        "and an intercept, estimated by ordinary least squares with the "
        "classical covariance sigma squared times the inverse of X prime X. "
        "If this case ever disagrees across implementations, the disagreement "
        "is about the residual degrees of freedom or the reference "
        "distribution used for inference, not about the estimator itself, "
        "because the coefficient vector is uniquely determined by the design.",
        "ols_nominal",
        "ols-001.csv",
        {
            "kind": "ols",
            "response": "y",
            "regressors": ["x1", "x2"],
            "intercept": True,
            "vcov": "nonrobust",
        },
        TIGHT,
    ),
    spec(
        "ols-002",
        "ols",
        "OLS with HC1 heteroskedasticity consistent errors",
        "violation",
        "The error variance grows with the absolute value of x1, so the "
        "classical standard errors are inconsistent while the coefficients "
        "remain unbiased. This case pins down the HC1 convention: the "
        "sandwich meat uses squared residuals and the correction factor is n "
        "over n minus k. R applies that factor inside sandwich::vcovHC while "
        "statsmodels applies it inside cov_type HC1, and inference stays on "
        "the t distribution with n minus k degrees of freedom on both sides.",
        "ols_heteroskedastic",
        "ols-002.csv",
        {
            "kind": "ols",
            "response": "y",
            "regressors": ["x1", "x2"],
            "intercept": True,
            "vcov": "HC1",
        },
        TIGHT,
    ),
    spec(
        "ols-003",
        "ols",
        "OLS with HC3 errors on a near collinear design",
        "edge",
        "x2 is x1 plus a perturbation with standard deviation 0.02, which "
        "makes X prime X nearly singular and the individual coefficients "
        "badly identified even though their sum is not. HC3 divides the "
        "squared residual by one minus the leverage squared, and leverage is "
        "exactly where an ill conditioned design bites hardest, so this case "
        "tests the leverage computation and the numerical stability of the "
        "solver at the same time.",
        "ols_near_collinear",
        "ols-003.csv",
        {
            "kind": "ols",
            "response": "y",
            "regressors": ["x1", "x2"],
            "intercept": True,
            "vcov": "HC3",
        },
        LOOSE,
    ),
    spec(
        "iv-001",
        "iv2sls",
        "Two stage least squares with strong overidentifying instruments",
        "nominal",
        "One endogenous regressor, one included exogenous control and two "
        "strong excluded instruments. The trap this case guards is the most "
        "common 2SLS implementation error: forming the residual from the "
        "first stage fitted values instead of the observed regressors, which "
        "silently shrinks every standard error. The overidentification degree "
        "of freedom must be one, and the first stage exclusion F must be "
        "large.",
        "iv_strong",
        "iv-001.csv",
        {
            "kind": "iv2sls",
            "response": "y",
            "exog": ["w"],
            "endog": ["d"],
            "instruments": ["z1", "z2"],
            "intercept": True,
        },
        TIGHT,
    ),
    spec(
        "iv-002",
        "iv2sls",
        "Two stage least squares with weak instruments",
        "violation",
        "Instrument relevance is deliberately near zero, so the first stage F "
        "falls far below the conventional threshold of ten and the 2SLS "
        "estimator is badly behaved. The estimate is not expected to be near "
        "the true parameter and the suite does not pretend otherwise: what is "
        "asserted is that both implementations report the same badly behaved "
        "number and the same diagnostic, so a user can see the weakness "
        "rather than inherit it silently.",
        "iv_weak",
        "iv-002.csv",
        {
            "kind": "iv2sls",
            "response": "y",
            "exog": ["w"],
            "endog": ["d"],
            "instruments": ["z1", "z2"],
            "intercept": True,
        },
        TIGHT,
    ),
    spec(
        "iv-003",
        "iv2sls",
        "Just identified two stage least squares",
        "edge",
        "One excluded instrument for one endogenous regressor. The projection "
        "step is exactly determined, the overidentification degree of freedom "
        "is zero and any Sargan style test is undefined. This is the boundary "
        "where implementations tend to differ: some report a degenerate test "
        "statistic rather than declining to compute one, so the spec asserts "
        "the degree of freedom explicitly.",
        "iv_just_identified",
        "iv-003.csv",
        {
            "kind": "iv2sls",
            "response": "y",
            "exog": ["w"],
            "endog": ["d"],
            "instruments": ["z1"],
            "intercept": True,
        },
        TIGHT,
    ),
    spec(
        "fe-001",
        "panel_fe",
        "One way fixed effects on a balanced panel",
        "nominal",
        "Sixty entities observed for eight periods each, with entity effects "
        "drawn independently of the regressors. The within estimator is "
        "algebraically identical to least squares with sixty dummy variables, "
        "and the only thing that can differ between implementations is the "
        "residual degrees of freedom, which must be n minus the number of "
        "entities minus the number of regressors. Getting that wrong inflates "
        "every t statistic.",
        "panel_balanced",
        "fe-001.csv",
        {
            "kind": "panel_fe",
            "response": "y",
            "regressors": ["x1", "x2"],
            "entity": "entity",
        },
        TIGHT,
    ),
    spec(
        "fe-002",
        "panel_fe",
        "Fixed effects where entity effects are correlated with a regressor",
        "violation",
        "The entity effect enters x1 with a coefficient of 0.8, so pooled "
        "ordinary least squares is biased upward while the within estimator "
        "remains consistent. This case exists to make the omitted variable "
        "bias visible as a number rather than as a claim: the report shows "
        "the within estimate next to the pooled estimate on the same fixture, "
        "and both implementations must agree on both.",
        "panel_confounded",
        "fe-002.csv",
        {
            "kind": "panel_fe",
            "response": "y",
            "regressors": ["x1", "x2"],
            "entity": "entity",
        },
        TIGHT,
    ),
    spec(
        "fe-003",
        "panel_fe",
        "One way fixed effects on an unbalanced panel",
        "edge",
        "Entity lengths vary between three and twelve periods, so the entity "
        "means used by the within transformation have different denominators. "
        "Implementations that demean with a common divisor, or that silently "
        "drop short entities, will disagree here. The number of entities and "
        "the total number of observations are both asserted so a silent drop "
        "cannot pass as agreement.",
        "panel_unbalanced",
        "fe-003.csv",
        {
            "kind": "panel_fe",
            "response": "y",
            "regressors": ["x1", "x2"],
            "entity": "entity",
        },
        TIGHT,
    ),
    spec(
        "adf-001",
        "adf",
        "Augmented Dickey Fuller regression on a stationary series",
        "nominal",
        "A stationary first order autoregression with coefficient 0.55 and an "
        "intercept in the test regression. The augmented Dickey Fuller "
        "statistic is the t ratio on the lagged level, and the whole point of "
        "specifying the regression explicitly, rather than calling a library "
        "routine, is that implementations disagree about how many "
        "observations to keep and where the trend term starts. Here the "
        "sample runs from observation lags plus two to the end, which leaves "
        "T minus lags minus one usable rows on both sides.",
        "series_stationary",
        "adf-001.csv",
        {"kind": "adf", "series": "y", "lags": 2, "trend": "c"},
        TIGHT,
    ),
    spec(
        "adf-002",
        "adf",
        "Augmented Dickey Fuller regression on a random walk",
        "violation",
        "A pure random walk, so a unit root is present by construction and the "
        "coefficient on the lagged level should sit near zero. The statistic "
        "is expected to be small in absolute value and far from any "
        "conventional critical value. No p-value is reported for any augmented "
        "Dickey Fuller case, because under the null the t ratio does not "
        "follow a Student t distribution and the usual tabulated values are "
        "interpolation schemes that differ between packages.",
        "series_random_walk",
        "adf-002.csv",
        {"kind": "adf", "series": "y", "lags": 1, "trend": "c"},
        TIGHT,
    ),
    spec(
        "adf-003",
        "adf",
        "Augmented Dickey Fuller regression with a deterministic trend",
        "edge",
        "A deterministic trend plus stationary autoregressive noise, tested "
        "with both a constant and a trend in the regression. The trend "
        "regressor is defined as the original one based position of each "
        "observation, not as one through the length of the trimmed sample, "
        "because those two conventions give different intercepts and "
        "different trend coefficients while leaving the statistic on the "
        "lagged level unchanged. Fixing the convention in the spec is what "
        "makes the intercept comparable at all.",
        "series_trend_stationary",
        "adf-003.csv",
        {"kind": "adf", "series": "y", "lags": 3, "trend": "ct"},
        TIGHT,
    ),
]

FAMILY_DIR = {
    "ols": "ols",
    "iv2sls": "iv-2sls",
    "panel_fe": "panel-fe",
    "adf": "adf",
}


def main() -> int:
    with open(os.path.join(ROOT, "fixtures", "manifest.json"), "r", encoding="utf-8") as h:
        manifest = json.load(h)
    written = 0
    for item in SPECS:
        filename = os.path.basename(item["fixture"]["path"])
        if filename not in manifest:
            raise SystemExit("fixture %s is not in the manifest" % filename)
        item["fixture"]["sha256"] = manifest[filename]["sha256"]
        directory = os.path.join(ROOT, "specs", FAMILY_DIR[item["family"]])
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "%s.json" % item["id"])
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(item, handle, indent=2)
            handle.write("\n")
        written += 1
    print("wrote %d specs" % written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
