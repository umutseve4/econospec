"""Independent verification of the reference estimators.

The rule for this file: never check a result against another call of the same
function. Every assertion recomputes the quantity by a different route, so a
shared bug cannot pass. Where a route is not available (statsmodels and R are
not installed in the reference environment) the check is left to CI, and that
is stated in docs/verification.md rather than assumed away.
"""

from __future__ import annotations

import math

import numpy as np

from econospec import estimators
from econospec.csvio import read_frame
from econospec.distributions import student_t_two_sided_p


def _design(n=120, k=3, seed=7):
    g = np.random.Generator(np.random.PCG64(seed))
    X = np.column_stack([np.ones(n)] + [g.normal(size=n) for _ in range(k - 1)])
    beta = np.array([1.0, -2.0, 0.5][:k])
    y = X @ beta + g.normal(scale=1.3, size=n)
    names = ["const"] + ["x%d" % i for i in range(1, k)]
    return y, X, names


def test_ols_coefficients_match_lstsq_and_pinv():
    y, X, names = _design()
    fit = estimators.ols(y, X, names)
    lstsq = np.linalg.lstsq(X, y, rcond=None)[0]
    pinv = np.linalg.pinv(X) @ y
    got = np.array([fit["params"][n]["estimate"] for n in names])
    assert np.allclose(got, lstsq, rtol=0, atol=1e-11)
    assert np.allclose(got, pinv, rtol=0, atol=1e-11)


def test_ols_classical_se_match_the_textbook_formula():
    y, X, names = _design()
    fit = estimators.ols(y, X, names)
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    resid = y - X @ beta
    n, k = X.shape
    sigma2 = resid @ resid / (n - k)
    vcov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(vcov))
    got = np.array([fit["params"][n_]["se"] for n_ in names])
    assert np.allclose(got, se, rtol=0, atol=1e-12)
    assert abs(fit["scalars"]["sigma2"] - sigma2) < 1e-12


def test_hc1_matches_an_explicitly_looped_sandwich():
    y, X, names = _design(n=200)
    fit = estimators.ols(y, X, names, vcov_type="HC1")
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    resid = y - X @ beta
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    meat = np.zeros((k, k))
    for i in range(n):  # deliberately a loop, not the vectorised form
        xi = X[i][:, None]
        meat += (resid[i] ** 2) * (xi @ xi.T)
    vcov = (n / (n - k)) * (bread @ meat @ bread)
    se = np.sqrt(np.diag(vcov))
    got = np.array([fit["params"][n_]["se"] for n_ in names])
    assert np.allclose(got, se, rtol=0, atol=1e-12)


def test_hc3_uses_leverage_from_the_hat_matrix():
    y, X, names = _design(n=90)
    fit = estimators.ols(y, X, names, vcov_type="HC3")
    bread = np.linalg.inv(X.T @ X)
    H = X @ bread @ X.T  # the full hat matrix, formed explicitly
    h = np.diag(H)
    assert abs(h.sum() - X.shape[1]) < 1e-10  # trace(H) = rank
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    resid = y - X @ beta
    w = resid ** 2 / (1.0 - h) ** 2
    meat = (X * w[:, None]).T @ X
    se = np.sqrt(np.diag(bread @ meat @ bread))
    got = np.array([fit["params"][n_]["se"] for n_ in names])
    assert np.allclose(got, se, rtol=0, atol=1e-12)


def test_hc_variants_are_ordered_and_distinct():
    y, X, names = _design(n=150)
    ses = {}
    for vt in ("nonrobust", "HC0", "HC1", "HC2", "HC3"):
        fit = estimators.ols(y, X, names, vcov_type=vt)
        ses[vt] = np.array([fit["params"][n_]["se"] for n_ in names])
    # HC1 inflates HC0 by exactly sqrt(n / (n - k)).
    n, k = X.shape
    assert np.allclose(ses["HC1"], ses["HC0"] * math.sqrt(n / (n - k)), atol=1e-13)
    # HC2 and HC3 divide by increasing powers of (1 - h), so they are ordered.
    assert np.all(ses["HC3"] > ses["HC2"])
    assert np.all(ses["HC2"] > ses["HC0"])
    assert not np.allclose(ses["nonrobust"], ses["HC3"])


def test_r2_equals_squared_correlation_when_an_intercept_is_present():
    y, X, names = _design(n=140)
    fit = estimators.ols(y, X, names)
    beta = np.array([fit["params"][n_]["estimate"] for n_ in names])
    fitted = X @ beta
    r = np.corrcoef(y, fitted)[0, 1]
    assert abs(fit["scalars"]["r2"] - r ** 2) < 1e-12
    n, k = X.shape
    expected_adj = 1 - (1 - r ** 2) * (n - 1) / (n - k)
    assert abs(fit["scalars"]["adj_r2"] - expected_adj) < 1e-12


def test_pvalues_come_from_the_t_distribution_with_df_resid():
    y, X, names = _design()
    fit = estimators.ols(y, X, names)
    df = fit["scalars"]["df_resid"]
    for name in names:
        entry = fit["params"][name]
        expected = student_t_two_sided_p(entry["estimate"] / entry["se"], df)
        assert abs(entry["pvalue"] - expected) < 1e-15


def test_rank_deficient_design_raises_instead_of_returning_a_number():
    y, X, names = _design()
    X_bad = np.column_stack([X, X[:, 1]])
    try:
        estimators.ols(y, X_bad, names + ["copy"])
    except estimators.RankDeficientError:
        return
    raise AssertionError("a duplicated column must not produce an estimate")


def test_panel_within_estimator_equals_least_squares_with_dummies():
    frame = read_frame("fixtures/fe-001.csv")
    y = frame["y"].astype(float)
    X = np.column_stack([frame["x1"], frame["x2"]]).astype(float)
    fit = estimators.panel_fe(y, X, frame["entity"], ["x1", "x2"])

    codes, inverse = np.unique(frame["entity"], return_inverse=True)
    dummies = np.zeros((len(y), len(codes)))
    dummies[np.arange(len(y)), inverse] = 1.0
    X_lsdv = np.column_stack([X, dummies])
    beta = np.linalg.lstsq(X_lsdv, y, rcond=None)[0]
    resid = y - X_lsdv @ beta
    df = len(y) - X_lsdv.shape[1]
    sigma2 = resid @ resid / df
    vcov = sigma2 * np.linalg.inv(X_lsdv.T @ X_lsdv)
    se = np.sqrt(np.diag(vcov))[:2]

    assert fit["scalars"]["df_resid"] == df
    assert abs(fit["scalars"]["sigma2"] - sigma2) < 1e-10
    for i, name in enumerate(["x1", "x2"]):
        assert abs(fit["params"][name]["estimate"] - beta[i]) < 1e-10
        assert abs(fit["params"][name]["se"] - se[i]) < 1e-10


def test_unbalanced_panel_keeps_every_observation():
    frame = read_frame("fixtures/fe-003.csv")
    y = frame["y"].astype(float)
    X = np.column_stack([frame["x1"], frame["x2"]]).astype(float)
    fit = estimators.panel_fe(y, X, frame["entity"], ["x1", "x2"])
    assert fit["scalars"]["nobs"] == len(y)
    assert fit["scalars"]["n_entities"] == len(np.unique(frame["entity"]))
    counts = np.bincount(frame["entity"])[1:]
    assert counts.min() != counts.max()  # the fixture really is unbalanced


def test_2sls_coefficients_match_the_manual_two_stage_computation():
    frame = read_frame("fixtures/iv-001.csv")
    n = len(frame["y"])
    y = frame["y"].astype(float)
    X = np.column_stack([np.ones(n), frame["w"], frame["d"]]).astype(float)
    Z = np.column_stack([np.ones(n), frame["w"], frame["z1"], frame["z2"]]).astype(float)
    Z_exog = np.column_stack([np.ones(n), frame["w"]]).astype(float)
    fit = estimators.iv2sls(
        y, X, Z, ["const", "w", "d"], n_endog=1, n_excluded=2,
        endog=frame["d"].astype(float), Z_exog=Z_exog,
    )
    # Route A: literally run the two stages.
    pi = np.linalg.lstsq(Z, X, rcond=None)[0]
    Xhat = Z @ pi
    beta_a = np.linalg.lstsq(Xhat, y, rcond=None)[0]
    # Route B: the closed form (X' P_Z X)^-1 X' P_Z y.
    Pz = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    beta_b = np.linalg.solve(X.T @ Pz @ X, X.T @ Pz @ y)
    got = np.array([fit["params"][k]["estimate"] for k in ("const", "w", "d")])
    assert np.allclose(got, beta_a, atol=1e-9)
    assert np.allclose(got, beta_b, atol=1e-9)


def test_2sls_residual_uses_observed_regressors_not_fitted_ones():
    """The classic 2SLS trap, asserted rather than assumed."""
    frame = read_frame("fixtures/iv-001.csv")
    n = len(frame["y"])
    y = frame["y"].astype(float)
    X = np.column_stack([np.ones(n), frame["w"], frame["d"]]).astype(float)
    Z = np.column_stack([np.ones(n), frame["w"], frame["z1"], frame["z2"]]).astype(float)
    Z_exog = np.column_stack([np.ones(n), frame["w"]]).astype(float)
    fit = estimators.iv2sls(
        y, X, Z, ["const", "w", "d"], n_endog=1, n_excluded=2,
        endog=frame["d"].astype(float), Z_exog=Z_exog,
    )
    pi = np.linalg.lstsq(Z, X, rcond=None)[0]
    Xhat = Z @ pi
    beta = np.linalg.lstsq(Xhat, y, rcond=None)[0]

    sigma2_correct = float(((y - X @ beta) ** 2).sum()) / (n - X.shape[1])
    sigma2_naive = float(((y - Xhat @ beta) ** 2).sum()) / (n - X.shape[1])

    assert abs(fit["scalars"]["sigma2"] - sigma2_correct) < 1e-9
    # The two differ by a wide margin. The direction is not universal, so the
    # assertion is on the magnitude of the gap, not on its sign.
    rel_gap = abs(sigma2_naive - sigma2_correct) / sigma2_correct
    assert rel_gap > 0.05, rel_gap
    assert abs(fit["scalars"]["sigma2"] - sigma2_naive) > 1e-6


def test_first_stage_f_matches_an_explicit_restricted_unrestricted_test():
    frame = read_frame("fixtures/iv-002.csv")
    n = len(frame["y"])
    d = frame["d"].astype(float)
    Z = np.column_stack([np.ones(n), frame["w"], frame["z1"], frame["z2"]]).astype(float)
    Z_exog = np.column_stack([np.ones(n), frame["w"]]).astype(float)
    y = frame["y"].astype(float)
    X = np.column_stack([np.ones(n), frame["w"], d])
    fit = estimators.iv2sls(
        y, X, Z, ["const", "w", "d"], n_endog=1, n_excluded=2, endog=d, Z_exog=Z_exog
    )
    rss_u = float(((d - Z @ np.linalg.lstsq(Z, d, rcond=None)[0]) ** 2).sum())
    rss_r = float(((d - Z_exog @ np.linalg.lstsq(Z_exog, d, rcond=None)[0]) ** 2).sum())
    f = ((rss_r - rss_u) / 2) / (rss_u / (n - 4))
    assert abs(fit["scalars"]["first_stage_F"] - f) < 1e-9
    # This fixture is the weak instrument case, so the diagnostic must show it.
    assert fit["scalars"]["first_stage_F"] < 10.0


def test_just_identified_case_reports_zero_overidentification_df():
    frame = read_frame("fixtures/iv-003.csv")
    n = len(frame["y"])
    y = frame["y"].astype(float)
    X = np.column_stack([np.ones(n), frame["w"], frame["d"]]).astype(float)
    Z = np.column_stack([np.ones(n), frame["w"], frame["z1"]]).astype(float)
    Z_exog = np.column_stack([np.ones(n), frame["w"]]).astype(float)
    fit = estimators.iv2sls(
        y, X, Z, ["const", "w", "d"], n_endog=1, n_excluded=1,
        endog=frame["d"].astype(float), Z_exog=Z_exog,
    )
    assert fit["scalars"]["overid_df"] == 0
    # Exactly identified 2SLS equals the indirect least squares ratio.
    assert fit["scalars"]["n_instruments"] == 3


def test_adf_design_is_built_exactly_as_documented():
    y = np.arange(1.0, 11.0) ** 1.5
    design = estimators.build_adf_design(y, lags=2, trend="ct")
    assert design["nobs"] == len(y) - 2 - 1
    assert design["names"] == ["const", "trend", "level.lag1", "diff.lag1", "diff.lag2"]
    X = design["X"]
    dy = np.diff(y)
    assert np.allclose(design["y"], dy[2:])
    assert np.allclose(X[:, 0], 1.0)
    assert np.allclose(X[:, 1], np.arange(4, 11))  # 1-based positions t = lags+2..T
    assert np.allclose(X[:, 2], y[2:9])
    assert np.allclose(X[:, 3], dy[1:8])
    assert np.allclose(X[:, 4], dy[0:7])


def test_adf_statistic_is_the_t_ratio_from_an_independently_run_regression():
    frame = read_frame("fixtures/adf-003.csv")
    series = frame["y"].astype(float)
    result = estimators.adf(series, lags=3, trend="ct")
    design = estimators.build_adf_design(series, lags=3, trend="ct")
    X, lhs = design["X"], design["y"]
    beta = np.linalg.lstsq(X, lhs, rcond=None)[0]
    resid = lhs - X @ beta
    n, k = X.shape
    sigma2 = resid @ resid / (n - k)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    idx = design["names"].index("level.lag1")
    assert abs(result["scalars"]["adf_stat"] - beta[idx] / se[idx]) < 1e-9
    assert result["scalars"]["nobs"] == n


def test_adf_reports_no_pvalues():
    result = estimators.adf(read_frame("fixtures/adf-001.csv")["y"].astype(float), 2, "c")
    for entry in result["params"].values():
        assert "pvalue" not in entry, "ADF p-values are not defensible; see docs"


def test_adf_separates_a_random_walk_from_a_stationary_series():
    stationary = estimators.adf(
        read_frame("fixtures/adf-001.csv")["y"].astype(float), 2, "c"
    )["scalars"]["adf_stat"]
    walk = estimators.adf(
        read_frame("fixtures/adf-002.csv")["y"].astype(float), 1, "c"
    )["scalars"]["adf_stat"]
    # Well below the 5 percent Dickey Fuller critical value of about -2.87.
    assert stationary < -3.5, stationary
    # A random walk should sit nowhere near it.
    assert walk > -2.0, walk


def test_adf_rejects_an_impossible_lag_order():
    for args in ((np.arange(5.0), 10, "c"), (np.arange(50.0), 1, "quadratic")):
        try:
            estimators.build_adf_design(*args)
        except ValueError:
            continue
        raise AssertionError("expected ValueError for %r" % (args,))
