"""Reference estimators, written directly against the linear algebra.

Nothing here delegates to a statistics library. That is the point: the values
in ``fixtures/golden`` must be defensible from first principles, so that when
statsmodels or R disagrees we can say which side is wrong instead of assuming
the library is right.

Conventions fixed by this module (documented because cross language agreement
is a convention question before it is a numerical one):

* ``df_resid = nobs - rank(X)``. Panel fixed effects additionally subtract the
  number of estimated entity intercepts.
* ``sigma2 = ss_resid / df_resid`` in every family, including 2SLS, where the
  residual is formed from the *observed* regressors and never from the first
  stage fitted values.
* Inference uses the Student t distribution with ``df_resid`` degrees of
  freedom, including under heteroskedasticity consistent covariance. This
  matches both ``sandwich::coeftest`` defaults in R and statsmodels ``OLS``
  with ``cov_type='HC1'``.
* ``r2`` is centred when the design contains an intercept and uncentred when
  it does not.
* Every floating point reduction goes through :mod:`econospec.linalg`, which
  sums exactly and rounds once. No reduction in this module is delegated to a
  BLAS kernel, because a BLAS kernel is chosen at run time from the CPU and the
  thread count and therefore is not a function of the input. That is not a
  hypothesis: docs/determinism.md records the same design returning a different
  last bit on two CPU families, and a different last bit on one machine when
  only the thread count changed. numpy is still used here for shape handling,
  slicing and elementwise arithmetic, all of which are exact or correctly
  rounded per element.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import linalg as la
from .distributions import student_t_two_sided_p

__all__ = [
    "RankDeficientError",
    "ols",
    "iv2sls",
    "panel_fe",
    "adf",
    "build_adf_design",
]

VCOV_TYPES = ("nonrobust", "HC0", "HC1", "HC2", "HC3")


class RankDeficientError(ValueError):
    """Raised when a design matrix does not have full column rank."""


def _check_rank(X: np.ndarray, name: str = "design") -> int:
    rank = la.rank(X)
    if rank < X.shape[1]:
        raise RankDeficientError(
            "%s matrix has %d columns but rank %d" % (name, X.shape[1], rank)
        )
    return rank


def _xtx_inv(X: np.ndarray) -> np.ndarray:
    """Inverse of X'X computed through the QR factorisation of X."""
    return la.xtx_inv(X)


def _solve_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return la.solve_ols(X, y)


def _sandwich(
    X: np.ndarray,
    resid: np.ndarray,
    xtx_inv: np.ndarray,
    vcov_type: str,
    df_resid: int,
) -> np.ndarray:
    n, k = X.shape
    if vcov_type == "HC0":
        w = resid ** 2
        scale = 1.0
    elif vcov_type == "HC1":
        w = resid ** 2
        scale = n / float(df_resid)
    elif vcov_type in ("HC2", "HC3"):
        h = la.leverage(X, xtx_inv)
        power = 1.0 if vcov_type == "HC2" else 2.0
        w = resid ** 2 / (1.0 - h) ** power
        scale = 1.0
    else:  # pragma: no cover - guarded by caller
        raise ValueError("unknown robust vcov type %r" % (vcov_type,))
    meat = la.weighted_crossprod(X, w)
    return scale * la.matmul(la.matmul(xtx_inv, meat), xtx_inv)


def _param_table(
    names: Sequence[str],
    beta: np.ndarray,
    vcov: np.ndarray,
    df_resid: int,
    with_pvalue: bool = True,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(names):
        se = math.sqrt(float(vcov[i][i]))
        tstat = float(beta[i]) / se
        entry = {
            "estimate": float(beta[i]),
            "se": se,
            "tstat": tstat,
        }
        if with_pvalue:
            entry["pvalue"] = float(student_t_two_sided_p(tstat, df_resid))
        out[name] = entry
    return out


def _fit_stats(
    y: np.ndarray, resid: np.ndarray, has_const: bool
) -> Dict[str, float]:
    ss_resid = la.sumsq(resid)
    if has_const:
        ss_total = la.sumsq(y - la.mean(y))
    else:
        ss_total = la.sumsq(y)
    return {"ss_resid": ss_resid, "ss_total": ss_total}


def ols(
    y: np.ndarray,
    X: np.ndarray,
    names: Sequence[str],
    vcov_type: str = "nonrobust",
    has_const: bool = True,
) -> Dict[str, object]:
    """Ordinary least squares with optional heteroskedasticity robust errors."""
    if vcov_type not in VCOV_TYPES:
        raise ValueError(
            "vcov_type must be one of %r, got %r" % (VCOV_TYPES, vcov_type)
        )
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y have different row counts")
    if len(names) != X.shape[1]:
        raise ValueError("names length does not match the number of columns")
    n, k = X.shape
    rank = _check_rank(X, "OLS design")
    df_resid = n - k
    if df_resid <= 0:
        raise ValueError("non positive residual degrees of freedom")

    beta = _solve_ols(X, y)
    resid = y - la.matvec(X, beta)
    ss = _fit_stats(y, resid, has_const)
    sigma2 = ss["ss_resid"] / df_resid
    xtx_inv = _xtx_inv(X)
    if vcov_type == "nonrobust":
        vcov = sigma2 * xtx_inv
    else:
        vcov = _sandwich(X, resid, xtx_inv, vcov_type, df_resid)

    df_model = k - 1 if has_const else k
    r2 = 1.0 - ss["ss_resid"] / ss["ss_total"]
    if has_const:
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / float(df_resid)
    else:
        adj_r2 = 1.0 - (1.0 - r2) * n / float(df_resid)

    return {
        "params": _param_table(names, beta, vcov, df_resid),
        "scalars": {
            "nobs": int(n),
            "df_resid": int(df_resid),
            "df_model": int(df_model),
            "rank": int(rank),
            "sigma2": float(sigma2),
            "ss_resid": ss["ss_resid"],
            "ss_total": ss["ss_total"],
            "r2": float(r2),
            "adj_r2": float(adj_r2),
        },
    }


def iv2sls(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    names: Sequence[str],
    n_endog: int,
    n_excluded: int,
    endog: Optional[np.ndarray] = None,
    Z_exog: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Two stage least squares.

    ``X`` holds the structural regressors (exogenous first, then endogenous)
    and ``Z`` holds the instrument set (the same exogenous block, then the
    excluded instruments). ``endog`` and ``Z_exog`` are supplied so the first
    stage exclusion F statistic can be computed without re-slicing here.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    n, k = X.shape
    if Z.shape[0] != n:
        raise ValueError("Z and X have different row counts")
    if Z.shape[1] < k:
        raise ValueError("under identified: fewer instruments than regressors")
    _check_rank(X, "structural design")
    _check_rank(Z, "instrument")

    ztz_inv = _xtx_inv(Z)
    pi = la.matmul(ztz_inv, la.crossprod(Z, X))
    xhat = la.matmul(Z, pi)
    _check_rank(xhat, "projected design")
    beta = _solve_ols(xhat, y)
    resid = y - la.matvec(X, beta)
    df_resid = n - k
    ss_resid = la.sumsq(resid)
    sigma2 = ss_resid / df_resid
    vcov = sigma2 * _xtx_inv(xhat)

    ss_total = la.sumsq(y - la.mean(y))
    r2 = 1.0 - ss_resid / ss_total

    scalars: Dict[str, float] = {
        "nobs": int(n),
        "df_resid": int(df_resid),
        "rank": int(k),
        "sigma2": float(sigma2),
        "ss_resid": ss_resid,
        "ss_total": ss_total,
        "r2": float(r2),
        "n_instruments": int(Z.shape[1]),
        "overid_df": int(Z.shape[1] - k),
    }

    if endog is not None and Z_exog is not None and n_endog == 1:
        d = np.asarray(endog, dtype=float).ravel()
        beta_u = _solve_ols(Z, d)
        rss_u = la.sumsq(d - la.matvec(Z, beta_u))
        beta_r = _solve_ols(Z_exog, d)
        rss_r = la.sumsq(d - la.matvec(Z_exog, beta_r))
        df_u = n - Z.shape[1]
        f_stat = ((rss_r - rss_u) / n_excluded) / (rss_u / df_u)
        scalars["first_stage_F"] = float(f_stat)
        scalars["first_stage_df_num"] = int(n_excluded)
        scalars["first_stage_df_den"] = int(df_u)

    return {
        "params": _param_table(names, beta, vcov, df_resid),
        "scalars": scalars,
    }


def panel_fe(
    y: np.ndarray,
    X: np.ndarray,
    entity: np.ndarray,
    names: Sequence[str],
) -> Dict[str, object]:
    """One way (entity) fixed effects through the within transformation.

    ``X`` must not contain an intercept: it is absorbed by the entity means.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    entity = np.asarray(entity)
    n, k = X.shape
    codes, inverse = np.unique(entity, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    n_entities = int(codes.shape[0])

    counts = la.group_counts(inverse, n_entities)
    y_mean = la.group_sums(inverse, y, n_entities) / counts
    yd = y - y_mean[inverse]
    Xd = np.empty_like(X)
    for j in range(k):
        col_mean = la.group_sums(inverse, X[:, j], n_entities) / counts
        Xd[:, j] = X[:, j] - col_mean[inverse]

    _check_rank(Xd, "within design")
    df_resid = n - n_entities - k
    if df_resid <= 0:
        raise ValueError("non positive residual degrees of freedom")

    beta = _solve_ols(Xd, yd)
    resid = yd - la.matvec(Xd, beta)
    ss_resid = la.sumsq(resid)
    sigma2 = ss_resid / df_resid
    vcov = sigma2 * _xtx_inv(Xd)
    ss_within = la.sumsq(yd)
    r2_within = 1.0 - ss_resid / ss_within

    return {
        "params": _param_table(names, beta, vcov, df_resid),
        "scalars": {
            "nobs": int(n),
            "n_entities": n_entities,
            "df_resid": int(df_resid),
            "rank": int(k),
            "sigma2": float(sigma2),
            "ss_resid": ss_resid,
            "ss_within": ss_within,
            "r2_within": float(r2_within),
        },
    }


def build_adf_design(
    series: np.ndarray, lags: int, trend: str
) -> Dict[str, object]:
    """Construct the augmented Dickey Fuller regression explicitly.

    The regression is

        d y_t = const + trend * t + rho * y_{t-1}
                + sum_{i=1..p} gamma_i d y_{t-i} + e_t

    estimated over ``t = lags + 2 ... T`` in 1-based indexing of the original
    series, which leaves ``T - lags - 1`` usable observations. ``trend`` takes
    the original 1-based position of the observation, so the design is fully
    determined by the spec and does not depend on how any library trims the
    sample.
    """
    if trend not in ("nc", "c", "ct"):
        raise ValueError("trend must be one of 'nc', 'c', 'ct', got %r" % (trend,))
    if lags < 0:
        raise ValueError("lags must be non negative")
    y = np.asarray(series, dtype=float).ravel()
    T = y.shape[0]
    n_eff = T - lags - 1
    if n_eff <= 0:
        raise ValueError("series too short for the requested lag order")

    dy = np.diff(y)  # dy[i] = y[i+1] - y[i], length T - 1
    start = lags  # index into dy of the first usable difference
    lhs = dy[start:]
    cols: List[np.ndarray] = []
    names: List[str] = []
    if trend in ("c", "ct"):
        cols.append(np.ones(n_eff))
        names.append("const")
    if trend == "ct":
        t_index = np.arange(lags + 2, T + 1, dtype=float)
        cols.append(t_index)
        names.append("trend")
    cols.append(y[start: start + n_eff])
    names.append("level.lag1")
    for i in range(1, lags + 1):
        cols.append(dy[start - i: start - i + n_eff])
        names.append("diff.lag%d" % i)

    return {
        "y": lhs,
        "X": np.column_stack(cols),
        "names": names,
        "nobs": int(n_eff),
        "has_const": trend in ("c", "ct"),
    }


def adf(series: np.ndarray, lags: int, trend: str) -> Dict[str, object]:
    """Augmented Dickey Fuller regression and test statistic.

    No p-value is reported. Under the unit root null the t ratio on
    ``level.lag1`` does not follow a Student t distribution, and the usual
    p-values come from interpolated MacKinnon tables that differ between
    implementations. EconoSpec reports the statistic and leaves the critical
    values to the reader; see docs/divergences.md.
    """
    design = build_adf_design(series, lags, trend)
    X = design["X"]
    lhs = design["y"]
    names = design["names"]
    n, k = X.shape
    _check_rank(X, "ADF design")
    df_resid = n - k
    if df_resid <= 0:
        raise ValueError("non positive residual degrees of freedom")

    beta = _solve_ols(X, lhs)
    resid = lhs - la.matvec(X, beta)
    ss_resid = la.sumsq(resid)
    sigma2 = ss_resid / df_resid
    vcov = sigma2 * _xtx_inv(X)
    params = _param_table(names, beta, vcov, df_resid, with_pvalue=False)

    return {
        "params": params,
        "scalars": {
            "nobs": int(n),
            "df_resid": int(df_resid),
            "rank": int(k),
            "sigma2": float(sigma2),
            "ss_resid": ss_resid,
            "adf_stat": float(params["level.lag1"]["tstat"]),
            "lags": int(lags),
        },
    }
