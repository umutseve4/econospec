#!/usr/bin/env python3
"""statsmodels and linearmodels adapter.

This script deliberately does not import ``econospec``. It reads the spec, the
fixture CSV and nothing else, so that agreement with the reference kernel is
evidence rather than tautology. Every scalar is recomputed here from the
adapter's own fitted objects using the definitions in the spec contract, so a
library changing what it names ``rsquared`` cannot quietly change the answer.

Usage:
    python adapters/python/run_statsmodels.py --out results/python-statsmodels
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

ADAPTER = "python-statsmodels"


def envelope(spec_id, params, scalars):
    return {
        "spec_id": spec_id,
        "adapter": ADAPTER,
        "params": params,
        "scalars": {k: float(v) for k, v in scalars.items()},
    }


def table(names, beta, se, df_resid, with_pvalue=True):
    from scipy import stats

    out = {}
    for i, name in enumerate(names):
        t = float(beta[i] / se[i])
        entry = {"estimate": float(beta[i]), "se": float(se[i]), "tstat": t}
        if with_pvalue:
            entry["pvalue"] = float(2.0 * stats.t.sf(abs(t), df_resid))
        out[name] = entry
    return out


def design(frame, regressors, intercept):
    cols, names = [], []
    if intercept:
        cols.append(np.ones(len(frame)))
        names.append("const")
    for r in regressors:
        cols.append(frame[r].to_numpy(dtype=float))
        names.append(r)
    return np.column_stack(cols), names


def run_ols(spec, frame):
    model = spec["model"]
    y = frame[model["response"]].to_numpy(dtype=float)
    X, names = design(frame, model["regressors"], model["intercept"])
    kwargs = {"use_t": True}
    if model["vcov"] != "nonrobust":
        kwargs["cov_type"] = model["vcov"]
    res = sm.OLS(y, X).fit(**kwargs)

    n, k = X.shape
    resid = y - X @ res.params
    ss_resid = float(resid @ resid)
    df_resid = n - k
    ss_total = float(((y - y.mean()) ** 2).sum()) if model["intercept"] else float((y ** 2).sum())
    r2 = 1.0 - ss_resid / ss_total
    if model["intercept"]:
        adj = 1.0 - (1.0 - r2) * (n - 1) / df_resid
    else:
        adj = 1.0 - (1.0 - r2) * n / df_resid
    scalars = {
        "nobs": n,
        "df_resid": df_resid,
        "df_model": k - 1 if model["intercept"] else k,
        "rank": np.linalg.matrix_rank(X),
        "sigma2": ss_resid / df_resid,
        "ss_resid": ss_resid,
        "ss_total": ss_total,
        "r2": r2,
        "adj_r2": adj,
    }
    return envelope(spec["id"], table(names, res.params, res.bse, df_resid), scalars)


def run_iv(spec, frame):
    from linearmodels.iv import IV2SLS

    model = spec["model"]
    y = frame[model["response"]]
    exog = frame[list(model["exog"])].copy()
    if model["intercept"]:
        exog.insert(0, "const", 1.0)
    endog = frame[list(model["endog"])]
    instr = frame[list(model["instruments"])]
    res = IV2SLS(y, exog, endog, instr).fit(cov_type="unadjusted", debiased=True)

    names = list(exog.columns) + list(endog.columns)
    beta = np.array([res.params[n] for n in names], dtype=float)
    se = np.array([res.std_errors[n] for n in names], dtype=float)
    n = len(frame)
    k = len(names)
    df_resid = n - k
    Xs = np.column_stack([exog.to_numpy(dtype=float), endog.to_numpy(dtype=float)])
    resid = y.to_numpy(dtype=float) - Xs @ beta
    ss_resid = float(resid @ resid)
    yv = y.to_numpy(dtype=float)
    ss_total = float(((yv - yv.mean()) ** 2).sum())

    Z = np.column_stack([exog.to_numpy(dtype=float), instr.to_numpy(dtype=float)])
    d = endog.to_numpy(dtype=float).ravel()
    rss_u = float(((d - Z @ np.linalg.lstsq(Z, d, rcond=None)[0]) ** 2).sum())
    Ze = exog.to_numpy(dtype=float)
    rss_r = float(((d - Ze @ np.linalg.lstsq(Ze, d, rcond=None)[0]) ** 2).sum())
    q = instr.shape[1]
    df_den = n - Z.shape[1]
    scalars = {
        "nobs": n,
        "df_resid": df_resid,
        "rank": k,
        "sigma2": ss_resid / df_resid,
        "ss_resid": ss_resid,
        "ss_total": ss_total,
        "r2": 1.0 - ss_resid / ss_total,
        "n_instruments": Z.shape[1],
        "overid_df": Z.shape[1] - k,
        "first_stage_F": ((rss_r - rss_u) / q) / (rss_u / df_den),
        "first_stage_df_num": q,
        "first_stage_df_den": df_den,
    }
    return envelope(spec["id"], table(names, beta, se, df_resid), scalars)


def run_panel(spec, frame):
    from linearmodels.panel import PanelOLS

    model = spec["model"]
    df = frame.copy()
    df["__period"] = np.arange(len(df))
    df = df.set_index([model["entity"], "__period"])
    y = df[model["response"]]
    X = df[list(model["regressors"])]
    res = PanelOLS(y, X, entity_effects=True).fit(cov_type="unadjusted", debiased=True)

    names = list(model["regressors"])
    beta = np.array([res.params[n] for n in names], dtype=float)
    se = np.array([res.std_errors[n] for n in names], dtype=float)
    n = len(frame)
    entities = frame[model["entity"]].to_numpy()
    n_entities = int(len(np.unique(entities)))
    k = len(names)
    df_resid = n - n_entities - k

    codes, inverse = np.unique(entities, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    yv = frame[model["response"]].to_numpy(dtype=float)
    yd = yv - (np.bincount(inverse, weights=yv) / counts)[inverse]
    Xv = frame[names].to_numpy(dtype=float)
    Xd = np.column_stack(
        [Xv[:, j] - (np.bincount(inverse, weights=Xv[:, j]) / counts)[inverse]
         for j in range(k)]
    )
    resid = yd - Xd @ beta
    ss_resid = float(resid @ resid)
    ss_within = float((yd ** 2).sum())
    scalars = {
        "nobs": n,
        "n_entities": n_entities,
        "df_resid": df_resid,
        "rank": k,
        "sigma2": ss_resid / df_resid,
        "ss_resid": ss_resid,
        "ss_within": ss_within,
        "r2_within": 1.0 - ss_resid / ss_within,
    }
    return envelope(spec["id"], table(names, beta, se, df_resid), scalars)


def run_adf(spec, frame):
    model = spec["model"]
    y = frame[model["series"]].to_numpy(dtype=float)
    lags, trend = int(model["lags"]), model["trend"]
    T = len(y)
    n_eff = T - lags - 1
    dy = np.diff(y)
    lhs = dy[lags:]
    cols, names = [], []
    if trend in ("c", "ct"):
        cols.append(np.ones(n_eff))
        names.append("const")
    if trend == "ct":
        cols.append(np.arange(lags + 2, T + 1, dtype=float))
        names.append("trend")
    cols.append(y[lags: lags + n_eff])
    names.append("level.lag1")
    for i in range(1, lags + 1):
        cols.append(dy[lags - i: lags - i + n_eff])
        names.append("diff.lag%d" % i)
    X = np.column_stack(cols)

    res = sm.OLS(lhs, X).fit()
    beta, se = res.params, res.bse
    df_resid = n_eff - X.shape[1]
    resid = lhs - X @ beta
    ss_resid = float(resid @ resid)

    # Independent cross check against the library routine. The trend
    # parameterisation differs (statsmodels restarts the trend at one), which
    # leaves the statistic on the lagged level unchanged because the two
    # parameterisations span the same column space together with the constant.
    stat = float(beta[names.index("level.lag1")] / se[names.index("level.lag1")])
    library_stat = adfuller(y, maxlag=lags, regression=trend, autolag=None)[0]
    if abs(stat - library_stat) > 1e-8:
        raise SystemExit(
            "ADF cross check failed for %s: explicit %r vs adfuller %r"
            % (spec["id"], stat, library_stat)
        )

    scalars = {
        "nobs": n_eff,
        "df_resid": df_resid,
        "rank": X.shape[1],
        "sigma2": ss_resid / df_resid,
        "ss_resid": ss_resid,
        "adf_stat": stat,
        "lags": lags,
    }
    return envelope(spec["id"], table(names, beta, se, df_resid, with_pvalue=False), scalars)


RUNNERS = {"ols": run_ols, "iv2sls": run_iv, "panel_fe": run_panel, "adf": run_adf}


def write_environment(out_dir):
    """Record what actually produced these numbers.

    A conformance claim without a version list is a claim about nothing, so
    this file is written next to the envelopes and rendered in the report.
    """
    import platform
    import sys as _sys

    import linearmodels
    import scipy
    import statsmodels

    payload = {
        "adapter": ADAPTER,
        "language": "python",
        "runtime": _sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "linearmodels": linearmodels.__version__,
        },
    }
    with open(os.path.join(out_dir, "environment.json"), "w",
              encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="results/python-statsmodels")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    write_environment(args.out)
    paths = sorted(glob.glob(os.path.join(args.root, "specs", "*", "*.json")))
    if not paths:
        raise SystemExit("no specs found under %s/specs" % args.root)
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            spec = json.load(handle)
        frame = pd.read_csv(os.path.join(args.root, spec["fixture"]["path"]))
        env = RUNNERS[spec["model"]["kind"]](spec, frame)
        with open(os.path.join(args.out, "%s.json" % spec["id"]), "w",
                  encoding="utf-8", newline="\n") as handle:
            json.dump(env, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print("%s ok" % spec["id"])
    print("wrote %d envelopes to %s" % (len(paths), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
