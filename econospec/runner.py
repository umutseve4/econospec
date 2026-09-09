"""Turn a spec into a canonical result envelope.

The envelope is the only thing adapters exchange. Its shape is fixed:

    {
      "spec_id": ...,
      "adapter": ...,
      "params": {name: {estimate, se, tstat[, pvalue]}},
      "scalars": {name: number}
    }

Adapters written in other languages must produce exactly these keys. A missing
key is a failure, not a skip.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np

from . import estimators
from .csvio import read_frame

__all__ = ["build_matrix", "run_spec", "flatten", "REFERENCE_ADAPTER"]

REFERENCE_ADAPTER = "reference-numpy"


def build_matrix(frame: Dict[str, np.ndarray], names: List[str], intercept: bool):
    cols = []
    out_names: List[str] = []
    if intercept:
        n = len(next(iter(frame.values())))
        cols.append(np.ones(n))
        out_names.append("const")
    for name in names:
        if name not in frame:
            raise KeyError("column %r is not in the fixture" % (name,))
        cols.append(np.asarray(frame[name], dtype=float))
        out_names.append(name)
    return np.column_stack(cols), out_names


def run_spec(spec: dict, root: str) -> dict:
    frame = read_frame(os.path.join(root, spec["fixture"]["path"]))
    model = spec["model"]
    kind = model["kind"]

    if kind == "ols":
        y = np.asarray(frame[model["response"]], dtype=float)
        X, names = build_matrix(frame, model["regressors"], model["intercept"])
        result = estimators.ols(
            y, X, names, vcov_type=model["vcov"], has_const=model["intercept"]
        )
    elif kind == "iv2sls":
        y = np.asarray(frame[model["response"]], dtype=float)
        X, names = build_matrix(
            frame, list(model["exog"]) + list(model["endog"]), model["intercept"]
        )
        Z, _ = build_matrix(
            frame, list(model["exog"]) + list(model["instruments"]), model["intercept"]
        )
        Z_exog, _ = build_matrix(frame, list(model["exog"]), model["intercept"])
        endog = np.column_stack(
            [np.asarray(frame[c], dtype=float) for c in model["endog"]]
        )
        result = estimators.iv2sls(
            y,
            X,
            Z,
            names,
            n_endog=len(model["endog"]),
            n_excluded=len(model["instruments"]),
            endog=endog,
            Z_exog=Z_exog,
        )
    elif kind == "panel_fe":
        y = np.asarray(frame[model["response"]], dtype=float)
        X, names = build_matrix(frame, model["regressors"], intercept=False)
        result = estimators.panel_fe(y, X, frame[model["entity"]], names)
    elif kind == "adf":
        series = np.asarray(frame[model["series"]], dtype=float)
        result = estimators.adf(series, lags=model["lags"], trend=model["trend"])
    else:  # pragma: no cover - validation catches this earlier
        raise ValueError("unknown model kind %r" % (kind,))

    return {
        "spec_id": spec["id"],
        "adapter": REFERENCE_ADAPTER,
        "params": result["params"],
        "scalars": result["scalars"],
    }


def flatten(envelope: dict) -> Dict[str, float]:
    """Flatten an envelope into dotted keys for comparison."""
    out: Dict[str, float] = {}
    for name, entry in envelope.get("params", {}).items():
        for stat, value in entry.items():
            out["params.%s.%s" % (name, stat)] = float(value)
    for name, value in envelope.get("scalars", {}).items():
        out["scalars.%s" % (name,)] = float(value)
    return out
