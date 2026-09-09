"""Deterministic data generating processes for the fixtures.

Fixtures are generated once in Python and committed as CSV. R never draws a
random number: it reads the same bytes. That removes cross language RNG
differences from the conformance question entirely, which is the only way a
disagreement can be attributed to the estimator rather than to the data.

Every generator takes an explicit seed and uses ``numpy.random.Generator`` with
the PCG64 bit generator, so regenerating on a different machine reproduces the
files byte for byte (asserted in tests/test_fixtures.py).
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

__all__ = ["GENERATORS", "generate"]

Frame = Dict[str, np.ndarray]


def _rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def ols_nominal(seed: int = 20260101, n: int = 240) -> Frame:
    """Well behaved homoskedastic linear model."""
    g = _rng(seed)
    x1 = g.normal(0.0, 1.0, n)
    x2 = g.normal(2.0, 1.5, n)
    u = g.normal(0.0, 1.0, n)
    y = 1.5 + 2.0 * x1 - 0.75 * x2 + u
    return {"y": y, "x1": x1, "x2": x2}


def ols_heteroskedastic(seed: int = 20260102, n: int = 300) -> Frame:
    """Variance grows with x1, so HC1 and the classical errors must differ."""
    g = _rng(seed)
    x1 = g.uniform(-2.0, 2.0, n)
    x2 = g.normal(0.0, 1.0, n)
    sigma = 0.35 + 0.9 * np.abs(x1)
    u = g.normal(0.0, 1.0, n) * sigma
    y = -0.5 + 1.25 * x1 + 0.4 * x2 + u
    return {"y": y, "x1": x1, "x2": x2}


def ols_near_collinear(seed: int = 20260103, n: int = 180) -> Frame:
    """x2 is x1 plus a small perturbation: a deliberately ill conditioned design."""
    g = _rng(seed)
    x1 = g.normal(0.0, 1.0, n)
    x2 = x1 + g.normal(0.0, 0.02, n)
    u = g.normal(0.0, 0.5, n)
    y = 0.25 + 1.0 * x1 + 1.0 * x2 + u
    return {"y": y, "x1": x1, "x2": x2}


def iv_strong(seed: int = 20260201, n: int = 400) -> Frame:
    """Endogenous regressor with two strong excluded instruments."""
    g = _rng(seed)
    z1 = g.normal(0.0, 1.0, n)
    z2 = g.normal(0.0, 1.0, n)
    w = g.normal(0.0, 1.0, n)
    e = g.normal(0.0, 1.0, n)
    v = g.normal(0.0, 1.0, n)
    d = 0.9 * z1 + 0.7 * z2 + 0.3 * w + 0.8 * e + v
    y = 1.0 + 1.5 * d + 0.5 * w + 2.0 * e
    return {"y": y, "d": d, "w": w, "z1": z1, "z2": z2}


def iv_weak(seed: int = 20260202, n: int = 400) -> Frame:
    """Instrument relevance is near zero: the first stage F should be small."""
    g = _rng(seed)
    z1 = g.normal(0.0, 1.0, n)
    z2 = g.normal(0.0, 1.0, n)
    w = g.normal(0.0, 1.0, n)
    e = g.normal(0.0, 1.0, n)
    v = g.normal(0.0, 1.0, n)
    d = 0.04 * z1 + 0.03 * z2 + 0.3 * w + 0.8 * e + v
    y = 1.0 + 1.5 * d + 0.5 * w + 2.0 * e
    return {"y": y, "d": d, "w": w, "z1": z1, "z2": z2}


def iv_just_identified(seed: int = 20260203, n: int = 350) -> Frame:
    """One instrument for one endogenous regressor: overid_df must be zero."""
    g = _rng(seed)
    z1 = g.normal(0.0, 1.0, n)
    w = g.normal(0.0, 1.0, n)
    e = g.normal(0.0, 1.0, n)
    v = g.normal(0.0, 1.0, n)
    d = 1.1 * z1 + 0.25 * w + 0.8 * e + v
    y = -0.4 + 0.9 * d + 0.6 * w + 2.0 * e
    return {"y": y, "d": d, "w": w, "z1": z1}


def panel_balanced(seed: int = 20260301, n_entities: int = 60, T: int = 8) -> Frame:
    """Balanced panel with entity effects uncorrelated with the regressors."""
    g = _rng(seed)
    alpha = g.normal(0.0, 1.5, n_entities)
    entity: List[int] = []
    period: List[int] = []
    x1: List[float] = []
    x2: List[float] = []
    y: List[float] = []
    for i in range(n_entities):
        for t in range(1, T + 1):
            a = g.normal(0.0, 1.0)
            b = g.normal(0.0, 1.0)
            e = g.normal(0.0, 0.8)
            entity.append(i + 1)
            period.append(t)
            x1.append(a)
            x2.append(b)
            y.append(alpha[i] + 1.2 * a - 0.6 * b + e)
    return {
        "entity": np.array(entity),
        "period": np.array(period),
        "y": np.array(y),
        "x1": np.array(x1),
        "x2": np.array(x2),
    }


def panel_confounded(seed: int = 20260302, n_entities: int = 50, T: int = 10) -> Frame:
    """Entity effects are correlated with x1, so pooled OLS is biased upward."""
    g = _rng(seed)
    alpha = g.normal(0.0, 2.0, n_entities)
    entity: List[int] = []
    period: List[int] = []
    x1: List[float] = []
    x2: List[float] = []
    y: List[float] = []
    for i in range(n_entities):
        for t in range(1, T + 1):
            a = 0.8 * alpha[i] + g.normal(0.0, 1.0)
            b = g.normal(0.0, 1.0)
            e = g.normal(0.0, 0.7)
            entity.append(i + 1)
            period.append(t)
            x1.append(a)
            x2.append(b)
            y.append(alpha[i] + 1.0 * a + 0.5 * b + e)
    return {
        "entity": np.array(entity),
        "period": np.array(period),
        "y": np.array(y),
        "x1": np.array(x1),
        "x2": np.array(x2),
    }


def panel_unbalanced(seed: int = 20260303, n_entities: int = 55) -> Frame:
    """Unbalanced panel: entity lengths vary between 3 and 12 periods."""
    g = _rng(seed)
    alpha = g.normal(0.0, 1.2, n_entities)
    lengths = g.integers(3, 13, n_entities)
    entity: List[int] = []
    period: List[int] = []
    x1: List[float] = []
    x2: List[float] = []
    y: List[float] = []
    for i in range(n_entities):
        for t in range(1, int(lengths[i]) + 1):
            a = g.normal(0.0, 1.0)
            b = g.normal(0.0, 1.3)
            e = g.normal(0.0, 0.9)
            entity.append(i + 1)
            period.append(t)
            x1.append(a)
            x2.append(b)
            y.append(alpha[i] + 0.7 * a - 1.1 * b + e)
    return {
        "entity": np.array(entity),
        "period": np.array(period),
        "y": np.array(y),
        "x1": np.array(x1),
        "x2": np.array(x2),
    }


def series_stationary(seed: int = 20260401, T: int = 300) -> Frame:
    """Stationary AR(1) with phi = 0.55."""
    g = _rng(seed)
    e = g.normal(0.0, 1.0, T)
    y = np.empty(T)
    y[0] = e[0]
    for t in range(1, T):
        y[t] = 0.55 * y[t - 1] + e[t]
    return {"t": np.arange(1, T + 1), "y": y}


def series_random_walk(seed: int = 20260402, T: int = 300) -> Frame:
    """Pure random walk: a unit root is present by construction."""
    g = _rng(seed)
    e = g.normal(0.0, 1.0, T)
    y = np.cumsum(e)
    return {"t": np.arange(1, T + 1), "y": y}


def series_trend_stationary(seed: int = 20260403, T: int = 320) -> Frame:
    """Deterministic trend plus stationary AR(1) noise."""
    g = _rng(seed)
    e = g.normal(0.0, 1.0, T)
    noise = np.empty(T)
    noise[0] = e[0]
    for t in range(1, T):
        noise[t] = 0.4 * noise[t - 1] + e[t]
    t_index = np.arange(1, T + 1, dtype=float)
    y = 2.0 + 0.05 * t_index + noise
    return {"t": np.arange(1, T + 1), "y": y}


GENERATORS: Dict[str, Callable[..., Frame]] = {
    "ols_nominal": ols_nominal,
    "ols_heteroskedastic": ols_heteroskedastic,
    "ols_near_collinear": ols_near_collinear,
    "iv_strong": iv_strong,
    "iv_weak": iv_weak,
    "iv_just_identified": iv_just_identified,
    "panel_balanced": panel_balanced,
    "panel_confounded": panel_confounded,
    "panel_unbalanced": panel_unbalanced,
    "series_stationary": series_stationary,
    "series_random_walk": series_random_walk,
    "series_trend_stationary": series_trend_stationary,
}


def generate(name: str) -> Frame:
    if name not in GENERATORS:
        raise KeyError("unknown generator %r" % (name,))
    return GENERATORS[name]()
