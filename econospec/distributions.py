"""Distribution functions implemented from scratch.

The reference kernel deliberately avoids SciPy. Every tail probability that
EconoSpec reports is produced here, so the numbers in the golden manifests can
be traced to code in this repository rather than to a third party library.

Accuracy targets (checked in tests/test_distributions.py against published
values): absolute error below 1e-12 for the Student t CDF and the normal CDF
over the ranges used by the fixtures.
"""

from __future__ import annotations

import math

__all__ = [
    "log_beta",
    "betacf",
    "betainc_regularized",
    "student_t_cdf",
    "student_t_sf",
    "student_t_two_sided_p",
    "normal_cdf",
    "normal_two_sided_p",
]

_MAXIT = 400
_EPS = 3.0e-16
_FPMIN = 1.0e-300


def log_beta(a: float, b: float) -> float:
    """Natural log of the Beta function."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def betacf(a: float, b: float, x: float) -> float:
    """Continued fraction expansion for the incomplete beta function.

    Modified Lentz algorithm. Converges for x < (a + 1) / (a + b + 2); the
    caller is responsible for applying the symmetry transformation otherwise.
    """
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            return h
    raise RuntimeError(
        "betacf failed to converge for a=%r b=%r x=%r" % (a, b, x)
    )


def betainc_regularized(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if not (0.0 <= x <= 1.0):
        raise ValueError("x must lie in [0, 1], got %r" % (x,))
    if a <= 0.0 or b <= 0.0:
        raise ValueError("a and b must be positive, got a=%r b=%r" % (a, b))
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    front = math.exp(
        a * math.log(x) + b * math.log1p(-x) - log_beta(a, b)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * betacf(a, b, x) / a
    return 1.0 - front * betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: float) -> float:
    """P(T <= t) for a Student t variate with ``df`` degrees of freedom."""
    if df <= 0:
        raise ValueError("df must be positive, got %r" % (df,))
    if t != t:  # NaN
        return float("nan")
    if math.isinf(t):
        return 1.0 if t > 0 else 0.0
    x = df / (df + t * t)
    tail = 0.5 * betainc_regularized(0.5 * df, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def student_t_sf(t: float, df: float) -> float:
    """Upper tail P(T > t)."""
    return 1.0 - student_t_cdf(t, df)


def student_t_two_sided_p(t: float, df: float) -> float:
    """Two sided p-value, computed without cancellation for large |t|."""
    if df <= 0:
        raise ValueError("df must be positive, got %r" % (df,))
    if t != t:
        return float("nan")
    if math.isinf(t):
        return 0.0
    x = df / (df + t * t)
    return betainc_regularized(0.5 * df, 0.5, x)


def normal_cdf(z: float) -> float:
    """Standard normal CDF via the complementary error function."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def normal_two_sided_p(z: float) -> float:
    """Two sided normal p-value, stable in the far tail."""
    return math.erfc(abs(z) / math.sqrt(2.0))
