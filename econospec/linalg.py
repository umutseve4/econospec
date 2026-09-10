"""Exactly rounded linear algebra for the reference kernel.

Why this module exists
----------------------

The suite asserts that the reference kernel reproduces the committed golden
manifest bit for bit. That claim is only meaningful if the kernel is a
function of its inputs alone. It was not. ``np.linalg.qr`` and
``np.linalg.solve`` enter LAPACK, LAPACK enters OpenBLAS, and OpenBLAS picks a
micro kernel from the instruction set it finds at run time and blocks the
reduction according to the thread count. The measurement in
docs/determinism.md shows the same design returning two different last bits on
two different CPU families, and shows the value changing on a *single* machine
when the thread count changes. No environment variable closes that gap,
because the dispatch is not about threads.

So the reference path does not call a dispatched kernel at all. Every
reduction here is ``math.fsum``, which computes the exact sum of its arguments
and rounds once at the end (Shewchuk's algorithm, implemented in CPython in C).
Every other operation used below is a single IEEE 754 double operation:
multiplication, subtraction, division and ``math.sqrt`` are all required by the
standard to be correctly rounded. A composition of correctly rounded operations
performed in a fixed order is a pure function of its inputs on every conforming
platform, whatever the CPU, the thread count or the BLAS build.

Two consequences worth stating plainly, because they are the cost of the
guarantee:

* This is slower than BLAS. On the fixtures in this repository the whole suite
  spends well under a second here, so the trade is free in practice, but it
  would not be for large problems. EconoSpec compares small published designs;
  it is not a numerical library.
* Order still matters, and is therefore fixed. ``fsum`` makes each individual
  reduction order independent, but the *algorithm* is not: a Householder
  reflection depends on which row sits at the pivot. Every routine below
  documents the order it uses, and that order is part of the specification.

The functions take and return numpy arrays so the estimators keep their shape
handling, but no numpy reduction is ever performed on the values.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

__all__ = [
    "EPS",
    "total",
    "mean",
    "dot",
    "sumsq",
    "matvec",
    "matmul",
    "crossprod",
    "weighted_crossprod",
    "leverage",
    "qr_rhs",
    "solve_upper",
    "inv_upper",
    "rank",
    "solve_ols",
    "xtx_inv",
    "group_sums",
    "group_counts",
]

#: Machine epsilon for IEEE 754 binary64, written out rather than read from
#: numpy so the tolerance below is a constant of the specification.
EPS = 2.220446049250313e-16


def _rows(A) -> List[List[float]]:
    return [[float(v) for v in row] for row in np.asarray(A, dtype=float)]


def _vec(x) -> List[float]:
    return [float(v) for v in np.asarray(x, dtype=float).ravel()]


def total(values) -> float:
    """Exactly rounded sum."""
    return math.fsum(_vec(values))


def mean(values) -> float:
    """Exactly rounded sum divided by the count, in that order."""
    v = _vec(values)
    if not v:
        raise ValueError("mean of an empty sequence")
    return math.fsum(v) / len(v)


def dot(a, b) -> float:
    """Exactly rounded inner product.

    Each product ``a_i * b_i`` is one correctly rounded multiplication and the
    products are then summed exactly, so the result depends on the multiset of
    products and not on the order they are visited in.
    """
    va = _vec(a)
    vb = _vec(b)
    if len(va) != len(vb):
        raise ValueError("dot of vectors with lengths %d and %d" % (len(va), len(vb)))
    return math.fsum([x * y for x, y in zip(va, vb)])


def sumsq(a) -> float:
    """Exactly rounded sum of squares."""
    return math.fsum([x * x for x in _vec(a)])


def matvec(A, x) -> np.ndarray:
    """``A @ x`` with an exactly rounded reduction per row."""
    rows = _rows(A)
    xv = _vec(x)
    if rows and len(rows[0]) != len(xv):
        raise ValueError("matvec shape mismatch")
    return np.array(
        [math.fsum([row[j] * xv[j] for j in range(len(xv))]) for row in rows],
        dtype=float,
    )


def matmul(A, B) -> np.ndarray:
    """``A @ B`` with an exactly rounded reduction per output cell."""
    ra = _rows(A)
    rb = _rows(B)
    if ra and rb and len(ra[0]) != len(rb):
        raise ValueError("matmul shape mismatch")
    inner = len(rb)
    cols = list(zip(*rb)) if rb else []
    return np.array(
        [[math.fsum([row[k] * col[k] for k in range(inner)]) for col in cols]
         for row in ra],
        dtype=float,
    )


def crossprod(A, B) -> np.ndarray:
    """``A.T @ B`` without materialising a transpose in numpy."""
    ra = _rows(A)
    rb = _rows(B)
    if len(ra) != len(rb):
        raise ValueError("crossprod row count mismatch")
    n = len(ra)
    ka = len(ra[0]) if ra else 0
    kb = len(rb[0]) if rb else 0
    return np.array(
        [[math.fsum([ra[i][a] * rb[i][b] for i in range(n)]) for b in range(kb)]
         for a in range(ka)],
        dtype=float,
    )


def weighted_crossprod(X, w) -> np.ndarray:
    """``(X * w[:, None]).T @ X``.

    The product order is ``(X[i, a] * w[i]) * X[i, b]``, matching the numpy
    expression it replaces, so the only difference from the previous kernel is
    the accumulation, not the multiplication order.
    """
    rx = _rows(X)
    wv = _vec(w)
    if len(rx) != len(wv):
        raise ValueError("weight length does not match the row count")
    n = len(rx)
    k = len(rx[0]) if rx else 0
    return np.array(
        [[math.fsum([(rx[i][a] * wv[i]) * rx[i][b] for i in range(n)])
          for b in range(k)] for a in range(k)],
        dtype=float,
    )


def leverage(X, M) -> np.ndarray:
    """Row wise ``x_i' M x_i``, the hat matrix diagonal when ``M`` is (X'X)^-1.

    Replaces ``np.einsum("ij,jk,ik->i", X, M, X)``. The inner product is formed
    first, then contracted with the row, both exactly rounded.
    """
    rx = _rows(X)
    rm = _rows(M)
    k = len(rm)
    out = []
    for row in rx:
        mid = [math.fsum([rm[a][b] * row[b] for b in range(k)]) for a in range(k)]
        out.append(math.fsum([row[a] * mid[a] for a in range(k)]))
    return np.array(out, dtype=float)


def qr_rhs(A, y=None) -> Tuple[np.ndarray, np.ndarray]:
    """Householder QR of ``A``, returning ``(R, Q.T @ y)``.

    ``R`` is the ``n by n`` upper triangular factor of the ``m by n`` design.
    When ``y`` is omitted the second element is an empty array. The reflections
    are applied in column order, and within a column to the trailing columns in
    index order; that order is part of the specification because it determines
    the rounding.

    Every reduction is ``fsum`` and every scalar step is a single correctly
    rounded operation, so the factor is reproducible on any conforming
    platform. Compare ``np.linalg.qr``, which is not.
    """
    rows = _rows(A)
    m = len(rows)
    n = len(rows[0]) if rows else 0
    if m < n:
        raise ValueError("QR requires at least as many rows as columns")
    cols = [[rows[i][j] for i in range(m)] for j in range(n)]
    rhs = _vec(y) if y is not None else []
    if rhs and len(rhs) != m:
        raise ValueError("right hand side length does not match the row count")

    for j in range(n):
        col = cols[j]
        norm = math.sqrt(math.fsum([col[i] * col[i] for i in range(j, m)]))
        if norm == 0.0:
            continue
        # Sign chosen away from col[j] so the subtraction never cancels.
        alpha = -norm if col[j] >= 0.0 else norm
        v = [0.0] * m
        for i in range(j, m):
            v[i] = col[i]
        v[j] = col[j] - alpha
        vtv = math.fsum([v[i] * v[i] for i in range(j, m)])
        if vtv == 0.0:
            continue
        for k in range(j, n):
            ck = cols[k]
            s = math.fsum([v[i] * ck[i] for i in range(j, m)])
            f = 2.0 * s / vtv
            for i in range(j, m):
                ck[i] = ck[i] - f * v[i]
        if rhs:
            s = math.fsum([v[i] * rhs[i] for i in range(j, m)])
            f = 2.0 * s / vtv
            for i in range(j, m):
                rhs[i] = rhs[i] - f * v[i]

    R = np.array(
        [[cols[j][i] if i <= j else 0.0 for j in range(n)] for i in range(n)],
        dtype=float,
    )
    qty = np.array(rhs[:n], dtype=float) if rhs else np.empty(0, dtype=float)
    return R, qty


def solve_upper(R, b) -> np.ndarray:
    """Back substitution against an upper triangular ``R``."""
    rr = _rows(R)
    bv = _vec(b)
    n = len(rr)
    if len(bv) != n:
        raise ValueError("right hand side length does not match R")
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = math.fsum([rr[i][k] * x[k] for k in range(i + 1, n)])
        pivot = rr[i][i]
        if pivot == 0.0:
            raise ZeroDivisionError("singular triangular factor at row %d" % i)
        x[i] = (bv[i] - s) / pivot
    return np.array(x, dtype=float)


def inv_upper(R) -> np.ndarray:
    """Inverse of an upper triangular matrix, one back substitution per column."""
    rr = _rows(R)
    n = len(rr)
    cols = []
    for j in range(n):
        e = [1.0 if i == j else 0.0 for i in range(n)]
        cols.append(solve_upper(rr, e))
    return np.array([[cols[j][i] for j in range(n)] for i in range(n)], dtype=float)


def rank(A) -> int:
    """Rank from the Householder QR diagonal.

    A column counts towards the rank when its diagonal entry exceeds
    ``max(m, n) * EPS * max_i |R_ii|``, the same shape of tolerance numpy
    applies to singular values in ``matrix_rank``. Without column pivoting this
    is a full rank *detector* rather than a general rank revealing
    factorisation, which is exactly what the callers need: they ask whether the
    design is deficient, and refuse to estimate when it is.

    tests/test_linalg.py asserts that this agrees with ``np.linalg.matrix_rank``
    on every fixture design in the repository, so the substitution is measured
    rather than assumed.
    """
    arr = np.asarray(A, dtype=float)
    m, n = arr.shape
    R, _ = qr_rhs(arr)
    diag = [abs(float(R[i][i])) for i in range(n)]
    largest = max(diag) if diag else 0.0
    if largest == 0.0:
        return 0
    tol = max(m, n) * EPS * largest
    return sum(1 for d in diag if d > tol)


def solve_ols(X, y) -> np.ndarray:
    """Least squares coefficients through the Householder QR."""
    R, qty = qr_rhs(X, y)
    return solve_upper(R, qty)


def xtx_inv(X) -> np.ndarray:
    """``(X'X)^-1`` as ``R^-1 R^-T`` from the QR factorisation of ``X``."""
    R, _ = qr_rhs(X)
    r_inv = inv_upper(R)
    return matmul(r_inv, r_inv.T)


def group_sums(inverse, values, n_groups: int) -> np.ndarray:
    """Exactly rounded sum of ``values`` within each group.

    Replaces ``np.bincount(inverse, weights=values)``. Membership is collected
    first and each group is then reduced with ``fsum``, so the result does not
    depend on how the accumulation is blocked.
    """
    idx = np.asarray(inverse).ravel()
    vals = _vec(values)
    if len(idx) != len(vals):
        raise ValueError("group index length does not match the value count")
    buckets: List[List[float]] = [[] for _ in range(n_groups)]
    for position, value in zip(idx, vals):
        buckets[int(position)].append(value)
    return np.array([math.fsum(bucket) for bucket in buckets], dtype=float)


def group_counts(inverse, n_groups: int) -> np.ndarray:
    """Number of members per group, counted in Python integers.

    Replaces ``np.bincount(inverse)``. Integer counting cannot round, so this
    is about keeping the reference path free of dispatched kernels rather than
    about accuracy.
    """
    counts = [0] * n_groups
    for position in np.asarray(inverse).ravel():
        counts[int(position)] += 1
    return np.array([float(c) for c in counts], dtype=float)
