"""The reference kernel must be a pure function of its inputs.

`docs/determinism.md` records what CI measured: the same source, the same
numpy, the same wheel and the same fixture produced two different values for
`adf-001 params.const.estimate` depending on which CPU the runner landed on
and on how many threads the linear algebra backend was allowed to use. Pinning
the backend to one thread did not close it. Two runners pinned to one thread
disagreed with each other.

The fix was to stop asking a dispatched kernel for a sum. Every reduction on
the reference path now goes through `econospec.linalg`, which accumulates with
`math.fsum`. `math.fsum` is exactly rounded, so its result is the correctly
rounded sum of the inputs and does not depend on the order it visits them,
the width of the vector unit, or the microarchitecture underneath.

These tests are the standing guard on that property. The first group checks
the primitives directly. The second group checks that the estimators never
reach past them, because a single `@` reinstated in a hot path would restore
the divergence without failing any accuracy test in the suite.
"""

from __future__ import annotations

import ast
import math
import os
import random

import numpy as np
import pytest

from econospec import estimators, linalg as la

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL_FILES = [
    os.path.join(HERE, "econospec", "linalg.py"),
    os.path.join(HERE, "econospec", "estimators.py"),
]


# ---------------------------------------------------------------------------
# the primitives
# ---------------------------------------------------------------------------


def test_fsum_total_is_exactly_rounded_where_naive_summation_cancels():
    """The property being bought, on the smallest case that shows it.

    Left to right accumulation of this sequence loses the 1.0 entirely, and
    the same three values in a different order give a different answer. An
    exactly rounded sum returns 1.0 whatever order it is handed.
    """
    values = [1e16, 1.0, -1e16]
    naive = 0.0
    for value in values:
        naive += value
    assert naive == 0.0
    assert la.total(values) == 1.0
    for _ in range(20):
        shuffled = list(values)
        random.shuffle(shuffled)
        assert la.total(shuffled) == 1.0


def test_dot_does_not_depend_on_the_order_of_the_terms():
    """Bit for bit, not within a tolerance. A tolerance would hide the bug."""
    rng = np.random.Generator(np.random.PCG64(11))
    for trial in range(50):
        n = int(rng.integers(2, 400))
        a = rng.normal(scale=10.0 ** rng.integers(-8, 8), size=n)
        b = rng.normal(scale=10.0 ** rng.integers(-8, 8), size=n)
        reference = la.dot(a, b)
        assert la.dot(a[::-1], b[::-1]) == reference, trial
        order = rng.permutation(n)
        assert la.dot(a[order], b[order]) == reference, trial


def test_dot_agrees_with_an_exact_rational_evaluation():
    """An independent route: the products summed exactly in rationals.

    The products themselves are rounded, exactly as IEEE-754 multiplication
    rounds them, so this measures the accumulation and nothing else. Summing
    them in `Fraction` and rounding once at the end is the correctly rounded
    sum by definition. `dot` has to hit it on the nose, not near it.
    """
    from fractions import Fraction

    rng = np.random.Generator(np.random.PCG64(12))
    for _ in range(25):
        a = rng.normal(size=64)
        b = rng.normal(size=64)
        exact = sum(
            (Fraction(float(x) * float(y)) for x, y in zip(a, b)), Fraction(0)
        )
        assert la.dot(a, b) == float(exact)


def test_qr_reproduces_the_normal_equations():
    rng = np.random.Generator(np.random.PCG64(13))
    x = rng.normal(size=(120, 5))
    y = rng.normal(size=120)
    r, _ = la.qr_rhs(x, y)
    rebuilt = la.crossprod(r, r)
    truth = la.crossprod(x, x)
    assert np.allclose(rebuilt, truth, rtol=0, atol=1e-9)


def test_solve_ols_matches_lstsq():
    rng = np.random.Generator(np.random.PCG64(14))
    x = rng.normal(size=(200, 4))
    y = rng.normal(size=200)
    ours = la.solve_ols(x, y)
    theirs = np.linalg.lstsq(x, y, rcond=None)[0]
    assert np.allclose(ours, theirs, rtol=0, atol=1e-11)


def test_xtx_inv_matches_the_dispatched_inverse():
    rng = np.random.Generator(np.random.PCG64(15))
    x = rng.normal(size=(150, 3))
    ours = la.xtx_inv(x)
    theirs = np.linalg.inv(x.T @ x)
    assert np.allclose(ours, theirs, rtol=0, atol=1e-10)


def test_deterministic_rank_agrees_with_numpy_matrix_rank():
    """Forty designs, including deficient ones, checked against numpy's SVD.

    `linalg.rank` decides from the QR diagonal rather than from singular
    values. The two are different criteria, so the agreement is measured here
    rather than assumed. If it ever stops holding, this test says so before a
    rank deficient design is silently fitted.
    """
    rng = np.random.Generator(np.random.PCG64(16))
    for trial in range(40):
        n = int(rng.integers(12, 90))
        k = int(rng.integers(2, 6))
        x = rng.normal(size=(n, k))
        if trial % 3 == 0:
            x[:, -1] = x[:, 0]
        if trial % 7 == 0:
            x[:, -1] = x[:, 0] * 2.0 - x[:, 1]
        assert la.rank(x) == int(np.linalg.matrix_rank(x)), trial


def test_group_sums_and_counts_match_bincount():
    rng = np.random.Generator(np.random.PCG64(17))
    codes = rng.integers(0, 9, size=400)
    values = rng.normal(size=400)
    assert np.allclose(
        la.group_sums(codes, values, 9),
        np.bincount(codes, weights=values, minlength=9),
        rtol=0,
        atol=1e-12,
    )
    assert np.array_equal(
        la.group_counts(codes, 9), np.bincount(codes, minlength=9).astype(float)
    )


def test_leverage_matches_the_hat_matrix_diagonal():
    rng = np.random.Generator(np.random.PCG64(18))
    x = rng.normal(size=(80, 4))
    inv = la.xtx_inv(x)
    theirs = np.einsum("ij,jk,ik->i", x, inv, x)
    assert np.allclose(la.leverage(x, inv), theirs, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# the estimators must not reach past the primitives
# ---------------------------------------------------------------------------

FORBIDDEN_NUMPY_NAMES = {
    "linalg",
    "dot",
    "vdot",
    "inner",
    "outer",
    "matmul",
    "tensordot",
    "einsum",
    "sum",
    "nansum",
    "prod",
    "cumsum",
    "mean",
    "average",
    "var",
    "std",
    "trace",
    "bincount",
    "cov",
    "corrcoef",
}
FORBIDDEN_METHODS = {
    "dot",
    "sum",
    "prod",
    "cumsum",
    "mean",
    "var",
    "std",
    "trace",
}
OUR_MODULES = {"la", "linalg", "math"}


def _dispatched_reductions(path: str):
    """Every place the file asks numpy to reduce something.

    Parsed, not grepped, so that the prose in the docstrings can name the
    functions it replaced without tripping the guard.
    """
    with open(path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            found.append("matmul operator '@' on line %d" % node.lineno)
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.MatMult):
            found.append("matmul operator '@=' on line %d" % node.lineno)
        elif isinstance(node, ast.Attribute):
            base = node.value
            if isinstance(base, ast.Name):
                if base.id in ("np", "numpy") and node.attr in FORBIDDEN_NUMPY_NAMES:
                    found.append("np.%s on line %d" % (node.attr, node.lineno))
                elif base.id not in OUR_MODULES and node.attr in FORBIDDEN_METHODS:
                    found.append("%s.%s on line %d" % (base.id, node.attr, node.lineno))
            elif node.attr in FORBIDDEN_METHODS:
                found.append("<expr>.%s on line %d" % (node.attr, node.lineno))
    return found


@pytest.mark.parametrize("path", KERNEL_FILES, ids=lambda p: os.path.basename(p))
def test_the_reference_kernel_contains_no_dispatched_reduction(path):
    """The regression guard for the whole episode.

    Any of these would hand a sum back to a kernel that is free to choose its
    own accumulation order, which is what made the golden manifest comparison
    machine dependent in the first place. Accuracy tests do not catch it: the
    values stay correct to fourteen digits either way. Only exact equality
    across machines catches it, and by then it is a red run on somebody else's
    pull request.
    """
    found = _dispatched_reductions(path)
    assert found == [], (
        "%s reduces through a dispatched kernel: %s. Route it through "
        "econospec.linalg instead, or the bit for bit claim in "
        "tests/test_conformance.py stops being true on some machine nobody "
        "here owns." % (os.path.basename(path), "; ".join(found))
    )


def test_the_guard_would_actually_fire(tmp_path):
    """A guard nobody has seen fail is a guard nobody has tested."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import numpy as np\n"
        "def f(a, b):\n"
        "    '''np.linalg.qr and a @ b are named here in prose only.'''\n"
        "    return a @ b, np.linalg.solve(a, b), a.sum(), np.einsum('i,i->', b, b)\n",
        encoding="utf-8",
    )
    found = _dispatched_reductions(str(sample))
    assert len(found) == 4, found


def test_the_estimators_still_work_when_numpys_solvers_are_removed(monkeypatch):
    """The runtime half of the same claim, in case the parser misses a route.

    If any estimator still reaches for a factorisation, one of these raises
    and the test fails with the name of the function that was called.
    """

    def refuse(name):
        def inner(*args, **kwargs):
            raise AssertionError(
                "the reference path called np.linalg.%s, which reduces in an "
                "order the platform chooses" % name
            )

        return inner

    for name in ("qr", "solve", "inv", "lstsq", "pinv", "matrix_rank", "svd", "eig"):
        monkeypatch.setattr(np.linalg, name, refuse(name))
    monkeypatch.setattr(np, "einsum", refuse("einsum"))
    monkeypatch.setattr(np, "dot", refuse("dot"))

    rng = np.random.Generator(np.random.PCG64(19))
    n = 240
    x = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    y = 1.0 + x[:, 1] * 0.5 - x[:, 2] * 0.25 + rng.normal(size=n)
    names = ["const", "x1", "x2"]

    for vcov in estimators.VCOV_TYPES:
        out = estimators.ols(y, x, names, vcov_type=vcov)
        assert math.isfinite(out["params"]["x1"]["estimate"])
        assert out["params"]["x1"]["se"] > 0

    entity = np.repeat(np.arange(24), 10)
    fe = estimators.panel_fe(y, x[:, 1:], entity, ["x1", "x2"])
    assert math.isfinite(fe["params"]["x1"]["estimate"])

    w = rng.normal(size=n)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    u = rng.normal(size=n)
    endog = 0.8 * z1 + 0.6 * z2 + 0.4 * w + 0.7 * u
    yiv = 1.0 + 0.5 * w + 1.2 * endog + u
    xiv = np.column_stack([np.ones(n), w, endog])
    ziv = np.column_stack([np.ones(n), w, z1, z2])
    iv = estimators.iv2sls(
        yiv,
        xiv,
        ziv,
        ["const", "w", "d"],
        n_endog=1,
        n_excluded=2,
        endog=endog,
        Z_exog=np.column_stack([np.ones(n), w]),
    )
    assert math.isfinite(iv["params"]["d"]["estimate"])

    series = np.empty(n)
    noise = rng.normal(size=n)
    series[0] = noise[0]
    for i in range(1, n):
        series[i] = 0.6 * series[i - 1] + noise[i]
    adf = estimators.adf(series, lags=2, trend="c")
    assert math.isfinite(adf["scalars"]["adf_stat"])


def test_repeated_fits_in_one_process_are_bit_identical():
    """Cheap, and it is the exact assertion the conformance suite makes."""
    rng = np.random.Generator(np.random.PCG64(20))
    n = 300
    x = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    y = rng.normal(size=n)
    names = ["const", "x1", "x2"]
    first = estimators.ols(y, x, names, vcov_type="HC1")
    for _ in range(9):
        again = estimators.ols(y, x, names, vcov_type="HC1")
        for key in first["params"]:
            for field in ("estimate", "se", "tstat"):
                assert again["params"][key][field] == first["params"][key][field]
