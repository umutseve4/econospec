"""Tests for the hand written distribution functions.

Every assertion here compares against something that can be checked without
this repository: a closed form, a published quantile, or an identity. Nothing
is compared against a previous run of the same code.
"""

from __future__ import annotations

import math

from econospec.distributions import (
    betainc_regularized,
    normal_cdf,
    normal_two_sided_p,
    student_t_cdf,
    student_t_two_sided_p,
)


def test_t_cdf_df1_matches_cauchy_closed_form():
    # With one degree of freedom the t distribution is standard Cauchy:
    # F(t) = 0.5 + arctan(t) / pi.
    for t in (-4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 4.0, 17.5):
        expected = 0.5 + math.atan(t) / math.pi
        assert abs(student_t_cdf(t, 1) - expected) < 1e-13, t


def test_t_cdf_df2_matches_closed_form():
    # With two degrees of freedom F(t) = 0.5 + t / (2 * sqrt(2 + t^2)).
    for t in (-9.0, -2.5, -0.5, 0.0, 0.5, 2.5, 9.0):
        expected = 0.5 + t / (2.0 * math.sqrt(2.0 + t * t))
        assert abs(student_t_cdf(t, 2) - expected) < 1e-13, t


def test_t_cdf_df4_matches_closed_form():
    # With four degrees of freedom, substituting t = 2 tan(theta) gives
    # F(t) = 0.5 + (3/8) x - x^3 / 32 with x = t / sqrt(1 + t^2/4).
    # Sanity check of the constants: as t grows, x tends to 2 and F tends to
    # 0.5 + 0.75 - 0.25 = 1.
    for t in (-30.0, -3.0, -1.0, 0.0, 1.0, 3.0, 30.0):
        x = t / math.sqrt(1.0 + t * t / 4.0)
        expected = 0.5 + 0.375 * x - x ** 3 / 32.0
        assert abs(student_t_cdf(t, 4) - expected) < 1e-13, t


def test_published_quantiles():
    # Standard tabulated two sided 5 percent critical values.
    published = {
        1: 12.706204736174698,
        5: 2.570581835636197,
        10: 2.228138851986273,
        30: 2.042272456301238,
        100: 1.983971518523590,
    }
    for df, q in published.items():
        assert abs(student_t_two_sided_p(q, df) - 0.05) < 1e-12, df
        assert abs(student_t_cdf(q, df) - 0.975) < 1e-12, df


def test_t_cdf_is_symmetric():
    for df in (3, 7, 25, 240):
        for t in (0.3, 1.1, 2.7, 6.5):
            assert abs(student_t_cdf(-t, df) - (1.0 - student_t_cdf(t, df))) < 1e-14


def test_two_sided_p_agrees_with_cdf_in_the_moderate_range():
    for df in (5, 20, 297):
        for t in (0.1, 1.0, 2.0, 3.5):
            direct = student_t_two_sided_p(t, df)
            viacdf = 2.0 * (1.0 - student_t_cdf(t, df))
            assert abs(direct - viacdf) < 1e-12, (df, t)


def test_two_sided_p_survives_the_far_tail():
    # The naive 2 * (1 - cdf) form cancels to zero here; the direct incomplete
    # beta evaluation must still return a positive number with real digits.
    p = student_t_two_sided_p(30.0, 237)
    assert 0.0 < p < 1e-80
    assert math.isfinite(math.log10(p))


def test_t_converges_to_normal_as_df_grows():
    for t in (0.5, 1.5, 2.5):
        assert abs(student_t_cdf(t, 2_000_000) - normal_cdf(t)) < 1e-6, t


def test_normal_cdf_published_points():
    assert abs(normal_cdf(0.0) - 0.5) < 1e-15
    assert abs(normal_cdf(1.959963984540054) - 0.975) < 1e-14
    assert abs(normal_cdf(-2.575829303548901) - 0.005) < 1e-14
    assert abs(normal_two_sided_p(1.959963984540054) - 0.05) < 1e-14


def test_incomplete_beta_identities():
    # I_x(a, b) = 1 - I_{1-x}(b, a), and I_x(1, 1) = x.
    for a, b, x in ((0.5, 2.5, 0.3), (3.0, 7.0, 0.62), (120.0, 0.5, 0.91)):
        left = betainc_regularized(a, b, x)
        right = 1.0 - betainc_regularized(b, a, 1.0 - x)
        assert abs(left - right) < 1e-13, (a, b, x)
    for x in (0.0, 0.25, 0.5, 1.0):
        assert abs(betainc_regularized(1.0, 1.0, x) - x) < 1e-15


def test_incomplete_beta_rejects_bad_input():
    for args in ((0.5, 1.0, 1.5), (0.0, 1.0, 0.5), (1.0, -1.0, 0.5)):
        try:
            betainc_regularized(*args)
        except ValueError:
            continue
        raise AssertionError("expected ValueError for %r" % (args,))
