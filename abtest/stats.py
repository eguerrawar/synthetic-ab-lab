"""Statistical primitives, implemented from scratch on the standard library.

Everything an A/B test needs to produce a decision lives here: normal and
Student-t distribution functions, the two-sample tests, and confidence
intervals. No SciPy. Each function is validated against published reference
values in tests/test_stats.py, because a test harness whose statistics are
wrong is worse than no harness at all -- it produces confident, wrong answers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --------------------------------------------------------------------------
# Normal distribution
# --------------------------------------------------------------------------


def norm_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def norm_cdf(z: float) -> float:
    """P(Z <= z) for standard normal, via the error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def norm_sf(z: float) -> float:
    """Upper tail P(Z > z). Computed via erfc to stay accurate far out."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


# Acklam's rational approximation to the inverse normal CDF, then one
# Halley refinement step. Accurate to roughly 1e-15 after refinement.
_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_P_LOW = 0.02425


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (the quantile function)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf requires 0 < p < 1, got {p}")

    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    elif p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        x = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / (
            ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )

    # Halley refinement against the true CDF.
    e = norm_cdf(x) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


# --------------------------------------------------------------------------
# Incomplete beta / gamma -- needed for Student-t and chi-square tails
# --------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction expansion for the incomplete beta (Lentz's method)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + b * math.log1p(-x) + a * math.log(x)
    ) * _betacf(b, a, 1.0 - x) / b


def t_sf(t: float, df: float) -> float:
    """Upper tail P(T > t) for Student's t with df degrees of freedom."""
    if df <= 0:
        raise ValueError("df must be positive")
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2.0, 0.5, x)
    return tail if t > 0 else 1.0 - tail


def t_cdf(t: float, df: float) -> float:
    return 1.0 - t_sf(t, df)


def t_ppf(p: float, df: float) -> float:
    """Inverse Student-t CDF by bisection on t_cdf. Fast enough; used rarely."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"t_ppf requires 0 < p < 1, got {p}")
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _gammq(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x)."""
    if x < 0.0 or a <= 0.0:
        raise ValueError("invalid arguments to gammq")
    if x == 0.0:
        return 1.0
    if x < a + 1.0:
        # Series representation for P(a, x).
        ap = a
        total = 1.0 / a
        delta = total
        for _ in range(500):
            ap += 1.0
            delta *= x / ap
            total += delta
            if abs(delta) < abs(total) * 3e-16:
                break
        return 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # Continued fraction for Q(a, x).
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-16:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(x: float, df: int) -> float:
    """Upper tail P(X > x) for a chi-square with df degrees of freedom."""
    if x <= 0:
        return 1.0
    return _gammq(df / 2.0, x / 2.0)


# --------------------------------------------------------------------------
# Test results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TestResult:
    """The full readout of one comparison -- never just a p-value.

    A p-value alone cannot support a ship decision. Effect size says whether
    the win is worth the engineering cost; the interval says how precisely we
    know it. Every scenario in this repo reports all three.
    """

    control_mean: float
    treatment_mean: float
    absolute_lift: float
    relative_lift: float  # as a fraction, e.g. 0.043 == +4.3%
    standard_error: float
    statistic: float
    p_value: float
    ci_low: float  # CI on the ABSOLUTE lift
    ci_high: float
    n_control: int
    n_treatment: int
    alpha: float

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    @property
    def ci_low_relative(self) -> float:
        return self.ci_low / self.control_mean if self.control_mean else float("nan")

    @property
    def ci_high_relative(self) -> float:
        return self.ci_high / self.control_mean if self.control_mean else float("nan")

    def summary(self) -> str:
        verdict = "SIGNIFICANT" if self.significant else "not significant"
        return (
            f"control={self.control_mean:.4f} (n={self.n_control:,})  "
            f"treatment={self.treatment_mean:.4f} (n={self.n_treatment:,})  "
            f"lift={self.relative_lift:+.2%} "
            f"[{self.ci_low_relative:+.2%}, {self.ci_high_relative:+.2%}]  "
            f"p={self.p_value:.4f}  -> {verdict}"
        )


def two_proportion_ztest(
    conv_c: int, n_c: int, conv_t: int, n_t: int, alpha: float = 0.05
) -> TestResult:
    """Two-sided z-test for a difference in conversion rates.

    The p-value uses the POOLED standard error (correct under the null we are
    testing, which is that both arms share one rate). The confidence interval
    uses the UNPOOLED standard error, because under the alternative the rates
    differ and pooling would understate the width. Mixing these up is a
    common and quiet source of intervals that disagree with their own p-value.
    """
    if n_c <= 0 or n_t <= 0:
        raise ValueError("both arms need at least one unit")
    p_c = conv_c / n_c
    p_t = conv_t / n_t
    diff = p_t - p_c

    p_pool = (conv_c + conv_t) / (n_c + n_t)
    se_pooled = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_c + 1.0 / n_t))
    se_unpooled = math.sqrt(p_c * (1.0 - p_c) / n_c + p_t * (1.0 - p_t) / n_t)

    z = diff / se_pooled if se_pooled > 0 else 0.0
    p_value = 2.0 * norm_sf(abs(z))

    crit = norm_ppf(1.0 - alpha / 2.0)
    return TestResult(
        control_mean=p_c,
        treatment_mean=p_t,
        absolute_lift=diff,
        relative_lift=diff / p_c if p_c > 0 else float("nan"),
        standard_error=se_unpooled,
        statistic=z,
        p_value=p_value,
        ci_low=diff - crit * se_unpooled,
        ci_high=diff + crit * se_unpooled,
        n_control=n_c,
        n_treatment=n_t,
        alpha=alpha,
    )


def welch_ttest(
    mean_c: float,
    var_c: float,
    n_c: int,
    mean_t: float,
    var_t: float,
    n_t: int,
    alpha: float = 0.05,
) -> TestResult:
    """Welch's t-test for continuous metrics (revenue, session time, ...).

    Welch rather than Student because the two arms rarely share a variance --
    a treatment that lifts revenue almost always widens its spread too.
    """
    if n_c < 2 or n_t < 2:
        raise ValueError("Welch's t-test needs at least 2 units per arm")
    diff = mean_t - mean_c
    vc, vt = var_c / n_c, var_t / n_t
    se = math.sqrt(vc + vt)
    if se == 0:
        raise ValueError("zero variance in both arms; t-test undefined")

    t = diff / se
    # Welch-Satterthwaite effective degrees of freedom.
    df = (vc + vt) ** 2 / (vc**2 / (n_c - 1) + vt**2 / (n_t - 1))
    p_value = 2.0 * t_sf(abs(t), df)

    crit = t_ppf(1.0 - alpha / 2.0, df)
    return TestResult(
        control_mean=mean_c,
        treatment_mean=mean_t,
        absolute_lift=diff,
        relative_lift=diff / mean_c if mean_c else float("nan"),
        standard_error=se,
        statistic=t,
        p_value=p_value,
        ci_low=diff - crit * se,
        ci_high=diff + crit * se,
        n_control=n_c,
        n_treatment=n_t,
        alpha=alpha,
    )


def welch_ttest_from_samples(
    control: list[float], treatment: list[float], alpha: float = 0.05
) -> TestResult:
    def mean_var(xs: list[float]) -> tuple[float, float]:
        n = len(xs)
        m = sum(xs) / n
        v = sum((x - m) ** 2 for x in xs) / (n - 1)
        return m, v

    mc, vc = mean_var(control)
    mt, vt = mean_var(treatment)
    return welch_ttest(mc, vc, len(control), mt, vt, len(treatment), alpha)


def srm_check(counts: list[int], expected_ratio: list[float], alpha: float = 0.001) -> dict:
    """Sample Ratio Mismatch: chi-square goodness-of-fit on arm assignment counts.

    If a coin flip meant to split traffic 50/50 delivers 50.4/49.6 on ten
    million users, the randomizer is broken -- and a broken randomizer means
    the two arms are not exchangeable, so every downstream number is suspect.
    Alpha is 0.001 rather than 0.05 by convention: this check runs on every
    experiment, so it needs a low false-alarm rate to stay trustworthy.
    """
    total = sum(counts)
    expected = [total * r for r in expected_ratio]
    chi2 = sum((o - e) ** 2 / e for o, e in zip(counts, expected) if e > 0)
    df = len(counts) - 1
    p = chi2_sf(chi2, df)
    return {
        "chi2": chi2,
        "df": df,
        "p_value": p,
        "observed": list(counts),
        "expected": expected,
        "observed_ratio": [c / total for c in counts],
        "failed": p < alpha,
        "alpha": alpha,
    }
