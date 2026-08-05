"""Power analysis: how many users, for how long, to detect what.

This is the single most valuable calculation in growth engineering, because
it is the one that happens BEFORE the experiment. An underpowered test does
not return "no result" -- it returns a coin flip dressed up as evidence, and
it burns two weeks of traffic doing it.

The vocabulary:
  alpha  false positive rate. Ship something that does nothing. Usually 0.05.
  power  1 - beta. Probability of detecting a real effect of the stated size.
         0.80 is the convention: one in five real wins is still missed.
  MDE    minimum detectable effect. The smallest lift the test can reliably
         see at the given sample size. Effects below the MDE are invisible
         no matter how the results are stared at.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .stats import norm_cdf, norm_ppf


@dataclass(frozen=True)
class PowerPlan:
    baseline_rate: float
    relative_mde: float
    alpha: float
    power: float
    n_per_arm: int
    n_total: int
    days_required: float | None = None
    daily_traffic: int | None = None

    def summary(self) -> str:
        lines = [
            f"  baseline conversion   {self.baseline_rate:.2%}",
            f"  target relative MDE   {self.relative_mde:+.1%} "
            f"({self.baseline_rate:.2%} -> {self.baseline_rate * (1 + self.relative_mde):.2%})",
            f"  alpha / power         {self.alpha} / {self.power:.0%}",
            f"  users per arm         {self.n_per_arm:,}",
            f"  users total           {self.n_total:,}",
        ]
        if self.days_required is not None:
            lines.append(
                f"  runtime at {self.daily_traffic:,}/day  {self.days_required:.1f} days"
            )
        return "\n".join(lines)


def sample_size_proportions(
    baseline_rate: float,
    relative_mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    daily_traffic: int | None = None,
) -> PowerPlan:
    """Users per arm needed to detect `relative_mde` on a conversion rate.

    Uses the unpooled two-proportion formula:

        n = (z_(1-a/2) * sqrt(2*p_bar*(1-p_bar)) + z_power * sqrt(p1*q1 + p2*q2))^2 / d^2

    which accounts for the fact that the treatment arm's variance differs from
    control's under the alternative. The simpler pooled formula understates n
    for large effects.
    """
    if not 0.0 < baseline_rate < 1.0:
        raise ValueError("baseline_rate must be strictly between 0 and 1")
    if relative_mde == 0:
        raise ValueError("relative_mde must be non-zero")

    p1 = baseline_rate
    p2 = baseline_rate * (1.0 + relative_mde)
    if not 0.0 < p2 < 1.0:
        raise ValueError(f"MDE pushes treatment rate to {p2:.4f}, outside (0, 1)")

    d = abs(p2 - p1)
    p_bar = (p1 + p2) / 2.0

    z_alpha = norm_ppf(1.0 - alpha / 2.0)
    z_power = norm_ppf(power)

    n = (
        z_alpha * math.sqrt(2.0 * p_bar * (1.0 - p_bar))
        + z_power * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    ) ** 2 / d**2
    n_per_arm = int(math.ceil(n))

    days = None
    if daily_traffic:
        days = (2 * n_per_arm) / daily_traffic

    return PowerPlan(
        baseline_rate=baseline_rate,
        relative_mde=relative_mde,
        alpha=alpha,
        power=power,
        n_per_arm=n_per_arm,
        n_total=2 * n_per_arm,
        days_required=days,
        daily_traffic=daily_traffic,
    )


def power_at_n(
    baseline_rate: float, relative_mde: float, n_per_arm: int, alpha: float = 0.05
) -> float:
    """Analytic power of a two-proportion z-test at a given sample size.

    The scenario `power_curve` checks this function against the empirical
    rejection rate of thousands of simulated experiments. If the curves match,
    both the formula and the simulator are working; if they diverge, one of
    them is wrong and the whole harness is suspect.
    """
    p1 = baseline_rate
    p2 = baseline_rate * (1.0 + relative_mde)
    d = abs(p2 - p1)
    if d == 0 or n_per_arm < 1:
        return alpha

    p_bar = (p1 + p2) / 2.0
    se_null = math.sqrt(2.0 * p_bar * (1.0 - p_bar) / n_per_arm)
    se_alt = math.sqrt((p1 * (1.0 - p1) + p2 * (1.0 - p2)) / n_per_arm)
    z_alpha = norm_ppf(1.0 - alpha / 2.0)

    # Two-sided: sum both rejection regions (the far tail is negligible but
    # cheap to include, and matters for tiny effects).
    upper = 1.0 - norm_cdf((z_alpha * se_null - d) / se_alt)
    lower = norm_cdf((-z_alpha * se_null - d) / se_alt)
    return upper + lower


def mde_at_n(
    baseline_rate: float,
    n_per_arm: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Invert the power calculation: what is the smallest visible lift at this n?

    Answers the question that actually gets asked in planning meetings --
    "we have two weeks of traffic, what can we learn?" -- by bisecting on the
    power function rather than using a closed form.
    """
    lo, hi = 1e-6, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if baseline_rate * (1.0 + mid) >= 1.0:
            hi = mid
            continue
        if power_at_n(baseline_rate, mid, n_per_arm, alpha) < power:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def sample_size_with_cuped(
    baseline_rate: float,
    relative_mde: float,
    correlation: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> tuple[int, int, float]:
    """Sample size before and after CUPED variance reduction.

    CUPED removes a `rho^2` fraction of the metric variance by regressing out
    a pre-experiment covariate, so the required n scales by (1 - rho^2).
    Returns (n_plain, n_cuped, fraction_saved).
    """
    plain = sample_size_proportions(baseline_rate, relative_mde, alpha, power).n_per_arm
    reduced = int(math.ceil(plain * (1.0 - correlation**2)))
    return plain, reduced, 1.0 - reduced / plain
