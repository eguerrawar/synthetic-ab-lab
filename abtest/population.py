"""The synthetic user population.

This is the "synthetic" in synthetic A/B testing. Rather than waiting weeks
for real traffic to learn whether an analysis method works, generate a
population where the ground truth is known by construction, run the method
against it, and check whether it recovers the answer.

That inversion is the whole point. On a real experiment the true lift is
unknown forever, so a wrong analysis is invisible. Here the true lift is a
parameter, so a wrong analysis is a failing test.

The population is deliberately not i.i.d. coin flips. Real traffic has:

  * segments with very different baseline rates (mobile converts worse than
    desktop; paid traffic worse than organic)
  * heterogeneous treatment effects -- the same feature helps one segment and
    hurts another
  * a pre-period covariate correlated with the outcome, which is what makes
    CUPED variance reduction possible
  * time structure, so novelty effects and day-of-week seasonality exist

Each of those exists because it breaks a naive analysis somewhere in
`scenarios/`. Nothing here is decoration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .rng import Rng
from .stats import norm_sf


@dataclass(frozen=True)
class Segment:
    """A user cohort with its own baseline rate and its own response."""

    name: str
    share: float  # fraction of the population
    baseline_rate: float  # conversion probability under control
    effect_multiplier: float = 1.0  # scales the global treatment effect

    def treated_rate(self, relative_effect: float) -> float:
        rate = self.baseline_rate * (1.0 + relative_effect * self.effect_multiplier)
        return min(max(rate, 1e-6), 1.0 - 1e-6)


@dataclass
class User:
    """One synthetic user. Attributes exist so the analysis can be tested on them."""

    user_id: int
    segment: str
    baseline_rate: float
    pre_period_value: float  # latent engagement, observed BEFORE assignment
    sessions_pre: float = 0.0  # continuous pre-period metric (CUPED covariate)
    day: int = 0
    arm: str = ""
    converted: int = 0
    revenue: float = 0.0
    sessions_post: float = 0.0  # continuous in-experiment metric


# Scale of the continuous engagement metric. Chosen so the numbers read like
# a plausible "sessions per week" rather than for any statistical reason.
SESSIONS_MEAN = 8.0
SESSIONS_SD = 3.0


# A default population loosely shaped like a consumer signup funnel.
DEFAULT_SEGMENTS = [
    Segment("mobile_organic", share=0.45, baseline_rate=0.052, effect_multiplier=1.4),
    Segment("mobile_paid", share=0.20, baseline_rate=0.031, effect_multiplier=0.6),
    Segment("desktop_organic", share=0.25, baseline_rate=0.118, effect_multiplier=1.0),
    Segment("desktop_paid", share=0.10, baseline_rate=0.086, effect_multiplier=0.3),
]


@dataclass
class PopulationSpec:
    """Ground truth for one simulated world."""

    segments: list[Segment] = field(default_factory=lambda: list(DEFAULT_SEGMENTS))
    true_relative_effect: float = 0.0  # 0.0 means this is an A/A test
    covariate_correlation: float = 0.60  # rho between pre-period value and outcome
    novelty_half_life_days: float | None = None  # None = stable effect
    n_days: int = 14

    def __post_init__(self):
        total = sum(s.share for s in self.segments)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"segment shares must sum to 1.0, got {total}")

    @property
    def blended_baseline(self) -> float:
        return sum(s.share * s.baseline_rate for s in self.segments)

    @property
    def blended_true_effect(self) -> float:
        """The population-average relative lift, accounting for heterogeneity.

        This is the number a correct analysis should recover -- NOT
        `true_relative_effect`, which is only the effect on a segment whose
        multiplier is 1.0. Conflating the two is a real and easy mistake, so
        the property exists to make the right answer unambiguous.
        """
        base = self.blended_baseline
        if base == 0:
            return 0.0
        treated = sum(
            s.share * s.treated_rate(self.true_relative_effect) for s in self.segments
        )
        return (treated - base) / base


def effect_on_day(spec: PopulationSpec, day: int) -> float:
    """The treatment effect actually in force on a given day.

    With a novelty half-life, the effect decays toward zero: users respond to
    the newness of a change, then habituate. An experiment stopped early --
    or a metric read only over its first three days -- overstates the durable
    lift, which is `scenarios/novelty.py`.
    """
    if spec.novelty_half_life_days is None:
        return spec.true_relative_effect
    return spec.true_relative_effect * (0.5 ** (day / spec.novelty_half_life_days))


def generate_users(spec: PopulationSpec, n: int, rng: Rng) -> list[User]:
    """Draw n users with segment membership, a pre-period covariate, and a day.

    The covariate is built so that its correlation with the eventual outcome
    is approximately `spec.covariate_correlation`. It is generated BEFORE
    assignment and is untouched by treatment, which is exactly the condition
    CUPED requires -- adjusting on a post-treatment variable would bias the
    estimate instead of just shrinking its variance.
    """
    names = [s.name for s in spec.segments]
    shares = [s.share for s in spec.segments]
    by_name = {s.name: s for s in spec.segments}

    # Draw all segment memberships in one call -- see Rng.choice_weighted_bulk.
    seg_names = rng.choice_weighted_bulk(names, shares, n)

    rho = spec.covariate_correlation
    w_shared = math.sqrt(rho)
    w_idio = math.sqrt(1.0 - rho)

    users: list[User] = []
    for i in range(n):
        seg = by_name[seg_names[i]]
        # Latent engagement: a standard normal reused when drawing outcomes,
        # which is what creates the pre/post correlation.
        latent = rng.normal()
        # Continuous pre-period metric. Weighting the shared latent by
        # sqrt(rho) and the idiosyncratic part by sqrt(1-rho) makes
        # corr(sessions_pre, sessions_post) exactly rho by construction, so
        # the CUPED variance reduction can be checked against rho^2 as an
        # exact prediction rather than a rough expectation.
        sessions_pre = SESSIONS_MEAN + SESSIONS_SD * (
            w_shared * latent + w_idio * rng.normal()
        )
        users.append(
            User(
                user_id=i,
                segment=seg.name,
                baseline_rate=seg.baseline_rate,
                pre_period_value=latent,
                sessions_pre=sessions_pre,
                day=int(rng.uniform() * spec.n_days),
            )
        )
    return users


def realize_outcomes(spec: PopulationSpec, users: list[User], rng: Rng) -> None:
    """Fill in `converted` and `revenue` for users that already have an arm.

    The conversion draw mixes the user's pre-period latent value with fresh
    noise, weighted by `covariate_correlation`. That is what makes the
    covariate predictive without making it deterministic.
    """
    rho = spec.covariate_correlation
    w_shared = math.sqrt(rho)
    w_idio = math.sqrt(1.0 - rho)
    by_name = {s.name: s for s in spec.segments}

    for u in users:
        seg = by_name[u.segment]
        treated = u.arm == "treatment"
        effect = effect_on_day(spec, u.day) if treated else 0.0
        rate = seg.treated_rate(effect) if treated else seg.baseline_rate

        # Continuous in-experiment metric, built from the same latent as
        # sessions_pre so that corr(pre, post) == spec.covariate_correlation.
        u.sessions_post = SESSIONS_MEAN * (1.0 + effect) + SESSIONS_SD * (
            w_shared * u.pre_period_value + w_idio * rng.normal()
        )

        # Correlated latent draw -> threshold at the segment's rate.
        combined = rho * u.pre_period_value + (1.0 - rho**2) ** 0.5 * rng.normal()
        # norm_sf maps the latent normal to a uniform, so thresholding at
        # `rate` keeps the marginal conversion probability exactly `rate`
        # while preserving correlation with the pre-period value. The upper
        # tail (sf, not cdf) is used so that a HIGHER latent means a MORE
        # engaged user: it makes the covariate positively correlated with
        # both conversion and sessions, which is what a real engagement
        # score looks like.
        u.converted = 1 if norm_sf(combined) < rate else 0
        # Revenue: heavy-tailed and only earned by converters. The long tail
        # is why revenue tests need far more traffic than conversion tests.
        u.revenue = rng.lognormal(3.2, 1.1) if u.converted else 0.0
