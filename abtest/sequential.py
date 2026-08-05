"""Always-valid inference: the fix for continuous monitoring.

The problem. A fixed-horizon t-test controls the false positive rate at one
pre-committed sample size. Look at the dashboard every day and stop the moment
p < 0.05, and the actual false positive rate climbs toward 25-30% -- because
each look is another chance for noise to cross the line. Nobody in growth has
ever actually waited two weeks without looking, so the theory has to change
rather than the humans.

The fix. A mixture Sequential Probability Ratio Test (mSPRT), the approach
Optimizely and others adopted for exactly this reason. Instead of testing the
null against one specific alternative, mix over a prior on the effect:

    delta ~ Normal(theta_0, tau^2)

and track the likelihood ratio of "some effect" against "no effect":

    Lambda_n = sqrt(V_n / (V_n + tau^2))
               * exp( tau^2 * (delta_hat - theta_0)^2
                      / (2 * V_n * (V_n + tau^2)) )

where V_n is the variance of the treatment-effect estimate at time n.
Lambda_n is a non-negative martingale under the null, so Ville's inequality
gives P(sup_n Lambda_n >= 1/alpha) <= alpha. The running minimum of
1/Lambda_n is therefore a p-value that is valid at EVERY n simultaneously --
it can be watched continuously, and stopping the moment it drops below alpha
still controls the false positive rate at alpha.

The price is real and worth stating out loud: to buy the freedom to peek, a
sequential test needs more samples than a fixed-horizon test to reach the same
power. `scenarios/peeking.py` measures both sides of that trade.

Reference: Johari, Pekelis & Walsh, "Always Valid Inference: Continuous
Monitoring of A/B Tests" (2015/2021).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SequentialState:
    """One always-valid readout at a point in time."""

    n_control: int
    n_treatment: int
    estimate: float  # delta_hat, the observed absolute lift
    variance: float  # V_n, variance of delta_hat
    log_lambda: float
    p_value: float  # always-valid: running min, safe to watch continuously
    ci_low: float
    ci_high: float

    @property
    def decisive(self) -> bool:
        """True once the confidence sequence excludes zero."""
        return self.ci_low > 0.0 or self.ci_high < 0.0


class MSPRT:
    """Mixture SPRT for a difference in means or proportions.

    tau is the prior standard deviation on the true effect, and it is the one
    real tuning knob. It should be set to the size of effect the team actually
    expects to ship -- roughly the MDE the experiment was powered for. Set it
    far too large and the test loses power against small effects; far too
    small and it takes forever to accumulate evidence for large ones. It does
    NOT affect validity: the false positive guarantee holds for any tau > 0.
    """

    def __init__(self, tau: float, alpha: float = 0.05, theta_0: float = 0.0):
        if tau <= 0:
            raise ValueError("tau must be positive")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.tau = tau
        self.alpha = alpha
        self.theta_0 = theta_0
        self._min_p = 1.0
        self.history: list[SequentialState] = []

    def update(
        self,
        mean_c: float,
        var_c: float,
        n_c: int,
        mean_t: float,
        var_t: float,
        n_t: int,
    ) -> SequentialState:
        """Feed in the running summary statistics; get an always-valid readout."""
        if n_c < 2 or n_t < 2:
            raise ValueError("need at least 2 units per arm")

        delta = mean_t - mean_c
        v = var_c / n_c + var_t / n_t  # V_n
        if v <= 0:
            v = 1e-12

        tau2 = self.tau**2
        centered = delta - self.theta_0

        log_lambda = 0.5 * math.log(v / (v + tau2)) + (
            tau2 * centered**2 / (2.0 * v * (v + tau2))
        )

        # Always-valid p-value: 1/Lambda, floored by its own running minimum
        # so that it is monotone and can be compared to alpha at any moment.
        p_now = min(1.0, math.exp(-log_lambda))
        self._min_p = min(self._min_p, p_now)

        half = self._confidence_half_width(v)
        state = SequentialState(
            n_control=n_c,
            n_treatment=n_t,
            estimate=delta,
            variance=v,
            log_lambda=log_lambda,
            p_value=self._min_p,
            ci_low=delta - half,
            ci_high=delta + half,
        )
        self.history.append(state)
        return state

    def update_proportions(
        self, conv_c: int, n_c: int, conv_t: int, n_t: int
    ) -> SequentialState:
        """Convenience wrapper for binary metrics (variance is p(1-p))."""
        p_c = conv_c / n_c
        p_t = conv_t / n_t
        return self.update(p_c, p_c * (1 - p_c), n_c, p_t, p_t * (1 - p_t), n_t)

    def _confidence_half_width(self, v: float) -> float:
        """Half-width of the always-valid confidence sequence.

        Derived by inverting the test: collect every theta_0 that the mixture
        statistic never rejects. Solving Lambda_n <= 1/alpha for theta_0 gives

            half = sqrt( V(V + tau^2)/tau^2 * log( (V + tau^2) / (V * alpha^2) ) )

        This sequence covers the true effect at all times simultaneously with
        probability 1 - alpha, unlike a fixed-horizon CI which only covers at
        the one sample size it was computed for.
        """
        tau2 = self.tau**2
        return math.sqrt((v * (v + tau2) / tau2) * math.log((v + tau2) / (v * self.alpha**2)))

    def reset(self) -> None:
        self._min_p = 1.0
        self.history = []


def fixed_horizon_peek_risk(n_looks: int, alpha: float = 0.05) -> float:
    """Rough upper bound on the inflated false positive rate from peeking.

    Assumes independent looks, which they are not -- consecutive looks share
    most of their data, so the true inflation is lower than this bound. Useful
    only as an order-of-magnitude sanity check against the simulated number,
    which is the one that should be believed.
    """
    return 1.0 - (1.0 - alpha) ** n_looks
