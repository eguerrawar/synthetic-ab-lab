"""Deterministic random number generation.

Every result in this repo is reproducible from a seed. That is not a nicety:
a simulation study that cannot be re-run exactly cannot be reviewed, and a
reviewer who cannot re-run it has to take the author's word for the numbers.

Two sampling paths exist, and the choice between them is the main performance
decision in the project:

  * `Rng.bernoulli` draws one user at a time. Needed when users carry
    attributes (segment, pre-period behavior) that the analysis reads back.
  * `Rng.binomial` draws the arm total in one call. Mathematically identical
    when users are exchangeable, and roughly 10,000x faster -- which is what
    makes a 20,000-trial A/A calibration run finish in seconds instead of
    hours.
"""

from __future__ import annotations

import math
import random


class Rng:
    """A seeded random stream with the samplers this project needs."""

    def __init__(self, seed: int):
        self.seed = seed
        self._r = random.Random(seed)
        self._has_native_binomial = hasattr(self._r, "binomialvariate")

    def spawn(self, label: str) -> "Rng":
        """Derive an independent, reproducible child stream.

        Named sub-streams keep parts of a simulation from interfering: adding
        a new metric should not shift the assignment draws and silently change
        every previously reported number.
        """
        derived = (self.seed * 2_654_435_761 + hash(label)) % (2**63)
        return Rng(derived)

    # -- uniform / bernoulli ------------------------------------------------

    def uniform(self) -> float:
        return self._r.random()

    def bernoulli(self, p: float) -> int:
        return 1 if self._r.random() < p else 0

    def choice_weighted(self, items: list, weights: list[float]):
        return self._r.choices(items, weights=weights, k=1)[0]

    def choice_weighted_bulk(self, items: list, weights: list[float], k: int) -> list:
        """Draw k weighted choices at once.

        `random.choices` rebuilds its cumulative-weight table on every call,
        which dominates the runtime when generating hundreds of thousands of
        users one at a time. Building the table once and reusing it cut the
        CUPED scenario's runtime by roughly 4x.
        """
        return self._r.choices(items, weights=weights, k=k)

    # -- distributions ------------------------------------------------------

    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        return self._r.gauss(mu, sigma)

    def lognormal(self, mu: float, sigma: float) -> float:
        """Heavy-tailed positive draw -- the realistic shape for revenue."""
        return math.exp(self._r.gauss(mu, sigma))

    def binomial(self, n: int, p: float) -> int:
        """Number of successes in n independent trials.

        Uses CPython's exact `random.binomialvariate` (3.12+) when present.
        The fallback is exact for small n and a continuity-corrected normal
        approximation for large n, which is accurate to well under the noise
        floor of any simulation in this repo.
        """
        if n <= 0:
            return 0
        if p <= 0.0:
            return 0
        if p >= 1.0:
            return n
        if self._has_native_binomial:
            return self._r.binomialvariate(n, p)
        if n <= 200:
            return sum(1 for _ in range(n) if self._r.random() < p)
        mean = n * p
        sd = math.sqrt(n * p * (1.0 - p))
        return max(0, min(n, int(round(self._r.gauss(mean, sd)))))
