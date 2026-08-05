"""Scenarios: each one demonstrates a way experiments go wrong, and the fix.

Every scenario follows the same contract:

    run(seed: int, **kwargs) -> dict

The returned dict is JSON-serializable and carries a `verdict` key with an
explicit PASS/FAIL, because a demonstration that only prints numbers leaves
the reader to decide whether the numbers are good. Stating the pass criterion
up front -- and letting it fail -- is the difference between a test and a demo.
"""

from . import aa_calibration, cuped, heterogeneity, novelty, peeking, power_curve, srm

REGISTRY = {
    "aa": aa_calibration,
    "power": power_curve,
    "peeking": peeking,
    "srm": srm,
    "heterogeneity": heterogeneity,
    "novelty": novelty,
    "cuped": cuped,
}

__all__ = ["REGISTRY"]
