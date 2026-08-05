"""synthetic-ab-lab -- a simulation harness for A/B test methodology.

Public surface:
    stats       distribution functions, two-sample tests, SRM check
    power       sample size, power, MDE
    sequential  mSPRT always-valid p-values and confidence sequences
    population  synthetic user generation with known ground truth
    experiment  assignment, measurement, and estimators
"""

from .power import mde_at_n, power_at_n, sample_size_proportions
from .sequential import MSPRT
from .stats import srm_check, two_proportion_ztest, welch_ttest

__version__ = "1.0.0"

__all__ = [
    "sample_size_proportions",
    "power_at_n",
    "mde_at_n",
    "MSPRT",
    "two_proportion_ztest",
    "welch_ttest",
    "srm_check",
]
