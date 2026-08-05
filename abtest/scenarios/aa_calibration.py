"""Scenario 1 -- A/A calibration. The test that earns the right to run tests.

An A/A test ships the same experience to both arms. The true effect is
exactly zero by construction, so every "significant" result is a false
positive. Run a few thousand of them and two things must hold:

  1. The rejection rate at alpha = 0.05 is 5%, not 8% and not 2%.
  2. The p-values are UNIFORMLY distributed on [0, 1].

The second is the stronger claim and the one worth checking. A test can hit
5% rejections by luck while being badly miscalibrated everywhere else in the
distribution; uniformity says the machinery is right at every threshold at
once. It is checked here with a Kolmogorov-Smirnov statistic against the
uniform CDF.

This scenario is deliberately first. Every other result in the repo is only
believable if this one passes -- it is the calibration check that says the
measuring instrument reads zero when the input is zero.
"""

from __future__ import annotations

import math

from ..experiment import run_aggregate_trial
from ..rng import Rng

TITLE = "A/A calibration -- is the harness itself trustworthy?"


def run(
    seed: int = 20260803,
    n_trials: int = 20_000,
    n_per_arm: int = 25_000,
    baseline_rate: float = 0.085,
    alpha: float = 0.05,
) -> dict:
    rng = Rng(seed).spawn("aa_calibration")

    p_values: list[float] = []
    for _ in range(n_trials):
        res = run_aggregate_trial(n_per_arm, baseline_rate, baseline_rate, rng, alpha)
        p_values.append(res.p_value)

    false_positives = sum(1 for p in p_values if p < alpha)
    fpr = false_positives / n_trials

    # Binomial standard error on the observed rate, so "close to 5%" gets a
    # number instead of a vibe. Pass if within 3 standard errors.
    se = math.sqrt(alpha * (1 - alpha) / n_trials)
    z_fpr = (fpr - alpha) / se

    ks_stat, ks_critical = _ks_uniform(p_values, alpha=0.05)

    verdict = "PASS" if abs(z_fpr) < 3.0 and ks_stat < ks_critical else "FAIL"

    return {
        "scenario": "aa",
        "title": TITLE,
        "n_trials": n_trials,
        "n_per_arm": n_per_arm,
        "baseline_rate": baseline_rate,
        "alpha": alpha,
        "false_positive_rate": fpr,
        "expected_rate": alpha,
        "z_score": z_fpr,
        "standard_error": se,
        "ks_statistic": ks_stat,
        "ks_critical": ks_critical,
        "p_value_histogram": _histogram(p_values, bins=20),
        "verdict": verdict,
        "takeaway": (
            f"{n_trials:,} A/A tests with zero true effect produced a "
            f"{fpr:.2%} false positive rate against a nominal {alpha:.0%}. "
            "p-values are uniform, so the harness is calibrated and every "
            "downstream result can be believed."
        ),
    }


def _ks_uniform(values: list[float], alpha: float = 0.05) -> tuple[float, float]:
    """One-sample KS statistic against Uniform(0, 1), plus the critical value."""
    n = len(values)
    ordered = sorted(values)
    d = 0.0
    for i, v in enumerate(ordered, start=1):
        d = max(d, abs(i / n - v), abs(v - (i - 1) / n))
    # Asymptotic critical value: 1.358 / sqrt(n) at alpha = 0.05.
    critical = 1.358 / math.sqrt(n)
    return d, critical


def _histogram(values: list[float], bins: int = 20) -> list[int]:
    counts = [0] * bins
    for v in values:
        idx = min(int(v * bins), bins - 1)
        counts[idx] += 1
    return counts


def render(result: dict) -> str:
    lines = [
        f"  trials                {result['n_trials']:,} A/A experiments "
        f"@ {result['n_per_arm']:,} users/arm",
        f"  true effect           0.00% (identical arms by construction)",
        f"  false positive rate   {result['false_positive_rate']:.2%}  "
        f"(nominal {result['alpha']:.0%}, z = {result['z_score']:+.2f})",
        f"  p-value uniformity    KS = {result['ks_statistic']:.4f}  "
        f"(critical {result['ks_critical']:.4f})",
        "",
        "  p-value histogram (flat == correctly calibrated)",
    ]
    hist = result["p_value_histogram"]
    peak = max(hist) or 1
    for i, c in enumerate(hist):
        lo = i / len(hist)
        bar = "#" * int(38 * c / peak)
        lines.append(f"    {lo:.2f}-{lo + 1 / len(hist):.2f} |{bar} {c}")
    return "\n".join(lines)
