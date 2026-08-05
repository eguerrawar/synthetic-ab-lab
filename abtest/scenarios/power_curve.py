"""Scenario 2 -- does the sample size calculator tell the truth?

The power formula in `power.py` promises that at n users per arm, an effect
of size d will be detected 80% of the time. That is a falsifiable claim, and
this scenario falsifies or confirms it: simulate thousands of experiments at
each of several effect sizes, count how often each one actually reaches
significance, and compare the empirical rejection rate to the analytic curve.

If the two agree, the planning numbers can be quoted to a PM with confidence.
If they diverge, every "we need two weeks of traffic" estimate the team has
ever given is wrong by the same amount.

This is the same idea as a bench validation sweep: drive the system with a
known input across its operating range and confirm the response matches the
spec sheet.
"""

from __future__ import annotations

from ..experiment import run_aggregate_trial
from ..power import mde_at_n, power_at_n, sample_size_proportions
from ..rng import Rng

TITLE = "Power validation -- analytic sample size vs. simulated reality"


def run(
    seed: int = 20260803,
    n_trials: int = 6000,
    baseline_rate: float = 0.085,
    alpha: float = 0.05,
    target_power: float = 0.80,
    design_mde: float = 0.05,
    effect_grid: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10),
) -> dict:
    rng = Rng(seed).spawn("power_curve")

    # Size the experiment for the effect we care about, then check what the
    # design actually delivers across the full range of possible truths.
    plan = sample_size_proportions(baseline_rate, design_mde, alpha, target_power)
    n = plan.n_per_arm

    points = []
    max_gap = 0.0
    for effect in effect_grid:
        treated = baseline_rate * (1.0 + effect)
        hits = 0
        for _ in range(n_trials):
            res = run_aggregate_trial(n, baseline_rate, treated, rng, alpha)
            if res.significant:
                hits += 1
        empirical = hits / n_trials
        analytic = power_at_n(baseline_rate, effect, n, alpha)
        gap = abs(empirical - analytic)
        max_gap = max(max_gap, gap)
        points.append(
            {
                "relative_effect": effect,
                "empirical_power": empirical,
                "analytic_power": analytic,
                "gap": gap,
            }
        )

    # Monte Carlo noise at n_trials replications is about 1.1 percentage
    # points (1 SE); allow 3 SE plus a small allowance for the normal
    # approximation in the power formula.
    tolerance = 3 * (0.25 / n_trials) ** 0.5 + 0.01
    verdict = "PASS" if max_gap < tolerance else "FAIL"

    return {
        "scenario": "power",
        "title": TITLE,
        "baseline_rate": baseline_rate,
        "design_mde": design_mde,
        "n_per_arm": n,
        "n_total": plan.n_total,
        "alpha": alpha,
        "target_power": target_power,
        "n_trials": n_trials,
        "points": points,
        "max_gap": max_gap,
        "tolerance": tolerance,
        "mde_table": _mde_table(baseline_rate, alpha, target_power),
        "verdict": verdict,
        "takeaway": (
            f"Sized for a {design_mde:+.0%} lift, the design needs "
            f"{plan.n_total:,} users. Simulated power tracks the analytic "
            f"curve to within {max_gap:.1%} everywhere, so the planning "
            "numbers are safe to quote."
        ),
    }


def _mde_table(baseline_rate: float, alpha: float, power: float) -> list[dict]:
    """What is detectable at each realistic traffic level -- the planning table."""
    rows = []
    for n in (5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000):
        rows.append(
            {
                "n_per_arm": n,
                "mde": mde_at_n(baseline_rate, n, alpha, power),
                "absolute_mde": baseline_rate * mde_at_n(baseline_rate, n, alpha, power),
            }
        )
    return rows


def render(result: dict) -> str:
    lines = [
        f"  design            baseline {result['baseline_rate']:.2%}, "
        f"MDE {result['design_mde']:+.0%}, power {result['target_power']:.0%}",
        f"  required n        {result['n_per_arm']:,} per arm "
        f"({result['n_total']:,} total)",
        f"  replications      {result['n_trials']:,} per grid point",
        "",
        "  true lift    simulated power    analytic power    gap",
    ]
    for p in result["points"]:
        lines.append(
            f"    {p['relative_effect']:+6.1%}        {p['empirical_power']:6.1%}"
            f"            {p['analytic_power']:6.1%}      {p['gap']:.3f}"
        )
    lines.append("")
    lines.append(f"  worst gap {result['max_gap']:.3f} vs tolerance {result['tolerance']:.3f}")
    lines.append("")
    lines.append("  planning table -- smallest lift visible at each traffic level")
    for row in result["mde_table"]:
        lines.append(
            f"    {row['n_per_arm']:>7,} users/arm  ->  MDE {row['mde']:+.2%} "
            f"({row['absolute_mde'] * 100:.3f} pp)"
        )
    return "\n".join(lines)
