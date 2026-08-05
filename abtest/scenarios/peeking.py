"""Scenario 3 -- the peeking problem, and what it actually costs to fix.

This is the headline result of the repo.

Nobody launches an experiment and then ignores the dashboard for two weeks.
People look daily, and they stop when the number turns green. That behavior
is completely rational and completely invalidates a fixed-horizon t-test,
whose 5% false positive guarantee only holds at the single sample size it was
committed to in advance.

The simulation makes the damage concrete: run A/A experiments (true effect
exactly zero) where data arrives daily, and stop the first time p < 0.05.

  * Fixed-horizon z-test, checked daily: false positive rate climbs to
    roughly 25-30%. Better than one in four "wins" is noise.
  * mSPRT always-valid p-value, checked daily: stays at or under 5%, which
    is what it promises for any stopping rule at all.

The second half of the scenario prices the fix honestly. Always-valid
inference is not free -- against a real effect it needs more data than a
fixed-horizon test to reach the same power. The trade is: pay some sample
size, buy the ability to stop whenever the evidence justifies it. For most
growth teams that is a good trade, because the alternative is not "wait
patiently" -- the alternative is peeking anyway with an uncontrolled error
rate.
"""

from __future__ import annotations

from ..rng import Rng
from ..sequential import MSPRT, fixed_horizon_peek_risk
from ..stats import two_proportion_ztest

TITLE = "Peeking -- how daily dashboard checks manufacture false positives"


def run(
    seed: int = 20260803,
    n_trials: int = 3000,
    baseline_rate: float = 0.085,
    users_per_day_per_arm: int = 4_000,
    n_days: int = 14,
    alpha: float = 0.05,
    true_effect_for_power: float = 0.05,
) -> dict:
    rng = Rng(seed).spawn("peeking")

    naive_fp = 0
    seq_fp = 0
    naive_stop_days: list[int] = []
    seq_stop_days: list[int] = []
    # Track the fixed-horizon FPR as a function of how many looks were taken,
    # which is the curve that makes the problem legible to a non-statistician.
    fp_by_look = [0] * n_days

    tau = baseline_rate * true_effect_for_power  # prior scale ~ the design MDE

    for _ in range(n_trials):
        msprt = MSPRT(tau=tau, alpha=alpha)
        n_c = n_t = conv_c = conv_t = 0
        naive_fired = False
        seq_fired = False

        for day in range(n_days):
            n_c += users_per_day_per_arm
            n_t += users_per_day_per_arm
            conv_c += rng.binomial(users_per_day_per_arm, baseline_rate)
            conv_t += rng.binomial(users_per_day_per_arm, baseline_rate)

            fixed = two_proportion_ztest(conv_c, n_c, conv_t, n_t, alpha)
            if fixed.significant and not naive_fired:
                naive_fired = True
                naive_stop_days.append(day + 1)
            if naive_fired:
                fp_by_look[day] += 1

            state = msprt.update_proportions(conv_c, n_c, conv_t, n_t)
            if state.p_value < alpha and not seq_fired:
                seq_fired = True
                seq_stop_days.append(day + 1)

        naive_fp += 1 if naive_fired else 0
        seq_fp += 1 if seq_fired else 0

    naive_rate = naive_fp / n_trials
    seq_rate = seq_fp / n_trials

    power = _power_comparison(
        rng, baseline_rate, true_effect_for_power, users_per_day_per_arm, n_days, alpha, tau
    )
    price = _price_of_always_valid(
        rng, baseline_rate, true_effect_for_power, users_per_day_per_arm, alpha, tau
    )
    tau_study = _tau_sensitivity(
        rng, baseline_rate, true_effect_for_power, users_per_day_per_arm, n_days, alpha
    )

    verdict = "PASS" if seq_rate <= alpha * 1.5 and naive_rate > alpha * 2 else "FAIL"

    return {
        "scenario": "peeking",
        "title": TITLE,
        "n_trials": n_trials,
        "n_days": n_days,
        "users_per_day_per_arm": users_per_day_per_arm,
        "baseline_rate": baseline_rate,
        "alpha": alpha,
        "naive_false_positive_rate": naive_rate,
        "sequential_false_positive_rate": seq_rate,
        "inflation_factor": naive_rate / alpha,
        "independent_looks_bound": fixed_horizon_peek_risk(n_days, alpha),
        "fpr_by_look": [c / n_trials for c in fp_by_look],
        "median_false_stop_day_naive": _median(naive_stop_days),
        "power": power,
        "price": price,
        "tau_study": tau_study,
        "tau": tau,
        "verdict": verdict,
        "takeaway": (
            f"Checking a fixed-horizon test every day for {n_days} days turns a "
            f"{alpha:.0%} false positive rate into {naive_rate:.1%} -- "
            f"{naive_rate / alpha:.1f}x inflation, on experiments with zero "
            f"true effect. The mSPRT held at {seq_rate:.1%} under the exact "
            "same stopping rule."
        ),
    }


def _power_comparison(
    rng: Rng,
    baseline_rate: float,
    effect: float,
    per_day: int,
    n_days: int,
    alpha: float,
    tau: float,
    n_trials: int = 1000,
) -> dict:
    """What the always-valid guarantee costs when the effect is real."""
    treated = baseline_rate * (1.0 + effect)
    fixed_hits = 0
    seq_hits = 0
    seq_days: list[int] = []

    for _ in range(n_trials):
        msprt = MSPRT(tau=tau, alpha=alpha)
        n_c = n_t = conv_c = conv_t = 0
        seq_fired = False
        for day in range(n_days):
            n_c += per_day
            n_t += per_day
            conv_c += rng.binomial(per_day, baseline_rate)
            conv_t += rng.binomial(per_day, treated)
            state = msprt.update_proportions(conv_c, n_c, conv_t, n_t)
            if state.p_value < alpha and not seq_fired:
                seq_fired = True
                seq_days.append(day + 1)
        # Fixed-horizon test, evaluated ONLY at the pre-committed end date --
        # the way it is supposed to be used.
        if two_proportion_ztest(conv_c, n_c, conv_t, n_t, alpha).significant:
            fixed_hits += 1
        if seq_fired:
            seq_hits += 1

    return {
        "true_effect": effect,
        "fixed_horizon_power_at_end": fixed_hits / n_trials,
        "sequential_power_any_time": seq_hits / n_trials,
        "median_days_to_detect_sequential": _median(seq_days),
        "n_trials": n_trials,
    }


def _price_of_always_valid(
    rng: Rng,
    baseline_rate: float,
    effect: float,
    per_day: int,
    alpha: float,
    tau: float,
    max_days: int = 70,
    n_trials: int = 1500,
    target_power: float = 0.80,
) -> dict:
    """How much extra traffic buys the right to peek?

    Run the mSPRT out to a long horizon and record the day it reaches the
    same power a fixed-horizon test achieves at its designed sample size.
    The ratio of those two sample sizes is the actual, quotable price of
    always-valid inference -- far more useful to a PM than "it has less
    power", which invites the wrong conclusion.
    """
    from ..power import sample_size_proportions

    treated = baseline_rate * (1.0 + effect)
    fired_by_day = [0] * max_days

    for _ in range(n_trials):
        msprt = MSPRT(tau=tau, alpha=alpha)
        n = conv_c = conv_t = 0
        fired_on = None
        for day in range(max_days):
            n += per_day
            conv_c += rng.binomial(per_day, baseline_rate)
            conv_t += rng.binomial(per_day, treated)
            if fired_on is None:
                state = msprt.update_proportions(conv_c, n, conv_t, n)
                if state.p_value < alpha:
                    fired_on = day
        if fired_on is not None:
            for d in range(fired_on, max_days):
                fired_by_day[d] += 1

    cumulative = [c / n_trials for c in fired_by_day]
    day_at_target = next(
        (i + 1 for i, p in enumerate(cumulative) if p >= target_power), None
    )

    fixed_n = sample_size_proportions(baseline_rate, effect, alpha, target_power).n_per_arm
    seq_n = day_at_target * per_day if day_at_target else None

    return {
        "target_power": target_power,
        "fixed_horizon_n_per_arm": fixed_n,
        "sequential_n_per_arm": seq_n,
        "sequential_days": day_at_target,
        "traffic_multiplier": (seq_n / fixed_n) if seq_n else None,
        "cumulative_power_by_day": cumulative,
        "max_days": max_days,
        "n_trials": n_trials,
    }


def _tau_sensitivity(
    rng: Rng,
    baseline_rate: float,
    effect: float,
    per_day: int,
    n_days: int,
    alpha: float,
    n_trials: int = 800,
) -> list[dict]:
    """Show that tau trades power, never validity.

    The mixture prior is the one judgement call in the method, so it is worth
    demonstrating what it does and does not affect: false positive control
    holds for every tau, while power peaks when tau is near the true effect.
    Set tau to the MDE the experiment was powered for.
    """
    truth_abs = baseline_rate * effect
    rows = []
    for mult in (0.25, 0.5, 1.0, 2.0, 4.0):
        tau = truth_abs * mult
        fp = hits = 0
        for _ in range(n_trials):
            # A/A run -- checks validity.
            m0 = MSPRT(tau=tau, alpha=alpha)
            n = c0 = t0 = 0
            fired = False
            for _day in range(n_days):
                n += per_day
                c0 += rng.binomial(per_day, baseline_rate)
                t0 += rng.binomial(per_day, baseline_rate)
                if not fired and m0.update_proportions(c0, n, t0, n).p_value < alpha:
                    fired = True
            fp += 1 if fired else 0

            # A/B run -- checks power.
            m1 = MSPRT(tau=tau, alpha=alpha)
            n = c1 = t1 = 0
            fired = False
            for _day in range(n_days):
                n += per_day
                c1 += rng.binomial(per_day, baseline_rate)
                t1 += rng.binomial(per_day, baseline_rate * (1 + effect))
                if not fired and m1.update_proportions(c1, n, t1, n).p_value < alpha:
                    fired = True
            hits += 1 if fired else 0

        rows.append(
            {
                "tau_multiple_of_truth": mult,
                "tau": tau,
                "false_positive_rate": fp / n_trials,
                "power": hits / n_trials,
            }
        )
    return rows


def _median(xs: list[int]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    m = len(s) // 2
    return float(s[m]) if len(s) % 2 else (s[m - 1] + s[m]) / 2.0


def render(result: dict) -> str:
    lines = [
        f"  setup                 {result['n_trials']:,} A/A experiments, "
        f"{result['users_per_day_per_arm']:,} users/arm/day, "
        f"checked daily for {result['n_days']} days",
        f"  true effect           0.00%  (every rejection below is a false positive)",
        "",
        f"  fixed-horizon z-test, peeked daily   "
        f"{result['naive_false_positive_rate']:6.1%}   "
        f"<-- {result['inflation_factor']:.1f}x the nominal {result['alpha']:.0%}",
        f"  mSPRT always-valid p-value           "
        f"{result['sequential_false_positive_rate']:6.1%}   <-- controlled",
        "",
        "  false positive rate as looks accumulate",
    ]
    for i, r in enumerate(result["fpr_by_look"], start=1):
        bar = "#" * int(60 * r)
        lines.append(f"    day {i:>2} |{bar} {r:.1%}")

    p = result["power"]
    pr = result["price"]
    lines += [
        "",
        f"  cost of the fix -- against a real {p['true_effect']:+.0%} lift, "
        f"same {result['n_days']}-day window:",
        f"    fixed-horizon power (correct use, ONE look at the end)  "
        f"{p['fixed_horizon_power_at_end']:.1%}",
        f"    mSPRT power (stop ANY time)                             "
        f"{p['sequential_power_any_time']:.1%}",
        "",
        "  priced properly -- traffic needed to reach 80% power:",
        f"    fixed horizon   {pr['fixed_horizon_n_per_arm']:>9,} users/arm",
    ]
    if pr["sequential_n_per_arm"]:
        lines += [
            f"    mSPRT           {pr['sequential_n_per_arm']:>9,} users/arm "
            f"({pr['sequential_days']} days)",
            f"    the price       {pr['traffic_multiplier']:.2f}x traffic to buy "
            "unlimited peeking",
        ]
    else:
        lines.append(
            f"    mSPRT           did not reach 80% within {pr['max_days']} days"
        )

    lines += [
        "",
        "  tau sensitivity -- the prior trades POWER, never VALIDITY:",
        "",
        "    tau / true effect    false positive rate    power",
    ]
    for row in result["tau_study"]:
        lines.append(
            f"    {row['tau_multiple_of_truth']:>13.2f}x    "
            f"{row['false_positive_rate']:>15.1%}    {row['power']:>6.1%}"
        )
    lines.append("")
    lines.append(
        "    Every row controls false positives. Only the power column moves."
    )
    return "\n".join(lines)
