"""Scenario 6 -- novelty decay, or why the day-3 read is not the answer.

Some treatment effects are real on day one and gone by day ten. Users notice
a change, click the new thing because it is new, and habituate. A redesigned
button, a new banner, a notification -- these routinely show a large early
lift that decays toward zero.

The failure mode is not statistical. The test is correctly calibrated, the
p-value is honest, and the early result is a true measurement -- of a
temporary effect. It gets shipped as if it were permanent, and the metric
quietly returns to baseline a month later, by which point nobody connects the
two events.

The simulation applies exponential decay with a configurable half-life and
compares three readings of the same experiment:

  cumulative-to-date  what the dashboard shows, and what gets shipped
  last-week-only      the durable effect, measured after habituation
  daily               the decay curve itself, which is the actual diagnosis

The operational rule this supports: for any surface-level UI change, do not
read the cumulative number. Read the final week in isolation, and require the
daily series to be flat before calling the effect durable. If it is still
sloping, the experiment is not finished.
"""

from __future__ import annotations

from ..population import PopulationSpec, effect_on_day
from ..rng import Rng
from ..stats import two_proportion_ztest

TITLE = "Novelty decay -- the early read measures a real effect that will not last"


def run(
    seed: int = 20260803,
    baseline_rate: float = 0.085,
    initial_effect: float = 0.12,
    half_life_days: float = 4.0,
    n_days: int = 28,
    users_per_day_per_arm: int = 80_000,
) -> dict:
    rng = Rng(seed).spawn("novelty")
    spec = PopulationSpec(
        true_relative_effect=initial_effect,
        novelty_half_life_days=half_life_days,
        n_days=n_days,
    )

    daily = []
    cum_c = cum_t = cum_n = 0
    for day in range(n_days):
        effect_today = effect_on_day(spec, day)
        c = rng.binomial(users_per_day_per_arm, baseline_rate)
        t = rng.binomial(users_per_day_per_arm, baseline_rate * (1.0 + effect_today))

        cum_c += c
        cum_t += t
        cum_n += users_per_day_per_arm

        cumulative = two_proportion_ztest(cum_c, cum_n, cum_t, cum_n)
        today = two_proportion_ztest(c, users_per_day_per_arm, t, users_per_day_per_arm)

        daily.append(
            {
                "day": day + 1,
                "true_effect_today": effect_today,
                "daily_measured_lift": today.relative_lift,
                "cumulative_measured_lift": cumulative.relative_lift,
                "cumulative_p_value": cumulative.p_value,
                "cumulative_significant": cumulative.significant,
            }
        )

    # Three readings of the same experiment.
    week1 = _window_estimate(rng, spec, baseline_rate, users_per_day_per_arm, 0, 7)
    final_week = _window_estimate(
        rng, spec, baseline_rate, users_per_day_per_arm, n_days - 7, n_days
    )
    cumulative_final = daily[-1]["cumulative_measured_lift"]

    true_final = final_week["true_window_effect"]

    # Pass criteria. The first is a property of the design and is essentially
    # deterministic. The second is a COVERAGE RATE over many replications
    # rather than a single-draw CI check: any one 95% interval misses 5% of
    # the time by construction, so testing one interval would make this
    # scenario fail at random roughly one run in twenty. Measuring the
    # coverage rate tests the estimator instead of testing a coin flip.
    coverage = _coverage_of_final_week_estimator(
        rng, spec, baseline_rate, users_per_day_per_arm, n_days, n_replications=400
    )
    overstates = cumulative_final > true_final + 0.02
    estimator_unbiased = 0.92 <= coverage["coverage_rate"] <= 0.98
    verdict = "PASS" if overstates and estimator_unbiased else "FAIL"

    return {
        "scenario": "novelty",
        "title": TITLE,
        "baseline_rate": baseline_rate,
        "initial_effect": initial_effect,
        "half_life_days": half_life_days,
        "n_days": n_days,
        "users_per_day_per_arm": users_per_day_per_arm,
        "daily": daily,
        "week1_lift": week1["measured_lift"],
        "week1_true": week1["true_window_effect"],
        "final_week_lift": final_week["measured_lift"],
        "final_week_ci": (final_week["ci_low"], final_week["ci_high"]),
        "final_week_significant": final_week["significant"],
        "cumulative_final_lift": cumulative_final,
        "true_effect_final_week": true_final,
        "coverage": coverage,
        "verdict": verdict,
        "takeaway": (
            f"A {initial_effect:+.0%} launch-day lift with a {half_life_days:.0f}-day "
            f"half-life reads {week1['measured_lift']:+.1%} in week 1 and "
            f"{cumulative_final:+.1%} cumulatively, but the durable effect measured "
            f"in the final week is {final_week['measured_lift']:+.1%} with a CI that "
            "covers zero. Ship on the cumulative number and the win evaporates in "
            "production a month later, when nobody connects the two events."
        ),
    }


def _window_estimate(
    rng: Rng,
    spec: PopulationSpec,
    baseline_rate: float,
    per_day: int,
    day_from: int,
    day_to: int,
) -> dict:
    """Re-simulate a window of days in isolation and measure just that window."""
    n = conv_c = conv_t = 0
    for day in range(day_from, day_to):
        eff = effect_on_day(spec, day)
        conv_c += rng.binomial(per_day, baseline_rate)
        conv_t += rng.binomial(per_day, baseline_rate * (1.0 + eff))
        n += per_day
    test = two_proportion_ztest(conv_c, n, conv_t, n)
    # The traffic-weighted true effect over the same window, so the measured
    # number can be compared against what it is actually estimating rather
    # than against the single final-day value.
    true_avg = sum(effect_on_day(spec, d) for d in range(day_from, day_to)) / (
        day_to - day_from
    )
    return {
        "measured_lift": test.relative_lift,
        "true_window_effect": true_avg,
        "ci_low": test.ci_low_relative,
        "ci_high": test.ci_high_relative,
        "p_value": test.p_value,
        "significant": test.significant,
        "days": (day_from + 1, day_to),
    }


def _coverage_of_final_week_estimator(
    rng: Rng,
    spec: PopulationSpec,
    baseline_rate: float,
    per_day: int,
    n_days: int,
    n_replications: int = 400,
) -> dict:
    """How often does the final-week CI actually contain the final-week truth?

    If the answer is 95%, the "read the last week" rule is not just good
    advice -- it is an unbiased estimator of the durable effect, with honest
    uncertainty. That is a much stronger statement than one interval that
    happened to land well.
    """
    day_from, day_to = n_days - 7, n_days
    true_avg = sum(effect_on_day(spec, d) for d in range(day_from, day_to)) / 7

    covered = 0
    estimates = []
    for _ in range(n_replications):
        n = conv_c = conv_t = 0
        for day in range(day_from, day_to):
            eff = effect_on_day(spec, day)
            conv_c += rng.binomial(per_day, baseline_rate)
            conv_t += rng.binomial(per_day, baseline_rate * (1.0 + eff))
            n += per_day
        test = two_proportion_ztest(conv_c, n, conv_t, n)
        estimates.append(test.relative_lift)
        if test.ci_low_relative <= true_avg <= test.ci_high_relative:
            covered += 1

    mean_est = sum(estimates) / len(estimates)
    return {
        "n_replications": n_replications,
        "true_window_effect": true_avg,
        "mean_estimate": mean_est,
        "bias": mean_est - true_avg,
        "coverage_rate": covered / n_replications,
    }


def render(result: dict) -> str:
    lines = [
        f"  design            {result['initial_effect']:+.0%} launch-day effect, "
        f"{result['half_life_days']:.0f}-day half-life, {result['n_days']} days",
        "",
        "    day   true    daily measured   cumulative (dashboard)",
    ]
    for d in result["daily"]:
        if d["day"] % 2 and d["day"] != 1:
            continue
        flag = " *" if d["cumulative_significant"] else "  "
        lines.append(
            f"    {d['day']:>3}  {d['true_effect_today']:+6.2%}   "
            f"{d['daily_measured_lift']:+8.2%}       "
            f"{d['cumulative_measured_lift']:+8.2%}{flag}"
        )
    lines += [
        "",
        "    three readings of the SAME experiment:",
        f"      week 1 only            {result['week1_lift']:+.2%}   "
        f"(true: {result['week1_true']:+.2%})  <- the excited Slack message",
        f"      cumulative day 1-{result['n_days']}    "
        f"{result['cumulative_final_lift']:+.2%}                  "
        "<- what the dashboard shows",
        f"      final week only        {result['final_week_lift']:+.2%}   "
        f"(true: {result['true_effect_final_week']:+.2%})  <- the DURABLE effect",
        f"                             95% CI [{result['final_week_ci'][0]:+.2%}, "
        f"{result['final_week_ci'][1]:+.2%}]",
        "",
        "    is 'read the last week' actually an unbiased rule? "
        f"({result['coverage']['n_replications']} replications)",
        f"      mean estimate    {result['coverage']['mean_estimate']:+.3%}  vs truth "
        f"{result['coverage']['true_window_effect']:+.3%}  "
        f"(bias {result['coverage']['bias']:+.3%})",
        f"      CI coverage      {result['coverage']['coverage_rate']:.1%}  "
        "(nominal 95%)",
        "",
        "    Read the last week, not the average. Require the daily series to be flat.",
    ]
    return "\n".join(lines)
