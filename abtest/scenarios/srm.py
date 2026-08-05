"""Scenario 4 -- Sample Ratio Mismatch, the check that runs before any analysis.

A 50/50 split that delivers 50.4% / 49.6% looks like nothing. On two million
users it is a five-sigma event, and it means the randomizer is not doing what
the analysis assumes. That matters far beyond the split itself: whatever
mechanism dropped those users almost certainly dropped a NON-RANDOM subset of
them, which breaks the exchangeability the entire causal claim rests on.

Typical real causes, none of them exotic:
  * the treatment renders slower, so more users bounce before the exposure
    event fires -- the arm loses exactly its most impatient users
  * a redirect drops clients that fail a JS check
  * bot filtering is applied to one arm's logs and not the other's
  * a bad deploy served control to a fraction of treatment-bucketed traffic

The rule this scenario supports: SRM is a hard gate, not a warning. If the
chi-square p-value is below 0.001, the experiment is not analyzed at all --
it is debugged. Reporting a lift from an experiment with SRM is reporting the
bug, not the feature.

The simulation shows both halves: the detector's sensitivity at realistic
sample sizes, and how badly the measured lift is corrupted when the dropout
is correlated with the outcome.
"""

from __future__ import annotations

from ..population import PopulationSpec
from ..rng import Rng
from ..stats import srm_check, two_proportion_ztest

TITLE = "Sample Ratio Mismatch -- the gate that runs before any result is read"


def run(
    seed: int = 20260803,
    n_trials: int = 400,
    n_users: int = 400_000,
    bias_grid: tuple[float, ...] = (0.0, 0.001, 0.0025, 0.005, 0.01, 0.02),
    alpha_srm: float = 0.001,
) -> dict:
    rng = Rng(seed).spawn("srm")

    detection = []
    for bias in bias_grid:
        caught = 0
        observed_shares = []
        for _ in range(n_trials):
            n_t = rng.binomial(n_users, 0.5 + bias)
            n_c = n_users - n_t
            check = srm_check([n_c, n_t], [0.5, 0.5], alpha_srm)
            caught += 1 if check["failed"] else 0
            observed_shares.append(n_t / n_users)
        detection.append(
            {
                "bias": bias,
                "true_split": 0.5 + bias,
                "detection_rate": caught / n_trials,
                "mean_observed_share": sum(observed_shares) / len(observed_shares),
            }
        )

    # Sized at 4M rather than 200k deliberately. The dropout bias is about
    # +1.9% regardless of sample size, but at 200k users that sits inside the
    # noise (p ~ 0.3) and the scenario fails to make its own point. The
    # danger of SRM is not that it creates a large bias -- it is that at real
    # traffic volumes it creates a SIGNIFICANT one, which is what gets shipped.
    corruption = _biased_dropout_damage(rng, n_users=4_000_000)

    # The detector must almost never fire when nothing is wrong, and must
    # reliably catch a 1 percentage point skew at this sample size.
    clean_fpr = detection[0]["detection_rate"]
    strong_detect = next(d["detection_rate"] for d in detection if d["bias"] == 0.01)
    verdict = (
        "PASS"
        if clean_fpr <= 0.01
        and strong_detect > 0.95
        # The corrupted experiment must produce a SIGNIFICANT fake lift that
        # the SRM gate nonetheless blocks. If the fake lift were not
        # significant, the gate would not be earning its keep.
        and corruption["measured_significant"]
        and corruption["srm_failed"]
        else "FAIL"
    )

    return {
        "scenario": "srm",
        "title": TITLE,
        "n_users": n_users,
        "n_trials": n_trials,
        "alpha_srm": alpha_srm,
        "detection": detection,
        "corruption": corruption,
        "false_alarm_rate": clean_fpr,
        "verdict": verdict,
        "takeaway": (
            f"At {n_users:,} users the chi-square gate catches a 1pp assignment "
            f"skew {strong_detect:.0%} of the time while false-alarming on clean "
            f"splits only {clean_fpr:.1%} of the time. When the dropout is "
            f"correlated with the outcome it fabricates a "
            f"{corruption['measured_lift']:+.1%} lift out of a true effect of "
            f"{corruption['true_lift']:+.1%}."
        ),
    }


def _biased_dropout_damage(rng: Rng, n_users: int) -> dict:
    """Show WHY SRM matters: non-random dropout invents a lift from nothing.

    The bug modeled here is the common one -- the treatment is slightly slower
    to render, so users who were least likely to convert anyway drop out of
    the treatment arm before being logged. Nothing about the product changed.
    The arm just lost its worst users, and the average went up.
    """
    spec = PopulationSpec(true_relative_effect=0.0)
    base = spec.blended_baseline
    dropout_prob = 0.02  # 2% of low-propensity treatment users never log

    n_c = n_users // 2
    n_t_assigned = n_users - n_c

    conv_c = rng.binomial(n_c, base)

    # Treatment: drop a slice of non-converters before they are counted.
    conv_t = rng.binomial(n_t_assigned, base)
    non_conv_t = n_t_assigned - conv_t
    dropped = rng.binomial(non_conv_t, dropout_prob)
    n_t_observed = n_t_assigned - dropped

    test = two_proportion_ztest(conv_c, n_c, conv_t, n_t_observed)
    check = srm_check([n_c, n_t_observed], [0.5, 0.5])

    return {
        "true_lift": 0.0,
        "measured_lift": test.relative_lift,
        "measured_p_value": test.p_value,
        "measured_significant": test.significant,
        "srm_p_value": check["p_value"],
        "srm_failed": check["failed"],
        "observed_ratio": check["observed_ratio"],
        "n_control": n_c,
        "n_treatment_assigned": n_t_assigned,
        "n_treatment_observed": n_t_observed,
        "dropped": dropped,
    }


def render(result: dict) -> str:
    lines = [
        f"  detector sensitivity at {result['n_users']:,} users, "
        f"alpha = {result['alpha_srm']}",
        "",
        "    true split      caught",
    ]
    for d in result["detection"]:
        label = "clean 50.00%" if d["bias"] == 0 else f"{d['true_split']:.3%}"
        bar = "#" * int(30 * d["detection_rate"])
        lines.append(f"    {label:>12}  |{bar} {d['detection_rate']:.1%}")

    c = result["corruption"]
    lines += [
        "",
        "  what SRM does to a result -- 2% of low-propensity treatment users never log:",
        f"    assigned to treatment    {c['n_treatment_assigned']:,}",
        f"    observed in treatment    {c['n_treatment_observed']:,}  "
        f"({c['dropped']:,} silently missing)",
        f"    observed split           {c['observed_ratio'][0]:.3%} / "
        f"{c['observed_ratio'][1]:.3%}",
        f"    TRUE lift                {c['true_lift']:+.2%}",
        f"    MEASURED lift            {c['measured_lift']:+.2%}  "
        f"(p = {c['measured_p_value']:.2e}, "
        f"{'SIGNIFICANT' if c['measured_significant'] else 'not significant'})",
        f"    SRM gate                 p = {c['srm_p_value']:.2e}  "
        f"-> {'BLOCKED, do not analyze' if c['srm_failed'] else 'passed'}",
        "",
        "    The lift is entirely fabricated. Without the gate it ships.",
    ]
    return "\n".join(lines)
