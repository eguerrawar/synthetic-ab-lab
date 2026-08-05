"""Scenario 5 -- Simpson's paradox, and why the pooled number can be a lie.

Construct an experiment where the treatment beats control in EVERY segment,
and loses overall. Not a trick: the arms have different segment mixes, and
the segments have very different baseline rates, so the pooled average is
dominated by the mix rather than by the treatment.

    segment    baseline   treated   control mix   treatment mix
    mobile       3.1%      3.4%         40%            70%
    desktop     11.8%     13.0%         60%            30%

Treatment wins in both rows. Pooled, treatment loses badly, because it is
carrying far more of the segment that converts worse for reasons that have
nothing to do with the treatment.

This is not a hypothetical failure mode. It is what a differential-exposure
bug looks like from the analyst's chair, and the pooled number is exactly
what a dashboard shows by default.

The fix is post-stratification: estimate the effect inside each segment,
then re-weight by the segment's share in the POPULATION rather than its share
within each arm. That removes the mix channel entirely and recovers the true
effect. It is also worth noting the corollary -- the mix imbalance that
causes this would have been caught upstream by the per-segment SRM check in
`scenarios/srm.py`. These failures travel together.
"""

from __future__ import annotations

from ..population import Segment
from ..rng import Rng
from ..stats import two_proportion_ztest

TITLE = "Simpson's paradox -- treatment wins every segment and loses the average"

SEGMENTS = [
    Segment("mobile", share=0.55, baseline_rate=0.031),
    Segment("desktop", share=0.45, baseline_rate=0.118),
]

# Fraction of each arm made up of mobile users. The gap is the bug.
CONTROL_MOBILE_MIX = 0.40
TREATMENT_MOBILE_MIX = 0.70


def run(
    seed: int = 20260803,
    n_per_arm: int = 300_000,
    true_relative_effect: float = 0.10,
) -> dict:
    rng = Rng(seed).spawn("heterogeneity")

    by_name = {s.name: s for s in SEGMENTS}
    arms = {}
    for arm, mobile_mix in (
        ("control", CONTROL_MOBILE_MIX),
        ("treatment", TREATMENT_MOBILE_MIX),
    ):
        n_mobile = int(n_per_arm * mobile_mix)
        n_desktop = n_per_arm - n_mobile
        effect = true_relative_effect if arm == "treatment" else 0.0
        counts = {}
        for seg_name, n_seg in (("mobile", n_mobile), ("desktop", n_desktop)):
            rate = by_name[seg_name].treated_rate(effect)
            counts[seg_name] = {"n": n_seg, "conv": rng.binomial(n_seg, rate), "rate_true": rate}
        arms[arm] = counts

    # --- the naive pooled read, i.e. what the dashboard shows ---------------
    n_c = sum(v["n"] for v in arms["control"].values())
    n_t = sum(v["n"] for v in arms["treatment"].values())
    conv_c = sum(v["conv"] for v in arms["control"].values())
    conv_t = sum(v["conv"] for v in arms["treatment"].values())
    pooled = two_proportion_ztest(conv_c, n_c, conv_t, n_t)

    # --- per-segment reads --------------------------------------------------
    per_segment = {}
    for seg in SEGMENTS:
        c = arms["control"][seg.name]
        t = arms["treatment"][seg.name]
        test = two_proportion_ztest(c["conv"], c["n"], t["conv"], t["n"])
        per_segment[seg.name] = {
            "share": seg.share,
            "rate_control": test.control_mean,
            "rate_treatment": test.treatment_mean,
            "relative_lift": test.relative_lift,
            "p_value": test.p_value,
            "significant": test.significant,
            "n_control": c["n"],
            "n_treatment": t["n"],
        }

    # --- post-stratified estimate ------------------------------------------
    abs_lift = sum(
        s.share * (per_segment[s.name]["rate_treatment"] - per_segment[s.name]["rate_control"])
        for s in SEGMENTS
    )
    baseline = sum(s.share * per_segment[s.name]["rate_control"] for s in SEGMENTS)
    stratified_lift = abs_lift / baseline

    all_segments_positive = all(v["relative_lift"] > 0 for v in per_segment.values())
    paradox = all_segments_positive and pooled.relative_lift < 0
    recovered = abs(stratified_lift - true_relative_effect) < 0.02

    verdict = "PASS" if paradox and recovered else "FAIL"

    return {
        "scenario": "heterogeneity",
        "title": TITLE,
        "n_per_arm": n_per_arm,
        "true_relative_effect": true_relative_effect,
        "control_mobile_mix": CONTROL_MOBILE_MIX,
        "treatment_mobile_mix": TREATMENT_MOBILE_MIX,
        "pooled_relative_lift": pooled.relative_lift,
        "pooled_p_value": pooled.p_value,
        "pooled_significant": pooled.significant,
        "stratified_relative_lift": stratified_lift,
        "per_segment": per_segment,
        "paradox_reproduced": paradox,
        "verdict": verdict,
        "takeaway": (
            f"Treatment lifts every segment by {true_relative_effect:+.0%}, yet the "
            f"pooled number reads {pooled.relative_lift:+.1%} because the arms carry "
            f"different device mixes. Post-stratifying on the population share "
            f"recovers {stratified_lift:+.1%}. Never ship on a pooled average without "
            "checking the mix."
        ),
    }


def render(result: dict) -> str:
    lines = [
        f"  design              {result['n_per_arm']:,} users/arm, "
        f"true effect {result['true_relative_effect']:+.0%} in EVERY segment",
        f"  the bug             control is {result['control_mobile_mix']:.0%} mobile, "
        f"treatment is {result['treatment_mobile_mix']:.0%} mobile",
        "",
        "    segment     control    treatment    lift        verdict",
    ]
    for name, v in result["per_segment"].items():
        lines.append(
            f"    {name:<10} {v['rate_control']:7.3%}    {v['rate_treatment']:7.3%}   "
            f"{v['relative_lift']:+7.2%}    "
            f"{'WIN (p=%.1e)' % v['p_value'] if v['significant'] else 'flat'}"
        )
    lines += [
        "",
        f"    POOLED     {result['pooled_relative_lift']:+7.2%}  "
        f"(p = {result['pooled_p_value']:.2e})   <-- treatment 'loses'",
        f"    STRATIFIED {result['stratified_relative_lift']:+7.2%}  "
        f"<-- recovers the truth ({result['true_relative_effect']:+.0%})",
        "",
        "    Every segment improved. The pooled average fell. Both are arithmetic.",
    ]
    return "\n".join(lines)
