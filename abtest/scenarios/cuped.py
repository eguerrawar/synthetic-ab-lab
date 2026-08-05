"""Scenario 7 -- CUPED, the closest thing to a free lunch in experimentation.

Every other lever for detecting smaller effects costs something: more users,
more days, or a looser alpha. CUPED costs a join against pre-period data and
buys a variance reduction of (1 - rho^2), where rho is the correlation
between the pre-period covariate and the outcome.

The mechanic:

    Y_cuped = Y - theta * (X - E[X]),   theta = Cov(Y, X) / Var(X)

X is measured BEFORE assignment, so its mean is equal across arms in
expectation and subtracting it cannot bias the treatment effect. It only
removes the part of Y that was already predictable from who the user was.
What remains is the part the treatment could have moved -- and it is much
less noisy.

This scenario makes two claims and tests both:

  1. The reduction really is rho^2, where rho is the ACHIEVED correlation
     between covariate and metric. Predicted and observed are printed side by
     side; if the implementation were wrong the columns would diverge.

  2. Which metric you apply it to matters enormously. On a continuous
     engagement metric, CUPED delivers the full rho^2. On a binary conversion
     flag it delivers far less -- and the reason is not a bug. A Bernoulli
     indicator with an 8.5% rate is mostly irreducible coin-flip noise, so
     even a perfect predictor of the underlying propensity can only correlate
     with the realized 0/1 outcome weakly. The nominal population rho of 0.9
     shows up as an achieved rho near 0.45 against the binary outcome, and
     0.45^2 is what CUPED actually recovers.

Claim 2 is the practically useful one. It is the difference between "we
tried CUPED and it did nothing" and knowing in advance which metrics it will
pay off on.

Reference: Deng, Xu, Kohavi & Walker, WSDM 2013.
"""

from __future__ import annotations

from ..experiment import cuped_estimate, run_user_level_trial
from ..population import PopulationSpec
from ..power import sample_size_with_cuped
from ..rng import Rng

TITLE = "CUPED -- same data, same answer, materially less noise"


def run(
    seed: int = 20260803,
    n_users: int = 60_000,
    true_effect: float = 0.05,
    correlations: tuple[float, ...] = (0.0, 0.3, 0.5, 0.7, 0.9),
    n_repeats: int = 6,
) -> dict:
    rng = Rng(seed).spawn("cuped")

    binary_rows = []
    continuous_rows = []
    worst_theory_gap = 0.0

    for rho in correlations:
        spec = PopulationSpec(true_relative_effect=true_effect, covariate_correlation=rho)
        acc = {
            "binary": {"pred": [], "obs": [], "rho": [], "se_ratio": []},
            "continuous": {"pred": [], "obs": [], "rho": [], "se_ratio": []},
        }

        for r in range(n_repeats):
            result = run_user_level_trial(spec, n_users, rng.spawn(f"rho{rho}_{r}"))

            variants = {
                "binary": cuped_estimate(
                    result,
                    metric=lambda u: float(u.converted),
                    covariate=lambda u: u.pre_period_value,
                ),
                "continuous": cuped_estimate(
                    result,
                    metric=lambda u: u.sessions_post,
                    covariate=lambda u: u.sessions_pre,
                ),
            }
            for kind, adj in variants.items():
                a = acc[kind]
                a["pred"].append(adj["predicted_reduction"])
                a["obs"].append(adj["variance_reduction"])
                a["rho"].append(adj["achieved_correlation"])
                a["se_ratio"].append(
                    adj["test"].standard_error / adj["plain_test"].standard_error
                )

        for kind, target in (("binary", binary_rows), ("continuous", continuous_rows)):
            a = acc[kind]
            mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
            pred, obs = mean(a["pred"]), mean(a["obs"])
            worst_theory_gap = max(worst_theory_gap, abs(pred - obs))
            target.append(
                {
                    "nominal_rho": rho,
                    "achieved_rho": mean(a["rho"]),
                    "predicted_reduction": pred,
                    "observed_reduction": obs,
                    "se_ratio": mean(a["se_ratio"]),
                    # Variance reduction converts directly into sample size:
                    # the same precision is reached on (1 - reduction) as
                    # many users. This is the number worth quoting to a PM.
                    "equivalent_traffic_fraction": 1.0 - obs,
                }
            )

    planning = []
    for rho in (0.3, 0.5, 0.7, 0.9):
        plain, reduced, saved = sample_size_with_cuped(0.085, 0.03, rho)
        planning.append(
            {
                "correlation": rho,
                "n_plain": plain,
                "n_cuped": reduced,
                "fraction_saved": saved,
            }
        )

    # The theory is the claim under test: observed reduction must track
    # achieved_rho^2 across every row and both metric types.
    theory_holds = worst_theory_gap < 0.03
    helps_continuous = continuous_rows[-1]["observed_reduction"] > 0.70
    verdict = "PASS" if theory_holds and helps_continuous else "FAIL"

    return {
        "scenario": "cuped",
        "title": TITLE,
        "n_users": n_users,
        "true_effect": true_effect,
        "n_repeats": n_repeats,
        "binary_rows": binary_rows,
        "continuous_rows": continuous_rows,
        "worst_theory_gap": worst_theory_gap,
        "planning": planning,
        "verdict": verdict,
        "takeaway": (
            f"Observed variance reduction tracks the achieved rho^2 to within "
            f"{worst_theory_gap:.3f} across both metric types, so the estimator is "
            f"correct. On the continuous metric CUPED removes "
            f"{continuous_rows[-1]['observed_reduction']:.0%} of the variance; on the "
            f"binary conversion flag only "
            f"{binary_rows[-1]['observed_reduction']:.0%}, because a Bernoulli outcome "
            "is mostly irreducible noise. Apply CUPED to engagement metrics first."
        ),
    }


def _render_table(rows: list[dict]) -> list[str]:
    out = [
        "    nominal rho   achieved rho   predicted (rho^2)   observed   SE ratio   "
        "traffic needed",
    ]
    for r in rows:
        out.append(
            f"    {r['nominal_rho']:>10.1f}   {r['achieved_rho']:>12.3f}   "
            f"{r['predicted_reduction']:>16.1%}   {r['observed_reduction']:>8.1%}   "
            f"{r['se_ratio']:>8.3f}   {r['equivalent_traffic_fraction']:>13.0%}"
        )
    return out


def render(result: dict) -> str:
    lines = [
        f"  design      {result['n_users']:,} users, true effect "
        f"{result['true_effect']:+.0%}, {result['n_repeats']} replications per row",
        "",
        "  CONTINUOUS metric (sessions per week) -- CUPED works as advertised",
    ]
    lines += _render_table(result["continuous_rows"])
    lines += [
        "",
        "  BINARY metric (converted yes/no) -- same code, much less to gain",
    ]
    lines += _render_table(result["binary_rows"])
    lines += [
        "",
        f"  predicted vs observed agree to within {result['worst_theory_gap']:.3f} "
        "everywhere -- the estimator is correct, the metric is the constraint",
        "",
        "  what the variance reduction is worth in planning",
        "  (baseline 8.5%, detect +3% relative, 80% power)",
        "",
        "    rho    users/arm plain    users/arm CUPED    saved",
    ]
    for p in result["planning"]:
        lines.append(
            f"    {p['correlation']:.1f}    {p['n_plain']:>13,}    {p['n_cuped']:>14,}    "
            f"{p['fraction_saved']:.0%}"
        )
    return "\n".join(lines)
