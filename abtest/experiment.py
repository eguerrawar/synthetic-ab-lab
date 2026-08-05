"""Assignment, measurement, and the analysis estimators.

Two ways to run an experiment live here, and the split is intentional:

  * `run_aggregate_trial` -- draws each arm's conversion count in a single
    binomial call. Used for calibration studies that need tens of thousands
    of repeated experiments.
  * `run_user_level_trial` -- materializes every user with their attributes.
    Slower, but required for anything that reads user-level structure:
    segment analysis, CUPED, novelty over time.

Also here: the three estimators the scenarios compare against each other.
  naive     -- pooled difference in means. Correct only if the arms are
               balanced on everything that matters.
  stratified -- weights each segment's effect by its POPULATION share rather
               than its in-arm share. Immune to mix shift; this is the fix
               for the Simpson's paradox scenario.
  CUPED     -- subtracts off the part of the outcome predictable from a
               pre-period covariate. Same expectation, lower variance.
"""

from __future__ import annotations

from dataclasses import dataclass

from .population import PopulationSpec, User, generate_users, realize_outcomes
from .rng import Rng
from .stats import (
    TestResult,
    srm_check,
    two_proportion_ztest,
    welch_ttest,
    welch_ttest_from_samples,
)


@dataclass
class ArmSummary:
    name: str
    n: int
    conversions: int

    @property
    def rate(self) -> float:
        return self.conversions / self.n if self.n else 0.0


def assign(rng: Rng, split: float = 0.5, bias: float = 0.0) -> str:
    """Bucket one user.

    `bias` deliberately corrupts the randomizer: it is the knob that creates a
    Sample Ratio Mismatch. A real SRM usually comes from something mundane --
    a redirect that drops slow mobile clients, a bot filter applied to only
    one arm, a treatment that crashes before the exposure event fires. The
    mechanism differs; the signature in the data is identical.
    """
    return "treatment" if rng.uniform() < (split + bias) else "control"


# --------------------------------------------------------------------------
# Fast path: aggregate trials for calibration studies
# --------------------------------------------------------------------------


def run_aggregate_trial(
    n_per_arm: int,
    control_rate: float,
    treatment_rate: float,
    rng: Rng,
    alpha: float = 0.05,
) -> TestResult:
    """One complete experiment, summarized in two binomial draws.

    Valid because users are exchangeable here: if nothing distinguishes one
    user from another, the sum of n Bernoulli draws IS a binomial draw. The
    shortcut is what makes 20,000 replications feasible in pure Python.
    """
    conv_c = rng.binomial(n_per_arm, control_rate)
    conv_t = rng.binomial(n_per_arm, treatment_rate)
    return two_proportion_ztest(conv_c, n_per_arm, conv_t, n_per_arm, alpha)


# --------------------------------------------------------------------------
# Full path: user-level experiments
# --------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    users: list[User]
    naive: TestResult
    srm: dict
    spec: PopulationSpec

    @property
    def control(self) -> list[User]:
        return [u for u in self.users if u.arm == "control"]

    @property
    def treatment(self) -> list[User]:
        return [u for u in self.users if u.arm == "treatment"]


def run_user_level_trial(
    spec: PopulationSpec,
    n_users: int,
    rng: Rng,
    split: float = 0.5,
    assignment_bias: float = 0.0,
    alpha: float = 0.05,
) -> ExperimentResult:
    """Generate users, assign them, realize outcomes, and run the health checks."""
    users = generate_users(spec, n_users, rng.spawn("users"))

    assign_rng = rng.spawn("assign")
    for u in users:
        u.arm = assign(assign_rng, split, assignment_bias)

    realize_outcomes(spec, users, rng.spawn("outcomes"))

    n_c = sum(1 for u in users if u.arm == "control")
    n_t = len(users) - n_c
    conv_c = sum(u.converted for u in users if u.arm == "control")
    conv_t = sum(u.converted for u in users if u.arm == "treatment")

    return ExperimentResult(
        users=users,
        naive=two_proportion_ztest(conv_c, n_c, conv_t, n_t, alpha),
        srm=srm_check([n_c, n_t], [1.0 - split, split]),
        spec=spec,
    )


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------


def stratified_estimate(result: ExperimentResult) -> dict:
    """Post-stratified treatment effect, weighted by population shares.

    The naive pooled estimate silently weights each segment by how many of
    its users landed in each arm. If the arms have different segment mixes --
    which happens whenever exposure logging is imperfect -- the pooled number
    reflects the mix difference as much as the treatment. Re-weighting by the
    known population share removes that channel entirely.
    """
    shares = {s.name: s.share for s in result.spec.segments}
    per_segment = {}

    for seg_name in shares:
        c = [u for u in result.control if u.segment == seg_name]
        t = [u for u in result.treatment if u.segment == seg_name]
        if not c or not t:
            continue
        rate_c = sum(u.converted for u in c) / len(c)
        rate_t = sum(u.converted for u in t) / len(t)
        per_segment[seg_name] = {
            "rate_control": rate_c,
            "rate_treatment": rate_t,
            "absolute_lift": rate_t - rate_c,
            "relative_lift": (rate_t - rate_c) / rate_c if rate_c else float("nan"),
            "n_control": len(c),
            "n_treatment": len(t),
            "share": shares[seg_name],
        }

    if not per_segment:
        return {"absolute_lift": float("nan"), "per_segment": {}}

    weight_total = sum(v["share"] for v in per_segment.values())
    abs_lift = sum(v["absolute_lift"] * v["share"] for v in per_segment.values()) / weight_total
    baseline = sum(v["rate_control"] * v["share"] for v in per_segment.values()) / weight_total

    return {
        "absolute_lift": abs_lift,
        "relative_lift": abs_lift / baseline if baseline else float("nan"),
        "baseline": baseline,
        "per_segment": per_segment,
    }


def cuped_estimate(
    result: ExperimentResult,
    alpha: float = 0.05,
    metric=lambda u: float(u.converted),
    covariate=lambda u: u.pre_period_value,
) -> dict:
    """CUPED: Controlled-experiment Using Pre-Experiment Data.

    Replace the outcome Y with the adjusted outcome

        Y_cuped = Y - theta * (X - E[X])

    where X is a pre-experiment covariate and theta = Cov(Y, X) / Var(X).
    Because X is measured before assignment, its mean is equal across arms in
    expectation, so subtracting it leaves the treatment effect unbiased while
    stripping out the variance X explains. The variance drops by a factor of
    (1 - rho^2), which converts directly into either a shorter experiment or
    a smaller detectable effect -- the reason CUPED is standard at every
    company running experiments at scale.

    `metric` and `covariate` are accessors, so the same estimator runs against
    the binary conversion flag and against the continuous engagement metric
    without duplicating any logic. That matters here: the two cases behave
    very differently, and the comparison is the point of the scenario.

    Reference: Deng, Xu, Kohavi & Walker, "Improving the Sensitivity of Online
    Controlled Experiments by Utilizing Pre-Experiment Data" (WSDM 2013).
    """
    users = result.users
    n = len(users)
    if n < 3:
        raise ValueError("not enough users for CUPED")

    xs = [covariate(u) for u in users]
    ys = [metric(u) for u in users]

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / (n - 1)
    var_x = sum((x - mean_x) ** 2 for x in xs) / (n - 1)
    var_y = sum((y - mean_y) ** 2 for y in ys) / (n - 1)
    theta = cov / var_x if var_x > 0 else 0.0

    def adjusted(u: User) -> float:
        return metric(u) - theta * (covariate(u) - mean_x)

    c_vals = [adjusted(u) for u in result.control]
    t_vals = [adjusted(u) for u in result.treatment]

    def mean_var(vals):
        m = sum(vals) / len(vals)
        v = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
        return m, v

    mc, vc = mean_var(c_vals)
    mt, vt = mean_var(t_vals)
    test = welch_ttest(mc, vc, len(c_vals), mt, vt, len(t_vals), alpha)

    raw_c = [metric(u) for u in result.control]
    raw_t = [metric(u) for u in result.treatment]
    plain_test = welch_ttest_from_samples(raw_c, raw_t, alpha)

    _, raw_vc = mean_var(raw_c)
    _, raw_vt = mean_var(raw_t)
    pooled_raw = (raw_vc + raw_vt) / 2.0
    pooled_adj = (vc + vt) / 2.0

    # The ACHIEVED correlation between covariate and metric. This is the
    # number that predicts the variance reduction -- not the rho the
    # population was configured with. For a binary outcome the two differ
    # sharply, because a Bernoulli indicator cannot correlate with a
    # continuous covariate as strongly as two continuous variables can.
    achieved_rho = cov / (var_x**0.5 * var_y**0.5) if var_x > 0 and var_y > 0 else 0.0

    return {
        "theta": theta,
        "achieved_correlation": achieved_rho,
        "predicted_reduction": achieved_rho**2,
        "test": test,
        "plain_test": plain_test,
        "variance_raw": pooled_raw,
        "variance_cuped": pooled_adj,
        "variance_reduction": 1.0 - pooled_adj / pooled_raw if pooled_raw > 0 else 0.0,
    }


def daily_cumulative(result: ExperimentResult) -> list[dict]:
    """Cumulative arm statistics by day -- the input to any time-series read."""
    days = max(u.day for u in result.users) + 1
    out = []
    n_c = n_t = conv_c = conv_t = 0
    for d in range(days):
        for u in result.users:
            if u.day != d:
                continue
            if u.arm == "control":
                n_c += 1
                conv_c += u.converted
            else:
                n_t += 1
                conv_t += u.converted
        if n_c > 1 and n_t > 1 and conv_c > 0 and conv_t > 0:
            out.append(
                {
                    "day": d + 1,
                    "n_control": n_c,
                    "n_treatment": n_t,
                    "conv_control": conv_c,
                    "conv_treatment": conv_t,
                    "rate_control": conv_c / n_c,
                    "rate_treatment": conv_t / n_t,
                    "test": two_proportion_ztest(conv_c, n_c, conv_t, n_t),
                }
            )
    return out
