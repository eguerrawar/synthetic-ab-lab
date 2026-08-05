"""Tests for population generation, assignment, and the estimators.

The population is the foundation of every result in the repo -- if the
synthetic users are not what they claim to be, every scenario is measuring
the generator's bugs rather than the analysis method's behavior. These tests
verify the generator's contract: correct marginal rates, correct correlation
structure, reproducibility from a seed, and estimators that recover a known
planted effect.
"""

import unittest

from abtest.experiment import (
    assign,
    cuped_estimate,
    daily_cumulative,
    run_aggregate_trial,
    run_user_level_trial,
    stratified_estimate,
)
from abtest.population import (
    DEFAULT_SEGMENTS,
    PopulationSpec,
    Segment,
    effect_on_day,
    generate_users,
)
from abtest.rng import Rng


class TestRng(unittest.TestCase):
    def test_same_seed_same_stream(self):
        a = [Rng(42).uniform() for _ in range(3)]
        b = [Rng(42).uniform() for _ in range(3)]
        self.assertEqual(a, b)

    def test_different_seeds_differ(self):
        self.assertNotEqual(Rng(1).uniform(), Rng(2).uniform())

    def test_spawn_is_deterministic_and_independent(self):
        self.assertEqual(Rng(5).spawn("x").uniform(), Rng(5).spawn("x").uniform())
        self.assertNotEqual(Rng(5).spawn("x").uniform(), Rng(5).spawn("y").uniform())

    def test_binomial_edge_cases(self):
        r = Rng(1)
        self.assertEqual(r.binomial(0, 0.5), 0)
        self.assertEqual(r.binomial(100, 0.0), 0)
        self.assertEqual(r.binomial(100, 1.0), 100)

    def test_binomial_mean_is_np(self):
        r = Rng(9)
        draws = [r.binomial(10_000, 0.085) for _ in range(400)]
        self.assertAlmostEqual(sum(draws) / len(draws), 850, delta=15)

    def test_bulk_choice_respects_weights(self):
        r = Rng(4)
        picks = r.choice_weighted_bulk(["a", "b"], [0.8, 0.2], 20_000)
        self.assertAlmostEqual(picks.count("a") / 20_000, 0.8, delta=0.02)


class TestSegments(unittest.TestCase):
    def test_treated_rate_applies_multiplier(self):
        seg = Segment("s", 1.0, 0.10, effect_multiplier=2.0)
        self.assertAlmostEqual(seg.treated_rate(0.05), 0.10 * 1.10, places=12)

    def test_treated_rate_clamped_to_valid_probability(self):
        seg = Segment("s", 1.0, 0.9, effect_multiplier=1.0)
        self.assertLess(seg.treated_rate(5.0), 1.0)

    def test_shares_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            PopulationSpec(segments=[Segment("a", 0.3, 0.1), Segment("b", 0.3, 0.1)])

    def test_default_segments_are_valid(self):
        spec = PopulationSpec()
        self.assertAlmostEqual(sum(s.share for s in spec.segments), 1.0, places=12)
        # 0.45(.052) + 0.20(.031) + 0.25(.118) + 0.10(.086) = 0.0677
        self.assertAlmostEqual(spec.blended_baseline, 0.0677, places=6)

    def test_blended_effect_differs_from_nominal_under_heterogeneity(self):
        """The population-average effect is not the per-segment parameter.

        With unequal effect multipliers these two genuinely differ, and
        conflating them would make every scenario compare against the wrong
        ground truth.
        """
        spec = PopulationSpec(true_relative_effect=0.10)
        self.assertNotAlmostEqual(spec.blended_true_effect, 0.10, places=3)
        self.assertGreater(spec.blended_true_effect, 0.0)

    def test_homogeneous_population_recovers_nominal_effect(self):
        spec = PopulationSpec(
            segments=[Segment("only", 1.0, 0.085, 1.0)], true_relative_effect=0.10
        )
        self.assertAlmostEqual(spec.blended_true_effect, 0.10, places=9)


class TestNoveltyDecay(unittest.TestCase):
    def test_no_half_life_means_constant_effect(self):
        spec = PopulationSpec(true_relative_effect=0.08)
        self.assertEqual(effect_on_day(spec, 0), 0.08)
        self.assertEqual(effect_on_day(spec, 27), 0.08)

    def test_halves_at_the_half_life(self):
        spec = PopulationSpec(true_relative_effect=0.12, novelty_half_life_days=4.0)
        self.assertAlmostEqual(effect_on_day(spec, 0), 0.12, places=12)
        self.assertAlmostEqual(effect_on_day(spec, 4), 0.06, places=12)
        self.assertAlmostEqual(effect_on_day(spec, 8), 0.03, places=12)


class TestPopulationGeneration(unittest.TestCase):
    def test_segment_shares_are_respected(self):
        spec = PopulationSpec()
        users = generate_users(spec, 60_000, Rng(2))
        for seg in DEFAULT_SEGMENTS:
            observed = sum(1 for u in users if u.segment == seg.name) / len(users)
            self.assertAlmostEqual(observed, seg.share, delta=0.01)

    def test_covariate_correlation_is_as_configured(self):
        """corr(sessions_pre, sessions_post) must equal the configured rho.

        This is the property the CUPED scenario's rho^2 prediction rests on.
        """
        for rho in (0.3, 0.7, 0.9):
            spec = PopulationSpec(covariate_correlation=rho, true_relative_effect=0.0)
            r = Rng(17)
            users = generate_users(spec, 40_000, r)
            for u in users:
                u.arm = "control"
            from abtest.population import realize_outcomes

            realize_outcomes(spec, users, r.spawn("out"))
            self.assertAlmostEqual(
                _pearson(
                    [u.sessions_pre for u in users], [u.sessions_post for u in users]
                ),
                rho,
                delta=0.02,
                msg=f"rho={rho}",
            )

    def test_reproducible_from_seed(self):
        spec = PopulationSpec(true_relative_effect=0.05)
        a = run_user_level_trial(spec, 5_000, Rng(99))
        b = run_user_level_trial(spec, 5_000, Rng(99))
        self.assertEqual(a.naive.p_value, b.naive.p_value)
        self.assertEqual(a.naive.treatment_mean, b.naive.treatment_mean)


class TestAssignment(unittest.TestCase):
    def test_unbiased_split_is_balanced(self):
        r = Rng(21)
        arms = [assign(r) for _ in range(40_000)]
        self.assertAlmostEqual(arms.count("treatment") / 40_000, 0.5, delta=0.01)

    def test_bias_shifts_the_split(self):
        r = Rng(22)
        arms = [assign(r, bias=0.05) for _ in range(40_000)]
        self.assertAlmostEqual(arms.count("treatment") / 40_000, 0.55, delta=0.01)

    def test_clean_experiment_passes_srm(self):
        spec = PopulationSpec(true_relative_effect=0.0)
        result = run_user_level_trial(spec, 80_000, Rng(23))
        self.assertFalse(result.srm["failed"])

    def test_biased_experiment_fails_srm(self):
        spec = PopulationSpec(true_relative_effect=0.0)
        result = run_user_level_trial(spec, 200_000, Rng(24), assignment_bias=0.02)
        self.assertTrue(result.srm["failed"])


class TestAggregateTrial(unittest.TestCase):
    def test_recovers_planted_effect_on_average(self):
        r = Rng(31)
        lifts = [
            run_aggregate_trial(200_000, 0.085, 0.085 * 1.10, r).relative_lift
            for _ in range(60)
        ]
        self.assertAlmostEqual(sum(lifts) / len(lifts), 0.10, delta=0.01)


class TestEstimators(unittest.TestCase):
    def test_stratified_recovers_effect_on_clean_data(self):
        spec = PopulationSpec(true_relative_effect=0.15)
        result = run_user_level_trial(spec, 250_000, Rng(41))
        strat = stratified_estimate(result)
        self.assertAlmostEqual(
            strat["relative_lift"], spec.blended_true_effect, delta=0.03
        )

    def test_stratified_reports_every_segment(self):
        spec = PopulationSpec(true_relative_effect=0.10)
        result = run_user_level_trial(spec, 60_000, Rng(42))
        self.assertEqual(len(strat_names(result)), len(spec.segments))

    def test_cuped_does_not_bias_the_estimate(self):
        """CUPED must change the variance, not the expected effect."""
        spec = PopulationSpec(true_relative_effect=0.0, covariate_correlation=0.9)
        result = run_user_level_trial(spec, 120_000, Rng(43))
        adj = cuped_estimate(
            result, metric=lambda u: u.sessions_post, covariate=lambda u: u.sessions_pre
        )
        self.assertAlmostEqual(
            adj["test"].absolute_lift, adj["plain_test"].absolute_lift, delta=0.05
        )

    def test_cuped_reduces_variance_by_rho_squared(self):
        spec = PopulationSpec(true_relative_effect=0.05, covariate_correlation=0.8)
        result = run_user_level_trial(spec, 120_000, Rng(44))
        adj = cuped_estimate(
            result, metric=lambda u: u.sessions_post, covariate=lambda u: u.sessions_pre
        )
        self.assertAlmostEqual(
            adj["variance_reduction"], adj["achieved_correlation"] ** 2, delta=0.02
        )

    def test_cuped_with_useless_covariate_does_no_harm(self):
        spec = PopulationSpec(true_relative_effect=0.05, covariate_correlation=0.0)
        result = run_user_level_trial(spec, 60_000, Rng(45))
        adj = cuped_estimate(
            result, metric=lambda u: u.sessions_post, covariate=lambda u: u.sessions_pre
        )
        self.assertGreaterEqual(adj["variance_reduction"], -0.01)


class TestDailyCumulative(unittest.TestCase):
    def test_counts_accumulate_monotonically(self):
        spec = PopulationSpec(true_relative_effect=0.05, n_days=10)
        result = run_user_level_trial(spec, 60_000, Rng(51))
        rows = daily_cumulative(result)
        self.assertGreater(len(rows), 1)
        ns = [r["n_control"] for r in rows]
        self.assertEqual(ns, sorted(ns))

    def test_final_row_matches_overall_test(self):
        spec = PopulationSpec(true_relative_effect=0.05, n_days=10)
        result = run_user_level_trial(spec, 60_000, Rng(52))
        rows = daily_cumulative(result)
        self.assertAlmostEqual(
            rows[-1]["test"].relative_lift, result.naive.relative_lift, places=9
        )


def strat_names(result):
    return stratified_estimate(result)["per_segment"].keys()


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy)


if __name__ == "__main__":
    unittest.main()
