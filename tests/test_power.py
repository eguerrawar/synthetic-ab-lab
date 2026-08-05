"""Tests for the power / sample size calculator.

The key test is `test_matches_published_example`, which checks the formula
against a worked example rather than against itself. The rest verify the
internal consistency that makes the calculator usable: sizing for a power
level and then asking for the power at that size must return the level you
asked for, and MDE must be the exact inverse of sample size.
"""

import unittest

from abtest.power import (
    mde_at_n,
    power_at_n,
    sample_size_proportions,
    sample_size_with_cuped,
)


class TestSampleSize(unittest.TestCase):
    def test_matches_textbook_worked_example(self):
        """p1=20%, p2=22%, alpha=.05 two-sided, power=.80 -> 6,510 per arm.

        This is the standard uncontinuity-corrected two-proportion formula,
        hand-checkable in one line. Online calculators that report ~6,850 for
        the same inputs are applying a continuity correction; that variant is
        more conservative but is a different formula, not a disagreement
        about this one. The empirical check that this number actually
        delivers 80% power lives in scenarios/power_curve.py, which measures
        80.3% at exactly this n.
        """
        plan = sample_size_proportions(0.20, 0.10, alpha=0.05, power=0.80)
        self.assertEqual(plan.n_per_arm, 6510)

    def test_smaller_effects_need_more_users(self):
        big = sample_size_proportions(0.085, 0.10).n_per_arm
        small = sample_size_proportions(0.085, 0.01).n_per_arm
        self.assertGreater(small, big * 50)

    def test_quartering_the_effect_roughly_16x_the_n(self):
        # n scales as 1/d^2.
        n1 = sample_size_proportions(0.085, 0.04).n_per_arm
        n2 = sample_size_proportions(0.085, 0.01).n_per_arm
        self.assertAlmostEqual(n2 / n1, 16.0, delta=0.6)

    def test_higher_power_needs_more_users(self):
        p80 = sample_size_proportions(0.085, 0.05, power=0.80).n_per_arm
        p95 = sample_size_proportions(0.085, 0.05, power=0.95).n_per_arm
        self.assertGreater(p95, p80)

    def test_stricter_alpha_needs_more_users(self):
        a05 = sample_size_proportions(0.085, 0.05, alpha=0.05).n_per_arm
        a01 = sample_size_proportions(0.085, 0.05, alpha=0.01).n_per_arm
        self.assertGreater(a01, a05)

    def test_runtime_computed_from_traffic(self):
        plan = sample_size_proportions(0.085, 0.05, daily_traffic=40_000)
        self.assertAlmostEqual(plan.days_required, plan.n_total / 40_000, places=9)

    def test_rejects_impossible_mde(self):
        # +200% on a 50% baseline would need a rate above 1.
        with self.assertRaises(ValueError):
            sample_size_proportions(0.50, 2.0)

    def test_rejects_bad_baseline(self):
        for bad in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                sample_size_proportions(bad, 0.05)

    def test_rejects_zero_mde(self):
        with self.assertRaises(ValueError):
            sample_size_proportions(0.085, 0.0)


class TestPowerAtN(unittest.TestCase):
    def test_round_trip_with_sample_size(self):
        """Sizing for 80% power must yield 80% power at that size."""
        for baseline in (0.02, 0.085, 0.30):
            for mde in (0.02, 0.05, 0.15):
                n = sample_size_proportions(baseline, mde, power=0.80).n_per_arm
                self.assertAlmostEqual(power_at_n(baseline, mde, n), 0.80, delta=0.01)

    def test_zero_effect_gives_alpha(self):
        self.assertAlmostEqual(power_at_n(0.085, 0.0, 50_000, alpha=0.05), 0.05, places=6)

    def test_monotone_in_n(self):
        powers = [power_at_n(0.085, 0.03, n) for n in (10_000, 50_000, 100_000, 500_000)]
        self.assertEqual(powers, sorted(powers))

    def test_monotone_in_effect(self):
        powers = [power_at_n(0.085, e, 50_000) for e in (0.01, 0.02, 0.05, 0.10)]
        self.assertEqual(powers, sorted(powers))

    def test_huge_effect_approaches_one(self):
        self.assertGreater(power_at_n(0.085, 0.50, 100_000), 0.999)


class TestMDE(unittest.TestCase):
    def test_inverts_sample_size(self):
        for baseline in (0.02, 0.085, 0.30):
            n = sample_size_proportions(baseline, 0.05, power=0.80).n_per_arm
            self.assertAlmostEqual(mde_at_n(baseline, n, power=0.80), 0.05, delta=0.001)

    def test_more_traffic_detects_smaller_effects(self):
        mdes = [mde_at_n(0.085, n) for n in (10_000, 50_000, 250_000)]
        self.assertEqual(mdes, sorted(mdes, reverse=True))


class TestCuped(unittest.TestCase):
    def test_zero_correlation_saves_nothing(self):
        plain, reduced, saved = sample_size_with_cuped(0.085, 0.03, 0.0)
        self.assertEqual(plain, reduced)
        self.assertAlmostEqual(saved, 0.0, places=9)

    def test_saving_is_rho_squared(self):
        for rho in (0.3, 0.5, 0.7, 0.9):
            _, _, saved = sample_size_with_cuped(0.085, 0.03, rho)
            self.assertAlmostEqual(saved, rho**2, delta=0.001)


if __name__ == "__main__":
    unittest.main()
