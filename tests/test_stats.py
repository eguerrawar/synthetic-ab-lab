"""Tests for the statistical primitives.

These check against PUBLISHED reference values, not against the code's own
output. A test that only asserts the function returns what it currently
returns will happily lock in a bug forever.
"""

import math
import unittest

from abtest.stats import (
    betainc,
    chi2_sf,
    norm_cdf,
    norm_ppf,
    norm_sf,
    srm_check,
    t_ppf,
    t_sf,
    two_proportion_ztest,
    welch_ttest,
    welch_ttest_from_samples,
)


class TestNormal(unittest.TestCase):
    def test_cdf_reference_values(self):
        # Standard normal table values.
        for z, expected in [
            (-3.0, 0.001350),
            # Note: Phi(-1.96) is 0.0249979, not exactly 0.025. The round
            # number belongs to -1.959964, which is why norm_ppf(0.025)
            # returns that and not -1.96.
            (-1.96, 0.024998),
            (-1.0, 0.158655),
            (0.0, 0.500000),
            (1.0, 0.841345),
            (1.645, 0.950015),
            (1.96, 0.975002),
            (2.576, 0.995002),
        ]:
            self.assertAlmostEqual(norm_cdf(z), expected, places=6, msg=f"z={z}")

    def test_sf_is_complement(self):
        for z in (-4.0, -1.0, 0.0, 0.5, 3.3):
            self.assertAlmostEqual(norm_sf(z), 1.0 - norm_cdf(z), places=12)

    def test_sf_accurate_in_far_tail(self):
        # 1 - cdf loses all precision here; erfc does not.
        self.assertAlmostEqual(norm_sf(8.0), 6.22096e-16, delta=1e-20)

    def test_ppf_reference_values(self):
        for p, expected in [
            (0.001, -3.090232),
            (0.025, -1.959964),
            (0.05, -1.644854),
            (0.5, 0.0),
            (0.95, 1.644854),
            (0.975, 1.959964),
            (0.999, 3.090232),
        ]:
            self.assertAlmostEqual(norm_ppf(p), expected, places=6, msg=f"p={p}")

    def test_ppf_inverts_cdf(self):
        for p in (0.001, 0.01, 0.2, 0.5, 0.8, 0.99, 0.999):
            self.assertAlmostEqual(norm_cdf(norm_ppf(p)), p, places=12)

    def test_ppf_rejects_out_of_range(self):
        for bad in (0.0, 1.0, -0.5, 2.0):
            with self.assertRaises(ValueError):
                norm_ppf(bad)


class TestStudentT(unittest.TestCase):
    def test_two_sided_critical_values(self):
        # Classic t-table: two-sided 5% critical values.
        for df, crit in [(1, 12.706), (5, 2.571), (10, 2.228), (30, 2.042), (100, 1.984)]:
            self.assertAlmostEqual(2 * t_sf(crit, df), 0.05, places=3, msg=f"df={df}")

    def test_converges_to_normal(self):
        self.assertAlmostEqual(t_sf(1.96, 1_000_000), norm_sf(1.96), places=5)

    def test_symmetry(self):
        for df in (3, 12, 60):
            self.assertAlmostEqual(t_sf(1.5, df), 1.0 - t_sf(-1.5, df), places=12)

    def test_ppf_inverts(self):
        for df in (4, 25, 200):
            for p in (0.01, 0.25, 0.5, 0.9, 0.99):
                self.assertAlmostEqual(1.0 - t_sf(t_ppf(p, df), df), p, places=6)


class TestBetaAndChiSquare(unittest.TestCase):
    def test_betainc_endpoints(self):
        self.assertEqual(betainc(2.0, 3.0, 0.0), 0.0)
        self.assertEqual(betainc(2.0, 3.0, 1.0), 1.0)

    def test_betainc_symmetry_identity(self):
        # I_x(a,b) == 1 - I_{1-x}(b,a)
        self.assertAlmostEqual(betainc(2.5, 3.5, 0.3), 1.0 - betainc(3.5, 2.5, 0.7), places=12)

    def test_chi2_critical_values(self):
        for df, crit in [(1, 3.841), (2, 5.991), (3, 7.815), (5, 11.070), (10, 18.307)]:
            self.assertAlmostEqual(chi2_sf(crit, df), 0.05, places=3, msg=f"df={df}")

    def test_chi2_one_df_matches_normal(self):
        # A 1-df chi-square is the square of a standard normal.
        for z in (0.5, 1.0, 1.96, 3.0):
            self.assertAlmostEqual(chi2_sf(z * z, 1), 2 * norm_sf(z), places=10)


class TestTwoProportionZTest(unittest.TestCase):
    def test_identical_arms_give_p_one(self):
        r = two_proportion_ztest(500, 10_000, 500, 10_000)
        self.assertAlmostEqual(r.p_value, 1.0, places=10)
        self.assertAlmostEqual(r.absolute_lift, 0.0, places=12)
        self.assertFalse(r.significant)

    def test_known_case(self):
        # 5.00% vs 5.50% on 20k per arm.
        r = two_proportion_ztest(1000, 20_000, 1100, 20_000)
        self.assertAlmostEqual(r.control_mean, 0.05, places=10)
        self.assertAlmostEqual(r.treatment_mean, 0.055, places=10)
        self.assertAlmostEqual(r.relative_lift, 0.10, places=10)
        self.assertAlmostEqual(r.p_value, 0.02500, places=4)
        self.assertTrue(r.significant)

    def test_confidence_interval_contains_estimate(self):
        r = two_proportion_ztest(1000, 20_000, 1100, 20_000)
        self.assertLess(r.ci_low, r.absolute_lift)
        self.assertGreater(r.ci_high, r.absolute_lift)

    def test_interval_uses_unpooled_se(self):
        # CI half-width must equal 1.96 * unpooled SE, not the pooled one.
        r = two_proportion_ztest(1000, 20_000, 1100, 20_000)
        half = (r.ci_high - r.ci_low) / 2
        self.assertAlmostEqual(half, norm_ppf(0.975) * r.standard_error, places=12)

    def test_wider_interval_at_smaller_n(self):
        small = two_proportion_ztest(50, 1_000, 55, 1_000)
        large = two_proportion_ztest(5000, 100_000, 5500, 100_000)
        self.assertGreater(small.ci_high - small.ci_low, large.ci_high - large.ci_low)

    def test_rejects_empty_arm(self):
        with self.assertRaises(ValueError):
            two_proportion_ztest(0, 0, 5, 100)


class TestWelch(unittest.TestCase):
    def test_identical_samples(self):
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        r = welch_ttest_from_samples(a, list(a))
        self.assertAlmostEqual(r.p_value, 1.0, places=10)

    def test_known_shift(self):
        control = [float(x) for x in range(1, 21)]
        treatment = [float(x) + 5.0 for x in range(1, 21)]
        r = welch_ttest_from_samples(control, treatment)
        self.assertAlmostEqual(r.absolute_lift, 5.0, places=10)
        self.assertTrue(r.significant)

    def test_unequal_variance_handled(self):
        # Welch must not blow up when the arms have very different spreads.
        r = welch_ttest(10.0, 1.0, 100, 10.5, 100.0, 100)
        self.assertGreater(r.p_value, 0.05)
        self.assertLess(r.p_value, 1.0)

    def test_requires_two_units(self):
        with self.assertRaises(ValueError):
            welch_ttest(1.0, 1.0, 1, 2.0, 1.0, 5)


class TestSRM(unittest.TestCase):
    def test_perfect_split_passes(self):
        check = srm_check([50_000, 50_000], [0.5, 0.5])
        self.assertFalse(check["failed"])
        self.assertAlmostEqual(check["p_value"], 1.0, places=10)

    def test_large_skew_caught(self):
        check = srm_check([50_000, 48_000], [0.5, 0.5])
        self.assertTrue(check["failed"])
        # chi2 = 2 * 1000^2 / 49000 = 40.82 on 1 df -> p ~ 1.7e-10
        self.assertLess(check["p_value"], 1e-9)

    def test_small_skew_on_small_sample_passes(self):
        # 51/49 on 200 users is ordinary noise, not a bug.
        check = srm_check([102, 98], [0.5, 0.5])
        self.assertFalse(check["failed"])

    def test_uneven_intended_split(self):
        # A deliberate 90/10 holdout must not be flagged.
        check = srm_check([90_000, 10_000], [0.9, 0.1])
        self.assertFalse(check["failed"])

    def test_three_arms(self):
        check = srm_check([33_333, 33_333, 33_334], [1 / 3, 1 / 3, 1 / 3])
        self.assertEqual(check["df"], 2)
        self.assertFalse(check["failed"])

    def test_chi2_matches_manual_computation(self):
        counts = [5100, 4900]
        check = srm_check(counts, [0.5, 0.5])
        expected = (5100 - 5000) ** 2 / 5000 + (4900 - 5000) ** 2 / 5000
        self.assertAlmostEqual(check["chi2"], expected, places=10)


if __name__ == "__main__":
    unittest.main()
