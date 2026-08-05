"""Tests for the mSPRT.

The load-bearing test is `test_false_positive_rate_under_continuous_peeking`:
it stops at the first crossing across many A/A runs and asserts the error
rate stays under alpha. That is the entire promise of always-valid inference,
so if only one test in this repo survives, it should be that one.
"""

import unittest

from abtest.rng import Rng
from abtest.sequential import MSPRT, fixed_horizon_peek_risk
from abtest.stats import two_proportion_ztest


class TestMSPRTBasics(unittest.TestCase):
    def test_rejects_bad_parameters(self):
        with self.assertRaises(ValueError):
            MSPRT(tau=0.0)
        with self.assertRaises(ValueError):
            MSPRT(tau=-1.0)
        with self.assertRaises(ValueError):
            MSPRT(tau=0.01, alpha=0.0)
        with self.assertRaises(ValueError):
            MSPRT(tau=0.01, alpha=1.0)

    def test_p_value_is_monotone_non_increasing(self):
        """The always-valid p-value is a running minimum and can never rise."""
        rng = Rng(7)
        m = MSPRT(tau=0.005)
        n = c = t = 0
        last = 1.0
        for _ in range(40):
            n += 2_000
            c += rng.binomial(2_000, 0.085)
            t += rng.binomial(2_000, 0.085)
            p = m.update_proportions(c, n, t, n).p_value
            self.assertLessEqual(p, last + 1e-15)
            last = p

    def test_no_effect_keeps_p_value_high(self):
        rng = Rng(11)
        m = MSPRT(tau=0.005)
        n = c = t = 0
        for _ in range(30):
            n += 5_000
            c += rng.binomial(5_000, 0.085)
            t += rng.binomial(5_000, 0.085)
            state = m.update_proportions(c, n, t, n)
        self.assertGreater(state.p_value, 0.05)

    def test_large_effect_is_detected(self):
        rng = Rng(13)
        m = MSPRT(tau=0.01)
        n = c = t = 0
        fired = False
        for _ in range(30):
            n += 5_000
            c += rng.binomial(5_000, 0.085)
            t += rng.binomial(5_000, 0.110)  # ~+29%, unmissable
            if m.update_proportions(c, n, t, n).p_value < 0.05:
                fired = True
                break
        self.assertTrue(fired)

    def test_history_is_recorded(self):
        rng = Rng(3)
        m = MSPRT(tau=0.005)
        n = c = t = 0
        for _ in range(5):
            n += 1_000
            c += rng.binomial(1_000, 0.085)
            t += rng.binomial(1_000, 0.085)
            m.update_proportions(c, n, t, n)
        self.assertEqual(len(m.history), 5)
        m.reset()
        self.assertEqual(len(m.history), 0)

    def test_requires_minimum_sample(self):
        m = MSPRT(tau=0.005)
        with self.assertRaises(ValueError):
            m.update(0.1, 0.09, 1, 0.1, 0.09, 50)


class TestConfidenceSequence(unittest.TestCase):
    def test_interval_contains_estimate(self):
        m = MSPRT(tau=0.005)
        s = m.update_proportions(850, 10_000, 900, 10_000)
        self.assertLessEqual(s.ci_low, s.estimate)
        self.assertGreaterEqual(s.ci_high, s.estimate)

    def test_interval_narrows_with_more_data(self):
        widths = []
        for n in (10_000, 100_000, 1_000_000):
            m = MSPRT(tau=0.005)
            s = m.update_proportions(int(0.085 * n), n, int(0.085 * n), n)
            widths.append(s.ci_high - s.ci_low)
        self.assertEqual(widths, sorted(widths, reverse=True))

    def test_wider_than_fixed_horizon_interval(self):
        """Always-valid intervals must be conservative -- that is the point.

        A confidence sequence covers at every sample size simultaneously,
        which cannot be done as tightly as covering at one pre-chosen size.
        If it were ever narrower than the fixed-horizon CI, something would
        be badly wrong.
        """
        n, conv_c, conv_t = 50_000, 4_250, 4_400
        fixed = two_proportion_ztest(conv_c, n, conv_t, n)
        seq = MSPRT(tau=0.005).update_proportions(conv_c, n, conv_t, n)
        self.assertGreater(
            seq.ci_high - seq.ci_low, fixed.ci_high - fixed.ci_low
        )


class TestAlwaysValidGuarantee(unittest.TestCase):
    def test_false_positive_rate_under_continuous_peeking(self):
        """Stop at the first crossing; the error rate must stay under alpha.

        This is the claim the whole method exists to support. The naive
        comparison in the same loop shows what a fixed-horizon test does
        under identical conditions.
        """
        rng = Rng(20260803).spawn("guarantee_test")
        alpha = 0.05
        trials, days, per_day = 600, 14, 4_000
        rate = 0.085

        seq_fp = naive_fp = 0
        for _ in range(trials):
            m = MSPRT(tau=rate * 0.05, alpha=alpha)
            n = c = t = 0
            seq_fired = naive_fired = False
            for _day in range(days):
                n += per_day
                c += rng.binomial(per_day, rate)
                t += rng.binomial(per_day, rate)
                if not seq_fired and m.update_proportions(c, n, t, n).p_value < alpha:
                    seq_fired = True
                if not naive_fired and two_proportion_ztest(c, n, t, n, alpha).significant:
                    naive_fired = True
            seq_fp += seq_fired
            naive_fp += naive_fired

        seq_rate = seq_fp / trials
        naive_rate = naive_fp / trials

        # The guarantee, with a small allowance for Monte Carlo noise.
        self.assertLess(seq_rate, alpha + 0.02)
        # And the contrast that makes it worth having.
        self.assertGreater(naive_rate, 2 * alpha)


class TestPeekRiskBound(unittest.TestCase):
    def test_single_look_is_alpha(self):
        self.assertAlmostEqual(fixed_horizon_peek_risk(1, 0.05), 0.05, places=12)

    def test_increases_with_looks(self):
        risks = [fixed_horizon_peek_risk(k, 0.05) for k in (1, 5, 14, 30)]
        self.assertEqual(risks, sorted(risks))


if __name__ == "__main__":
    unittest.main()
