# Build notes: what Claude Code actually did here

This project was built with Claude Code. Since the interview is partly about
whether I can work with these tools, here is an honest account rather than a
demo-day version.

## The division of labor

**I decided:** the scope (which seven failure modes are worth demonstrating),
the zero-dependency constraint, the scenario contract, that mSPRT was the right
answer to peeking, that CUPED needed both a binary and a continuous metric, and
that every scenario must declare a pass criterion before showing numbers.

**Claude wrote:** most of the implementation, once those decisions were made.
Distribution functions, the SVG chart layer, the test suite scaffolding.

**The actual work was reviewing output.** Three examples below, because "I used
AI and it worked" is not an interesting claim. What matters is the failure rate
and how it gets caught.

## Bug 1 — a demo that couldn't support its own conclusion

The SRM scenario simulates a bug where 2% of low-propensity users silently drop
out of the treatment arm, and shows that this fabricates a lift out of nothing.
It ran at 200,000 users and reported a **+1.79% fabricated lift**.

That number was fine. The problem surfaced when I checked the p-value it was
being reported alongside: **0.284**. Not significant.

So the scenario was demonstrating a bias that a real analyst would have
correctly ignored as noise. The write-up claimed the lift was significant and
would ship. It wouldn't have.

The fix wasn't to soften the claim — it was to recognize that the demo was
underpowered to make its own point. The bias is ~2% at any sample size, but
only becomes *significant* at real traffic volumes. Re-running at 4M users:
**+2.00% at p=8.9e-08**, blocked by the SRM gate at p=5e-79. That is the actual
lesson: SRM is dangerous because scale converts a small bias into a significant
one.

I also added the significance of the fake lift to the scenario's own pass
criterion, so it can't silently regress.

**Takeaway:** the model produced correct code and a plausible narrative that
the code didn't support. Checking the numbers against the claim is the job.

## Bug 2 — a test that failed 5% of the time by design

The novelty scenario asserted that the final-week confidence interval contained
the true final-week effect. It failed on one run.

Nothing was wrong. A 95% interval misses 5% of the time — that's what 95%
means. The test was asserting on a single random draw, so it was guaranteed to
fail roughly one run in twenty, which would be indistinguishable from a real
regression.

Replaced with a **coverage rate over 400 replications**: measure how often the
interval covers, and assert it lands between 92% and 98%. It now reports 94.8%
coverage and −0.05pp bias, which is a much stronger statement — it says the
estimator is unbiased, not that one interval happened to land well.

**Takeaway:** in stochastic code, "the test failed" and "the code is broken"
are different events, and knowing which one you're looking at requires
understanding the statistics.

## Bug 3 — four wrong expected values in the test suite

First full test run: 4 failures out of 89. All four were wrong numbers in the
*tests*, not bugs in the implementation:

- `Φ(-1.96)` asserted as exactly 0.025. It's 0.0249979 — the round number
  belongs to −1.959964.
- Sample size for a 20%→22% lift asserted as 6,850 (a figure I'd half-remembered
  from an online calculator). The standard uncorrected two-proportion formula
  gives **6,510**; calculators reporting ~6,850 apply a continuity correction,
  which is a different formula, not a disagreement.
- A blended baseline rate asserted as 0.0713 when the segment shares multiply
  out to 0.0677.
- An SRM p-value threshold of 1e-10 when the true value is 1.67e-10.

I verified each by hand before touching anything. The temptation with a failing
test is to adjust the code until it passes; here that would have introduced
four real bugs to satisfy four wrong assertions.

**Takeaway:** this is the failure mode people worry about with AI-assisted
code, and the defense is having independent reference values rather than
trusting either the code or the test.

## Two rendering bugs, for completeness

The HTML report had a y-axis that padded an explicit `y_lo=0` into negative
territory (a "−6%" tick on a chart of variance reduction), and a dashed
reference line drawn thick enough to hide the measured series underneath it —
which mattered precisely because the two overlap when the theory is correct.
Both caught by looking at the rendered output rather than the code.

## What I'd tell a teammate

Claude Code was a large multiplier on implementation speed — this is roughly a
week of evenings compressed into a day. It was **not** a substitute for knowing
what the numbers should be. Every one of the bugs above was caught by domain
knowledge, not by the tooling, and two of them were cases where the code was
correct and the surrounding claim or test was wrong.

The workflow that worked: make the architectural and statistical decisions
myself, let the model implement, then review output against independent
expectations — reference tables, hand calculations, and "does this number
actually support the sentence next to it."
