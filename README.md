# synthetic-ab-lab

A simulation harness for A/B testing methodology. It generates synthetic user
populations where the true treatment effect is **known by construction**, runs
real analysis methods against them, and reports whether each method recovered
the truth.

The point is inversion. On a live experiment the true lift is unknown forever,
so a broken analysis is invisible — it just returns a confident number that
nobody can check. Here the truth is a parameter, so a broken analysis is a
failing test.

```bash
python run.py run all --report report.html
```

No dependencies. No install step. No network. Python 3.12+ standard library
only — the statistics, the samplers, and the SVG charts are all implemented
from scratch in this repo.

---

## Results

All seven scenarios pass their stated criteria; full run takes ~37 seconds.

| # | Scenario | The failure it demonstrates | Headline result |
|---|---|---|---|
| 1 | **A/A calibration** | — (validates the harness) | 20,000 null experiments → **4.88%** false positive rate, p-values uniform (KS 0.008) |
| 2 | **Power validation** | Sample-size math nobody checks | Simulated power tracks the analytic curve to within **0.6pp** everywhere |
| 3 | **Peeking** | Checking the dashboard daily | False positives **5% → 22.1%** (4.4×). mSPRT holds at **0.87%** |
| 4 | **Sample Ratio Mismatch** | Non-random dropout | Fabricates a **+2.0%** lift at **p=9e-08** from a true 0%; gate blocks it at p=5e-79 |
| 5 | **Simpson's paradox** | Segment mix imbalance | Treatment wins *every* segment, pooled reads **−24.9%**; stratifying recovers **+9.2%** |
| 6 | **Novelty decay** | Reading the cumulative number | Week 1 says **+8.0%**, cumulative says **+2.7%**, durable effect is **+0.0%** |
| 7 | **CUPED** | Leaving variance on the table | Removes **81%** of variance on a continuous metric, **21%** on a binary one |

Every scenario prints a `PASS`/`FAIL` against a criterion declared **before**
the numbers are shown. A demo that only prints numbers leaves the reader to
decide whether they are good ones.

---

## The three results worth understanding

### Peeking is the expensive one

Nobody launches an experiment and then ignores the dashboard for two weeks.
People look daily and stop when the number turns green. That behavior is
rational and it completely invalidates a fixed-horizon test, whose 5%
guarantee holds only at the single sample size committed to in advance.

Simulated with 3,000 A/A experiments — identical arms, zero true effect, so
every rejection is an error:

```
  fixed-horizon z-test, peeked daily    22.1%   <-- 4.4x the nominal 5%
  mSPRT always-valid p-value             0.9%   <-- controlled
```

The fix is a mixture Sequential Probability Ratio Test. The likelihood ratio
is a martingale under the null, so Ville's inequality bounds the probability
it *ever* crosses `1/α` — which makes the running minimum of `1/Λ` a p-value
valid at every sample size simultaneously.

It is not free, and the repo prices it rather than hand-waving:

```
  traffic needed to reach 80% power:
    fixed horizon      69,121 users/arm
    mSPRT             120,000 users/arm (30 days)
    the price       1.74x traffic to buy unlimited peeking
```

**1.74× traffic for the right to stop whenever the evidence justifies it.**
For most teams that is a good trade, because the realistic alternative is not
"wait patiently" — it is peeking anyway with an uncontrolled error rate.

### Sample Ratio Mismatch is a hard gate, not a warning

A 50/50 split delivering 50.47/49.53 looks like nothing. It is a five-sigma
event, and it means whatever dropped those users dropped a *non-random* subset
— which breaks the exchangeability the entire causal claim rests on.

Modeled here as the common bug: treatment renders slightly slower, so 2% of
users who weren't going to convert anyway bounce before the exposure event
fires. Nothing about the product changed. The arm just lost its worst users.

```
    assigned to treatment    2,000,000
    observed in treatment    1,962,532  (37,468 silently missing)
    observed split           50.473% / 49.527%
    TRUE lift                +0.00%
    MEASURED lift            +2.00%  (p = 8.94e-08, SIGNIFICANT)
    SRM gate                 p = 4.95e-79  -> BLOCKED, do not analyze
```

The lift is entirely fabricated, and it is *statistically significant*. Without
the gate it ships. This is why SRM runs before analysis, at α=0.001 — it fires
on every experiment, so it needs a low false-alarm rate to stay trusted.

The sample size here is deliberate. The dropout bias is ~2% at any n, but at
200k users that sits inside the noise (p≈0.3) and looks harmless. The danger of
SRM is not that it produces a *large* bias — it is that at real traffic volumes
it produces a **significant** one. Scale is what turns the bug into a shipped
feature.

### CUPED pays off on the metrics you'd least expect

CUPED replaces the outcome `Y` with `Y - θ(X - E[X])` for a pre-experiment
covariate `X`. Because `X` is measured before assignment, subtracting it cannot
bias the effect — it only removes variance the treatment could never have
caused.

The scenario tests the actual theoretical claim, that reduction equals `ρ²` for
the **achieved** correlation:

```
  CONTINUOUS metric (sessions per week)
    nominal rho   achieved rho   predicted (rho^2)   observed   traffic needed
           0.9          0.898              80.6%      81.0%             19%

  BINARY metric (converted yes/no)
           0.9          0.457              20.9%      20.9%             79%
```

Predicted and observed agree to **0.4pp** across every row — the estimator is
correct. But the same population, the same covariate, and the same code deliver
81% variance reduction on sessions and only 21% on conversion. A Bernoulli
outcome at an 8.5% rate is mostly irreducible coin-flip noise, so even a perfect
predictor of the underlying propensity correlates weakly with the realized 0/1.

That distinction is the practically useful one. It is the difference between
"we tried CUPED and it did nothing" and knowing in advance which metrics it
will pay for.

---

## Layout

```
abtest/
  stats.py        normal/t/chi-square distributions, two-sample tests, SRM check
  power.py        sample size, power, MDE, CUPED-adjusted planning
  sequential.py   mSPRT always-valid p-values + confidence sequences
  population.py   synthetic users: segments, covariates, novelty decay
  experiment.py   assignment, measurement, and three estimators
  report.py       self-contained HTML + hand-built SVG charts
  scenarios/      the seven studies, each with a declared pass criterion
tests/            89 unit tests
run.py            CLI
```

### Why no NumPy or SciPy

Two reasons, one practical and one substantive.

Practical: the repo runs anywhere Python does. Clone and run, no virtualenv, no
version pinning, nothing to break in six months.

Substantive: implementing `norm_ppf`, the incomplete beta function, Welch's
degrees of freedom, and the mSPRT martingale means understanding them rather
than calling them. Every function is checked against published reference values
in `tests/`, not against its own output — a test that asserts a function returns
what it currently returns locks in bugs forever.

The one genuine performance decision is in `rng.py`: calibration studies draw
each arm's conversion *count* from a single binomial rather than simulating
users individually. Identical mathematics when users are exchangeable, ~10,000×
faster, and it is what makes the 20,000-replication A/A study finish in 0.25
seconds instead of an hour.

---

## CLI

```bash
python run.py list                        # the seven scenarios
python run.py run peeking                 # one scenario
python run.py run all --report out.html   # everything, plus the HTML report
python run.py run all --json results.json # raw results for further analysis
```

The planner is usable on its own for real experiment design:

```bash
python run.py plan --baseline 0.085 --mde 0.03 --traffic 40000
```

```
  users per arm         190,308
  runtime at 40,000/day  9.5 days

  if you can only afford a shorter run:
     3 days  ->     60,000 users/arm  ->  smallest visible lift +5.37%
     7 days  ->    140,000 users/arm  ->  smallest visible lift +3.50%
    14 days  ->    280,000 users/arm  ->  smallest visible lift +2.47%
```

That last table is the one that matters in planning meetings. Anything smaller
than the MDE is invisible to the test — running it anyway and then interpreting
the noise is how the peeking problem starts.

---

## Tests

```bash
python -m unittest discover -s tests -t .
```

89 tests, ~5 seconds. Three things they cover that matter:

- **Reference values, not self-agreement.** Distribution functions are checked
  against standard normal and t-tables; the sample-size formula against a
  hand-computable worked example.
- **The always-valid guarantee itself.** `test_sequential.py` stops at the first
  crossing across hundreds of A/A runs and asserts the error rate stays under α,
  with the naive comparison in the same loop for contrast.
- **The generator's contract.** The CUPED `ρ²` prediction is only meaningful if
  `corr(sessions_pre, sessions_post)` really equals the configured ρ, so that is
  asserted directly.

---

## Known limits

- The mSPRT is **conservative**: it holds ~1% false positives against a nominal
  5%. Ville's inequality bounds a supremum over all time, and that bound is not
  tight. Real power is left on the table; the 1.74× figure above already
  reflects it.
- `blended_true_effect` is the population-average lift under heterogeneous
  segment response, and it is *not* the per-segment parameter. Conflating them
  would compare every scenario against the wrong ground truth, so the two are
  separate properties with a test asserting they differ.
- The novelty scenario's daily estimates are noisy by construction. The pass
  criterion is a **coverage rate over 400 replications**, not a single-draw CI
  check — testing one 95% interval would fail at random one run in twenty.
- `Rng.binomial` uses CPython 3.12+'s exact `random.binomialvariate`. The
  fallback for older versions is exact below n=200 and a continuity-corrected
  normal approximation above it.
