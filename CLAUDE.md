# CLAUDE.md

Conventions for working in this repo.

## What this project is

A simulation harness that validates A/B testing *methodology*. Synthetic
populations with a known planted effect; analysis methods run against them;
each scenario reports whether the method recovered the truth.

The ground truth is the whole point. Any change that makes the "correct"
answer unknowable defeats the purpose of the repo.

## Hard constraints

- **Standard library only.** No NumPy, SciPy, pandas, matplotlib, or pytest.
  If a statistical function is needed, implement it in `abtest/stats.py` and
  test it against published reference values. This is not a style preference —
  it is what lets the repo be cloned and run anywhere with zero setup.
- **Python 3.12+.** Relied on for `random.binomialvariate`. A fallback exists
  in `rng.py`; keep it working.
- **Everything is seeded.** All randomness flows through `abtest/rng.py`.
  Never call `random` directly. Use `Rng.spawn("label")` for sub-streams so
  that adding a new metric does not shift assignment draws and silently
  change every previously reported number.

## Scenario contract

Every module in `abtest/scenarios/` exports:

```python
TITLE: str
def run(seed: int = 20260803, **kwargs) -> dict   # JSON-serializable
def render(result: dict) -> str                    # console output
```

The returned dict **must** contain `verdict` (`"PASS"`/`"FAIL"`) and
`takeaway` (one paragraph, plain English, quoting the actual numbers).

Register new scenarios in `scenarios/__init__.py::REGISTRY`.

## Pass criteria are the point

State the criterion before showing the numbers, and let it fail. Two rules
learned the hard way while building this:

1. **Never assert on a single random draw.** A 95% CI misses 5% of the time
   by construction, so a one-shot coverage check makes a scenario fail at
   random one run in twenty. Assert on a *rate* over replications instead —
   see `novelty.py::_coverage_of_final_week_estimator`.
2. **Make sure the scenario is powered to show its own effect.** The SRM
   scenario originally ran at 200k users, where the fabricated lift was real
   but not significant (p≈0.3), which undercut the entire claim. It runs at
   4M now. If a demo needs a specific sample size to land, say why in a
   comment.

## Testing

```bash
python -m unittest discover -s tests -t .
```

Test against **published reference values**, not against the code's current
output. A test asserting that a function returns what it returns today locks
in bugs forever. Normal/t/chi-square go against standard tables; the sample
size formula against a hand-computable worked example.

When a test fails, check the expected value before changing the code — four
of the first failures in this repo were wrong numbers in the tests, not bugs
in the implementation.

## Style

- Comments explain **why**, especially for statistical choices. "Pooled SE for
  the p-value, unpooled for the CI" needs the reason attached or someone will
  'fix' it.
- Cite papers for non-obvious methods (mSPRT → Johari et al.; CUPED → Deng et
  al.).
- Docstrings on scenarios describe the real-world failure being modeled, not
  just the code.
- Keep `render()` output under ~80 columns.
