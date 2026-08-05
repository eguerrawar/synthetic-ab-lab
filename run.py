#!/usr/bin/env python3
"""synthetic-ab-lab command line.

    python run.py list                       show the scenarios
    python run.py run aa                     run one scenario
    python run.py run all                    run everything
    python run.py run all --report out.html  ...and write the HTML report
    python run.py plan --baseline 0.085 --mde 0.03 --traffic 40000

Standard library only. No install step, no virtualenv, no network.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from abtest.power import mde_at_n, sample_size_proportions
from abtest.scenarios import REGISTRY

BANNER = r"""
  synthetic-ab-lab
  simulation harness for A/B test methodology -- known ground truth,
  falsifiable claims, stdlib only
"""


def cmd_list(_args: argparse.Namespace) -> int:
    print(BANNER)
    print("  scenarios:\n")
    for key, mod in REGISTRY.items():
        print(f"    {key:<15} {mod.TITLE}")
    print("\n  run one with:  python run.py run <name>")
    print("  run them all:  python run.py run all --report report.html\n")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    names = list(REGISTRY) if args.scenario == "all" else [args.scenario]
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        print(f"unknown scenario(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"available: {', '.join(REGISTRY)}", file=sys.stderr)
        return 2

    print(BANNER)
    results = []
    failures = 0

    for name in names:
        mod = REGISTRY[name]
        print("=" * 78)
        print(f"  [{name}]  {mod.TITLE}")
        print("=" * 78)
        started = time.time()
        result = mod.run(seed=args.seed)
        elapsed = time.time() - started

        console = mod.render(result)
        print(console)
        print()
        # Keep the console rendering so the HTML report can show the exact
        # numbers behind each chart rather than only the picture.
        result["console"] = console
        print(f"  VERDICT: {result['verdict']}   ({elapsed:.1f}s)")
        print(f"  {_wrap(result['takeaway'], 74, '  ')}")
        print()

        result["elapsed_seconds"] = elapsed
        results.append(result)
        if result["verdict"] != "PASS":
            failures += 1

    print("=" * 78)
    passed = len(results) - failures
    print(f"  {passed}/{len(results)} scenarios passed their stated criteria")
    print("=" * 78)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"  wrote {args.json}")

    if args.report:
        from abtest.report import write_report

        write_report(results, args.report)
        print(f"  wrote {args.report}")

    return 1 if failures else 0


def cmd_plan(args: argparse.Namespace) -> int:
    plan = sample_size_proportions(
        baseline_rate=args.baseline,
        relative_mde=args.mde,
        alpha=args.alpha,
        power=args.power,
        daily_traffic=args.traffic,
    )
    print(BANNER)
    print("  EXPERIMENT PLAN\n")
    print(plan.summary())
    print()
    if args.traffic:
        print("  if you can only afford a shorter run:\n")
        for days in (3, 7, 14, 21, 28):
            n = int(args.traffic * days / 2)
            print(
                f"    {days:>2} days  ->  {n:>9,} users/arm  ->  "
                f"smallest visible lift {mde_at_n(args.baseline, n, args.alpha, args.power):+.2%}"
            )
        print()
    print("  Anything smaller than the MDE is invisible to this test. Do not")
    print("  run it and then interpret the noise.\n")
    return 0


def _wrap(text: str, width: int, indent: str) -> str:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return f"\n{indent}".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run.py", description="synthetic A/B testing lab"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list available scenarios")
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="run a scenario (or 'all')")
    p_run.add_argument("scenario", help="scenario name, or 'all'")
    p_run.add_argument("--seed", type=int, default=20260803)
    p_run.add_argument("--report", help="write an HTML report to this path")
    p_run.add_argument("--json", help="write raw results as JSON to this path")
    p_run.set_defaults(func=cmd_run)

    p_plan = sub.add_parser("plan", help="power / sample size calculator")
    p_plan.add_argument("--baseline", type=float, required=True, help="baseline rate, e.g. 0.085")
    p_plan.add_argument("--mde", type=float, required=True, help="relative MDE, e.g. 0.03 for +3%%")
    p_plan.add_argument("--alpha", type=float, default=0.05)
    p_plan.add_argument("--power", type=float, default=0.80)
    p_plan.add_argument("--traffic", type=int, default=None, help="users per day, both arms")
    p_plan.set_defaults(func=cmd_plan)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
