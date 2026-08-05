"""Self-contained HTML report generation.

Charts are hand-built SVG rather than a plotting library, for the same reason
the statistics are hand-built: the project has no dependencies, so it runs on
any machine with Python and nothing else. The output is one file that opens
in a browser with no server, no CDN, and no network access.
"""

from __future__ import annotations

import datetime
import html

# --------------------------------------------------------------------------
# SVG primitives
# --------------------------------------------------------------------------

W, H = 720, 300
PAD_L, PAD_R, PAD_T, PAD_B = 62, 20, 24, 46


def _x(i: int, n: int) -> float:
    if n <= 1:
        return PAD_L
    return PAD_L + i * (W - PAD_L - PAD_R) / (n - 1)


def _y(v: float, lo: float, hi: float) -> float:
    if hi == lo:
        return H - PAD_B
    return H - PAD_B - (v - lo) / (hi - lo) * (H - PAD_T - PAD_B)


def _axes(y_lo: float, y_hi: float, y_fmt, x_labels: list[str], every: int = 1) -> str:
    parts = [
        f'<line class="axis" x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{H - PAD_B}"/>',
        f'<line class="axis" x1="{PAD_L}" y1="{H - PAD_B}" '
        f'x2="{W - PAD_R}" y2="{H - PAD_B}"/>',
    ]
    for k in range(5):
        v = y_lo + (y_hi - y_lo) * k / 4
        y = _y(v, y_lo, y_hi)
        parts.append(
            f'<line class="grid" x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{PAD_L - 8}" y="{y + 4:.1f}" '
            f'text-anchor="end">{y_fmt(v)}</text>'
        )
    n = len(x_labels)
    for i, lab in enumerate(x_labels):
        if i % every:
            continue
        parts.append(
            f'<text class="tick" x="{_x(i, n):.1f}" y="{H - PAD_B + 18}" '
            f'text-anchor="middle">{html.escape(lab)}</text>'
        )
    return "".join(parts)


def _series(values: list[float], lo: float, hi: float, cls: str) -> str:
    pts = " ".join(
        f"{_x(i, len(values)):.1f},{_y(v, lo, hi):.1f}" for i, v in enumerate(values)
    )
    dots = "".join(
        f'<circle class="{cls} dot" cx="{_x(i, len(values)):.1f}" '
        f'cy="{_y(v, lo, hi):.1f}" r="3"/>'
        for i, v in enumerate(values)
    )
    return f'<polyline class="{cls} line" points="{pts}"/>{dots}'


def _legend(entries: list[tuple[str, str]]) -> str:
    out = []
    for i, (cls, label) in enumerate(entries):
        x = PAD_L + i * 220
        out.append(
            f'<rect class="{cls} swatch" x="{x}" y="6" width="14" height="4" rx="2"/>'
            f'<text class="legend" x="{x + 20}" y="11">{html.escape(label)}</text>'
        )
    return "".join(out)


def line_chart(
    x_labels: list[str],
    series: list[tuple[str, str, list[float]]],
    y_fmt=lambda v: f"{v:.0%}",
    y_lo: float | None = None,
    y_hi: float | None = None,
    every: int = 1,
    rules: list[tuple[float, str]] | None = None,
) -> str:
    all_vals = [v for _, _, vals in series for v in vals]
    if rules:
        all_vals += [r[0] for r in rules]
    lo = min(all_vals) if y_lo is None else y_lo
    hi = max(all_vals) if y_hi is None else y_hi
    if hi == lo:
        hi = lo + 1e-9
    # Only pad bounds that were inferred. An explicit y_lo=0 means the axis
    # should start at zero, not at -6% because of headroom padding.
    span = hi - lo
    if y_lo is None:
        lo -= span * 0.08
    if y_hi is None:
        hi += span * 0.08

    body = [_axes(lo, hi, y_fmt, x_labels, every)]
    for rule_v, rule_label in rules or []:
        y = _y(rule_v, lo, hi)
        body.append(
            f'<line class="rule" x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}"/>'
            f'<text class="rulelabel" x="{W - PAD_R - 4}" y="{y - 6:.1f}" '
            f'text-anchor="end">{html.escape(rule_label)}</text>'
        )
    for cls, _label, vals in series:
        body.append(_series(vals, lo, hi, cls))
    body.append(_legend([(cls, label) for cls, label, _ in series]))
    return (
        f'<svg viewBox="0 0 {W} {H}" role="img" preserveAspectRatio="xMidYMid meet">'
        + "".join(body)
        + "</svg>"
    )


def bar_chart(
    labels: list[str],
    values: list[float],
    y_fmt=lambda v: f"{v:.0%}",
    classes: list[str] | None = None,
) -> str:
    hi = max(values) if values else 1.0
    hi = hi * 1.15 if hi > 0 else 1.0
    n = len(values)
    inner = W - PAD_L - PAD_R
    bw = inner / n * 0.62

    body = [_axes(0.0, hi, y_fmt, labels)]
    for i, v in enumerate(values):
        cx = PAD_L + inner * (i + 0.5) / n
        y = _y(v, 0.0, hi)
        cls = (classes[i] if classes else "s1") + " bar"
        body.append(
            f'<rect class="{cls}" x="{cx - bw / 2:.1f}" y="{y:.1f}" '
            f'width="{bw:.1f}" height="{H - PAD_B - y:.1f}" rx="2"/>'
        )
        body.append(
            f'<text class="barval" x="{cx:.1f}" y="{y - 6:.1f}" '
            f'text-anchor="middle">{y_fmt(v)}</text>'
        )
    body.append(
        f'<line class="axis" x1="{PAD_L}" y1="{H - PAD_B}" x2="{W - PAD_R}" '
        f'y2="{H - PAD_B}"/>'
    )
    return (
        f'<svg viewBox="0 0 {W} {H}" role="img" preserveAspectRatio="xMidYMid meet">'
        + "".join(body)
        + "</svg>"
    )


# --------------------------------------------------------------------------
# Per-scenario figures
# --------------------------------------------------------------------------


def _fig_aa(r: dict) -> str:
    hist = r["p_value_histogram"]
    total = sum(hist)
    labels = [f"{i / len(hist):.2f}" if i % 4 == 0 else "" for i in range(len(hist))]
    return bar_chart(
        labels,
        [c / total for c in hist],
        y_fmt=lambda v: f"{v:.1%}",
        classes=["s3" if i == 0 else "s1" for i in range(len(hist))],
    )


def _fig_power(r: dict) -> str:
    labels = [f"{p['relative_effect']:+.0%}" for p in r["points"]]
    return line_chart(
        labels,
        [
            ("s1", "analytic power", [p["analytic_power"] for p in r["points"]]),
            ("s2", "simulated power", [p["empirical_power"] for p in r["points"]]),
        ],
        y_lo=0.0,
        y_hi=1.0,
        rules=[(r["target_power"], "80% target")],
    )


def _fig_peeking(r: dict) -> str:
    n = len(r["fpr_by_look"])
    return line_chart(
        [str(i + 1) for i in range(n)],
        [
            ("s3", "fixed-horizon, peeked daily", r["fpr_by_look"]),
            (
                "s2",
                "mSPRT always-valid",
                [r["sequential_false_positive_rate"]] * n,
            ),
        ],
        y_lo=0.0,
        rules=[(r["alpha"], "nominal 5%")],
    )


def _fig_srm(r: dict) -> str:
    labels = [
        "50.00%" if d["bias"] == 0 else f"{d['true_split']:.2%}" for d in r["detection"]
    ]
    return bar_chart(
        labels,
        [d["detection_rate"] for d in r["detection"]],
        classes=["s3" if d["bias"] == 0 else "s2" for d in r["detection"]],
    )


def _fig_heterogeneity(r: dict) -> str:
    labels = list(r["per_segment"].keys()) + ["POOLED", "STRATIFIED"]
    values = [v["relative_lift"] for v in r["per_segment"].values()] + [
        r["pooled_relative_lift"],
        r["stratified_relative_lift"],
    ]
    classes = ["s2"] * len(r["per_segment"]) + ["s3", "s1"]
    hi = max(values) * 1.2
    lo = min(values) * 1.2
    n = len(values)
    inner = W - PAD_L - PAD_R
    bw = inner / n * 0.55
    zero_y = _y(0.0, lo, hi)

    body = [_axes(lo, hi, lambda v: f"{v:+.0%}", labels)]
    body.append(
        f'<line class="axis" x1="{PAD_L}" y1="{zero_y:.1f}" x2="{W - PAD_R}" '
        f'y2="{zero_y:.1f}"/>'
    )
    for i, v in enumerate(values):
        cx = PAD_L + inner * (i + 0.5) / n
        y = _y(v, lo, hi)
        top, height = (y, zero_y - y) if v >= 0 else (zero_y, y - zero_y)
        body.append(
            f'<rect class="{classes[i]} bar" x="{cx - bw / 2:.1f}" y="{top:.1f}" '
            f'width="{bw:.1f}" height="{abs(height):.1f}" rx="2"/>'
        )
        ty = top - 6 if v >= 0 else top + height + 16
        body.append(
            f'<text class="barval" x="{cx:.1f}" y="{ty:.1f}" '
            f'text-anchor="middle">{v:+.1%}</text>'
        )
    return (
        f'<svg viewBox="0 0 {W} {H}" role="img" preserveAspectRatio="xMidYMid meet">'
        + "".join(body)
        + "</svg>"
    )


def _fig_novelty(r: dict) -> str:
    daily = r["daily"]
    return line_chart(
        [str(d["day"]) for d in daily],
        [
            ("s1", "true effect", [d["true_effect_today"] for d in daily]),
            (
                "s3",
                "cumulative (dashboard)",
                [d["cumulative_measured_lift"] for d in daily],
            ),
        ],
        y_fmt=lambda v: f"{v:+.0%}",
        every=3,
        rules=[(0.0, "")],
    )


def _fig_cuped(r: dict) -> str:
    cont, binr = r["continuous_rows"], r["binary_rows"]
    return line_chart(
        [f"{row['nominal_rho']:.1f}" for row in cont],
        [
            ("s1", "continuous metric", [row["observed_reduction"] for row in cont]),
            ("s2", "binary metric", [row["observed_reduction"] for row in binr]),
            ("s4", "theoretical rho^2", [row["nominal_rho"] ** 2 for row in cont]),
        ],
        y_lo=0.0,
    )


FIGURES = {
    "aa": _fig_aa,
    "power": _fig_power,
    "peeking": _fig_peeking,
    "srm": _fig_srm,
    "heterogeneity": _fig_heterogeneity,
    "novelty": _fig_novelty,
    "cuped": _fig_cuped,
}

CAPTIONS = {
    "aa": "p-value distribution across A/A tests with zero true effect. Flat is "
    "correct; the highlighted first bar is the 5% that cross significance.",
    "power": "Simulated rejection rate against the analytic power curve at the "
    "designed sample size.",
    "peeking": "False positive rate as daily looks accumulate. Both arms are "
    "identical -- every rejection is an error.",
    "srm": "Chi-square detection rate against the true assignment split.",
    "heterogeneity": "Measured lift by segment, pooled, and post-stratified. "
    "Same data, three answers.",
    "novelty": "True effect versus what the cumulative dashboard number reports "
    "as the novelty wears off.",
    "cuped": "Variance removed by CUPED against the theoretical rho^2 ceiling, "
    "for a continuous and a binary metric.",
}


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

CSS = """
:root{
  --bg:#fbfbfa; --panel:#fff; --ink:#1f1e1c; --muted:#6b6862; --line:#e4e1db;
  --s1:#4f7cff; --s2:#12a594; --s3:#e5484d; --s4:#8e8b85; --pass:#12a594; --fail:#e5484d;
}
@media (prefers-color-scheme:dark){
  :root{ --bg:#141413; --panel:#1c1c1a; --ink:#eeece7; --muted:#9a968e; --line:#2e2d2a;
         --s1:#7aa0ff; --s2:#2ec4b0; --s3:#ff6b6f; --s4:#8e8b85; }
}
:root[data-theme=dark]{
  --bg:#141413; --panel:#1c1c1a; --ink:#eeece7; --muted:#9a968e; --line:#2e2d2a;
  --s1:#7aa0ff; --s2:#2ec4b0; --s3:#ff6b6f; --s4:#8e8b85;
}
:root[data-theme=light]{
  --bg:#fbfbfa; --panel:#fff; --ink:#1f1e1c; --muted:#6b6862; --line:#e4e1db;
  --s1:#4f7cff; --s2:#12a594; --s3:#e5484d; --s4:#8e8b85;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;}
.wrap{max-width:860px;margin:0 auto;padding:48px 20px 80px}
header{border-bottom:1px solid var(--line);padding-bottom:24px;margin-bottom:8px}
h1{font-size:26px;margin:0 0 6px;letter-spacing:-.02em}
.sub{color:var(--muted);font-size:14px;margin:0}
.summary{display:flex;gap:10px;flex-wrap:wrap;margin:24px 0 8px}
.chip{background:var(--panel);border:1px solid var(--line);border-radius:8px;
  padding:8px 12px;font-size:13px}
.chip b{font-variant-numeric:tabular-nums}
section{background:var(--panel);border:1px solid var(--line);border-radius:12px;
  padding:22px 24px;margin:22px 0}
h2{font-size:17px;margin:0 0 2px;letter-spacing:-.01em}
.slug{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em;
  margin:0 0 14px}
.verdict{float:right;font-size:12px;font-weight:600;padding:3px 10px;border-radius:20px}
.PASS{background:color-mix(in srgb,var(--pass) 16%,transparent);color:var(--pass)}
.FAIL{background:color-mix(in srgb,var(--fail) 16%,transparent);color:var(--fail)}
.take{border-left:3px solid var(--s1);padding:2px 0 2px 14px;margin:18px 0 0;
  color:var(--ink);font-size:14.5px}
figure{margin:16px 0 0}
svg{width:100%;height:auto;display:block}
figcaption{color:var(--muted);font-size:12.5px;margin-top:8px}
.axis{stroke:var(--muted);stroke-width:1;opacity:.55}
.grid{stroke:var(--line);stroke-width:1}
.rule{stroke:var(--s4);stroke-width:1;stroke-dasharray:4 4;opacity:.8}
.rulelabel{fill:var(--muted);font-size:10px}
.tick{fill:var(--muted);font-size:10.5px;font-variant-numeric:tabular-nums}
.legend{fill:var(--muted);font-size:11px}
.barval{fill:var(--muted);font-size:10px;font-variant-numeric:tabular-nums}
.line{fill:none;stroke-width:2.25;stroke-linejoin:round;stroke-linecap:round}
.dot{stroke:none}
.s1.line{stroke:var(--s1)} .s1.dot,.s1.bar,.s1.swatch{fill:var(--s1)}
.s2.line{stroke:var(--s2)} .s2.dot,.s2.bar,.s2.swatch{fill:var(--s2)}
.s3.line{stroke:var(--s3)} .s3.dot,.s3.bar,.s3.swatch{fill:var(--s3)}
/* Reference series: thin and widely dashed so the measured series it sits on
   top of stays visible through the gaps. When observed matches theory the two
   overlap exactly, and both still need to read. */
.s4.line{stroke:var(--s4);stroke-width:1.4;stroke-dasharray:3 6}
.s4.dot{fill:none} .s4.bar,.s4.swatch{fill:var(--s4)}
pre{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:14px;
  overflow-x:auto;font:12px/1.55 ui-monospace,"Cascadia Code",Consolas,monospace;
  margin:16px 0 0}
footer{color:var(--muted);font-size:12.5px;margin-top:40px;text-align:center}
"""


def write_report(results: list[dict], path: str) -> None:
    passed = sum(1 for r in results if r["verdict"] == "PASS")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    total_time = sum(r.get("elapsed_seconds", 0) for r in results)

    parts = [
        f"<style>{CSS}</style>",
        '<div class="wrap">',
        "<header>",
        "<h1>synthetic-ab-lab</h1>",
        '<p class="sub">Simulation harness for A/B test methodology. Every scenario '
        "plants a known ground truth, runs an analysis against it, and states a "
        "pass criterion up front.</p>",
        "</header>",
        '<div class="summary">',
        f'<div class="chip"><b>{passed}/{len(results)}</b> scenarios passed</div>',
        f'<div class="chip"><b>{total_time:.1f}s</b> total runtime</div>',
        '<div class="chip"><b>0</b> dependencies</div>',
        f'<div class="chip">generated {stamp}</div>',
        "</div>",
    ]

    for r in results:
        key = r["scenario"]
        fig = FIGURES.get(key)
        parts.append("<section>")
        parts.append(f'<span class="verdict {r["verdict"]}">{r["verdict"]}</span>')
        parts.append(f"<h2>{html.escape(r['title'])}</h2>")
        parts.append(f'<p class="slug">scenario: {html.escape(key)}</p>')
        if fig:
            parts.append("<figure>")
            parts.append(fig(r))
            parts.append(f"<figcaption>{html.escape(CAPTIONS.get(key, ''))}</figcaption>")
            parts.append("</figure>")
        parts.append(f'<p class="take">{html.escape(r["takeaway"])}</p>')
        if r.get("console"):
            parts.append(f"<pre>{html.escape(r['console'])}</pre>")
        parts.append("</section>")

    parts.append(
        '<footer>Python standard library only &middot; every figure regenerates from '
        "<code>python run.py run all --report report.html</code></footer>"
    )
    parts.append("</div>")

    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>synthetic-ab-lab report</title></head><body>"
        + "".join(parts)
        + "</body></html>"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
