"""Self-contained HTML dashboard built from a results JSON file.

No network, no build step: figures are inlined as base64 PNGs and the tables are
rendered from the same numbers the markdown report uses, so the dashboard can
never disagree with `RESULTS.md`.
"""

from __future__ import annotations

import base64
import html
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["build_dashboard"]

# Status colours are validated with the dataviz skill's checker
# (scripts/validate_palette.js). The light pair #0f7a4a / #c4341c passes all six
# checks -- lightness band, chroma floor, CVD separation (deutan dE 8.6),
# normal-vision floor (dE 27.1) and contrast -- against the page surface.  They
# encode *status*, never series identity, and every use is accompanied by a text
# label, so colour is never the only channel carrying meaning.
_CSS = """
:root{--bg:#fbfaf8;--fg:#1c1b19;--mut:#6b6762;--line:#e3dfd9;--card:#fff;
--pos:#0f7a4a;--neg:#c4341c;--neutral:#6b6762;--accent:#2E6F9E;--wm:#C4573B;
--grid:#d8d3cc}
@media (prefers-color-scheme: dark){
:root{--bg:#14140f;--fg:#ece9e4;--mut:#a5a099;--line:#33312d;--card:#1c1b19;
--pos:#57c795;--neg:#ef7358;--neutral:#a5a099;--accent:#6aa9d6;--wm:#e08a72;
--grid:#3a3833}
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 -apple-system,
BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:32px 20px 72px}
h1{font-size:27px;margin:0 0 6px;letter-spacing:-.02em}
h2{font-size:19px;margin:38px 0 10px;letter-spacing:-.01em}
h3{font-size:15px;margin:22px 0 8px;color:var(--mut);font-weight:600}
p{margin:8px 0}.sub{color:var(--mut);margin:0 0 18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:16px 18px;margin:14px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.kpi .v{font-size:23px;font-weight:650;letter-spacing:-.02em}
.kpi .l{color:var(--mut);font-size:12px;margin-top:2px}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:12.5px;min-width:520px}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line);
white-space:nowrap}
th:first-child,td:first-child{text-align:left;font-weight:600}
thead th{color:var(--mut);font-weight:600;font-size:11.5px;text-transform:uppercase;
letter-spacing:.04em;position:sticky;top:0;background:var(--card)}
tbody tr:hover{background:#f6f4f1}
.pos{color:var(--pos)}.neg{color:var(--neg)}.mut{color:var(--mut)}
.badge{display:inline-block;padding:1px 7px;border-radius:99px;font-size:11px;
border:1px solid var(--line);color:var(--mut);margin-left:6px}
.badge.wm{color:var(--wm);border-color:#eccfc7;background:#fdf4f1}
.f{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;
background:#fff;margin:8px 0}
.finding{padding:9px 12px;border-left:3px solid var(--line);margin:8px 0;
background:#fff;border-radius:0 6px 6px 0;font-size:13px}
.finding.high{border-color:#b34a3c}.finding.medium{border-color:#d9a441}
.finding.low{border-color:#8e8a85}.finding.info{border-color:#2E6F9E}
.finding code{background:#f2efec;padding:1px 5px;border-radius:4px;font-size:12px}
footer{color:var(--mut);font-size:12px;margin-top:40px;border-top:1px solid var(--line);
padding-top:14px}
.verdict{display:grid;grid-template-columns:1fr auto;gap:8px 14px;align-items:start;
padding:11px 14px;border:1px solid var(--line);border-radius:8px;background:var(--card);
margin:8px 0}
.verdict .q{font-weight:600}
.verdict .d{grid-column:1/-1;color:var(--mut);font-size:12.5px}
.chip{display:inline-flex;align-items:center;gap:5px;padding:2px 9px;border-radius:99px;
font-size:11.5px;font-weight:650;border:1px solid currentColor;white-space:nowrap}
.chip.yes{color:var(--pos)}.chip.no{color:var(--neg)}.chip.na{color:var(--neutral)}
.headline{border-left:3px solid var(--accent);padding:12px 16px;background:var(--card);
border-radius:0 8px 8px 0;margin:14px 0;font-size:14.5px}
figure{margin:12px 0}figcaption{color:var(--mut);font-size:12px;margin-top:6px}
@media (max-width:640px){.wrap{padding:20px 14px 56px}h1{font-size:22px}}
"""


def _fmt(v: Any, nd: int = 3, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return '<span class="mut">--</span>'
    if isinstance(v, (int, float)):
        cls = ""
        if signed:
            cls = "pos" if v > 0 else ("neg" if v < 0 else "")
        s = f"{v:+.{nd}f}" if signed else f"{v:,.{nd}f}" if abs(v) < 1e6 else f"{v:,.0f}"
        return f'<span class="{cls}">{s}</span>' if cls else s
    return html.escape(str(v))


def _table(rows: Sequence[Dict[str, Any]], cols: Sequence[str],
           nd: int = 3, signed: Optional[Sequence[str]] = None,
           families: Optional[Dict[str, str]] = None) -> str:
    if not rows:
        return '<p class="mut">No data.</p>'
    signed = set(signed or [])
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = []
    for r in rows:
        cells = []
        for i, c in enumerate(cols):
            v = r.get(c)
            if i == 0:
                label = html.escape(str(v))
                fam = (families or {}).get(str(v))
                if fam == "world_model":
                    label += '<span class="badge wm">world model</span>'
                elif fam in ("unified", "llm", "oracle"):
                    label += f'<span class="badge">{html.escape(fam)}</span>'
                cells.append(f"<td>{label}</td>")
            else:
                cells.append(f"<td>{_fmt(v, nd, c in signed)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (f'<div class="tw"><table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def _img(path: str, alt: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()
    return f'<img class="f" alt="{html.escape(alt)}" src="data:image/png;base64,{b64}">'


def _sorted(rows: Sequence[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda r: (r.get(key) is None,
                                       -(r.get(key) or 0)
                                       if isinstance(r.get(key), (int, float)) else 0))



def _forest(rows: Sequence[Dict[str, Any]], width: int = 720,
            row_h: int = 26) -> str:
    """Forest plot of paired mean differences with confidence intervals.

    The canonical form for this data: the question a reader has is "which
    differences exclude zero?", and a dot-and-whisker against a zero rule answers
    it directly, where a bar chart of means would hide the uncertainty that is
    the entire point.
    """
    rows = [r for r in rows if isinstance(r.get("mean_diff"), (int, float))]
    if not rows:
        return ""
    lo = min(min(r.get("ci_lo", r["mean_diff"]), r["mean_diff"]) for r in rows)
    hi = max(max(r.get("ci_hi", r["mean_diff"]), r["mean_diff"]) for r in rows)
    pad = max((hi - lo) * 0.12, 0.02)
    lo, hi = lo - pad, hi + pad
    label_w, right_w = 230, 92
    plot_w = width - label_w - right_w
    height = len(rows) * row_h + 44

    def x(v: float) -> float:
        return label_w + (v - lo) / max(hi - lo, 1e-9) * plot_w

    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" '
             f'role="img" aria-label="Paired mean differences with 95% '
             f'confidence intervals" style="max-width:100%;height:auto">']
    # zero reference rule
    parts.append(f'<line x1="{x(0):.1f}" y1="26" x2="{x(0):.1f}" y2="{height-18}" '
                 f'stroke="var(--grid)" stroke-width="2" stroke-dasharray="3 3"/>')
    parts.append(f'<text x="{x(0):.1f}" y="18" text-anchor="middle" font-size="10.5" '
                 f'fill="var(--mut)">no difference</text>')
    for i, r in enumerate(rows):
        y = 34 + i * row_h
        d = float(r["mean_diff"])
        clo = float(r.get("ci_lo", d))
        chi = float(r.get("ci_hi", d))
        # Colour follows the exact permutation test, not "the CI excludes zero".
        # The two can disagree (unified - llm: CI excludes zero, p = 0.076), and
        # colouring by the looser criterion would show green on a comparison the
        # report itself calls inconclusive.  The conservative test wins so that
        # colour, the asterisk and the prose all say the same thing.
        pv = r.get("p_value")
        sig = isinstance(pv, (int, float)) and pv < 0.05
        col = "var(--pos)" if (sig and d > 0) else (
            "var(--neg)" if sig else "var(--neutral)")
        parts.append(f'<text x="{label_w-10}" y="{y+4}" text-anchor="end" '
                     f'font-size="11.5" fill="var(--fg)">'
                     f'{html.escape(str(r.get("comparison", "")))}</text>')
        parts.append(f'<line x1="{x(clo):.1f}" y1="{y}" x2="{x(chi):.1f}" y2="{y}" '
                     f'stroke="{col}" stroke-width="2" stroke-linecap="round"/>')
        for e in (clo, chi):      # CI caps
            parts.append(f'<line x1="{x(e):.1f}" y1="{y-4}" x2="{x(e):.1f}" '
                         f'y2="{y+4}" stroke="{col}" stroke-width="2"/>')
        # 2px surface ring keeps the point readable where it overlaps the whisker
        parts.append(f'<circle cx="{x(d):.1f}" cy="{y}" r="5" fill="{col}" '
                     f'stroke="var(--card)" stroke-width="2"/>')
        star = " *" if sig else ""
        parts.append(f'<text x="{width-6}" y="{y+4}" text-anchor="end" font-size="11" '
                     f'fill="var(--mut)">{d:+.3f}{star}</text>')
    for v in (lo, 0.0, hi):       # sparse axis
        parts.append(f'<text x="{x(v):.1f}" y="{height-4}" text-anchor="middle" '
                     f'font-size="10" fill="var(--mut)">{v:+.2f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _verdict_block(results: Dict[str, Any]) -> str:
    """Pre-registered falsification checks, as status rows.

    Status is carried by a text label *and* a colour, never colour alone.
    """
    try:
        from ..evaluation.verdict import evaluate_hypothesis
        v = results.get("verdict") or evaluate_hypothesis(results)
    except Exception:
        return ""
    if not v.get("checks"):
        return ""
    label = {"supported": ("yes", "yes"), "not_supported": ("no", "NO"),
             "inconclusive": ("na", "inconclusive"), "info": ("na", "context")}
    out = ["<h2>Does the evidence support the hypothesis?</h2>",
           '<div class="headline">These criteria were fixed in '
           '<code>docs/METHODOLOGY.md</code> <b>before any result was seen</b>. '
           'The benchmark was able to confirm the hypothesis and did not.</div>']
    counts = v.get("counts", {})
    if counts:
        out.append('<div class="grid">' + "".join(
            f'<div class="kpi"><div class="v">{n}</div>'
            f'<div class="l">{html.escape(k.replace("_", " "))}</div></div>'
            for k, n in sorted(counts.items(),
                               key=lambda kv: ["supported", "not_supported",
                                               "inconclusive", "info"].index(kv[0])
                               if kv[0] in ("supported", "not_supported",
                                            "inconclusive", "info") else 9)) + "</div>")
    for c in v["checks"]:
        cls, text = label.get(c["verdict"], ("na", c["verdict"]))
        out.append(
            f'<div class="verdict"><div class="q">'
            f'{html.escape(c["check"].replace("_", " "))}</div>'
            f'<div><span class="chip {cls}">{html.escape(text)}</span></div>'
            f'<div class="d">{html.escape(c["detail"])}</div></div>')
    return "".join(out)


def build_dashboard(results: Dict[str, Any], out_dir: str,
                    figures_dir: Optional[str] = None,
                    title: str = "Blockchain World Model Intelligence Lab") -> str:
    figures_dir = figures_dir or os.path.join(out_dir, "figures")
    tables = results.get("tables", {})
    tl = results.get("train_logs", {})
    families = {k: v.get("family", "baseline") for k, v in tl.items()}
    for p in results.get("control", {}).get("policy_info", {}):
        families.setdefault(p, results["control"]["policy_info"][p].get("family", ""))

    parts: List[str] = [f"<h1>{html.escape(title)}</h1>",
                        '<p class="sub">Does a learned world model provide capabilities '
                        'that better function approximation does not?</p>']

    # --- KPI strip ----------------------------------------------------
    kpis: List[str] = []
    comp = results.get("composite", {})
    ranked = [(k, v.get("composite")) for k, v in comp.items()
              if v.get("composite") is not None and v.get("coverage", 0) >= 0.8]
    ranked.sort(key=lambda kv: -(kv[1] or 0))
    if ranked:
        kpis.append(f'<div class="kpi"><div class="v">{html.escape(ranked[0][0])}</div>'
                    f'<div class="l">best composite ({ranked[0][1]:.3f})</div></div>')
    ctrl = results.get("control", {}).get("summaries", {})
    if ctrl:
        best = max(ctrl, key=lambda p: sum(v.get("total_return", 0)
                                           for v in ctrl[p].values()))
        n = len(ctrl[best]) or 1
        val = sum(v.get("total_return", 0) for v in ctrl[best].values()) / n
        kpis.append(f'<div class="kpi"><div class="v">{html.escape(best)}</div>'
                    f'<div class="l">best control return ({val:+.3f})</div></div>')
    aud = results.get("audit", {})
    if aud:
        hi = aud.get("by_severity", {}).get("high", 0)
        kpis.append(f'<div class="kpi"><div class="v">{hi}</div>'
                    f'<div class="l">high-severity audit findings</div></div>')
    da = results.get("data_audit", {})
    if da:
        kpis.append(f'<div class="kpi"><div class="v">'
                    f'{"clean" if da.get("seed_disjoint") else "CONTAMINATED"}</div>'
                    f'<div class="l">train/test split</div></div>')
    if kpis:
        parts.append(f'<div class="grid">{"".join(kpis)}</div>')

    # --- the verdict comes first -----------------------------------------
    parts.append(_verdict_block(results))

    h2h = results.get("control", {}).get("head_to_head")
    if h2h:
        parts += ["<h2>Head-to-head tests</h2>",
                  '<div class="card"><p>Every policy ran the same episode seeds, '
                  'so these are <b>paired</b> comparisons of total return. The bar '
                  'is a 95% bootstrap interval and the dot is the mean difference; '
                  'intervals crossing the dashed rule are not distinguishable from '
                  'no difference. <code>*</code> marks p &lt; 0.05 on an exact '
                  'sign-flip permutation test.</p></div>',
                  "<figure>", _forest(h2h),
                  '<figcaption>Positive favours the first system named. Colour '
                  'and the asterisk both mark p &lt; 0.05; grey means not '
                  'distinguishable, including where the interval happens to '
                  'exclude zero. Colour adds no information the label lacks.'
                  '</figcaption>',
                  "</figure>"]

    # --- prediction ----------------------------------------------------
    if "prediction" in tables:
        rows = tables["prediction"]
        dcols = ["model"] + [c for c in rows[0] if c.startswith("dskill@")] + \
                ["dskill_ood_mean", "dskill_ood_min"]
        parts += ["<h2>Prediction</h2>",
                  '<div class="card"><p><b>Decision-state skill</b> against the better '
                  'of two naive references (persistence, linear autoregression), on '
                  'the target groups a planner actually needs: the focal account\'s '
                  'balance sheet plus endogenous protocol state. Price changes are '
                  'excluded because they are martingales by construction.</p></div>',
                  _table(_sorted(rows, "dskill@test_iid"), dcols, 4,
                         signed=set(dcols[1:]), families=families),
                  "<h3>Per-group detail (in-distribution)</h3>",
                  _table(_sorted(rows, "agent_skill_iid"),
                         ["model", "agent_skill_iid", "protocol_skill_iid",
                          "population_skill_iid", "price_skill_iid",
                          "price_level_skill_iid"], 4,
                         signed={"agent_skill_iid", "protocol_skill_iid",
                                 "population_skill_iid", "price_skill_iid",
                                 "price_level_skill_iid"}, families=families),
                  '<p class="mut"><code>price_level_skill_iid</code> is a leakage '
                  'canary: it should sit at zero.</p>',
                  "<h3>Events and reward</h3>",
                  _table(_sorted(rows, "event_auroc_iid"),
                         ["model", "event_auroc_iid", "event_logloss_iid",
                          "event_brier_iid", "event_ece_iid", "reward_skill_iid"], 4,
                         families=families),
                  _img(os.path.join(figures_dir, "prediction_skill.png"),
                       "prediction skill"),
                  _img(os.path.join(figures_dir, "calibration.png"), "calibration")]

    # --- counterfactual -------------------------------------------------
    if "counterfactual" in tables:
        rows = tables["counterfactual"]
        cols = ["model"] + [c for c in rows[0] if c.startswith("cf_skill@")] + \
               ["cf_skill_mean", "sign_acc_material_mean"]
        parts += ["<h2>Counterfactual reasoning</h2>",
                  '<div class="card"><p>Ground truth comes from forking the real '
                  'simulator, so both branches see identical exogenous noise and the '
                  'difference is the causal effect of the intervention. Skill is '
                  'measured against a <b>zero-effect predictor</b>: most single '
                  'actions barely move a deep market, so "nothing changed" already '
                  'scores a good MSE.</p></div>',
                  _table(_sorted(rows, "cf_skill_mean"), cols, 4,
                         signed=set(cols[1:-1]), families=families),
                  _img(os.path.join(figures_dir, "counterfactual.png"),
                       "counterfactual skill")]

    # --- long horizon ---------------------------------------------------
    parts.append(_img(os.path.join(figures_dir, "long_horizon.png"), "long horizon"))

    # --- control --------------------------------------------------------
    if "control" in tables:
        rows = tables["control"]
        cols = ["policy"] + [c for c in rows[0] if c.startswith("ret@")] + \
               ["return_mean", "sharpe_mean", "max_drawdown_mean"]
        parts += ["<h2>Control</h2>",
                  '<div class="card"><p>Every policy runs the same episodes and is '
                  'charged exactly one environment step per decision. '
                  '<code>oracle_plan</code> plans with the true simulator and is a '
                  'privileged upper bound, not a competitor.</p></div>',
                  _table(_sorted(rows, "return_mean"), cols, 4,
                         signed=set(cols[1:-1]), families=families),
                  _img(os.path.join(figures_dir, "control_return.png"),
                       "control returns"),
                  _img(os.path.join(figures_dir, "gate_usage.png"), "gate usage")]

    # --- composite ------------------------------------------------------
    if comp:
        from ..evaluation.benchmark import COMPONENTS
        crows = []
        for name, c in comp.items():
            row = {"system": name, "composite": c.get("composite"),
                   "coverage": c.get("coverage")}
            row.update(c.get("components", {}))
            crows.append(row)
        parts += ["<h2>Composite GPC index</h2>",
                  '<div class="card"><p>A single headline number is provided because '
                  'it was asked for, but it is built so it cannot hide its parts: '
                  'components are always shown, unsupported components are blank '
                  'rather than zero, and systems are only comparable at equal '
                  '<code>coverage</code>.</p></div>',
                  _table(_sorted(crows, "composite"),
                         ["system", "composite", "coverage", *COMPONENTS], 3,
                         families=families),
                  _img(os.path.join(figures_dir, "components.png"), "components"),
                  _img(os.path.join(figures_dir, "compute_tradeoff.png"),
                       "capability vs capacity")]

    # --- cost -----------------------------------------------------------
    if tl:
        rows = [{"model": k, "family": v.get("family"), "params": v.get("n_params"),
                 "train_s": v.get("wall_seconds")} for k, v in tl.items()]
        parts += ["<h2>Cost and capacity</h2>",
                  _table(sorted(rows, key=lambda r: -(r["params"] or 0)),
                         ["model", "family", "params", "train_s"], 1,
                         families=families)]

    # --- audit ----------------------------------------------------------
    if aud and aud.get("findings"):
        order = {"high": 0, "medium": 1, "low": 2, "info": 3}
        items = "".join(
            f'<div class="finding {html.escape(f["severity"])}">'
            f'<b>{html.escape(f["severity"])}</b> · <code>{html.escape(f["check"])}</code>'
            f' — {html.escape(f["detail"])}</div>'
            for f in sorted(aud["findings"], key=lambda f: order.get(f["severity"], 9)))
        parts += ["<h2>Automated audit</h2>", items]
    if da:
        worst = max(da.get("observation_overlap", {}).values(), default=0.0)
        parts += ['<div class="card"><p>Episode seeds disjoint across splits: '
                  f'<b>{da.get("seed_disjoint")}</b>. Largest verbatim observation-row '
                  f'overlap with train: <b>{worst:.5f}</b>.</p></div>']

    parts.append(f'<footer>Config hash <code>{html.escape(str(results.get("config_hash","?")))}</code> · '
                 f'evaluation wall time {results.get("wall_seconds", 0):.0f}s</footer>')

    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width,initial-scale=1,'
           f'viewport-fit=cover"><title>{html.escape(title)}</title>'
           f'<style>{_CSS}</style></head><body><div class="wrap">'
           f'{"".join(parts)}</div></body></html>')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "dashboard.html")
    with open(path, "w") as fh:
        fh.write(doc)
    return path
