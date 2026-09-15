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

_CSS = """
:root{--bg:#fbfaf8;--fg:#1c1b19;--mut:#6b6762;--line:#e3dfd9;--card:#fff;
--pos:#2f7d5c;--neg:#b34a3c;--accent:#2E6F9E;--wm:#C4573B}
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
