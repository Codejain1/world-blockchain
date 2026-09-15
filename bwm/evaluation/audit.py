"""Post-hoc audits: contamination, leakage, degenerate models, reward hacking.

Run automatically at the end of every experiment.  The point is that the failure
modes most likely to produce a spurious positive result should be *checked*, not
assumed absent.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["audit_results", "format_audit"]

#: A model that scores above this on unpredictable price levels is suspect.
CANARY_THRESHOLD = 0.05


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def audit_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Return a list of findings, each with a severity and an explanation."""
    findings: List[Dict[str, str]] = []
    pred = results.get("prediction", {})
    cf = results.get("counterfactual", {})
    control = results.get("control", {})

    # --- 1. information leakage ---------------------------------------
    for model, per_split in pred.items():
        for split, v in per_split.items():
            g = v.get("groups", {}).get("price_level", {})
            s = g.get("skill_vs_best_naive")
            if _finite(s) and s > CANARY_THRESHOLD:
                findings.append({
                    "severity": "high", "check": "leakage_canary",
                    "detail": (f"{model} scores {s:+.3f} on price-level prediction "
                               f"in {split}; price changes are martingales by "
                               f"construction, so this suggests it can see the "
                               f"exogenous process.")})

    # --- 2. train/test contamination ----------------------------------
    audit = results.get("data_audit", {})
    if audit.get("seed_disjoint") is False:
        findings.append({"severity": "high", "check": "split_contamination",
                         "detail": f"Splits share episode seeds: {audit.get('collisions')}"})
    ov = audit.get("observation_overlap", {})
    for split, frac in ov.items():
        if _finite(frac) and frac > 0.01:
            findings.append({
                "severity": "medium", "check": "duplicate_states",
                "detail": (f"{frac:.1%} of {split} observation rows appear verbatim "
                           f"in train; memorisation is possible.")})

    # --- 3. degenerate / collapsed models ------------------------------
    for model, per_split in pred.items():
        iid = per_split.get("test_iid", {})
        ns = iid.get("next_state", {})
        if _finite(ns.get("skill_vs_persistence")) and \
                abs(ns["skill_vs_persistence"]) < 1e-6 and model != "persistence":
            findings.append({
                "severity": "low", "check": "degenerate_predictor",
                "detail": f"{model} is numerically identical to persistence in-distribution."})
        ev = iid.get("event", {})
        if _finite(ev.get("mean_auroc")) and ev["mean_auroc"] < 0.52 \
                and model not in ("persistence", "ar"):
            findings.append({
                "severity": "low", "check": "event_head_uninformative",
                "detail": (f"{model} event AUROC {ev['mean_auroc']:.3f} is at chance; "
                           f"its event head learned nothing.")})

    # --- 4. counterfactual signal vs the zero-effect control -----------
    for model, per_setting in cf.items():
        vals = [v.get("skill_vs_zero_effect") for v in per_setting.values()
                if isinstance(v, dict)]
        vals = [v for v in vals if _finite(v)]
        if vals and max(vals) <= 0.0:
            findings.append({
                "severity": "info", "check": "no_causal_signal",
                "detail": (f"{model} never beats the zero-effect predictor "
                           f"(best {max(vals):+.4f}); it has no measurable causal "
                           f"understanding in this world.")})

    # --- 5. reward hacking / implausible control outcomes ---------------
    for policy, per_sc in control.get("summaries", {}).items():
        for sc, s in per_sc.items():
            r = s.get("total_return")
            if _finite(r) and r > 1.0:
                findings.append({
                    "severity": "high", "check": "implausible_return",
                    "detail": (f"{policy} returned {r:+.1%} on {sc}; returns above "
                               f"100% per episode usually indicate a mark-to-market "
                               f"or mechanism artifact, not skill.")})
            dd = s.get("max_drawdown")
            if _finite(dd) and dd > 0.95:
                findings.append({
                    "severity": "medium", "check": "near_total_loss",
                    "detail": f"{policy} lost {dd:.0%} peak-to-trough on {sc}."})

    # --- 6. unfair compute -------------------------------------------
    tl = results.get("train_logs", {})
    params = {k: v.get("n_params", 0) for k, v in tl.items()
              if v.get("n_params") and v.get("family") != "baseline" or
              k in ("mlp", "transformer", "gnn", "obsspace")}
    params = {k: v for k, v in params.items() if v > 1000}
    if params:
        lo, hi = min(params.values()), max(params.values())
        if hi / max(lo, 1) > 10.0:
            findings.append({
                "severity": "medium", "check": "capacity_mismatch",
                "detail": (f"Parameter counts span {lo:,}-{hi:,} ({hi/max(lo,1):.1f}x); "
                           f"capability differences may be capacity differences.")})

    # --- 7. the environment must reward dynamics knowledge -------------
    summaries = control.get("summaries", {})
    if "oracle_plan" in summaries and "noop" in summaries:
        def mean_ret(p: str) -> float:
            vals = [v.get("total_return", 0.0) for v in summaries[p].values()]
            return float(np.mean(vals)) if vals else 0.0
        gap = mean_ret("oracle_plan") - mean_ret("noop")
        if gap <= 0.0:
            findings.append({
                "severity": "high", "check": "uninformative_environment",
                "detail": (f"Planning with the true simulator does not beat doing "
                           f"nothing (gap {gap:+.4f}). The world does not reward "
                           f"dynamics knowledge, so the whole comparison is "
                           f"uninformative.")})
        else:
            findings.append({
                "severity": "info", "check": "environment_is_informative",
                "detail": (f"True-simulator planning beats do-nothing by {gap:+.3f} "
                           f"return, so accurate dynamics are worth something here.")})

    return {"n_findings": len(findings),
            "by_severity": {s: sum(1 for f in findings if f["severity"] == s)
                            for s in ("high", "medium", "low", "info")},
            "findings": findings}


def format_audit(audit: Dict[str, Any]) -> str:
    lines = ["## Automated audit", ""]
    if not audit.get("findings"):
        lines.append("No findings.")
        return "\n".join(lines)
    counts = audit["by_severity"]
    lines.append("Counts: " + ", ".join(f"{k}={v}" for k, v in counts.items() if v))
    lines.append("")
    order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    for f in sorted(audit["findings"], key=lambda f: order.get(f["severity"], 9)):
        lines.append(f"- **{f['severity']}** · `{f['check']}` — {f['detail']}")
    lines.append("")
    return "\n".join(lines)
