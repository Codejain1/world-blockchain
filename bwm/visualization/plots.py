"""Figures for the report.  Every plot is derived from the saved results JSON."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["make_all_plots"]

_PALETTE = ["#2E6F9E", "#C4573B", "#4F9D69", "#8E6BB0", "#D9A441", "#5C5C5C",
            "#3AA9A0", "#B0466F", "#7A8B3C", "#9C6B3F", "#4666B0", "#A03A3A"]


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _family(name: str, train_logs: Dict[str, Any]) -> str:
    return train_logs.get(name, {}).get("family", "baseline")


def plot_prediction_skill(results: Dict[str, Any], out: str) -> None:
    pred = results.get("prediction", {})
    if not pred:
        return
    models = list(pred)
    splits = list(next(iter(pred.values())).keys())
    tl = results.get("train_logs", {})
    fig, ax = plt.subplots(figsize=(1.6 + 0.9 * len(splits), 4.2))
    w = 0.8 / max(len(models), 1)
    x = np.arange(len(splits))
    for i, m in enumerate(models):
        vals = [pred[m].get(s, {}).get("decision_state", {}).get(
            "skill_vs_best_naive", np.nan) for s in splits]
        hatch = "//" if _family(m, tl) == "world_model" else None
        ax.bar(x + i * w, vals, w, label=m, color=_PALETTE[i % len(_PALETTE)],
               hatch=hatch, edgecolor="white", linewidth=0.4)
    ax.axhline(0.0, color="k", linewidth=1.0)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels([s.replace("ood_", "") for s in splits], rotation=30,
                       ha="right", fontsize=8)
    _style(ax, "Decision-state prediction skill vs best naive reference "
               "(hatched = world model)", "", "skill score")
    ax.legend(fontsize=6.5, ncol=2, frameon=False)
    _save(fig, os.path.join(out, "prediction_skill.png"))


def plot_long_horizon(results: Dict[str, Any], out: str, split: str = "test_iid") -> None:
    pred = results.get("prediction", {})
    if not pred:
        return
    tl = results.get("train_logs", {})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, sp, title in ((axes[0], split, "in-distribution"),
                          (axes[1], _first_ood(pred), "held-out world")):
        if sp is None:
            continue
        for i, m in enumerate(pred):
            lh = pred[m].get(sp, {}).get("long_horizon", {})
            if not lh:
                continue
            hs = sorted(int(k) for k in lh)
            ys = [lh[str(h)].get("decision_skill_vs_best_naive", np.nan) for h in hs]
            ax.plot(hs, ys, marker="o", ms=3.5, lw=1.4, label=m,
                    color=_PALETTE[i % len(_PALETTE)],
                    ls="-" if _family(m, tl) == "world_model" else "--")
        ax.axhline(0.0, color="k", lw=1.0)
        _style(ax, f"Long-horizon decision-state skill ({title}: {sp})",
               "horizon (steps)", "skill")
    axes[1].legend(fontsize=6.5, ncol=2, frameon=False)
    _save(fig, os.path.join(out, "long_horizon.png"))


def _first_ood(pred: Dict[str, Any]) -> Optional[str]:
    for m in pred.values():
        for k in m:
            if k.startswith("ood_"):
                return k
    return None


def plot_calibration(results: Dict[str, Any], out: str, split: str = "test_iid",
                     top: int = 6) -> None:
    pred = results.get("prediction", {})
    if not pred:
        return
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], color="k", lw=1.0, ls=":", label="perfect")
    n = 0
    for i, m in enumerate(pred):
        rel = pred[m].get(split, {}).get("event", {}).get("reliability")
        if not rel or n >= top:
            continue
        c = np.asarray(rel["confidence"], float)
        f = np.asarray(rel["frequency"], float)
        keep = (c >= 0) & (f >= 0)
        if keep.sum() < 2:
            continue
        ax.plot(c[keep], f[keep], marker="o", ms=3.5, lw=1.3, label=m,
                color=_PALETTE[i % len(_PALETTE)])
        n += 1
    _style(ax, f"Event calibration ({split})", "predicted probability",
           "observed frequency")
    ax.legend(fontsize=7, frameon=False)
    _save(fig, os.path.join(out, "calibration.png"))


def plot_counterfactual(results: Dict[str, Any], out: str) -> None:
    rows = results.get("tables", {}).get("counterfactual")
    if not rows:
        return
    rows = [r for r in rows if r.get("cf_skill_mean") is not None]
    rows.sort(key=lambda r: r["cf_skill_mean"])
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(rows) + 2))
    tl = results.get("train_logs", {})
    colors = ["#C4573B" if _family(r["model"], tl) == "world_model" else "#2E6F9E"
              for r in rows]
    ax.barh([r["model"] for r in rows], [r["cf_skill_mean"] for r in rows],
            color=colors, edgecolor="white")
    ax.axvline(0.0, color="k", lw=1.0)
    _style(ax, "Counterfactual skill vs a zero-effect predictor "
               "(red = world model)", "skill score", "")
    _save(fig, os.path.join(out, "counterfactual.png"))


def plot_control(results: Dict[str, Any], out: str) -> None:
    ctrl = results.get("control")
    if not ctrl:
        return
    summaries = ctrl["summaries"]
    scenarios = ctrl["scenarios"]
    pols = list(summaries)
    fig, ax = plt.subplots(figsize=(1.8 + 1.1 * len(scenarios), 4.4))
    w = 0.8 / max(len(pols), 1)
    x = np.arange(len(scenarios))
    for i, p in enumerate(pols):
        vals = [summaries[p].get(s, {}).get("total_return", np.nan) for s in scenarios]
        errs = [summaries[p].get(s, {}).get("total_return_se", 0.0) for s in scenarios]
        ax.bar(x + i * w, vals, w, yerr=errs, capsize=1.5, label=p,
               color=_PALETTE[i % len(_PALETTE)], edgecolor="white", linewidth=0.4)
    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(scenarios, rotation=25, ha="right", fontsize=8)
    _style(ax, "Control: total return per episode (error bars = s.e. over episodes)",
           "", "total return")
    ax.legend(fontsize=6.5, ncol=2, frameon=False)
    _save(fig, os.path.join(out, "control_return.png"))


def plot_compute_tradeoff(results: Dict[str, Any], out: str) -> None:
    comp = results.get("composite", {})
    tl = results.get("train_logs", {})
    pts = [(tl.get(k, {}).get("n_params", 0), v.get("composite"), k)
           for k, v in comp.items()
           if v.get("composite") is not None and k in tl]
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for i, (p, c, k) in enumerate(pts):
        ax.scatter(max(p, 1), c, s=42, color=_PALETTE[i % len(_PALETTE)],
                   marker="^" if _family(k, tl) == "world_model" else "o")
        ax.annotate(k, (max(p, 1), c), fontsize=6.5, xytext=(4, 3),
                    textcoords="offset points")
    ax.set_xscale("log")
    _style(ax, "Capability vs capacity (triangles = world models)",
           "parameters (log)", "composite GPC index")
    _save(fig, os.path.join(out, "compute_tradeoff.png"))


def plot_components(results: Dict[str, Any], out: str) -> None:
    from ..evaluation.benchmark import COMPONENTS
    comp = results.get("composite", {})
    rows = [(k, v) for k, v in comp.items() if v.get("coverage", 0) >= 0.8]
    if not rows:
        rows = list(comp.items())
    rows = sorted(rows, key=lambda kv: -(kv[1].get("composite") or 0))[:10]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(1.6 + 1.0 * len(COMPONENTS), 4.4))
    x = np.arange(len(COMPONENTS))
    w = 0.8 / max(len(rows), 1)
    for i, (name, c) in enumerate(rows):
        vals = [c["components"].get(k) if c["components"].get(k) is not None else 0.0
                for k in COMPONENTS]
        ax.bar(x + i * w, vals, w, label=name, color=_PALETTE[i % len(_PALETTE)],
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(COMPONENTS, rotation=22, ha="right", fontsize=8)
    _style(ax, "Composite components (missing components shown as 0)", "", "score 0-1")
    ax.legend(fontsize=6.5, ncol=2, frameon=False)
    _save(fig, os.path.join(out, "components.png"))


def plot_gate(results: Dict[str, Any], out: str) -> None:
    info = results.get("control", {}).get("policy_info", {})
    rows = [(k, v["extra"]["gate_usage"]) for k, v in info.items()
            if v.get("extra", {}).get("gate_usage")]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(6, 0.7 * len(rows) + 2))
    labels = [r[0] for r in rows]
    shares = np.asarray([r[1]["share"] for r in rows], float)
    left = np.zeros(len(rows))
    for j, nm in enumerate(["reason", "plan", "recall"]):
        ax.barh(labels, shares[:, j], left=left, label=nm, color=_PALETTE[j],
                edgecolor="white")
        left += shares[:, j]
    _style(ax, "Unified system: share of decisions per pathway", "share", "")
    ax.legend(fontsize=8, frameon=False, ncol=3)
    _save(fig, os.path.join(out, "gate_usage.png"))


def make_all_plots(results: Dict[str, Any], out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    for fn in (plot_prediction_skill, plot_long_horizon, plot_calibration,
               plot_counterfactual, plot_control, plot_compute_tradeoff,
               plot_components, plot_gate):
        try:
            fn(results, out_dir)
        except Exception as e:
            print(f"  (plot {fn.__name__} failed: {type(e).__name__}: {e})")
    return sorted(os.listdir(out_dir))
