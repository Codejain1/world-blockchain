"""Metric primitives.

Prediction quality is reported with *both* a sharpness metric (log loss, MSE) and
a calibration metric (Brier decomposition, ECE).  A model can be accurate and
badly calibrated, and for a planner the calibration is what matters: an
overconfident liquidation probability produces confidently wrong plans.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["mse", "mae", "rmse", "r2", "nrmse", "skill_score", "log_loss",
           "brier", "expected_calibration_error", "reliability_curve", "auroc",
           "sharpe", "max_drawdown", "sortino", "cumulative_return",
           "summarize_regression", "summarize_binary"]

EPS = 1e-12


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


# ------------------------------------------------------------- regression
def mse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(mse(y, p)))


def r2(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, EPS)


def nrmse(y: np.ndarray, p: np.ndarray) -> float:
    """RMSE normalised by the target's own standard deviation."""
    s = float(np.std(np.asarray(y, float)))
    return rmse(y, p) / max(s, EPS)


def skill_score(y: np.ndarray, p: np.ndarray, baseline: np.ndarray) -> float:
    """1 - MSE(model)/MSE(baseline).  Positive means better than the baseline."""
    return 1.0 - mse(y, p) / max(mse(y, baseline), EPS)


# ------------------------------------------------------- binary / events
def log_loss(y: np.ndarray, prob: np.ndarray, eps: float = 1e-7) -> float:
    y = np.asarray(y, float)
    p = np.clip(np.asarray(prob, float), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def brier(y: np.ndarray, prob: np.ndarray) -> float:
    return float(np.mean((np.asarray(prob, float) - np.asarray(y, float)) ** 2))


def expected_calibration_error(y: np.ndarray, prob: np.ndarray, n_bins: int = 15
                               ) -> float:
    y = np.asarray(y, float).reshape(-1)
    p = np.asarray(prob, float).reshape(-1)
    if y.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        ece += (m.mean()) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def reliability_curve(y: np.ndarray, prob: np.ndarray, n_bins: int = 10
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, float).reshape(-1)
    p = np.asarray(prob, float).reshape(-1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    conf = np.full(n_bins, np.nan)
    freq = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins)
    for b in range(n_bins):
        m = idx == b
        cnt[b] = m.sum()
        if m.any():
            conf[b] = p[m].mean()
            freq[b] = y[m].mean()
    return conf, freq, cnt


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC; ``nan`` when a label has only one class present."""
    y = np.asarray(y, float).reshape(-1)
    s = np.asarray(score, float).reshape(-1)
    pos, neg = y > 0.5, y <= 0.5
    if not pos.any() or not neg.any():
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n_pos, n_neg = float(pos.sum()), float(neg.sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def summarize_regression(y: np.ndarray, p: np.ndarray,
                         baseline: Optional[np.ndarray] = None) -> Dict[str, float]:
    out = {"mse": mse(y, p), "mae": mae(y, p), "rmse": rmse(y, p),
           "r2": r2(y.reshape(-1), p.reshape(-1)), "nrmse": nrmse(y, p)}
    if baseline is not None:
        out["skill_vs_persistence"] = skill_score(y, p, baseline)
    return out


def summarize_binary(y: np.ndarray, logit: np.ndarray,
                     names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Per-label and pooled event metrics."""
    y = np.asarray(y, float)
    p = _sigmoid(np.asarray(logit, float))
    out: Dict[str, Any] = {
        "log_loss": log_loss(y, p), "brier": brier(y, p),
        "ece": expected_calibration_error(y, p),
    }
    per: Dict[str, Dict[str, float]] = {}
    aucs: List[float] = []
    for j in range(y.shape[1]):
        nm = names[j] if names is not None else f"event_{j}"
        a = auroc(y[:, j], p[:, j])
        per[nm] = {"base_rate": float(y[:, j].mean()),
                   "log_loss": log_loss(y[:, j], p[:, j]),
                   "brier": brier(y[:, j], p[:, j]),
                   "ece": expected_calibration_error(y[:, j], p[:, j]),
                   "auroc": a}
        if np.isfinite(a):
            aucs.append(a)
    out["mean_auroc"] = float(np.mean(aucs)) if aucs else float("nan")
    out["per_event"] = per
    return out


# --------------------------------------------------------------- control
def cumulative_return(rewards: np.ndarray, initial: float) -> float:
    return float(np.sum(rewards) / max(abs(initial), EPS))


def sharpe(rewards: np.ndarray, scale: Optional[float] = None) -> float:
    """Sharpe-like ratio of per-step returns (no risk-free rate, no annualisation).

    Reported as a *relative* risk-adjustment measure between systems evaluated on
    identical episodes, not as a financial statistic.
    """
    r = np.asarray(rewards, float)
    if scale is not None and scale > 0:
        r = r / scale
    s = float(np.std(r))
    return float(np.mean(r) / max(s, EPS)) if r.size > 1 else 0.0


def sortino(rewards: np.ndarray) -> float:
    r = np.asarray(rewards, float)
    down = r[r < 0.0]
    d = float(np.std(down)) if down.size > 1 else 0.0
    return float(np.mean(r) / max(d, EPS))


def max_drawdown(equity: np.ndarray) -> float:
    e = np.asarray(equity, float)
    if e.size == 0:
        return 0.0
    peak = np.maximum.accumulate(e)
    dd = (peak - e) / np.maximum(np.abs(peak), EPS)
    return float(np.max(dd))
