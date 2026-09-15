"""Prediction tasks, scored by *mechanism-based target group*.

Why the targets are grouped
---------------------------
Measuring one pooled MSE over all 112 observation features is misleading in this
world, and the grouping below exists because the pooled number was measured and
found to be dominated by two artefacts:

``price``       ``ret1_*`` and ``pool_move_*`` are *already* one-step
                differences, so predicting their change is mostly the identity
                "next return ~ its mean".  A ridge fit scores **+0.99** skill
                there while learning nothing.  ``logprice_*`` is the opposite:
                the generative process is a regime-switching geometric random
                walk, so its one-step change is a martingale that *nobody* can
                predict.  We therefore report ``price_level`` separately as a
                **leakage canary** -- a system scoring clearly above zero there
                is reading something it should not be able to read.

``protocol``    endogenous protocol state (utilisation, rates, pool depth,
                basis, realised volatility, gas).  Driven by agent behaviour,
                genuinely learnable, best linear skill ~+0.06 with individual
                features up to +0.29.

``agent``       the focal account's own balance sheet.  Directly action
                dependent -- best linear skill ~+0.45 -- and therefore the
                channel where action-conditioned understanding shows up.

``population``  per-cluster aggregates of the other agents.

The composite index uses ``agent`` and ``protocol``; ``price`` is reported but
excluded, because rewarding a model for the return identity would reward
bookkeeping rather than understanding.

Reference models
----------------
Skill is reported against persistence, against a fitted linear autoregression,
and against ``best_naive`` = whichever of the two has lower error on that split.
The last one is the headline: a fitted linear reference can blow up on held-out
worlds (it does, on ``ood_unseen_agents``), which would otherwise hand every
model a free skill of 1.0.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..data.dataset import BatchSpec, Normalizer, TrajectorySet
from ..environment.types import EVENT_NAMES
from .metrics import (log_loss, mae, mse, r2, skill_score, summarize_binary,
                      summarize_regression)

__all__ = ["TARGET_GROUPS", "COMPOSITE_GROUPS", "group_indices", "evaluate_prediction",
           "sample_eval_batches", "reference_predictions", "market_feature_indices"]

_PRICE = ("logprice_", "ret1_", "pool_move_")
_PRICE_LEVEL = ("logprice_",)
_AGENT = ("bal_frac_", "sup_frac_", "bor_frac_", "lpv_frac_", "cash_frac", "lp_frac",
          "stake_frac", "leverage", "inv_health", "borrow_cap_frac", "log_net_worth",
          "pnl_frac")
_POPULATION = ("clu",)

#: group name -> membership test applied to a feature name.
TARGET_GROUPS: Tuple[str, ...] = ("price", "price_level", "protocol", "agent",
                                  "population")

#: Groups that enter the composite index (see the module docstring).
COMPOSITE_GROUPS: Tuple[str, ...] = ("agent", "protocol")

#: Retained for backward compatibility with older result files.
MARKET_FEATURE_PREFIXES: Tuple[str, ...] = (
    "logprice_", "basis_", "pool_logtvl_", "util_", "borrow_rate_",
    "log_total_tvl", "log_gas_base_fee")


def _group_of(name: str) -> str:
    if name.startswith(_PRICE):
        return "price"
    if name.startswith(_AGENT):
        return "agent"
    if name.startswith(_POPULATION):
        return "population"
    return "protocol"


def group_indices(feature_names: Sequence[str]) -> Dict[str, np.ndarray]:
    """Map each target group to the observation indices it covers."""
    out: Dict[str, List[int]] = {g: [] for g in TARGET_GROUPS}
    for i, n in enumerate(feature_names):
        out[_group_of(n)].append(i)
        if n.startswith(_PRICE_LEVEL):
            out["price_level"].append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in out.items() if v}


def market_feature_indices(feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray([i for i, n in enumerate(feature_names)
                       if n.startswith(MARKET_FEATURE_PREFIXES)], dtype=np.int64)


# --------------------------------------------------------------------------
def sample_eval_batches(ts: TrajectorySet, spec: BatchSpec, normalizer: Normalizer,
                        n_samples: int = 6000, seed: int = 0,
                        with_graph: bool = True) -> Dict[str, np.ndarray]:
    """One fixed evaluation batch, shared by every model on that split."""
    idx = ts.index(spec.history, spec.horizon)
    rng = np.random.Generator(np.random.PCG64(seed))
    if len(idx) > n_samples:
        idx = idx[rng.choice(len(idx), n_samples, replace=False)]
    return ts.batch(idx, spec, normalizer, with_graph=with_graph)


def reference_predictions(model, batch: Dict[str, np.ndarray], horizon: int,
                          chunk: int = 1024) -> Dict[str, np.ndarray]:
    """Cache a reference model's predictions for reuse across every system."""
    n = int(batch["obs_hist"].shape[0])
    deltas, obs = [], []
    for s in range(0, n, chunk):
        sub = {k: v[s:s + chunk] for k, v in batch.items()}
        deltas.append(model.predict_step(sub).delta)
        obs.append(model.predict_rollout(sub, horizon).obs)
    return {"delta": np.concatenate(deltas, 0), "obs": np.concatenate(obs, 0)}


def _group_scores(y: np.ndarray, pred: np.ndarray, groups: Dict[str, np.ndarray],
                  refs: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Per-group MSE and skill against persistence / AR / the better of the two."""
    out: Dict[str, Dict[str, float]] = {}
    for g, idx in groups.items():
        yy, pp = y[:, idx], pred[:, idx]
        m = mse(yy, pp)
        row: Dict[str, float] = {"mse": m, "n_features": int(idx.size),
                                 "target_rms": float(np.sqrt(np.mean(yy ** 2)))}
        ref_mses: Dict[str, float] = {}
        for rname, rpred in refs.items():
            rm = mse(yy, rpred[:, idx])
            ref_mses[rname] = rm
            row[f"skill_vs_{rname}"] = 1.0 - m / max(rm, 1e-12)
        if ref_mses:
            best = min(ref_mses.values())
            row["skill_vs_best_naive"] = 1.0 - m / max(best, 1e-12)
            row["best_naive"] = min(ref_mses, key=lambda k: ref_mses[k])
        out[g] = row
    return out


def evaluate_prediction(model, batch: Dict[str, np.ndarray], spec: BatchSpec,
                        feature_names: Sequence[str],
                        horizons: Sequence[int] = (1, 2, 4, 8),
                        chunk: int = 1024,
                        reference: Optional[Dict[str, np.ndarray]] = None
                        ) -> Dict[str, Any]:
    """Score one :class:`PredictiveModel` on one evaluation batch."""
    n = int(batch["obs_hist"].shape[0])
    groups = group_indices(feature_names)
    out: Dict[str, Any] = {}

    # ---- one-step ------------------------------------------------------
    deltas, ev_logits, rewards = [], [], []
    for s in range(0, n, chunk):
        sub = {k: v[s:s + chunk] for k, v in batch.items()}
        p = model.predict_step(sub)
        deltas.append(p.delta)
        ev_logits.append(p.event_logit)
        rewards.append(p.reward)
    delta_hat = np.concatenate(deltas, 0)
    ev_hat = np.concatenate(ev_logits, 0)
    rw_hat = np.concatenate(rewards, 0)

    y_delta = batch["delta"]
    zero = np.zeros_like(y_delta)
    refs: Dict[str, np.ndarray] = {"persistence": zero}
    if reference is not None:
        refs["ar"] = reference["delta"]

    out["next_state"] = summarize_regression(y_delta, delta_hat, baseline=zero)
    out["next_state"]["skill_vs_persistence"] = skill_score(y_delta, delta_hat, zero)
    if reference is not None:
        out["next_state"]["skill_vs_ar"] = skill_score(y_delta, delta_hat,
                                                       reference["delta"])
    out["groups"] = _group_scores(y_delta, delta_hat, groups, refs)

    # Combined decision-relevant score: the groups the composite uses, pooled.
    comp_idx = np.concatenate([groups[g] for g in COMPOSITE_GROUPS if g in groups])
    m = mse(y_delta[:, comp_idx], delta_hat[:, comp_idx])
    ref_m = min(mse(y_delta[:, comp_idx], r[:, comp_idx]) for r in refs.values())
    out["decision_state"] = {"mse": m, "skill_vs_best_naive": 1.0 - m / max(ref_m, 1e-12)}

    out["event"] = summarize_binary(batch["event"], ev_hat, EVENT_NAMES)
    y_r = batch["reward"]
    out["reward"] = summarize_regression(y_r, rw_hat,
                                         baseline=np.full_like(y_r, float(y_r.mean())))
    out["reward"]["skill_vs_mean"] = skill_score(y_r, rw_hat,
                                                 np.full_like(y_r, float(y_r.mean())))

    # ---- multi-step ----------------------------------------------------
    hmax = min(int(max(horizons)), int(batch["fut_obs"].shape[1]))
    obs_pred, ev_pred, rw_pred = [], [], []
    for s in range(0, n, chunk):
        sub = {k: v[s:s + chunk] for k, v in batch.items()}
        r = model.predict_rollout(sub, hmax)
        obs_pred.append(r.obs)
        ev_pred.append(r.event_logit)
        rw_pred.append(r.reward)
    obs_pred = np.concatenate(obs_pred, 0)
    ev_pred = np.concatenate(ev_pred, 0)
    rw_pred = np.concatenate(rw_pred, 0)

    last = batch["obs_hist"][:, -1]
    per_h: Dict[str, Dict[str, Any]] = {}
    for h in horizons:
        if h > hmax:
            continue
        i = h - 1
        y = batch["fut_obs"][:, i]
        hrefs: Dict[str, np.ndarray] = {"persistence": last}
        if reference is not None and reference["obs"].shape[1] > i:
            hrefs["ar"] = reference["obs"][:, i]
        g = _group_scores(y, obs_pred[:, i], groups, hrefs)
        comp_m = mse(y[:, comp_idx], obs_pred[:, i][:, comp_idx])
        comp_ref = min(mse(y[:, comp_idx], r[:, comp_idx]) for r in hrefs.values())
        per_h[str(h)] = {
            "obs_mse": mse(y, obs_pred[:, i]),
            "obs_skill_vs_persistence": skill_score(y, obs_pred[:, i], last),
            "decision_skill_vs_best_naive": 1.0 - comp_m / max(comp_ref, 1e-12),
            "event_log_loss": log_loss(batch["fut_event"][:, i],
                                       1.0 / (1.0 + np.exp(-np.clip(ev_pred[:, i],
                                                                    -60, 60)))),
            "reward_mse": mse(batch["fut_reward"][:, i], rw_pred[:, i]),
            "groups": {k: {"skill_vs_best_naive": v.get("skill_vs_best_naive"),
                           "mse": v["mse"]} for k, v in g.items()},
        }
    out["long_horizon"] = per_h
    out["n_samples"] = n
    return out
