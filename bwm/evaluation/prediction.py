"""Prediction tasks: next state, events, market state, long horizon, reward.

Every model is scored on *identical* batches drawn with a fixed seed, and every
regression number is additionally reported as a skill score against persistence,
because raw MSE on normalised features is hard to interpret and easy to
misrepresent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..data.dataset import BatchSpec, Normalizer, TrajectorySet
from ..environment.types import EVENT_NAMES
from .metrics import (log_loss, mse, r2, skill_score, summarize_binary,
                      summarize_regression)

__all__ = ["MARKET_FEATURE_PREFIXES", "evaluate_prediction", "sample_eval_batches"]

#: Features that constitute "market state" for the price/market-state task.
MARKET_FEATURE_PREFIXES: Tuple[str, ...] = (
    "logprice_", "basis_", "pool_logtvl_", "util_", "borrow_rate_",
    "log_total_tvl", "log_gas_base_fee")


def market_feature_indices(feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray([i for i, n in enumerate(feature_names)
                       if n.startswith(MARKET_FEATURE_PREFIXES)], dtype=np.int64)


def sample_eval_batches(ts: TrajectorySet, spec: BatchSpec, normalizer: Normalizer,
                        n_samples: int = 6000, seed: int = 0,
                        with_graph: bool = True) -> Dict[str, np.ndarray]:
    """One fixed evaluation batch, shared by every model."""
    idx = ts.index(spec.history, spec.horizon)
    rng = np.random.Generator(np.random.PCG64(seed))
    if len(idx) > n_samples:
        idx = idx[rng.choice(len(idx), n_samples, replace=False)]
    return ts.batch(idx, spec, normalizer, with_graph=with_graph)


def evaluate_prediction(model, batch: Dict[str, np.ndarray], spec: BatchSpec,
                        feature_names: Sequence[str],
                        horizons: Sequence[int] = (1, 2, 4, 8),
                        chunk: int = 1024) -> Dict[str, Any]:
    """Score a :class:`PredictiveModel` on one evaluation batch.

    Returns a nested dict with one entry per task.  ``chunk`` bounds peak memory
    for the heavier models; it does not change the result.
    """
    n = int(batch["obs_hist"].shape[0])
    market_idx = market_feature_indices(feature_names)
    out: Dict[str, Any] = {}

    # ---- one step -----------------------------------------------------
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
    out["next_state"] = summarize_regression(y_delta, delta_hat, baseline=zero)
    out["next_state"]["skill_vs_persistence"] = skill_score(y_delta, delta_hat, zero)

    out["market_state"] = summarize_regression(
        y_delta[:, market_idx], delta_hat[:, market_idx],
        baseline=zero[:, market_idx])

    out["event"] = summarize_binary(batch["event"], ev_hat, EVENT_NAMES)

    y_r = batch["reward"]
    out["reward"] = summarize_regression(y_r, rw_hat,
                                         baseline=np.full_like(y_r, float(y_r.mean())))

    # ---- long horizon --------------------------------------------------
    hmax = int(max(horizons))
    hmax = min(hmax, int(batch["fut_obs"].shape[1]))
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

    per_h: Dict[str, Dict[str, float]] = {}
    last = batch["obs_hist"][:, -1]
    for h in horizons:
        if h > hmax:
            continue
        i = h - 1
        y = batch["fut_obs"][:, i]
        persist = last                        # "nothing changes" over h steps
        per_h[str(h)] = {
            "obs_mse": mse(y, obs_pred[:, i]),
            "obs_skill_vs_persistence": skill_score(y, obs_pred[:, i], persist),
            "market_mse": mse(y[:, market_idx], obs_pred[:, i][:, market_idx]),
            "event_log_loss": log_loss(batch["fut_event"][:, i],
                                       1.0 / (1.0 + np.exp(-np.clip(ev_pred[:, i],
                                                                    -60, 60)))),
            "reward_mse": mse(batch["fut_reward"][:, i], rw_pred[:, i]),
        }
    out["long_horizon"] = per_h
    out["n_samples"] = n
    return out
