"""Windowed views over trajectories, with train-only normalisation.

All models -- baselines, world models and the unified system -- consume batches
produced here.  Giving every system the *same* window length ``H`` and the same
feature normalisation is how the "identical observation budget" fairness
requirement is enforced in code rather than by convention.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..environment.runner import EXTRA_NAMES, Trajectory

__all__ = ["Normalizer", "TrajectorySet", "make_batch", "BatchSpec"]


# --------------------------------------------------------------------------
@dataclass
class BatchSpec:
    """Shapes shared by every model in an experiment."""

    history: int = 8        # H: observations visible to *every* system
    horizon: int = 8        # L: imagination / long-horizon prediction depth
    obs_dim: int = 0
    n_actions: int = 0
    n_events: int = 0
    n_nodes: int = 0
    node_dim: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return dict(history=self.history, horizon=self.horizon, obs_dim=self.obs_dim,
                    n_actions=self.n_actions, n_events=self.n_events,
                    n_nodes=self.n_nodes, node_dim=self.node_dim)


class Normalizer:
    """Per-feature standardisation fitted on the training split only."""

    def __init__(self, mean: np.ndarray, std: np.ndarray,
                 node_mean: Optional[np.ndarray] = None,
                 node_std: Optional[np.ndarray] = None,
                 reward_scale: float = 1.0) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.node_mean = None if node_mean is None else np.asarray(node_mean, np.float32)
        self.node_std = None if node_std is None else np.asarray(node_std, np.float32)
        self.reward_scale = float(reward_scale)

    @staticmethod
    def fit(trajs: Sequence[Trajectory], clip: float = 1e-6) -> "Normalizer":
        obs = np.concatenate([t.obs for t in trajs], axis=0)
        mean, std = obs.mean(axis=0), obs.std(axis=0)
        std = np.maximum(std, clip)
        nm = ns = None
        if trajs[0].node_feat.ndim == 3 and trajs[0].node_feat.shape[1] > 1:
            nf = np.concatenate([t.node_feat for t in trajs], axis=0)
            nm, ns = nf.mean(axis=(0,)), np.maximum(nf.std(axis=(0,)), clip)
        rew = np.concatenate([t.rewards for t in trajs])
        scale = float(max(np.std(rew), 1.0))
        return Normalizer(mean, std, nm, ns, scale)

    # -- transforms ------------------------------------------------------
    def obs(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def inv_obs(self, z: np.ndarray) -> np.ndarray:
        return (z * self.std + self.mean).astype(np.float32)

    def obs_delta(self, d: np.ndarray) -> np.ndarray:
        """Deltas share the observation scale but not its offset."""
        return (d / self.std).astype(np.float32)

    def inv_obs_delta(self, d: np.ndarray) -> np.ndarray:
        return (d * self.std).astype(np.float32)

    def node(self, x: np.ndarray) -> np.ndarray:
        if self.node_mean is None:
            return x.astype(np.float32)
        return ((x - self.node_mean) / self.node_std).astype(np.float32)

    def reward(self, r: np.ndarray) -> np.ndarray:
        return (np.asarray(r, dtype=np.float32) / self.reward_scale).astype(np.float32)

    def inv_reward(self, r: np.ndarray) -> np.ndarray:
        return (np.asarray(r, dtype=np.float32) * self.reward_scale).astype(np.float32)

    # -- io ---------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std,
                 node_mean=np.array([]) if self.node_mean is None else self.node_mean,
                 node_std=np.array([]) if self.node_std is None else self.node_std,
                 reward_scale=np.array(self.reward_scale))

    @staticmethod
    def load(path: str) -> "Normalizer":
        z = np.load(path)
        nm = z["node_mean"] if z["node_mean"].size else None
        ns = z["node_std"] if z["node_std"].size else None
        return Normalizer(z["mean"], z["std"], nm, ns, float(z["reward_scale"]))


# --------------------------------------------------------------------------
class TrajectorySet:
    """A split: a list of trajectories plus a window index."""

    def __init__(self, trajs: Sequence[Trajectory], name: str = "") -> None:
        if not trajs:
            raise ValueError("TrajectorySet requires at least one trajectory")
        self.trajs: List[Trajectory] = list(trajs)
        self.name = name

    def __len__(self) -> int:
        return len(self.trajs)

    @property
    def obs_dim(self) -> int:
        return int(self.trajs[0].obs.shape[1])

    @property
    def n_events(self) -> int:
        return int(self.trajs[0].events.shape[1])

    @property
    def n_actions(self) -> int:
        return int(self.trajs[0].meta.get("n_actions", int(max(t.actions.max()
                                                              for t in self.trajs)) + 1))

    @property
    def node_shape(self) -> Tuple[int, int]:
        nf = self.trajs[0].node_feat
        return (int(nf.shape[1]), int(nf.shape[2]))

    def spec(self, history: int = 8, horizon: int = 8) -> BatchSpec:
        n, f = self.node_shape
        return BatchSpec(history=history, horizon=horizon, obs_dim=self.obs_dim,
                         n_actions=self.n_actions, n_events=self.n_events,
                         n_nodes=n, node_dim=f)

    def index(self, history: int = 8, horizon: int = 8) -> np.ndarray:
        """All valid ``(trajectory, t)`` pairs with a full window and horizon."""
        pairs: List[Tuple[int, int]] = []
        for i, tr in enumerate(self.trajs):
            lo, hi = history - 1, tr.T - horizon
            pairs.extend((i, t) for t in range(lo, max(hi, lo)))
        return np.asarray(pairs, dtype=np.int64)

    def batch(self, pairs: np.ndarray, spec: BatchSpec,
              normalizer: Optional[Normalizer] = None,
              with_graph: bool = True) -> Dict[str, np.ndarray]:
        return make_batch(self.trajs, pairs, spec, normalizer, with_graph)

    def stats(self) -> Dict[str, Any]:
        ev = np.concatenate([t.events for t in self.trajs], axis=0)
        rw = np.concatenate([t.rewards for t in self.trajs])
        return {
            "n_episodes": len(self.trajs),
            "n_steps": int(sum(t.T for t in self.trajs)),
            "event_rate": ev.mean(axis=0).tolist(),
            "reward_mean": float(rw.mean()),
            "reward_std": float(rw.std()),
            "scenario": self.trajs[0].meta.get("split_scenario", ""),
            "kind": self.trajs[0].meta.get("kind", ""),
        }


def make_batch(trajs: Sequence[Trajectory], pairs: np.ndarray, spec: BatchSpec,
               normalizer: Optional[Normalizer] = None,
               with_graph: bool = True) -> Dict[str, np.ndarray]:
    """Assemble one batch.

    Conventions
    -----------
    ``obs_hist[b]``  = obs[t-H+1 .. t]            (H, D)
    ``act_hist[b]``  = actions[t-H+1 .. t]        (H,)  -- last entry is ``a_t``
    ``next_obs[b]``  = obs[t+1]                   (D,)
    ``fut_obs[b]``   = obs[t+1 .. t+L]            (L, D)
    ``fut_act[b]``   = actions[t .. t+L-1]        (L,)  -- drives imagination
    """
    H, L = spec.history, spec.horizon
    B = len(pairs)
    D, E = spec.obs_dim, spec.n_events
    N, F = spec.n_nodes, spec.node_dim

    obs_hist = np.zeros((B, H, D), np.float32)
    act_hist = np.zeros((B, H), np.int64)
    next_obs = np.zeros((B, D), np.float32)
    delta = np.zeros((B, D), np.float32)
    event = np.zeros((B, E), np.float32)
    reward = np.zeros((B,), np.float32)
    fut_obs = np.zeros((B, L, D), np.float32)
    fut_act = np.zeros((B, L), np.int64)
    fut_event = np.zeros((B, L, E), np.float32)
    fut_reward = np.zeros((B, L), np.float32)
    regime = np.zeros((B,), np.int64)
    node_hist = np.zeros((B, H, N, F), np.float32) if with_graph else np.zeros((B, 1), np.float32)

    for b, (i, t) in enumerate(pairs):
        tr = trajs[int(i)]
        t = int(t)
        obs_hist[b] = tr.obs[t - H + 1: t + 1]
        act_hist[b] = tr.actions[t - H + 1: t + 1]
        next_obs[b] = tr.obs[t + 1]
        delta[b] = tr.obs[t + 1] - tr.obs[t]
        event[b] = tr.events[t]
        reward[b] = tr.rewards[t]
        fut_obs[b] = tr.obs[t + 1: t + 1 + L]
        fut_act[b] = tr.actions[t: t + L]
        fut_event[b] = tr.events[t: t + L]
        fut_reward[b] = tr.rewards[t: t + L]
        regime[b] = tr.regime[t]
        if with_graph:
            node_hist[b] = tr.node_feat[t - H + 1: t + 1]

    out: Dict[str, np.ndarray] = {
        "obs_hist": obs_hist, "act_hist": act_hist, "next_obs": next_obs,
        "delta": delta, "event": event, "reward": reward,
        "fut_obs": fut_obs, "fut_act": fut_act, "fut_event": fut_event,
        "fut_reward": fut_reward, "regime": regime, "node_hist": node_hist,
    }
    if normalizer is not None:
        out["obs_hist"] = normalizer.obs(obs_hist)
        out["next_obs"] = normalizer.obs(next_obs)
        out["fut_obs"] = normalizer.obs(fut_obs)
        out["delta"] = normalizer.obs_delta(delta)
        out["reward"] = normalizer.reward(reward)
        out["fut_reward"] = normalizer.reward(fut_reward)
        if with_graph:
            out["node_hist"] = normalizer.node(node_hist)
    return out
