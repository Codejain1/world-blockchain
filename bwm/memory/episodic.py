"""Long-term memory: episodic recall and consequence tracking.

Two components, both fed by the *same* consequence stream every system receives
through ``Policy.observe_outcome``:

``EpisodicMemory``
    A bounded store of (state key, action, outcome) tuples with nearest-neighbour
    retrieval.  It answers "the last few times the world looked like this and I
    did that, what happened?".

``ConsequenceTracker``
    Running per-action value estimates plus an exponential moving average of the
    world model's own one-step error.  The second signal is what lets the unified
    system notice that its model has stopped being trustworthy -- which is the
    whole point of not hard-wiring it to always simulate.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["MemoryEntry", "EpisodicMemory", "ConsequenceTracker"]


@dataclass
class MemoryEntry:
    key: np.ndarray            # compact state descriptor
    action: int
    reward: float
    events: np.ndarray
    t: int
    meta: Dict[str, Any] = field(default_factory=dict)


class EpisodicMemory:
    """Bounded episodic store with cosine/L2 nearest-neighbour retrieval."""

    def __init__(self, key_dim: int, capacity: int = 4096, n_actions: int = 1) -> None:
        self.key_dim = int(key_dim)
        self.capacity = int(capacity)
        self.n_actions = int(n_actions)
        self._keys = np.zeros((self.capacity, self.key_dim), dtype=np.float32)
        self._entries: List[Optional[MemoryEntry]] = [None] * self.capacity
        self._n = 0
        self._ptr = 0

    def __len__(self) -> int:
        return self._n

    def clear(self) -> None:
        self._n = 0
        self._ptr = 0
        self._keys[:] = 0.0
        self._entries = [None] * self.capacity

    def add(self, key: np.ndarray, action: int, reward: float,
            events: np.ndarray, t: int, meta: Optional[Dict[str, Any]] = None) -> None:
        k = np.asarray(key, dtype=np.float32).reshape(-1)[: self.key_dim]
        if k.size < self.key_dim:
            k = np.pad(k, (0, self.key_dim - k.size))
        self._keys[self._ptr] = k
        self._entries[self._ptr] = MemoryEntry(k, int(action), float(reward),
                                               np.asarray(events, np.float32),
                                               int(t), dict(meta or {}))
        self._ptr = (self._ptr + 1) % self.capacity
        self._n = min(self._n + 1, self.capacity)

    def retrieve(self, key: np.ndarray, k: int = 8
                 ) -> Tuple[List[MemoryEntry], np.ndarray]:
        if self._n == 0:
            return [], np.zeros(0, np.float32)
        q = np.asarray(key, dtype=np.float32).reshape(-1)[: self.key_dim]
        if q.size < self.key_dim:
            q = np.pad(q, (0, self.key_dim - q.size))
        keys = self._keys[: self._n]
        d = np.linalg.norm(keys - q[None, :], axis=1)
        idx = np.argsort(d)[: min(k, self._n)]
        return [self._entries[int(i)] for i in idx], d[idx]

    def action_values(self, key: np.ndarray, k: int = 16,
                      temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Distance-weighted value estimate per action from recalled neighbours.

        Returns ``(values, support)`` where ``support`` is the accumulated weight
        behind each action -- so a caller can tell "good" from "never tried".
        """
        vals = np.zeros(self.n_actions, np.float32)
        sup = np.zeros(self.n_actions, np.float32)
        entries, dists = self.retrieve(key, k)
        if not entries:
            return vals, sup
        w = np.exp(-dists / max(temperature * (np.median(dists) + 1e-6), 1e-6))
        for e, wi in zip(entries, w):
            vals[e.action] += wi * e.reward
            sup[e.action] += wi
        nz = sup > 0
        vals[nz] /= sup[nz]
        return vals, sup

    def mean_distance(self, key: np.ndarray, k: int = 8) -> float:
        _, d = self.retrieve(key, k)
        return float(d.mean()) if d.size else float("inf")


class ConsequenceTracker:
    """Online statistics used for continual adaptation."""

    def __init__(self, n_actions: int, decay: float = 0.98) -> None:
        self.n_actions = int(n_actions)
        self.decay = float(decay)
        self.action_value = np.zeros(self.n_actions, np.float32)
        self.action_count = np.zeros(self.n_actions, np.float32)
        self.model_error_ema: float = 0.0
        self.model_error_n: int = 0
        self.reward_ema: float = 0.0
        self.recent_rewards: Deque[float] = deque(maxlen=64)
        # Running scale of the reward stream, used to z-score the learning signal.
        self._rew_m2: float = 0.0
        self._rew_mean: float = 0.0
        self._rew_n: int = 0

    def reset(self) -> None:
        self.action_value[:] = 0.0
        self.action_count[:] = 0.0
        self.model_error_ema = 0.0
        self.model_error_n = 0
        self.reward_ema = 0.0
        self.recent_rewards.clear()
        self._rew_m2 = 0.0
        self._rew_mean = 0.0
        self._rew_n = 0

    def update_action(self, action: int, reward: float) -> None:
        a = int(action) % self.n_actions
        self.action_count[a] = self.decay * self.action_count[a] + 1.0
        lr = 1.0 / max(self.action_count[a], 1.0)
        self.action_value[a] += lr * (float(reward) - self.action_value[a])
        self.reward_ema = 0.95 * self.reward_ema + 0.05 * float(reward)
        self.recent_rewards.append(float(reward))
        # Welford update for the running reward scale.
        self._rew_n += 1
        d = float(reward) - self._rew_mean
        self._rew_mean += d / self._rew_n
        self._rew_m2 += d * (float(reward) - self._rew_mean)

    def update_model_error(self, err: float) -> None:
        w = 0.1 if self.model_error_n > 0 else 1.0
        self.model_error_ema = (1 - w) * self.model_error_ema + w * float(err)
        self.model_error_n += 1

    @property
    def reward_scale(self) -> float:
        """Running standard deviation of the reward stream (>= a small floor)."""
        if self._rew_n < 2:
            return 1.0
        return float(max(np.sqrt(self._rew_m2 / (self._rew_n - 1)), 1e-9))

    def standardize(self, reward: float, clip: float = 3.0) -> float:
        """Z-score a reward against its own running scale.

        The bandit gate needs an O(1) learning signal.  Dividing by a *global*
        dataset reward scale makes the focal agent's per-step rewards far
        smaller than the UCB exploration bonus, so the gate would never leave
        exploration and the unified system would be handicapped by its own
        arbitration layer rather than by its components.
        """
        if self._rew_n < 4:
            return 0.0
        z = (float(reward) - self._rew_mean) / self.reward_scale
        return float(np.clip(z, -clip, clip))

    @property
    def reward_volatility(self) -> float:
        return float(np.std(self.recent_rewards)) if len(self.recent_rewards) > 2 else 0.0

    def ucb(self, c: float = 1.0) -> np.ndarray:
        total = max(self.action_count.sum(), 1.0)
        bonus = c * np.sqrt(np.log(total + 1.0) / np.maximum(self.action_count, 1e-3))
        return self.action_value + bonus
