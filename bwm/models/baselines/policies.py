"""Control baselines: do nothing, act at random, buy and hold, follow the crowd.

These bound the control tasks from below and from the side.  "Do nothing" is a
surprisingly strong policy in a world with gas fees, and any system that cannot
beat it is not adding value.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ...evaluation.compute import ComputeMeter
from ..base import DecisionContext, ModelInfo, Policy

__all__ = ["NoopPolicy", "RandomPolicy", "BuyAndHoldPolicy", "GreedyMemoryPolicy"]


class _Base(Policy):
    family = "baseline"

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.meter = ComputeMeter(name=name)

    def _timed(self, fn):
        t0 = time.perf_counter()
        a = fn()
        self.meter.add_inference(time.perf_counter() - t0, 1)
        return a


class NoopPolicy(_Base):
    """Hold the initial portfolio and pay no gas."""

    def __init__(self) -> None:
        super().__init__("noop")

    def act(self, ctx: DecisionContext) -> int:
        return self._timed(lambda: 0)


class RandomPolicy(_Base):
    """Uniform over feasible actions -- the exploration floor."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__("random")
        self.seed = int(seed)
        self._rng = np.random.Generator(np.random.PCG64(self.seed))

    def reset(self, episode_seed: Optional[int] = None) -> None:
        self._rng = np.random.Generator(np.random.PCG64(
            self.seed if episode_seed is None else int(episode_seed) ^ self.seed))

    def act(self, ctx: DecisionContext) -> int:
        def pick() -> int:
            idx = np.flatnonzero(ctx.feasible)
            return int(self._rng.choice(idx)) if idx.size else 0
        return self._timed(pick)


class BuyAndHoldPolicy(_Base):
    """Convert cash into the risk asset once, then hold.

    The classic passive benchmark: any active system must beat simply being long.
    """

    def __init__(self, action_names: Sequence[str], token: str = "ETHX") -> None:
        super().__init__("buy_and_hold")
        self.buy = 0
        for i, n in enumerate(action_names):
            if n.startswith(f"SWAP[USD->{token}]@0.5"):
                self.buy = i
                break
        self._done = False

    def reset(self, episode_seed: Optional[int] = None) -> None:
        self._done = False

    def act(self, ctx: DecisionContext) -> int:
        def pick() -> int:
            if not self._done and ctx.feasible[self.buy]:
                self._done = True
                return self.buy
            return 0
        return self._timed(pick)


class GreedyMemoryPolicy(_Base):
    """Memory-only control: replay whatever worked in the most similar past state.

    Isolates the contribution of episodic memory with no model and no reasoning.
    """

    def __init__(self, n_actions: int, capacity: int = 4096, explore: float = 0.1,
                 seed: int = 0) -> None:
        from ...memory.episodic import EpisodicMemory
        super().__init__("memory_only")
        self.memory = EpisodicMemory(key_dim=16, capacity=capacity, n_actions=n_actions)
        self.explore = float(explore)
        self.seed = int(seed)
        self._rng = np.random.Generator(np.random.PCG64(seed))

    def reset(self, episode_seed: Optional[int] = None) -> None:
        self.memory.clear()
        self._rng = np.random.Generator(np.random.PCG64(
            self.seed if episode_seed is None else int(episode_seed) ^ self.seed))

    def act(self, ctx: DecisionContext) -> int:
        def pick() -> int:
            raw = ctx.raw_obs if ctx.raw_obs is not None else ctx.obs_hist[-1]
            if len(self.memory) < 16 or self._rng.random() < self.explore:
                idx = np.flatnonzero(ctx.feasible)
                return int(self._rng.choice(idx)) if idx.size else 0
            vals, sup = self.memory.action_values(np.asarray(raw, np.float32)[:16])
            score = np.where(ctx.feasible & (sup > 0.3), vals, -np.inf)
            return int(np.argmax(score)) if np.isfinite(score).any() else 0
        return self._timed(pick)

    def observe_outcome(self, ctx, action, reward, next_obs, events) -> None:
        raw = ctx.raw_obs if ctx.raw_obs is not None else ctx.obs_hist[-1]
        self.memory.add(np.asarray(raw, np.float32)[:16], int(action), float(reward),
                        events, int(ctx.t))
