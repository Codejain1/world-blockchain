"""Planning without a learned world model (ablation F).

The reasoner proposes candidate actions; each candidate is scored by a *static,
analytic* value function computed from the current observation alone -- expected
one-step edge minus gas, plus a solvency term.  There is no learned dynamics
model and no simulator, so this isolates "search over a menu" from "search
through a model of consequences".
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..evaluation.compute import ComputeMeter
from ..models.base import DecisionContext, ModelInfo, Policy

__all__ = ["AnalyticPlannerPolicy"]


class AnalyticPlannerPolicy(Policy):
    family = "baseline"

    def __init__(self, feature_names: Sequence[str], action_names: Sequence[str],
                 reasoner=None, name: str = "planner_no_model",
                 gas_penalty: float = 0.02, n_candidates: int = 16,
                 seed: int = 0) -> None:
        super().__init__(name)
        self.f = {n: i for i, n in enumerate(feature_names)}
        self.action_names = list(action_names)
        self.reasoner = reasoner
        self.gas_penalty = float(gas_penalty)
        self.n_candidates = int(n_candidates)
        self.meter = ComputeMeter(name=name)
        self.seed = int(seed)
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self.tokens = sorted({n.split("_", 1)[1] for n in feature_names
                              if n.startswith("basis_")})

    def reset(self, episode_seed: Optional[int] = None) -> None:
        self._rng = np.random.Generator(np.random.PCG64(
            self.seed if episode_seed is None else int(episode_seed) ^ self.seed))

    def _get(self, obs: np.ndarray, name: str, default: float = 0.0) -> float:
        i = self.f.get(name)
        return float(obs[i]) if i is not None else float(default)

    def _value(self, obs: np.ndarray, a: int) -> float:
        """Analytic one-step value of an action, from observable quantities only."""
        name = self.action_names[a]
        if name == "NOOP":
            return 0.0
        v = -self.gas_penalty * (1.0 + self._get(obs, "gas_vs_baseline"))
        inv_h = self._get(obs, "inv_health")
        if name.startswith(("REPAY", "SUPPLY")) and inv_h > 0.42:
            v += 4.0 * (inv_h - 0.42)
        if name.startswith("BORROW") and inv_h > 0.40:
            v -= 3.0 * (inv_h - 0.40)
        if name.startswith("SWAP["):
            inner = name[len("SWAP["):name.index("]")]
            src, dst = inner.split("->")
            frac = float(name.rsplit("@", 1)[1])
            if dst == "USD" and src in self.tokens:
                v += frac * 2.0 * self._get(obs, f"basis_{src}")
            elif src == "USD" and dst in self.tokens:
                v += frac * 2.0 * (-self._get(obs, f"basis_{dst}"))
        if name.startswith("ADD_LIQ"):
            rvol = max((self._get(obs, f"rvol_{t}") for t in self.tokens), default=0.0)
            v += 0.05 - 8.0 * rvol
        if name.startswith("REM_LIQ"):
            rvol = max((self._get(obs, f"rvol_{t}") for t in self.tokens), default=0.0)
            v += 8.0 * rvol - 0.05
        if name.startswith("LIQUIDATE"):
            v += 2.0 * self._get(obs, "frac_unhealthy")
        return float(v)

    def act(self, ctx: DecisionContext) -> int:
        t0 = time.perf_counter()
        raw = ctx.raw_obs if ctx.raw_obs is not None else ctx.obs_hist[-1]
        cands: List[int] = [0]
        if self.reasoner is not None:
            try:
                cands.append(int(self.reasoner.act(ctx)))
            except Exception:
                pass
        legal = np.flatnonzero(ctx.feasible)
        if legal.size:
            extra = self._rng.choice(legal, size=min(self.n_candidates, legal.size),
                                     replace=False)
            cands.extend(int(x) for x in extra)
        best, best_v = 0, -np.inf
        for a in dict.fromkeys(cands):
            if not ctx.feasible[a]:
                continue
            v = self._value(raw, a)
            if v > best_v:
                best, best_v = a, v
        self.meter.add_inference(time.perf_counter() - t0, 1)
        return int(best)

    def info(self) -> ModelInfo:
        return ModelInfo(name=self.name, family="baseline", n_params=0,
                         notes="search over an analytic value function; no learned dynamics")
