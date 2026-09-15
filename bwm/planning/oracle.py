"""Planning with the *true* simulator instead of a learned model.

This is ablation E and a deliberate upper bound: it forks the real world, rolls
candidate action sequences through the actual transition function, and picks the
best.  It has perfect dynamics, so it bounds how much any learned model could
gain from better prediction alone.

It is also enormously more expensive, and its cost is recorded honestly in
``ComputeMeter.oracle_sim_steps``.  Where it wins, the right conclusion is "the
learned model is not yet accurate enough", not "planning does not work".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..environment.types import EventType
from ..evaluation.compute import ComputeMeter
from .planner import RISK_EVENTS, PlannerConfig

__all__ = ["OracleSimulatorPlanner"]


@dataclass
class OracleConfig(PlannerConfig):
    horizon: int = 4
    n_candidates: int = 8
    n_iters: int = 1
    method: str = "shooting"


class OracleSimulatorPlanner:
    """Random-shooting MPC over forks of the real environment."""

    def __init__(self, action_space, cfg: Optional[OracleConfig] = None,
                 meter: Optional[ComputeMeter] = None) -> None:
        self.action_space = action_space
        self.cfg = cfg or OracleConfig()
        self.meter = meter or ComputeMeter(name="oracle_planner")
        self._rng = np.random.Generator(np.random.PCG64(self.cfg.seed))

    def reset(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.Generator(np.random.PCG64(
            self.cfg.seed if seed is None else int(seed)))

    def plan(self, world, agent: int, feasible: Optional[np.ndarray] = None
             ) -> Tuple[int, Dict[str, Any]]:
        cfg = self.cfg
        n_actions = self.action_space.n
        mask = (np.ones(n_actions, dtype=bool) if feasible is None
                else np.asarray(feasible, dtype=bool).copy())
        if not mask.any():
            mask[0] = True
        allowed = np.flatnonzero(mask)
        firsts = self._rng.choice(
            allowed, size=min(cfg.n_candidates, len(allowed)), replace=False)

        best_a, best_score = int(firsts[0]), -np.inf
        for a0 in firsts:
            branch = world.fork()
            score, disc = 0.0, 1.0
            seq = [int(a0)] + [int(x) for x in
                               self._rng.integers(0, n_actions, cfg.horizon - 1)]
            for l, a in enumerate(seq):
                act = self.action_space.decode(a, agent,
                                               base_fee=float(branch.state.gas_base_fee))
                res = branch.step({agent: act})
                self.meter.oracle_sim_steps += 1
                r = float(res.rewards[agent])
                pen = (cfg.risk_lambda * float(res.events[list(RISK_EVENTS)].mean())
                       if cfg.risk_lambda > 0 else 0.0)
                score += disc * (r - pen)
                disc *= cfg.gamma
            if score > best_score:
                best_score, best_a = score, int(a0)
        return best_a, {"score": best_score, "method": "oracle_sim"}
