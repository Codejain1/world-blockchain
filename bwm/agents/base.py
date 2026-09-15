"""Base types for scripted agents that populate the world.

Scripted agents are *part of the simulator*, not systems under test.  They are
what makes the world a multi-agent economy with endogenous prices, liquidity and
liquidation cascades.  Because they are simulator components they may read raw
state (including the exogenous ``fundamental`` price, which stands in for an
off-chain reference market); the systems under evaluation may not.

Every policy must be a pure function of ``(state, block, agent_id, base_seed)``
so that forked worlds replay identically -- see the determinism contract in
:mod:`bwm.environment.world`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..environment.types import Action, ActionType

__all__ = ["AgentSpec", "ScriptedPolicy"]


@dataclass
class AgentSpec:
    """Per-agent parameters drawn once at genesis."""

    agent_id: int
    archetype: str
    activity: float = 0.15        # probability of submitting a tx in a block
    risk_aversion: float = 0.5    # 0 = risk seeking, 1 = risk averse
    aggression: float = 0.3       # typical trade size as a fraction of balance
    horizon: int = 32             # memory length for momentum-style rules
    wealth_multiplier: float = 1.0
    tip_multiple: float = 0.1     # priority fee as a multiple of the base fee
    extra: Dict[str, float] = field(default_factory=dict)


class ScriptedPolicy:
    """Interface for a scripted agent policy."""

    archetype: str = "base"

    def __init__(self, spec: AgentSpec) -> None:
        self.spec = spec

    # ``rng`` is created by the population from a counter-based stream.
    def act(self, world, rng: np.random.Generator) -> Optional[Action]:
        raise NotImplementedError

    # -- helpers ---------------------------------------------------------
    def _tip(self, world) -> float:
        return self.spec.tip_multiple * float(world.state.gas_base_fee)

    def noop(self) -> Optional[Action]:
        return None

    def make(self, world, atype: ActionType, target: int = 0, side: int = 0,
             frac: float = 0.0, tip_scale: float = 1.0) -> Action:
        return Action(agent=self.spec.agent_id, atype=atype, target=int(target),
                      side=int(side), frac=float(np.clip(frac, 0.0, 1.0)),
                      tip=self._tip(world) * float(tip_scale))
