"""Assembly of a heterogeneous agent population."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..environment.config import EnvConfig
from ..environment.types import Action
from .archetypes import ARCHETYPES, make_policy
from .base import AgentSpec, ScriptedPolicy

__all__ = ["AgentPopulation", "ARCHETYPE_ORDER"]

#: Stable ordering so archetype -> cluster index is reproducible across worlds.
ARCHETYPE_ORDER: List[str] = [
    "retail", "whale", "arbitrageur", "market_maker", "liquidity_provider",
    "borrower", "keeper", "governance", "random", "adversarial", "coordinated",
]
_ARCH_INDEX = {a: i for i, a in enumerate(ARCHETYPE_ORDER)}

#: Per-archetype defaults for (activity, risk_aversion, aggression, wealth x).
_DEFAULTS: Dict[str, Tuple[float, float, float, float]] = {
    "retail": (0.16, 0.55, 0.14, 1.0),
    "whale": (0.10, 0.45, 0.22, 1.0),           # wealth multiplier applied from cfg
    "arbitrageur": (0.90, 0.20, 0.50, 12.0),
    "market_maker": (0.40, 0.45, 0.30, 6.0),
    "liquidity_provider": (0.25, 0.60, 0.25, 3.0),
    "borrower": (0.35, 0.30, 0.35, 2.0),
    "keeper": (0.98, 0.15, 0.40, 6.0),
    "governance": (0.20, 0.50, 0.20, 2.0),
    "random": (0.35, 0.50, 0.25, 1.0),
    "adversarial": (0.35, 0.10, 0.35, 5.0),
    "coordinated": (0.50, 0.30, 0.20, 2.0),
}


class AgentPopulation:
    """Builds and drives the scripted agents of a world.

    Implements :class:`bwm.environment.world.PopulationProtocol`.
    """

    def __init__(self, cfg: EnvConfig, spec_seed: Optional[int] = None,
                 n_blocs: int = 2) -> None:
        self.cfg = cfg
        self.spec_seed = int(cfg.seed if spec_seed is None else spec_seed)
        self.n_blocs = int(max(n_blocs, 1))
        self.specs: List[AgentSpec] = self._build_specs()
        self.policies: List[ScriptedPolicy] = [make_policy(s) for s in self.specs]
        self.agent_types: np.ndarray = np.asarray(
            [_ARCH_INDEX.get(s.archetype, 0) for s in self.specs], dtype=np.int64)

    # ------------------------------------------------------------------
    def _build_specs(self) -> List[AgentSpec]:
        cfg = self.cfg
        rng = np.random.Generator(np.random.PCG64(self.spec_seed ^ 0xA5A5))
        counts = dict(cfg.population)
        for name in counts:
            if name not in ARCHETYPES:
                raise KeyError(f"Unknown archetype in population: {name!r}")
        total = sum(counts.values())
        if total != cfg.n_agents:
            # Scale proportionally, then fix the remainder with the largest group.
            scaled = {k: int(round(v * cfg.n_agents / max(total, 1))) for k, v in counts.items()}
            diff = cfg.n_agents - sum(scaled.values())
            if scaled:
                big = max(scaled, key=lambda k: scaled[k])
                scaled[big] = max(scaled[big] + diff, 0)
            counts = scaled
            drift = cfg.n_agents - sum(counts.values())
            if drift != 0 and counts:
                any_key = sorted(counts)[0]
                counts[any_key] = max(counts[any_key] + drift, 0)

        specs: List[AgentSpec] = []
        aid = 0
        bloc = 0
        for name in sorted(counts):
            for _ in range(counts[name]):
                if aid >= cfg.n_agents:
                    break
                act, risk, aggr, wmul = _DEFAULTS.get(name, (0.2, 0.5, 0.2, 1.0))
                if name == "whale":
                    wmul = cfg.whale_wealth_multiplier
                extra: Dict[str, float] = {}
                if name == "coordinated":
                    extra = {"bloc": float(bloc % self.n_blocs),
                             "bloc_activity": 0.25 if not cfg.coordinated_attack else 0.55}
                    bloc += 1
                specs.append(AgentSpec(
                    agent_id=aid,
                    archetype=name,
                    activity=float(np.clip(act * rng.uniform(0.8, 1.2), 0.01, 1.0)),
                    risk_aversion=float(np.clip(risk + rng.normal(0, 0.15), 0.0, 1.0)),
                    aggression=float(np.clip(aggr * rng.uniform(0.7, 1.3), 0.01, 0.9)),
                    horizon=int(rng.integers(8, 64)),
                    wealth_multiplier=float(wmul),
                    tip_multiple=float(np.clip(rng.uniform(0.05, 0.3), 0.01, 1.0)),
                    extra=extra,
                ))
                aid += 1
        while aid < cfg.n_agents:      # pad with retail if rounding left a gap
            specs.append(AgentSpec(agent_id=aid, archetype="retail", activity=0.16,
                                   risk_aversion=0.55, aggression=0.14))
            aid += 1
        return specs

    # ------------------------------------------------------------------
    def reset(self, world) -> None:
        """Scale genesis balances by each archetype's wealth multiplier."""
        st = world.state
        mult = np.asarray([s.wealth_multiplier for s in self.specs], dtype=np.float64)
        st.balances *= mult[:, None]
        st.net_worth_prev = world.net_worth()
        world._init_net_worth = st.net_worth_prev.copy()

    def act(self, world, exclude: Sequence[int] = ()) -> List[Action]:
        """Collect this block's scripted actions.

        The RNG for agent ``i`` at block ``t`` is ``rng("agent", t, i)``, so an
        agent's decision noise is independent of every other agent and of the
        order of evaluation -- a requirement for fork-consistent counterfactuals.
        """
        t = int(world.state.t)
        skip = set(int(x) for x in exclude)
        nw = world.net_worth()
        out: List[Action] = []
        for i, pol in enumerate(self.policies):
            if i in skip:
                continue
            rng = world.seeds.rng("agent", t, i)
            act = pol.act(world, rng)
            if act is None:
                continue
            if not self._can_afford(world, act, nw[i]):
                continue          # priced out of the block by the base fee
            out.append(act)
        return out

    def _can_afford(self, world, act: Action, net_worth: float) -> bool:
        """Demand side of the fee market.

        An account submits only if the gas cost is small relative to the value it
        is moving.  Without this, nothing stops the EIP-1559 base fee from
        rising without bound, because agents would keep bidding at any price.
        """
        gas = act.gas()
        if gas == 0:
            return True
        fee = gas * (float(world.state.gas_base_fee) + max(act.tip, 0.0))
        if float(world.state.balances[act.agent, 0]) < fee:
            return False      # cannot pay for the block space it is bidding for
        trade_value = max(float(act.frac), 0.05) * max(float(net_worth), 0.0)
        budget = float(getattr(world.cfg, "gas_budget_frac", 0.005)) * trade_value
        return fee <= max(budget, 0.0)

    # -- introspection ---------------------------------------------------
    def archetype_of(self, agent: int) -> str:
        return self.specs[int(agent)].archetype

    def agents_of(self, archetype: str) -> List[int]:
        return [s.agent_id for s in self.specs if s.archetype == archetype]

    def summary(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for s in self.specs:
            out[s.archetype] = out.get(s.archetype, 0) + 1
        return out
