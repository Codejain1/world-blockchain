"""Episode execution and trajectory logging.

A :class:`Trajectory` is the unit of data in this lab.  It records, for one
focal agent in one world:

``obs[t]``      observation *before* the action at step ``t``  (length T+1)
``actions[t]``  discrete action index executed at step ``t``   (length T)
``rewards[t]``  focal agent's change in net worth at step ``t``
``events[t]``   world events produced by step ``t``
``regime[t]``   the hidden regime -- **analysis only**, never a model input

Storing ``T+1`` observations makes next-state prediction, multi-step rollout and
counterfactual comparison all read off the same array without index gymnastics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..agents.population import AgentPopulation
from ..utils.seeding import make_rng
from .action_space import DiscreteActionSpace
from .config import EnvConfig
from .observation import ObservationBuilder
from .types import Action, ActionType, N_EVENTS
from .world import BlockchainWorld

__all__ = ["Trajectory", "BehaviorPolicy", "EpisodeRunner", "EXTRA_NAMES"]

#: Auxiliary scalar targets logged alongside observations (prediction targets
#: for the "price/market-state" task).  Always derived from public state.
EXTRA_NAMES: List[str] = ["tvl", "total_borrow_usd", "frac_unhealthy",
                          "gas_base_fee", "mean_net_worth", "amm_basis_mean"]


@dataclass
class Trajectory:
    obs: np.ndarray            # (T+1, D) float32
    actions: np.ndarray        # (T,)    int64
    rewards: np.ndarray        # (T,)    float32
    events: np.ndarray         # (T, E)  float32
    node_feat: np.ndarray      # (T+1, N, F) float32
    regime: np.ndarray         # (T+1,)  int8   -- analysis only
    prices: np.ndarray         # (T+1, K) float32 oracle prices
    extras: np.ndarray         # (T+1, len(EXTRA_NAMES)) float32
    net_worth: np.ndarray      # (T+1,)  float32 focal agent
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.actions.shape[0])

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        np.savez_compressed(
            path, obs=self.obs, actions=self.actions, rewards=self.rewards,
            events=self.events, node_feat=self.node_feat, regime=self.regime,
            prices=self.prices, extras=self.extras, net_worth=self.net_worth,
            meta=np.array(repr(self.meta), dtype=object))

    @staticmethod
    def load(path: str) -> "Trajectory":
        z = np.load(path, allow_pickle=True)
        meta = eval(str(z["meta"].item())) if "meta" in z else {}   # noqa: S307
        return Trajectory(
            obs=z["obs"], actions=z["actions"], rewards=z["rewards"],
            events=z["events"], node_feat=z["node_feat"], regime=z["regime"],
            prices=z["prices"], extras=z["extras"], net_worth=z["net_worth"], meta=meta)


class BehaviorPolicy:
    """Data-collection policy for the focal agent.

    Mixes the focal agent's own scripted archetype (projected onto the discrete
    action menu) with uniform exploration.  Exploration is essential: an
    action-conditioned world model trained only on one policy's actions cannot
    answer counterfactual questions about actions that policy never takes.
    """

    def __init__(self, action_space: DiscreteActionSpace, epsilon: float = 0.35,
                 seed: int = 0, use_archetype: bool = True) -> None:
        self.action_space = action_space
        self.epsilon = float(epsilon)
        self.seed = int(seed)
        self.use_archetype = bool(use_archetype)

    def __call__(self, world: BlockchainWorld, agent: int,
                 population: Optional[AgentPopulation] = None) -> int:
        t = int(world.state.t)
        rng = make_rng(self.seed, "behavior", t, agent)
        mask = self.action_space.feasible_mask(world, agent)
        if rng.random() < self.epsilon or not self.use_archetype or population is None:
            idx = np.flatnonzero(mask)
            return int(rng.choice(idx)) if idx.size else 0
        pol = population.policies[agent]
        act = pol.act(world, make_rng(self.seed, "behavior_pol", t, agent))
        if act is None:
            return 0
        i = self.action_space.project(act)
        return int(i) if mask[i] else 0


class EpisodeRunner:
    """Runs one episode of a world with one focal agent under evaluation."""

    def __init__(self, cfg: EnvConfig, focal_agent: Optional[int] = None,
                 n_clusters: int = 8, action_space: Optional[DiscreteActionSpace] = None,
                 obs_builder: Optional[ObservationBuilder] = None) -> None:
        self.cfg = cfg
        self.population = AgentPopulation(cfg)
        self.obs_builder = obs_builder or ObservationBuilder(cfg, n_clusters=n_clusters)
        self.action_space = action_space or DiscreteActionSpace(cfg)
        self.focal_agent = (int(focal_agent) if focal_agent is not None
                            else self._default_focal())
        self.world = BlockchainWorld(cfg, self.population)

    def _default_focal(self) -> int:
        """Pick a retail account as the default focal agent.

        Retail is the most common archetype and holds no special powers, so
        evaluating on it avoids handing any system an unusual balance sheet.
        """
        retail = self.population.agents_of("retail")
        return int(retail[0]) if retail else 0

    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.cfg.seed = int(seed)
        self.world = BlockchainWorld(self.cfg, self.population)
        return self.observe()

    def observe(self) -> np.ndarray:
        return self.obs_builder.observe(self.world, self.focal_agent)

    def graph(self) -> np.ndarray:
        return self.obs_builder.graph(self.world, self.focal_agent,
                                      self.population.agent_types).node_feat

    def extras(self) -> np.ndarray:
        w, st, cfg = self.world, self.world.state, self.cfg
        bor = float((st.total_borrow * st.borrow_index * st.oracle_price).sum())
        hf = w.health_factor()
        nw = w.net_worth()
        basis = float(np.mean([abs(w.amm_usd_price(k) / max(st.oracle_price[k], 1e-12) - 1.0)
                               for k in range(1, cfg.n_tokens)]))
        return np.asarray([w.total_tvl(), bor, float(np.mean(hf < 1.0)),
                           float(st.gas_base_fee), float(np.mean(nw)), basis],
                          dtype=np.float32)

    def step_with_index(self, action_index: int):
        """Execute a discrete action index for the focal agent."""
        act = self.action_space.decode(int(action_index), self.focal_agent,
                                       base_fee=float(self.world.state.gas_base_fee))
        return self.world.step({self.focal_agent: act})

    # ------------------------------------------------------------------
    def rollout(self, policy, steps: Optional[int] = None,
                seed: Optional[int] = None, record_graph: bool = True) -> Trajectory:
        """Run an episode, logging a :class:`Trajectory`.

        ``policy`` is any callable ``(world, agent, population) -> action index``.
        """
        T = int(steps or self.cfg.episode_length)
        self.reset(seed)
        D = self.obs_builder.obs_dim
        N, F = self.obs_builder.n_nodes, self.obs_builder.node_feat_dim
        K = self.cfg.n_tokens

        obs = np.zeros((T + 1, D), dtype=np.float32)
        node = np.zeros((T + 1, N, F), dtype=np.float32) if record_graph \
            else np.zeros((T + 1, 1, 1), dtype=np.float32)
        acts = np.zeros(T, dtype=np.int64)
        rews = np.zeros(T, dtype=np.float32)
        evts = np.zeros((T, N_EVENTS), dtype=np.float32)
        regs = np.zeros(T + 1, dtype=np.int8)
        prices = np.zeros((T + 1, K), dtype=np.float32)
        extras = np.zeros((T + 1, len(EXTRA_NAMES)), dtype=np.float32)
        nws = np.zeros(T + 1, dtype=np.float32)

        for t in range(T):
            obs[t] = self.observe()
            if record_graph:
                node[t] = self.graph()
            regs[t] = self.world.state.regime
            prices[t] = self.world.state.oracle_price
            extras[t] = self.extras()
            nws[t] = self.world.net_worth()[self.focal_agent]
            a = int(policy(self.world, self.focal_agent, self.population))
            acts[t] = a
            res = self.step_with_index(a)
            rews[t] = float(res.rewards[self.focal_agent])
            evts[t] = res.events
        obs[T] = self.observe()
        if record_graph:
            node[T] = self.graph()
        regs[T] = self.world.state.regime
        prices[T] = self.world.state.oracle_price
        extras[T] = self.extras()
        nws[T] = self.world.net_worth()[self.focal_agent]

        meta = {
            "scenario": self.cfg.name,
            "seed": int(self.cfg.seed),
            "focal_agent": int(self.focal_agent),
            "focal_archetype": self.population.archetype_of(self.focal_agent),
            "n_agents": int(self.cfg.n_agents),
            "obs_dim": int(D),
            "n_actions": int(self.action_space.n),
            "population": self.population.summary(),
        }
        return Trajectory(obs=obs, actions=acts, rewards=rews, events=evts,
                          node_feat=node, regime=regs, prices=prices,
                          extras=extras, net_worth=nws, meta=meta)
