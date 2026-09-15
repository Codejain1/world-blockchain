"""Live control evaluation: action selection, planning, wealth and risk.

Every policy is run on the *same* episodes (same world seeds, same focal agent,
same action space, same observation window) and is charged for exactly one real
environment step per decision.  Differences in outcome therefore come from the
decisions, not from extra interaction.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..data.dataset import Normalizer
from ..environment.config import EnvConfig
from ..environment.observation import ObservationBuilder
from ..environment.runner import EpisodeRunner
from ..environment.types import EVENT_NAMES
from ..models.base import DecisionContext, Policy
from .compute import ComputeMeter, Timer
from .metrics import max_drawdown, sharpe, sortino

__all__ = ["ControlResult", "run_policy_episode", "summarize_control", "regret_table"]


@dataclass
class ControlResult:
    policy: str
    scenario: str
    seed: int
    rewards: np.ndarray
    net_worth: np.ndarray
    events: np.ndarray
    actions: np.ndarray
    seconds: float
    meter: Dict[str, Any] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        nw = self.net_worth
        init = float(nw[0]) if nw.size else 1.0
        final = float(nw[-1]) if nw.size else init
        return {
            "policy": self.policy, "scenario": self.scenario, "seed": self.seed,
            "final_wealth_ratio": final / max(abs(init), 1e-9),
            "total_return": (final - init) / max(abs(init), 1e-9),
            "mean_step_reward": float(self.rewards.mean()) if self.rewards.size else 0.0,
            "sharpe": sharpe(self.rewards / max(abs(init), 1e-9)),
            "sortino": sortino(self.rewards / max(abs(init), 1e-9)),
            "max_drawdown": max_drawdown(nw),
            "n_steps": int(self.rewards.size),
            "seconds": self.seconds,
            "action_entropy": _entropy(self.actions),
            "frac_noop": float(np.mean(self.actions == 0)) if self.actions.size else 0.0,
        }


def _entropy(actions: np.ndarray) -> float:
    if actions.size == 0:
        return 0.0
    _, c = np.unique(actions, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


def run_policy_episode(policy: Policy, cfg: EnvConfig, normalizer: Normalizer,
                       history: int = 8, steps: Optional[int] = None,
                       seed: Optional[int] = None, focal_agent: Optional[int] = None,
                       allow_simulator: bool = False, needs_graph: bool = False,
                       runner: Optional[EpisodeRunner] = None) -> ControlResult:
    """Run one policy for one episode and record wealth, risk and compute."""
    runner = runner or EpisodeRunner(cfg, focal_agent=focal_agent)
    T = int(steps or cfg.episode_length)
    runner.reset(seed if seed is not None else cfg.seed)
    policy.reset(episode_seed=seed if seed is not None else cfg.seed)

    agent = runner.focal_agent
    world = runner.world
    asp = runner.action_space

    raw_win: deque = deque(maxlen=history)
    act_win: deque = deque(maxlen=history)
    node_win: deque = deque(maxlen=history)
    o0 = runner.observe()
    for _ in range(history):
        raw_win.append(o0)
        act_win.append(0)
        node_win.append(runner.graph() if needs_graph else np.zeros((1, 1), np.float32))

    rewards = np.zeros(T, np.float32)
    nws = np.zeros(T + 1, np.float32)
    events = np.zeros((T, len(EVENT_NAMES)), np.float32)
    actions = np.zeros(T, np.int64)
    nws[0] = float(world.net_worth()[agent])

    with Timer() as timer:
        for t in range(T):
            raw = np.asarray(raw_win[-1], np.float32)
            raw_hist = np.stack(list(raw_win))
            obs_hist = normalizer.obs(raw_hist)
            ctx = DecisionContext(
                obs_hist=obs_hist, act_hist=np.asarray(list(act_win), np.int64),
                t=t, agent=agent, action_space=asp,
                feasible=asp.feasible_mask(world, agent),
                world=world if allow_simulator else None,
                allow_simulator=allow_simulator, raw_obs=raw,
                info={"node_hist": (normalizer.node(np.stack(list(node_win)))
                                    if needs_graph else None),
                      "raw_obs_hist": raw_hist})
            a = int(policy.act(ctx))
            if not ctx.feasible[a]:
                a = 0
            res = runner.step_with_index(a)
            nxt = runner.observe()
            rewards[t] = float(res.rewards[agent])
            events[t] = res.events
            actions[t] = a
            nws[t + 1] = float(world.net_worth()[agent])
            policy.observe_outcome(ctx, a, rewards[t], nxt, res.events)
            raw_win.append(nxt)
            act_win.append(a)
            if needs_graph:
                node_win.append(runner.graph())

    meter = getattr(policy, "meter", None)
    md = meter.to_dict() if isinstance(meter, ComputeMeter) else {}
    if isinstance(meter, ComputeMeter):
        meter.env_steps += T
        md = meter.to_dict()
    return ControlResult(policy=policy.name, scenario=cfg.name,
                         seed=int(seed if seed is not None else cfg.seed),
                         rewards=rewards, net_worth=nws, events=events,
                         actions=actions, seconds=timer.seconds, meter=md,
                         info={"focal_archetype": runner.population.archetype_of(agent),
                               **(policy.info().extra or {})})


def summarize_control(results: Sequence[ControlResult]) -> Dict[str, Any]:
    """Aggregate episode summaries for one policy on one scenario."""
    if not results:
        return {}
    rows = [r.summary() for r in results]
    keys = ["final_wealth_ratio", "total_return", "mean_step_reward", "sharpe",
            "sortino", "max_drawdown", "action_entropy", "frac_noop", "seconds"]
    out: Dict[str, Any] = {"policy": rows[0]["policy"], "scenario": rows[0]["scenario"],
                           "n_episodes": len(rows)}
    for k in keys:
        v = np.asarray([r[k] for r in rows], float)
        out[k] = float(np.mean(v))
        out[f"{k}_std"] = float(np.std(v))
        out[f"{k}_se"] = float(np.std(v) / max(np.sqrt(len(v)), 1.0))
    ev = np.concatenate([r.events for r in results], axis=0)
    out["event_rate"] = {n: float(ev[:, i].mean()) for i, n in enumerate(EVENT_NAMES)}
    out["total_env_steps"] = int(sum(r.rewards.size for r in results))
    return out


def regret_table(per_policy: Dict[str, List[ControlResult]]) -> Dict[str, float]:
    """Mean per-episode regret against the best policy on that same episode.

    Computed episode-wise so that regret is not contaminated by which episodes a
    policy happened to be lucky on.
    """
    by_ep: Dict[Tuple[str, int], Dict[str, float]] = {}
    for name, results in per_policy.items():
        for r in results:
            by_ep.setdefault((r.scenario, r.seed), {})[name] = r.summary()["total_return"]
    regrets: Dict[str, List[float]] = {k: [] for k in per_policy}
    for _, d in by_ep.items():
        best = max(d.values())
        for name, v in d.items():
            regrets[name].append(best - v)
    return {k: float(np.mean(v)) if v else float("nan") for k, v in regrets.items()}
