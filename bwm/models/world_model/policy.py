"""System 2: acting by planning inside a learned world model."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from ...data.dataset import Normalizer
from ...evaluation.compute import ComputeMeter
from ...memory.episodic import ConsequenceTracker, EpisodicMemory
from ...planning.planner import LatentPlanner, PlannerConfig
from ..base import DecisionContext, ModelInfo, Policy
from .base import LatentWorldModel

__all__ = ["WorldModelPolicy"]


class WorldModelPolicy(Policy):
    """Encode -> imagine -> score -> act, with the simulator used only to execute.

    ``use_planning=False`` gives ablation D (a world model used as a one-step
    greedy value estimate, with no lookahead), which isolates how much of the
    model's value comes from *planning* rather than from *prediction*.
    """

    family = "world_model"

    def __init__(self, model: LatentWorldModel, action_space, normalizer: Normalizer,
                 planner_cfg: Optional[PlannerConfig] = None, name: str = "world_model",
                 use_planning: bool = True, use_memory: bool = False,
                 memory_capacity: int = 4096, memory_weight: float = 0.3,
                 needs_graph: bool = False) -> None:
        super().__init__(name)
        self.model = model
        self.action_space = action_space
        self.normalizer = normalizer
        self.use_planning = bool(use_planning)
        self.needs_graph = bool(needs_graph)
        self.planner = LatentPlanner(model, action_space.n, planner_cfg or PlannerConfig())
        self.memory = (EpisodicMemory(key_dim=16, capacity=memory_capacity,
                                      n_actions=action_space.n) if use_memory else None)
        self.memory_weight = float(memory_weight)
        self.tracker = ConsequenceTracker(action_space.n)
        self.meter = ComputeMeter(name=name, n_params=model.n_params())
        self._pending: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    def reset(self, episode_seed: Optional[int] = None) -> None:
        self.planner.reset(episode_seed)
        self.tracker.reset()
        if self.memory is not None:
            self.memory.clear()
        self._pending = None

    @staticmethod
    def _key(obs: np.ndarray) -> np.ndarray:
        """Compact state descriptor for memory lookup (first 16 features)."""
        return np.asarray(obs, dtype=np.float32)[:16]

    def _greedy_one_step(self, ctx: DecisionContext) -> int:
        """Ablation D: rank feasible actions by predicted immediate reward."""
        feas = np.flatnonzero(ctx.feasible)
        if feas.size == 0:
            return 0
        node = ctx.info.get("node_hist") if self.needs_graph else None
        state = self.model.encode_numpy(ctx.obs_hist[None], ctx.act_hist[None],
                                        None if node is None else node[None])
        st = LatentWorldModel.expand_state(state, len(feas))
        a = torch.as_tensor(feas.reshape(-1, 1), dtype=torch.long,
                            device=self.model.cfg.device)
        out = self.model.imagine_sequences(st, a)
        return int(feas[int(torch.argmax(out["reward"][:, 0]).item())])

    def act(self, ctx: DecisionContext) -> int:
        t0 = time.perf_counter()
        node = ctx.info.get("node_hist") if self.needs_graph else None
        if self.use_planning:
            a, _ = self.planner.plan(ctx.obs_hist, ctx.act_hist, ctx.feasible, node)
        else:
            a = self._greedy_one_step(ctx)

        if self.memory is not None and len(self.memory) > 32:
            # Blend the planner's choice with recalled outcomes: if memory has
            # strong, consistent evidence that a *feasible* action did well in
            # similar states, prefer it.
            vals, sup = self.memory.action_values(self._key(ctx.raw_obs
                                                            if ctx.raw_obs is not None
                                                            else ctx.obs_hist[-1]))
            vals = np.where(ctx.feasible, vals, -np.inf)
            sup = np.where(ctx.feasible, sup, 0.0)
            best = int(np.argmax(np.where(sup > 0.5, vals, -np.inf)))
            if sup[best] > 0.5 and vals[best] > self.memory_weight * \
                    max(abs(self.tracker.reward_ema), 1e-6):
                a = best

        # Record the model's own prediction so the realised outcome can score it.
        try:
            state = self.model.encode_numpy(ctx.obs_hist[None], ctx.act_hist[None],
                                            None if node is None else node[None])
            aa = torch.as_tensor([[int(a)]], dtype=torch.long,
                                 device=self.model.cfg.device)
            pred = self.model.imagine_sequences(state, aa)
            self._pending = {"obs": pred["obs"][0, 0].cpu().numpy(), "action": int(a)}
        except Exception:
            self._pending = None
        self.meter.add_inference(time.perf_counter() - t0, 1)
        self.meter.imagined_steps = self.model.meter.imagined_steps
        return int(a)

    def observe_outcome(self, ctx, action, reward, next_obs, events) -> None:
        self.tracker.update_action(int(action), float(reward))
        if self._pending is not None:
            err = float(np.mean(np.abs(self._pending["obs"] - self.normalizer.obs(next_obs))))
            self.tracker.update_model_error(err)
        if self.memory is not None:
            key = self._key(ctx.raw_obs if ctx.raw_obs is not None else ctx.obs_hist[-1])
            self.memory.add(key, int(action), float(reward), events, int(ctx.t))
        self._pending = None

    def n_params(self) -> int:
        return self.model.n_params()

    def info(self) -> ModelInfo:
        return ModelInfo(name=self.name, family="world_model",
                         n_params=self.n_params(),
                         notes=f"planning={self.use_planning} memory={self.memory is not None}",
                         extra={"planner": self.planner.cfg.to_dict(),
                                "imagined_steps": int(self.model.meter.imagined_steps)})
