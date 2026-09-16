"""System 3: unified foundation reasoning + latent world model + memory + planner.

Architecture
------------
Three decision *pathways* share one action interface:

``reason``  the foundation-model (or rule-based) reasoner acting on raw
            observations -- fast, robust to novelty, no simulation
``plan``    model-predictive control inside the learned latent world model --
            strong when the model is accurate, dangerous when it is not
``recall``  episodic memory: replay whatever actually worked in similar states

A contextual bandit (LinUCB) arbitrates between them, and it is **learned from
consequences during the episode**, not hard-coded.  The brief is explicit that
the unified system must not be wired to always trust the world model; the gate's
context deliberately includes the world model's own recent one-step error, so the
system can learn to stop simulating when its model has drifted -- which is
exactly what should happen in the held-out worlds.

Every pathway consumes the same observations and the same number of real
environment steps.  Only imagination and token counts differ, and both are
metered.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ...data.dataset import Normalizer
from ...evaluation.compute import ComputeMeter
from ...memory.episodic import ConsequenceTracker, EpisodicMemory
from ...planning.planner import LatentPlanner, PlannerConfig
from ..base import DecisionContext, ModelInfo, Policy
from ..llm.agent import LLMPolicy, RuleBasedReasoner
from ..world_model.base import LatentWorldModel

__all__ = ["LinUCBGate", "UnifiedPolicy", "PATHWAYS"]

PATHWAYS: Tuple[str, ...] = ("reason", "plan", "recall")


class LinUCBGate:
    """Contextual bandit over decision pathways.

    LinUCB is chosen over a learned neural gate for a boring but important
    reason: with only a few hundred decisions per episode, a neural gate cannot
    be fitted reliably, and an unreliable gate would confound the comparison.
    LinUCB has closed-form updates and an explicit exploration term.
    """

    def __init__(self, n_arms: int, dim: int, alpha: float = 0.6,
                 ridge: float = 1.0) -> None:
        self.n_arms, self.dim, self.alpha = int(n_arms), int(dim), float(alpha)
        # Stored explicitly.  It used to be recovered inside reset() by reading
        # A[0][0, 0], which stops being the ridge after the first update: the
        # context carries a bias term of 1.0, so that cell grows by one per pull.
        # reset() therefore restored a prior that compounded every episode
        # (1 -> 54 -> 107 -> ...), leaving the gate effectively frozen by the end
        # of an evaluation and making results depend on evaluation order.
        self.ridge = float(ridge)
        self.A = np.stack([np.eye(dim) * ridge for _ in range(n_arms)])
        self.b = np.zeros((n_arms, dim))
        self.counts = np.zeros(n_arms, dtype=np.int64)
        self.rewards = np.zeros(n_arms)

    def reset(self) -> None:
        self.A = np.stack([np.eye(self.dim) * self.ridge for _ in range(self.n_arms)])
        self.b[:] = 0.0
        self.counts[:] = 0
        self.rewards[:] = 0.0

    def scores(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            Ainv = np.linalg.inv(self.A[a])
            theta = Ainv @ self.b[a]
            out[a] = float(theta @ x + self.alpha * np.sqrt(max(x @ Ainv @ x, 0.0)))
        return out

    def select(self, x: np.ndarray, allowed: Optional[Sequence[int]] = None) -> int:
        s = self.scores(x)
        if allowed is not None:
            mask = np.full(self.n_arms, -np.inf)
            for a in allowed:
                mask[a] = s[a]
            s = mask
        return int(np.argmax(s))

    def update(self, arm: int, x: np.ndarray, reward: float) -> None:
        a = int(arm)
        self.A[a] += np.outer(x, x)
        self.b[a] += float(reward) * x
        self.counts[a] += 1
        self.rewards[a] += float(reward)

    def fit(self, X: np.ndarray, arms: np.ndarray, rewards: np.ndarray) -> "LinUCBGate":
        """Batch-fit from logged ``(context, arm, reward)`` triples.

        LinUCB's update is additive, so an offline fit is just the online update
        applied to every logged decision at once.  This is what lets the gate
        arrive at evaluation already competent, instead of spending the whole
        episode in exploration -- the same treatment every other learned
        component gets.
        """
        X = np.asarray(X, dtype=np.float64)
        arms = np.asarray(arms, dtype=np.int64)
        rewards = np.asarray(rewards, dtype=np.float64)
        for a in range(self.n_arms):
            m = arms == a
            if not np.any(m):
                continue
            Xa = X[m]
            self.A[a] += Xa.T @ Xa
            self.b[a] += Xa.T @ rewards[m]
            self.counts[a] += int(m.sum())
            self.rewards[a] += float(rewards[m].sum())
        return self

    def state_dict(self) -> Dict[str, Any]:
        return {"A": self.A.tolist(), "b": self.b.tolist(),
                "counts": self.counts.tolist(), "rewards": self.rewards.tolist(),
                "n_arms": self.n_arms, "dim": self.dim, "alpha": self.alpha,
                "ridge": self.ridge}

    def load_state_dict(self, d: Dict[str, Any]) -> "LinUCBGate":
        self.A = np.asarray(d["A"], dtype=np.float64)
        self.b = np.asarray(d["b"], dtype=np.float64)
        self.counts = np.asarray(d["counts"], dtype=np.int64)
        self.rewards = np.asarray(d["rewards"], dtype=np.float64)
        self.ridge = float(d.get("ridge", self.ridge))
        return self

    def usage(self) -> Dict[str, Any]:
        tot = max(int(self.counts.sum()), 1)
        return {"counts": self.counts.tolist(),
                "share": (self.counts / tot).round(4).tolist(),
                "mean_reward": (self.rewards / np.maximum(self.counts, 1)).round(4).tolist()}


class UnifiedPolicy(Policy):
    """The unified intelligence system (System 3)."""

    family = "unified"

    GATE_DIM = 9

    def __init__(self, model: LatentWorldModel, action_space, normalizer: Normalizer,
                 reasoner: LLMPolicy, feature_names: Sequence[str],
                 planner_cfg: Optional[PlannerConfig] = None, name: str = "unified",
                 use_memory: bool = True, use_planning: bool = True,
                 use_reasoner: bool = True, learn_gate: bool = True,
                 gate_alpha: float = 0.4, memory_capacity: int = 4096,
                 needs_graph: bool = False, persist_gate: bool = False,
                 gate_explore: float = 0.0, gate_seed: int = 0) -> None:
        super().__init__(name)
        self.model = model
        self.action_space = action_space
        self.normalizer = normalizer
        self.reasoner = reasoner
        self.needs_graph = bool(needs_graph)
        self.use_memory = bool(use_memory)
        self.use_planning = bool(use_planning)
        self.use_reasoner = bool(use_reasoner)
        self.learn_gate = bool(learn_gate)
        self.planner = LatentPlanner(model, action_space.n, planner_cfg or PlannerConfig())
        self.memory = EpisodicMemory(key_dim=16, capacity=memory_capacity,
                                     n_actions=action_space.n)
        self.tracker = ConsequenceTracker(action_space.n)
        self.gate = LinUCBGate(len(PATHWAYS), self.GATE_DIM, alpha=gate_alpha)
        self.meter = ComputeMeter(name=name, n_params=model.n_params())
        self.fidx = {n: i for i, n in enumerate(feature_names)}
        self._tokens = sorted({n.split("_", 1)[1] for n in feature_names
                               if n.startswith("basis_")})
        self._pending: Optional[Dict[str, Any]] = None
        self.path_log: List[int] = []
        # Keeping the gate across episodes is the difference between ~53 pulls
        # per arm and ~1000: LinUCB over a 9-dimensional context cannot be
        # identified from one episode, which is why the per-episode gate
        # collapses onto whichever arm it happened to try first.
        self.persist_gate = bool(persist_gate)
        #: probability of taking a uniformly random arm -- used only when
        #: *collecting* data to fit the gate offline, never at evaluation.
        self.gate_explore = float(gate_explore)
        self._grng = np.random.Generator(np.random.PCG64(int(gate_seed)))
        #: logged (context, arm, standardised reward) for offline fitting.
        self.gate_log: List[Tuple[np.ndarray, int, float]] = []

    # ------------------------------------------------------------------
    def reset(self, episode_seed: Optional[int] = None) -> None:
        self.planner.reset(episode_seed)
        self.tracker.reset()
        self.memory.clear()
        if not self.persist_gate:
            self.gate.reset()
        self.reasoner.reset(episode_seed)
        self._pending = None
        self.path_log = []

    def _allowed_arms(self) -> List[int]:
        arms = []
        if self.use_reasoner:
            arms.append(0)
        if self.use_planning:
            arms.append(1)
        if self.use_memory:
            arms.append(2)
        return arms or [0]

    def _f(self, obs: np.ndarray, name: str, default: float = 0.0) -> float:
        i = self.fidx.get(name)
        return float(obs[i]) if i is not None else float(default)

    def _context(self, ctx: DecisionContext, raw: np.ndarray) -> np.ndarray:
        """Gate features: how novel is this state, and is my model still right?"""
        rvol = max((self._f(raw, f"rvol_{t}") for t in self._tokens), default=0.0)
        basis = max((abs(self._f(raw, f"basis_{t}")) for t in self._tokens), default=0.0)
        mem_d = (self.memory.mean_distance(self._key(raw))
                 if len(self.memory) > 8 else 10.0)
        x = np.array([
            1.0,
            float(np.tanh(self.tracker.model_error_ema)),      # model trustworthiness
            float(np.tanh(self.tracker.reward_volatility)),
            float(np.clip(self._f(raw, "inv_health"), 0.0, 1.0)),
            float(np.tanh(50.0 * rvol)),
            float(np.tanh(10.0 * basis)),
            float(np.tanh(mem_d / 5.0)),
            float(np.clip(self._f(raw, "time_frac"), 0.0, 1.0)),
            float(np.tanh(self._f(raw, "frac_unhealthy") * 10.0)),
        ], dtype=np.float64)
        return x

    @staticmethod
    def _key(obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)[:16]

    # ------------------------------------------------------------------
    def act(self, ctx: DecisionContext) -> int:
        t0 = time.perf_counter()
        raw = ctx.raw_obs if ctx.raw_obs is not None else ctx.obs_hist[-1]
        x = self._context(ctx, raw)
        arms = self._allowed_arms()
        if self.gate_explore > 0.0 and self._grng.random() < self.gate_explore:
            arm = int(self._grng.choice(arms))          # data-collection only
        elif self.learn_gate:
            arm = self.gate.select(x, arms)
        else:
            arm = arms[0]

        node = ctx.info.get("node_hist") if self.needs_graph else None
        a = 0
        if arm == 1 and self.use_planning:
            a, _ = self.planner.plan(ctx.obs_hist, ctx.act_hist, ctx.feasible, node)
        elif arm == 2 and self.use_memory and len(self.memory) > 16:
            vals, sup = self.memory.action_values(self._key(raw))
            score = np.where(ctx.feasible & (sup > 0.3), vals, -np.inf)
            a = int(np.argmax(score)) if np.isfinite(score).any() else 0
        else:
            arm = 0
            a = self.reasoner.act(ctx)
        if not ctx.feasible[a]:
            a = 0

        # Record the world model's prediction for this action regardless of the
        # pathway taken: the model must be scored even on steps it did not drive,
        # otherwise the gate can never learn that the model has become unreliable.
        try:
            state = self.model.encode_numpy(ctx.obs_hist[None], ctx.act_hist[None],
                                            None if node is None else node[None])
            aa = torch.as_tensor([[int(a)]], dtype=torch.long,
                                 device=self.model.cfg.device)
            pred = self.model.imagine_sequences(state, aa)
            pred_obs = pred["obs"][0, 0].cpu().numpy()
        except Exception:
            pred_obs = None
        self._pending = {"arm": int(arm), "x": x, "pred_obs": pred_obs}
        self.path_log.append(int(arm))
        self.meter.add_inference(time.perf_counter() - t0, 1)
        self.meter.imagined_steps = self.model.meter.imagined_steps
        self.meter.llm_calls = self.reasoner.meter.llm_calls
        self.meter.llm_prompt_tokens = self.reasoner.meter.llm_prompt_tokens
        self.meter.llm_completion_tokens = self.reasoner.meter.llm_completion_tokens
        return int(a)

    def observe_outcome(self, ctx, action, reward, next_obs, events) -> None:
        self.tracker.update_action(int(action), float(reward))
        # The gate learns from a *self-standardised* reward so the signal is
        # comparable to its exploration bonus (see ConsequenceTracker.standardize).
        r_gate = self.tracker.standardize(float(reward))
        self.reasoner.observe_outcome(ctx, action, reward, next_obs, events)
        if self._pending is not None:
            if self._pending["pred_obs"] is not None:
                err = float(np.mean(np.abs(self._pending["pred_obs"]
                                           - self.normalizer.obs(next_obs))))
                self.tracker.update_model_error(err)
            self.gate_log.append((self._pending["x"], int(self._pending["arm"]),
                                  float(r_gate)))
            if self.learn_gate:
                self.gate.update(self._pending["arm"], self._pending["x"], r_gate)
        if self.use_memory:
            key = self._key(ctx.raw_obs if ctx.raw_obs is not None else ctx.obs_hist[-1])
            self.memory.add(key, int(action), r_gate, events, int(ctx.t))
        self._pending = None

    def n_params(self) -> int:
        return self.model.n_params() + self.gate.n_arms * self.gate.dim

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name, family="unified", n_params=self.n_params(),
            notes=(f"memory={self.use_memory} planning={self.use_planning} "
                   f"reasoner={self.use_reasoner} gate={self.learn_gate} "
                   f"backend={self.reasoner.backend}"),
            extra={"gate_usage": self.gate.usage(),
                   "pathways": list(PATHWAYS),
                   "model_error_ema": float(self.tracker.model_error_ema),
                   "imagined_steps": int(self.model.meter.imagined_steps)})
