"""Interfaces shared by every intelligence system in the lab.

Two abstractions cover everything:

``PredictiveModel``  answers "what happens next?"  -- used for the prediction,
                     event, counterfactual and long-horizon tasks.
``Policy``           answers "what should I do?"   -- used for the control,
                     planning and adaptation tasks.

Baselines, world models, the LLM system and the unified system all implement one
or both.  Because the evaluation harness only ever talks to these interfaces, no
system can quietly receive a different input, a different action space or a
different number of environment interactions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["StepPrediction", "RolloutPrediction", "PredictiveModel",
           "DecisionContext", "Policy", "ModelInfo"]


@dataclass
class StepPrediction:
    """One-step prediction.  ``delta`` and ``reward`` are in normalised units."""

    delta: np.ndarray            # (B, D) predicted obs_{t+1} - obs_t
    event_logit: np.ndarray      # (B, E) logits for the binary event vector
    reward: np.ndarray           # (B,)   predicted focal reward


@dataclass
class RolloutPrediction:
    """Multi-step prediction produced *without* touching the simulator."""

    obs: np.ndarray              # (B, L, D) absolute normalised observations
    event_logit: np.ndarray      # (B, L, E)
    reward: np.ndarray           # (B, L)


@dataclass
class ModelInfo:
    """Static description of a system, recorded with every result."""

    name: str
    family: str                  # baseline | world_model | llm | unified
    n_params: int = 0
    notes: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class PredictiveModel(ABC):
    """A system that predicts the world's next state, events and reward."""

    family: str = "baseline"

    def __init__(self, name: str) -> None:
        self.name = name

    # -- training --------------------------------------------------------
    @abstractmethod
    def fit(self, train, val, spec, normalizer, **kwargs) -> Dict[str, Any]:
        """Fit on a :class:`~bwm.data.dataset.TrajectorySet`; return a log."""

    # -- inference -------------------------------------------------------
    @abstractmethod
    def predict_step(self, batch: Dict[str, np.ndarray]) -> StepPrediction:
        """One-step prediction from ``obs_hist``/``act_hist``."""

    def predict_rollout(self, batch: Dict[str, np.ndarray], horizon: int
                        ) -> RolloutPrediction:
        """Multi-step prediction.

        The default is honest autoregression: feed each prediction back in as the
        next observation.  Models with native latent rollouts override this.
        """
        obs_hist = np.array(batch["obs_hist"], copy=True)          # (B, H, D)
        act_hist = np.array(batch["act_hist"], copy=True)          # (B, H)
        fut_act = batch["fut_act"]                                  # (B, L)
        B, H, D = obs_hist.shape
        E = None
        obs_out: List[np.ndarray] = []
        ev_out: List[np.ndarray] = []
        rw_out: List[np.ndarray] = []
        node_hist = batch.get("node_hist")
        for l in range(horizon):
            act_hist[:, -1] = fut_act[:, l]
            sub = {"obs_hist": obs_hist, "act_hist": act_hist}
            if node_hist is not None and node_hist.ndim == 4:
                sub["node_hist"] = node_hist
            p = self.predict_step(sub)
            nxt = obs_hist[:, -1] + p.delta
            obs_out.append(nxt)
            ev_out.append(p.event_logit)
            rw_out.append(p.reward)
            obs_hist = np.concatenate([obs_hist[:, 1:], nxt[:, None, :]], axis=1)
            act_hist = np.concatenate([act_hist[:, 1:], act_hist[:, -1:]], axis=1)
            if node_hist is not None and node_hist.ndim == 4:
                # Graph features are not predicted by non-graph models; hold the
                # last snapshot fixed and note the limitation in the results.
                node_hist = np.concatenate(
                    [node_hist[:, 1:], node_hist[:, -1:]], axis=1)
        return RolloutPrediction(
            obs=np.stack(obs_out, axis=1),
            event_logit=np.stack(ev_out, axis=1),
            reward=np.stack(rw_out, axis=1))

    # -- bookkeeping -----------------------------------------------------
    def n_params(self) -> int:
        return 0

    def info(self) -> ModelInfo:
        return ModelInfo(name=self.name, family=self.family, n_params=self.n_params())

    def save(self, path: str) -> None:       # pragma: no cover - optional
        raise NotImplementedError

    def load(self, path: str) -> None:       # pragma: no cover - optional
        raise NotImplementedError


# --------------------------------------------------------------------------
@dataclass
class DecisionContext:
    """Everything a policy is allowed to see when choosing an action.

    ``world`` is present only because two *explicitly privileged* systems need
    it: the oracle-simulator planner (ablation E, an upper bound) and the
    feasibility mask.  Ordinary systems must use ``obs_hist``/``act_hist`` only;
    the evaluation harness asserts this by constructing contexts with
    ``allow_simulator=False`` for them.
    """

    obs_hist: np.ndarray            # (H, D) normalised observation window
    act_hist: np.ndarray            # (H,)   past action indices
    t: int
    agent: int
    action_space: Any
    feasible: np.ndarray            # (n_actions,) bool
    world: Any = None
    allow_simulator: bool = False
    raw_obs: Optional[np.ndarray] = None   # unnormalised, for LLM prompting
    info: Dict[str, Any] = field(default_factory=dict)


class Policy(ABC):
    """A system that selects actions."""

    family: str = "baseline"

    def __init__(self, name: str) -> None:
        self.name = name

    def reset(self, episode_seed: Optional[int] = None) -> None:
        """Called at the start of every evaluation episode."""

    @abstractmethod
    def act(self, ctx: DecisionContext) -> int:
        """Return a discrete action index."""

    def observe_outcome(self, ctx: DecisionContext, action: int, reward: float,
                        next_obs: np.ndarray, events: np.ndarray) -> None:
        """Consequence feedback.

        Systems with memory or continual learning use this; stateless systems
        ignore it.  Every system receives exactly the same feedback signal.
        """

    def n_params(self) -> int:
        return 0

    def info(self) -> ModelInfo:
        return ModelInfo(name=self.name, family=self.family, n_params=self.n_params())
