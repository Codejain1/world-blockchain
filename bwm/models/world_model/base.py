"""Scaffolding shared by every latent world model.

A world model here is anything that can

1. ``encode`` an observation/action history into a latent state ``z_t``,
2. ``imagine`` forward in that latent space -- ``z_{t+1} ~ p(z_{t+1} | z_t, a_t)``
   -- **without executing anything in the simulator**, and
3. ``readout`` predicted observations, rewards and events from a latent state.

Training objective
------------------
World models are trained *open loop*: encode the history, then roll the latent
forward for ``L`` steps driven only by the action sequence, and supervise the
readouts at every step.  That open-loop objective is the defining difference
from a one-step predictor, and it is exactly what the planner later relies on.

Because "trained for multi-step prediction" is itself a plausible explanation for
any advantage, the ablation suite also trains (a) world models with ``L = 1`` and
(b) flat baselines with the same multi-step unroll.  If the flat baseline closes
the gap once it gets the same objective, the advantage was the objective, not the
latent state -- and the report says so.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...data.dataset import BatchSpec, Normalizer
from ...training.trainer import LossModule, TorchPredictiveModel, TrainConfig, to_torch
from ..base import RolloutPrediction, StepPrediction
from ..baselines.neural import LOSS_WEIGHTS

__all__ = ["LatentState", "WorldModelModule", "LatentWorldModel"]

LatentState = Dict[str, torch.Tensor]


class WorldModelModule(LossModule):
    """Abstract latent world model.  Subclasses implement three methods."""

    latent_dim: int = 0
    #: weight of any architecture-specific regulariser (KL, VICReg, ...)
    aux_weight: float = 1.0
    #: Decode observations as a *residual* on the previous observation estimate.
    #:
    #: Fairness fix.  The flat baselines predict ``obs_{t+1} - obs_t`` directly,
    #: so at initialisation they output ~0 and therefore start from the strong
    #: persistence prior.  A world model that decodes absolute observations from
    #: its latent starts from the dataset mean instead, which is far worse -- an
    #: architecture-induced handicap that has nothing to do with whether latent
    #: world models work.  Residual decoding removes it: every system now starts
    #: at persistence and is scored on what it adds.
    residual_decode: bool = True

    # -- required API ----------------------------------------------------
    def encode(self, b: Dict[str, torch.Tensor]) -> LatentState:
        """History -> latent state at the last observed step."""
        raise NotImplementedError

    def imagine(self, state: LatentState, action: torch.Tensor) -> LatentState:
        """One action-conditioned latent transition (no simulator involved)."""
        raise NotImplementedError

    def _readout_raw(self, state: LatentState) -> Dict[str, torch.Tensor]:
        """Latent -> (residual or absolute) observation, reward, event logits."""
        raise NotImplementedError

    def readout(self, state: LatentState) -> Dict[str, torch.Tensor]:
        out = self._readout_raw(state)
        if self.residual_decode and "obs_prev" in state:
            out = dict(out)
            out["obs"] = state["obs_prev"] + out["obs"]
        return out

    @staticmethod
    def advance(state: LatentState, out: Dict[str, torch.Tensor]) -> LatentState:
        """Carry the predicted observation forward as the next residual anchor."""
        nxt = dict(state)
        nxt["obs_prev"] = out["obs"]
        return nxt

    def aux_loss(self, state: LatentState, b: Dict[str, torch.Tensor],
                 step: int) -> torch.Tensor:
        """Architecture-specific extra loss (KL for RSSM, VICReg for JEPA)."""
        return torch.zeros((), device=next(self.parameters()).device)

    # -- shared open-loop objective --------------------------------------
    def rollout(self, b: Dict[str, torch.Tensor], horizon: int
                ) -> Tuple[List[Dict[str, torch.Tensor]], List[LatentState]]:
        state = self.encode(b)
        outs, states = [], []
        for l in range(horizon):
            state = self.imagine(state, b["fut_act"][:, l])
            out = self.readout(state)
            outs.append(out)
            states.append(state)
            state = self.advance(state, out)
        return outs, states

    def loss(self, b: Dict[str, torch.Tensor], horizon: Optional[int] = None
             ) -> Tuple[torch.Tensor, Dict[str, float]]:
        L = horizon or int(b["fut_act"].shape[1])
        wd, we, wr = LOSS_WEIGHTS
        outs, states = self.rollout(b, L)
        l_obs = l_ev = l_rw = l_aux = 0.0
        # Geometric discounting over the horizon keeps the one-step term
        # dominant while still supervising long-range consistency.
        weights = torch.tensor([0.9 ** i for i in range(L)],
                               device=b["obs_hist"].device)
        weights = weights / weights.sum()
        for l, (o, st) in enumerate(zip(outs, states)):
            w = weights[l]
            l_obs = l_obs + w * F.mse_loss(o["obs"], b["fut_obs"][:, l])
            l_ev = l_ev + w * F.binary_cross_entropy_with_logits(
                o["event_logit"], b["fut_event"][:, l])
            l_rw = l_rw + w * F.mse_loss(o["reward"], b["fut_reward"][:, l])
            # The architecture regulariser (RSSM's KL, JEPA's VICReg) is charged
            # at FULL weight, not discounted by the horizon weight.
            #
            # It used to be multiplied by w[l].  Because the regulariser is
            # charged once, at l = 0, that made its effective strength depend on
            # the horizon: w[0] = 1.00 at L = 1 but only 0.213 at L = 6, so the
            # one-step ablation regularised its latent 4.7x harder than the
            # model it was meant to be compared against.  The ablation would then
            # have measured "more KL" as well as "shorter horizon", which is
            # exactly the confound it exists to rule out.
            l_aux = l_aux + self.aux_loss(st, b, l)
        total = wd * l_obs + we * l_ev + wr * l_rw + self.aux_weight * l_aux
        return total, {"obs": float(l_obs.detach()), "event": float(l_ev.detach()),
                       "reward": float(l_rw.detach()),
                       "aux": float(l_aux.detach()) if torch.is_tensor(l_aux) else 0.0}

    @torch.no_grad()
    def predict(self, b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """One-step prediction expressed as a delta, for the shared interface."""
        state = self.encode(b)
        state = self.imagine(state, b["act_hist"][:, -1])
        out = self.readout(state)
        return {"delta": out["obs"] - b["obs_hist"][:, -1],
                "event_logit": out["event_logit"], "reward": out["reward"]}


class LatentWorldModel(TorchPredictiveModel):
    """Evaluation-harness wrapper: native latent rollouts, no simulator access."""

    family = "world_model"

    def __init__(self, name: str, module: WorldModelModule, cfg: TrainConfig,
                 with_graph: bool = False, train_horizon: Optional[int] = None) -> None:
        super().__init__(name, module, cfg, with_graph=with_graph)
        self.train_horizon = train_horizon
        self.imagined_steps = 0

    def fit(self, train, val, spec, normalizer, **kw):
        if self.train_horizon is not None:
            # Wrap the module loss so the trainer uses the configured horizon.
            base_loss = self.module.loss
            h = int(self.train_horizon)
            self.module.loss = lambda b: base_loss(b, horizon=h)   # type: ignore[assignment]
        out = super().fit(train, val, spec, normalizer, **kw)
        out["train_horizon"] = self.train_horizon or spec.horizon
        return out

    # -- native latent imagination --------------------------------------
    def predict_rollout(self, batch: Dict[str, np.ndarray], horizon: int
                        ) -> RolloutPrediction:
        self.module.eval()
        b = to_torch(batch, self.cfg.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            state = self.module.encode(b)
            obs, ev, rw = [], [], []
            for l in range(horizon):
                state = self.module.imagine(state, b["fut_act"][:, l])
                o = self.module.readout(state)
                obs.append(o["obs"])
                ev.append(o["event_logit"])
                rw.append(o["reward"])
                state = self.module.advance(state, o)
        B = int(batch["obs_hist"].shape[0])
        self.imagined_steps += B * horizon
        self.meter.imagined_steps += B * horizon
        self.meter.add_inference(time.perf_counter() - t0, B)
        return RolloutPrediction(
            obs=torch.stack(obs, 1).cpu().numpy(),
            event_logit=torch.stack(ev, 1).cpu().numpy(),
            reward=torch.stack(rw, 1).cpu().numpy())

    # -- planner-facing API ---------------------------------------------
    @torch.no_grad()
    def encode_numpy(self, obs_hist: np.ndarray, act_hist: np.ndarray,
                     node_hist: Optional[np.ndarray] = None) -> LatentState:
        b = {"obs_hist": obs_hist.astype(np.float32),
             "act_hist": act_hist.astype(np.int64)}
        if node_hist is not None:
            b["node_hist"] = node_hist.astype(np.float32)
        return self.module.encode(to_torch(b, self.cfg.device))

    @torch.no_grad()
    def imagine_sequences(self, state: LatentState, actions: torch.Tensor
                          ) -> Dict[str, torch.Tensor]:
        """Roll ``(B, L)`` action sequences forward from a batched latent state.

        Returns stacked predicted rewards, event logits and observations.  This
        is the only thing the model-based planner is allowed to call -- it never
        touches the simulator.
        """
        L = int(actions.shape[1])
        rewards, events, obs = [], [], []
        for l in range(L):
            state = self.module.imagine(state, actions[:, l])
            o = self.module.readout(state)
            rewards.append(o["reward"])
            events.append(o["event_logit"])
            obs.append(o["obs"])
            state = self.module.advance(state, o)
        n = int(actions.shape[0]) * L
        self.imagined_steps += n
        self.meter.imagined_steps += n
        return {"reward": torch.stack(rewards, 1), "event_logit": torch.stack(events, 1),
                "obs": torch.stack(obs, 1)}

    @staticmethod
    def expand_state(state: LatentState, n: int) -> LatentState:
        """Tile a single latent state into ``n`` parallel imagination threads."""
        return {k: v.repeat_interleave(n, dim=0) if v.dim() > 0 else v
                for k, v in state.items()}
