"""JEPA-style world model: prediction in representation space, no reconstruction.

The encoder is trained so that a predictor can map ``(z_t, a_t)`` onto the
representation an *EMA target encoder* assigns to the next observation.  Nothing
in that objective asks the model to reproduce pixels -- or, here, the 112
observation features -- which is LeCun's central architectural claim.

Collapse is the obvious failure mode of such an objective (a constant encoder has
zero prediction error), so we add VICReg variance and covariance terms and report
the realised latent standard deviation in the results.  A collapsed JEPA would be
a finding, not something to hide.

Observation read-outs are produced by a decoder trained on **detached** latents,
so reconstruction gradients never shape the representation; the decoder exists
only to make JEPA comparable on the shared prediction benchmark.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...training.trainer import TrainConfig
from ..nn import MLP, TemporalTransformer
from .base import LatentState, LatentWorldModel, WorldModelModule

__all__ = ["JEPAModule", "JEPAWorldModel"]


def vicreg_terms(z: torch.Tensor, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """Variance and covariance regularisers that make collapse costly."""
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = F.relu(1.0 - std).mean()
    n, d = z.shape
    cov = (z.T @ z) / max(n - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    cov_loss = (off ** 2).sum() / d
    return var_loss, cov_loss


class JEPAModule(WorldModelModule):
    def __init__(self, obs_dim: int, n_actions: int, n_events: int, history: int,
                 d_model: int = 128, latent: int = 128, hidden: int = 256,
                 act_emb: int = 32, n_layers: int = 2, ema: float = 0.99,
                 pred_weight: float = 2.0, var_weight: float = 1.0,
                 cov_weight: float = 0.04) -> None:
        super().__init__()
        self.latent_dim = latent
        self.ema = float(ema)
        self.pred_weight = float(pred_weight)
        self.var_weight = float(var_weight)
        self.cov_weight = float(cov_weight)
        self.aux_weight = 1.0
        self.history = history

        self.act_emb = nn.Embedding(n_actions, act_emb)
        self.ctx = TemporalTransformer(obs_dim + act_emb, d_model, 4, n_layers)
        self.to_latent = MLP([d_model, hidden, latent], norm=True)
        # Target encoder sees a single observation; it is an EMA copy and never
        # receives gradients (the standard JEPA anti-collapse construction).
        self.target_enc = MLP([obs_dim, hidden, latent], norm=True)
        self.online_enc = MLP([obs_dim, hidden, latent], norm=True)
        for p in self.target_enc.parameters():
            p.requires_grad_(False)
        self.target_enc.load_state_dict(self.online_enc.state_dict())

        self.predictor = MLP([latent + act_emb, hidden, hidden, latent], norm=True)
        self.dec_obs = MLP([latent, hidden, hidden, obs_dim])
        self.dec_reward = MLP([latent, hidden, 1])
        self.dec_event = MLP([latent, hidden, n_events])

    # -- required API ----------------------------------------------------
    def encode(self, b: Dict[str, torch.Tensor]) -> LatentState:
        a = self.act_emb(b["act_hist"])
        prev = torch.cat([torch.zeros_like(a[:, :1]), a[:, :-1]], dim=1)
        h = self.ctx(torch.cat([b["obs_hist"], prev], dim=-1))[:, -1]
        return {"z": self.to_latent(h)}

    def imagine(self, state: LatentState, action: torch.Tensor) -> LatentState:
        z = self.predictor(torch.cat([state["z"], self.act_emb(action)], dim=-1))
        return {"z": z}

    def readout(self, state: LatentState) -> Dict[str, torch.Tensor]:
        z = state["z"]
        # Decoder gradients are blocked so that reconstruction cannot shape the
        # representation -- otherwise this stops being a JEPA.
        zd = z.detach()
        return {"obs": self.dec_obs(zd), "reward": self.dec_reward(zd).squeeze(-1),
                "event_logit": self.dec_event(zd)}

    @torch.no_grad()
    def _update_target(self) -> None:
        for tp, op in zip(self.target_enc.parameters(), self.online_enc.parameters()):
            tp.mul_(self.ema).add_(op.detach(), alpha=1.0 - self.ema)

    def loss(self, b: Dict[str, torch.Tensor], horizon: Optional[int] = None):
        L = horizon or int(b["fut_act"].shape[1])
        total, logs = super().loss(b, horizon=L)

        # --- representation-space prediction (the actual JEPA objective) ---
        state = self.encode(b)
        pred_loss = torch.zeros((), device=total.device)
        var_loss = torch.zeros((), device=total.device)
        cov_loss = torch.zeros((), device=total.device)
        weights = torch.tensor([0.9 ** i for i in range(L)], device=total.device)
        weights = weights / weights.sum()
        for l in range(L):
            state = self.imagine(state, b["fut_act"][:, l])
            with torch.no_grad():
                tgt = self.target_enc(b["fut_obs"][:, l])
            zc = self.online_enc(b["fut_obs"][:, l])     # keeps the online encoder alive
            pred_loss = pred_loss + weights[l] * (
                F.mse_loss(state["z"], tgt) + 0.5 * F.mse_loss(zc, tgt.detach()))
            v, c = vicreg_terms(state["z"])
            var_loss = var_loss + weights[l] * v
            cov_loss = cov_loss + weights[l] * c
        total = total + self.pred_weight * pred_loss \
            + self.var_weight * var_loss + self.cov_weight * cov_loss
        if self.training:
            self._update_target()
        logs.update({"jepa_pred": float(pred_loss.detach()),
                     "jepa_var": float(var_loss.detach()),
                     "jepa_cov": float(cov_loss.detach())})
        return total, logs

    @torch.no_grad()
    def latent_std(self, b: Dict[str, torch.Tensor]) -> float:
        """Diagnostic: mean per-dimension latent std.  Near zero means collapse."""
        return float(self.encode(b)["z"].std(dim=0).mean())


class JEPAWorldModel(LatentWorldModel):
    def __init__(self, spec, cfg: Optional[TrainConfig] = None,
                 train_horizon: Optional[int] = None, name: str = "wm_jepa",
                 **kw) -> None:
        cfg = cfg or TrainConfig()
        mod = JEPAModule(spec.obs_dim, spec.n_actions, spec.n_events, spec.history, **kw)
        super().__init__(name, mod, cfg, train_horizon=train_horizon)

    def collapse_diagnostic(self, batch) -> float:
        from ...training.trainer import to_torch
        return self.module.latent_std(to_torch(batch, self.cfg.device))
