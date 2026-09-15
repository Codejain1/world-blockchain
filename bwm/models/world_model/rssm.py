"""RSSM: recurrent state-space model with stochastic latents (Dreamer-style).

State is split into a deterministic recurrent part ``h`` and a stochastic part
``s``.  The *prior* ``p(s_t | h_t)`` is what imagination uses; the *posterior*
``q(s_t | h_t, o_t)`` is available only when a real observation arrives.  Training
minimises prediction error plus ``KL(q || p)``, which forces the prior -- the
thing the planner rolls forward -- to be a usable model of the world rather than
a decoder of the current frame.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from ...training.trainer import TrainConfig
from ..nn import MLP
from .base import LatentState, LatentWorldModel, WorldModelModule

__all__ = ["RSSMModule", "RSSMWorldModel"]


class RSSMModule(WorldModelModule):
    def __init__(self, obs_dim: int, n_actions: int, n_events: int,
                 deter: int = 192, stoch: int = 32, hidden: int = 256,
                 act_emb: int = 32, free_nats: float = 1.0,
                 kl_weight: float = 1.0, kl_balance: float = 0.8) -> None:
        super().__init__()
        self.obs_dim, self.deter, self.stoch = obs_dim, deter, stoch
        self.latent_dim = deter + stoch
        self.free_nats = float(free_nats)
        self.aux_weight = float(kl_weight)
        self.kl_balance = float(kl_balance)

        self.act_emb = nn.Embedding(n_actions, act_emb)
        self.obs_enc = MLP([obs_dim, hidden, hidden], out_act=True, norm=True)
        self.cell = nn.GRUCell(stoch + act_emb, deter)
        self.prior_net = MLP([deter, hidden, 2 * stoch])
        self.post_net = MLP([deter + hidden, hidden, 2 * stoch])

        feat = deter + stoch
        self.dec_obs = MLP([feat, hidden, hidden, obs_dim])
        self.dec_reward = MLP([feat, hidden, 1])
        self.dec_event = MLP([feat, hidden, n_events])

    # -- distributions ---------------------------------------------------
    @staticmethod
    def _dist(params: torch.Tensor) -> D.Normal:
        mean, std = params.chunk(2, dim=-1)
        std = F.softplus(std) + 0.1
        return D.Normal(mean, std)

    def _feat(self, state: LatentState) -> torch.Tensor:
        return torch.cat([state["h"], state["s"]], dim=-1)

    # -- required API ----------------------------------------------------
    def encode(self, b: Dict[str, torch.Tensor]) -> LatentState:
        obs, acts = b["obs_hist"], b["act_hist"]
        B, H, _ = obs.shape
        dev = obs.device
        h = torch.zeros(B, self.deter, device=dev)
        s = torch.zeros(B, self.stoch, device=dev)
        # Action at index i-1 leads into observation i; the first step has none.
        prev = torch.cat([torch.zeros_like(acts[:, :1]), acts[:, :-1]], dim=1)
        kl_acc = torch.zeros((), device=dev)
        for i in range(H):
            h = self.cell(torch.cat([s, self.act_emb(prev[:, i])], dim=-1), h)
            prior = self._dist(self.prior_net(h))
            e = self.obs_enc(obs[:, i])
            post = self._dist(self.post_net(torch.cat([h, e], dim=-1)))
            s = post.rsample()
            kl_acc = kl_acc + self._kl(post, prior)
        return {"h": h, "s": s, "kl": kl_acc / max(H, 1)}

    def imagine(self, state: LatentState, action: torch.Tensor) -> LatentState:
        h = self.cell(torch.cat([state["s"], self.act_emb(action)], dim=-1), state["h"])
        prior = self._dist(self.prior_net(h))
        s = prior.rsample()
        return {"h": h, "s": s, "kl": state.get(
            "kl", torch.zeros((), device=h.device))}

    def readout(self, state: LatentState) -> Dict[str, torch.Tensor]:
        f = self._feat(state)
        return {"obs": self.dec_obs(f), "reward": self.dec_reward(f).squeeze(-1),
                "event_logit": self.dec_event(f)}

    def _kl(self, post: D.Normal, prior: D.Normal) -> torch.Tensor:
        """KL balancing (Dreamer-v2): the prior is pulled harder than the posterior."""
        kl_lhs = D.kl_divergence(D.Normal(post.mean.detach(), post.stddev.detach()),
                                 prior).sum(-1)
        kl_rhs = D.kl_divergence(
            post, D.Normal(prior.mean.detach(), prior.stddev.detach())).sum(-1)
        a = self.kl_balance
        kl = a * kl_lhs + (1.0 - a) * kl_rhs
        return torch.clamp(kl, min=self.free_nats).mean()

    def aux_loss(self, state: LatentState, b, step: int) -> torch.Tensor:
        # The representation KL is accumulated once during `encode`; charge it on
        # the first imagination step only so it is not counted L times.
        if step == 0 and "kl" in state:
            return state["kl"]
        return torch.zeros((), device=b["obs_hist"].device)


class RSSMWorldModel(LatentWorldModel):
    def __init__(self, spec, cfg: Optional[TrainConfig] = None,
                 train_horizon: Optional[int] = None, name: str = "wm_rssm",
                 **kw) -> None:
        cfg = cfg or TrainConfig()
        mod = RSSMModule(spec.obs_dim, spec.n_actions, spec.n_events, **kw)
        super().__init__(name, mod, cfg, train_horizon=train_horizon)
