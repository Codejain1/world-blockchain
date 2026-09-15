"""Neural scientific baselines: MLP, temporal Transformer, temporal graph network.

These are *not* optional extras.  If a plain Transformer over the same window
matches a latent world model on every task, the world-model hypothesis is not
supported by this environment, and the comparison has to be able to show that.

All three share the same heads, the same loss weighting and the same trainer, so
they differ only in how they encode the window.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...training.trainer import LossModule, TorchPredictiveModel, TrainConfig
from ..nn import MLP, GraphEncoder, TemporalTransformer

__all__ = ["PredictionHeads", "MLPBaseline", "TransformerBaseline", "GNNBaseline",
           "LOSS_WEIGHTS"]

#: Shared loss weighting (delta, event, reward).  Identical for every learned
#: system in the lab so that no model is implicitly tuned for one task.
LOSS_WEIGHTS: Tuple[float, float, float] = (1.0, 1.0, 0.5)


class PredictionHeads(nn.Module):
    """Delta / event / reward heads on top of any encoder representation."""

    def __init__(self, d_model: int, obs_dim: int, n_events: int,
                 hidden: int = 256) -> None:
        super().__init__()
        self.delta = MLP([d_model, hidden, obs_dim])
        self.event = MLP([d_model, hidden, n_events])
        self.reward = MLP([d_model, hidden, 1])

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"delta": self.delta(h), "event_logit": self.event(h),
                "reward": self.reward(h).squeeze(-1)}


def prediction_loss(out: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor],
                    weights: Tuple[float, float, float] = LOSS_WEIGHTS
                    ) -> Tuple[torch.Tensor, Dict[str, float]]:
    wd, we, wr = weights
    l_delta = F.mse_loss(out["delta"], b["delta"])
    l_event = F.binary_cross_entropy_with_logits(out["event_logit"], b["event"])
    l_reward = F.mse_loss(out["reward"], b["reward"])
    total = wd * l_delta + we * l_event + wr * l_reward
    return total, {"delta": float(l_delta.detach()), "event": float(l_event.detach()),
                   "reward": float(l_reward.detach())}


# --------------------------------------------------------------------------
class _MLPModule(LossModule):
    def __init__(self, obs_dim: int, n_actions: int, n_events: int, history: int,
                 hidden: int = 256, n_layers: int = 3, act_emb: int = 16,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.history = history
        self.act_emb = nn.Embedding(n_actions, act_emb)
        d_in = history * (obs_dim + act_emb)
        sizes = [d_in] + [hidden] * n_layers
        self.trunk = MLP(sizes, out_act=True, norm=True, dropout=dropout)
        self.heads = PredictionHeads(hidden, obs_dim, n_events)

    def encode(self, b: Dict[str, torch.Tensor]) -> torch.Tensor:
        a = self.act_emb(b["act_hist"])                       # (B, H, A)
        x = torch.cat([b["obs_hist"], a], dim=-1).flatten(1)
        return self.trunk(x)

    def loss(self, b):
        return prediction_loss(self.heads(self.encode(b)), b)

    @torch.no_grad()
    def predict(self, b):
        return self.heads(self.encode(b))


class MLPBaseline(TorchPredictiveModel):
    def __init__(self, spec, cfg: Optional[TrainConfig] = None, **kw) -> None:
        cfg = cfg or TrainConfig()
        mod = _MLPModule(spec.obs_dim, spec.n_actions, spec.n_events, spec.history, **kw)
        super().__init__("mlp", mod, cfg)


# --------------------------------------------------------------------------
class _TransformerModule(LossModule):
    def __init__(self, obs_dim: int, n_actions: int, n_events: int, history: int,
                 d_model: int = 128, n_heads: int = 4, n_layers: int = 3,
                 act_emb: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.act_emb = nn.Embedding(n_actions, act_emb)
        self.tfm = TemporalTransformer(obs_dim + act_emb, d_model, n_heads,
                                       n_layers, dropout=dropout)
        self.heads = PredictionHeads(d_model, obs_dim, n_events)

    def encode(self, b):
        a = self.act_emb(b["act_hist"])
        h = self.tfm(torch.cat([b["obs_hist"], a], dim=-1))
        return h[:, -1]

    def loss(self, b):
        return prediction_loss(self.heads(self.encode(b)), b)

    @torch.no_grad()
    def predict(self, b):
        return self.heads(self.encode(b))


class TransformerBaseline(TorchPredictiveModel):
    def __init__(self, spec, cfg: Optional[TrainConfig] = None, **kw) -> None:
        cfg = cfg or TrainConfig()
        mod = _TransformerModule(spec.obs_dim, spec.n_actions, spec.n_events,
                                 spec.history, **kw)
        super().__init__("transformer", mod, cfg)


# --------------------------------------------------------------------------
class _GNNModule(LossModule):
    """Temporal graph network: graph encoder per step, GRU across steps.

    The graph carries the *same information* as the flat vector (see
    ``ObservationBuilder._cluster_features``); what differs is the relational
    structure -- tokens wired to the pools and markets that reference them, and
    agent clusters wired to the venues they use.
    """

    def __init__(self, obs_dim: int, n_actions: int, n_events: int, history: int,
                 node_dim: int, edge_index: np.ndarray, edge_feat: np.ndarray,
                 node_type: np.ndarray, d_model: int = 64, n_gnn_layers: int = 2,
                 d_rnn: int = 192, act_emb: int = 16) -> None:
        super().__init__()
        self.register_buffer("edge_index", torch.as_tensor(edge_index, dtype=torch.long))
        self.register_buffer("edge_feat", torch.as_tensor(edge_feat, dtype=torch.float32))
        self.register_buffer("node_type", torch.as_tensor(node_type, dtype=torch.long))
        self.gnn = GraphEncoder(node_dim, d_model, edge_feat.shape[1], n_gnn_layers,
                                n_node_types=int(node_type.max()) + 1)
        self.act_emb = nn.Embedding(n_actions, act_emb)
        # Pooled graph state + focal-node state + the flat vector.  The flat
        # vector is included so the graph model is never *weaker* in information
        # than the MLP/Transformer -- the comparison is about structure.
        d_step = 2 * d_model + obs_dim + act_emb
        self.step_proj = MLP([d_step, d_rnn], out_act=True, norm=True)
        self.rnn = nn.GRU(d_rnn, d_rnn, batch_first=True)
        self.heads = PredictionHeads(d_rnn, obs_dim, n_events)

    def encode(self, b):
        nh = b["node_hist"]                                   # (B, H, N, F)
        B, H, N, Fd = nh.shape
        h = self.gnn(nh.reshape(B * H, N, Fd), self.edge_index, self.edge_feat,
                     self.node_type)                          # (B*H, N, d)
        pooled = h.mean(dim=1)
        focal = h[:, -1]
        g = torch.cat([pooled, focal], dim=-1).reshape(B, H, -1)
        a = self.act_emb(b["act_hist"])
        x = self.step_proj(torch.cat([g, b["obs_hist"], a], dim=-1))
        out, _ = self.rnn(x)
        return out[:, -1]

    def loss(self, b):
        return prediction_loss(self.heads(self.encode(b)), b)

    @torch.no_grad()
    def predict(self, b):
        return self.heads(self.encode(b))


class GNNBaseline(TorchPredictiveModel):
    """Temporal graph baseline.

    Note on autoregressive rollout: this model consumes graph snapshots it does
    not itself predict, so multi-step rollouts hold the last graph fixed.  That
    is a genuine limitation of using a graph *encoder* as a predictor, and it is
    reported rather than hidden -- the graph *world model* does predict its own
    latent forward, which is precisely the distinction under test.
    """

    def __init__(self, spec, obs_builder, cfg: Optional[TrainConfig] = None, **kw) -> None:
        cfg = cfg or TrainConfig()
        cfg.with_graph = True
        g = obs_builder._build_graph_skeleton()
        node_type = np.zeros(obs_builder.n_nodes, dtype=np.int64)
        K, P, L = obs_builder.cfg.n_tokens, obs_builder.cfg.n_pools, obs_builder.cfg.n_lend
        node_type[K:K + P] = obs_builder.NODE_POOL
        node_type[K + P:K + P + L] = obs_builder.NODE_MARKET
        node_type[K + P + L:-1] = obs_builder.NODE_CLUSTER
        node_type[-1] = obs_builder.NODE_FOCAL
        mod = _GNNModule(spec.obs_dim, spec.n_actions, spec.n_events, spec.history,
                         spec.node_dim, g[0], g[1], node_type, **kw)
        super().__init__("temporal_gnn", mod, cfg, with_graph=True)
