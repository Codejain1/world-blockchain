"""Transformer, graph and observation-space latent dynamics.

Three further points on the architecture axis:

``TransformerDynamicsWorldModel``
    Deterministic latent, rolled forward by a causal Transformer over the
    *imagined* latent sequence.  Tests whether sequence modelling in latent
    space is enough, without RSSM's stochastic state or JEPA's EMA target.

``GraphDynamicsWorldModel``
    Latent state is the full set of graph node embeddings; a message-passing
    step *is* the transition.  Tests whether relational structure helps when the
    world is literally a graph of tokens, pools, markets and agent clusters.

``ObsSpaceDynamicsModel``
    The control that matters most: identical multi-step training objective,
    identical capacity, but the "latent" is the observation vector itself.  If
    this matches the latent models, then any world-model advantage in this lab
    is attributable to the training objective, not to learned latent state.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ...training.trainer import TrainConfig
from ..nn import MLP, GraphEncoder, GraphBlock, TemporalTransformer
from .base import LatentState, LatentWorldModel, WorldModelModule

__all__ = ["TransformerDynamicsWorldModel", "GraphDynamicsWorldModel",
           "ObsSpaceDynamicsModel"]


# --------------------------------------------------------------------------
class _TfmDynamicsModule(WorldModelModule):
    def __init__(self, obs_dim: int, n_actions: int, n_events: int, history: int,
                 d_model: int = 128, latent: int = 128, hidden: int = 256,
                 act_emb: int = 32, n_layers: int = 2, ctx_len: int = 16) -> None:
        super().__init__()
        self.latent_dim = latent
        self.ctx_len = int(ctx_len)
        self.act_emb = nn.Embedding(n_actions, act_emb)
        self.enc = TemporalTransformer(obs_dim + act_emb, d_model, 4, n_layers)
        self.to_latent = MLP([d_model, hidden, latent], norm=True)
        self.dyn = TemporalTransformer(latent + act_emb, d_model, 4, n_layers,
                                       max_len=ctx_len + 8)
        self.from_dyn = MLP([d_model, hidden, latent], norm=True)
        self.dec_obs = MLP([latent, hidden, hidden, obs_dim])
        self.dec_reward = MLP([latent, hidden, 1])
        self.dec_event = MLP([latent, hidden, n_events])

    def encode(self, b: Dict[str, torch.Tensor]) -> LatentState:
        a = self.act_emb(b["act_hist"])
        prev = torch.cat([torch.zeros_like(a[:, :1]), a[:, :-1]], dim=1)
        h = self.enc(torch.cat([b["obs_hist"], prev], dim=-1))
        z = self.to_latent(h)                       # (B, H, latent)
        z = z[:, -self.ctx_len:]
        pad = torch.zeros(z.shape[0], z.shape[1], self.act_emb.embedding_dim,
                          device=z.device)
        return {"zs": z, "as": pad, "z": z[:, -1]}

    def imagine(self, state: LatentState, action: torch.Tensor) -> LatentState:
        ae = self.act_emb(action).unsqueeze(1)
        acts = torch.cat([state["as"][:, 1:], ae], dim=1) if state["as"].shape[1] == \
            state["zs"].shape[1] else torch.cat([state["as"], ae], dim=1)
        seq = torch.cat([state["zs"], acts[:, -state["zs"].shape[1]:]], dim=-1)
        h = self.dyn(seq)[:, -1]
        z = state["z"] + self.from_dyn(h)           # residual latent transition
        zs = torch.cat([state["zs"][:, 1:], z.unsqueeze(1)], dim=1) \
            if state["zs"].shape[1] >= self.ctx_len else \
            torch.cat([state["zs"], z.unsqueeze(1)], dim=1)
        as_ = torch.cat([state["as"][:, 1:], ae], dim=1) \
            if state["as"].shape[1] >= self.ctx_len else \
            torch.cat([state["as"], ae], dim=1)
        return {"zs": zs, "as": as_[:, -zs.shape[1]:], "z": z}

    def readout(self, state: LatentState) -> Dict[str, torch.Tensor]:
        z = state["z"]
        return {"obs": self.dec_obs(z), "reward": self.dec_reward(z).squeeze(-1),
                "event_logit": self.dec_event(z)}


class TransformerDynamicsWorldModel(LatentWorldModel):
    def __init__(self, spec, cfg: Optional[TrainConfig] = None,
                 train_horizon: Optional[int] = None, name: str = "wm_transformer",
                 **kw) -> None:
        cfg = cfg or TrainConfig()
        mod = _TfmDynamicsModule(spec.obs_dim, spec.n_actions, spec.n_events,
                                 spec.history, **kw)
        super().__init__(name, mod, cfg, train_horizon=train_horizon)


# --------------------------------------------------------------------------
class _GraphDynamicsModule(WorldModelModule):
    def __init__(self, obs_dim: int, n_actions: int, n_events: int, history: int,
                 node_dim: int, edge_index: np.ndarray, edge_feat: np.ndarray,
                 node_type: np.ndarray, d_model: int = 64, n_gnn_layers: int = 2,
                 n_dyn_layers: int = 2, hidden: int = 256, act_emb: int = 32) -> None:
        super().__init__()
        self.register_buffer("edge_index", torch.as_tensor(edge_index, dtype=torch.long))
        self.register_buffer("edge_feat", torch.as_tensor(edge_feat, dtype=torch.float32))
        self.register_buffer("node_type", torch.as_tensor(node_type, dtype=torch.long))
        self.n_nodes = int(node_type.shape[0])
        self.d_model = d_model
        self.latent_dim = d_model * self.n_nodes

        self.gnn = GraphEncoder(node_dim, d_model, edge_feat.shape[1], n_gnn_layers,
                                n_node_types=int(node_type.max()) + 1)
        self.act_emb = nn.Embedding(n_actions, act_emb)
        self.time_mix = nn.GRU(d_model, d_model, batch_first=True)
        self.flat_mix = MLP([obs_dim, hidden, d_model], norm=True)
        self.dyn_blocks = nn.ModuleList(
            [GraphBlock(d_model, edge_feat.shape[1]) for _ in range(n_dyn_layers)])
        self.act_inject = MLP([act_emb, hidden, d_model], norm=True)
        self.dec_obs = MLP([2 * d_model, hidden, hidden, obs_dim])
        self.dec_reward = MLP([2 * d_model, hidden, 1])
        self.dec_event = MLP([2 * d_model, hidden, n_events])

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        return torch.cat([h.mean(dim=1), h[:, -1]], dim=-1)

    def encode(self, b: Dict[str, torch.Tensor]) -> LatentState:
        nh = b["node_hist"]                                     # (B, H, N, F)
        B, H, N, Fd = nh.shape
        h = self.gnn(nh.reshape(B * H, N, Fd), self.edge_index, self.edge_feat,
                     self.node_type).reshape(B, H, N, self.d_model)
        # Mix across time per node, then fold in the flat observation so the
        # graph model is never informationally weaker than the flat models.
        seq = h.permute(0, 2, 1, 3).reshape(B * N, H, self.d_model)
        out, _ = self.time_mix(seq)
        hN = out[:, -1].reshape(B, N, self.d_model)
        hN = hN + self.flat_mix(b["obs_hist"][:, -1]).unsqueeze(1)
        return {"h": hN}

    def imagine(self, state: LatentState, action: torch.Tensor) -> LatentState:
        h = state["h"]
        # The action enters at the focal-agent node and propagates outward --
        # the graph analogue of "my trade moves the pool I traded in first".
        inj = torch.zeros_like(h)
        inj[:, -1] = self.act_inject(self.act_emb(action))
        h = h + inj
        for blk in self.dyn_blocks:
            h = blk(h, self.edge_index, self.edge_feat)
        return {"h": h}

    def readout(self, state: LatentState) -> Dict[str, torch.Tensor]:
        z = self._pool(state["h"])
        return {"obs": self.dec_obs(z), "reward": self.dec_reward(z).squeeze(-1),
                "event_logit": self.dec_event(z)}


def _node_types(obs_builder) -> np.ndarray:
    nt = np.zeros(obs_builder.n_nodes, dtype=np.int64)
    K = obs_builder.cfg.n_tokens
    P = obs_builder.cfg.n_pools
    L = obs_builder.cfg.n_lend
    nt[K:K + P] = obs_builder.NODE_POOL
    nt[K + P:K + P + L] = obs_builder.NODE_MARKET
    nt[K + P + L:-1] = obs_builder.NODE_CLUSTER
    nt[-1] = obs_builder.NODE_FOCAL
    return nt


class GraphDynamicsWorldModel(LatentWorldModel):
    def __init__(self, spec, obs_builder, cfg: Optional[TrainConfig] = None,
                 train_horizon: Optional[int] = None, name: str = "wm_graph",
                 **kw) -> None:
        cfg = cfg or TrainConfig()
        cfg.with_graph = True
        ei, ef = obs_builder._build_graph_skeleton()
        mod = _GraphDynamicsModule(spec.obs_dim, spec.n_actions, spec.n_events,
                                   spec.history, spec.node_dim, ei, ef,
                                   _node_types(obs_builder), **kw)
        super().__init__(name, mod, cfg, with_graph=True, train_horizon=train_horizon)


# --------------------------------------------------------------------------
class _ObsSpaceModule(WorldModelModule):
    """Autoregressive dynamics *in observation space* -- no learned latent.

    Same open-loop objective, same capacity, no representation learning.  This is
    the control that separates "world model" from "multi-step training".
    """

    def __init__(self, obs_dim: int, n_actions: int, n_events: int, history: int,
                 hidden: int = 256, n_layers: int = 3, act_emb: int = 32) -> None:
        super().__init__()
        self.obs_dim, self.history = obs_dim, history
        self.latent_dim = obs_dim
        self.act_emb = nn.Embedding(n_actions, act_emb)
        self.trunk = MLP([history * obs_dim + act_emb] + [hidden] * n_layers,
                         out_act=True, norm=True)
        self.delta = MLP([hidden, hidden, obs_dim])
        self.dec_reward = MLP([hidden, hidden, 1])
        self.dec_event = MLP([hidden, hidden, n_events])

    def encode(self, b: Dict[str, torch.Tensor]) -> LatentState:
        return {"win": b["obs_hist"].clone(), "z": b["obs_hist"][:, -1].clone(),
                "feat": torch.zeros(b["obs_hist"].shape[0], 1,
                                    device=b["obs_hist"].device)}

    def imagine(self, state: LatentState, action: torch.Tensor) -> LatentState:
        win = state["win"]
        h = self.trunk(torch.cat([win.flatten(1), self.act_emb(action)], dim=-1))
        nxt = win[:, -1] + self.delta(h)
        return {"win": torch.cat([win[:, 1:], nxt.unsqueeze(1)], dim=1),
                "z": nxt, "feat": h}

    def readout(self, state: LatentState) -> Dict[str, torch.Tensor]:
        h = state["feat"]
        return {"obs": state["z"], "reward": self.dec_reward(h).squeeze(-1),
                "event_logit": self.dec_event(h)}


class ObsSpaceDynamicsModel(LatentWorldModel):
    def __init__(self, spec, cfg: Optional[TrainConfig] = None,
                 train_horizon: Optional[int] = None, name: str = "dyn_obsspace",
                 **kw) -> None:
        cfg = cfg or TrainConfig()
        mod = _ObsSpaceModule(spec.obs_dim, spec.n_actions, spec.n_events,
                              spec.history, **kw)
        super().__init__(name, mod, cfg, train_horizon=train_horizon)
        self.family = "baseline"
