"""Observation construction -- the *single* information channel of the lab.

Every intelligence system (LLM, world model, unified, and all baselines) sees
exactly the vector produced here and nothing else.  Two facts are deliberately
withheld from all of them:

* the hidden market ``regime``, and
* the exogenous ``fundamental`` price (only the protocol ``oracle_price`` is
  visible, which can diverge from it under glitch/TWAP configurations).

Latent-state inference is therefore a real problem rather than a lookup, and no
system can "cheat" by reading the generative parameters.  Aggregate population
statistics *are* exposed, because on a real chain they are public; they are
exposed identically to every system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import EnvConfig
from .state import EPS

__all__ = ["ObservationBuilder", "GraphObservation"]


def _safe_log(x: np.ndarray | float, floor: float = 1e-12) -> np.ndarray | float:
    return np.log(np.maximum(x, floor))


@dataclass
class GraphObservation:
    """Heterogeneous graph view of the world used by graph encoders."""

    node_feat: np.ndarray    # (N, F_node)
    node_type: np.ndarray    # (N,) int
    edge_index: np.ndarray   # (2, E) int
    edge_feat: np.ndarray    # (E, F_edge)


class ObservationBuilder:
    """Builds fixed-size observation vectors and graph views."""

    NODE_TOKEN, NODE_POOL, NODE_MARKET, NODE_CLUSTER, NODE_FOCAL = 0, 1, 2, 3, 4

    def __init__(self, cfg: EnvConfig, n_clusters: int = 8) -> None:
        self.cfg = cfg
        self.n_clusters = int(n_clusters)
        self._init_prices = np.asarray(cfg.init_prices, dtype=np.float64)
        self._market_names = self._build_market_names()
        self._agent_names = self._build_agent_names()
        self._edge_index, self._edge_static = self._build_graph_skeleton()

    # ------------------------------------------------------------------
    # names / dimensions
    # ------------------------------------------------------------------
    def _build_market_names(self) -> List[str]:
        cfg = self.cfg
        names: List[str] = []
        for k in range(cfg.n_tokens):
            if k == 0:
                continue
            t = cfg.token_names[k]
            names += [f"logprice_{t}", f"ret1_{t}", f"basis_{t}", f"rvol_{t}"]
        for p, (ta, tb) in enumerate(cfg.pools):
            tag = f"{cfg.token_names[ta]}_{cfg.token_names[tb]}"
            names += [f"pool_logtvl_{tag}", f"pool_fee_{tag}", f"pool_move_{tag}",
                      f"pool_imbal_{tag}"]
        for k in cfg.lend_tokens:
            t = cfg.token_names[k]
            names += [f"util_{t}", f"borrow_rate_{t}", f"supply_rate_{t}",
                      f"log_supply_{t}", f"log_borrow_{t}"]
        names += [
            "log_gas_base_fee", "gas_vs_baseline", "block_gas_frac", "staked_frac", "staking_rate",
            "n_live_proposals", "log_total_tvl", "n_liquidations_prev",
            "frac_unhealthy", "agg_leverage", "bad_debt_frac", "depeg_dev",
            "time_frac", "pop_cash_frac", "pop_lev_mean",
        ]
        return names

    def _build_agent_names(self) -> List[str]:
        cfg = self.cfg
        names = ["log_net_worth", "pnl_frac", "inv_health", "borrow_cap_frac",
                 "cash_frac", "lp_frac", "stake_frac", "leverage"]
        names += [f"bal_frac_{cfg.token_names[k]}" for k in range(cfg.n_tokens)]
        names += [f"sup_frac_{cfg.token_names[k]}" for k in cfg.lend_tokens]
        names += [f"bor_frac_{cfg.token_names[k]}" for k in cfg.lend_tokens]
        names += [f"lpv_frac_{p}" for p in range(cfg.n_pools)]
        return names

    @property
    def market_names(self) -> List[str]:
        return list(self._market_names)

    @property
    def agent_names(self) -> List[str]:
        return list(self._agent_names)

    @property
    def feature_names(self) -> List[str]:
        return self._market_names + self._agent_names

    @property
    def market_dim(self) -> int:
        return len(self._market_names)

    @property
    def agent_dim(self) -> int:
        return len(self._agent_names)

    @property
    def obs_dim(self) -> int:
        return self.market_dim + self.agent_dim

    # ------------------------------------------------------------------
    # feature extraction
    # ------------------------------------------------------------------
    def market_features(self, world) -> np.ndarray:
        cfg, st = self.cfg, world.state
        ph = st.price_history
        out: List[float] = []

        amm = np.array([world.amm_usd_price(k) for k in range(cfg.n_tokens)])
        if ph is not None and ph.shape[0] >= 2:
            logs = _safe_log(ph)
            diffs = np.diff(logs, axis=0)
            ret1 = diffs[-1]
            rvol = diffs.std(axis=0)
        else:
            ret1 = np.zeros(cfg.n_tokens)
            rvol = np.zeros(cfg.n_tokens)

        for k in range(cfg.n_tokens):
            if k == 0:
                continue
            out += [
                float(_safe_log(st.oracle_price[k] / max(self._init_prices[k], EPS))),
                float(ret1[k]),
                float(_safe_log(max(amm[k], EPS) / max(st.oracle_price[k], EPS))),
                float(rvol[k]),
            ]

        pool_tvl = st.pool_value_usd(pools=cfg.pools)
        init_depth = np.asarray(cfg.pool_depth_usd) * cfg.liquidity_scale
        for p, (ta, tb) in enumerate(cfg.pools):
            x, y = st.reserves[p]
            va, vb = x * st.oracle_price[ta], y * st.oracle_price[tb]
            imbal = (va - vb) / max(va + vb, EPS)
            move = float(ret1[ta] - ret1[tb]) if ph is not None else 0.0
            out += [
                float(_safe_log(max(pool_tvl[p], EPS) / max(init_depth[p], EPS))),
                float(st.pool_fee_bps[p] / 100.0),
                float(move),
                float(imbal),
            ]

        util = st.utilization()
        from .protocols import borrow_rate as _br
        br = _br(util, cfg.base_rate, cfg.slope1, cfg.slope2, cfg.kink)
        sr = br * util * (1.0 - cfg.reserve_factor)
        sup_usd = st.total_supply * st.supply_index * st.oracle_price
        bor_usd = st.total_borrow * st.borrow_index * st.oracle_price
        for k in cfg.lend_tokens:
            out += [
                float(util[k]), float(br[k]), float(sr[k]),
                float(_safe_log(1.0 + sup_usd[k]) / 15.0),
                float(_safe_log(1.0 + bor_usd[k]) / 15.0),
            ]

        nw = world.net_worth()
        hf = world.health_factor()
        tot_sup, tot_bor = float(sup_usd.sum()), float(bor_usd.sum())
        gov_supply = float(st.balances[:, cfg.gov_token].sum()
                           + st.total_staked * st.staking_index)
        cash = float(st.balances[:, 0].sum())
        tot_nw = float(np.maximum(nw, 0.0).sum())
        live = sum(1 for p in st.proposals
                   if not p.executed and not p.rejected and p.closes_at > st.t)
        out += [
            float(_safe_log(st.gas_base_fee / max(world._init_base_fee, EPS))),
            float(_safe_log(st.gas_base_fee / max(st.gas_baseline, EPS))),
            float(st.last_block_gas / max(cfg.block_gas_limit, 1)),
            float(st.total_staked * st.staking_index / max(gov_supply, EPS)),
            float(st.params.get("staking_reward_rate", cfg.staking_reward_rate)),
            float(live) / 4.0,
            float(_safe_log(max(world.total_tvl(), EPS) / max(world._init_tvl, EPS))),
            float(min(world._last_liquidations, 20)) / 20.0,
            float(np.mean(hf < 1.0)),
            float(tot_bor / max(tot_sup, EPS)),
            float((st.bad_debt * st.oracle_price).sum() / max(world._init_tvl, EPS)),
            float(abs(st.oracle_price[0] - 1.0)),
            float(st.t / max(cfg.episode_length, 1)),
            float(cash / max(tot_nw, EPS)),
            float(tot_bor / max(tot_nw, EPS)),
        ]
        return np.asarray(out, dtype=np.float32)

    def agent_features(self, world, agent: int) -> np.ndarray:
        cfg, st = self.cfg, world.state
        price = st.oracle_price
        nw_vec = world.net_worth()
        nw = float(nw_vec[agent])
        scale = max(abs(nw), 1.0)
        hf = float(world.health_factor()[agent])
        from .protocols import borrow_capacity

        bal_val = st.balances[agent] * price
        sup_val = st.supplied[agent] * st.supply_index * price
        bor_val = st.borrowed[agent] * st.borrow_index * price
        lp_val = st.lp_value_usd(cfg.pools, price)[agent]
        stake_val = float(st.staked[agent] * st.staking_index * price[cfg.gov_token])
        prev = world._init_net_worth[agent] if world._init_net_worth is not None else nw
        pnl = float(nw - (st.net_worth_prev[agent] if st.net_worth_prev is not None else nw))

        out = [
            float(_safe_log(max(nw, 1.0) / max(abs(prev), 1.0))),
            float(np.clip(pnl / scale, -5.0, 5.0)),
            float(1.0 / (1.0 + max(hf, 0.0)) if np.isfinite(hf) else 0.0),
            float(np.clip(borrow_capacity(st, agent, world.collateral_factor) / scale, 0, 10)),
            float(bal_val[0] / scale),
            float(lp_val.sum() / scale),
            float(stake_val / scale),
            float(bor_val.sum() / max(sup_val.sum() + bal_val.sum(), EPS)),
        ]
        out += [float(bal_val[k] / scale) for k in range(cfg.n_tokens)]
        out += [float(sup_val[k] / scale) for k in cfg.lend_tokens]
        out += [float(bor_val[k] / scale) for k in cfg.lend_tokens]
        out += [float(lp_val[p] / scale) for p in range(cfg.n_pools)]
        return np.asarray(out, dtype=np.float32)

    def observe(self, world, agent: int) -> np.ndarray:
        return np.concatenate([self.market_features(world),
                               self.agent_features(world, agent)]).astype(np.float32)

    # ------------------------------------------------------------------
    # graph view
    # ------------------------------------------------------------------
    def _build_graph_skeleton(self) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        K, P, L, C = cfg.n_tokens, cfg.n_pools, cfg.n_lend, self.n_clusters
        tok0, pool0, mkt0, clus0 = 0, K, K + P, K + P + L
        focal = clus0 + C
        src: List[int] = []
        dst: List[int] = []
        etype: List[int] = []

        def add(u: int, v: int, e: int) -> None:
            src.append(u); dst.append(v); etype.append(e)
            src.append(v); dst.append(u); etype.append(e)

        for p, (ta, tb) in enumerate(cfg.pools):
            add(tok0 + ta, pool0 + p, 0)
            add(tok0 + tb, pool0 + p, 0)
        for j, k in enumerate(cfg.lend_tokens):
            add(tok0 + k, mkt0 + j, 1)
        for c in range(C):
            for p in range(P):
                add(clus0 + c, pool0 + p, 2)
            for j in range(L):
                add(clus0 + c, mkt0 + j, 3)
        for p in range(P):
            add(focal, pool0 + p, 4)
        for j in range(L):
            add(focal, mkt0 + j, 5)
        for c in range(C):
            add(focal, clus0 + c, 6)
        edge_index = np.asarray([src, dst], dtype=np.int64)
        edge_static = np.eye(7, dtype=np.float32)[np.asarray(etype, dtype=np.int64)]
        return edge_index, edge_static

    @property
    def n_nodes(self) -> int:
        cfg = self.cfg
        return cfg.n_tokens + cfg.n_pools + cfg.n_lend + self.n_clusters + 1

    @property
    def node_feat_dim(self) -> int:
        return 8

    @property
    def edge_feat_dim(self) -> int:
        return 7

    def graph(self, world, agent: int, agent_types: Optional[np.ndarray] = None
              ) -> GraphObservation:
        """Heterogeneous temporal-graph snapshot at the current block."""
        cfg, st = self.cfg, world.state
        K, P, L, C = cfg.n_tokens, cfg.n_pools, cfg.n_lend, self.n_clusters
        N, F = self.n_nodes, self.node_feat_dim
        nf = np.zeros((N, F), dtype=np.float32)
        nt = np.zeros(N, dtype=np.int64)
        price = st.oracle_price
        util = st.utilization()

        for k in range(K):
            nf[k, 0] = float(_safe_log(price[k] / max(self._init_prices[k], EPS)))
            nf[k, 1] = float(_safe_log(max(world.amm_usd_price(k), EPS) / max(price[k], EPS)))
            nf[k, 2] = float(st.balances[:, k].sum() * price[k] / max(world._init_tvl, EPS))
            nf[k, 3] = float(util[k])
            nt[k] = self.NODE_TOKEN

        pool_tvl = st.pool_value_usd(pools=cfg.pools)
        init_depth = np.asarray(cfg.pool_depth_usd) * cfg.liquidity_scale
        for p in range(P):
            i = K + p
            nf[i, 0] = float(_safe_log(max(pool_tvl[p], EPS) / max(init_depth[p], EPS)))
            nf[i, 1] = float(st.pool_fee_bps[p] / 100.0)
            nf[i, 2] = float(st.lp_shares[:, p].sum() / max(st.pool_shares[p], EPS))
            nf[i, 3] = float(pool_tvl[p] / max(world._init_tvl, EPS))
            nt[i] = self.NODE_POOL

        sup_usd = st.total_supply * st.supply_index * price
        bor_usd = st.total_borrow * st.borrow_index * price
        for j, k in enumerate(cfg.lend_tokens):
            i = K + P + j
            nf[i, 0] = float(util[k])
            nf[i, 1] = float(_safe_log(1.0 + sup_usd[k]) / 15.0)
            nf[i, 2] = float(_safe_log(1.0 + bor_usd[k]) / 15.0)
            nf[i, 3] = float(world.collateral_factor[k])
            nt[i] = self.NODE_MARKET

        types = agent_types if agent_types is not None else np.zeros(cfg.n_agents, dtype=int)
        nw = world.net_worth()
        hf = world.health_factor()
        for c in range(C):
            i = K + P + L + c
            m = (np.asarray(types) % C) == c
            nt[i] = self.NODE_CLUSTER
            if not np.any(m):
                continue
            nf[i, 0] = float(m.sum()) / max(cfg.n_agents, 1)
            nf[i, 1] = float(np.maximum(nw[m], 0).sum() / max(world._init_tvl, EPS))
            nf[i, 2] = float(np.mean(hf[m] < 1.0))
            nf[i, 3] = float((st.balances[m, 0]).sum() / max(world._init_tvl, EPS))
            nf[i, 4] = float((st.borrowed[m] * st.borrow_index * price).sum()
                             / max(world._init_tvl, EPS))
            nf[i, 5] = float(st.lp_shares[m].sum() / max(st.pool_shares.sum(), EPS))

        fi = N - 1
        nt[fi] = self.NODE_FOCAL
        af = self.agent_features(world, agent)
        nf[fi, : min(F, af.size)] = af[: min(F, af.size)]
        return GraphObservation(node_feat=nf, node_type=nt,
                                edge_index=self._edge_index, edge_feat=self._edge_static)
