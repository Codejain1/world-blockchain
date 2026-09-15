"""The world state ``S_t`` and its (de)serialisation.

``WorldState`` is a plain container of NumPy arrays plus a few scalars.  It is
fully value-typed: ``copy()`` gives an independent world that can be simulated
forward without touching the original.  That is the mechanism behind
counterfactual forks.
"""

from __future__ import annotations

import copy as _copy
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["Proposal", "WorldState"]

EPS = 1e-12


@dataclass(slots=True)
class Proposal:
    """A governance proposal to change one protocol parameter."""

    pid: int
    param: str
    new_value: float
    created_at: int
    closes_at: int
    yes: float = 0.0
    no: float = 0.0
    executed: bool = False
    rejected: bool = False
    voters: set = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: getattr(self, k) for k in
             ("pid", "param", "new_value", "created_at", "closes_at", "yes", "no",
              "executed", "rejected")}
        d["voters"] = sorted(self.voters)
        return d


@dataclass
class WorldState:
    """Complete, self-contained state of the simulated economy at block ``t``."""

    t: int
    # --- accounts -------------------------------------------------------
    balances: np.ndarray        # (A, K) liquid token balances
    supplied: np.ndarray        # (A, K) scaled lending supply (multiply by supply_index)
    borrowed: np.ndarray        # (A, K) scaled lending debt (multiply by borrow_index)
    lp_shares: np.ndarray       # (A, P)
    staked: np.ndarray          # (A,) scaled stake (multiply by staking_index)

    # --- venues ---------------------------------------------------------
    reserves: np.ndarray        # (P, 2) constant-product reserves
    pool_shares: np.ndarray     # (P,) total LP shares outstanding
    pool_fee_bps: np.ndarray    # (P,)

    supply_index: np.ndarray    # (K,)
    borrow_index: np.ndarray    # (K,)
    total_supply: np.ndarray    # (K,) scaled
    total_borrow: np.ndarray    # (K,) scaled
    protocol_reserves: np.ndarray  # (K,) fees retained by the lending protocol
    bad_debt: np.ndarray        # (K,) debt written off (insolvency)

    staking_index: float
    total_staked: float         # scaled

    # --- market & chain -------------------------------------------------
    oracle_price: np.ndarray    # (K,) price used by the protocol
    fundamental: np.ndarray     # (K,) exogenous "true" price
    gas_base_fee: float
    gas_baseline: float          # EMA of the base fee; the reference for HIGH_GAS
    last_block_gas: int
    regime: int                 # HIDDEN from observations by construction
    fees_burned: float

    # --- governance -----------------------------------------------------
    proposals: List[Proposal] = field(default_factory=list)
    params: Dict[str, float] = field(default_factory=dict)
    next_proposal_id: int = 0

    # --- bookkeeping ----------------------------------------------------
    price_history: Optional[np.ndarray] = None   # (W, K) ring buffer for TWAP oracles
    net_worth_prev: Optional[np.ndarray] = None  # (A,) for reward computation

    # -------------------------------------------------------------------
    @property
    def n_agents(self) -> int:
        return int(self.balances.shape[0])

    @property
    def n_tokens(self) -> int:
        return int(self.balances.shape[1])

    @property
    def n_pools(self) -> int:
        return int(self.reserves.shape[0])

    def copy(self) -> "WorldState":
        """Deep, independent copy (arrays are copied, not viewed)."""
        return WorldState(
            t=self.t,
            balances=self.balances.copy(),
            supplied=self.supplied.copy(),
            borrowed=self.borrowed.copy(),
            lp_shares=self.lp_shares.copy(),
            staked=self.staked.copy(),
            reserves=self.reserves.copy(),
            pool_shares=self.pool_shares.copy(),
            pool_fee_bps=self.pool_fee_bps.copy(),
            supply_index=self.supply_index.copy(),
            borrow_index=self.borrow_index.copy(),
            total_supply=self.total_supply.copy(),
            total_borrow=self.total_borrow.copy(),
            protocol_reserves=self.protocol_reserves.copy(),
            bad_debt=self.bad_debt.copy(),
            staking_index=float(self.staking_index),
            total_staked=float(self.total_staked),
            oracle_price=self.oracle_price.copy(),
            fundamental=self.fundamental.copy(),
            gas_base_fee=float(self.gas_base_fee),
            gas_baseline=float(self.gas_baseline),
            last_block_gas=int(self.last_block_gas),
            regime=int(self.regime),
            fees_burned=float(self.fees_burned),
            proposals=[_copy.deepcopy(p) for p in self.proposals],
            params=dict(self.params),
            next_proposal_id=int(self.next_proposal_id),
            price_history=None if self.price_history is None else self.price_history.copy(),
            net_worth_prev=None if self.net_worth_prev is None else self.net_worth_prev.copy(),
        )

    # -- derived quantities ---------------------------------------------
    def supplied_actual(self) -> np.ndarray:
        return self.supplied * self.supply_index[None, :]

    def borrowed_actual(self) -> np.ndarray:
        return self.borrowed * self.borrow_index[None, :]

    def pool_value_usd(self, price: Optional[np.ndarray] = None,
                       pools: Optional[List[tuple]] = None) -> np.ndarray:
        """USD value locked in each pool."""
        price = self.oracle_price if price is None else price
        assert pools is not None
        vals = np.empty(self.n_pools, dtype=np.float64)
        for p, (a, b) in enumerate(pools):
            vals[p] = self.reserves[p, 0] * price[a] + self.reserves[p, 1] * price[b]
        return vals

    def lp_value_usd(self, pools: List[tuple], price: Optional[np.ndarray] = None) -> np.ndarray:
        """(A, P) USD value of each agent's LP position."""
        price = self.oracle_price if price is None else price
        tvl = self.pool_value_usd(price, pools)
        share = np.divide(self.lp_shares, np.maximum(self.pool_shares[None, :], EPS))
        return share * tvl[None, :]

    def net_worth(self, pools: List[tuple], gov_token: int,
                  price: Optional[np.ndarray] = None) -> np.ndarray:
        """Mark-to-market net worth per agent, in USD."""
        price = self.oracle_price if price is None else price
        nw = self.balances @ price
        nw += self.supplied_actual() @ price
        nw -= self.borrowed_actual() @ price
        nw += self.lp_value_usd(pools, price).sum(axis=1)
        nw += self.staked * self.staking_index * price[gov_token]
        return nw

    def health_factor(self, collateral_factor: np.ndarray,
                      price: Optional[np.ndarray] = None) -> np.ndarray:
        """Aave-style health factor; ``inf`` when an account has no debt."""
        price = self.oracle_price if price is None else price
        coll = (self.supplied_actual() * (price * collateral_factor)[None, :]).sum(axis=1)
        debt = (self.borrowed_actual() * price[None, :]).sum(axis=1)
        hf = np.full(self.n_agents, np.inf, dtype=np.float64)
        has = debt > 1e-8
        hf[has] = coll[has] / debt[has]
        return hf

    def utilization(self) -> np.ndarray:
        sup = self.total_supply * self.supply_index
        bor = self.total_borrow * self.borrow_index
        return np.divide(bor, np.maximum(sup, EPS))

    # -- hashing / equality ---------------------------------------------
    def digest(self) -> str:
        """Content hash of the full state -- used by determinism tests."""
        h = hashlib.blake2b(digest_size=16)
        for arr in (self.balances, self.supplied, self.borrowed, self.lp_shares,
                    self.staked, self.reserves, self.pool_shares, self.pool_fee_bps,
                    self.supply_index, self.borrow_index, self.total_supply,
                    self.total_borrow, self.protocol_reserves, self.bad_debt,
                    self.oracle_price, self.fundamental):
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        # NumPy scalars and Python floats repr differently (``np.float64(1.0)``
        # vs ``1.0``), so every scalar is coerced before hashing.  Without this
        # a state and its own copy can hash differently.
        scalars = (
            int(self.t), round(float(self.staking_index), 12),
            round(float(self.total_staked), 6), round(float(self.gas_base_fee), 18),
            round(float(self.gas_baseline), 18), int(self.last_block_gas),
            int(self.regime), round(float(self.fees_burned), 8),
            int(self.next_proposal_id),
            sorted((str(k), round(float(v), 10)) for k, v in self.params.items()),
            [self._proposal_key(p) for p in self.proposals],
        )
        h.update(repr(scalars).encode())
        return h.hexdigest()

    @staticmethod
    def _proposal_key(p: "Proposal") -> tuple:
        return (int(p.pid), str(p.param), round(float(p.new_value), 10),
                int(p.created_at), int(p.closes_at), round(float(p.yes), 6),
                round(float(p.no), 6), bool(p.executed), bool(p.rejected),
                tuple(sorted(int(v) for v in p.voters)))
