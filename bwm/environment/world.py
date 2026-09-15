"""The blockchain world: ``S_t + A_t -> S_{t+1}, R_t, E_t``.

Determinism contract
--------------------
1. All exogenous randomness is drawn from *counter-based* streams keyed by
   ``(base_seed, stream_name, block)`` -- see :mod:`bwm.utils.seeding`.  The
   noise at block ``t`` therefore does not depend on how many random numbers
   were consumed before ``t``.
2. Scripted agent policies are pure functions of ``(state, block, agent_id,
   seed)``.
3. Consequently ``world.fork()`` followed by *different* actions isolates the
   causal effect of those actions: both branches see identical world noise.

That contract is what makes the counterfactual experiments in this lab
meaningful rather than a measurement of RNG drift.

Genesis liquidity convention
----------------------------
Pools and lending markets are seeded with protocol-owned liquidity that no
agent holds a claim on (locked LP shares / protocol supply).  This keeps market
depth realistic without distorting per-agent wealth accounting.  Gas fees
(base fee *and* tip) are burned rather than paid to a validator agent, so no
agent can profit from ordering; this is stated explicitly because it removes a
whole class of MEV strategies from the world.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np

from ..utils.seeding import SeedSequenceFactory
from .config import EnvConfig
from .protocols import (
    accrue_interest,
    add_liquidity,
    apply_swap,
    borrow_capacity,
    liquidation_amounts,
    remove_liquidity,
    spot_price,
)
from .state import EPS, Proposal, WorldState
from .types import Action, ActionType, EventType, N_EVENTS, Receipt, StepResult

__all__ = ["BlockchainWorld", "PopulationProtocol"]


class PopulationProtocol(Protocol):
    """Duck-typed interface the world uses to obtain scripted agent actions."""

    def act(self, world: "BlockchainWorld", exclude: Sequence[int]) -> List[Action]:
        ...


class BlockchainWorld:
    """A deterministic, configurable DeFi economy."""

    def __init__(self, cfg: EnvConfig, population: Optional[PopulationProtocol] = None) -> None:
        self.cfg = cfg
        self.population = population
        self.seeds = SeedSequenceFactory(cfg.seed)
        self.state: WorldState = None  # type: ignore[assignment]
        self._cf = np.zeros(cfg.n_tokens)          # collateral factor vector (mutable by gov)
        self.trace: List[Dict[str, Any]] = []      # per-block replay log
        self.record_trace: bool = False
        self._init_tvl: float = 1.0
        self._init_base_fee: float = cfg.init_base_fee
        self._init_net_worth: Optional[np.ndarray] = None
        self._last_liquidations: int = 0
        self.reset()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> WorldState:
        cfg = self.cfg
        if seed is not None:
            cfg.seed = int(seed)
        self.seeds = SeedSequenceFactory(cfg.seed)
        K, P, A = cfg.n_tokens, cfg.n_pools, cfg.n_agents
        rng = self.seeds.rng("genesis")

        price = np.asarray(cfg.init_prices, dtype=np.float64).copy()

        # --- accounts ---------------------------------------------------
        wealth = cfg.init_wealth_median * np.exp(
            rng.normal(0.0, cfg.init_wealth_sigma, size=A))
        balances = np.zeros((A, K), dtype=np.float64)
        risk_frac = np.clip(rng.normal(cfg.init_risk_fraction, 0.12, size=A), 0.0, 0.9)
        volatile = [k for k in range(K) if k != 0]
        for a in range(A):
            balances[a, 0] = wealth[a] * (1.0 - risk_frac[a])
            if volatile:
                w = rng.dirichlet(np.full(len(volatile), 2.0))
                for j, k in enumerate(volatile):
                    balances[a, k] = wealth[a] * risk_frac[a] * w[j] / price[k]

        # --- pools (protocol-owned genesis liquidity) -------------------
        reserves = np.zeros((P, 2), dtype=np.float64)
        pool_shares = np.zeros(P, dtype=np.float64)
        for p, (ta, tb) in enumerate(cfg.pools):
            depth = cfg.pool_depth_usd[p] * cfg.liquidity_scale
            reserves[p, 0] = (depth / 2.0) / price[ta]
            reserves[p, 1] = (depth / 2.0) / price[tb]
            pool_shares[p] = float(np.sqrt(reserves[p, 0] * reserves[p, 1]))

        # --- lending (protocol-owned seed supply) -----------------------
        total_supply = np.zeros(K, dtype=np.float64)
        for k in cfg.lend_tokens:
            total_supply[k] = (cfg.lending_seed_usd * cfg.liquidity_scale) / price[k]

        params = {
            "amm_fee_bps": float(cfg.amm_fee_bps),
            "liquidation_bonus": float(cfg.liquidation_bonus),
            "staking_reward_rate": float(cfg.staking_reward_rate),
        }
        for k in range(K):
            params[f"collateral_factor_{k}"] = float(cfg.collateral_factor.get(k, 0.0))

        st = WorldState(
            t=0,
            balances=balances,
            supplied=np.zeros((A, K)),
            borrowed=np.zeros((A, K)),
            lp_shares=np.zeros((A, P)),
            staked=np.zeros(A),
            reserves=reserves,
            pool_shares=pool_shares,
            pool_fee_bps=np.full(P, float(cfg.amm_fee_bps)),
            supply_index=np.ones(K),
            borrow_index=np.ones(K),
            total_supply=total_supply,
            total_borrow=np.zeros(K),
            protocol_reserves=np.zeros(K),
            bad_debt=np.zeros(K),
            staking_index=1.0,
            total_staked=0.0,
            oracle_price=price.copy(),
            fundamental=price.copy(),
            gas_base_fee=float(cfg.init_base_fee),
            gas_baseline=float(cfg.init_base_fee),
            last_block_gas=0,
            regime=int(cfg.init_regime),
            fees_burned=0.0,
            params=params,
            price_history=np.tile(price, (max(cfg.oracle_twap_window, 1), 1)),
        )
        self.state = st
        self._sync_collateral_factor()
        st.net_worth_prev = self.net_worth()
        self._init_net_worth = st.net_worth_prev.copy()
        self._init_tvl = float(max(self.total_tvl(), 1.0))
        self._init_base_fee = float(cfg.init_base_fee)
        self._last_liquidations = 0
        self.trace = []
        if self.population is not None and hasattr(self.population, "reset"):
            self.population.reset(self)
        return st

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _sync_collateral_factor(self) -> None:
        cf = np.zeros(self.cfg.n_tokens)
        for k in range(self.cfg.n_tokens):
            cf[k] = self.state.params.get(f"collateral_factor_{k}",
                                          self.cfg.collateral_factor.get(k, 0.0))
        self._cf = cf

    @property
    def collateral_factor(self) -> np.ndarray:
        return self._cf

    def net_worth(self) -> np.ndarray:
        return self.state.net_worth(self.cfg.pools, self.cfg.gov_token)

    def health_factor(self) -> np.ndarray:
        return self.state.health_factor(self._cf)

    def total_tvl(self) -> float:
        st = self.state
        pool_tvl = float(st.pool_value_usd(pools=self.cfg.pools).sum())
        lend_tvl = float((st.total_supply * st.supply_index * st.oracle_price).sum())
        return pool_tvl + lend_tvl

    def amm_usd_price(self, token: int) -> float:
        """AMM-implied USD price of ``token`` (1.0 for the numeraire)."""
        if token == 0:
            return 1.0
        for p, (ta, tb) in enumerate(self.cfg.pools):
            if ta == token and tb == 0:
                return spot_price(self.state.reserves, p)
            if tb == token and ta == 0:
                sp = spot_price(self.state.reserves, p)
                return 1.0 / sp if sp > EPS else 0.0
        return float(self.state.fundamental[token])

    def fork(self) -> "BlockchainWorld":
        """Independent copy of the world, sharing the same exogenous noise."""
        clone = object.__new__(BlockchainWorld)
        clone.cfg = self.cfg
        clone.population = self.population
        clone.seeds = SeedSequenceFactory(self.cfg.seed)
        clone.state = self.state.copy()
        clone._cf = self._cf.copy()
        clone.trace = []
        clone.record_trace = False
        clone._init_tvl = self._init_tvl
        clone._init_base_fee = self._init_base_fee
        clone._init_net_worth = (None if self._init_net_worth is None
                                 else self._init_net_worth.copy())
        clone._last_liquidations = int(self._last_liquidations)
        return clone

    def snapshot(self) -> WorldState:
        return self.state.copy()

    def restore(self, snap: WorldState) -> None:
        self.state = snap.copy()
        self._sync_collateral_factor()

    # ------------------------------------------------------------------
    # exogenous dynamics
    # ------------------------------------------------------------------
    def _regime_step(self, t: int) -> None:
        cfg = self.cfg
        if cfg.forced_regime_schedule:
            sched = cfg.forced_regime_schedule
            self.state.regime = int(sched[t % len(sched)])
            return
        rng = self.seeds.rng("regime", t)
        row = np.asarray(cfg.regime_transition[self.state.regime], dtype=np.float64)
        self.state.regime = int(rng.choice(len(row), p=row / row.sum()))

    def _price_step(self, t: int) -> np.ndarray:
        """Advance fundamental prices; returns per-token log returns."""
        cfg, st = self.cfg, self.state
        from .types import Regime

        rp = cfg.regimes[Regime(st.regime).name]
        rng = self.seeds.rng("price", t)
        K = cfg.n_tokens
        dt = cfg.dt_years
        common = rng.normal()
        idio = rng.normal(size=K)
        rho = np.clip(cfg.market_corr, 0.0, 0.999)
        z = rho * common + np.sqrt(1.0 - rho ** 2) * idio

        logret = np.zeros(K)
        for k in range(K):
            if k == 0:
                continue  # numeraire is pegged unless a depeg shock fires
            # GOV is a lower-beta asset; ALT-style tokens are higher beta.
            beta = 1.0 if k != cfg.gov_token else 0.7
            logret[k] = (rp.mu * beta - 0.5 * (rp.sigma * beta) ** 2) * dt \
                + rp.sigma * beta * np.sqrt(dt) * z[k]
            if rng.random() < rp.jump_lambda:
                logret[k] += rng.normal(rp.jump_mu, rp.jump_sigma)
        st.fundamental = np.maximum(st.fundamental * np.exp(logret), 1e-8)
        # Slow mean reversion of the peg toward 1.0 after a depeg.
        st.fundamental[0] += (1.0 - st.fundamental[0]) * 0.02
        return logret

    def _apply_shocks(self, t: int) -> Dict[str, Any]:
        """Sample and apply exogenous shocks.  Returns a description dict."""
        cfg, st = self.cfg, self.state
        fired: Dict[str, Any] = {}
        if not cfg.shock_enabled:
            return fired
        from .types import Regime

        rng = self.seeds.rng("shock", t)
        mult = cfg.shock_crisis_multiplier if st.regime == int(Regime.CRISIS) else 1.0
        mag = cfg.shock_magnitude

        if rng.random() < cfg.shock_probs.get("liquidity_withdrawal", 0.0) * mult:
            p = int(rng.integers(cfg.n_pools))
            frac = float(np.clip(0.15 + 0.35 * rng.random(), 0.0, 0.9)) * mag
            frac = min(frac, 0.9)
            holders = np.where(st.lp_shares[:, p] > 0)[0]
            for a in holders:
                remove_liquidity(st, p, int(a), st.lp_shares[a, p] * frac)
            # Protocol-owned (locked) liquidity flees too: burn reserves pro-rata.
            locked = float(st.pool_shares[p] - st.lp_shares[:, p].sum())
            if locked > EPS and st.pool_shares[p] > EPS:
                burn = locked * frac
                r = burn / st.pool_shares[p]
                st.reserves[p] *= (1.0 - r)
                st.pool_shares[p] -= burn
            fired["liquidity_withdrawal"] = {"pool": p, "frac": frac}

        if rng.random() < cfg.shock_probs.get("whale_dump", 0.0) * mult:
            k = int(rng.choice([i for i in range(cfg.n_tokens) if i != 0]))
            pool = self._pool_for(k, 0)
            if pool is not None:
                holder = int(np.argmax(st.balances[:, k]))
                amt = float(st.balances[holder, k]) * float(0.3 + 0.5 * rng.random()) * mag
                if amt > 0:
                    side = 0 if cfg.pools[pool][0] == k else 1
                    dy, _, move = apply_swap(st, pool, side, amt)
                    st.balances[holder, k] -= amt
                    other = cfg.pools[pool][1 - side]
                    st.balances[holder, other] += dy
                    fired["whale_dump"] = {"token": k, "amount": amt, "move": move}

        if rng.random() < cfg.shock_probs.get("depeg", 0.0) * mult:
            depth = float(0.02 + 0.10 * rng.random()) * mag
            st.fundamental[0] = max(1.0 - depth, 0.05)
            fired["depeg"] = {"depth": depth}

        if rng.random() < cfg.shock_probs.get("protocol_exploit", 0.0) * mult:
            p = int(rng.integers(cfg.n_pools))
            frac = float(0.05 + 0.25 * rng.random()) * mag
            st.reserves[p] *= (1.0 - min(frac, 0.9))
            fired["protocol_exploit"] = {"pool": p, "frac": frac}

        if rng.random() < cfg.shock_probs.get("gas_spike", 0.0) * mult:
            f = float(2.0 + 6.0 * rng.random()) * mag
            st.gas_base_fee *= f
            fired["gas_spike"] = {"factor": f}

        glitch = np.ones(cfg.n_tokens)
        if rng.random() < cfg.shock_probs.get("oracle_glitch", 0.0) * mult:
            k = int(rng.choice([i for i in range(cfg.n_tokens) if i != 0]))
            off = float(rng.normal(0.0, 0.05 * mag))
            glitch[k] = max(1.0 + off, 0.2)
            fired["oracle_glitch"] = {"token": k, "offset": off}
        fired["_glitch"] = glitch
        return fired

    def _pool_for(self, a: int, b: int) -> Optional[int]:
        for p, (ta, tb) in enumerate(self.cfg.pools):
            if (ta, tb) == (a, b) or (ta, tb) == (b, a):
                return p
        return None

    def _update_oracle(self, glitch: np.ndarray) -> None:
        cfg, st = self.cfg, self.state
        amm = np.array([self.amm_usd_price(k) for k in range(cfg.n_tokens)])
        amm[0] = st.fundamental[0]
        if st.price_history is not None:
            st.price_history = np.roll(st.price_history, -1, axis=0)
            st.price_history[-1] = amm
        if cfg.oracle_source == "amm_twap" and st.price_history is not None:
            base = st.price_history.mean(axis=0)
            base[0] = st.fundamental[0]
        else:
            base = st.fundamental.copy()
        st.oracle_price = np.maximum(base * glitch, 1e-8)

    def _update_dynamic_fee(self, logret: np.ndarray) -> None:
        """Novel mechanism (OOD only): AMM fee reacts to realised volatility."""
        if not self.cfg.enable_dynamic_fee:
            return
        vol = float(np.abs(logret).max())
        base = float(self.state.params["amm_fee_bps"])
        self.state.pool_fee_bps[:] = np.clip(base * (1.0 + 300.0 * vol), 1.0, 300.0)

    # ------------------------------------------------------------------
    # action execution
    # ------------------------------------------------------------------
    def _charge_gas(self, act: Action) -> Optional[float]:
        """Deduct the gas fee in USD; ``None`` if the account cannot pay."""
        st = self.state
        gas = act.gas()
        if gas == 0:
            return 0.0
        fee = gas * (st.gas_base_fee + max(act.tip, 0.0))
        if st.balances[act.agent, 0] < fee:
            return None
        st.balances[act.agent, 0] -= fee
        st.fees_burned += fee
        return float(fee)

    def _execute(self, act: Action) -> Receipt:
        cfg, st = self.cfg, self.state
        a = int(act.agent)
        at = ActionType(int(act.atype))
        frac = float(np.clip(act.frac, 0.0, 1.0))
        detail: Dict[str, Any] = {}

        if at == ActionType.NOOP:
            return Receipt(act, True, "noop", 0, 0.0)

        fee = self._charge_gas(act)
        if fee is None:
            return Receipt(act, False, "insufficient_gas", 0, 0.0)
        gas = act.gas()

        def fail(reason: str) -> Receipt:
            return Receipt(act, False, reason, gas, fee)

        try:
            if at == ActionType.SWAP:
                p = int(act.target) % cfg.n_pools
                side = int(act.side) % 2
                tok_in = cfg.pools[p][side]
                amt = st.balances[a, tok_in] * frac
                if amt <= 1e-12:
                    return fail("zero_size")
                dy, swap_fee, move = apply_swap(st, p, side, amt)
                if dy <= 0.0:
                    return fail("no_output")
                st.balances[a, tok_in] -= amt
                st.balances[a, cfg.pools[p][1 - side]] += dy
                detail = {"pool": p, "side": side, "amount_in": amt,
                          "amount_out": dy, "move": move}

            elif at == ActionType.ADD_LIQUIDITY:
                p = int(act.target) % cfg.n_pools
                ta, tb = cfg.pools[p]
                amt_b = st.balances[a, tb] * frac
                x, y = st.reserves[p]
                if amt_b <= 1e-12 or y <= EPS:
                    return fail("zero_size")
                need_a = x * (amt_b / y)
                if st.balances[a, ta] < need_a:      # rescale to the binding side
                    if x <= EPS:
                        return fail("empty_pool")
                    amt_b = st.balances[a, ta] * (y / x)
                    need_a = st.balances[a, ta]
                if amt_b <= 1e-12:
                    return fail("zero_size")
                used_a, used_b, minted = add_liquidity(st, p, a, amt_b)
                if minted <= 0.0:
                    return fail("no_shares")
                st.balances[a, ta] -= used_a
                st.balances[a, tb] -= used_b
                detail = {"pool": p, "used_a": used_a, "used_b": used_b, "shares": minted}

            elif at == ActionType.REMOVE_LIQUIDITY:
                p = int(act.target) % cfg.n_pools
                shares = st.lp_shares[a, p] * frac
                if shares <= 1e-14:
                    return fail("no_position")
                out_a, out_b = remove_liquidity(st, p, a, shares)
                ta, tb = cfg.pools[p]
                st.balances[a, ta] += out_a
                st.balances[a, tb] += out_b
                detail = {"pool": p, "out_a": out_a, "out_b": out_b}

            elif at == ActionType.SUPPLY:
                k = int(act.target) % cfg.n_tokens
                if k not in cfg.lend_tokens:
                    return fail("no_market")
                amt = st.balances[a, k] * frac
                if amt <= 1e-12:
                    return fail("zero_size")
                scaled = amt / st.supply_index[k]
                st.balances[a, k] -= amt
                st.supplied[a, k] += scaled
                st.total_supply[k] += scaled
                detail = {"token": k, "amount": amt}

            elif at == ActionType.WITHDRAW:
                k = int(act.target) % cfg.n_tokens
                if k not in cfg.lend_tokens:
                    return fail("no_market")
                scaled = st.supplied[a, k] * frac
                amt = scaled * st.supply_index[k]
                if amt <= 1e-12:
                    return fail("no_position")
                free = (st.total_supply[k] - st.total_borrow[k] * st.borrow_index[k]
                        / max(st.supply_index[k], EPS))
                if scaled > max(free, 0.0):
                    return fail("insufficient_liquidity")
                st.supplied[a, k] -= scaled
                st.total_supply[k] -= scaled
                st.balances[a, k] += amt
                if self.health_factor()[a] < 1.0:     # revert if it breaks solvency
                    st.supplied[a, k] += scaled
                    st.total_supply[k] += scaled
                    st.balances[a, k] -= amt
                    return fail("would_be_unhealthy")
                detail = {"token": k, "amount": amt}

            elif at == ActionType.BORROW:
                k = int(act.target) % cfg.n_tokens
                if k not in cfg.lend_tokens:
                    return fail("no_market")
                cap_usd = borrow_capacity(st, a, self._cf)
                amt = (cap_usd / max(st.oracle_price[k], EPS)) * frac
                avail = (st.total_supply[k] * st.supply_index[k]
                         - st.total_borrow[k] * st.borrow_index[k])
                amt = min(amt, max(avail, 0.0) * 0.99)
                if cfg.enable_borrow_cap:
                    tot_sup = float((st.total_supply * st.supply_index * st.oracle_price).sum())
                    tot_bor = float((st.total_borrow * st.borrow_index * st.oracle_price).sum())
                    room = cfg.borrow_cap_frac * tot_sup - tot_bor
                    amt = min(amt, max(room, 0.0) / max(st.oracle_price[k], EPS))
                sup_k = float(st.total_supply[k] * st.supply_index[k])
                bor_k = float(st.total_borrow[k] * st.borrow_index[k])
                room = max(cfg.max_utilization * sup_k - bor_k, 0.0)
                amt = min(amt, room)
                if amt <= 1e-12:
                    return fail("no_capacity")
                scaled = amt / st.borrow_index[k]
                st.borrowed[a, k] += scaled
                st.total_borrow[k] += scaled
                st.balances[a, k] += amt
                detail = {"token": k, "amount": amt}

            elif at == ActionType.REPAY:
                k = int(act.target) % cfg.n_tokens
                debt = st.borrowed[a, k] * st.borrow_index[k]
                amt = min(debt * frac, st.balances[a, k])
                if amt <= 1e-12:
                    return fail("nothing_to_repay")
                scaled = amt / st.borrow_index[k]
                st.borrowed[a, k] -= scaled
                st.total_borrow[k] -= scaled
                st.balances[a, k] -= amt
                detail = {"token": k, "amount": amt}

            elif at == ActionType.STAKE:
                g = cfg.gov_token
                amt = st.balances[a, g] * frac
                if amt <= 1e-12:
                    return fail("zero_size")
                scaled = amt / st.staking_index
                st.balances[a, g] -= amt
                st.staked[a] += scaled
                st.total_staked += scaled
                detail = {"amount": amt}

            elif at == ActionType.UNSTAKE:
                scaled = st.staked[a] * frac
                amt = scaled * st.staking_index
                if amt <= 1e-12:
                    return fail("no_position")
                st.staked[a] -= scaled
                st.total_staked -= scaled
                st.balances[a, cfg.gov_token] += amt
                detail = {"amount": amt}

            elif at == ActionType.VOTE:
                if not cfg.governance_enabled:
                    return fail("governance_disabled")
                live = [p for p in st.proposals
                        if not p.executed and not p.rejected and p.closes_at > st.t]
                if not live:
                    return fail("no_proposal")
                prop = live[int(act.target) % len(live)]
                if a in prop.voters:
                    return fail("already_voted")
                weight = float(st.staked[a] * st.staking_index)
                if weight <= 0.0:
                    return fail("no_voting_power")
                prop.voters.add(a)
                if int(act.side) == 1:
                    prop.yes += weight
                else:
                    prop.no += weight
                detail = {"proposal": prop.pid, "weight": weight, "side": int(act.side)}

            elif at == ActionType.TRANSFER:
                k = int(act.target) % cfg.n_tokens
                to = int(act.side) % cfg.n_agents
                amt = st.balances[a, k] * frac
                if amt <= 1e-12 or to == a:
                    return fail("zero_size")
                st.balances[a, k] -= amt
                st.balances[to, k] += amt
                detail = {"token": k, "to": to, "amount": amt}

            elif at == ActionType.LIQUIDATE:
                hf = self.health_factor()
                victim = int(act.side)
                if victim < 0 or victim >= cfg.n_agents or hf[victim] >= 1.0:
                    cand = np.where(hf < 1.0)[0]
                    if cand.size == 0:
                        return fail("no_target")
                    victim = int(cand[np.argmin(hf[cand])])
                dt_ = int(act.target) % cfg.n_tokens
                debts = st.borrowed[victim] * st.borrow_index
                if debts[dt_] <= 1e-9:
                    dt_ = int(np.argmax(debts))
                if debts[dt_] <= 1e-9:
                    return fail("no_debt")
                colls = st.supplied[victim] * st.supply_index * st.oracle_price * self._cf
                ct = int(np.argmax(colls))
                repay, seize, shortfall = liquidation_amounts(
                    st, victim, dt_, ct, cfg.close_factor,
                    float(st.params["liquidation_bonus"]), self._cf)
                if repay <= 1e-12 or st.balances[a, dt_] < repay:
                    return fail("cannot_fund")
                st.balances[a, dt_] -= repay
                rs = repay / st.borrow_index[dt_]
                st.borrowed[victim, dt_] -= rs
                st.total_borrow[dt_] -= rs
                ss = seize / st.supply_index[ct]
                st.supplied[victim, ct] -= ss
                st.total_supply[ct] -= ss
                st.balances[a, ct] += seize
                if shortfall:
                    resid = st.borrowed[victim, dt_] * st.borrow_index[dt_]
                    if (st.supplied[victim] * st.supply_index * st.oracle_price).sum() < 1e-6 \
                            and resid > 1e-6:
                        st.bad_debt[dt_] += resid
                        st.total_borrow[dt_] -= st.borrowed[victim, dt_]
                        st.borrowed[victim, dt_] = 0.0
                        detail["bad_debt"] = float(resid)
                detail.update({"victim": victim, "repay": repay, "seize": seize,
                               "debt_token": dt_, "coll_token": ct})
            else:
                return fail("unknown_action")
        except FloatingPointError:  # pragma: no cover - defensive
            return fail("numeric_error")

        return Receipt(act, True, "ok", gas, fee, detail)

    # ------------------------------------------------------------------
    # governance & staking
    # ------------------------------------------------------------------
    def _governance_step(self, t: int) -> bool:
        cfg, st = self.cfg, self.state
        if not cfg.governance_enabled:
            return False
        executed = False
        if t > 0 and t % cfg.proposal_interval == 0:
            rng = self.seeds.rng("gov", t)
            names = sorted(cfg.governable)
            name = names[int(rng.integers(len(names)))]
            lo, hi = cfg.governable[name]
            st.proposals.append(Proposal(
                pid=st.next_proposal_id, param=name,
                new_value=float(rng.uniform(lo, hi)),
                created_at=t, closes_at=t + cfg.voting_blocks))
            st.next_proposal_id += 1
        total_power = st.total_staked * st.staking_index
        for prop in st.proposals:
            if prop.executed or prop.rejected or prop.closes_at > t:
                continue
            quorum = cfg.quorum_frac * max(total_power, EPS)
            if prop.yes + prop.no >= quorum and prop.yes > prop.no:
                prop.executed = True
                st.params[prop.param] = float(prop.new_value)
                if prop.param == "amm_fee_bps":
                    st.pool_fee_bps[:] = float(prop.new_value)
                if prop.param.startswith("collateral_factor_"):
                    self._sync_collateral_factor()
                executed = True
            else:
                prop.rejected = True
        # Bound memory: proposals older than two intervals cannot change anything.
        cutoff = t - 4 * cfg.proposal_interval
        st.proposals = [p for p in st.proposals if p.closes_at >= cutoff]
        return executed

    def _staking_step(self) -> None:
        st = self.state
        rate = float(st.params.get("staking_reward_rate", self.cfg.staking_reward_rate))
        st.staking_index *= (1.0 + rate / self.cfg.blocks_per_year)

    # ------------------------------------------------------------------
    # the transition
    # ------------------------------------------------------------------
    def step(self, external_actions: Optional[Dict[int, Action]] = None) -> StepResult:
        """Advance the world by one block.

        ``external_actions`` maps agent id -> action for agents controlled by an
        intelligence system.  Every other agent is driven by the scripted
        population (if one is attached).
        """
        cfg, st = self.cfg, self.state
        t = int(st.t)
        ext = dict(external_actions or {})

        # --- pre-block snapshot for event labelling ---------------------
        price_before = st.oracle_price.copy()
        fundamental_before = st.fundamental.copy()
        tvl_before = self.total_tvl()
        pool_tvl_before = st.pool_value_usd(pools=cfg.pools)
        nw_before = st.net_worth_prev if st.net_worth_prev is not None else self.net_worth()

        # --- 1. exogenous world update ---------------------------------
        self._regime_step(t)
        logret = self._price_step(t)
        shocks = self._apply_shocks(t)
        glitch = shocks.pop("_glitch", np.ones(cfg.n_tokens))
        self._update_dynamic_fee(logret)

        # --- 2. protocol accrual ---------------------------------------
        accrue_interest(st, cfg)
        self._staking_step()
        self._update_oracle(glitch)

        # --- 3. gather actions -----------------------------------------
        actions: List[Action] = []
        if self.population is not None:
            actions.extend(self.population.act(self, exclude=tuple(ext.keys())))
        actions.extend(ext.values())

        # --- 4. mempool ordering (priority fee auction, deterministic) ---
        actions.sort(key=lambda a: (-float(a.tip), int(a.agent), int(a.atype), int(a.target)))
        included: List[Action] = []
        gas_used = 0
        for a in actions:
            g = a.gas()
            if gas_used + g > cfg.block_gas_limit:
                continue
            included.append(a)
            gas_used += g

        # --- 5. execution ----------------------------------------------
        receipts: List[Receipt] = []
        n_liquidations = 0
        max_move = 0.0
        bad_debt_created = 0.0
        for a in included:
            rc = self._execute(a)
            receipts.append(rc)
            if rc.success and rc.action.atype == ActionType.LIQUIDATE:
                n_liquidations += 1
                bad_debt_created += float(rc.detail.get("bad_debt", 0.0))
            if rc.success and rc.action.atype == ActionType.SWAP:
                max_move = max(max_move, abs(float(rc.detail.get("move", 0.0))))
        if "whale_dump" in shocks:
            max_move = max(max_move, abs(float(shocks["whale_dump"].get("move", 0.0))))

        self._last_liquidations = n_liquidations

        # --- 6. governance ---------------------------------------------
        gov_executed = self._governance_step(t)

        # --- 7. gas base fee (EIP-1559) ---------------------------------
        st.last_block_gas = gas_used
        delta = (gas_used - cfg.block_gas_target) / max(cfg.block_gas_target, 1)
        st.gas_base_fee = max(
            st.gas_base_fee * (1.0 + delta / cfg.base_fee_max_change_denom),
            cfg.min_base_fee)
        # Trailing baseline: HIGH_GAS means "expensive *relative to recent
        # conditions*".  Comparing against the genesis fee instead would make the
        # label fire permanently once the fee market found a new equilibrium.
        d = float(cfg.gas_baseline_decay)
        st.gas_baseline = d * st.gas_baseline + (1.0 - d) * st.gas_base_fee

        # --- 8. rewards & events ----------------------------------------
        st.t = t + 1
        nw_after = self.net_worth()
        rewards = nw_after - nw_before
        st.net_worth_prev = nw_after

        events = np.zeros(N_EVENTS, dtype=np.float64)
        events[EventType.LIQUIDATION] = float(n_liquidations > 0)
        events[EventType.CASCADE] = float(n_liquidations >= cfg.cascade_threshold)
        events[EventType.LARGE_SWAP] = float(max_move >= cfg.large_swap_move)
        jump = float(np.max(np.abs(np.log(np.maximum(st.fundamental, 1e-12))
                                   - np.log(np.maximum(fundamental_before, 1e-12)))))
        events[EventType.PRICE_JUMP] = float(jump >= cfg.price_jump_threshold)
        pool_tvl_after = st.pool_value_usd(pools=cfg.pools)
        drop = np.divide(pool_tvl_before - pool_tvl_after, np.maximum(pool_tvl_before, EPS))
        events[EventType.LIQUIDITY_DROP] = float(np.max(drop) >= cfg.liquidity_drop_frac)
        events[EventType.GOV_EXECUTED] = float(gov_executed)
        events[EventType.DEPEG] = float(abs(st.fundamental[0] - 1.0) >= cfg.depeg_band)
        events[EventType.HIGH_GAS] = float(
            st.gas_base_fee >= cfg.high_gas_multiple * max(st.gas_baseline, cfg.min_base_fee))
        events[EventType.INSOLVENCY] = float(bad_debt_created > 0.0)
        util = st.utilization()
        lend_util = util[cfg.lend_tokens] if cfg.n_lend else np.zeros(1)
        events[EventType.UTILIZATION_SPIKE] = float(np.max(lend_util) >= cfg.utilization_spike)

        info: Dict[str, Any] = {
            "regime": int(st.regime),
            "n_liquidations": n_liquidations,
            "gas_used": gas_used,
            "n_included": len(included),
            "n_submitted": len(actions),
            "shocks": {k: v for k, v in shocks.items()},
            "tvl": self.total_tvl(),
            "tvl_before": tvl_before,
            "max_swap_move": max_move,
            "bad_debt": float(st.bad_debt.sum()),
            "n_unhealthy": int(np.sum(self.health_factor() < 1.0)),
            "price_before": price_before,
            "logret": logret,
        }
        result = StepResult(t=t, rewards=rewards, events=events, receipts=receipts, info=info)

        if self.record_trace:
            self.trace.append({
                "t": t,
                "actions": [a.as_tuple() for a in included],
                "events": events.copy(),
                "regime": int(st.regime),
                "digest": st.digest(),
            })
        return result
