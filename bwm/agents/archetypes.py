"""The scripted agent archetypes that make the economy move.

Each archetype implements a recognisable market role.  Together they generate
the phenomena the lab is about: mean-reverting prices (arbitrageurs), endogenous
liquidity (LPs and market makers), leverage cycles and cascades (borrowers and
keepers), governance drift, and deliberate manipulation (adversarial and
coordinated blocs).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..environment.protocols import borrow_capacity, spot_price
from ..environment.state import EPS
from ..environment.types import Action, ActionType, Regime
from .base import AgentSpec, ScriptedPolicy

__all__ = ["ARCHETYPES", "make_policy"]


def _usd_pool(cfg, token: int) -> Optional[int]:
    """Index of the ``token``/USD pool, if one exists."""
    for p, (ta, tb) in enumerate(cfg.pools):
        if (ta, tb) == (token, 0) or (ta, tb) == (0, token):
            return p
    return None


def _pool_price_usd(world, pool: int) -> float:
    """Marginal USD price of the pool's non-numeraire token."""
    cfg = world.cfg
    ta, tb = cfg.pools[pool]
    sp = spot_price(world.state.reserves, pool)
    if tb == 0:
        return sp
    if ta == 0:
        return 1.0 / sp if sp > EPS else 0.0
    return sp * float(world.state.fundamental[tb])


class RetailTrader(ScriptedPolicy):
    """Noise + momentum trader, biased by the (hidden) regime's flow bias."""

    archetype = "retail"

    def act(self, world, rng):
        cfg, st = world.cfg, world.state
        if rng.random() > self.spec.activity:
            return self.noop()
        volatile = [k for k in range(cfg.n_tokens) if k != 0]
        k = int(rng.choice(volatile))
        pool = _usd_pool(cfg, k)
        if pool is None:
            return self.noop()
        bias = cfg.regimes[Regime(st.regime).name].flow_bias
        ph = st.price_history
        mom = 0.0
        if ph is not None and ph.shape[0] >= 2:
            mom = float(np.log(max(ph[-1, k], EPS)) - np.log(max(ph[0, k], EPS)))
        # Retail chases momentum and follows sentiment; risk aversion damps size.
        score = bias + 6.0 * mom + 0.6 * rng.normal()
        buy = score > 0.0
        ta, tb = cfg.pools[pool]
        side = (1 if tb == 0 else 0) if buy else (0 if ta == k else 1)
        frac = float(np.clip(self.spec.aggression * (1.0 - 0.5 * self.spec.risk_aversion)
                             * abs(rng.normal(1.0, 0.4)), 0.01, 0.6))
        return self.make(world, ActionType.SWAP, pool, side, frac)


class Whale(ScriptedPolicy):
    """Large, patient, contrarian capital that trades rarely but decisively."""

    archetype = "whale"

    def act(self, world, rng):
        cfg, st = world.cfg, world.state
        if rng.random() > self.spec.activity:
            return self.noop()
        volatile = [k for k in range(cfg.n_tokens) if k != 0]
        k = int(rng.choice(volatile))
        pool = _usd_pool(cfg, k)
        if pool is None:
            return self.noop()
        amm = _pool_price_usd(world, pool)
        fund = float(st.fundamental[k])
        dev = (amm - fund) / max(fund, EPS)
        if abs(dev) < 0.004 and rng.random() < 0.7:
            return self.noop()
        ta, tb = cfg.pools[pool]
        buy = dev < 0.0     # cheap on-chain -> accumulate
        side = (1 if tb == 0 else 0) if buy else (0 if ta == k else 1)
        frac = float(np.clip(self.spec.aggression * (1.0 + 8.0 * abs(dev)), 0.02, 0.5))
        return self.make(world, ActionType.SWAP, pool, side, frac, tip_scale=2.0)


class Arbitrageur(ScriptedPolicy):
    """Closes the gap between pool price and the external reference price.

    Trade size is computed from the constant-product invariant: to move the
    marginal price to ``p*`` the reserve of the sold token must become
    ``sqrt(k / p*)``.  This makes arbitrage *quantitatively* correct rather than
    a hand-tuned nudge, which is what keeps on-chain prices tethered to the
    exogenous process.
    """

    archetype = "arbitrageur"

    def act(self, world, rng):
        cfg, st = world.cfg, world.state
        if rng.random() > self.spec.activity:
            return self.noop()
        best = None
        for p, (ta, tb) in enumerate(cfg.pools):
            if 0 not in (ta, tb):
                continue
            k = ta if tb == 0 else tb
            amm = _pool_price_usd(world, p)
            fund = float(st.fundamental[k])
            if fund <= EPS or amm <= EPS:
                continue
            dev = (amm - fund) / fund
            fee = float(st.pool_fee_bps[p]) / 10_000.0
            if abs(dev) <= fee * 1.2:      # no edge after fees
                continue
            if best is None or abs(dev) > abs(best[1]):
                best = (p, dev, k)
        if best is None:
            return self.noop()
        p, dev, k = best
        ta, tb = cfg.pools[p]
        x, y = st.reserves[p]                      # x: token_a, y: token_b
        kk = float(x * y)
        pull = 0.5 + 0.4 * (1.0 - self.spec.risk_aversion)   # close most of the gap
        if dev > 0:      # pool price too high -> sell token k into the pool
            sell_tok, sell_idx = k, (0 if ta == k else 1)
            target_a = float(st.fundamental[k]) if tb == 0 else 1.0 / max(
                float(st.fundamental[k]), EPS)
            cur_a = spot_price(st.reserves, p)
            tgt = cur_a + (target_a - cur_a) * pull
            need_x = np.sqrt(kk / max(tgt, EPS)) if sell_idx == 0 else None
            amount = (need_x - x) if sell_idx == 0 else (np.sqrt(kk * max(tgt, EPS)) - y)
        else:            # pool price too low -> buy token k with USD
            sell_tok, sell_idx = 0, (0 if ta == 0 else 1)
            target_a = float(st.fundamental[k]) if tb == 0 else 1.0 / max(
                float(st.fundamental[k]), EPS)
            cur_a = spot_price(st.reserves, p)
            tgt = cur_a + (target_a - cur_a) * pull
            amount = (np.sqrt(kk / max(tgt, EPS)) - x) if sell_idx == 0 else (
                np.sqrt(kk * max(tgt, EPS)) - y)
        amount = float(amount)
        if amount <= 0:
            return self.noop()
        a = self.spec.agent_id
        bal = float(st.balances[a, sell_tok])

        # Unwind any short once the position has done its job.
        debt_here = float(st.borrowed[a, sell_tok] * st.borrow_index[sell_tok])
        if debt_here > 1e-9 and bal > debt_here * 0.5 and abs(dev) < 0.01:
            return self.make(world, ActionType.REPAY, sell_tok, 0, 1.0, tip_scale=3.0)

        # Inventory constraint: an arbitrageur that has sold all of an asset can
        # no longer correct a positive basis.  Real desks solve this by
        # borrowing the asset and shorting it, so we let them do the same --
        # which also couples the AMM and the lending market, a coupling the
        # world model has to learn.
        if bal < 0.5 * amount and sell_tok in cfg.lend_tokens \
                and float(st.utilization()[sell_tok]) < 0.95:
            from ..environment.protocols import borrow_capacity as _cap
            cap = _cap(st, a, world.collateral_factor)
            need_usd = amount * float(st.oracle_price[sell_tok])
            if cap < need_usd * 1.5:
                idle = int(np.argmax(st.balances[a] * st.oracle_price
                                     * world.collateral_factor))
                if st.balances[a, idle] > 1e-9 and world.collateral_factor[idle] > 0:
                    return self.make(world, ActionType.SUPPLY, idle, 0, 0.9, tip_scale=3.0)
            elif cap > 1.0:
                frac_b = float(np.clip(need_usd / max(cap, EPS), 0.02, 0.9))
                return self.make(world, ActionType.BORROW, sell_tok, 0, frac_b, tip_scale=3.0)

        if bal <= EPS:
            return self.noop()
        frac = float(np.clip(amount / bal, 0.0, 1.0))
        if frac < 1e-4:
            return self.noop()
        return self.make(world, ActionType.SWAP, p, sell_idx, frac, tip_scale=4.0)


class MarketMaker(ScriptedPolicy):
    """Supplies liquidity when volatility is low, withdraws when it spikes."""

    archetype = "market_maker"

    def act(self, world, rng):
        cfg, st = world.cfg, world.state
        if rng.random() > self.spec.activity:
            return self.noop()
        ph = st.price_history
        vol = 0.0
        if ph is not None and ph.shape[0] >= 3:
            vol = float(np.abs(np.diff(np.log(np.maximum(ph, EPS)), axis=0)).mean())
        p = int(rng.integers(cfg.n_pools))
        held = float(st.lp_shares[self.spec.agent_id, p])
        stress = vol * 400.0 * (0.5 + self.spec.risk_aversion)
        if stress > 1.0 and held > 0:
            return self.make(world, ActionType.REMOVE_LIQUIDITY, p, 0,
                             float(np.clip(0.3 * stress, 0.05, 0.9)), tip_scale=3.0)
        if stress < 0.6:
            return self.make(world, ActionType.ADD_LIQUIDITY, p, 0,
                             float(np.clip(self.spec.aggression, 0.02, 0.5)))
        return self.noop()


class LiquidityProvider(ScriptedPolicy):
    """Passive LP: deposits steadily, exits on drawdown."""

    archetype = "liquidity_provider"

    def act(self, world, rng):
        cfg, st = world.cfg, world.state
        if rng.random() > self.spec.activity:
            return self.noop()
        p = int(rng.integers(cfg.n_pools))
        nw = float(world.net_worth()[self.spec.agent_id])
        init = float(world._init_net_worth[self.spec.agent_id]) if \
            world._init_net_worth is not None else nw
        drawdown = 1.0 - nw / max(init, EPS)
        if drawdown > 0.15 * (1.0 + self.spec.risk_aversion) and \
                st.lp_shares[self.spec.agent_id, p] > 0:
            return self.make(world, ActionType.REMOVE_LIQUIDITY, p, 0, 0.5, tip_scale=2.0)
        return self.make(world, ActionType.ADD_LIQUIDITY, p, 0,
                         float(np.clip(self.spec.aggression * 0.8, 0.02, 0.4)))


class Borrower(ScriptedPolicy):
    """Levered user: supplies collateral, borrows, deleverages near the edge.

    Risk-seeking borrowers run thin health factors and are the fuel for
    liquidation cascades when a shock lands.
    """

    archetype = "borrower"

    def act(self, world, rng):
        cfg, st = world.cfg, world.state
        if rng.random() > self.spec.activity:
            return self.noop()
        a = self.spec.agent_id
        hf = float(world.health_factor()[a])
        target_hf = 1.25 + 1.2 * self.spec.risk_aversion
        lend = list(cfg.lend_tokens)

        # Rate sensitivity: when utilisation spikes the borrow rate explodes, and
        # levered users deleverage.  Without this the lending market has no
        # negative feedback and utilisation pins at 100%.
        from ..environment.protocols import borrow_rate as _brate
        util = st.utilization()
        rates = _brate(util, cfg.base_rate, cfg.slope1, cfg.slope2, cfg.kink)
        debts_now = st.borrowed[a] * st.borrow_index
        if debts_now.sum() > 1e-9:
            kd = int(np.argmax(debts_now * rates))
            if rates[kd] > 0.25 + 0.5 * (1.0 - self.spec.risk_aversion) \
                    and st.balances[a, kd] > 1e-9:
                return self.make(world, ActionType.REPAY, kd, 0, 0.6, tip_scale=2.0)
        if np.isfinite(hf) and hf < max(target_hf * 0.8, 1.05):
            debts = st.borrowed[a] * st.borrow_index
            k = int(np.argmax(debts))
            if debts[k] > 1e-9 and st.balances[a, k] > 1e-9:
                return self.make(world, ActionType.REPAY, k, 0, 0.5, tip_scale=3.0)
            coll = st.supplied[a] * st.supply_index * st.oracle_price
            j = int(np.argmax(coll))
            if st.balances[a, j] > 1e-9:
                return self.make(world, ActionType.SUPPLY, j, 0, 0.6, tip_scale=3.0)
            return self.noop()
        supplied_usd = float((st.supplied[a] * st.supply_index * st.oracle_price).sum())
        if supplied_usd < 1e-6 or rng.random() < 0.25:
            k = int(rng.choice([t for t in lend if world.collateral_factor[t] > 0])) \
                if any(world.collateral_factor[t] > 0 for t in lend) else lend[0]
            if st.balances[a, k] > 1e-9:
                return self.make(world, ActionType.SUPPLY, k, 0,
                                 float(np.clip(self.spec.aggression + 0.3, 0.05, 0.9)))
        cap = borrow_capacity(st, a, world.collateral_factor)
        if cap > 1.0 and (not np.isfinite(hf) or hf > target_hf):
            k = int(rng.choice(lend))
            frac = float(np.clip(0.9 - 0.6 * self.spec.risk_aversion, 0.05, 0.95))
            return self.make(world, ActionType.BORROW, k, 0, frac)
        # Deploy borrowed cash into the risk asset (the classic leverage loop).
        volatile = [k for k in range(cfg.n_tokens) if k != 0]
        k = int(rng.choice(volatile))
        pool = _usd_pool(cfg, k)
        if pool is not None and st.balances[a, 0] > 1e-6:
            ta, tb = cfg.pools[pool]
            side = 1 if tb == 0 else 0
            return self.make(world, ActionType.SWAP, pool, side, 0.5)
        return self.noop()


class Keeper(ScriptedPolicy):
    """Liquidation bot.  Keeps the protocol solvent and creates cascades."""

    archetype = "keeper"

    def act(self, world, rng):
        st = world.state
        hf = world.health_factor()
        unhealthy = np.where(hf < 1.0)[0]
        if unhealthy.size == 0:
            # Idle keepers hold dry powder in the numeraire.
            if rng.random() < 0.02:
                return self.make(world, ActionType.WITHDRAW, 0, 0, 1.0)
            return self.noop()
        victim = int(unhealthy[np.argmin(hf[unhealthy])])
        debts = st.borrowed[victim] * st.borrow_index
        dt_ = int(np.argmax(debts))
        return self.make(world, ActionType.LIQUIDATE, dt_, victim, 0.0, tip_scale=8.0)


class GovernanceParticipant(ScriptedPolicy):
    """Stakes and votes; preferences depend on its own balance-sheet exposure."""

    archetype = "governance"

    def act(self, world, rng):
        cfg, st = world.cfg, world.state
        a = self.spec.agent_id
        if rng.random() > self.spec.activity:
            return self.noop()
        live = [p for p in st.proposals
                if not p.executed and not p.rejected and p.closes_at > st.t]
        if live and st.staked[a] > 0:
            prop = live[int(rng.integers(len(live)))]
            if a in prop.voters:
                return self.noop()
            debt = float((st.borrowed[a] * st.borrow_index * st.oracle_price).sum())
            lp = float(st.lp_value_usd(cfg.pools)[a].sum())
            cur = float(st.params.get(prop.param, 0.0))
            delta = prop.new_value - cur
            # Borrowers want looser collateral rules; LPs want higher fees.
            if prop.param.startswith("collateral_factor"):
                pref = 1 if (delta > 0) == (debt > 0) else 0
            elif prop.param == "amm_fee_bps":
                pref = 1 if (delta > 0) == (lp > 0) else 0
            elif prop.param == "liquidation_bonus":
                pref = 1 if (delta > 0) == (debt <= 0) else 0
            else:
                pref = int(rng.random() < 0.5)
            return self.make(world, ActionType.VOTE, live.index(prop), pref, 0.0)
        if st.balances[a, cfg.gov_token] > 1e-9:
            return self.make(world, ActionType.STAKE, cfg.gov_token, 0, 0.6)
        return self.noop()


class RandomAgent(ScriptedPolicy):
    """Uniform exploration over the action menu -- keeps the data distribution wide."""

    archetype = "random"

    def act(self, world, rng):
        cfg = world.cfg
        if rng.random() > self.spec.activity:
            return self.noop()
        at = ActionType(int(rng.integers(1, len(ActionType))))
        if at == ActionType.LIQUIDATE:
            at = ActionType.SWAP
        target = int(rng.integers(max(cfg.n_pools, cfg.n_tokens)))
        return self.make(world, at, target, int(rng.integers(2)),
                         float(rng.uniform(0.02, 0.5)))


class Adversarial(ScriptedPolicy):
    """Deliberately hunts liquidations by pushing a thin pool's price.

    When enough accounts sit just above the liquidation threshold, dumping into
    the thinnest pool can push them under and let the attacker (or its allied
    keepers) capture the liquidation bonus.
    """

    archetype = "adversarial"

    def act(self, world, rng):
        cfg, st = world.cfg, world.state
        a = self.spec.agent_id
        if rng.random() > self.spec.activity:
            return self.noop()
        hf = world.health_factor()
        near = np.sum((hf >= 1.0) & (hf < 1.25))
        if near >= 2:
            tvl = st.pool_value_usd(pools=cfg.pools)
            order = np.argsort(tvl)
            for p in order:
                ta, tb = cfg.pools[int(p)]
                if 0 not in (ta, tb):
                    continue
                k = ta if tb == 0 else tb
                if world.collateral_factor[k] <= 0:
                    continue
                side = 0 if ta == k else 1
                if st.balances[a, k] > 1e-9:
                    return self.make(world, ActionType.SWAP, int(p), side,
                                     float(np.clip(self.spec.aggression * 2.0, 0.1, 0.9)),
                                     tip_scale=6.0)
        # Otherwise accumulate ammunition in the collateral asset.
        volatile = [k for k in range(cfg.n_tokens)
                    if k != 0 and world.collateral_factor[k] > 0]
        if volatile and st.balances[a, 0] > 1e-6:
            k = int(rng.choice(volatile))
            pool = _usd_pool(cfg, k)
            if pool is not None:
                ta, tb = cfg.pools[pool]
                return self.make(world, ActionType.SWAP, pool, 1 if tb == 0 else 0, 0.3)
        return self.noop()


class Coordinated(ScriptedPolicy):
    """Member of a bloc that acts as one.

    All members share a *bloc seed*, so the whole group takes the same directional
    decision in the same block.  Individually each trade looks ordinary; together
    they move the market.  Used to create held-out "coordinated agents" worlds.
    """

    archetype = "coordinated"

    def act(self, world, rng):
        cfg, st = world.cfg, world.state
        a = self.spec.agent_id
        bloc = world.seeds.rng("bloc", st.t, int(self.spec.extra.get("bloc", 0)))
        if bloc.random() > float(self.spec.extra.get("bloc_activity", 0.25)):
            return self.noop()
        volatile = [k for k in range(cfg.n_tokens) if k != 0]
        k = int(volatile[int(bloc.integers(len(volatile)))])
        pool = _usd_pool(cfg, k)
        if pool is None:
            return self.noop()
        sell = bloc.random() < 0.5
        ta, tb = cfg.pools[pool]
        side = (0 if ta == k else 1) if sell else (1 if tb == 0 else 0)
        frac = float(np.clip(self.spec.aggression * 1.5, 0.05, 0.8))
        return self.make(world, ActionType.SWAP, pool, side, frac, tip_scale=5.0)


ARCHETYPES = {
    "retail": RetailTrader,
    "whale": Whale,
    "arbitrageur": Arbitrageur,
    "market_maker": MarketMaker,
    "liquidity_provider": LiquidityProvider,
    "borrower": Borrower,
    "keeper": Keeper,
    "governance": GovernanceParticipant,
    "random": RandomAgent,
    "adversarial": Adversarial,
    "coordinated": Coordinated,
}


def make_policy(spec: AgentSpec) -> ScriptedPolicy:
    if spec.archetype not in ARCHETYPES:
        raise KeyError(f"Unknown archetype {spec.archetype!r}; "
                       f"known: {sorted(ARCHETYPES)}")
    return ARCHETYPES[spec.archetype](spec)
