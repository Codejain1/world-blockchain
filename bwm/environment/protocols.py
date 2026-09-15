"""Protocol mechanics: constant-product AMM, lending market, staking.

These are deliberately *textbook* implementations (Uniswap-v2 style pricing,
Aave-style kinked interest with a health factor and a close factor).  Using
well-known mechanisms rather than invented ones means observed model failures
are attributable to the learning problem, not to exotic simulator rules.

All functions are pure with respect to randomness: given a state they are
deterministic.  Stochasticity enters only through the exogenous streams in
:mod:`bwm.environment.world`.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .state import EPS, WorldState

__all__ = [
    "spot_price", "swap_output", "apply_swap", "add_liquidity", "remove_liquidity",
    "borrow_rate", "accrue_interest", "borrow_capacity", "liquidation_amounts",
]


# --------------------------------------------------------------------------
# AMM
# --------------------------------------------------------------------------
def spot_price(reserves: np.ndarray, pool: int) -> float:
    """Marginal price of token_a denominated in token_b (no fee)."""
    x, y = reserves[pool]
    if x <= EPS:
        return 0.0
    return float(y / x)


def swap_output(x_in_reserve: float, y_out_reserve: float, dx: float, fee_bps: float
                ) -> Tuple[float, float]:
    """Constant-product swap.  Returns ``(dy_out, fee_paid_in_input_token)``."""
    if dx <= 0.0 or x_in_reserve <= EPS or y_out_reserve <= EPS:
        return 0.0, 0.0
    fee = dx * (fee_bps / 10_000.0)
    dx_eff = dx - fee
    dy = y_out_reserve * dx_eff / (x_in_reserve + dx_eff)
    # Numerical guard: never drain a pool completely.
    dy = min(dy, y_out_reserve * (1.0 - 1e-9))
    return float(dy), float(fee)


def apply_swap(state: WorldState, pool: int, side: int, dx: float
               ) -> Tuple[float, float, float]:
    """Execute a swap against ``pool``.

    ``side == 0`` sells token_a for token_b, ``side == 1`` sells token_b.
    Returns ``(dy_out, fee, price_move)`` where ``price_move`` is the signed
    relative change of the pool's marginal price (used for LARGE_SWAP events).
    """
    p0 = spot_price(state.reserves, pool)
    fee_bps = float(state.pool_fee_bps[pool])
    if side == 0:
        dy, fee = swap_output(state.reserves[pool, 0], state.reserves[pool, 1], dx, fee_bps)
        state.reserves[pool, 0] += dx
        state.reserves[pool, 1] -= dy
    else:
        dy, fee = swap_output(state.reserves[pool, 1], state.reserves[pool, 0], dx, fee_bps)
        state.reserves[pool, 1] += dx
        state.reserves[pool, 0] -= dy
    p1 = spot_price(state.reserves, pool)
    move = 0.0 if p0 <= EPS else (p1 - p0) / p0
    return dy, fee, float(move)


def add_liquidity(state: WorldState, pool: int, agent: int, amount_b: float
                  ) -> Tuple[float, float, float]:
    """Deposit ``amount_b`` of token_b plus the matching amount of token_a.

    Returns ``(used_a, used_b, shares_minted)``.  Deposits must be
    balanced, which is the standard v2 rule and avoids "free" arbitrage from
    single-sided adds.
    """
    x, y = state.reserves[pool]
    if amount_b <= 0.0 or x <= EPS or y <= EPS:
        return 0.0, 0.0, 0.0
    ratio = amount_b / y
    used_a = x * ratio
    used_b = amount_b
    total = float(state.pool_shares[pool])
    minted = total * ratio if total > EPS else float(np.sqrt(max(used_a * used_b, EPS)))
    state.reserves[pool, 0] += used_a
    state.reserves[pool, 1] += used_b
    state.pool_shares[pool] += minted
    state.lp_shares[agent, pool] += minted
    return float(used_a), float(used_b), float(minted)


def remove_liquidity(state: WorldState, pool: int, agent: int, shares: float
                     ) -> Tuple[float, float]:
    """Burn ``shares`` LP tokens, returning the pro-rata token amounts."""
    total = float(state.pool_shares[pool])
    shares = min(float(shares), float(state.lp_shares[agent, pool]))
    if shares <= 0.0 or total <= EPS:
        return 0.0, 0.0
    frac = shares / total
    out_a = float(state.reserves[pool, 0] * frac)
    out_b = float(state.reserves[pool, 1] * frac)
    state.reserves[pool, 0] -= out_a
    state.reserves[pool, 1] -= out_b
    state.pool_shares[pool] -= shares
    state.lp_shares[agent, pool] -= shares
    return out_a, out_b


# --------------------------------------------------------------------------
# Lending
# --------------------------------------------------------------------------
def borrow_rate(u: np.ndarray, base: float, slope1: float, slope2: float, kink: float
                ) -> np.ndarray:
    """Kinked (Aave/Compound-style) utilisation curve, annualised."""
    u = np.clip(u, 0.0, 1.0)
    below = base + slope1 * np.divide(u, max(kink, EPS))
    above = base + slope1 + slope2 * np.divide(u - kink, max(1.0 - kink, EPS))
    return np.where(u <= kink, below, above)


def accrue_interest(state: WorldState, cfg) -> None:
    """Advance the supply/borrow indices by one block."""
    u = state.utilization()
    br = borrow_rate(u, cfg.base_rate, cfg.slope1, cfg.slope2, cfg.kink)
    per_block = br / cfg.blocks_per_year
    sr = per_block * u * (1.0 - cfg.reserve_factor)

    borrow_before = state.total_borrow * state.borrow_index
    state.borrow_index *= (1.0 + per_block)
    state.supply_index *= (1.0 + sr)
    # Reserve factor accrues to the protocol, keeping the books balanced.
    interest = state.total_borrow * state.borrow_index - borrow_before
    state.protocol_reserves += interest * cfg.reserve_factor


def borrow_capacity(state: WorldState, agent: int, collateral_factor: np.ndarray,
                    price: Optional[np.ndarray] = None) -> float:
    """Remaining USD an account may borrow before its health factor hits 1."""
    price = state.oracle_price if price is None else price
    coll = float((state.supplied[agent] * state.supply_index * price * collateral_factor).sum())
    debt = float((state.borrowed[agent] * state.borrow_index * price).sum())
    return max(coll - debt, 0.0)


def liquidation_amounts(state: WorldState, victim: int, debt_token: int, coll_token: int,
                        close_factor: float, bonus: float,
                        collateral_factor: np.ndarray) -> Tuple[float, float, bool]:
    """Compute a liquidation.

    Returns ``(repay_amount_in_debt_token, seize_amount_in_coll_token,
    shortfall)`` where ``shortfall`` is True when the victim's collateral could
    not cover the seizure -- i.e. bad debt is created.
    """
    price = state.oracle_price
    debt = float(state.borrowed[victim, debt_token] * state.borrow_index[debt_token])
    if debt <= 1e-9:
        return 0.0, 0.0, False
    repay = debt * close_factor
    repay_value = repay * price[debt_token]
    seize_value = repay_value * (1.0 + bonus)
    coll_avail = float(state.supplied[victim, coll_token] * state.supply_index[coll_token])
    seize = seize_value / max(price[coll_token], EPS)
    shortfall = False
    if seize > coll_avail:
        seize = coll_avail
        # Scale the repayment down to what the seizable collateral supports.
        repay_value = seize * price[coll_token] / (1.0 + bonus)
        repay = repay_value / max(price[debt_token], EPS)
        shortfall = True
    return float(repay), float(seize), shortfall
