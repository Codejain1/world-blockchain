"""Core value types for the blockchain world: actions, events, results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = [
    "ActionType",
    "Action",
    "EventType",
    "EVENT_NAMES",
    "N_EVENTS",
    "Regime",
    "REGIME_NAMES",
    "Receipt",
    "StepResult",
]


class ActionType(IntEnum):
    """Every state-mutating operation an account can submit to a block."""

    NOOP = 0
    SWAP = 1              # target=pool, side=0 -> sell token_a, side=1 -> sell token_b
    ADD_LIQUIDITY = 2     # target=pool, frac of USD-side balance
    REMOVE_LIQUIDITY = 3  # target=pool, frac of LP shares
    SUPPLY = 4            # target=token, frac of balance
    WITHDRAW = 5          # target=token, frac of supplied
    BORROW = 6            # target=token, frac of remaining borrow capacity
    REPAY = 7             # target=token, frac of debt
    STAKE = 8             # frac of governance-token balance
    UNSTAKE = 9           # frac of staked
    VOTE = 10             # target=proposal, side=1 -> yes, 0 -> no
    TRANSFER = 11         # target=token, side=recipient agent id
    LIQUIDATE = 12        # target=debt token; victim resolved from `side` (-1 = auto)


#: Gas units consumed by each action type (mirrors relative EVM costs).
GAS_UNITS: Dict[int, int] = {
    ActionType.NOOP: 0,
    ActionType.SWAP: 120_000,
    ActionType.ADD_LIQUIDITY: 160_000,
    ActionType.REMOVE_LIQUIDITY: 140_000,
    ActionType.SUPPLY: 90_000,
    ActionType.WITHDRAW: 100_000,
    ActionType.BORROW: 130_000,
    ActionType.REPAY: 95_000,
    ActionType.STAKE: 80_000,
    ActionType.UNSTAKE: 85_000,
    ActionType.VOTE: 60_000,
    ActionType.TRANSFER: 21_000,
    ActionType.LIQUIDATE: 250_000,
}


@dataclass(slots=True)
class Action:
    """A transaction submitted to the mempool.

    Attributes
    ----------
    agent: submitting account id.
    atype: :class:`ActionType`.
    target: pool id / token id / proposal id depending on ``atype``.
    side: direction flag, recipient, or victim id (semantics per ``atype``).
    frac: size as a fraction of the relevant capacity (balance, debt, shares).
          Keeping sizes *relative* makes the action space scale-free, so the
          same discrete action index means the same thing for a whale and for a
          retail account.
    tip: priority fee (USD per gas unit) used for mempool ordering.
    """

    agent: int
    atype: ActionType = ActionType.NOOP
    target: int = 0
    side: int = 0
    frac: float = 0.0
    tip: float = 0.0

    def gas(self) -> int:
        return GAS_UNITS[int(self.atype)]

    def as_tuple(self) -> tuple:
        return (int(self.agent), int(self.atype), int(self.target), int(self.side),
                float(self.frac), float(self.tip))


class EventType(IntEnum):
    """Binary, per-block world events used as prediction targets.

    These are the "did something notable happen?" labels used for event
    prediction, Brier score, log-loss and calibration.  They are computed from
    the transition itself, never from hidden state, so every system can in
    principle predict them from observations.
    """

    LIQUIDATION = 0
    CASCADE = 1            # >= cascade_threshold liquidations in one block
    LARGE_SWAP = 2         # swap moving a pool by more than large_swap_move
    PRICE_JUMP = 3         # |oracle log-return| above jump threshold
    LIQUIDITY_DROP = 4     # pool TVL fell more than liquidity_drop frac
    GOV_EXECUTED = 5       # a governance proposal changed protocol parameters
    DEPEG = 6              # numeraire token left its peg band
    HIGH_GAS = 7           # base fee above high_gas multiple of its baseline
    INSOLVENCY = 8         # bad debt created (collateral could not cover debt)
    UTILIZATION_SPIKE = 9  # lending utilisation above spike threshold


EVENT_NAMES: List[str] = [e.name for e in EventType]
N_EVENTS: int = len(EVENT_NAMES)


class Regime(IntEnum):
    """Hidden market regime.

    Deliberately *not* part of any observation: inferring it from observable
    dynamics is exactly the kind of latent-state problem a world model is
    supposed to be good at.
    """

    BULL = 0
    BEAR = 1
    CRAB = 2
    CRISIS = 3


REGIME_NAMES: List[str] = [r.name for r in Regime]


@dataclass(slots=True)
class Receipt:
    """Execution outcome of a single action (the analogue of a tx receipt)."""

    action: Action
    success: bool
    reason: str = ""
    gas_used: int = 0
    fee_paid: float = 0.0
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StepResult:
    """Return value of :meth:`BlockchainWorld.step`.

    Implements the required interface ``S_t + A_t -> S_{t+1}, R_t, E_t``.
    """

    t: int
    rewards: np.ndarray            # (n_agents,) change in net worth, USD
    events: np.ndarray             # (N_EVENTS,) float 0/1 labels
    receipts: List[Receipt]
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def event_dict(self) -> Dict[str, float]:
        return {EVENT_NAMES[i]: float(self.events[i]) for i in range(N_EVENTS)}
