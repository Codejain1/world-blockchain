"""A flat, discrete action space shared by *every* intelligence system.

Scientific-fairness note
------------------------
LLM, world-model planner, unified system and all baselines choose from the
*same* enumerated set of ``n`` actions with the same semantics.  No system can
express an action another cannot.  Sizes are expressed as fractions of the
relevant capacity so the index space is scale-free (action 7 means the same
thing for a whale and for a retail account).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import EnvConfig
from .types import Action, ActionType

__all__ = ["DiscreteActionSpace"]

DEFAULT_FRACS: Tuple[float, ...] = (0.10, 0.25, 0.50)


@dataclass(frozen=True)
class _Slot:
    atype: ActionType
    target: int
    side: int
    frac: float
    name: str


class DiscreteActionSpace:
    """Enumerates the legal action menu for one agent in a given world config."""

    def __init__(self, cfg: EnvConfig, fracs: Sequence[float] = DEFAULT_FRACS,
                 tip_multiple: float = 0.10) -> None:
        self.cfg = cfg
        self.fracs = tuple(float(f) for f in fracs)
        self.tip_multiple = float(tip_multiple)
        slots: List[_Slot] = [_Slot(ActionType.NOOP, 0, 0, 0.0, "NOOP")]
        tok = cfg.token_names

        for p, (ta, tb) in enumerate(cfg.pools):
            for side in (0, 1):
                sell, buy = (ta, tb) if side == 0 else (tb, ta)
                for f in self.fracs:
                    slots.append(_Slot(ActionType.SWAP, p, side, f,
                                       f"SWAP[{tok[sell]}->{tok[buy]}]@{f:g}"))
        for p, (ta, tb) in enumerate(cfg.pools):
            for f in self.fracs:
                slots.append(_Slot(ActionType.ADD_LIQUIDITY, p, 0, f,
                                   f"ADD_LIQ[{tok[ta]}/{tok[tb]}]@{f:g}"))
        for p, (ta, tb) in enumerate(cfg.pools):
            for f in self.fracs:
                slots.append(_Slot(ActionType.REMOVE_LIQUIDITY, p, 0, f,
                                   f"REM_LIQ[{tok[ta]}/{tok[tb]}]@{f:g}"))
        for at, label in ((ActionType.SUPPLY, "SUPPLY"), (ActionType.WITHDRAW, "WITHDRAW"),
                          (ActionType.BORROW, "BORROW"), (ActionType.REPAY, "REPAY")):
            for k in cfg.lend_tokens:
                for f in self.fracs:
                    slots.append(_Slot(at, k, 0, f, f"{label}[{tok[k]}]@{f:g}"))
        for f in self.fracs:
            slots.append(_Slot(ActionType.STAKE, cfg.gov_token, 0, f, f"STAKE@{f:g}"))
        for f in self.fracs:
            slots.append(_Slot(ActionType.UNSTAKE, cfg.gov_token, 0, f, f"UNSTAKE@{f:g}"))
        slots.append(_Slot(ActionType.VOTE, 0, 1, 0.0, "VOTE[yes]"))
        slots.append(_Slot(ActionType.VOTE, 0, 0, 0.0, "VOTE[no]"))
        slots.append(_Slot(ActionType.LIQUIDATE, 0, -1, 0.0, "LIQUIDATE[auto]"))

        self._slots: Tuple[_Slot, ...] = tuple(slots)
        self.names: Tuple[str, ...] = tuple(s.name for s in slots)

    def __len__(self) -> int:
        return len(self._slots)

    @property
    def n(self) -> int:
        return len(self._slots)

    def decode(self, index: int, agent: int, base_fee: float = 0.0) -> Action:
        """Turn an action index into a concrete :class:`Action` for ``agent``."""
        s = self._slots[int(index) % len(self._slots)]
        tip = self.tip_multiple * float(base_fee) if s.atype != ActionType.NOOP else 0.0
        return Action(agent=int(agent), atype=s.atype, target=s.target,
                      side=s.side, frac=s.frac, tip=tip)

    def describe(self, index: int) -> str:
        return self.names[int(index) % len(self._slots)]

    def one_hot(self, index: int) -> np.ndarray:
        v = np.zeros(self.n, dtype=np.float32)
        v[int(index) % self.n] = 1.0
        return v

    def index_of(self, name: str) -> int:
        return self.names.index(name)

    def project(self, action: Action) -> int:
        """Nearest discrete index for an arbitrary :class:`Action`.

        Lets scripted archetype policies drive a focal agent while still logging
        a valid action *index*, which is what action-conditioned world models
        need.  Matching is exact on (type, target, side) and nearest on size.
        """
        at = ActionType(int(action.atype))
        if at == ActionType.NOOP:
            return 0
        best, best_cost = 0, float("inf")
        for i, s in enumerate(self._slots):
            if s.atype != at:
                continue
            cost = 0.0
            if at in (ActionType.SWAP, ActionType.ADD_LIQUIDITY,
                      ActionType.REMOVE_LIQUIDITY):
                if s.target != int(action.target) % self.cfg.n_pools:
                    continue
            elif at in (ActionType.SUPPLY, ActionType.WITHDRAW,
                        ActionType.BORROW, ActionType.REPAY):
                if s.target != int(action.target) % self.cfg.n_tokens:
                    continue
            if at in (ActionType.SWAP, ActionType.VOTE) and s.side != int(action.side) % 2:
                continue
            cost += abs(s.frac - float(action.frac))
            if cost < best_cost:
                best, best_cost = i, cost
        return best

    # -- crude legality filter (used to shrink planner search, never to give a
    #    system extra information another system lacks) --------------------
    def feasible_mask(self, world, agent: int) -> np.ndarray:
        """Boolean mask of actions that are not trivially impossible.

        Derived purely from the agent's *own* observable position, so applying
        it to one system and not another changes efficiency, not information.
        """
        st = world.state
        cfg = self.cfg
        mask = np.ones(self.n, dtype=bool)
        hf = world.health_factor()
        any_unhealthy = bool(np.any(hf < 1.0))
        for i, s in enumerate(self._slots):
            if s.atype == ActionType.SWAP:
                tok_in = cfg.pools[s.target][s.side]
                mask[i] = st.balances[agent, tok_in] > 1e-9
            elif s.atype == ActionType.ADD_LIQUIDITY:
                ta, tb = cfg.pools[s.target]
                mask[i] = st.balances[agent, ta] > 1e-9 and st.balances[agent, tb] > 1e-9
            elif s.atype == ActionType.REMOVE_LIQUIDITY:
                mask[i] = st.lp_shares[agent, s.target] > 1e-12
            elif s.atype == ActionType.SUPPLY:
                mask[i] = st.balances[agent, s.target] > 1e-9
            elif s.atype in (ActionType.WITHDRAW,):
                mask[i] = st.supplied[agent, s.target] > 1e-12
            elif s.atype == ActionType.REPAY:
                mask[i] = (st.borrowed[agent, s.target] > 1e-12
                           and st.balances[agent, s.target] > 1e-9)
            elif s.atype == ActionType.BORROW:
                mask[i] = float((st.supplied[agent] * st.supply_index).sum()) > 1e-12
            elif s.atype == ActionType.STAKE:
                mask[i] = st.balances[agent, cfg.gov_token] > 1e-9
            elif s.atype == ActionType.UNSTAKE:
                mask[i] = st.staked[agent] > 1e-12
            elif s.atype == ActionType.VOTE:
                live = [p for p in st.proposals
                        if not p.executed and not p.rejected and p.closes_at > st.t]
                mask[i] = bool(live) and st.staked[agent] > 1e-12
            elif s.atype == ActionType.LIQUIDATE:
                mask[i] = any_unhealthy
        mask[0] = True   # NOOP is always available
        return mask
