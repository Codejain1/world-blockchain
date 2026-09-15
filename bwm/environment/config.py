"""Environment configuration.

Everything that defines a *world* lives here, so held-out ("OOD") worlds are
produced by changing configuration only -- never by changing simulator source.
That separation is what lets us claim a generalisation test is honest: the same
code path runs train and test worlds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = ["EnvConfig", "RegimeParams"]


@dataclass
class RegimeParams:
    """Price-process parameters for one hidden market regime."""

    mu: float = 0.05           # annualised drift of volatile tokens
    sigma: float = 0.6         # annualised volatility
    jump_lambda: float = 0.002  # per-block jump probability
    jump_mu: float = -0.02     # mean jump size in log space
    jump_sigma: float = 0.08   # jump dispersion
    flow_bias: float = 0.0     # nudges scripted agents toward buying (+) / selling (-)


def _default_regimes() -> Dict[str, RegimeParams]:
    return {
        "BULL": RegimeParams(mu=0.9, sigma=0.55, jump_lambda=0.0015, jump_mu=0.03,
                             jump_sigma=0.05, flow_bias=0.35),
        "BEAR": RegimeParams(mu=-0.7, sigma=0.75, jump_lambda=0.004, jump_mu=-0.05,
                             jump_sigma=0.09, flow_bias=-0.35),
        "CRAB": RegimeParams(mu=0.02, sigma=0.35, jump_lambda=0.001, jump_mu=0.0,
                             jump_sigma=0.04, flow_bias=0.0),
        "CRISIS": RegimeParams(mu=-2.2, sigma=1.6, jump_lambda=0.02, jump_mu=-0.12,
                               jump_sigma=0.18, flow_bias=-0.8),
    }


def _default_transition() -> List[List[float]]:
    # Rows/cols ordered BULL, BEAR, CRAB, CRISIS.  Persistent regimes with rare
    # crisis entry and fast crisis exit -- crises are short and violent.
    return [
        [0.985, 0.004, 0.010, 0.001],
        [0.006, 0.980, 0.012, 0.002],
        [0.010, 0.010, 0.979, 0.001],
        [0.030, 0.120, 0.020, 0.830],
    ]


@dataclass
class EnvConfig:
    """Full specification of a simulated blockchain economy."""

    # --- identity -------------------------------------------------------
    name: str = "base"
    seed: int = 0
    episode_length: int = 256

    # --- tokens & venues ------------------------------------------------
    token_names: List[str] = field(default_factory=lambda: ["USD", "ETHX", "ALT", "GOV"])
    #: constant-product pools as ``(token_a, token_b)`` index pairs.
    pools: List[Tuple[int, int]] = field(default_factory=lambda: [(1, 0), (2, 0), (3, 0), (1, 2)])
    #: tokens with a lending market.
    lend_tokens: List[int] = field(default_factory=lambda: [0, 1, 2])
    gov_token: int = 3

    # --- population -----------------------------------------------------
    n_agents: int = 120
    #: archetype -> count.  Resolved by :mod:`bwm.agents.population`.
    population: Dict[str, int] = field(default_factory=lambda: {
        "retail": 48, "whale": 6, "arbitrageur": 8, "market_maker": 8,
        "liquidity_provider": 10, "borrower": 28, "keeper": 5,
        "governance": 4, "random": 3,
    })
    init_wealth_median: float = 50_000.0
    init_wealth_sigma: float = 1.1          # lognormal shape -> heavy tail
    whale_wealth_multiplier: float = 30.0
    #: fraction of an agent's initial wealth held in volatile tokens.
    init_risk_fraction: float = 0.45

    # --- AMM ------------------------------------------------------------
    amm_fee_bps: float = 30.0
    #: USD-equivalent depth of each pool at genesis.
    pool_depth_usd: List[float] = field(default_factory=lambda: [1.8e7, 7.5e6, 4.5e6, 2.4e6])
    #: multiplies all pool depths -- the knob for "different liquidity structure".
    liquidity_scale: float = 1.0
    #: Maximum size of a single swap, as a fraction of the input-side reserve.
    #:
    #: Without this cap a thin pool can be dislocated by tens of times in one
    #: block: the governance token has no lending market, so arbitrageurs cannot
    #: short it and cannot correct an over-price, while on the other side they
    #: (and the leverage loop) push far past the fair price.  Net worth is marked
    #: at the protocol oracle, so a dislocated pool becomes free money for anyone
    #: who can see it -- which a true-simulator planner can.  Capping per-trade
    #: price impact is both realistic (routers and desks enforce slippage limits)
    #: and enough to keep the AMM tethered to the oracle.
    max_swap_frac: float = 0.15

    # --- lending --------------------------------------------------------
    base_rate: float = 0.01
    slope1: float = 0.06
    slope2: float = 3.0
    kink: float = 0.80
    reserve_factor: float = 0.10
    collateral_factor: Dict[int, float] = field(default_factory=lambda: {0: 0.90, 1: 0.75, 2: 0.60, 3: 0.0})
    close_factor: float = 0.5
    liquidation_bonus: float = 0.08
    blocks_per_year: float = 8760.0          # one step == one hour of chain time
    lending_seed_usd: float = 2.4e6          # protocol-owned initial supply depth

    # --- staking & governance ------------------------------------------
    staking_reward_rate: float = 0.12        # annualised, paid in GOV emissions
    governance_enabled: bool = True
    proposal_interval: int = 64
    voting_blocks: int = 24
    quorum_frac: float = 0.10
    #: parameters a passed proposal may change, as ``name -> (lo, hi)``.
    governable: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "amm_fee_bps": (5.0, 100.0),
        "liquidation_bonus": (0.03, 0.20),
        "collateral_factor_1": (0.40, 0.85),
        "staking_reward_rate": (0.02, 0.35),
    })

    # --- gas ------------------------------------------------------------
    block_gas_target: int = 2_000_000
    block_gas_limit: int = 6_000_000
    init_base_fee: float = 1.0e-6            # USD per gas unit
    base_fee_max_change_denom: float = 8.0
    min_base_fee: float = 1.0e-7
    #: an account will not pay more than this fraction of its trade value in gas,
    #: which gives EIP-1559 a demand side and stops the base fee running away.
    gas_budget_frac: float = 0.005

    # --- price process --------------------------------------------------
    init_prices: List[float] = field(default_factory=lambda: [1.0, 2500.0, 8.0, 15.0])
    dt_years: float = 1.0 / 8760.0           # one step == one hour
    market_corr: float = 0.55                # common-factor loading across tokens
    regimes: Dict[str, RegimeParams] = field(default_factory=_default_regimes)
    regime_transition: List[List[float]] = field(default_factory=_default_transition)
    init_regime: int = 2                     # CRAB
    #: 'fundamental' = honest oracle; 'amm_twap' = manipulable oracle (novel mechanism).
    oracle_source: str = "fundamental"
    oracle_twap_window: int = 8

    # --- shocks ---------------------------------------------------------
    shock_enabled: bool = True
    #: per-block probability of each shock, multiplied by the crisis multiplier
    #: while the hidden regime is CRISIS.
    shock_probs: Dict[str, float] = field(default_factory=lambda: {
        "liquidity_withdrawal": 0.004,
        "whale_dump": 0.004,
        "depeg": 0.0020,
        "protocol_exploit": 0.0006,
        "oracle_glitch": 0.0010,
        "gas_spike": 0.0030,
    })
    shock_crisis_multiplier: float = 6.0
    shock_magnitude: float = 1.0             # global scaling knob for shock severity

    # --- event thresholds ----------------------------------------------
    large_swap_move: float = 0.06            # |pool price move| to flag LARGE_SWAP
    price_jump_threshold: float = 0.02       # |log return| to flag PRICE_JUMP
    liquidity_drop_frac: float = 0.05
    cascade_threshold: int = 2
    high_gas_multiple: float = 2.5
    #: smoothing of the trailing base-fee baseline used by the HIGH_GAS label.
    gas_baseline_decay: float = 0.98
    #: hard ceiling on lending utilisation enforced at borrow time.
    max_utilization: float = 0.99
    depeg_band: float = 0.02
    utilization_spike: float = 0.85

    # --- scenario switches used by held-out worlds ----------------------
    #: enables a protocol mechanism absent from the training distribution.
    enable_dynamic_fee: bool = False         # AMM fee scales with realised volatility
    enable_borrow_cap: bool = False          # hard cap on protocol-wide borrow
    borrow_cap_frac: float = 0.6
    coordinated_attack: bool = False         # coordinated agents act as one bloc
    forced_regime_schedule: Optional[List[int]] = None  # deterministic regime path

    # -------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.pools = [tuple(int(x) for x in p) for p in self.pools]  # type: ignore[misc]
        self.lend_tokens = [int(x) for x in self.lend_tokens]
        self.collateral_factor = {int(k): float(v) for k, v in self.collateral_factor.items()}
        self.regimes = {
            k: (v if isinstance(v, RegimeParams) else RegimeParams(**dict(v)))
            for k, v in self.regimes.items()
        }
        self.governable = {k: (float(v[0]), float(v[1])) for k, v in self.governable.items()}
        if len(self.init_prices) != len(self.token_names):
            raise ValueError("init_prices must have one entry per token")
        if len(self.pool_depth_usd) != len(self.pools):
            raise ValueError("pool_depth_usd must have one entry per pool")
        tm = np.asarray(self.regime_transition, dtype=np.float64)
        if tm.shape != (4, 4):
            raise ValueError("regime_transition must be 4x4")
        if not np.allclose(tm.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("regime_transition rows must sum to 1")

    # -- convenience -----------------------------------------------------
    @property
    def n_tokens(self) -> int:
        return len(self.token_names)

    @property
    def n_pools(self) -> int:
        return len(self.pools)

    @property
    def n_lend(self) -> int:
        return len(self.lend_tokens)

    @classmethod
    def from_dict(cls, d: Optional[Mapping[str, Any]]) -> "EnvConfig":
        d = dict(d or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown EnvConfig fields: {sorted(unknown)}")
        if "regimes" in d and d["regimes"] is not None:
            d["regimes"] = {k: (v if isinstance(v, RegimeParams) else RegimeParams(**dict(v)))
                            for k, v in d["regimes"].items()}
        return cls(**d)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["pools"] = [list(p) for p in self.pools]
        out["regimes"] = {k: asdict(v) for k, v in self.regimes.items()}
        out["collateral_factor"] = {int(k): float(v) for k, v in self.collateral_factor.items()}
        out["governable"] = {k: list(v) for k, v in self.governable.items()}
        return out

    def variant(self, **changes: Any) -> "EnvConfig":
        """Return a copy with ``changes`` applied (used to build held-out worlds)."""
        d = self.to_dict()
        d.update(changes)
        return EnvConfig.from_dict(d)
