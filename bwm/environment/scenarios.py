"""Named worlds: the training distribution and the held-out test distributions.

The generalisation claim of this lab rests entirely on this file.  Worlds differ
*only* by configuration, and the held-out worlds change axes that the training
distribution never varies:

======================  =====================================================
scenario                what is different from training
======================  =====================================================
``iid``                 nothing (disjoint seeds only) -- the interpolation test
``liquidity_crisis``    market depth collapses and shocks are frequent
``unseen_agents``       archetypes absent from every training episode
``coordinated_attack``  an adversarial bloc that acts as one
``token_economics``     new price process, new token values, new staking rate
``amm_params``          different fee and a different depth profile
``novel_mechanism``     protocol rules that do not exist in training
                        (volatility-linked AMM fee, borrow cap, TWAP oracle)
``shock_storm``         forced crisis regimes and amplified shocks
======================  =====================================================

Note that ``unseen_agents`` and ``coordinated_attack`` introduce *behaviours*,
``liquidity_crisis``/``amm_params``/``token_economics`` change *parameters*, and
``novel_mechanism`` changes the *transition rules* themselves.  Reporting those
separately matters: a model can extrapolate over parameters without being able
to cope with a rule it has never seen.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from .config import EnvConfig, RegimeParams

__all__ = ["TRAIN_POPULATION", "base_config", "sample_train_config",
           "SCENARIOS", "make_scenario", "scenario_names"]

#: Archetypes the training distribution is allowed to contain.  ``adversarial``
#: and ``coordinated`` are deliberately excluded so that "unseen agent type" is
#: a real held-out axis rather than a relabelling.
TRAIN_POPULATION: Dict[str, int] = {
    "retail": 48, "whale": 6, "arbitrageur": 8, "market_maker": 8,
    "liquidity_provider": 10, "borrower": 28, "keeper": 5,
    "governance": 4, "random": 3,
}


def base_config(seed: int = 0, n_agents: int = 120, episode_length: int = 256,
                **overrides) -> EnvConfig:
    """The canonical in-distribution world."""
    cfg = EnvConfig(name="base", seed=int(seed), n_agents=int(n_agents),
                    episode_length=int(episode_length),
                    population=dict(TRAIN_POPULATION))
    return cfg.variant(**overrides) if overrides else cfg


def sample_train_config(seed: int, n_agents: int = 120, episode_length: int = 256
                        ) -> EnvConfig:
    """Draw one world from the *training distribution*.

    Nuisance parameters are jittered so models cannot memorise a single world,
    but every varied quantity stays inside a narrow band.  The held-out
    scenarios move well outside these bands (see :data:`SCENARIOS`).
    """
    rng = np.random.Generator(np.random.PCG64(int(seed) ^ 0x5EED))
    cfg = base_config(seed=seed, n_agents=n_agents, episode_length=episode_length)
    return cfg.variant(
        name="train",
        liquidity_scale=float(rng.uniform(0.85, 1.20)),
        amm_fee_bps=float(rng.uniform(25.0, 35.0)),
        init_regime=int(rng.integers(0, 3)),          # never starts in CRISIS
        init_risk_fraction=float(rng.uniform(0.38, 0.52)),
        staking_reward_rate=float(rng.uniform(0.08, 0.16)),
        shock_magnitude=float(rng.uniform(0.85, 1.15)),
        market_corr=float(rng.uniform(0.45, 0.65)),
    )


# --------------------------------------------------------------------------
# held-out worlds
# --------------------------------------------------------------------------
def _iid(cfg: EnvConfig) -> EnvConfig:
    return cfg.variant(name="iid")


def _liquidity_crisis(cfg: EnvConfig) -> EnvConfig:
    probs = dict(cfg.shock_probs)
    probs["liquidity_withdrawal"] *= 8.0
    probs["whale_dump"] *= 4.0
    return cfg.variant(
        name="liquidity_crisis",
        liquidity_scale=0.30,
        lending_seed_usd=cfg.lending_seed_usd * 0.35,
        shock_probs=probs,
        shock_magnitude=1.8,
        shock_crisis_multiplier=10.0,
    )


def _unseen_agents(cfg: EnvConfig) -> EnvConfig:
    pop = dict(TRAIN_POPULATION)
    pop["retail"] = max(pop["retail"] - 20, 4)
    pop["adversarial"] = 10          # never present during training
    pop["coordinated"] = 10          # never present during training
    return cfg.variant(name="unseen_agents", population=pop)


def _coordinated_attack(cfg: EnvConfig) -> EnvConfig:
    pop = dict(TRAIN_POPULATION)
    pop["retail"] = max(pop["retail"] - 26, 4)
    pop["adversarial"] = 8
    pop["coordinated"] = 18
    return cfg.variant(name="coordinated_attack", population=pop,
                       coordinated_attack=True, liquidity_scale=0.7)


def _token_economics(cfg: EnvConfig) -> EnvConfig:
    regimes = {k: RegimeParams(**{**vars(v), "sigma": v.sigma * 1.7,
                                  "mu": v.mu * 1.4,
                                  "jump_lambda": v.jump_lambda * 2.5})
               for k, v in cfg.regimes.items()}
    return cfg.variant(
        name="token_economics",
        init_prices=[1.0, 400.0, 62.0, 3.2],        # different value scales
        regimes={k: vars(v) for k, v in regimes.items()},
        staking_reward_rate=0.45,
        market_corr=0.15,                            # nearly idiosyncratic tokens
        init_risk_fraction=0.65,
    )


def _amm_params(cfg: EnvConfig) -> EnvConfig:
    depths = list(cfg.pool_depth_usd)
    depths = [depths[2], depths[3], depths[0], depths[1]]   # inverted depth profile
    return cfg.variant(
        name="amm_params",
        amm_fee_bps=85.0,
        pool_depth_usd=depths,
        kink=0.55, slope1=0.15, slope2=6.0,
        liquidation_bonus=0.16, close_factor=0.75,
        collateral_factor={0: 0.95, 1: 0.55, 2: 0.40, 3: 0.0},
    )


def _novel_mechanism(cfg: EnvConfig) -> EnvConfig:
    """Rules that literally do not exist in the training worlds."""
    return cfg.variant(
        name="novel_mechanism",
        enable_dynamic_fee=True,       # AMM fee reacts to realised volatility
        enable_borrow_cap=True,        # protocol-wide borrow ceiling
        borrow_cap_frac=0.45,
        oracle_source="amm_twap",      # oracle becomes manipulable
        oracle_twap_window=12,
    )


def _shock_storm(cfg: EnvConfig) -> EnvConfig:
    probs = {k: v * 3.0 for k, v in cfg.shock_probs.items()}
    # Deterministic crisis windows: 64 calm steps, then 32 of crisis, repeating.
    sched: List[int] = ([2] * 64 + [3] * 32) * 8
    return cfg.variant(name="shock_storm", shock_probs=probs, shock_magnitude=2.2,
                       forced_regime_schedule=sched, init_regime=3)


SCENARIOS: Dict[str, Callable[[EnvConfig], EnvConfig]] = {
    "iid": _iid,
    "liquidity_crisis": _liquidity_crisis,
    "unseen_agents": _unseen_agents,
    "coordinated_attack": _coordinated_attack,
    "token_economics": _token_economics,
    "amm_params": _amm_params,
    "novel_mechanism": _novel_mechanism,
    "shock_storm": _shock_storm,
}

#: Grouping used when reporting results.
SCENARIO_KIND: Dict[str, str] = {
    "iid": "interpolation",
    "liquidity_crisis": "parameter_shift",
    "amm_params": "parameter_shift",
    "token_economics": "parameter_shift",
    "unseen_agents": "behaviour_shift",
    "coordinated_attack": "behaviour_shift",
    "novel_mechanism": "rule_shift",
    "shock_storm": "rule_shift",
}


def scenario_names() -> List[str]:
    return list(SCENARIOS)


def make_scenario(name: str, seed: int, n_agents: int = 120,
                  episode_length: int = 256) -> EnvConfig:
    """Build a held-out world.

    The base world is drawn from the *training* distribution first, so held-out
    worlds differ from training only along the axis the scenario changes.
    """
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario {name!r}; known: {sorted(SCENARIOS)}")
    base = sample_train_config(seed, n_agents=n_agents, episode_length=episode_length)
    return SCENARIOS[name](base)
