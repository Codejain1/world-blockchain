"""Environment correctness: determinism, replay, conservation, action semantics."""

from __future__ import annotations

import numpy as np
import pytest

from bwm.agents.population import AgentPopulation
from bwm.environment.action_space import DiscreteActionSpace
from bwm.environment.config import EnvConfig
from bwm.environment.observation import ObservationBuilder
from bwm.environment.protocols import borrow_rate, swap_output
from bwm.environment.runner import BehaviorPolicy, EpisodeRunner
from bwm.environment.scenarios import make_scenario, sample_train_config, scenario_names
from bwm.environment.types import Action, ActionType, N_EVENTS
from bwm.environment.world import BlockchainWorld
from bwm.utils.seeding import make_rng, stream_seed


def build(seed: int = 0, n_agents: int = 40, **kw):
    cfg = EnvConfig(n_agents=n_agents, seed=seed, **kw)
    pop = AgentPopulation(cfg)
    return cfg, pop, BlockchainWorld(cfg, pop)


# ---------------------------------------------------------------- seeding
def test_counter_based_streams_are_position_independent():
    a = make_rng(3, "price", 17).normal(size=5)
    b = make_rng(3, "price", 17).normal(size=5)
    assert np.array_equal(a, b)
    assert stream_seed(3, "price", 17) != stream_seed(3, "price", 18)
    assert stream_seed(3, "price", 17) != stream_seed(4, "price", 17)


# ------------------------------------------------------------ determinism
def test_same_seed_same_trajectory():
    digests = []
    for _ in range(2):
        _, _, w = build(seed=5)
        d = [w.step().info["regime"] for _ in range(50)]
        digests.append((d, w.state.digest()))
    assert digests[0] == digests[1]


def test_different_seed_diverges():
    _, _, w1 = build(seed=5)
    _, _, w2 = build(seed=6)
    for _ in range(50):
        w1.step(); w2.step()
    assert w1.state.digest() != w2.state.digest()


def test_snapshot_restore_roundtrip():
    _, _, w = build(seed=9)
    for _ in range(20):
        w.step()
    snap = w.snapshot()
    ref = w.state.digest()
    for _ in range(10):
        w.step()
    assert w.state.digest() != ref
    w.restore(snap)
    assert w.state.digest() == ref
    # and continuing from the restored point reproduces the same future
    after = [w.step().info["n_liquidations"] for _ in range(10)]
    w.restore(snap)
    assert after == [w.step().info["n_liquidations"] for _ in range(10)]


# ------------------------------------------------- counterfactual forking
def test_fork_with_null_intervention_is_identical():
    _, _, w = build(seed=13)
    for _ in range(30):
        w.step()
    a, b = w.fork(), w.fork()
    for _ in range(15):
        a.step(); b.step(None)
    assert a.state.digest() == b.state.digest()


def test_fork_with_intervention_diverges_and_isolates_the_action():
    """Same world noise, different action => difference is caused by the action."""
    _, _, w = build(seed=13)
    for _ in range(30):
        w.step()
    base, cf = w.fork(), w.fork()
    act = Action(agent=3, atype=ActionType.SWAP, target=0, side=1, frac=0.9, tip=1e-5)
    for t in range(15):
        base.step()
        cf.step({3: act} if t == 0 else None)
    assert base.state.digest() != cf.state.digest()
    # the intervening agent's own position must differ
    assert not np.allclose(base.state.balances[3], cf.state.balances[3])
    # the exogenous process is untouched by the intervention
    assert np.allclose(base.state.fundamental, cf.state.fundamental)


# ------------------------------------------------------------- protocols
def test_constant_product_invariant_grows_with_fees():
    x, y, dx = 1000.0, 2000.0, 37.0
    dy, fee = swap_output(x, y, dx, 30.0)
    assert 0 < dy < y
    assert fee == pytest.approx(dx * 0.003)
    assert (x + dx) * (y - dy) >= x * y          # fee accrues to the pool


def test_borrow_rate_is_monotone_and_kinked():
    u = np.linspace(0.0, 1.0, 21)
    r = borrow_rate(u, 0.01, 0.06, 3.0, 0.8)
    assert np.all(np.diff(r) >= -1e-12)
    assert r[0] == pytest.approx(0.01)
    assert r[-1] > r[16]                          # steep above the kink


def test_swap_moves_price_in_the_expected_direction():
    cfg, pop, w = build(seed=2)
    p0 = w.amm_usd_price(1)
    w.state.balances[0, 1] = 500.0
    rc = w._execute(Action(agent=0, atype=ActionType.SWAP, target=0, side=0, frac=1.0))
    assert rc.success, rc.reason
    assert w.amm_usd_price(1) < p0                # selling ETHX lowers its price


def test_borrow_requires_collateral_and_respects_health():
    cfg, pop, w = build(seed=4)
    a = 0
    w.state.borrowed[a] = 0.0
    w.state.supplied[a] = 0.0
    rc = w._execute(Action(agent=a, atype=ActionType.BORROW, target=1, frac=1.0))
    assert not rc.success and rc.reason == "no_capacity"
    w.state.balances[a, 0] = 10_000.0
    assert w._execute(Action(agent=a, atype=ActionType.SUPPLY, target=0, frac=0.9)).success
    rc = w._execute(Action(agent=a, atype=ActionType.BORROW, target=1, frac=0.5))
    assert rc.success, rc.reason
    assert w.health_factor()[a] >= 1.0


def test_liquidation_only_hits_unhealthy_accounts():
    cfg, pop, w = build(seed=6)
    hf = w.health_factor()
    assert np.all(np.isinf(hf))                   # nobody has debt at genesis
    rc = w._execute(Action(agent=0, atype=ActionType.LIQUIDATE, target=1, side=1))
    assert not rc.success and rc.reason == "no_target"


def test_token_conservation_under_transfer():
    cfg, pop, w = build(seed=8)
    w.state.balances[0, 1] = 100.0
    before = w.state.balances[:, 1].sum()
    rc = w._execute(Action(agent=0, atype=ActionType.TRANSFER, target=1, side=2, frac=0.5))
    assert rc.success
    assert w.state.balances[:, 1].sum() == pytest.approx(before)
    assert w.state.balances[0, 1] == pytest.approx(50.0)
    assert w.state.balances[2, 1] > 0.0


def test_gas_is_charged_and_burned():
    cfg, pop, w = build(seed=10)
    st = w.state
    usd0, burned0 = st.balances[0, 0], st.fees_burned
    w._execute(Action(agent=0, atype=ActionType.SWAP, target=0, side=1, frac=0.01,
                      tip=1e-6))
    assert st.balances[0, 0] < usd0
    assert st.fees_burned > burned0


# ---------------------------------------------------------- action space
def test_action_space_is_complete_and_projection_is_consistent():
    cfg = EnvConfig(n_agents=20)
    sp = DiscreteActionSpace(cfg)
    assert sp.n == len(set(sp.names))             # no duplicate labels
    assert sp.describe(0) == "NOOP"
    for i in range(sp.n):
        a = sp.decode(i, agent=7, base_fee=1e-6)
        assert a.agent == 7 and 0.0 <= a.frac <= 1.0
        if a.atype != ActionType.NOOP:
            assert sp.project(a) == i or sp.names[sp.project(a)] == sp.names[i]


def test_feasible_mask_excludes_positions_the_agent_lacks():
    cfg, pop, w = build(seed=12)
    sp = DiscreteActionSpace(cfg)
    m = sp.feasible_mask(w, 0)
    assert m[0]                                    # NOOP always legal
    for i, name in enumerate(sp.names):
        if name.startswith("REM_LIQ") or name.startswith("UNSTAKE"):
            assert not m[i]                        # no LP or stake at genesis


# ----------------------------------------------------------- observations
def test_observation_excludes_hidden_state():
    """No observation feature may reveal the regime or the fundamental price."""
    cfg, pop, w = build(seed=14)
    ob = ObservationBuilder(cfg)
    names = ob.feature_names
    assert not any("regime" in n or "fundamental" in n for n in names)
    o = ob.observe(w, 0)
    assert o.shape == (ob.obs_dim,) and np.isfinite(o).all()
    # Two worlds differing only in regime but with identical prices/positions
    # must produce identical observations.
    w2 = w.fork()
    w2.state.regime = (w.state.regime + 1) % 4
    assert np.allclose(ob.observe(w, 0), ob.observe(w2, 0))


def test_observations_stay_finite_over_a_long_episode():
    cfg = sample_train_config(21, episode_length=200)
    r = EpisodeRunner(cfg)
    tr = r.rollout(BehaviorPolicy(r.action_space, 0.5, seed=21), seed=21)
    assert np.isfinite(tr.obs).all() and np.isfinite(tr.node_feat).all()
    assert np.isfinite(tr.rewards).all()
    assert tr.events.shape == (tr.T, N_EVENTS)
    assert set(np.unique(tr.events)).issubset({0.0, 1.0})


# -------------------------------------------------------------- scenarios
@pytest.mark.parametrize("name", scenario_names())
def test_every_scenario_runs_and_is_reproducible(name):
    cfg = make_scenario(name, seed=777, n_agents=60, episode_length=64)
    outs = []
    for _ in range(2):
        r = EpisodeRunner(cfg)
        tr = r.rollout(BehaviorPolicy(r.action_space, 0.4, seed=777), seed=777)
        outs.append(tr)
    assert np.array_equal(outs[0].obs, outs[1].obs)
    assert np.isfinite(outs[0].obs).all()


def test_held_out_scenarios_differ_from_training_worlds():
    train = sample_train_config(31)
    for name in ("liquidity_crisis", "unseen_agents", "novel_mechanism", "amm_params"):
        ood = make_scenario(name, seed=31)
        assert ood.to_dict() != train.to_dict(), name


def test_training_population_never_contains_held_out_archetypes():
    for seed in range(20):
        cfg = sample_train_config(seed)
        assert "adversarial" not in cfg.population
        assert "coordinated" not in cfg.population
