"""Planner, memory, LLM interface and the unified gate."""

from __future__ import annotations

import numpy as np
import pytest

from bwm.data.dataset import Normalizer
from bwm.environment.action_space import DiscreteActionSpace
from bwm.environment.scenarios import base_config, sample_train_config
from bwm.experiments.registry import build_predictive_model
from bwm.memory.episodic import ConsequenceTracker, EpisodicMemory
from bwm.models.base import DecisionContext
from bwm.models.llm.agent import LLMPolicy, RuleBasedReasoner, StatePresenter, parse_action
from bwm.models.llm.client import LLMConfig, MockClient, make_client
from bwm.models.unified.unified import PATHWAYS, LinUCBGate, UnifiedPolicy
from bwm.models.world_model.policy import WorldModelPolicy
from bwm.planning.analytic import AnalyticPlannerPolicy
from bwm.planning.planner import LatentPlanner, PlannerConfig
from bwm.training.trainer import TrainConfig

FAST = TrainConfig(epochs=1, max_batches_per_epoch=2, batch_size=32, val_batches=1)


@pytest.fixture(scope="module")
def fitted_wm(tiny_set, obs_builder):
    ts, vs, spec, nz = tiny_set
    m = build_predictive_model("wm_rssm", spec, obs_builder, FAST)
    m.fit(ts, vs, spec, nz)
    return m, spec, nz


# ----------------------------------------------------------------- planner
def test_planner_returns_a_feasible_action_and_counts_imagination(fitted_wm, tiny_env_cfg):
    model, spec, nz = fitted_wm
    asp = DiscreteActionSpace(tiny_env_cfg)
    pl = LatentPlanner(model, asp.n, PlannerConfig(horizon=3, n_candidates=12, n_iters=2))
    oh = np.zeros((spec.history, spec.obs_dim), np.float32)
    ah = np.zeros(spec.history, np.int64)
    feas = np.zeros(asp.n, bool)
    feas[[0, 5, 9, 40]] = True
    before = model.meter.imagined_steps
    a, info = pl.plan(oh, ah, feas)
    assert feas[a], "planner chose an infeasible action"
    assert model.meter.imagined_steps > before, "imagined steps were not metered"
    assert np.isfinite(info["score"])


def test_planner_is_deterministic_for_a_fixed_seed(fitted_wm, tiny_env_cfg):
    model, spec, nz = fitted_wm
    asp = DiscreteActionSpace(tiny_env_cfg)
    cfg = PlannerConfig(horizon=3, n_candidates=12, n_iters=2, seed=7)
    oh = np.random.default_rng(0).normal(size=(spec.history, spec.obs_dim)).astype(np.float32)
    ah = np.zeros(spec.history, np.int64)
    outs = []
    for _ in range(2):
        pl = LatentPlanner(model, asp.n, cfg)
        pl.reset(7)
        outs.append(pl.plan(oh, ah, None)[0])
    assert outs[0] == outs[1]


@pytest.mark.parametrize("method", ["cem", "beam", "shooting"])
def test_all_search_methods_produce_valid_actions(fitted_wm, tiny_env_cfg, method):
    model, spec, nz = fitted_wm
    asp = DiscreteActionSpace(tiny_env_cfg)
    pl = LatentPlanner(model, asp.n,
                       PlannerConfig(horizon=3, n_candidates=10, n_iters=2,
                                     beam_width=3, beam_branch=4, method=method))
    a, _ = pl.plan(np.zeros((spec.history, spec.obs_dim), np.float32),
                   np.zeros(spec.history, np.int64), None)
    assert 0 <= a < asp.n


def test_planner_prefers_the_action_its_model_rewards(fitted_wm, tiny_env_cfg):
    """With a rigged reward head the planner must find the favoured action."""
    import torch
    model, spec, nz = fitted_wm
    asp = DiscreteActionSpace(tiny_env_cfg)
    target = 11

    class Rigged:
        cfg = model.cfg
        meter = model.meter

        def encode_numpy(self, oh, ah, nh=None):
            return {"z": torch.zeros(oh.shape[0], 1)}

        def imagine_sequences(self, state, actions):
            r = (actions == target).float()
            return {"reward": r, "event_logit": torch.zeros(*actions.shape, 10),
                    "obs": torch.zeros(*actions.shape, spec.obs_dim)}

    from bwm.models.world_model.base import LatentWorldModel
    Rigged.expand_state = staticmethod(LatentWorldModel.expand_state)
    pl = LatentPlanner(Rigged(), asp.n,
                       PlannerConfig(horizon=4, n_candidates=64, n_iters=4, seed=1))
    a, _ = pl.plan(np.zeros((spec.history, spec.obs_dim), np.float32),
                   np.zeros(spec.history, np.int64), None)
    assert a == target


def test_analytic_planner_needs_no_model(tiny_env_cfg):
    from bwm.environment.observation import ObservationBuilder
    ob = ObservationBuilder(tiny_env_cfg)
    asp = DiscreteActionSpace(tiny_env_cfg)
    p = AnalyticPlannerPolicy(ob.feature_names, asp.names)
    obs = np.zeros(ob.obs_dim, np.float32)
    obs[ob.feature_names.index("inv_health")] = 0.9      # near liquidation
    ctx = DecisionContext(obs_hist=np.tile(obs, (4, 1)), act_hist=np.zeros(4, np.int64),
                          t=0, agent=0, action_space=asp,
                          feasible=np.ones(asp.n, bool), raw_obs=obs)
    assert asp.names[p.act(ctx)].startswith(("REPAY", "SUPPLY"))


# ------------------------------------------------------------------ memory
def test_memory_recalls_the_action_that_worked():
    m = EpisodicMemory(key_dim=4, capacity=64, n_actions=5)
    good = np.array([1.0, 0.0, 0.0, 0.0], np.float32)
    for _ in range(10):
        m.add(good, 3, 5.0, np.zeros(10), 0)
        m.add(-good, 1, -5.0, np.zeros(10), 0)
    vals, sup = m.action_values(good, k=8)
    assert int(np.argmax(vals)) == 3 and sup[3] > 0


def test_memory_is_bounded_and_evicts_oldest():
    m = EpisodicMemory(key_dim=2, capacity=5, n_actions=3)
    for i in range(20):
        m.add(np.array([i, 0.0]), i % 3, float(i), np.zeros(10), i)
    assert len(m) == 5


def test_consequence_tracker_learns_action_values_and_model_error():
    c = ConsequenceTracker(4)
    for i in range(200):
        c.update_action(i % 4, float(i % 4))
    assert int(np.argmax(c.action_value)) == 3
    c.update_model_error(1.0)
    c.update_model_error(1.0)
    assert c.model_error_ema == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------- LLM
def test_llm_parse_is_robust():
    assert parse_action('{"action_index": 5}', 94)[:1] == (5,)
    assert parse_action("I pick action_index 12 because", 94)[0] == 12
    assert parse_action("total nonsense", 94) == (0, "total nonsense"[:200], False)
    assert parse_action('{"action_index": 500}', 94)[2] is False


def test_llm_policy_uses_the_mock_transport_and_counts_tokens(tiny_env_cfg):
    from bwm.environment.observation import ObservationBuilder
    ob = ObservationBuilder(tiny_env_cfg)
    asp = DiscreteActionSpace(tiny_env_cfg)
    client = MockClient(LLMConfig(provider="mock", cache_dir=None), n_actions=asp.n)
    p = LLMPolicy(ob.feature_names, asp.names, LLMConfig(provider="mock", cache_dir=None),
                  client=client)
    obs = np.zeros(ob.obs_dim, np.float32)
    ctx = DecisionContext(obs_hist=np.tile(obs, (4, 1)), act_hist=np.zeros(4, np.int64),
                          t=3, agent=0, action_space=asp,
                          feasible=np.ones(asp.n, bool), raw_obs=obs,
                          info={"raw_obs_hist": np.tile(obs, (4, 1))})
    a = p.act(ctx)
    assert 0 <= a < asp.n
    assert p.meter.llm_calls == 1 and p.meter.llm_prompt_tokens > 0
    assert p.backend == "mock"


def test_offline_backend_is_labelled_as_a_surrogate(tiny_env_cfg):
    from bwm.environment.observation import ObservationBuilder
    ob = ObservationBuilder(tiny_env_cfg)
    asp = DiscreteActionSpace(tiny_env_cfg)
    p = LLMPolicy(ob.feature_names, asp.names,
                  LLMConfig(provider="anthropic", cache_dir=None))
    if p.client.is_offline:
        assert p.backend == "rule_based_offline"
        assert "NOT an LLM result" in p.info().notes


def test_prompt_contains_every_feature_and_the_action_menu(tiny_env_cfg):
    from bwm.environment.observation import ObservationBuilder
    ob = ObservationBuilder(tiny_env_cfg)
    asp = DiscreteActionSpace(tiny_env_cfg)
    sp = StatePresenter(ob.feature_names, asp.names, max_history=4)
    txt = sp.render(np.zeros((4, ob.obs_dim), np.float32), np.zeros(4, np.int64),
                    np.ones(asp.n, bool), t=7)
    for n in ob.feature_names:
        assert n in txt
    assert "AVAILABLE ACTIONS" in txt and "NOOP" in txt


# ----------------------------------------------------------------- unified
def test_gate_learns_to_prefer_the_better_pathway():
    g = LinUCBGate(3, 9, alpha=0.4)
    rng = np.random.default_rng(0)
    for _ in range(600):
        x = rng.normal(size=9); x[0] = 1.0
        a = g.select(x)
        g.update(a, x, [0.0, 1.0, -1.0][a] + 0.05 * rng.normal())
    assert g.usage()["share"][1] > g.usage()["share"][2]


def test_unified_ablations_restrict_the_pathways(fitted_wm, tiny_env_cfg):
    from bwm.environment.observation import ObservationBuilder
    model, spec, nz = fitted_wm
    ob = ObservationBuilder(tiny_env_cfg)
    asp = DiscreteActionSpace(tiny_env_cfg)
    reasoner = LLMPolicy(ob.feature_names, asp.names, LLMConfig(provider="offline"))
    full = UnifiedPolicy(model, asp, nz, reasoner, ob.feature_names, name="unified")
    no_mem = UnifiedPolicy(model, asp, nz, reasoner, ob.feature_names,
                           name="unified_no_memory", use_memory=False)
    no_roll = UnifiedPolicy(model, asp, nz, reasoner, ob.feature_names,
                            name="unified_no_rollout", use_planning=False)
    assert full._allowed_arms() == [0, 1, 2]
    assert 2 not in no_mem._allowed_arms()
    assert 1 not in no_roll._allowed_arms()
    assert len(PATHWAYS) == 3


def test_gate_offline_fit_recovers_the_better_arm():
    """A batch-fitted gate must generalise to contexts it was not fitted on."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 9)); X[:, 0] = 1.0
    arms = rng.integers(0, 3, 600)
    true = lambda x, a: [0.0, 0.9 * x[1], -0.6][a]
    rew = np.array([true(x, a) for x, a in zip(X, arms)]) + 0.05 * rng.normal(size=600)
    g = LinUCBGate(3, 9, alpha=0.4).fit(X, arms, rew)

    Xt = rng.normal(size=(300, 9)); Xt[:, 0] = 1.0
    chosen = np.array([g.select(x) for x in Xt])
    best = np.array([int(np.argmax([true(x, a) for a in range(3)])) for x in Xt])
    assert (chosen == best).mean() > 0.9, "offline-fit gate did not learn the policy"


def test_gate_state_round_trips():
    g = LinUCBGate(3, 9)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 9)); X[:, 0] = 1.0
    g.fit(X, rng.integers(0, 3, 50), rng.normal(size=50))
    g2 = LinUCBGate(3, 9).load_state_dict(g.state_dict())
    assert np.allclose(g2.A, g.A) and np.allclose(g2.b, g.b)
    assert np.array_equal(g2.counts, g.counts)


def test_gate_persistence_flag_controls_reset(tiny_env_cfg, fitted_wm):
    """persist_gate must survive reset(); the default must still clear it."""
    from bwm.environment.observation import ObservationBuilder
    model, spec, nz = fitted_wm
    ob = ObservationBuilder(tiny_env_cfg)
    asp = DiscreteActionSpace(tiny_env_cfg)
    reasoner = LLMPolicy(ob.feature_names, asp.names, LLMConfig(provider="offline"))
    x = np.ones(UnifiedPolicy.GATE_DIM)
    for persist, expect_kept in ((True, True), (False, False)):
        u = UnifiedPolicy(model, asp, nz, reasoner, ob.feature_names,
                          name="u", persist_gate=persist)
        u.gate.update(1, x, 1.0)
        u.reset(episode_seed=0)
        kept = int(u.gate.counts.sum()) > 0
        assert kept is expect_kept, f"persist_gate={persist} behaved wrongly"


def test_gate_reset_restores_the_original_prior():
    """reset() must be idempotent across episodes.

    It used to recover the ridge by reading A[0][0, 0], which stops being the
    ridge after the first update because the context's bias term adds 1.0 per
    pull. The restored prior then compounded every episode, freezing the gate
    and making evaluation results depend on episode ORDER.
    """
    g = LinUCBGate(3, 9, alpha=0.4, ridge=1.0)
    x = np.ones(9)
    for _ in range(3):
        for _ in range(50):
            g.update(0, x, 0.1)
        g.reset()
        assert g.A[0][0, 0] == pytest.approx(1.0), "ridge drifted across resets"
        assert g.counts.sum() == 0 and np.allclose(g.b, 0.0)


def test_gate_selection_is_order_independent_after_reset():
    """Two gates given the same episode must agree, regardless of history."""
    rng = np.random.default_rng(0)
    warm, fresh = LinUCBGate(3, 9, alpha=0.4), LinUCBGate(3, 9, alpha=0.4)
    for _ in range(80):                       # give one of them a prior episode
        warm.update(int(rng.integers(3)), rng.normal(size=9), float(rng.normal()))
    warm.reset()
    probe = rng.normal(size=(40, 9)); probe[:, 0] = 1.0
    assert [warm.select(x) for x in probe] == [fresh.select(x) for x in probe]
