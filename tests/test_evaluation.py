"""Evaluation harness: metrics, counterfactual ground truth, control fairness."""

from __future__ import annotations

import numpy as np
import pytest

from bwm.data.dataset import Normalizer
from bwm.environment.scenarios import sample_train_config
from bwm.evaluation.benchmark import (COMPONENTS, brier_skill, clip01,
                                      composite_index, to_markdown)
from bwm.evaluation.control import run_policy_episode, summarize_control, regret_table
from bwm.evaluation.counterfactual import collect_probes, evaluate_counterfactual
from bwm.evaluation.metrics import (auroc, brier, expected_calibration_error,
                                    log_loss, max_drawdown, skill_score)
from bwm.evaluation.prediction import evaluate_prediction, reference_predictions
from bwm.experiments.registry import build_predictive_model
from bwm.models.baselines.policies import NoopPolicy, RandomPolicy
from bwm.training.trainer import TrainConfig

FAST = TrainConfig(epochs=1, max_batches_per_epoch=2, batch_size=32, val_batches=1)


# ------------------------------------------------------------------ metrics
def test_metric_edge_cases():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    perfect = np.array([0.001, 0.999, 0.001, 0.999])
    assert log_loss(y, perfect) < 0.01
    assert brier(y, perfect) < 0.001
    assert auroc(y, perfect) == pytest.approx(1.0)
    assert np.isnan(auroc(np.zeros(4), perfect))          # single-class label
    # A constant 0.5 forecast on a 50% base rate is perfectly calibrated ...
    assert expected_calibration_error(y, np.full(4, 0.5)) == pytest.approx(0.0)
    # ... but the same forecast on an event that never happens is not.
    assert expected_calibration_error(np.zeros(4), np.full(4, 0.5)) == pytest.approx(0.5)
    assert max_drawdown(np.array([1.0, 2.0, 1.0])) == pytest.approx(0.5)
    assert skill_score(y, y, np.zeros_like(y)) == pytest.approx(1.0)


def test_brier_skill_is_zero_for_the_base_rate_forecast():
    rate = 0.2
    y = np.random.default_rng(0).binomial(1, rate, 20_000).astype(float)
    bs = brier(y, np.full_like(y, rate))
    assert brier_skill(bs, [rate]) == pytest.approx(0.0, abs=0.02)


def test_composite_never_hides_missing_components():
    c = composite_index({}, None, None)
    assert c["composite"] is None and c["coverage"] == 0.0
    assert all(v is None for v in c["components"].values())
    assert clip01(float("nan")) is None
    assert clip01(5.0) == 1.0


def test_markdown_renders_missing_values_as_dashes():
    md = to_markdown([{"a": 1.0, "b": None}])
    assert "--" in md and "1.0000" in md


# ---------------------------------------------------------- counterfactual
def test_counterfactual_ground_truth_is_causal_not_rng_drift(tiny_train):
    """A null intervention must produce exactly zero measured effect."""
    nz = Normalizer.fit(tiny_train)
    cfg = sample_train_config(424242, n_agents=40, episode_length=120)
    probes = collect_probes(cfg, nz, history=4, horizon=3, n_probes=2, seed=424242,
                            burn_in=20, stride=10, kind="substitute")
    assert probes
    for p in probes:
        # The intervened action differs from the factual one ...
        assert p.cf_actions[0] != p.fact_actions[0] or np.allclose(p.true_effect, 0.0)
        # ... and the tail is shared, isolating the intervention.
        assert np.array_equal(p.cf_actions[1:], p.fact_actions[1:])


def test_zero_effect_predictor_scores_zero_skill(tiny_train, tiny_set, obs_builder):
    ts, vs, spec, nz = tiny_set
    cfg = sample_train_config(515151, n_agents=40, episode_length=120)
    probes = collect_probes(cfg, nz, history=spec.history, horizon=spec.horizon,
                            n_probes=3, seed=515151, burn_in=20, stride=10)
    m = build_predictive_model("persistence", spec, obs_builder, FAST)
    m.fit(ts, vs, spec, nz)
    r = evaluate_counterfactual(m, probes)
    # Persistence predicts the same future whatever the action, so its measured
    # causal effect is identically zero and its skill must be exactly zero.
    assert r["skill_vs_zero_effect"] == pytest.approx(0.0, abs=1e-9)
    assert r["true_effect_rms"] > 0.0, "the intervention had no measurable effect"


def test_counterfactual_effects_are_reproducible(tiny_train):
    nz = Normalizer.fit(tiny_train)
    cfg = sample_train_config(606060, n_agents=40, episode_length=120)
    kw = dict(history=4, horizon=3, n_probes=2, seed=606060, burn_in=20, stride=10)
    a = collect_probes(cfg, nz, **kw)
    b = collect_probes(sample_train_config(606060, n_agents=40, episode_length=120),
                       nz, **kw)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert np.array_equal(x.true_effect, y.true_effect)


# ----------------------------------------------------------------- control
def test_every_policy_gets_exactly_one_env_step_per_decision(tiny_train):
    nz = Normalizer.fit(tiny_train)
    cfg = sample_train_config(717171, n_agents=40, episode_length=64)
    for pol in (NoopPolicy(), RandomPolicy(3)):
        r = run_policy_episode(pol, cfg, nz, history=4, steps=32, seed=717171)
        assert r.rewards.size == 32
        assert pol.meter.env_steps == 32
        assert pol.meter.n_decisions == 32


def test_control_is_reproducible_for_a_fixed_seed(tiny_train):
    nz = Normalizer.fit(tiny_train)
    cfg = sample_train_config(818181, n_agents=40, episode_length=64)
    a = run_policy_episode(RandomPolicy(5), cfg, nz, history=4, steps=32, seed=818181)
    b = run_policy_episode(RandomPolicy(5),
                           sample_train_config(818181, n_agents=40, episode_length=64),
                           nz, history=4, steps=32, seed=818181)
    assert np.array_equal(a.actions, b.actions)
    assert np.allclose(a.rewards, b.rewards)


def test_noop_policy_never_acts_and_summary_is_well_formed(tiny_train):
    nz = Normalizer.fit(tiny_train)
    cfg = sample_train_config(919191, n_agents=40, episode_length=64)
    res = [run_policy_episode(NoopPolicy(), cfg, nz, history=4, steps=24, seed=919191)]
    assert (res[0].actions == 0).all()
    s = summarize_control(res)
    for k in ("total_return", "sharpe", "max_drawdown", "adaptation_delta"):
        assert k in s and np.isfinite(s[k])


def test_regret_is_zero_for_the_best_policy_on_every_episode():
    from bwm.evaluation.control import ControlResult
    def mk(name, ret):
        nw = np.array([100.0, 100.0 * (1 + ret)], np.float32)
        return ControlResult(name, "iid", 1, np.array([ret * 100], np.float32),
                             nw, np.zeros((1, 10), np.float32),
                             np.zeros(1, np.int64), 0.0)
    r = regret_table({"good": [mk("good", 0.1)], "bad": [mk("bad", -0.1)]})
    assert r["good"] == pytest.approx(0.0)
    assert r["bad"] == pytest.approx(0.2, abs=1e-6)


# -------------------------------------------------------------- prediction
def test_reference_scoring_makes_the_reference_score_zero(tiny_set, obs_builder):
    ts, vs, spec, nz = tiny_set
    ar = build_predictive_model("ar", spec, obs_builder, FAST, {"max_samples": 500})
    ar.fit(ts, vs, spec, nz)
    idx = ts.index(spec.history, spec.horizon)[:64]
    batch = ts.batch(idx, spec, nz, with_graph=False)
    ref = reference_predictions(ar, batch, spec.horizon)
    out = evaluate_prediction(ar, batch, spec, obs_builder.feature_names,
                              horizons=[1], reference=ref)
    assert out["next_state"]["skill_vs_ar"] == pytest.approx(0.0, abs=1e-9)
    for g, row in out["groups"].items():
        assert row["skill_vs_ar"] == pytest.approx(0.0, abs=1e-9), g
        # Scored against the *better* of the two naive references, the AR model
        # can only be <= 0: persistence may beat it on some groups.
        assert row["skill_vs_best_naive"] <= 1e-9, g


def test_target_groups_partition_the_observation_vector(obs_builder):
    from bwm.evaluation.prediction import COMPOSITE_GROUPS, group_indices
    g = group_indices(obs_builder.feature_names)
    # price_level is a subset of price, the rest partition the vector exactly.
    core = np.concatenate([g[k] for k in ("price", "protocol", "agent", "population")])
    assert sorted(core.tolist()) == list(range(obs_builder.obs_dim))
    assert set(g["price_level"].tolist()) <= set(g["price"].tolist())
    assert set(COMPOSITE_GROUPS) == {"agent", "protocol"}


# ------------------------------------------------------------ significance
def test_paired_tests_detect_a_real_effect_and_reject_noise():
    from bwm.evaluation.stats import paired_bootstrap, paired_permutation_test
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, 14)
    b = a - 0.4 + rng.normal(0, 0.02, 14)          # consistent paired effect
    bs = paired_bootstrap(a, b, seed=0)
    assert bs["lo"] > 0 and bs["diff"] == pytest.approx(0.4, abs=0.05)
    assert paired_permutation_test(a, b, seed=0) < 0.01

    c = rng.normal(0.0, 1.0, 14)                    # identical distributions
    d = rng.normal(0.0, 1.0, 14)
    assert paired_permutation_test(c, d, seed=0) > 0.05


def test_permutation_test_is_exact_and_bounded_for_small_n():
    from bwm.evaluation.stats import paired_permutation_test
    # With n=4 the smallest attainable two-sided p-value is 2/2^4 = 0.125,
    # so no per-scenario claim at n=4 can reach p<0.05.  Worth asserting so
    # nobody reads a 4-episode result as significant.
    assert paired_permutation_test([1, 1, 1, 1], [0, 0, 0, 0]) == pytest.approx(0.125)
    assert paired_permutation_test([1] * 8, [0] * 8) == pytest.approx(2 / 256)


def test_compare_to_reference_pairs_by_episode():
    from bwm.evaluation.control import ControlResult
    from bwm.evaluation.stats import compare_to_reference

    def mk(name, sc, seed, ret):
        nw = np.array([100.0, 100.0 * (1 + ret)], np.float32)
        return ControlResult(name, sc, seed, np.array([ret * 100], np.float32), nw,
                             np.zeros((1, 10), np.float32), np.zeros(1, np.int64), 0.0)

    per = {"good": {"iid": [mk("good", "iid", s, 0.1) for s in range(6)]},
           "noop": {"iid": [mk("noop", "iid", s, 0.0) for s in range(6)]}}
    rows = compare_to_reference(per, "noop")
    assert len(rows) == 1
    assert rows[0]["policy"] == "good"
    assert rows[0]["mean_diff"] == pytest.approx(0.1, abs=1e-6)
    assert rows[0]["n_pairs"] == 6
