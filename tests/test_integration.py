"""End-to-end integration: observation -> prediction -> planning -> action -> consequence."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from bwm.data.audit import audit_observation_overlap, audit_split_disjointness
from bwm.data.dataset import Normalizer, TrajectorySet
from bwm.data.generate import GenSpec, generate_dataset, load_manifest, load_split
from bwm.environment.scenarios import sample_train_config
from bwm.evaluation.audit import audit_results, format_audit
from bwm.evaluation.control import run_policy_episode
from bwm.evaluation.counterfactual import collect_probes, evaluate_counterfactual
from bwm.evaluation.prediction import evaluate_prediction, sample_eval_batches
from bwm.experiments.evaluate import run_evaluation
from bwm.experiments.registry import build_policies, build_predictive_model
from bwm.experiments.train import save_bundle, train_all
from bwm.models.llm.client import LLMConfig
from bwm.planning.planner import PlannerConfig
from bwm.training.trainer import TrainConfig
from bwm.utils.config import load_config

FAST = TrainConfig(epochs=1, max_batches_per_epoch=2, batch_size=32, val_batches=1)


def test_complete_decision_cycle(tiny_set, obs_builder, tiny_env_cfg):
    """One episode through the whole stack, with the world model driving."""
    from bwm.environment.action_space import DiscreteActionSpace
    from bwm.models.world_model.policy import WorldModelPolicy

    ts, vs, spec, nz = tiny_set
    wm = build_predictive_model("wm_rssm", spec, obs_builder, FAST)
    wm.fit(ts, vs, spec, nz)

    asp = DiscreteActionSpace(tiny_env_cfg)
    policy = WorldModelPolicy(wm, asp, nz,
                              PlannerConfig(horizon=3, n_candidates=12, n_iters=2),
                              name="wm_plan")
    cfg = sample_train_config(313131, n_agents=40, episode_length=48)
    res = run_policy_episode(policy, cfg, nz, history=spec.history, steps=24,
                             seed=313131)

    assert res.rewards.size == 24
    assert res.net_worth.size == 25
    assert np.isfinite(res.rewards).all() and np.isfinite(res.net_worth).all()
    # the planner imagined, the simulator executed exactly 24 steps
    assert wm.meter.imagined_steps > 0
    assert policy.meter.env_steps == 24
    # consequences were fed back
    assert policy.tracker.model_error_n > 0
    assert policy.tracker.action_count.sum() > 0
    s = res.summary()
    for k in ("total_return", "sharpe", "max_drawdown", "adaptation_delta"):
        assert np.isfinite(s[k])


def test_unified_system_runs_and_uses_multiple_pathways(tiny_set, obs_builder,
                                                        tiny_env_cfg):
    from bwm.environment.action_space import DiscreteActionSpace

    ts, vs, spec, nz = tiny_set
    wm = build_predictive_model("wm_rssm", spec, obs_builder, FAST)
    wm.fit(ts, vs, spec, nz)
    asp = DiscreteActionSpace(tiny_env_cfg)
    pols = build_policies(["unified"], world_models={"wm_rssm": wm},
                          action_space=asp, obs_builder=obs_builder, normalizer=nz,
                          planner_cfg=PlannerConfig(horizon=3, n_candidates=8,
                                                    n_iters=2),
                          llm_cfg=LLMConfig(provider="offline", cache_dir=None))
    u = pols["unified"]
    cfg = sample_train_config(414141, n_agents=40, episode_length=48)
    run_policy_episode(u, cfg, nz, history=spec.history, steps=30, seed=414141)
    usage = u.gate.usage()
    assert sum(usage["counts"]) == 30
    assert len([c for c in usage["counts"] if c > 0]) >= 2, \
        "the gate collapsed onto a single pathway immediately"
    assert u.tracker.model_error_n > 0, "the world model was not scored on every step"


def test_generated_dataset_is_leakage_free(tmp_path):
    spec = GenSpec(n_train=3, n_val=2, n_test=2, n_ood=1, episode_length=32,
                   n_agents=30, out_dir=str(tmp_path / "ds"), n_workers=1,
                   scenarios=["liquidity_crisis", "novel_mechanism"])
    manifest = generate_dataset(spec, verbose=False)
    audit = audit_split_disjointness(manifest)
    assert audit["ok"], audit["collisions"]

    train = load_split(spec.out_dir, "train")
    for split in ("test_iid", "ood_liquidity_crisis"):
        other = load_split(spec.out_dir, split)
        o = audit_observation_overlap(train, other)
        assert o["overlap_frac_b"] < 0.01, (split, o)


def test_full_pipeline_end_to_end(tmp_path, monkeypatch):
    """Train, evaluate, audit and report -- the whole run, in miniature."""
    ds = tmp_path / "ds"
    spec = GenSpec(n_train=6, n_val=2, n_test=2, n_ood=2, episode_length=40,
                   n_agents=30, out_dir=str(ds), n_workers=1,
                   scenarios=["liquidity_crisis"])
    generate_dataset(spec, verbose=False)

    out = tmp_path / "run"
    cfg = load_config("configs/smoke.yaml", [
        f"data.dir={ds}", f"experiment.out_dir={out}",
        "data.max_train_episodes=6", "data.max_val_episodes=2",
        "data.history=4", "data.horizon=3",
        "train.epochs=1", "train.max_batches_per_epoch=2", "train.val_batches=1",
        "models.predictive=['persistence','ar','wm_rssm']",
        "eval.n_samples=120", "eval.horizons=[1,3]",
        "eval.splits=['test_iid','ood_liquidity_crisis']",
        "counterfactual.scenarios=['iid']", "counterfactual.n_episodes=1",
        "counterfactual.n_probes=2", "counterfactual.horizon=3",
        "counterfactual.kinds=['substitute']",
        "control.scenarios=['iid']", "control.n_episodes=1", "control.steps=20",
        "control.policies=['noop','wm_plan']",
        "planner.horizon=3", "planner.n_candidates=8", "planner.n_iters=2",
    ])
    bundle = train_all(cfg, verbose=False)
    save_bundle(bundle, str(out))
    results = run_evaluation(bundle, cfg, str(out), verbose=False)

    assert os.path.exists(out / "results.json")
    assert set(results["prediction"]) == {"persistence", "ar", "wm_rssm"}
    assert results["data_audit"]["seed_disjoint"] is True
    assert "audit" in results and "findings" in results["audit"]

    # persistence must score exactly zero causal skill -- the control that keeps
    # the counterfactual metric honest.
    for v in results["counterfactual"]["persistence"].values():
        assert v["skill_vs_zero_effect"] == pytest.approx(0.0, abs=1e-9)

    comp = results["composite"]
    assert comp["wm_rssm"]["coverage"] > 0
    # every reported component must be accompanied by its raw value
    for name, c in comp.items():
        for k, v in c["components"].items():
            assert v is None or 0.0 <= v <= 1.0, (name, k, v)

    from bwm.visualization.report import write_report
    write_report(results, str(out), make_plots=True)
    assert os.path.exists(out / "RESULTS.md")
    md = (out / "RESULTS.md").read_text()
    assert "Prediction" in md and "Contamination check" in md
    assert os.path.isdir(out / "figures")


def test_experiment_is_reproducible(tmp_path):
    """Two runs of the same config must produce identical numbers."""
    ds = tmp_path / "ds"
    spec = GenSpec(n_train=4, n_val=2, n_test=2, n_ood=1, episode_length=32,
                   n_agents=30, out_dir=str(ds), n_workers=1,
                   scenarios=["liquidity_crisis"])
    generate_dataset(spec, verbose=False)
    outs = []
    for i in range(2):
        cfg = load_config("configs/smoke.yaml", [
            f"data.dir={ds}", f"experiment.out_dir={tmp_path}/r{i}",
            "data.max_train_episodes=4", "data.max_val_episodes=2",
            "data.history=4", "data.horizon=3",
            "train.epochs=1", "train.max_batches_per_epoch=2", "train.val_batches=1",
            "models.predictive=['ar','wm_rssm']",
            "eval.n_samples=100", "eval.horizons=[1]", "eval.splits=['test_iid']",
        ])
        bundle = train_all(cfg, verbose=False)
        outs.append(run_evaluation(bundle, cfg, f"{tmp_path}/r{i}",
                                   stages=("prediction",), verbose=False))
    for m in ("ar", "wm_rssm"):
        a = outs[0]["prediction"][m]["test_iid"]["next_state"]["mse"]
        b = outs[1]["prediction"][m]["test_iid"]["next_state"]["mse"]
        assert a == pytest.approx(b, rel=1e-9), m


def test_audit_flags_an_implausible_result():
    bad = {"control": {"summaries": {"x": {"iid": {"total_return": 4.0}},
                                     "noop": {"iid": {"total_return": 0.0}},
                                     "oracle_plan": {"iid": {"total_return": 0.1}}}}}
    a = audit_results(bad)
    assert any(f["check"] == "implausible_return" for f in a["findings"])
    assert "implausible_return" in format_audit(a)
