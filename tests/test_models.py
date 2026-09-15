"""Model interfaces: every system fits, predicts, rolls out and stays finite."""

from __future__ import annotations

import numpy as np
import pytest

from bwm.data.dataset import Normalizer
from bwm.environment.action_space import DiscreteActionSpace
from bwm.experiments.registry import (GRAPH_MODELS, PREDICTIVE_MODELS,
                                      build_predictive_model)
from bwm.models.base import RolloutPrediction, StepPrediction
from bwm.training.trainer import TrainConfig

FAST = TrainConfig(epochs=1, max_batches_per_epoch=2, batch_size=32, val_batches=1)
NAMES = [n for n in PREDICTIVE_MODELS if n != "gbdt"]   # gbdt is covered separately


@pytest.mark.parametrize("name", NAMES)
def test_every_model_fits_predicts_and_rolls_out(name, tiny_set, obs_builder):
    ts, vs, spec, nz = tiny_set
    overrides = {"max_samples": 500} if name in ("ar", "linear") else {}
    if name in ("gnn", "wm_graph"):
        overrides = {"d_model": 16, "n_gnn_layers": 1}
    model = build_predictive_model(name, spec, obs_builder, FAST, overrides)
    log = model.fit(ts, vs, spec, nz)
    assert isinstance(log, dict)

    idx = ts.index(spec.history, spec.horizon)[:16]
    batch = ts.batch(idx, spec, nz, with_graph=True)

    p = model.predict_step(batch)
    assert isinstance(p, StepPrediction)
    assert p.delta.shape == (16, spec.obs_dim)
    assert p.event_logit.shape == (16, spec.n_events)
    assert p.reward.shape == (16,)
    for arr in (p.delta, p.event_logit, p.reward):
        assert np.isfinite(arr).all(), f"{name} produced non-finite predictions"

    r = model.predict_rollout(batch, spec.horizon)
    assert isinstance(r, RolloutPrediction)
    assert r.obs.shape == (16, spec.horizon, spec.obs_dim)
    assert np.isfinite(r.obs).all()


def test_gbdt_backend_available_and_fits(tiny_set, obs_builder):
    ts, vs, spec, nz = tiny_set
    m = build_predictive_model("gbdt", spec, obs_builder, FAST,
                               {"n_estimators": 5, "max_samples": 400,
                                "top_k_targets": 3})
    log = m.fit(ts, vs, spec, nz)
    assert log["backend"] in ("lightgbm", "xgboost")
    idx = ts.index(spec.history, spec.horizon)[:8]
    p = m.predict_step(ts.batch(idx, spec, nz, with_graph=False))
    assert np.isfinite(p.event_logit).all()


def test_persistence_predicts_zero_change_and_base_rates(tiny_set, obs_builder):
    ts, vs, spec, nz = tiny_set
    m = build_predictive_model("persistence", spec, obs_builder, FAST)
    m.fit(ts, vs, spec, nz)
    idx = ts.index(spec.history, spec.horizon)[:8]
    p = m.predict_step(ts.batch(idx, spec, nz, with_graph=False))
    assert np.allclose(p.delta, 0.0)
    probs = 1.0 / (1.0 + np.exp(-p.event_logit))
    ev = np.concatenate([t.events for t in ts.trajs], axis=0).mean(axis=0)
    assert np.allclose(probs[0], np.clip(ev, 1e-4, 1 - 1e-4), atol=1e-3)


@pytest.mark.parametrize("name", ["wm_rssm", "wm_jepa", "wm_transformer"])
def test_latent_rollout_does_not_touch_the_simulator(name, tiny_set, obs_builder):
    """Imagination must be a pure function of latents and actions."""
    ts, vs, spec, nz = tiny_set
    m = build_predictive_model(name, spec, obs_builder, FAST)
    m.fit(ts, vs, spec, nz)
    idx = ts.index(spec.history, spec.horizon)[:8]
    b = ts.batch(idx, spec, nz, with_graph=False)
    r1 = m.predict_rollout(b, spec.horizon)
    r2 = m.predict_rollout(b, spec.horizon)
    assert np.allclose(r1.obs, r2.obs), "latent rollout is not deterministic"
    # A different action sequence must change the imagined future.
    b2 = dict(b)
    b2["fut_act"] = (b["fut_act"] + 7) % spec.n_actions
    r3 = m.predict_rollout(b2, spec.horizon)
    assert not np.allclose(r1.obs, r3.obs), "imagination ignores the action"


def test_jepa_latents_do_not_collapse(tiny_set, obs_builder):
    ts, vs, spec, nz = tiny_set
    cfg = TrainConfig(epochs=2, max_batches_per_epoch=6, batch_size=32, val_batches=1)
    m = build_predictive_model("wm_jepa", spec, obs_builder, cfg)
    m.fit(ts, vs, spec, nz)
    idx = ts.index(spec.history, spec.horizon)[:64]
    std = m.collapse_diagnostic(ts.batch(idx, spec, nz, with_graph=False))
    assert std > 1e-3, f"JEPA latent collapsed (std={std})"


def test_one_step_world_model_uses_horizon_one(tiny_set, obs_builder):
    ts, vs, spec, nz = tiny_set
    m = build_predictive_model("wm_rssm_1step", spec, obs_builder, FAST)
    log = m.fit(ts, vs, spec, nz)
    assert log["train_horizon"] == 1


def test_parameter_counts_are_within_one_order_of_magnitude(tiny_set, obs_builder):
    """Guards the fairness claim that no system has a capacity blowout."""
    ts, vs, spec, nz = tiny_set
    counts = {}
    for n in ("mlp", "transformer", "obsspace", "wm_rssm", "wm_jepa", "wm_transformer"):
        counts[n] = build_predictive_model(n, spec, obs_builder, FAST).n_params()
    lo, hi = min(counts.values()), max(counts.values())
    assert hi / max(lo, 1) < 10.0, counts
