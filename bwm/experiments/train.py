"""Training stage: fit every predictive system under one shared budget.

Fairness rules enforced here, not by convention:

* the normaliser is fitted on the **training split only** and reused everywhere;
* every learned system gets the same number of optimisation steps
  (``epochs x max_batches_per_epoch``) and the same batch size, window length and
  imagination horizon;
* wall-clock, parameter count and peak memory are recorded per system so a win
  bought with more compute is visible.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..data.dataset import BatchSpec, Normalizer, TrajectorySet
from ..data.generate import load_manifest, load_split
from ..environment.observation import ObservationBuilder
from ..environment.scenarios import base_config
from ..training.trainer import TrainConfig
from ..utils.config import Config
from ..utils.seeding import seed_everything
from .registry import GRAPH_MODELS, build_predictive_model

__all__ = ["TrainedBundle", "load_data", "train_all", "save_bundle", "load_bundle"]


class TrainedBundle:
    """Everything downstream stages need: fitted models, spec, normaliser."""

    def __init__(self, models: Dict[str, Any], spec: BatchSpec, normalizer: Normalizer,
                 obs_builder: ObservationBuilder, logs: Dict[str, Any],
                 cfg: Config) -> None:
        self.models = models
        self.spec = spec
        self.normalizer = normalizer
        self.obs_builder = obs_builder
        self.logs = logs
        self.cfg = cfg


def load_data(cfg: Config):
    d = cfg.get_path("data.dir", "datasets/main")
    H = int(cfg.get_path("data.history", 8))
    L = int(cfg.get_path("data.horizon", 6))
    train = load_split(d, "train", cfg.get_path("data.max_train_episodes"))
    val = load_split(d, "val", cfg.get_path("data.max_val_episodes"))
    ts, vs = TrajectorySet(train, "train"), TrajectorySet(val, "val")
    spec = ts.spec(history=H, horizon=L)
    normalizer = Normalizer.fit(train)          # train split only -- no leakage
    return ts, vs, spec, normalizer


def _train_cfg(cfg: Config) -> TrainConfig:
    t = cfg.sub("train")
    return TrainConfig(
        epochs=int(t.get("epochs", 15)), batch_size=int(t.get("batch_size", 256)),
        lr=float(t.get("lr", 1e-3)), weight_decay=float(t.get("weight_decay", 1e-5)),
        grad_clip=float(t.get("grad_clip", 1.0)), patience=int(t.get("patience", 5)),
        seed=int(t.get("seed", 0)), device=str(t.get("device", "cpu")),
        max_batches_per_epoch=int(t.get("max_batches_per_epoch", 150)),
        val_batches=int(t.get("val_batches", 30)))


def train_all(cfg: Config, verbose: bool = True) -> TrainedBundle:
    seed_everything(int(cfg.get_path("experiment.seed", 0)))
    ts, vs, spec, normalizer = load_data(cfg)
    env_cfg = base_config(n_agents=int(ts.trajs[0].meta.get("n_agents", 120)))
    obs_builder = ObservationBuilder(env_cfg)
    if obs_builder.obs_dim != spec.obs_dim:      # guard against silent drift
        raise ValueError(f"Observation dim mismatch: builder={obs_builder.obs_dim} "
                         f"dataset={spec.obs_dim}. Regenerate the dataset.")

    names: List[str] = list(cfg.get_path("models.predictive", []))
    overrides: Dict[str, Dict[str, Any]] = dict(cfg.get_path("models.overrides", {}) or {})
    tc = _train_cfg(cfg)
    models: Dict[str, Any] = {}
    logs: Dict[str, Any] = {}

    if verbose:
        n_steps = tc.epochs * tc.max_batches_per_epoch
        print(f"training {len(names)} systems | budget {n_steps} steps "
              f"| window H={spec.history} horizon L={spec.horizon} "
              f"| {len(ts)} train episodes", flush=True)

    for name in names:
        t0 = time.perf_counter()
        model = build_predictive_model(name, spec, obs_builder, tc, overrides.get(name))
        log = model.fit(ts, vs, spec, normalizer)
        dt = time.perf_counter() - t0
        models[name] = model
        meter = getattr(model, "meter", None)
        logs[name] = {
            "family": getattr(model, "family", "baseline"),
            "n_params": int(model.n_params()),
            "wall_seconds": dt,
            "best_val": log.get("best_val"),
            "needs_graph": name in GRAPH_MODELS,
            "train_log": {k: v for k, v in log.items() if k != "history"},
            "history": log.get("history", []),
            "peak_rss_mb": float(getattr(meter, "peak_rss_mb", 0.0)),
        }
        if verbose:
            bv = log.get("best_val")
            bv_s = f"{bv:.4f}" if isinstance(bv, float) else "n/a"
            print(f"  {name:<16s} params={model.n_params():>8d}  "
                  f"val={bv_s}  {dt:6.1f}s", flush=True)
    return TrainedBundle(models, spec, normalizer, obs_builder, logs, cfg)


def save_bundle(bundle: TrainedBundle, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    ck = os.path.join(out_dir, "checkpoints")
    os.makedirs(ck, exist_ok=True)
    bundle.normalizer.save(os.path.join(out_dir, "normalizer.npz"))
    with open(os.path.join(out_dir, "train_logs.json"), "w") as fh:
        json.dump(bundle.logs, fh, indent=2, default=float)
    with open(os.path.join(out_dir, "spec.json"), "w") as fh:
        json.dump(bundle.spec.to_dict(), fh, indent=2)
    bundle.cfg.save(os.path.join(out_dir, "config.yaml"))
    for name, m in bundle.models.items():
        try:
            m.save(os.path.join(ck, f"{name}.pt"))
        except NotImplementedError:
            pass
        except Exception:
            pass


def load_bundle(cfg: Config, out_dir: str, verbose: bool = False) -> TrainedBundle:
    """Rebuild fitted torch models from checkpoints (classical models refit)."""
    ts, vs, spec, _ = load_data(cfg)
    normalizer = Normalizer.load(os.path.join(out_dir, "normalizer.npz"))
    env_cfg = base_config(n_agents=int(ts.trajs[0].meta.get("n_agents", 120)))
    obs_builder = ObservationBuilder(env_cfg)
    tc = _train_cfg(cfg)
    overrides = dict(cfg.get_path("models.overrides", {}) or {})
    with open(os.path.join(out_dir, "train_logs.json")) as fh:
        logs = json.load(fh)
    models: Dict[str, Any] = {}
    for name in cfg.get_path("models.predictive", []):
        m = build_predictive_model(name, spec, obs_builder, tc, overrides.get(name))
        ckpt = os.path.join(out_dir, "checkpoints", f"{name}.pt")
        if os.path.exists(ckpt):
            m.load(ckpt)
            m.spec, m.normalizer = spec, normalizer
        else:
            if verbose:
                print(f"  refitting {name} (no checkpoint)", flush=True)
            m.fit(ts, vs, spec, normalizer)
        models[name] = m
    return TrainedBundle(models, spec, normalizer, obs_builder, logs, cfg)
