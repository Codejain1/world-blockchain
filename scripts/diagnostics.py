#!/usr/bin/env python3
"""Representation health checks for trained world models.

Answers one question the benchmark cannot: *is the latent space actually
carrying information?*  A collapsed representation can still post respectable
observation MSE through its decoder while being useless for planning, so a world
model that loses on the benchmark should be checked for collapse before the loss
is attributed to the idea rather than to the fit.

Reports per model:

``latent_std``        mean per-dimension standard deviation of the encoded latent.
                      Near zero means collapse.
``effective_dim``     participation ratio of the latent covariance spectrum,
                      i.e. how many dimensions are really being used.
``action_sensitivity`` mean absolute change in the *predicted next observation*
                      when the action is swapped, relative to the prediction's
                      own scale.  Zero means the model ignores actions, which
                      makes counterfactual reasoning impossible by construction.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

from bwm.data.dataset import Normalizer, TrajectorySet
from bwm.data.generate import load_split
from bwm.environment.observation import ObservationBuilder
from bwm.environment.scenarios import base_config
from bwm.experiments.registry import GRAPH_MODELS, build_predictive_model
from bwm.experiments.train import _train_cfg, load_data
from bwm.utils.config import load_config


def effective_dim(z: np.ndarray) -> float:
    """Participation ratio: (sum eig)^2 / sum(eig^2).  Ranges 1..dim."""
    z = z - z.mean(axis=0, keepdims=True)
    cov = np.cov(z, rowvar=False)
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 0.0, None)
    s1, s2 = ev.sum(), (ev ** 2).sum()
    return float(s1 ** 2 / max(s2, 1e-12))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--results", default="results/main")
    ap.add_argument("--n", type=int, default=512)
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    ts, vs, spec, _ = load_data(cfg)
    nz = Normalizer.load(os.path.join(a.results, "normalizer.npz"))
    ob = ObservationBuilder(base_config(n_agents=int(ts.trajs[0].meta.get("n_agents", 120))))
    tc = _train_cfg(cfg)
    overrides = dict(cfg.get_path("models.overrides", {}) or {})

    idx = ts.index(spec.history, spec.horizon)
    rng = np.random.Generator(np.random.PCG64(0))
    sel = idx[rng.choice(len(idx), min(a.n, len(idx)), replace=False)]
    batch = ts.batch(sel, spec, nz, with_graph=True)

    rows = []
    for name in cfg.get_path("models.predictive", []):
        ckpt = os.path.join(a.results, "checkpoints", f"{name}.pt")
        if not name.startswith(("wm_", "dyn_", "obsspace")) or not os.path.exists(ckpt):
            continue
        model = build_predictive_model(name, spec, ob, tc, overrides.get(name))
        model.load(ckpt)
        model.spec, model.normalizer = spec, nz
        mod = model.module
        b = {k: (torch.as_tensor(v) if v.dtype != np.int64
                 else torch.as_tensor(v, dtype=torch.long)) for k, v in batch.items()}
        with torch.no_grad():
            state = mod.encode(b)
            z = state.get("z")
            if z is None:
                z = torch.cat([state[k].flatten(1) for k in ("h", "s")
                               if k in state], dim=-1) if "h" in state else None
            if z is None and "h" in state:
                z = state["h"].flatten(1)
            zn = z.cpu().numpy() if z is not None else None

            p1 = model.predict_step(batch)
            alt = dict(batch)
            alt["act_hist"] = batch["act_hist"].copy()
            alt["act_hist"][:, -1] = (alt["act_hist"][:, -1] + 37) % spec.n_actions
            p2 = model.predict_step(alt)
            sens = float(np.mean(np.abs(p1.delta - p2.delta)))
            scale = float(np.mean(np.abs(p1.delta))) + 1e-12

        rows.append({
            "model": name,
            "latent_dim": int(zn.shape[1]) if zn is not None else None,
            "latent_std": float(zn.std(axis=0).mean()) if zn is not None else None,
            "effective_dim": effective_dim(zn) if zn is not None else None,
            "action_sensitivity": sens,
            "action_sensitivity_rel": sens / scale,
        })
        r = rows[-1]
        print(f"  {name:<18s} latent_dim={r['latent_dim']!s:>5s} "
              f"std={r['latent_std']:.4f} eff_dim={r['effective_dim']:.1f} "
              f"action_sens={r['action_sensitivity']:.5f} "
              f"(rel {r['action_sensitivity_rel']:.3f})")
        if r["latent_std"] is not None and r["latent_std"] < 1e-3:
            print(f"    WARNING: {name} latent has collapsed.")
        if r["action_sensitivity_rel"] < 1e-3:
            print(f"    WARNING: {name} predictions barely depend on the action; "
                  f"counterfactual reasoning is impossible for it.")

    out = os.path.join(a.results, "diagnostics.json")
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
