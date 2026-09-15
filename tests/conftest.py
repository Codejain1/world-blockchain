"""Shared fixtures: a tiny synthetic dataset so tests never need the real corpus."""

from __future__ import annotations

import numpy as np
import pytest

from bwm.data.dataset import BatchSpec, Normalizer, TrajectorySet
from bwm.environment.observation import ObservationBuilder
from bwm.environment.runner import BehaviorPolicy, EpisodeRunner
from bwm.environment.scenarios import base_config, sample_train_config


@pytest.fixture(scope="session")
def tiny_env_cfg():
    return base_config(n_agents=40, episode_length=48)


@pytest.fixture(scope="session")
def obs_builder(tiny_env_cfg):
    return ObservationBuilder(tiny_env_cfg)


def _episodes(n: int, seed0: int, steps: int = 48):
    out = []
    for i in range(n):
        cfg = sample_train_config(seed0 + i, n_agents=40, episode_length=steps)
        r = EpisodeRunner(cfg)
        out.append(r.rollout(BehaviorPolicy(r.action_space, 0.5, seed=seed0 + i),
                             seed=seed0 + i))
    return out


@pytest.fixture(scope="session")
def tiny_train():
    return _episodes(4, 10_000)


@pytest.fixture(scope="session")
def tiny_val():
    return _episodes(2, 20_000)


@pytest.fixture(scope="session")
def tiny_set(tiny_train, tiny_val):
    ts = TrajectorySet(tiny_train, "train")
    vs = TrajectorySet(tiny_val, "val")
    spec = ts.spec(history=4, horizon=3)
    nz = Normalizer.fit(tiny_train)
    return ts, vs, spec, nz
