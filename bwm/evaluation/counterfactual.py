"""The counterfactual experiment.

Question 1  "What would happen if agent X took action A instead of B?"
Question 2  "What would have happened if A had not occurred?"  (B = do-nothing)

Ground truth is obtained by *forking the real simulator*.  Because exogenous
noise is counter-based (see :mod:`bwm.environment.world`), both branches
experience identical world noise, so the difference between them is the causal
effect of the intervention and nothing else.

The model is then asked the same question in imagination -- encode the history at
``t``, roll the two action sequences forward in latent space, and subtract.

The control that matters
------------------------
A model that predicts "no effect" gets a respectable MSE, because most single
actions barely move a deep market.  So the headline number is the **skill score
against the zero-effect predictor**.  Positive skill means the model has learned
something causal; zero or negative means it has not, however good its one-step
MSE looks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..agents.population import AgentPopulation
from ..data.dataset import Normalizer
from ..environment.action_space import DiscreteActionSpace
from ..environment.config import EnvConfig
from ..environment.observation import ObservationBuilder
from ..environment.runner import BehaviorPolicy, EpisodeRunner
from ..environment.world import BlockchainWorld
from .metrics import mse, r2, skill_score

__all__ = ["CounterfactualProbe", "collect_probes", "evaluate_counterfactual"]


@dataclass
class CounterfactualProbe:
    """One (state, factual action, counterfactual action) triple with ground truth."""

    obs_hist: np.ndarray        # (H, D) normalised history ending at t
    act_hist: np.ndarray        # (H,)
    node_hist: np.ndarray       # (H, N, F)
    fact_actions: np.ndarray    # (L,) actions actually taken from t onwards
    cf_actions: np.ndarray      # (L,) intervened sequence (differs at index 0)
    true_effect: np.ndarray     # (L, D) normalised obs difference cf - factual
    fact_obs: np.ndarray        # (L, D) normalised factual future
    cf_obs: np.ndarray          # (L, D)
    fact_reward: np.ndarray     # (L,)
    cf_reward: np.ndarray       # (L,)
    t: int
    scenario: str
    kind: str                   # "substitute" | "removal"


def _window(obs_buf: List[np.ndarray], H: int) -> np.ndarray:
    win = obs_buf[-H:]
    while len(win) < H:
        win = [win[0]] + win
    return np.stack(win, axis=0)


def collect_probes(cfg: EnvConfig, normalizer: Normalizer, history: int = 8,
                   horizon: int = 6, n_probes: int = 8, seed: int = 0,
                   burn_in: int = 40, stride: int = 16,
                   kind: str = "substitute") -> List[CounterfactualProbe]:
    """Run one episode and fork it at several points to build ground truth."""
    runner = EpisodeRunner(cfg)
    ob: ObservationBuilder = runner.obs_builder
    asp: DiscreteActionSpace = runner.action_space
    behav = BehaviorPolicy(asp, epsilon=0.35, seed=seed)
    rng = np.random.Generator(np.random.PCG64(seed ^ 0xCF))

    runner.reset(cfg.seed)
    world = runner.world
    agent = runner.focal_agent
    obs_buf: List[np.ndarray] = []
    act_buf: List[int] = []
    node_buf: List[np.ndarray] = []
    probes: List[CounterfactualProbe] = []
    total = burn_in + stride * n_probes + horizon + 2

    for t in range(total):
        obs_buf.append(runner.observe())
        node_buf.append(runner.graph())
        a = int(behav(world, agent, runner.population))

        if t >= burn_in and (t - burn_in) % stride == 0 and len(probes) < n_probes:
            feas = asp.feasible_mask(world, agent)
            legal = np.flatnonzero(feas)
            if kind == "removal":
                a_fact = a if a != 0 else int(rng.choice(legal[legal != 0])) \
                    if (legal != 0).any() else 0
                a_cf = 0                                    # "A did not occur"
            else:
                a_fact = a
                choices = legal[legal != a_fact]
                a_cf = int(rng.choice(choices)) if choices.size else 0

            # --- factual branch (fork, do a_fact, then a fixed tail) --------
            fbr = world.fork()
            tail: List[int] = []
            f_obs, f_rew = [], []
            fr = EpisodeRunner.__new__(EpisodeRunner)   # lightweight view
            fr.cfg, fr.population = runner.cfg, runner.population
            fr.obs_builder, fr.action_space = ob, asp
            fr.focal_agent, fr.world = agent, fbr
            for l in range(horizon):
                act_i = a_fact if l == 0 else int(behav(fbr, agent, runner.population))
                tail.append(act_i)
                res = fr.step_with_index(act_i)
                f_obs.append(fr.observe())
                f_rew.append(float(res.rewards[agent]))

            # --- counterfactual branch (same tail, different first action) ---
            cbr = world.fork()
            cr = EpisodeRunner.__new__(EpisodeRunner)
            cr.cfg, cr.population = runner.cfg, runner.population
            cr.obs_builder, cr.action_space = ob, asp
            cr.focal_agent, cr.world = agent, cbr
            c_obs, c_rew = [], []
            cf_seq = [a_cf] + tail[1:]
            for l in range(horizon):
                res = cr.step_with_index(cf_seq[l])
                c_obs.append(cr.observe())
                c_rew.append(float(res.rewards[agent]))

            f_n = normalizer.obs(np.stack(f_obs))
            c_n = normalizer.obs(np.stack(c_obs))
            probes.append(CounterfactualProbe(
                obs_hist=normalizer.obs(_window(obs_buf, history)),
                act_hist=np.asarray((act_buf + [a_fact])[-history:]
                                    if len(act_buf) + 1 >= history else
                                    [0] * (history - len(act_buf) - 1)
                                    + act_buf + [a_fact], dtype=np.int64),
                node_hist=normalizer.node(np.stack(node_buf[-history:])
                                          if len(node_buf) >= history else
                                          np.stack([node_buf[0]] *
                                                   (history - len(node_buf)) + node_buf)),
                fact_actions=np.asarray(tail, dtype=np.int64),
                cf_actions=np.asarray(cf_seq, dtype=np.int64),
                true_effect=(c_n - f_n).astype(np.float32),
                fact_obs=f_n.astype(np.float32), cf_obs=c_n.astype(np.float32),
                fact_reward=normalizer.reward(np.asarray(f_rew)),
                cf_reward=normalizer.reward(np.asarray(c_rew)),
                t=t, scenario=cfg.name, kind=kind))

        act_buf.append(a)
        runner.step_with_index(a)
    return probes


def evaluate_counterfactual(model, probes: Sequence[CounterfactualProbe],
                            needs_graph: bool = False) -> Dict[str, Any]:
    """Compare imagined causal effects against simulator ground truth."""
    if not probes:
        return {"n_probes": 0}
    H = probes[0].obs_hist.shape[0]
    L = probes[0].fact_actions.shape[0]
    D = probes[0].obs_hist.shape[1]
    B = len(probes)

    def stack(attr: str) -> np.ndarray:
        return np.stack([getattr(p, attr) for p in probes], axis=0)

    batch_common = {
        "obs_hist": stack("obs_hist").astype(np.float32),
        "act_hist": stack("act_hist").astype(np.int64),
    }
    if needs_graph:
        batch_common["node_hist"] = stack("node_hist").astype(np.float32)

    def roll(actions: np.ndarray):
        b = dict(batch_common)
        b["fut_act"] = actions.astype(np.int64)
        return model.predict_rollout(b, L)

    r_f = roll(stack("fact_actions"))
    r_c = roll(stack("cf_actions"))
    pred_effect = r_c.obs - r_f.obs                       # (B, L, D)
    true_effect = stack("true_effect")
    zero = np.zeros_like(true_effect)

    pred_rw = r_c.reward - r_f.reward
    true_rw = stack("cf_reward") - stack("fact_reward")

    # Per-horizon breakdown
    per_h: Dict[str, Dict[str, float]] = {}
    for l in range(L):
        per_h[str(l + 1)] = {
            "effect_mse": mse(true_effect[:, l], pred_effect[:, l]),
            "skill_vs_zero_effect": skill_score(true_effect[:, l],
                                                pred_effect[:, l], zero[:, l]),
        }

    flat_t, flat_p = true_effect.reshape(-1), pred_effect.reshape(-1)
    corr = float(np.corrcoef(flat_t, flat_p)[0, 1]) if np.std(flat_p) > 1e-12 else 0.0
    sign = float(np.mean(np.sign(flat_t) == np.sign(flat_p)))
    # Sign agreement restricted to effects that are actually material.
    big = np.abs(flat_t) > np.percentile(np.abs(flat_t), 75)
    sign_big = float(np.mean(np.sign(flat_t[big]) == np.sign(flat_p[big]))) \
        if big.any() else float("nan")

    return {
        "n_probes": B,
        "kind": probes[0].kind,
        "effect_mse": mse(true_effect, pred_effect),
        "zero_effect_mse": mse(true_effect, zero),
        "skill_vs_zero_effect": skill_score(true_effect, pred_effect, zero),
        "effect_corr": corr,
        "sign_accuracy": sign,
        "sign_accuracy_material": sign_big,
        "reward_effect_mse": mse(true_rw, pred_rw),
        "reward_effect_skill": skill_score(true_rw, pred_rw, np.zeros_like(true_rw)),
        "true_effect_rms": float(np.sqrt(np.mean(true_effect ** 2))),
        "per_horizon": per_h,
    }
