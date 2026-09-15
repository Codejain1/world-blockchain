"""Factories that build every system in the comparison from configuration.

One place defines what "the MLP baseline" or "ablation D" means, so the training
script, the benchmark script and the tests cannot drift apart.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from ..data.dataset import BatchSpec, Normalizer
from ..environment.action_space import DiscreteActionSpace
from ..environment.observation import ObservationBuilder
from ..models.baselines.classical import (ARBaseline, GBDTBaseline, LinearBaseline,
                                          PersistenceBaseline)
from ..models.baselines.neural import GNNBaseline, MLPBaseline, TransformerBaseline
from ..models.baselines.policies import (BuyAndHoldPolicy, GreedyMemoryPolicy,
                                         NoopPolicy, RandomPolicy)
from ..models.llm.agent import LLMPolicy
from ..models.llm.client import LLMConfig
from ..models.unified.unified import UnifiedPolicy
from ..models.world_model.jepa import JEPAWorldModel
from ..models.world_model.latent import (GraphDynamicsWorldModel, ObsSpaceDynamicsModel,
                                         TransformerDynamicsWorldModel)
from ..models.world_model.policy import WorldModelPolicy
from ..models.world_model.rssm import RSSMWorldModel
from ..planning.analytic import AnalyticPlannerPolicy
from ..planning.oracle import OracleConfig, OraclePlannerPolicy
from ..planning.planner import PlannerConfig
from ..training.trainer import TrainConfig

__all__ = ["PREDICTIVE_MODELS", "GRAPH_MODELS", "build_predictive_model",
           "build_predictive_models", "ABLATIONS", "build_policies"]

#: Every predictive system, by short name.
PREDICTIVE_MODELS: Dict[str, str] = {
    "persistence": "baseline",
    "ar": "baseline",
    "linear": "baseline",
    "gbdt": "baseline",
    "mlp": "baseline",
    "transformer": "baseline",
    "gnn": "baseline",
    "obsspace": "baseline",        # multi-step objective, no latent
    "wm_rssm": "world_model",
    "wm_jepa": "world_model",
    "wm_transformer": "world_model",
    "wm_graph": "world_model",
    "wm_rssm_1step": "world_model",  # ablation: latent model, one-step objective
}

#: Models that consume the graph view and therefore need ``node_hist`` batches.
GRAPH_MODELS = {"gnn", "wm_graph"}


def build_predictive_model(name: str, spec: BatchSpec, obs_builder: ObservationBuilder,
                           train_cfg: Optional[TrainConfig] = None,
                           overrides: Optional[Dict[str, Any]] = None):
    """Construct one predictive system by name."""
    import copy

    cfg = copy.deepcopy(train_cfg or TrainConfig())
    kw = dict(overrides or {})
    if name == "persistence":
        return PersistenceBaseline()
    if name == "ar":
        return ARBaseline(**kw)
    if name == "linear":
        return LinearBaseline(**kw)
    if name == "gbdt":
        return GBDTBaseline(**kw)
    if name == "mlp":
        return MLPBaseline(spec, cfg, **kw)
    if name == "transformer":
        return TransformerBaseline(spec, cfg, **kw)
    if name == "gnn":
        return GNNBaseline(spec, obs_builder, cfg, **kw)
    if name == "obsspace":
        return ObsSpaceDynamicsModel(spec, cfg, **kw)
    if name == "wm_rssm":
        return RSSMWorldModel(spec, cfg, **kw)
    if name == "wm_rssm_1step":
        return RSSMWorldModel(spec, cfg, train_horizon=1, name="wm_rssm_1step", **kw)
    if name == "wm_jepa":
        return JEPAWorldModel(spec, cfg, **kw)
    if name == "wm_transformer":
        return TransformerDynamicsWorldModel(spec, cfg, **kw)
    if name == "wm_graph":
        return GraphDynamicsWorldModel(spec, obs_builder, cfg, **kw)
    raise KeyError(f"Unknown predictive model {name!r}; "
                   f"known: {sorted(PREDICTIVE_MODELS)}")


def build_predictive_models(names: Sequence[str], spec: BatchSpec,
                            obs_builder: ObservationBuilder,
                            train_cfg: Optional[TrainConfig] = None,
                            overrides: Optional[Dict[str, Dict[str, Any]]] = None
                            ) -> Dict[str, Any]:
    overrides = overrides or {}
    return {n: build_predictive_model(n, spec, obs_builder, train_cfg,
                                      overrides.get(n)) for n in names}


# --------------------------------------------------------------------------
#: Ablation letters from the project brief, mapped to the policy that realises them.
ABLATIONS: Dict[str, str] = {
    "A": "llm",                     # LLM / reasoner only
    "B": "wm_plan",                 # world model + planning, no memory, no reasoner
    "C": "unified",                 # LLM + world model + memory + gate
    "D": "wm_greedy",               # world model, no planning (one-step greedy)
    "E": "oracle_plan",             # planning with the true (non-learned) simulator
    "F": "planner_no_model",        # reasoner + search, no learned dynamics
    "G": "wm_plan_memory",          # world model + memory
    "H": "wm_plan",                 # world model without memory (== B)
    "I": "unified_no_memory",
    "J": "unified_no_rollout",      # unified without counterfactual rollouts
}


def build_policies(names: Sequence[str], *, world_models: Dict[str, Any],
                   action_space: DiscreteActionSpace, obs_builder: ObservationBuilder,
                   normalizer: Normalizer, planner_cfg: Optional[PlannerConfig] = None,
                   llm_cfg: Optional[LLMConfig] = None,
                   primary_wm: str = "wm_rssm", seed: int = 0) -> Dict[str, Any]:
    """Build control policies, including every ablation variant.

    ``world_models`` maps name -> fitted :class:`LatentWorldModel`.
    """
    fnames = obs_builder.feature_names
    anames = action_space.names
    pcfg = planner_cfg or PlannerConfig()
    lcfg = llm_cfg or LLMConfig()
    out: Dict[str, Any] = {}

    def reasoner(tag: str = "llm") -> LLMPolicy:
        return LLMPolicy(fnames, anames, lcfg, name=tag)

    def wm(name: Optional[str] = None):
        key = name or primary_wm
        if key not in world_models:
            raise KeyError(f"World model {key!r} not fitted; have {sorted(world_models)}")
        return world_models[key]

    for n in names:
        if n == "noop":
            out[n] = NoopPolicy()
        elif n == "random":
            out[n] = RandomPolicy(seed)
        elif n == "buy_and_hold":
            out[n] = BuyAndHoldPolicy(anames)
        elif n == "memory_only":
            out[n] = GreedyMemoryPolicy(action_space.n, seed=seed)
        elif n == "llm":
            out[n] = reasoner("llm")
        elif n == "planner_no_model":
            out[n] = AnalyticPlannerPolicy(fnames, anames, reasoner("llm_inner"), seed=seed)
        elif n == "oracle_plan":
            out[n] = OraclePlannerPolicy(action_space, OracleConfig(seed=seed))
        elif n.startswith("wm_plan") or n == "wm_greedy":
            base = primary_wm
            if "@" in n:
                n, base = n.split("@", 1)
            out[n if "@" not in n else n] = WorldModelPolicy(
                wm(base), action_space, normalizer, pcfg, name=n,
                use_planning=(n != "wm_greedy"),
                use_memory=n.endswith("memory"),
                needs_graph=base in GRAPH_MODELS)
        elif n.startswith("unified"):
            out[n] = UnifiedPolicy(
                wm(), action_space, normalizer, reasoner(f"{n}_reasoner"), fnames,
                planner_cfg=pcfg, name=n,
                use_memory="no_memory" not in n,
                use_planning="no_rollout" not in n,
                use_reasoner=True,
                learn_gate="no_gate" not in n,
                needs_graph=primary_wm in GRAPH_MODELS)
        else:
            raise KeyError(f"Unknown policy {n!r}")
    return out
