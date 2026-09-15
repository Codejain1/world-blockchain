"""System 1: the LLM-only agent (and its offline rule-based stand-in).

The LLM receives exactly the observation window every other system receives --
rendered as a named table -- plus the identical action menu, and returns one
action index.  It is given **no** access to any world-model latent state, to the
hidden regime, or to the simulator.

Offline operation
-----------------
When no API credentials are present the transport degrades to
:class:`~bwm.models.llm.client.OfflineClient` and this policy falls back to
:class:`RuleBasedReasoner`: an explicit, interpretable trading heuristic over the
*same* observation vector.  That fallback is a legitimate "symbolic reasoning
without a world model" baseline, but it is **not an LLM result**.  Every results
row records ``backend`` so the two can never be confused.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...evaluation.compute import ComputeMeter
from ..base import DecisionContext, ModelInfo, Policy
from .client import LLMClient, LLMConfig, make_client

__all__ = ["StatePresenter", "RuleBasedReasoner", "LLMPolicy", "SYSTEM_PROMPT"]

SYSTEM_PROMPT = """You are an autonomous trading agent operating inside a simulated \
blockchain economy with AMM pools, a lending market with liquidations, staking, \
on-chain governance and a gas fee market.

Your objective is to maximise the risk-adjusted growth of your own net worth over \
the remainder of the episode. Liquidation of your borrow position is very costly. \
Gas is charged on every action, so doing nothing is often correct.

You will be given:
  * a table of the last few observation vectors (named features, most recent last)
  * the menu of actions available to you this step

Respond with ONLY a JSON object:
{"action_index": <integer from the menu>, "reasoning": "<one or two sentences>"}
"""


class StatePresenter:
    """Renders the shared observation window as text for a language model."""

    def __init__(self, feature_names: Sequence[str], action_names: Sequence[str],
                 max_history: int = 8, precision: int = 4,
                 key_features: Optional[Sequence[str]] = None) -> None:
        self.feature_names = list(feature_names)
        self.action_names = list(action_names)
        self.max_history = int(max_history)
        self.precision = int(precision)
        self.key_features = list(key_features) if key_features else None

    def render(self, obs_hist: np.ndarray, act_hist: np.ndarray,
               feasible: np.ndarray, t: int, recent_rewards: Optional[Sequence[float]] = None,
               notes: str = "") -> str:
        H = min(self.max_history, obs_hist.shape[0])
        win = obs_hist[-H:]
        acts = act_hist[-H:]
        p = self.precision
        lines: List[str] = [f"STEP: {t}", ""]
        lines.append("OBSERVATION WINDOW (one row per step, oldest first):")
        lines.append("step," + ",".join(self.feature_names))
        for i in range(H):
            vals = ",".join(f"{v:.{p}g}" for v in win[i])
            lines.append(f"{t - H + 1 + i},{vals}")
        lines.append("")
        lines.append("YOUR RECENT ACTIONS (oldest first): " + ", ".join(
            f"{self.action_names[int(a)]}" for a in acts))
        if recent_rewards is not None and len(recent_rewards):
            lines.append("YOUR RECENT STEP RETURNS (USD): " + ", ".join(
                f"{r:.1f}" for r in list(recent_rewards)[-H:]))
        lines.append("")
        avail = [i for i in range(len(self.action_names)) if feasible[i]]
        lines.append(f"AVAILABLE ACTIONS ({len(avail)} of {len(self.action_names)}):")
        lines.append("; ".join(f"{i}={self.action_names[i]}" for i in avail))
        if notes:
            lines.append("")
            lines.append(notes)
        lines.append("")
        lines.append('Reply with JSON only: {"action_index": int, "reasoning": str}')
        return "\n".join(lines)


def parse_action(text: str, n_actions: int,
                 action_names: Optional[Sequence[str]] = None) -> Tuple[int, str, bool]:
    """Extract ``(action_index, reasoning, ok)`` from a model reply."""
    if not text:
        return 0, "", False
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            if "action_index" in d:
                i = int(d["action_index"])
                if 0 <= i < n_actions:
                    return i, str(d.get("reasoning", "")), True
            if action_names and "action_name" in d:
                name = str(d["action_name"])
                if name in action_names:
                    return action_names.index(name), str(d.get("reasoning", "")), True
        except Exception:
            pass
    m = re.search(r"action[_ ]?index\D{0,6}(\d+)", text, re.I)
    if m:
        i = int(m.group(1))
        if 0 <= i < n_actions:
            return i, text[:200], True
    m = re.search(r"\b(\d{1,3})\b", text)
    if m:
        i = int(m.group(1))
        if 0 <= i < n_actions:
            return i, text[:200], True
    return 0, text[:200], False


# --------------------------------------------------------------------------
class RuleBasedReasoner:
    """Interpretable heuristic policy over the shared observation vector.

    Encodes the standard playbook a competent DeFi participant would follow:
    protect the health factor first, take a clear basis when one exists, de-risk
    into volatility, earn yield on idle cash, otherwise do nothing (gas is real).
    """

    def __init__(self, feature_names: Sequence[str], action_names: Sequence[str],
                 risk_aversion: float = 0.5) -> None:
        self.f = {n: i for i, n in enumerate(feature_names)}
        self.a = {n: i for i, n in enumerate(action_names)}
        self.action_names = list(action_names)
        self.risk_aversion = float(risk_aversion)
        self.tokens = sorted({n.split("_", 1)[1] for n in feature_names
                              if n.startswith("basis_")})

    def _get(self, obs: np.ndarray, name: str, default: float = 0.0) -> float:
        i = self.f.get(name)
        return float(obs[i]) if i is not None else float(default)

    def _find(self, *patterns: str) -> Optional[int]:
        for p in patterns:
            if p in self.a:
                return self.a[p]
        for p in patterns:
            for n, i in self.a.items():
                if n.startswith(p):
                    return i
        return None

    def act(self, obs: np.ndarray, feasible: np.ndarray) -> Tuple[int, str]:
        def ok(i: Optional[int]) -> bool:
            return i is not None and bool(feasible[i])

        inv_health = self._get(obs, "inv_health")
        # inv_health = 1/(1+hf): 0.5 -> hf = 1, 0.45 -> hf ~= 1.22
        if inv_health > 0.44:
            for tok in ["USD", "ETHX", "ALT"]:
                i = self._find(f"REPAY[{tok}]@0.5")
                if ok(i):
                    return i, f"health factor is critical ({inv_health:.2f}); repaying debt"
            i = self._find("SUPPLY[USD]@0.5")
            if ok(i):
                return i, "health factor is critical; adding collateral"

        rvol = max((self._get(obs, f"rvol_{t}") for t in self.tokens), default=0.0)
        if rvol > 0.02 * (1.0 + self.risk_aversion) and self._get(obs, "lp_frac") > 0.05:
            i = self._find("REM_LIQ")
            if ok(i):
                return i, f"realised volatility {rvol:.3f} is high; pulling liquidity"

        # Basis trade: the AMM price has drifted from the protocol oracle price.
        best_tok, best_basis = None, 0.0
        for tok in self.tokens:
            b = self._get(obs, f"basis_{tok}")
            if abs(b) > abs(best_basis):
                best_tok, best_basis = tok, b
        if best_tok is not None and abs(best_basis) > 0.01:
            if best_basis > 0:
                i = self._find(f"SWAP[{best_tok}->USD]@0.25")
                why = f"{best_tok} trades {best_basis:+.2%} above oracle; selling"
            else:
                i = self._find(f"SWAP[USD->{best_tok}]@0.25")
                why = f"{best_tok} trades {best_basis:+.2%} below oracle; buying"
            if ok(i):
                return i, why

        if self._get(obs, "cash_frac") > 0.45:
            for tok in ["USD", "ETHX"]:
                i = self._find(f"SUPPLY[{tok}]@0.25")
                if ok(i):
                    return i, "idle cash; earning lending yield"

        mom = float(np.mean([self._get(obs, f"ret1_{t}") for t in self.tokens])) \
            if self.tokens else 0.0
        if mom > 0.004 and self.risk_aversion < 0.7 and best_tok:
            i = self._find(f"SWAP[USD->{best_tok}]@0.1")
            if ok(i):
                return i, f"positive momentum ({mom:+.3f}); adding risk"
        if mom < -0.006 and best_tok:
            i = self._find(f"SWAP[{best_tok}->USD]@0.25")
            if ok(i):
                return i, f"negative momentum ({mom:+.3f}); de-risking"
        return 0, "no edge worth paying gas for"


# --------------------------------------------------------------------------
class LLMPolicy(Policy):
    """System 1 -- reasoning over raw observations, no world model."""

    family = "llm"

    def __init__(self, feature_names: Sequence[str], action_names: Sequence[str],
                 llm_cfg: Optional[LLMConfig] = None, name: str = "llm",
                 max_history: int = 8, risk_aversion: float = 0.5,
                 client: Optional[LLMClient] = None) -> None:
        super().__init__(name)
        self.llm_cfg = llm_cfg or LLMConfig()
        self.client = client or make_client(self.llm_cfg, n_actions=len(action_names))
        self.presenter = StatePresenter(feature_names, action_names, max_history)
        self.fallback = RuleBasedReasoner(feature_names, action_names, risk_aversion)
        self.action_names = list(action_names)
        self.meter = ComputeMeter(name=name)
        self.parse_failures = 0
        self.recent_rewards: List[float] = []
        self.last_reasoning: str = ""

    @property
    def backend(self) -> str:
        return "rule_based_offline" if self.client.is_offline else self.llm_cfg.provider

    def reset(self, episode_seed: Optional[int] = None) -> None:
        self.recent_rewards = []

    def act(self, ctx: DecisionContext) -> int:
        t0 = time.perf_counter()
        obs = ctx.raw_obs if ctx.raw_obs is not None else ctx.obs_hist[-1]
        if self.client.is_offline:
            a, why = self.fallback.act(obs, ctx.feasible)
            self.last_reasoning = why
        else:
            raw_hist = ctx.info.get("raw_obs_hist", ctx.obs_hist)
            prompt = self.presenter.render(raw_hist, ctx.act_hist, ctx.feasible,
                                           ctx.t, self.recent_rewards)
            resp = self.client.complete(SYSTEM_PROMPT, prompt)
            a, why, ok = parse_action(resp.text, len(self.action_names),
                                      self.action_names)
            if not ok:
                self.parse_failures += 1
            if not ctx.feasible[a]:
                a = 0
            self.last_reasoning = why
            self.meter.llm_calls += 1
            self.meter.llm_prompt_tokens += resp.prompt_tokens
            self.meter.llm_completion_tokens += resp.completion_tokens
        self.meter.add_inference(time.perf_counter() - t0, 1)
        return int(a)

    def observe_outcome(self, ctx, action, reward, next_obs, events) -> None:
        self.recent_rewards.append(float(reward))
        if len(self.recent_rewards) > 64:
            self.recent_rewards = self.recent_rewards[-64:]

    def info(self) -> ModelInfo:
        return ModelInfo(name=self.name, family="llm", n_params=0,
                         notes=("offline rule-based surrogate -- NOT an LLM result"
                                if self.client.is_offline else
                                f"{self.llm_cfg.provider}:{self.llm_cfg.model}"),
                         extra={"backend": self.backend,
                                "parse_failures": self.parse_failures,
                                **self.client.stats()})
