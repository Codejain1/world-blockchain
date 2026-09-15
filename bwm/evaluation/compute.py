"""Computational-cost accounting.

The hypothesis under test is about *capability*, not about who was allowed to
spend more.  Every system therefore carries a meter recording parameters, wall
clock, environment steps, imagined rollout steps and LLM tokens.  Results tables
report these next to accuracy so a win bought with 100x compute is visible as
such.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

__all__ = ["ComputeMeter", "Timer"]


@dataclass
class ComputeMeter:
    """Mutable counters attached to one system for one evaluation."""

    name: str = ""
    n_params: int = 0
    train_seconds: float = 0.0
    train_samples: int = 0
    inference_seconds: float = 0.0
    n_decisions: int = 0
    env_steps: int = 0             # real simulator transitions consumed
    imagined_steps: int = 0        # latent rollout transitions
    oracle_sim_steps: int = 0      # privileged true-simulator rollouts
    llm_calls: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    peak_rss_mb: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- helpers ---------------------------------------------------------
    def add_inference(self, seconds: float, n: int = 1) -> None:
        self.inference_seconds += float(seconds)
        self.n_decisions += int(n)

    @property
    def latency_ms(self) -> float:
        return 1000.0 * self.inference_seconds / max(self.n_decisions, 1)

    def sample_memory(self) -> None:
        try:
            import resource
            mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            self.peak_rss_mb = max(self.peak_rss_mb, float(mb))
        except Exception:      # pragma: no cover - platform dependent
            pass

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["latency_ms"] = self.latency_ms
        return d

    def merge(self, other: "ComputeMeter") -> "ComputeMeter":
        out = ComputeMeter(name=self.name or other.name,
                           n_params=max(self.n_params, other.n_params))
        for f in ("train_seconds", "train_samples", "inference_seconds", "n_decisions",
                  "env_steps", "imagined_steps", "oracle_sim_steps", "llm_calls",
                  "llm_prompt_tokens", "llm_completion_tokens"):
            setattr(out, f, getattr(self, f) + getattr(other, f))
        out.peak_rss_mb = max(self.peak_rss_mb, other.peak_rss_mb)
        out.extra = {**self.extra, **other.extra}
        return out


class Timer:
    """``with Timer() as t: ...`` then read ``t.seconds``."""

    def __init__(self) -> None:
        self.seconds = 0.0
        self._t0 = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.seconds = time.perf_counter() - self._t0
