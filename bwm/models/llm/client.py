"""Provider-agnostic LLM client with on-disk caching and token accounting.

Supported backends
------------------
``anthropic``  Claude models via the official SDK (e.g. Claude Opus class).
``openai``     GPT models via the official SDK (e.g. GPT-5 class).
``mock``       Deterministic canned responses -- used by the unit tests so the
               prompt construction and parsing paths are covered without a key.
``offline``    No transport at all.  Signals to :class:`LLMPolicy` that it must
               fall back to the rule-based reasoner, and makes that substitution
               visible in every results table.

Caching is keyed by the full request, so re-running an experiment costs nothing
and produces byte-identical decisions -- a reproducibility requirement that an
API-backed system otherwise fails.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["LLMResponse", "LLMConfig", "LLMClient", "make_client",
           "AnthropicClient", "OpenAIClient", "MockClient", "OfflineClient"]


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False
    error: Optional[str] = None


@dataclass
class LLMConfig:
    provider: str = "offline"     # anthropic | openai | mock | offline
    model: str = "claude-opus-5"
    max_tokens: int = 512
    temperature: float = 0.0      # deterministic decisions by default
    cache_dir: Optional[str] = "datasets/llm_cache"
    timeout_s: float = 60.0
    max_retries: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _DiskCache:
    def __init__(self, path: Optional[str]) -> None:
        self.path = path
        if path:
            os.makedirs(path, exist_ok=True)

    def _file(self, key: str) -> str:
        return os.path.join(self.path or ".", f"{key}.json")

    def get(self, key: str) -> Optional[LLMResponse]:
        if not self.path:
            return None
        f = self._file(key)
        if not os.path.exists(f):
            return None
        try:
            with open(f) as fh:
                d = json.load(fh)
            return LLMResponse(text=d["text"], prompt_tokens=d.get("prompt_tokens", 0),
                               completion_tokens=d.get("completion_tokens", 0),
                               latency_s=0.0, cached=True)
        except Exception:
            return None

    def put(self, key: str, resp: LLMResponse) -> None:
        if not self.path or resp.error:
            return
        try:
            with open(self._file(key), "w") as fh:
                json.dump({"text": resp.text, "prompt_tokens": resp.prompt_tokens,
                           "completion_tokens": resp.completion_tokens}, fh)
        except Exception:
            pass


class LLMClient:
    """Base class: handles caching, retries and accounting."""

    is_offline: bool = False

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.cache = _DiskCache(cfg.cache_dir)
        self.n_calls = 0
        self.n_cache_hits = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_latency = 0.0
        self.errors: List[str] = []

    def _key(self, system: str, user: str) -> str:
        blob = json.dumps({"p": self.cfg.provider, "m": self.cfg.model,
                           "t": self.cfg.temperature, "mt": self.cfg.max_tokens,
                           "s": system, "u": user}, sort_keys=True)
        return hashlib.blake2b(blob.encode(), digest_size=16).hexdigest()

    def complete(self, system: str, user: str) -> LLMResponse:
        key = self._key(system, user)
        hit = self.cache.get(key)
        if hit is not None:
            self.n_cache_hits += 1
            self.prompt_tokens += hit.prompt_tokens
            self.completion_tokens += hit.completion_tokens
            return hit
        last: Optional[LLMResponse] = None
        for attempt in range(max(self.cfg.max_retries, 0) + 1):
            t0 = time.perf_counter()
            resp = self._call(system, user)
            resp.latency_s = time.perf_counter() - t0
            self.n_calls += 1
            self.total_latency += resp.latency_s
            self.prompt_tokens += resp.prompt_tokens
            self.completion_tokens += resp.completion_tokens
            if resp.error is None:
                self.cache.put(key, resp)
                return resp
            self.errors.append(resp.error)
            last = resp
            time.sleep(min(2.0 ** attempt, 4.0))
        return last or LLMResponse(text="", error="unknown")

    def _call(self, system: str, user: str) -> LLMResponse:   # pragma: no cover
        raise NotImplementedError

    def stats(self) -> Dict[str, Any]:
        return {"provider": self.cfg.provider, "model": self.cfg.model,
                "calls": self.n_calls, "cache_hits": self.n_cache_hits,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_latency_s": self.total_latency,
                "n_errors": len(self.errors)}


class AnthropicClient(LLMClient):
    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__(cfg)
        import anthropic  # noqa: F401  (import error surfaces at construction)
        self._client = anthropic.Anthropic(timeout=cfg.timeout_s)

    def _call(self, system: str, user: str) -> LLMResponse:
        try:
            msg = self._client.messages.create(
                model=self.cfg.model, max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature, system=system,
                messages=[{"role": "user", "content": user}])
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            return LLMResponse(text=text,
                               prompt_tokens=int(msg.usage.input_tokens),
                               completion_tokens=int(msg.usage.output_tokens))
        except Exception as e:                       # pragma: no cover - network
            return LLMResponse(text="", error=f"{type(e).__name__}: {e}")


class OpenAIClient(LLMClient):
    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__(cfg)
        import openai  # noqa: F401
        self._client = openai.OpenAI(timeout=cfg.timeout_s)

    def _call(self, system: str, user: str) -> LLMResponse:
        try:
            r = self._client.chat.completions.create(
                model=self.cfg.model, max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            u = r.usage
            return LLMResponse(text=r.choices[0].message.content or "",
                               prompt_tokens=int(getattr(u, "prompt_tokens", 0)),
                               completion_tokens=int(getattr(u, "completion_tokens", 0)))
        except Exception as e:                       # pragma: no cover - network
            return LLMResponse(text="", error=f"{type(e).__name__}: {e}")


class MockClient(LLMClient):
    """Deterministic stand-in used by the tests.

    Picks an action by hashing the prompt, so parsing and prompt construction are
    exercised end-to-end without a network call.  It is *not* a scientific
    baseline and is never used in reported experiments.
    """

    def __init__(self, cfg: LLMConfig, n_actions: int = 94,
                 responses: Optional[List[str]] = None) -> None:
        super().__init__(cfg)
        self.n_actions = int(n_actions)
        self.responses = responses
        self._i = 0

    def _call(self, system: str, user: str) -> LLMResponse:
        if self.responses:
            text = self.responses[self._i % len(self.responses)]
            self._i += 1
        else:
            h = int(hashlib.blake2b(user.encode(), digest_size=8).hexdigest(), 16)
            text = json.dumps({"action_index": h % self.n_actions,
                               "reasoning": "deterministic mock selection"})
        return LLMResponse(text=text, prompt_tokens=len(user) // 4,
                           completion_tokens=len(text) // 4)


class OfflineClient(LLMClient):
    """No transport available; the policy must use its rule-based fallback."""

    is_offline = True

    def __init__(self, cfg: LLMConfig) -> None:
        super().__init__(cfg)
        self.cache = _DiskCache(None)

    def _call(self, system: str, user: str) -> LLMResponse:
        return LLMResponse(text="", error="offline")


def make_client(cfg: LLMConfig, n_actions: int = 94) -> LLMClient:
    """Build a client, degrading to ``offline`` when credentials are absent.

    The degradation is never silent: :meth:`LLMClient.stats` and the policy's
    ``ModelInfo`` both record which backend actually ran.
    """
    p = (cfg.provider or "offline").lower()
    if p == "mock":
        return MockClient(cfg, n_actions=n_actions)
    if p == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return OfflineClient(cfg)
        try:
            return AnthropicClient(cfg)
        except Exception:
            return OfflineClient(cfg)
    if p == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            return OfflineClient(cfg)
        try:
            return OpenAIClient(cfg)
        except Exception:
            return OfflineClient(cfg)
    return OfflineClient(cfg)
