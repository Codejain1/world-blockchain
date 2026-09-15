"""Lightweight, dependency-free configuration system.

Design decision (documented per the project's "simplest defensible option"
rule): we use plain nested ``dict``s loaded from YAML rather than a heavyweight
config framework.  Configs support

* ``_base_``: single- or multi-inheritance from other YAML files,
* dotted-path overrides from the CLI (``--set env.n_agents=200``),
* recursive merge, and
* a content hash so every artefact can record *exactly* which config produced it.

Nothing about models, environments, seeds or datasets is hard-coded in source.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml

__all__ = ["Config", "load_config", "merge", "config_hash"]


def merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (returns a new dict)."""
    out: Dict[str, Any] = copy.deepcopy(dict(base))
    for k, v in override.items():
        if k in out and isinstance(out[k], Mapping) and isinstance(v, Mapping):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _coerce(text: str) -> Any:
    """Parse a CLI override value using YAML scalar rules."""
    try:
        return yaml.safe_load(text)
    except Exception:
        return text


class Config(dict):
    """A ``dict`` with dotted-path access helpers.

    ``Config`` deliberately remains a plain dict subclass so that it serialises
    to YAML/JSON without custom encoders.
    """

    def get_path(self, path: str, default: Any = None) -> Any:
        node: Any = self
        for part in path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node: Any = self
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = value

    def sub(self, path: str) -> "Config":
        value = self.get_path(path, {})
        return Config(copy.deepcopy(value) if isinstance(value, Mapping) else {})

    @property
    def hash(self) -> str:
        return config_hash(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(_plain(self), sort_keys=True)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as fh:
            fh.write(self.to_yaml())


def _plain(obj: Any) -> Any:
    """Convert nested Config/dict structures to plain builtins."""
    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def config_hash(cfg: Mapping[str, Any]) -> str:
    """Stable 12-hex-char content hash of a config."""
    blob = json.dumps(_plain(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=6).hexdigest()


def load_config(
    path: str,
    overrides: Optional[Iterable[str]] = None,
    _seen: Optional[List[str]] = None,
) -> Config:
    """Load a YAML config, resolving ``_base_`` inheritance and CLI overrides.

    ``overrides`` entries look like ``"trainer.lr=3e-4"``.
    """
    path = os.path.abspath(path)
    _seen = list(_seen or [])
    if path in _seen:
        raise ValueError(f"Circular _base_ inheritance detected at {path}")
    _seen.append(path)

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config {path} must contain a mapping at top level")

    bases = raw.pop("_base_", None)
    merged: Dict[str, Any] = {}
    if bases:
        if isinstance(bases, str):
            bases = [bases]
        for b in bases:
            b_path = b if os.path.isabs(b) else os.path.join(os.path.dirname(path), b)
            merged = merge(merged, load_config(b_path, None, _seen))
    merged = merge(merged, raw)

    cfg = Config(merged)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, _, value = item.partition("=")
        cfg.set_path(key.strip(), _coerce(value.strip()))
    return cfg
