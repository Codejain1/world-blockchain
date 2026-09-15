"""Deterministic, counter-based random number generation.

The whole lab depends on one property: *the exogenous noise of the world is a
pure function of (base_seed, block, stream_name)*.  It does **not** depend on
how many random numbers were consumed earlier in the episode.

Why this matters scientifically
-------------------------------
Counterfactual evaluation ("what would have happened if agent X had done A
instead of B?") is only well posed if both branches of the world experience the
*same* exogenous randomness.  With a single mutable ``Generator`` threaded
through the simulator, taking a different action consumes a different number of
random draws, so the two branches would silently diverge for reasons that have
nothing to do with the action.  Any measured "causal effect" would be dominated
by that artefact.

Counter-based seeding removes the artefact: block ``t`` always draws price
shocks from ``stream("price", t)``, whatever happened before.  Forking the world
at ``t`` and running two different action sequences therefore isolates the
causal effect of the action.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np

__all__ = ["stream_seed", "make_rng", "SeedSequenceFactory", "seed_everything"]


def _hash_to_uint64(parts: Iterable[object]) -> int:
    """Stable 64-bit hash of a tuple of objects.

    ``hash()`` is not usable here: Python randomises string hashing per process
    (PYTHONHASHSEED), which would silently break reproducibility across runs.
    BLAKE2b is stable across processes, platforms and Python versions.
    """
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x00")
    return int.from_bytes(h.digest(), "big", signed=False)


def stream_seed(base_seed: int, stream: str, *counters: object) -> int:
    """Derive a reproducible seed for ``(base_seed, stream, *counters)``."""
    return _hash_to_uint64((int(base_seed), str(stream), *counters))


def make_rng(base_seed: int, stream: str, *counters: object) -> np.random.Generator:
    """Return a fresh ``Generator`` for a named, counter-indexed stream."""
    return np.random.Generator(np.random.PCG64(stream_seed(base_seed, stream, *counters)))


class SeedSequenceFactory:
    """Convenience wrapper binding a base seed so call sites stay short."""

    __slots__ = ("base_seed",)

    def __init__(self, base_seed: int) -> None:
        self.base_seed = int(base_seed)

    def rng(self, stream: str, *counters: object) -> np.random.Generator:
        return make_rng(self.base_seed, stream, *counters)

    def seed(self, stream: str, *counters: object) -> int:
        return stream_seed(self.base_seed, stream, *counters)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SeedSequenceFactory(base_seed={self.base_seed})"


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch.

    Used for *model training* reproducibility.  The environment does not rely on
    global state -- it uses the counter-based streams above.
    """
    import random

    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # torch is optional for the pure-simulator paths
        pass
