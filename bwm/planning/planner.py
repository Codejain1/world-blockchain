"""Model-predictive planning in a learned latent space.

The loop required by the project brief, implemented literally:

1. encode the current observation history into ``z_t``
2. propose candidate action sequences
3. roll them forward **through the learned model only**
4. score the predicted outcomes
5. pick the best sequence
6. execute only its first action in the real environment
7. observe the real consequence
8. re-encode and re-plan

Two search strategies are provided (CEM and beam search) plus random shooting as
a control.  Every imagined transition is counted in the compute meter, because a
planner that wins by imagining 100x more than it is charged for has not won.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..environment.types import EventType
from ..models.world_model.base import LatentWorldModel

__all__ = ["PlannerConfig", "LatentPlanner", "RISK_EVENTS"]

#: Events a risk-aware objective should avoid.  Used only through the model's own
#: predicted probabilities -- the planner never sees the true future.
RISK_EVENTS: Tuple[int, ...] = (int(EventType.LIQUIDATION), int(EventType.CASCADE),
                                int(EventType.INSOLVENCY), int(EventType.LIQUIDITY_DROP))


@dataclass
class PlannerConfig:
    horizon: int = 8
    n_candidates: int = 64
    n_iters: int = 3              # CEM refinement rounds
    n_elites: int = 8
    gamma: float = 0.95
    risk_lambda: float = 0.0      # weight on predicted risk-event probability
    method: str = "cem"           # cem | beam | shooting
    beam_width: int = 8
    beam_branch: int = 12
    temperature: float = 1.0
    seed: int = 0
    keep_noop_prior: float = 0.10  # prior mass on "do nothing"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LatentPlanner:
    """MPC over a :class:`LatentWorldModel`.  Never touches the simulator."""

    def __init__(self, model: LatentWorldModel, n_actions: int,
                 cfg: Optional[PlannerConfig] = None) -> None:
        self.model = model
        self.n_actions = int(n_actions)
        self.cfg = cfg or PlannerConfig()
        self._rng = np.random.Generator(np.random.PCG64(self.cfg.seed))

    def reset(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.Generator(np.random.PCG64(
            self.cfg.seed if seed is None else int(seed)))

    # ------------------------------------------------------------------
    def _score(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Discounted predicted return, minus a risk penalty."""
        L = out["reward"].shape[1]
        g = torch.tensor([self.cfg.gamma ** i for i in range(L)],
                         device=out["reward"].device)
        ret = (out["reward"] * g).sum(dim=1)
        if self.cfg.risk_lambda > 0.0:
            p = torch.sigmoid(out["event_logit"][:, :, list(RISK_EVENTS)])
            ret = ret - self.cfg.risk_lambda * (p.mean(dim=-1) * g).sum(dim=1)
        return ret

    def _imagine(self, state: Dict[str, torch.Tensor], actions: np.ndarray
                 ) -> torch.Tensor:
        n = actions.shape[0]
        st = LatentWorldModel.expand_state(state, n)
        a = torch.as_tensor(actions, dtype=torch.long, device=self.model.cfg.device)
        out = self.model.imagine_sequences(st, a)
        return self._score(out)

    # ------------------------------------------------------------------
    def plan(self, obs_hist: np.ndarray, act_hist: np.ndarray,
             feasible: Optional[np.ndarray] = None,
             node_hist: Optional[np.ndarray] = None) -> Tuple[int, Dict[str, Any]]:
        """Return the first action of the best imagined sequence."""
        cfg = self.cfg
        state = self.model.encode_numpy(obs_hist[None], act_hist[None],
                                        None if node_hist is None else node_hist[None])
        mask = (np.ones(self.n_actions, dtype=bool) if feasible is None
                else np.asarray(feasible, dtype=bool).copy())
        if not mask.any():
            mask[0] = True
        if cfg.method == "beam":
            return self._plan_beam(state, mask)
        return self._plan_cem(state, mask)

    # -- CEM / random shooting -------------------------------------------
    def _plan_cem(self, state, mask: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        cfg = self.cfg
        L, N = cfg.horizon, cfg.n_candidates
        A = self.n_actions
        logits = np.zeros((L, A), dtype=np.float64)
        logits[:, ~mask] = -1e9
        logits[:, 0] += np.log(max(cfg.keep_noop_prior, 1e-6) * A)
        iters = 1 if cfg.method == "shooting" else cfg.n_iters
        best_a, best_score = 0, -np.inf
        scores_hist: List[float] = []

        for it in range(iters):
            probs = _softmax(logits / max(cfg.temperature, 1e-6))
            cand = np.stack([self._rng.choice(A, size=N, p=probs[l]) for l in range(L)],
                            axis=1)                                  # (N, L)
            cand[:, 0] = np.where(mask[cand[:, 0]], cand[:, 0], 0)
            scores = self._imagine(state, cand).cpu().numpy()
            scores_hist.append(float(scores.mean()))
            order = np.argsort(-scores)
            elite = cand[order[: max(cfg.n_elites, 1)]]
            if scores[order[0]] > best_score:
                best_score = float(scores[order[0]])
                best_a = int(cand[order[0], 0])
            # Refit the per-step categorical toward the elites (CEM update).
            counts = np.full((L, A), 1e-2)
            for l in range(L):
                np.add.at(counts[l], elite[:, l], 1.0)
            counts[:, ~mask] = 1e-9
            logits = np.log(counts / counts.sum(axis=1, keepdims=True))
        return best_a, {"score": best_score, "mean_scores": scores_hist,
                        "method": cfg.method}

    # -- beam search ------------------------------------------------------
    def _plan_beam(self, state, mask: np.ndarray) -> Tuple[int, Dict[str, Any]]:
        """Breadth-limited tree search over the same latent model."""
        cfg = self.cfg
        allowed = np.flatnonzero(mask)
        branch = min(cfg.beam_branch, len(allowed))
        # Depth 1: evaluate a sample of first actions with a short lookahead.
        firsts = self._rng.choice(allowed, size=branch, replace=False)
        beams = [[int(a)] for a in firsts]
        for depth in range(1, cfg.horizon):
            expanded: List[List[int]] = []
            for b in beams:
                for a in self._rng.choice(self.n_actions,
                                          size=min(3, self.n_actions), replace=False):
                    expanded.append(b + [int(a)])
            pad = [b + [0] * (cfg.horizon - len(b)) for b in expanded]
            sc = self._imagine(state, np.asarray(pad, dtype=np.int64)).cpu().numpy()
            keep = np.argsort(-sc)[: cfg.beam_width]
            beams = [expanded[i] for i in keep]
            if depth >= cfg.horizon - 1:
                best = beams[0]
                return int(best[0]), {"score": float(sc[keep[0]]), "method": "beam"}
        pad = [b + [0] * (cfg.horizon - len(b)) for b in beams]
        sc = self._imagine(state, np.asarray(pad, dtype=np.int64)).cpu().numpy()
        i = int(np.argmax(sc))
        return int(beams[i][0]), {"score": float(sc[i]), "method": "beam"}


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(np.clip(x, -700, 700))
    s = e.sum(axis=-1, keepdims=True)
    out = e / np.maximum(s, 1e-300)
    # Guard against an all-masked row.
    bad = ~np.isfinite(out).all(axis=-1)
    if np.any(bad):
        out[bad] = 0.0
        out[bad, 0] = 1.0
    return out
