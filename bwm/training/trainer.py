"""Generic supervised trainer shared by every learned system.

Using one trainer for baselines, world models and the unified system means the
optimiser, schedule, early stopping and batch construction are *identical*
across the comparison.  Differences in results then come from the models, not
from one of them having a better-tuned training loop.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..data.dataset import BatchSpec, Normalizer, TrajectorySet
from ..evaluation.compute import ComputeMeter
from ..models.base import PredictiveModel, StepPrediction

__all__ = ["TrainConfig", "to_torch", "TorchPredictiveModel", "LossModule"]


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    patience: int = 6                 # early-stopping patience in epochs
    seed: int = 0
    device: str = "cpu"
    max_batches_per_epoch: int = 200  # caps epoch cost so every model gets a
                                      # comparable optimisation budget
    val_batches: int = 40
    with_graph: bool = False
    log_every: int = 5
    warmup_frac: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def to_torch(batch: Dict[str, np.ndarray], device: str = "cpu") -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if v.dtype == np.int64:
            out[k] = torch.as_tensor(v, dtype=torch.long, device=device)
        else:
            out[k] = torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float32,
                                     device=device)
    return out


class LossModule(nn.Module):
    """A module that can score a batch and emit predictions."""

    def loss(self, b: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class TorchPredictiveModel(PredictiveModel):
    """Glue between :class:`LossModule` and the evaluation harness."""

    family = "baseline"

    def __init__(self, name: str, module: LossModule, cfg: TrainConfig,
                 with_graph: bool = False) -> None:
        super().__init__(name)
        self.module = module
        self.cfg = cfg
        self.with_graph = bool(with_graph or cfg.with_graph)
        self.spec: Optional[BatchSpec] = None
        self.normalizer: Optional[Normalizer] = None
        self.meter = ComputeMeter(name=name)
        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.module.parameters()))

    def _batches(self, ts: TrajectorySet, spec: BatchSpec, rng: np.random.Generator,
                 n_batches: int, shuffle: bool = True) -> Iterable[Dict[str, torch.Tensor]]:
        idx = ts.index(spec.history, spec.horizon)
        if idx.size == 0:
            return
        order = rng.permutation(len(idx)) if shuffle else np.arange(len(idx))
        bs = self.cfg.batch_size
        n = min(n_batches, max(1, len(order) // bs))
        for i in range(n):
            sel = idx[order[i * bs: (i + 1) * bs]]
            if len(sel) == 0:
                continue
            yield to_torch(ts.batch(sel, spec, self.normalizer, self.with_graph),
                           self.cfg.device)

    def fit(self, train: TrajectorySet, val: Optional[TrajectorySet], spec: BatchSpec,
            normalizer: Normalizer, **kwargs) -> Dict[str, Any]:
        self.spec, self.normalizer = spec, normalizer
        torch.manual_seed(self.cfg.seed)
        rng = np.random.Generator(np.random.PCG64(self.cfg.seed))
        self.module.to(self.cfg.device)
        opt = torch.optim.AdamW(self.module.parameters(), lr=self.cfg.lr,
                                weight_decay=self.cfg.weight_decay)
        total = self.cfg.epochs * self.cfg.max_batches_per_epoch
        warm = max(int(self.cfg.warmup_frac * total), 1)

        def lr_at(step: int) -> float:
            if step < warm:
                return step / warm
            p = (step - warm) / max(total - warm, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
        best, best_state, bad = float("inf"), None, 0
        t0 = time.perf_counter()
        n_samples = 0

        for ep in range(self.cfg.epochs):
            self.module.train()
            tr_loss, nb = 0.0, 0
            for b in self._batches(train, spec, rng, self.cfg.max_batches_per_epoch):
                loss, _ = self.module.loss(b)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.module.parameters(), self.cfg.grad_clip)
                opt.step()
                sched.step()
                tr_loss += float(loss.detach())
                nb += 1
                n_samples += int(b["obs_hist"].shape[0])
            tr_loss /= max(nb, 1)

            va_loss = tr_loss
            if val is not None:
                self.module.eval()
                vl, vb = 0.0, 0
                vrng = np.random.Generator(np.random.PCG64(12345))
                with torch.no_grad():
                    for b in self._batches(val, spec, vrng, self.cfg.val_batches,
                                           shuffle=False):
                        loss, _ = self.module.loss(b)
                        vl += float(loss)
                        vb += 1
                va_loss = vl / max(vb, 1)

            self.history.append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss})
            if va_loss < best - 1e-5:
                best, bad = va_loss, 0
                best_state = copy.deepcopy(self.module.state_dict())
            else:
                bad += 1
                if bad >= self.cfg.patience:
                    break
        if best_state is not None:
            self.module.load_state_dict(best_state)
        self.module.eval()
        self.meter.train_seconds = time.perf_counter() - t0
        self.meter.train_samples = n_samples
        self.meter.n_params = self.n_params()
        self.meter.sample_memory()
        return {"history": self.history, "best_val": best,
                "train_seconds": self.meter.train_seconds,
                "n_params": self.meter.n_params, "config": self.cfg.to_dict()}

    # ------------------------------------------------------------------
    def predict_step(self, batch: Dict[str, np.ndarray]) -> StepPrediction:
        self.module.eval()
        b = to_torch(batch, self.cfg.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.module.predict(b)
        self.meter.add_inference(time.perf_counter() - t0,
                                 int(batch["obs_hist"].shape[0]))
        return StepPrediction(
            delta=out["delta"].cpu().numpy(),
            event_logit=out["event_logit"].cpu().numpy(),
            reward=out["reward"].cpu().numpy().reshape(-1))

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.module.state_dict(),
                    "cfg": self.cfg.to_dict(),
                    "history": self.history}, path)

    def load(self, path: str) -> None:
        blob = torch.load(path, map_location=self.cfg.device, weights_only=False)
        self.module.load_state_dict(blob["state_dict"])
        self.module.eval()
