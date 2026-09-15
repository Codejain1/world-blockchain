"""Classical baselines: persistence, autoregression, linear/logistic, GBDT.

These set the floor the whole study is measured against.  A world model that
cannot beat ridge regression on next-state prediction has not demonstrated
anything, however sophisticated its latent space is.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ...data.dataset import BatchSpec, Normalizer, TrajectorySet
from ...evaluation.compute import ComputeMeter
from ..base import PredictiveModel, StepPrediction

__all__ = ["PersistenceBaseline", "ARBaseline", "LinearBaseline", "GBDTBaseline"]


def _flatten(batch: Dict[str, np.ndarray], n_actions: int) -> np.ndarray:
    """Window features shared by all classical models: obs window + action one-hots."""
    o = batch["obs_hist"]                       # (B, H, D)
    a = batch["act_hist"]                       # (B, H)
    B, H, D = o.shape
    oh = np.zeros((B, H, n_actions), dtype=np.float32)
    np.put_along_axis(oh, a[:, :, None], 1.0, axis=2)
    return np.concatenate([o.reshape(B, H * D), oh.reshape(B, H * n_actions)],
                          axis=1).astype(np.float32)


def _gather(ts: TrajectorySet, spec: BatchSpec, normalizer: Normalizer,
            max_samples: int, seed: int = 0) -> Dict[str, np.ndarray]:
    idx = ts.index(spec.history, spec.horizon)
    rng = np.random.Generator(np.random.PCG64(seed))
    if len(idx) > max_samples:
        idx = idx[rng.choice(len(idx), max_samples, replace=False)]
    return ts.batch(idx, spec, normalizer, with_graph=False)


class _ClassicalBase(PredictiveModel):
    family = "baseline"

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.spec: Optional[BatchSpec] = None
        self.normalizer: Optional[Normalizer] = None
        self.meter = ComputeMeter(name=name)

    def _timed(self, fn, n: int):
        t0 = time.perf_counter()
        out = fn()
        self.meter.add_inference(time.perf_counter() - t0, n)
        return out


# --------------------------------------------------------------------------
class PersistenceBaseline(_ClassicalBase):
    """"Nothing changes."  Predicts zero delta and the training base rates.

    This is the reference every other number should be read against: a model
    that only matches persistence has learned nothing about dynamics.
    """

    def __init__(self) -> None:
        super().__init__("persistence")
        self.event_rate: Optional[np.ndarray] = None
        self.reward_mean: float = 0.0

    def fit(self, train, val, spec, normalizer, **kw):
        self.spec, self.normalizer = spec, normalizer
        ev = np.concatenate([t.events for t in train.trajs], axis=0)
        self.event_rate = np.clip(ev.mean(axis=0), 1e-4, 1 - 1e-4)
        rw = np.concatenate([t.rewards for t in train.trajs])
        self.reward_mean = float(normalizer.reward(rw).mean())
        return {"event_rate": self.event_rate.tolist(), "n_params": 0}

    def predict_step(self, batch):
        B = batch["obs_hist"].shape[0]
        D = batch["obs_hist"].shape[2]
        logit = np.log(self.event_rate / (1.0 - self.event_rate))
        return self._timed(lambda: StepPrediction(
            delta=np.zeros((B, D), np.float32),
            event_logit=np.tile(logit.astype(np.float32), (B, 1)),
            reward=np.full(B, self.reward_mean, np.float32)), B)

    def n_params(self) -> int:
        return int(self.event_rate.size + 1) if self.event_rate is not None else 0


# --------------------------------------------------------------------------
class ARBaseline(_ClassicalBase):
    """Per-feature autoregression on the observation window (a VAR-lite).

    Fitted in closed form by ridge least squares on the *same* window every other
    model sees, so it is a like-for-like linear time-series reference.
    """

    def __init__(self, ridge: float = 1.0, max_samples: int = 60_000) -> None:
        super().__init__("ar_timeseries")
        self.ridge = float(ridge)
        self.max_samples = int(max_samples)
        self.W: Optional[np.ndarray] = None
        self.event_rate: Optional[np.ndarray] = None
        self.reward_mean: float = 0.0

    def fit(self, train, val, spec, normalizer, **kw):
        self.spec, self.normalizer = spec, normalizer
        b = _gather(train, spec, normalizer, self.max_samples)
        X = b["obs_hist"].reshape(len(b["obs_hist"]), -1)
        X = np.concatenate([X, np.ones((len(X), 1), np.float32)], axis=1)
        Y = b["delta"]
        A = X.T @ X + self.ridge * np.eye(X.shape[1], dtype=np.float64)
        self.W = np.linalg.solve(A, X.T @ Y).astype(np.float32)
        self.event_rate = np.clip(b["event"].mean(axis=0), 1e-4, 1 - 1e-4)
        self.reward_mean = float(b["reward"].mean())
        return {"n_params": int(self.W.size), "n_train": int(len(X))}

    def predict_step(self, batch):
        o = batch["obs_hist"]
        B = o.shape[0]
        X = np.concatenate([o.reshape(B, -1), np.ones((B, 1), np.float32)], axis=1)
        logit = np.log(self.event_rate / (1.0 - self.event_rate)).astype(np.float32)
        return self._timed(lambda: StepPrediction(
            delta=(X @ self.W).astype(np.float32),
            event_logit=np.tile(logit, (B, 1)),
            reward=np.full(B, self.reward_mean, np.float32)), B)

    def n_params(self) -> int:
        return int(self.W.size) if self.W is not None else 0


# --------------------------------------------------------------------------
class LinearBaseline(_ClassicalBase):
    """Ridge regression for the state delta and reward, logistic for events."""

    def __init__(self, alpha: float = 10.0, max_samples: int = 60_000) -> None:
        super().__init__("linear_logistic")
        self.alpha = float(alpha)
        self.max_samples = int(max_samples)
        self.delta_model = None
        self.reward_model = None
        self.event_models: List[Any] = []
        self.event_const: List[float] = []

    def fit(self, train, val, spec, normalizer, **kw):
        from sklearn.linear_model import LogisticRegression, Ridge

        self.spec, self.normalizer = spec, normalizer
        b = _gather(train, spec, normalizer, self.max_samples)
        X = _flatten(b, spec.n_actions)
        t0 = time.perf_counter()
        self.delta_model = Ridge(alpha=self.alpha).fit(X, b["delta"])
        self.reward_model = Ridge(alpha=self.alpha).fit(X, b["reward"])
        self.event_models, self.event_const = [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for e in range(spec.n_events):
                y = b["event"][:, e]
                if len(np.unique(y)) < 2:
                    # A label that never fires in training: fall back to its base
                    # rate rather than pretending to have fitted a classifier.
                    self.event_models.append(None)
                    self.event_const.append(float(np.clip(y.mean(), 1e-4, 1 - 1e-4)))
                    continue
                m = LogisticRegression(max_iter=400, C=1.0 / self.alpha,
                                       solver="lbfgs")
                self.event_models.append(m.fit(X, y))
                self.event_const.append(0.0)
        self.meter.train_seconds = time.perf_counter() - t0
        self.meter.train_samples = int(len(X))
        return {"n_params": self.n_params(), "n_train": int(len(X)),
                "train_seconds": self.meter.train_seconds}

    def predict_step(self, batch):
        X = _flatten(batch, self.spec.n_actions)
        B = X.shape[0]

        def run():
            logits = np.zeros((B, self.spec.n_events), np.float32)
            for e, m in enumerate(self.event_models):
                if m is None:
                    p = self.event_const[e]
                    logits[:, e] = np.log(p / (1.0 - p))
                else:
                    logits[:, e] = m.decision_function(X)
            return StepPrediction(
                delta=self.delta_model.predict(X).astype(np.float32),
                event_logit=logits,
                reward=self.reward_model.predict(X).astype(np.float32).reshape(-1))
        return self._timed(run, B)

    def n_params(self) -> int:
        n = 0
        if self.delta_model is not None:
            n += int(self.delta_model.coef_.size + self.delta_model.intercept_.size)
        if self.reward_model is not None:
            n += int(np.size(self.reward_model.coef_) + np.size(self.reward_model.intercept_))
        for m in self.event_models:
            if m is not None:
                n += int(m.coef_.size + m.intercept_.size)
        return n


# --------------------------------------------------------------------------
class GBDTBaseline(_ClassicalBase):
    """Gradient-boosted trees (LightGBM, falling back to XGBoost).

    Trees need one model per output.  Predicting all 100+ observation dimensions
    that way is prohibitively slow, so the delta head is fitted on the
    ``top_k_targets`` highest-variance features and the remainder fall back to
    persistence.  This is a real limitation of the baseline and is stated in the
    results rather than hidden: GBDT is strongest on the *event* task, where it
    fits every label.
    """

    def __init__(self, n_estimators: int = 120, max_samples: int = 40_000,
                 top_k_targets: int = 24, num_leaves: int = 31) -> None:
        super().__init__("gbdt")
        self.n_estimators = int(n_estimators)
        self.max_samples = int(max_samples)
        self.top_k = int(top_k_targets)
        self.num_leaves = int(num_leaves)
        self.delta_models: Dict[int, Any] = {}
        self.event_models: List[Any] = []
        self.event_const: List[float] = []
        self.reward_model = None
        self.backend = "none"

    @staticmethod
    def _make(kind: str, n_estimators: int, num_leaves: int):
        try:
            import lightgbm as lgb
            cls = lgb.LGBMRegressor if kind == "reg" else lgb.LGBMClassifier
            return cls(n_estimators=n_estimators, num_leaves=num_leaves,
                       learning_rate=0.08, verbose=-1, n_jobs=4,
                       min_child_samples=40), "lightgbm"
        except ImportError:
            pass
        try:
            import xgboost as xgb
            cls = xgb.XGBRegressor if kind == "reg" else xgb.XGBClassifier
            return cls(n_estimators=n_estimators, max_depth=6, learning_rate=0.08,
                       n_jobs=4, verbosity=0), "xgboost"
        except ImportError:
            return None, "none"

    def fit(self, train, val, spec, normalizer, **kw):
        self.spec, self.normalizer = spec, normalizer
        b = _gather(train, spec, normalizer, self.max_samples)
        X = _flatten(b, spec.n_actions)
        t0 = time.perf_counter()
        var = b["delta"].var(axis=0)
        self.targets = np.argsort(-var)[: self.top_k]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for j in self.targets:
                m, backend = self._make("reg", self.n_estimators, self.num_leaves)
                self.backend = backend
                if m is None:
                    break
                self.delta_models[int(j)] = m.fit(X, b["delta"][:, int(j)])
            m, _ = self._make("reg", self.n_estimators, self.num_leaves)
            self.reward_model = None if m is None else m.fit(X, b["reward"])
            for e in range(spec.n_events):
                y = b["event"][:, e]
                if len(np.unique(y)) < 2:
                    self.event_models.append(None)
                    self.event_const.append(float(np.clip(y.mean(), 1e-4, 1 - 1e-4)))
                    continue
                m, _ = self._make("clf", self.n_estimators, self.num_leaves)
                self.event_models.append(None if m is None else m.fit(X, y))
                self.event_const.append(0.0)
        self.meter.train_seconds = time.perf_counter() - t0
        self.meter.train_samples = int(len(X))
        return {"backend": self.backend, "n_targets": int(len(self.delta_models)),
                "train_seconds": self.meter.train_seconds, "n_params": self.n_params()}

    def predict_step(self, batch):
        X = _flatten(batch, self.spec.n_actions)
        B, D, E = X.shape[0], self.spec.obs_dim, self.spec.n_events

        def run():
            delta = np.zeros((B, D), np.float32)
            for j, m in self.delta_models.items():
                delta[:, j] = m.predict(X)
            logits = np.zeros((B, E), np.float32)
            for e, m in enumerate(self.event_models):
                if m is None:
                    p = self.event_const[e]
                    logits[:, e] = np.log(p / (1.0 - p))
                else:
                    pr = np.clip(m.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
                    logits[:, e] = np.log(pr / (1.0 - pr))
            rew = (np.zeros(B, np.float32) if self.reward_model is None
                   else self.reward_model.predict(X).astype(np.float32))
            return StepPrediction(delta=delta, event_logit=logits, reward=rew)
        return self._timed(run, B)

    def n_params(self) -> int:
        """Trees have no weights; we report total leaf count as the size proxy."""
        n = 0
        for m in list(self.delta_models.values()) + self.event_models + [self.reward_model]:
            if m is None:
                continue
            try:
                n += int(getattr(m, "n_estimators", 0)) * self.num_leaves
            except Exception:
                pass
        return n
