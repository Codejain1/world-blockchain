# Reproducibility

## Environment

Developed and run on Python 3.11, CPU only (4 cores, no GPU).

```bash
pip install -e .            # editable install exposes `bwm` everywhere
pytest -q                   # 75 tests
```

Pinned versions used for the reported run are in `requirements.txt`; the run
actually executed with NumPy 2.4, SciPy 1.17, PyTorch 2.14 (CPU), scikit-learn
1.9, LightGBM 4.7, XGBoost 3.2, pandas 3.0, matplotlib 3.11.

## Full pipeline

```bash
python scripts/generate_data.py --out datasets/main          # ~10 min, 4 workers
python scripts/run_experiment.py --config configs/main.yaml  # training + benchmark
```

Intermediate stages can be run on their own:

```bash
python scripts/train_models.py --config configs/main.yaml
python scripts/run_experiment.py --config configs/main.yaml --reuse-checkpoints \
       --stages prediction,counterfactual,control
python scripts/audit.py --data datasets/main --results results/main
```

## What is fixed, and where

| quantity | where it comes from |
|---|---|
| world noise | `(seed, stream, block)` via `bwm.utils.seeding` |
| scripted agent decisions | `rng("agent", t, agent_id)` |
| agent parameters | `spec_seed` in `AgentPopulation` |
| data-collection policy | `BehaviorPolicy(seed=episode_seed)` |
| split membership | disjoint seed ranges, recorded in `manifest.json` |
| model init and batching | `TrainConfig.seed`, `seed_everything` |
| planner sampling | `PlannerConfig.seed`, re-seeded per episode |
| evaluation batches | `experiment.seed` |
| LLM decisions | `temperature = 0` plus an on-disk response cache |

Every artefact records the config hash that produced it (`Config.hash`), and the
full config is written next to the results.

## Determinism guarantees

Asserted by the test suite:

- the same seed reproduces a trajectory byte-for-byte, and a different seed does not;
- `snapshot()` / `restore()` round-trips exactly, and the restored world replays
  the same future;
- a fork with a **null** intervention is bit-identical to its parent;
- a fork with a real intervention diverges, while the exogenous process stays
  untouched;
- counterfactual ground-truth effects are reproducible across runs;
- latent rollouts are deterministic and action-sensitive;
- control episodes replay identically for a fixed seed.

One historical bug is worth noting: `WorldState.digest()` originally hashed
`repr()` of NumPy scalars, so a state and its own copy hashed differently
(`np.float64(1.0)` vs `1.0`). The determinism test caught it. Scalars are now
coerced before hashing.

## Cost

On 4 CPU cores:

| stage | wall clock |
|---|---|
| dataset generation (650 episodes × 256 steps) | ~10 min |
| training 13 predictive systems (2250 gradient steps each) | ~2 h |
| prediction over 8 splits | ~20 min |
| counterfactual probes | ~10 min |
| control (13 policies × 5 worlds × 4 episodes) | ~35 min |

Reduce with `--set data.max_train_episodes=… train.epochs=… control.n_episodes=…`,
or use `configs/medium.yaml`.

## Using a real LLM

```bash
export ANTHROPIC_API_KEY=...
python scripts/run_experiment.py --config configs/main.yaml \
    --set llm.provider=anthropic llm.model=claude-opus-5
```

Without a key the client degrades to the offline rule-based reasoner. The
degradation is never silent: `RESULTS.md` prints the backend actually used and
marks those rows as a symbolic-reasoning baseline rather than an LLM result.
Responses are cached on disk keyed by the full request, so a rerun is free and
produces byte-identical decisions.
