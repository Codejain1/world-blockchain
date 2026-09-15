# Architecture

## Layers

```
                        configs/*.yaml   (everything is configuration)
                                 |
  bwm/environment  ── the world ─┼─ state.py      S_t as plain NumPy arrays, copyable
                                 │   protocols.py AMM / lending / staking math
                                 │   world.py     the transition, mempool, shocks, events
                                 │   observation.py  the single information channel
                                 │   action_space.py 94 discrete actions, shared by all
                                 │   scenarios.py train distribution + 8 held-out worlds
                                 │   runner.py    episodes -> Trajectory
  bwm/agents       ── the crowd ─┤   archetypes.py 11 roles; population.py assembly
                                 │
  bwm/data         ── the corpus ┤   generate.py  seed-disjoint splits + manifest
                                 │   dataset.py   windowed batches, train-only normaliser
                                 │   audit.py     contamination checks
                                 │
  bwm/models       ── the systems┤   baselines/  classical, neural, control policies
                                 │   llm/        System 1
                                 │   world_model/ System 2 (4 architectures)
                                 │   unified/    System 3
  bwm/planning     ── the search ┤   planner.py (CEM/beam), oracle.py, analytic.py
  bwm/memory       ── recall ────┤   episodic.py
  bwm/training     ── one trainer┤   trainer.py  shared by every learned system
  bwm/evaluation   ── the science┤   metrics / prediction / counterfactual / control
                                 │   benchmark.py composite index and tables
  bwm/experiments  ── the glue ──┘   registry.py, train.py, evaluate.py
  bwm/visualization                  plots.py, report.py
```

## The transition

`BlockchainWorld.step(external_actions)` produces one block:

1. **Exogenous update** — hidden regime transition, regime-conditioned price
   process with jumps, then shock sampling (liquidity flight, whale dump, depeg,
   exploit, oracle glitch, gas spike).
2. **Protocol accrual** — kinked-curve interest indices, staking index, oracle
   refresh (honest fundamental, or a manipulable AMM TWAP in the novel-mechanism
   world).
3. **Action collection** — scripted agents act; systems under test supply actions
   for their focal account.
4. **Mempool ordering** — sorted by priority fee, deterministic tie-break, cut at
   the block gas limit.
5. **Execution** — each action charged gas and applied, with a receipt recording
   success or the reason for failure.
6. **Governance** — proposals created, voted, executed; execution mutates live
   protocol parameters.
7. **Fee market** — EIP-1559 base fee update plus a trailing baseline.
8. **Rewards and events** — reward is the change in mark-to-market net worth;
   ten binary events are labelled from the transition itself.

## Determinism

Three properties, all covered by tests:

1. **Counter-based noise.** Every exogenous stream is seeded by
   `(base_seed, stream_name, block)`, so block `t`'s noise is independent of how
   many draws happened earlier.
2. **Pure scripted policies.** Agent `i` at block `t` draws from
   `rng("agent", t, i)` only.
3. **Value-typed state.** `WorldState.copy()` yields an independent world.

Together these give `world.fork()`: two branches that share exogenous noise and
differ only in the action taken. Without property 1, taking a different action
would consume a different number of random draws and the branches would diverge
for reasons unrelated to the action — every "causal effect" would be RNG drift.

## The observation channel

`ObservationBuilder` is the only path from world state to any system. It produces
112 features: per-token prices and basis, per-pool depth/fee/imbalance,
per-market utilisation and rates, gas, staking, governance, aggregate solvency,
per-cluster population aggregates, and the focal account's own balance sheet.

Withheld from everyone: the hidden regime and the exogenous fundamental price.

The same builder produces a heterogeneous **graph** view — tokens wired to the
pools and markets that reference them, agent clusters wired to the venues they
use, and a focal-agent node. The cluster aggregates appear in *both* views, so the
graph carries no extra information; it carries extra *structure*.

## World-model interface

```python
state = model.encode(batch)                  # history -> latent
state = model.imagine(state, action)         # z_{t+1} ~ p(z_{t+1} | z_t, a_t)
out   = model.readout(state)                 # obs, reward, event logits
state = model.advance(state, out)            # carry the residual anchor forward
```

`imagine` never touches the simulator. All four architectures implement exactly
this, so the planner, the unified system and the counterfactual harness are
architecture-agnostic.

Readouts are **residual** on the previous observation estimate. Flat baselines
predict `obs_{t+1} - obs_t` and so start at persistence; a world model decoding
absolute observations would start at the dataset mean instead. Residual decoding
removes that architecture-induced handicap.

Training is **open loop**: encode the history, then roll the latent forward for
`L` steps driven only by the action sequence, supervising observation, reward and
event readouts at every step, with a geometric discount that keeps the one-step
term dominant. RSSM adds KL-balanced divergence between posterior and prior; JEPA
adds representation-space prediction against an EMA target encoder with VICReg
variance and covariance terms, and trains its decoder on detached latents so
reconstruction never shapes the representation.

## Planning

```
encode current state
  -> sample candidate action sequences
  -> roll them through the learned model
  -> score: discounted predicted reward - risk_lambda * predicted risk events
  -> take the elite set, refit the per-step categorical (CEM)
  -> execute only the first action of the best sequence
  -> observe the real consequence, re-encode, replan
```

Feasibility masks come from the agent's own observable position, so they change
efficiency, not information. Every imagined transition is counted.

## Unified system

```
        observation window
                |
   +------------+-------------+
   |            |             |
 reason       plan          recall
 (LLM /     (CEM in the   (episodic
  rules)     world model)  memory)
   |            |             |
   +------------+-------------+
                |
          LinUCB gate  <-- context includes the world model's own recent
                |            one-step error, realised volatility, solvency,
             action          memory-match distance and time
                |
        real environment step
                |
         consequence feedback --> memory, action values, model error, gate
```

The world model is scored on **every** step, including steps another pathway
drove; otherwise the gate could never learn that the model had become unreliable.
Ablations disable pathways by restricting the arm set.
