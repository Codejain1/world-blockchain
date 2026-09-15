# Blockchain World Model Intelligence Lab

A runnable research system for testing one hypothesis:

> **LLM-only < World Model < Unified (LLM + World Model)**

The lab builds a deterministic, configurable blockchain economy, populates it with
heterogeneous autonomous agents, and evaluates three intelligence systems plus a
full set of scientific baselines on prediction, counterfactual reasoning,
planning, control and generalisation to held-out worlds.

It is built to be able to **falsify** the hypothesis. The benchmark, the fairness
controls and the artifact audits all exist so that a negative result would be
visible rather than absorbed.

---

## Quick start

```bash
pip install -e .                      # or: pip install -r requirements.txt

python scripts/generate_data.py --out datasets/main      # ~10 min on 4 cores
python scripts/run_experiment.py --config configs/main.yaml
```

Results land in `results/main/`: `RESULTS.md`, `results.json`, `tables/*.csv`,
`figures/*.png`.

Fast end-to-end check (a few minutes, exercises every stage):

```bash
python scripts/generate_data.py --out datasets/main --n-train 12 --n-val 4 \
    --n-test 4 --n-ood 2 --episode-length 64
python scripts/run_experiment.py --config configs/smoke.yaml
pytest -q
```

---

## The world

A DeFi economy exposing a clean transition `S_t + A_t -> S_{t+1}, R_t, E_t`.

| Mechanism | Implementation |
|---|---|
| Tokens & wallets | 4 tokens (numeraire, two risk assets, a governance token) |
| AMM / DEX | constant-product pools with fees, LP shares, a per-trade price-impact cap |
| Lending | Aave-style kinked rates, collateral factors, health factors, close factor, liquidation bonus, bad debt |
| Staking & governance | staked voting power; passed proposals mutate live protocol parameters |
| Gas | EIP-1559 base fee with a real demand side (accounts refuse to overpay) |
| Market | regime-switching price process (bull/bear/crab/crisis) with jumps |
| Shocks | liquidity flight, whale dump, depeg, protocol exploit, oracle glitch, gas spike |

One simulation step is one hour of chain time. Observations are 112 features;
**the hidden regime and the exogenous fundamental price are never observable**.

### Agents

Eleven archetypes: retail, whale, arbitrageur, market maker, liquidity provider,
borrower, keeper, governance participant, random, adversarial, coordinated.
Arbitrageurs size trades from the constant-product invariant and can short through
the lending market, which couples the AMM and the money market.

The population reproduces the qualitative signature of real DeFi without being
told to: arbitrageurs and whales profit, LPs bleed impermanent loss, and random
traders lose heavily to fees and adverse selection.

| archetype | mean episode return |
|---|---|
| arbitrageur | +19.8% |
| whale | +11.3% |
| keeper | +1.1% |
| governance | −1.8% |
| market maker | −2.4% |
| retail | −4.5% |
| borrower | −4.1% |
| liquidity provider | −8.6% |
| random | −37.8% |

### Determinism and counterfactuals

Exogenous randomness is **counter-based**: the noise at block `t` is a pure
function of `(seed, stream, t)` and does not depend on how many random numbers
were drawn before. Forking the world and taking a different action therefore
isolates the causal effect of that action instead of measuring RNG drift. This is
what makes the counterfactual experiments meaningful, and it is enforced by tests.

---

## The three systems

**System 1 — LLM.** Reasons over the raw observation window rendered as a named
table, choosing from the same action menu as everyone else. No world-model latent,
no simulator. Backends: Anthropic, OpenAI, a deterministic mock for tests, and an
offline path. *Without API credentials the policy falls back to an explicit
rule-based reasoner, and every results row is labelled `rule_based_offline`. Those
rows are a symbolic-reasoning baseline and are not evidence about LLMs.*

**System 2 — World model.** A learned latent model with four architectures
(RSSM, JEPA, Transformer latent dynamics, graph latent dynamics) sharing one
`encode / imagine / readout` interface, trained open loop so the latent must
survive multi-step rollout. Model-predictive control (CEM, beam search or random
shooting) proposes action sequences, rolls them through the *learned model only*,
and executes just the first action.

**System 3 — Unified.** Foundation reasoning + latent world model + episodic
memory + planner, arbitrated by a LinUCB gate over three pathways (`reason`,
`plan`, `recall`). The gate's context includes the world model's own recent
one-step error, so the system can **learn to stop trusting its model** rather than
being wired to trust it.

### Baselines

Persistence, closed-form autoregression, ridge + logistic regression, LightGBM /
XGBoost, MLP, temporal Transformer, temporal graph network — plus control
policies (do-nothing, random, buy-and-hold, memory-only) and an
observation-space dynamics model that gets the *identical* multi-step objective
with no learned latent.

---

## Fairness

Enforced in code, not by convention:

- identical observation window `H` and action menu for every system;
- the hidden regime and fundamental price withheld from all of them;
- per-cluster aggregates added to the flat observation so the graph view carries
  the **same information** and any graph advantage is inductive bias, not data;
- world models decode residually so no architecture starts from a better prior;
- one shared trainer: same optimiser, schedule, batch size and **same number of
  gradient steps** for every learned system;
- exactly one real environment step per decision, for every policy;
- parameters, wall clock, environment steps, imagined rollouts, oracle simulator
  steps and LLM tokens are all metered and reported next to accuracy.

The only system with privileged access is the oracle planner (ablation E), which
plans with the true simulator. It is labelled as an upper bound and its cost is
reported.

---

## Benchmark

Targets are grouped by mechanism, because a pooled MSE over all features was
measured and found to be dominated by artifacts — see
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). The headline is **decision-state
skill**: the focal agent's balance sheet plus endogenous protocol state, scored
against whichever naive reference (persistence or linear AR) does better on that
split. Price-level skill is reported as a **leakage canary**.

Reported metrics: skill scores, R², log loss, Brier, ECE, reliability curves,
AUROC, counterfactual skill against a zero-effect predictor, long-horizon error
curves, cumulative return, Sharpe, Sortino, drawdown, episode-wise regret,
adaptation within episode, and compute cost.

A composite **GPC** index (Generalization + Planning + Counterfactual) is
produced, but it is constructed so it cannot hide its components: every component
is stored raw alongside it, unsupported components are `None` rather than zero,
and systems are only comparable at equal `coverage`.

### Held-out worlds

Train on the base distribution; test on worlds that differ only by configuration:

| scenario | shift type |
|---|---|
| `iid` | interpolation (disjoint seeds only) |
| `liquidity_crisis`, `amm_params`, `token_economics` | parameter shift |
| `unseen_agents`, `coordinated_attack` | behaviour shift (archetypes absent from training) |
| `novel_mechanism`, `shock_storm` | rule shift (mechanisms that do not exist in training) |

Splits are defined by **disjoint episode-seed ranges**, never by slicing inside an
episode, and the disjointness is verified mechanically.

### Ablations

`A` LLM only · `B` world model only · `C` unified · `D` world model without
planning · `E` planning with the true simulator · `F` reasoner + search without a
learned model · `G` world model + memory · `H` world model without memory ·
`I` unified without memory · `J` unified without counterfactual rollouts.

---

## Repository layout

```
bwm/
  environment/   world state, protocols, transition, observations, action space, scenarios
  agents/        archetypes and population assembly
  data/          trajectory generation, windowed datasets, contamination audits
  models/
    baselines/   classical + neural baselines, control policies
    llm/         provider-agnostic client, System 1 policy, rule-based reasoner
    world_model/ RSSM, JEPA, Transformer and graph latent dynamics + the WM policy
    unified/     System 3 and the LinUCB gate
  planning/      CEM / beam MPC, oracle-simulator planner, analytic planner
  memory/        episodic memory and consequence tracking
  training/      shared supervised trainer
  evaluation/    metrics, prediction, counterfactual, control, composite benchmark
  experiments/   registry, training and evaluation pipelines
  visualization/ figures and report generation
configs/  tests/  scripts/  docs/
```

Everything (models, environments, seeds, datasets, hyperparameters) is driven by
YAML configs with `_base_` inheritance and dotted CLI overrides:

```bash
python scripts/run_experiment.py --config configs/main.yaml \
    --set train.epochs=30 planner.n_candidates=128 llm.provider=openai
```

---

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the pieces fit together
- [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — experimental design and what each metric means
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — exact commands, seeds and environment
- [`docs/RESULTS.md`](docs/RESULTS.md) — findings, analysis, limitations and the verdict on the hypothesis
