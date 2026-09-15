# Benchmark results

Config hash `576e10437fe5`. Total evaluation wall time 76.8s.

## Training cost and capacity

| model | family | params | train_s | peak_rss_mb |
|---|---|---|---|---|
| persistence | baseline | 11 | 0.0 | 0.0 |
| ar | baseline | 100464 | 1.7 | 0.0 |
| wm_rssm | world_model | 714811 | 364.6 | 1732.7 |
| wm_rssm_1step | world_model | 714811 | 237.3 | 1732.7 |

## Prediction

Targets are grouped by mechanism, because a single pooled MSE over all 112 features is dominated by two artefacts in this world:

* `ret1_*` / `pool_move_*` are already one-step differences, so predicting their change is mostly the identity "next return ~ its mean" (a ridge fit scores +0.99 there while learning nothing);
* `logprice_*` changes are a martingale by construction, so nobody can predict them.

`dskill` is therefore the headline: skill on the **decision-relevant** groups (the focal agent's own balance sheet plus endogenous protocol state), scored against whichever naive reference -- persistence or a fitted linear autoregression -- does better on that split.

`price_level_skill_iid` is a **leakage canary**: price changes are unpredictable by construction, so a clearly positive value there means a model is reading something it should not be able to read.

### Decision-state skill vs the best naive reference (headline)

| model | dskill@test_iid | dskill@ood_liquidity_crisis | dskill@ood_unseen_agents | dskill@ood_coordinated_attack | dskill@ood_token_economics | dskill@ood_amm_params | dskill@ood_novel_mechanism | dskill@ood_shock_storm | dskill_ood_mean | dskill_ood_min | ood_retention |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ar | 0.0000 | 0.0000 | -1483.8984 | -638.5958 | 0.0000 | -0.4455 | -4.5397 | -0.5405 | -304.0029 | -1483.8984 | -- |
| wm_rssm_1step | -0.2553 | -0.0268 | -0.0172 | -0.0166 | -0.2592 | 0.1552 | 0.0012 | 0.0740 | -0.0128 | -0.2592 | -- |
| wm_rssm | -0.6158 | -0.1483 | 0.0013 | -0.0009 | -0.5688 | 0.0036 | 0.0001 | 0.0027 | -0.1015 | -0.5688 | -- |
| persistence | -0.6350 | -0.1508 | 0.0000 | 0.0000 | -0.5866 | 0.0000 | 0.0000 | 0.0000 | -0.1053 | -0.5866 | -- |

### Per-group detail (in-distribution)

| model | agent_skill_iid | protocol_skill_iid | population_skill_iid | price_skill_iid | price_level_skill_iid | next_state_skill_pers_iid |
|---|---|---|---|---|---|---|
| wm_rssm | 0.1940 | -0.6820 | -0.0841 | -320.8657 | -0.0013 | 0.0028 |
| wm_rssm_1step | 0.0580 | -0.2810 | -0.0766 | -132.5156 | -0.0129 | 0.5012 |
| ar | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0216 | 0.8527 |
| persistence | -0.0022 | -0.6868 | -0.0909 | -320.9113 | 0.0000 | 0.0000 |

### Event and reward prediction (in-distribution)

| model | event_logloss_iid | event_brier_iid | event_ece_iid | event_auroc_iid | reward_skill_iid |
|---|---|---|---|---|---|
| wm_rssm_1step | 0.0937 | 0.0268 | 0.0036 | 0.8044 | -0.0007 |
| wm_rssm | 0.1041 | 0.0308 | 0.0048 | 0.7862 | -0.0695 |
| persistence | 0.2099 | 0.0629 | 0.0023 | 0.5000 | -0.0012 |
| ar | 0.2099 | 0.0629 | 0.0041 | 0.5000 | -0.0014 |

## Counterfactual reasoning

`cf_skill` is the skill score against a predictor that says the action changes nothing. Positive means genuine causal signal.

| model | cf_skill@iid:substitute | cf_skill@iid:removal | cf_skill@liquidity_crisis:substitute | cf_skill@liquidity_crisis:removal | cf_skill@novel_mechanism:substitute | cf_skill@novel_mechanism:removal | cf_skill_mean | sign_acc_material_mean |
|---|---|---|---|---|---|---|---|---|
| wm_rssm | 0.2507 | 0.1567 | 0.1167 | 0.1761 | 0.1790 | -0.0576 | 0.1369 | 0.6000 |
| persistence | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ar | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| wm_rssm_1step | 0.0499 | 0.0261 | -0.0078 | -0.1538 | 0.0768 | -0.0434 | -0.0087 | 0.5985 |

## Control (total return per episode)

| policy | ret@iid | return_mean | sharpe_mean | max_drawdown_mean |
|---|---|---|---|---|
| noop | 0.0204 | 0.0204 | 0.0416 | 0.0314 |
| wm_plan | -0.0121 | -0.0121 | -0.0121 | 0.0572 |

### Episode-wise regret (lower is better)

| policy | regret@iid | regret_mean |
|---|---|---|
| noop | 0.0023 | 0.0023 |
| wm_plan | 0.0347 | 0.0347 |

### Paired comparison against `noop`

| policy | mean_diff | ci_lo | ci_hi | p_value | significant_05 | n_pairs |
|---|---|---|---|---|---|---|
| wm_plan | -0.0325 | -0.0695 | 0.0045 | 1.0000 | False | 2 |

## Composite GPC index (never read without the components)

Systems are only comparable at equal `coverage`.

| system | composite | coverage | interpolation | extrapolation | event_calibration | counterfactual | long_horizon | planning |
|---|---|---|---|---|---|---|---|---|
| wm_rssm | 0.0228 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1369 | 0.0000 | 0.0000 |
| policy:wm_plan | 0.0228 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1369 | 0.0000 | 0.0000 |
| persistence | 0.0000 | 0.8333 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -- |
| ar | 0.0000 | 0.8333 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -- |
| wm_rssm_1step | 0.0000 | 0.8333 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -- |
| policy:noop | 0.0000 | 0.1667 | -- | -- | -- | -- | -- | 0.0000 |

## Pre-registered falsification checks

These criteria were written in `docs/METHODOLOGY.md` before the results were seen.

Counts: inconclusive=10

| check | verdict | detail |
|---|---|---|
| `latent_vs_same_objective` | **--** | counterfactual scores unavailable for the comparison |
| `beyond_sequence_modelling` | **--** | long-horizon scores unavailable |
| `planning_pays` | **--** | wm_plan vs wm_greedy not available |
| `world_model_beats_reasoner` | **--** | wm_plan vs llm not available |
| `unified_beats_world_model` | **--** | unified vs wm_plan not available |
| `unified_beats_reasoner` | **--** | unified vs llm not available |
| `environment_rewards_dynamics` | **--** | oracle/noop unavailable |
| `memory_helps_world_model` | **--** | wm_plan_memory vs wm_plan not available |
| `memory_helps_unified` | **--** | unified vs unified_no_memory not available |
| `rollouts_help_unified` | **--** | unified vs unified_no_rollout not available |


## Automated audit

Counts: info=2

- **info** · `no_causal_signal` — persistence never beats the zero-effect predictor (best +0.0000); it has no measurable causal understanding in this world.
- **info** · `no_causal_signal` — ar never beats the zero-effect predictor (best +0.0000); it has no measurable causal understanding in this world.


## Contamination check

- episode seeds disjoint across splits: **True**
- largest verbatim observation-row overlap with train: **0.00000**
