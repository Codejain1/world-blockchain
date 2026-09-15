# Benchmark results

Config hash `aa8b3c823993`. Total evaluation wall time 1953.7s.

## Training cost and capacity

| model | family | params | train_s | peak_rss_mb |
|---|---|---|---|---|
| persistence | baseline | 11 | 0.0 | 0.0 |
| ar | baseline | 100464 | 1.6 | 0.0 |
| linear | baseline | 202827 | 73.0 | 0.0 |
| gbdt | baseline | 130200 | 390.5 | 0.0 |
| mlp | baseline | 626011 | 53.2 | 2309.7 |
| transformer | baseline | 546395 | 235.5 | 2309.7 |
| gnn | baseline | 462571 | 793.5 | 2309.7 |
| obsspace | baseline | 602939 | 144.0 | 2309.7 |
| wm_rssm | world_model | 714811 | 338.6 | 2309.7 |
| wm_jepa | world_model | 815291 | 441.1 | 2309.7 |
| wm_transformer | world_model | 901947 | 1334.4 | 2309.7 |
| wm_graph | world_model | 296683 | 2368.9 | 2659.9 |
| wm_rssm_1step | world_model | 714811 | 241.5 | 2659.9 |

## Prediction

Targets are grouped by mechanism, because a single pooled MSE over all 112 features is dominated by two artefacts in this world:

* `ret1_*` / `pool_move_*` are already one-step differences, so predicting their change is mostly the identity "next return ~ its mean" (a ridge fit scores +0.99 there while learning nothing);
* `logprice_*` changes are a martingale by construction, so nobody can predict them.

`dskill` is therefore the headline: skill on the **decision-relevant** groups (the focal agent's own balance sheet plus endogenous protocol state), scored against whichever naive reference -- persistence or a fitted linear autoregression -- does better on that split.

`price_level_skill_iid` is a **leakage canary**: price changes are unpredictable by construction, so a clearly positive value there means a model is reading something it should not be able to read.

### Decision-state skill vs the best naive reference (headline)

| model | dskill@test_iid | dskill@ood_liquidity_crisis | dskill@ood_unseen_agents | dskill@ood_coordinated_attack | dskill@ood_token_economics | dskill@ood_amm_params | dskill@ood_novel_mechanism | dskill@ood_shock_storm | dskill_ood_mean | dskill_ood_min | ood_retention |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gbdt | 0.2140 | 0.3578 | 0.4006 | 0.3424 | 0.2014 | 0.4698 | 0.0031 | 0.2560 | 0.2901 | 0.0031 | 1.3558 |
| transformer | 0.1577 | 0.2924 | -0.5950 | -0.5368 | 0.1380 | 0.4178 | 0.0024 | 0.2027 | -0.0112 | -0.5950 | -0.0710 |
| gnn | 0.1451 | 0.3099 | -0.4858 | -0.1635 | 0.1243 | 0.4079 | 0.0026 | 0.2189 | 0.0592 | -0.4858 | 0.4078 |
| mlp | 0.1038 | 0.2243 | -0.4798 | -0.2765 | 0.0851 | 0.3731 | 0.0031 | 0.2002 | 0.0185 | -0.4798 | 0.1782 |
| wm_transformer | 0.0889 | 0.2538 | -1.0694 | -1.0454 | 0.0873 | 0.3371 | 0.0024 | 0.1663 | -0.1811 | -1.0694 | -2.0366 |
| linear | 0.0312 | 0.1253 | 0.3019 | 0.2774 | 0.0331 | -0.2903 | -2.5847 | -0.4040 | -0.3630 | -2.5847 | -- |
| obsspace | 0.0059 | 0.1769 | -3.2995 | -2.0435 | -0.0209 | 0.2618 | 0.0024 | 0.1637 | -0.6799 | -3.2995 | -- |
| ar | 0.0000 | 0.0000 | -1483.8984 | -638.5958 | 0.0000 | -0.4455 | -4.5397 | -0.5405 | -304.0029 | -1483.8984 | -- |
| wm_graph | -0.0280 | 0.1359 | -0.5036 | -0.3458 | -0.0249 | 0.2219 | 0.0019 | 0.1266 | -0.0554 | -0.5036 | -- |
| wm_rssm_1step | -0.1826 | 0.0099 | -0.0018 | 0.0015 | -0.2045 | 0.1488 | 0.0016 | 0.0911 | 0.0066 | -0.2045 | -- |
| wm_jepa | -0.1919 | -0.0225 | -0.0932 | -0.2893 | -0.2376 | 0.0709 | 0.0009 | 0.0960 | -0.0678 | -0.2893 | -- |
| wm_rssm | -0.6156 | -0.1483 | -0.0072 | -0.0053 | -0.5668 | 0.0042 | 0.0001 | 0.0006 | -0.1033 | -0.5668 | -- |
| persistence | -0.6350 | -0.1508 | 0.0000 | 0.0000 | -0.5866 | 0.0000 | 0.0000 | 0.0000 | -0.1053 | -0.5866 | -- |

### Per-group detail (in-distribution)

| model | agent_skill_iid | protocol_skill_iid | population_skill_iid | price_skill_iid | price_level_skill_iid | next_state_skill_pers_iid |
|---|---|---|---|---|---|---|
| linear | 0.5348 | -0.0100 | -0.0271 | -0.0179 | -0.0346 | 0.8567 |
| wm_graph | 0.2824 | -0.0534 | -0.0399 | -62.2471 | -0.0222 | 0.7002 |
| wm_rssm | 0.1525 | -0.6784 | -0.0578 | -320.8516 | -0.0019 | 0.0031 |
| wm_rssm_1step | 0.0726 | -0.2035 | -0.0849 | -129.1187 | -0.0067 | 0.5192 |
| transformer | 0.0502 | 0.1665 | 0.0660 | -3.8879 | -0.0282 | 0.8656 |
| gnn | 0.0224 | 0.1552 | -0.0114 | -4.9957 | -0.0241 | 0.8606 |
| obsspace | 0.0136 | 0.0053 | -0.2114 | -17.7010 | -0.0380 | 0.8096 |
| mlp | 0.0014 | 0.1122 | -0.0231 | -5.9333 | -0.0097 | 0.8526 |
| wm_transformer | 0.0012 | 0.0961 | -0.0992 | -10.0018 | -0.0278 | 0.8403 |
| wm_jepa | 0.0008 | -0.2077 | -0.0784 | -181.1631 | -0.0089 | 0.3939 |
| ar | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0216 | 0.8527 |
| persistence | -0.0022 | -0.6868 | -0.0909 | -320.9113 | 0.0000 | 0.0000 |
| gbdt | -0.0022 | 0.2317 | 0.0709 | -8.1626 | 0.0000 | 0.8631 |

### Event and reward prediction (in-distribution)

| model | event_logloss_iid | event_brier_iid | event_ece_iid | event_auroc_iid | reward_skill_iid |
|---|---|---|---|---|---|
| gnn | 0.0753 | 0.0218 | 0.0055 | 0.8704 | -0.0157 |
| mlp | 0.0773 | 0.0225 | 0.0034 | 0.8667 | -0.0187 |
| wm_transformer | 0.0774 | 0.0223 | 0.0037 | 0.8645 | -0.0573 |
| gbdt | 0.0719 | 0.0199 | 0.0031 | 0.8623 | -0.2298 |
| transformer | 0.0740 | 0.0214 | 0.0047 | 0.8571 | -0.0290 |
| obsspace | 0.0847 | 0.0246 | 0.0087 | 0.8533 | -0.0841 |
| linear | 0.0822 | 0.0242 | 0.0042 | 0.8396 | -0.1723 |
| wm_graph | 0.0836 | 0.0244 | 0.0055 | 0.8146 | -0.0182 |
| wm_rssm_1step | 0.0935 | 0.0265 | 0.0044 | 0.8033 | -0.0102 |
| wm_rssm | 0.0964 | 0.0282 | 0.0042 | 0.7991 | -0.0083 |
| wm_jepa | 0.1164 | 0.0337 | 0.0068 | 0.7764 | -0.0048 |
| persistence | 0.2099 | 0.0629 | 0.0023 | 0.5000 | -0.0012 |
| ar | 0.2099 | 0.0629 | 0.0041 | 0.5000 | -0.0014 |

## Counterfactual reasoning

`cf_skill` is the skill score against a predictor that says the action changes nothing. Positive means genuine causal signal.

| model | cf_skill@iid:substitute | cf_skill@iid:removal | cf_skill@liquidity_crisis:substitute | cf_skill@liquidity_crisis:removal | cf_skill@unseen_agents:substitute | cf_skill@unseen_agents:removal | cf_skill@novel_mechanism:substitute | cf_skill@novel_mechanism:removal | cf_skill_mean | sign_acc_material_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| linear | 0.5521 | 0.4937 | 0.3476 | 0.3744 | 0.5930 | 0.5113 | 0.6170 | 0.6473 | 0.5171 | 0.6362 |
| wm_rssm | 0.1694 | 0.1757 | 0.0972 | 0.0659 | 0.0751 | -0.0085 | 0.3304 | 0.2449 | 0.1438 | 0.5990 |
| transformer | 0.0580 | 0.0500 | 0.0240 | 0.0122 | 0.0008 | 0.0017 | -0.1016 | 0.0325 | 0.0097 | 0.5870 |
| wm_rssm_1step | 0.0413 | 0.0228 | 0.0067 | -0.3108 | -0.0199 | 0.0895 | 0.0908 | 0.0895 | 0.0012 | 0.6036 |
| persistence | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ar | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| obsspace | -0.0134 | -0.0405 | -0.0200 | -0.0826 | 0.0000 | 0.0000 | -0.0443 | -0.0437 | -0.0306 | 0.5406 |
| gnn | 0.0316 | 0.0037 | -0.0673 | -0.1619 | 0.0003 | 0.0003 | -0.0618 | -0.0166 | -0.0340 | 0.5833 |
| wm_jepa | -0.0228 | -0.0555 | -0.0448 | -0.0848 | -0.3090 | -0.5902 | -0.0317 | -0.0139 | -0.1441 | 0.5064 |
| mlp | -0.0051 | -0.0313 | -0.3611 | -1.1760 | 0.0000 | 0.0000 | -0.0307 | -0.0924 | -0.2121 | 0.5762 |
| wm_transformer | 0.0087 | -0.0188 | -0.0171 | -0.0808 | -1.2543 | -0.7709 | 0.0028 | 0.0088 | -0.2652 | 0.6108 |
| gbdt | -2.2441 | -1.9776 | -0.7407 | -0.3084 | -1.6952 | -0.4370 | -0.4857 | -1.5392 | -1.1785 | 0.0186 |
| wm_graph | 0.2849 | 0.3081 | -0.4448 | -1.2589 | -7.2736 | -16.4993 | 0.4303 | 0.2678 | -3.0232 | 0.5916 |

## Control (total return per episode)

| policy | ret@iid | ret@liquidity_crisis | ret@unseen_agents | ret@novel_mechanism | ret@shock_storm | return_mean | sharpe_mean | max_drawdown_mean |
|---|---|---|---|---|---|---|---|---|
| oracle_plan | 0.2051 | 0.4443 | 0.1286 | 0.0866 | 0.0933 | 0.1916 | 0.1438 | 0.0675 |
| noop | 0.0201 | -0.0105 | 0.0030 | 0.0026 | -0.0035 | 0.0023 | 0.0077 | 0.0672 |
| buy_and_hold | 0.0259 | -0.0157 | 0.0020 | -0.0449 | -0.0051 | -0.0076 | -0.0071 | 0.0820 |
| memory_only | 0.0080 | -0.0612 | -0.0699 | -0.0400 | -0.0387 | -0.0404 | -0.0439 | 0.1110 |
| wm_greedy | 0.0266 | 0.0665 | -0.1345 | -0.0153 | -0.1541 | -0.0422 | -0.0685 | 0.1155 |
| wm_plan | -0.0211 | 0.0343 | -0.1344 | -0.0705 | -0.0686 | -0.0520 | -0.0959 | 0.1264 |
| wm_plan_memory | -0.0107 | 0.0344 | -0.1378 | -0.0704 | -0.1012 | -0.0571 | -0.0979 | 0.1274 |
| random | -0.0009 | -0.0634 | -0.0475 | -0.0880 | -0.0924 | -0.0584 | -0.0523 | 0.1199 |
| planner_no_model | -0.0382 | -0.2150 | -0.0711 | 0.0115 | -0.0621 | -0.0750 | -0.0760 | 0.1842 |
| unified_no_memory | -0.0387 | -0.1013 | -0.1230 | -0.1121 | -0.0354 | -0.0821 | -0.0908 | 0.1470 |
| unified | -0.0741 | -0.3052 | -0.0600 | -0.0612 | -0.1523 | -0.1306 | -0.1016 | 0.2406 |
| unified_no_rollout | -0.0628 | -0.3712 | -0.1371 | -0.1425 | -0.1266 | -0.1680 | -0.1347 | 0.2553 |
| llm | -0.0791 | -0.5759 | -0.1877 | -0.1245 | -0.1254 | -0.2185 | -0.1488 | 0.3102 |

### Episode-wise regret (lower is better)

| policy | regret@iid | regret@liquidity_crisis | regret@unseen_agents | regret@novel_mechanism | regret@shock_storm | regret_mean |
|---|---|---|---|---|---|---|
| oracle_plan | 0.0401 | 0.0000 | 0.0000 | 0.0815 | 0.0667 | 0.0377 |
| noop | 0.2251 | 0.4549 | 0.1256 | 0.1654 | 0.1635 | 0.2269 |
| buy_and_hold | 0.2193 | 0.4600 | 0.1266 | 0.2130 | 0.1651 | 0.2368 |
| memory_only | 0.2372 | 0.5055 | 0.1985 | 0.2081 | 0.1987 | 0.2696 |
| wm_greedy | 0.2186 | 0.3778 | 0.2631 | 0.1834 | 0.3141 | 0.2714 |
| wm_plan | 0.2663 | 0.4100 | 0.2629 | 0.2386 | 0.2286 | 0.2813 |
| wm_plan_memory | 0.2559 | 0.4099 | 0.2663 | 0.2385 | 0.2612 | 0.2864 |
| random | 0.2461 | 0.5077 | 0.1761 | 0.2561 | 0.2524 | 0.2877 |
| planner_no_model | 0.2834 | 0.6594 | 0.1997 | 0.1566 | 0.2221 | 0.3042 |
| unified_no_memory | 0.2839 | 0.5456 | 0.2515 | 0.2802 | 0.1954 | 0.3113 |
| unified | 0.3193 | 0.7495 | 0.1885 | 0.2292 | 0.3124 | 0.3598 |
| unified_no_rollout | 0.3080 | 0.8155 | 0.2657 | 0.3106 | 0.2866 | 0.3973 |
| llm | 0.3243 | 1.0202 | 0.3163 | 0.2926 | 0.2854 | 0.4478 |

### Head-to-head tests (paired over identical episodes)

Every policy ran the same episode seeds, so these are paired comparisons: a bootstrap CI on the mean difference in total return, plus an exact sign-flip permutation p-value.

| comparison | mean_diff | ci_lo | ci_hi | p_value | n_pairs |
|---|---|---|---|---|---|
| wm_plan - llm | 0.1665 | 0.0338 | 0.3045 | 0.0278 | 20 |
| unified - wm_plan | -0.0785 | -0.2013 | 0.0324 | 0.2224 | 20 |
| unified - llm | 0.0880 | 0.0043 | 0.1838 | 0.0761 | 20 |
| wm_plan - wm_greedy | -0.0099 | -0.0482 | 0.0278 | 0.6273 | 20 |
| wm_plan_memory - wm_plan | -0.0051 | -0.0224 | 0.0097 | 0.5793 | 20 |
| unified - unified_no_memory | -0.0485 | -0.1554 | 0.0563 | 0.4005 | 20 |
| unified - unified_no_rollout | 0.0375 | -0.0112 | 0.0895 | 0.1805 | 20 |
| oracle_plan - wm_plan | 0.2436 | 0.1794 | 0.3182 | 0.0000 | 20 |
| planner_no_model - wm_plan | -0.0229 | -0.1222 | 0.0715 | 0.6606 | 20 |
| wm_plan - buy_and_hold | -0.0445 | -0.0861 | -0.0039 | 0.0499 | 20 |
| llm - noop | -0.2209 | -0.3407 | -0.1007 | 0.0022 | 20 |

### Paired comparison against `noop`

| policy | mean_diff | ci_lo | ci_hi | p_value | significant_05 | n_pairs |
|---|---|---|---|---|---|---|
| oracle_plan | 0.1892 | 0.1144 | 0.2873 | 0.0000 | True | 20 |
| buy_and_hold | -0.0099 | -0.0237 | 0.0038 | 0.1862 | False | 20 |
| memory_only | -0.0427 | -0.0716 | -0.0163 | 0.0072 | True | 20 |
| wm_greedy | -0.0445 | -0.0983 | 0.0084 | 0.1258 | False | 20 |
| wm_plan | -0.0544 | -0.0942 | -0.0160 | 0.0137 | True | 20 |
| wm_plan_memory | -0.0595 | -0.1038 | -0.0171 | 0.0158 | True | 20 |
| random | -0.0608 | -0.0884 | -0.0344 | 0.0003 | True | 20 |
| planner_no_model | -0.0773 | -0.1648 | 0.0075 | 0.1038 | False | 20 |
| unified_no_memory | -0.0844 | -0.1221 | -0.0371 | 0.0015 | True | 20 |
| unified | -0.1329 | -0.2438 | -0.0301 | 0.0247 | True | 20 |
| unified_no_rollout | -0.1704 | -0.2791 | -0.0650 | 0.0058 | True | 20 |
| llm | -0.2209 | -0.3407 | -0.1007 | 0.0022 | True | 20 |

## Composite GPC index (never read without the components)

Systems are only comparable at equal `coverage`.

| system | composite | coverage | interpolation | extrapolation | event_calibration | counterfactual | long_horizon | planning |
|---|---|---|---|---|---|---|---|---|
| policy:oracle_plan | 1.0000 | 0.1667 | -- | -- | -- | -- | -- | 1.0000 |
| gbdt | 0.2402 | 0.8333 | 0.2140 | 0.2901 | 0.6250 | 0.0000 | 0.0718 | -- |
| linear | 0.1097 | 0.8333 | 0.0312 | 0.0000 | 0.0000 | 0.5171 | 0.0000 | -- |
| wm_transformer | 0.0411 | 0.8333 | 0.0889 | 0.0000 | 0.1164 | 0.0000 | 0.0000 | -- |
| gnn | 0.0409 | 0.8333 | 0.1451 | 0.0592 | 0.0000 | 0.0000 | 0.0000 | -- |
| transformer | 0.0335 | 0.8333 | 0.1577 | 0.0000 | 0.0000 | 0.0097 | 0.0000 | -- |
| mlp | 0.0245 | 0.8333 | 0.1038 | 0.0185 | 0.0000 | 0.0000 | 0.0000 | -- |
| wm_rssm | 0.0240 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1438 | 0.0000 | 0.0000 |
| policy:wm_greedy | 0.0240 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1438 | 0.0000 | 0.0000 |
| policy:wm_plan | 0.0240 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1438 | 0.0000 | 0.0000 |
| policy:wm_plan_memory | 0.0240 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1438 | 0.0000 | 0.0000 |
| policy:unified | 0.0240 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1438 | 0.0000 | 0.0000 |
| policy:unified_no_memory | 0.0240 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1438 | 0.0000 | 0.0000 |
| policy:unified_no_rollout | 0.0240 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1438 | 0.0000 | 0.0000 |
| wm_jepa | 0.0230 | 0.8333 | 0.0000 | 0.0000 | 0.1152 | 0.0000 | 0.0000 | -- |
| wm_rssm_1step | 0.0016 | 0.8333 | 0.0000 | 0.0066 | 0.0000 | 0.0012 | 0.0000 | -- |
| obsspace | 0.0012 | 0.8333 | 0.0059 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -- |
| persistence | 0.0000 | 0.8333 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -- |
| ar | 0.0000 | 0.8333 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -- |
| wm_graph | 0.0000 | 0.8333 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -- |
| policy:noop | 0.0000 | 0.1667 | -- | -- | -- | -- | -- | 0.0000 |
| policy:random | 0.0000 | 0.1667 | -- | -- | -- | -- | -- | 0.0000 |
| policy:buy_and_hold | 0.0000 | 0.1667 | -- | -- | -- | -- | -- | 0.0000 |
| policy:memory_only | 0.0000 | 0.1667 | -- | -- | -- | -- | -- | 0.0000 |
| policy:llm | 0.0000 | 0.1667 | -- | -- | -- | -- | -- | 0.0000 |
| policy:planner_no_model | 0.0000 | 0.1667 | -- | -- | -- | -- | -- | 0.0000 |

## Unified gate: which pathway did it choose?

Pathways are `reason`, `plan`, `recall`.

| policy | reason | plan | recall | reward_reason | reward_plan | reward_recall |
|---|---|---|---|---|---|---|
| unified | 0.9938 | 0.0062 | 0.0000 | 0.1617 | 0.0000 | 0.0000 |
| unified_no_memory | 0.0125 | 0.9875 | 0.0000 | 0.0000 | 0.1176 | 0.0000 |
| unified_no_rollout | 1.0000 | 0.0000 | 0.0000 | 0.1702 | 0.0000 | 0.0000 |

## LLM backend actually used

- `llm`: **rule_based_offline**

> Rows marked `rule_based_offline` were produced by the rule-based reasoner, not by a language model. They are a symbolic-reasoning baseline and must not be read as evidence about LLM capability.

## Pre-registered falsification checks

These criteria were written in `docs/METHODOLOGY.md` before the results were seen.

Counts: inconclusive=2, info=1, not_supported=4, supported=4

| check | verdict | detail |
|---|---|---|
| `latent_vs_same_objective` | **yes** | best world model (wm_rssm, +0.1438) beats the observation-space control with the identical multi-step objective (obsspace, -0.0306) on counterfactual skill |
| `beyond_sequence_modelling` | **yes** | best world model (wm_graph, +0.0570) beats the Transformer baseline (-0.0425) at the deepest horizon |
| `planning_pays` | **NO** | planning does not beat one-step greedy (-0.0099 return, p=0.627) |
| `world_model_beats_reasoner` | **yes** | wm_plan beats llm by +0.1665 return (p=0.028, n=20) |
| `unified_beats_world_model` | **NO** | unified does not beat wm_plan (-0.0785 return, p=0.222) |
| `unified_beats_reasoner` | **--** | unified leads llm by +0.0880 but not distinguishably (p=0.076, n=20) |
| `environment_rewards_dynamics` | **yes** | planning with the true simulator beats do-nothing by +0.1892 return, so accurate dynamics are worth something in this world |
| `learned_vs_perfect_dynamics` | **info** | the true-simulator planner beats the learned planner by +0.2436 return (p=0.000); this is the headroom a better learned model could recover |
| `memory_helps_world_model` | **NO** | wm_plan_memory - wm_plan = -0.0051 return (p=0.579, n=20) |
| `memory_helps_unified` | **NO** | unified - unified_no_memory = -0.0485 return (p=0.400, n=20) |
| `rollouts_help_unified` | **--** | unified - unified_no_rollout = +0.0375 return (p=0.180, n=20) |


## Automated audit

Counts: info=5

- **info** · `no_causal_signal` — persistence never beats the zero-effect predictor (best +0.0000); it has no measurable causal understanding in this world.
- **info** · `no_causal_signal` — ar never beats the zero-effect predictor (best +0.0000); it has no measurable causal understanding in this world.
- **info** · `no_causal_signal` — gbdt never beats the zero-effect predictor (best -0.3084); it has no measurable causal understanding in this world.
- **info** · `no_causal_signal` — wm_jepa never beats the zero-effect predictor (best -0.0139); it has no measurable causal understanding in this world.
- **info** · `environment_is_informative` — True-simulator planning beats do-nothing by +0.189 return, so accurate dynamics are worth something here.


## Contamination check

- episode seeds disjoint across splits: **True**
- largest verbatim observation-row overlap with train: **0.00000**
