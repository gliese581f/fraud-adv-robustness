# Where every result comes from

This file maps each table, figure and quoted number of the study (main text and supplement) to the code that
produces it and to the CSV or figure it is in. Numbers such as "Table III" or "Table S-XI" follow the current
version of the study.

Each result goes through two stages:

1. **Run.** A command attacks, trains or analyses models, and writes raw outputs under `runs/<dataset>/`,
   `runs_consistent/<dataset>/` (the consistent attacker) or `runs_ivisvar/` (the ivis variants).
2. **Summarize.** The report modules turn the raw outputs into CSVs and figures (`.pdf` and `.png`) in `reports/`:
   `python -m fraudadv.report.make_report` writes the summary CSVs and most figures, and the other modules write
   the tables that need more computation (`reports/tables/`, `reports/causal/`, `reports/attack_views/`,
   `reports/shap/`).

`bash scripts/make_results.sh` runs stage 2 in the right order from saved results. `scripts/reproduce.sh <dataset>`
runs stage 1 for one dataset. Below, "Output" is the file under `reports/` (unless another folder is named), "Made
by" the stage-2 code and "Run step" the stage-1 command. Every module path is under `src/fraudadv/`. Rates are
shares in [0, 1].

`scripts/reproduce.sh` uses the options of the study's runs: every attack grid ran with `--full-bb` and an
exact-MILP budget of 3,600 s per model and threat model (`--milp-budget 3600`); foundation models were attacked on a
random 240 frauds of the sample (`--fm-full-n 240`), or on all of them on Sparkov. The study's runs split these steps
across several machines; `scripts/reproduce.sh` runs the same commands on one.

## Notebooks

The notebooks in `notebooks/` run stage 2 interactively, one group of results at a time, and show each result
inline. Each ends by checking the files it wrote against a previous build of `reports/` (`FRAUDADV_REFERENCE`).

| Notebook | Results |
|---|---|
| `00_build_all_results` | everything below, in the order of `scripts/make_results.sh` |
| `01_data_and_threat_models` | Table II; feature schema; the three threat models on a few frauds |
| `02_models_and_clean_performance` | Tables S-II, S-IV; clean performance (Section VI-A) |
| `03_attack_grid` | Tables IV, S-V to S-XI; Figures S1, S2; RQ1 and RQ2 numbers |
| `04_consistent_attacker` | Table III; Tables S-XIII to S-XX; Figure S4 |
| `05_transfer_and_thickness` | Figures 1, S3, S5; Table S-XXIV; RQ3 numbers |
| `06_history_and_foundation_models` | Tables V, S-XII, S-XXVIII, S-XLIII |
| `07_explanations` | Tables S-XXXII, S-XXXIII; Figures S8 to S12 |
| `08_ablations_and_attack_views` | Table VI; Tables S-XXXIV to S-XXXVII; Figures 2, S13 to S18 |
| `09_defenses_and_ranking_metrics` | Table VII; Tables S-XXIX (graph and foundation-model defenses), S-XXX, S-XXXI; Figures S6, S7 |
| `10_attack_checks_and_statistics` | Tables S-XXI to S-XXIII, S-XXXVIII to S-XLII |
| `11_ivis_variants` | Tables S-XXV to S-XXVII |

## Main text

| Item | Output | Made by | Run step (inputs) |
|---|---|---|---|
| Table I, prior evaluations | none (a literature summary) | n/a | n/a |
| Table II, datasets | `tables/datasets.csv` | `report/tables.py: datasets` | `data/prepare.py` (`data/processed/<d>/meta.json`), the grid's fraud sample (`runs/<d>/attacks/sample.npz`) |
| Table III, worst case under the three attackers, eps 0.25 | `tables/worst_case_three_attackers.csv`; mean normalized rank: `tables/worst_case_rank.csv` | `report/consistent_tables.py: worst_case` | grid (`attacks/run_attacks.py`) for U and C; `attacks/run_attacks.py --modes consistent` for K |
| Table IV, attack survey (mean ASR per family, attack and budget) | `table2_robust.csv` (mean over datasets) | `report/make_report.py: table_robust` | grid: `runs/<d>/attacks/results*.csv` |
| Table V, foundation models | `tables/foundation_models.csv` | `report/tables.py: foundation_models` (history column from `table10_history.csv`) | grid of `chronos2`/`timesfm`, `attacks/history.py`, `runs/<d>/<fm>/metrics.json`, `ablation_noemb.json` (`hpo/fm_ablation.py`) |
| Table VI, controlled ablations (why trees are easier to evade) | `causal/contrasts.csv`, rows with dataset `pooled` | `causal/analyze.py` | `causal/ablate.py <d>` on Sparkov, IEEE-CIS, TabFormer, BAF (`runs/<d>/causal/`) |
| Table VII, defenses under adaptive attack (tabular models) | `table7_defenses_by_dataset.csv` (per dataset), `table7_defenses.csv` (mean over datasets) | `report/make_report.py: defense_tables` | `defenses/run_defenses.py <d>` (`runs/<d>/defenses/results.csv`); ivis's adversarial augmentation `defenses/run_defenses_consistent.py <d> --families ivis --defenses adv_aug --both` (`results_ivis_adv_aug.csv`) |
| Figure 1, transfer rates (constrained) | `fig_transfer_constrained.{pdf,png}` | `report/make_report.py: fig_transfer("constrained")` | `attacks/transfer.py <d>` (`transfer_constrained.npz`, from the grid's `adv_constrained_*.npz`) |
| Figure 2, what an evasion looks like | `fig_attack_views.{pdf,png}`; numbers quoted with it: `attack_views/attack_view_numbers.csv` | `report/attack_views.py figures` (drawing in `report/attack_figures.py`) | `report/attack_views.py compute <d>` on the ablation bases (`runs/<d>/causal/views.pkl`) |

### Numbers in the text

Every number in the text is a value of one of the CSVs above or below. The ones that are not in a table come from:

| Section | Numbers | Source |
|---|---|---|
| VI-A Clean performance | AUPRC, AUC, KS per model | `reports/table1_clean.csv` (`make_report.table_clean`) |
| VI-B RQ1, VI-C RQ2 | ASR per dataset, model and budget, ASR ranges, constrained/unconstrained ratios (81/78/51/25%, 0.73/0.55/0.52/0.37), attack-by-attack comparisons, budget check (7 of 92 cells) | `reports/tables/grid_per_dataset.csv`; `python scripts/check_claims.py` prints the ranges and ratios from the grid; budget check: `reports/tables/budget_check.csv` |
| VI-C Consistent attacker | C beside K per attack, history injection and best defense under K | `reports/tables/consistent_summary.csv`, `consistent_grid.csv`, `consistent_history.csv`, `consistent_defenses.csv` (`report/consistent_tables.py`) |
| VI-C Realizable records | shares of impossible records, medians 85/95/100% | `reports/analysis/consistent/realizable.csv` (`report/realizable_share.py`), `reports/tables/realizable_summary.csv` (`report/consistent_tables.py: realizable`) |
| VI-D Transferability | mean transfer rates, 67-97% to trees, consistent change <= 0.01 | `runs/<d>/attacks/transfer_*.npz`, `runs_consistent/<d>/attacks/transfer_consistent.npz` (`attacks/transfer.py <d> --modes consistent`), drawn in Figures 1, S3 and S4 |
| VI-E History injection | ASR after k injected transactions; 41-64% among identified IEEE-CIS clients | `reports/table10_history.csv` (`make_report.history_table`), `reports/table12_groups.csv` (`make_report.groups_table`) |
| VI-F Foundation models | accuracy and robustness comparisons, paired tests | `reports/tables/foundation_models.csv`, `reports/table1_clean.csv`, `reports/table6_mcnemar.csv` |
| VI-G RQ3 thickness | rank correlations -0.64/-0.60 (U), -0.29/-0.32 (C), p-values | `reports/table4_thickness_corr.csv` (`make_report.thickness_analysis`) |
| VI-H RQ4 explanations | attribution shares, Kendall tau, SHAP shift | `reports/table5_explain_summary.csv`, `table5_explain_agreement.csv` (`make_report.explain_summary`), `reports/shap/shap_shift_xgboost_constrained.csv` |
| VI-I Ablations | representation effect, distillation | `reports/causal/contrasts.csv`, `factorial_lpm.csv`, `factorial_effects.csv`, `variants_clean.csv` (`causal/analyze.py`, `causal/report.py`) |
| VI-J RQ5 Defenses | medians of improvement, mean worst case per group (0.32-0.35 etc.), ivis's augmentation; graph and foundation models: median gain of the better of PGD-AT and TRADES outside PaySim, squeezing | `reports/table7_defenses_by_dataset.csv`, `reports/table7_defense_tradeoff.csv` (`make_report.defense_tables`); graph and foundation models: `reports/tables/defenses_context.csv` |
| VI-K Ranking metrics | AUC/KS lost, median ratios 3.2/3.1 | `reports/table9_robust_auc.csv` (`make_report.robust_auc_table`, from `runs/<d>/attacks/robust_auc*.csv`), `reports/tables/robust_auc_mean.csv` |
| VI-L Statistics | mixed model, Friedman, McNemar pairs, Wilson intervals | `reports/table6_*.csv` (`make_report.stats_tables`, using `stats/inference.py`) |

## Supplement

| Item | Output | Made by | Run step (inputs) |
|---|---|---|---|
| Table S-I, attacks and settings | `table3_attacks.md` (the catalog); the settings are the defaults of `attacks/run_attacks.py` and `attacks/blackwhite.py` | `make_report.table_attacks` | none (settings) |
| Table S-II, models and hyperparameters | `tables/hpo_trials.csv` (finished trials); chosen hyperparameters in `runs/<d>/<f>/best_params.json` | `report/tables.py: hpo_trials` | `hpo/train_all.py`, `hpo/train_context.py` (`best_params.json`, `hpo_v2.json`) |
| Table S-III, hyperparameter search spaces | none (the spaces are defined in `hpo/train_all.py` and `hpo/train_context.py`) | n/a | n/a |
| Table S-IV, clean performance | `table1_clean.csv` | `make_report.table_clean` | `runs/<d>/<f>/metrics*.json` |
| Tables S-V to S-X, per-dataset grid | `tables/grid_per_dataset.csv` (one dataset per table) | `report/tables.py: grid_per_dataset` | grid `results*.csv` |
| Table S-XI, full attack survey | `table2_robust.csv`; per dataset `table2b_robust_<d>.md` | `make_report.table_robust` | grid |
| Table S-XII, history injection | `table10_history.csv` | `make_report.history_table` | `attacks/history.py <d>` (`history*.csv`) |
| Table S-XIII, impossible records | `analysis/consistent/realizable.csv`; range over models `tables/realizable_summary.csv` | `report/realizable_share.py`; `report/consistent_tables.py: realizable` | the stored `adv_*.npz` of all three attackers |
| Table S-XIV, consistent attacker in brief | `tables/consistent_summary.csv` | `report/consistent_tables.py: summary` | consistent grid, history, defenses |
| Tables S-XV to S-XVII, consistent grid per dataset | `tables/consistent_grid.csv` | `report/consistent_tables.py: grid` | `attacks/run_attacks.py --modes consistent` |
| Table S-XVIII, consistent ranking metrics | `tables/consistent_robust_auc.csv` | `report/consistent_tables.py: robust_auc` | `attacks/robust_auc.py <d> <families> --modes consistent --eps 0.25` |
| Table S-XIX, consistent history injection | `tables/consistent_history.csv` | `report/consistent_tables.py: history` | `attacks/history_consistent.py` |
| Table S-XX, consistent defenses | `tables/consistent_defenses.csv` | `report/consistent_tables.py: defenses` | `defenses/run_defenses_consistent.py` |
| Table S-XXI, five-times budget check | `tables/budget_check.csv` | `report/tables.py: budget_check` | `attacks/budget_check.py` (`runs/<d>/attacks/budget_check.csv`) |
| Table S-XXII, certification of AATS by the exact MILP | `table8_aats_vs_milp.csv`, modes `constrained` and `unconstrained` | `make_report.certify_table` | `attacks/certify.py <d> --mode constrained/unconstrained` |
| Table S-XXIII, certification, continuous threat model | `table8_aats_vs_milp.csv`, mode `continuous` | `make_report.certify_table` | first-grid `attacks/certify.py --mode continuous` |
| Table S-XXIV, boundary thickness | `table4_thickness_corr.csv` (per model: `table4_thickness.csv`) | `make_report.thickness_analysis` | grid `thickness*.csv`, `boundary/run_thickness.py` |
| Table S-XXV, ivis variants: clean AUPRC and worst case | `runs_ivisvar/report/clean.csv`, `worst_case.csv` | `scripts/analysis/ivis_variants.py report` | `scripts/analysis/ivis_variants.py tune/train/attack/adaptive` (`runs_ivisvar/`); consistent: the same `attack`/`adaptive` with `--modes consistent` |
| Table S-XXVI, ivis variants: paired differences | `runs_ivisvar/report/paired_holm.csv` | `scripts/analysis/ivis_variants_paired.py` (from `paired.csv` of `ivis_variants.py report`) | as Table S-XXV |
| Table S-XXVII, ivis variants: embedding geometry | `runs_ivisvar/report/geometry.csv` | `scripts/analysis/ivis_variants.py diag` | as Table S-XXV |
| Table S-XXVIII, LoRA and zero-shot foundation models | `table17_fm_adapt.csv` | `report/fm_adapt.py --adapt-runs <dir>` (RL columns from `FRAUDADV_RL_RESULTS`) | `hpo/train_context.py` with the families of `models/foundation_adapt.py`, attacked by the grid |
| Table S-XXIX, defenses of the graph and foundation models (C and K) | `tables/defenses_context.csv` | `report/consistent_tables.py: defenses_context` (C from `table7_defenses_by_dataset.csv`) | `defenses/run_defenses_consistent.py <d> --families <f> --defenses <x> --eps 0.25 --tag _<f>_<x> --both` (`runs/<d>/defenses/results_<f>_<x>.csv`, `runs_consistent/<d>/defenses/results_consistent_<f>_<x>.csv`) |
| Table S-XXX, ranking metrics, mean over datasets | `tables/robust_auc_mean.csv` | `report/tables.py: robust_auc_mean` (from `table9_robust_auc.csv`) | `attacks/robust_auc.py` |
| Table S-XXXI, ranking metrics, full | `table9_robust_auc.csv` | `make_report.robust_auc_table` | `attacks/robust_auc.py` |
| Table S-XXXII, explanations | `table5_explain_summary.csv` (column `share_controllable`), `table5_explain_agreement.csv` | `make_report.explain_tables`, `explain_summary` | `explain/attributions.py`, `explain/shap_values.py` |
| Table S-XXXIII, drivers per dataset | `tables/drivers.csv` | `report/tables.py: drivers` | `explain/attributions.py` (`global_importance.csv`, `attack_shift.csv`) |
| Table S-XXXIV, ablation variants, clean performance | `causal/variants_clean.csv` | `causal/report.py: variants_clean` | `causal/ablate.py` |
| Table S-XXXV, ablation effects per dataset and pooled | `causal/contrasts.csv` | `causal/analyze.py` | `causal/ablate.py` |
| Table S-XXXVI, factorial model | `causal/factorial_lpm.csv`; derived effects `causal/factorial_effects.csv` | `causal/analyze.py`; `causal/report.py: factorial_effects` | `causal/ablate.py` |
| Table S-XXXVII, evasion examples | `attack_views/attack_examples.csv`; score percentiles before and after `attack_views/attack_examples_scores.csv` | `report/attack_figures.py: tab_examples` | `report/attack_views.py compute` |
| Table S-XXXVIII, Bayesian mixed model | `table6_mixed_model.csv` (without PaySim: `table6_mixed_model_nopaysim.csv`) | `make_report.stats_tables` | grid `per_example*.npz` |
| Table S-XXXIX, Friedman tests | `table6_friedman.csv` | `make_report.stats_tables` | grid `per_example*.npz` |
| Table S-XL, first vs second half of the sample | `table11_halves.csv` (per model); mean per dataset `tables/halves_by_dataset.csv` | `make_report.halves_table`; `report/tables.py: halves_by_dataset` | grid |
| Table S-XLI, first grid vs final grid | `table14_rerun_by_attack.csv` (cells: `table14_rerun_by_attack_cells.csv`) | `make_report.rerun_attack_table` | first grid archived in `runs/<d>/archive_attacks_20261114/`, final grid |
| Table S-XLII, five-start HopSkipJump | `table15_hsj_starts.csv` | `make_report.hsj_starts_table` | `attacks/hsj_starts.py` |
| Table S-XLIII, clients with and without an identity | `table12_groups.csv` | `make_report.groups_table` | grid and history on IEEE-CIS |
| Figures S1 and S2, ASR against budget | `fig_robust_curves_{constrained,unconstrained}.{pdf,png}` | `make_report.fig_curves` | grid |
| Figure S3, transfer (unconstrained) | `fig_transfer_unconstrained.{pdf,png}` | `make_report.fig_transfer` | `attacks/transfer.py` |
| Figure S4, transfer (consistent) | `fig_transfer_consistent.{pdf,png}` | `report/consistent_tables.py: figure_transfer` | `attacks/transfer.py <d> --modes consistent` |
| Figure S5, thickness against ASR | `fig_thickness_vs_asr.{pdf,png}` | `make_report.thickness_analysis` | grid `thickness*.csv` |
| Figure S6, defense trade-off | `fig_defense_tradeoff.{pdf,png}` (data: `table7_defense_tradeoff.csv`) | `make_report.fig_defense_tradeoff` | `defenses/run_defenses.py`, `defenses/run_defenses_consistent.py --both` |
| Figure S7, clean-robust Pareto frontiers | `fig_pareto.{pdf,png}` (points: `table16_pareto.csv`; LoRA / zero-shot points: `pareto_extra_fm_adapt.csv`) | `make_report.pareto_figure` (`report/pareto.py`) | grid, defenses, LoRA / zero-shot extras |
| Figures S8, S9 and S11, SHAP summaries (XGBoost, CatBoost, MLP) | `fig_shap_xgboost`, `fig_shap_catboost`, `fig_shap_mlp` `.{pdf,png}` (three-dataset XGBoost version: `fig_shap_xgboost_three`) | `report/shap_figures.py` | `explain/shap_values.py` |
| Figure S10, SHAP shift under attack | `fig_shap_shift.{pdf,png}` (data: `shap/shap_shift_xgboost_constrained.csv`) | `report/shap_figures.py` | `explain/shap_values.py` |
| Figure S12, attribution heatmap | `fig_attr_heatmap.{pdf,png}` (data: `shap/attribution_share_top_features.csv`) | `report/shap_figures.py` | `explain/attributions.py` |
| Figure S13, ablation effects | `fig_causal_effects.{pdf,png}` | `causal/report.py: fig_effects` | `causal/ablate.py` |
| Figure S14, ablation dose-response | `fig_causal_dose.{pdf,png}` (data: `causal/dose_response.csv`) | `causal/report.py: fig_dose` | `causal/ablate.py` |
| Figure S15, decision-function slices | `fig_attack_slices.{pdf,png}` | `report/attack_views.py figures` | `report/attack_views.py compute` |
| Figure S16, distance to the threshold | `fig_attack_distance.{pdf,png}` (data: `attack_views/distance_to_boundary.csv`) | `report/attack_views.py figures` | `report/attack_views.py compute` |
| Figure S17, UMAP of adversarial examples | `fig_attack_map.{pdf,png}` (k-NN summary: `attack_views/knn_fraud_share.csv`) | `report/attack_views.py figures` | `report/attack_views.py compute` |
| Figure S18, perturbation profile | `fig_attack_profile.{pdf,png}` (data: `attack_views/perturbation_profile.csv`) | `report/attack_views.py figures` | `report/attack_views.py compute` |

The feature schema (who controls which feature, described in an appendix) is set per dataset in `data/prepare.py`
(written to `meta.json`: `mutable`, bounds, integer and categorical levels) and turned into the threat models by
`attacks/constraints.py`.

## What each run step is

| Step | Command | Writes |
|---|---|---|
| Data | `python -m fraudadv.data.prepare <d> [--context]` | `data/processed/<d>/` |
| Training | `python -m fraudadv.hpo.train_all <d>`; `python -m fraudadv.hpo.train_context <d> --families hgnn,hgt,chronos2,timesfm`; second tuning passes `hpo/topup.py`, `hpo/topup_context.py` | `runs/<d>/<family>/` |
| Attack grid | `python -m fraudadv.attacks.run_attacks <d> --families ... --full-bb --milp-budget 3600` (foundation models: `--fm-full-n 240`) | `runs/<d>/attacks/results*.csv`, `per_example*.npz`, `attack_scores*.npz`, `adv_*.npz`, `thickness*.csv`, `milp_status*.csv` |
| Transfer | `python -m fraudadv.attacks.transfer <d>` | `transfer_<mode>.npz` |
| Certification | `python -m fraudadv.attacks.certify <d> --mode constrained` (and `unconstrained`) | `certify_<family>_<mode>.csv` |
| Ranking metrics | `python -m fraudadv.attacks.robust_auc <d> <families>` | `robust_auc*.csv` |
| Attack strength | `python -m fraudadv.attacks.budget_check <d> <families>`; `python -m fraudadv.attacks.hsj_starts <d> <families>` | `budget_check*.csv`, `hsj_starts*.csv` |
| Boundary thickness (class pairs) | `python -m fraudadv.boundary.run_thickness <d> <families>` | `thickness_pairs.csv` |
| History injection | `python -m fraudadv.attacks.history <d> <families> --tag _pe`, then `history_pe.csv` copied to `history.csv` (as the study's run did; the history-group table reads `history_per_example_pe.npz`); foundation models `--tag _fm_<family>` | `history*.csv`, `history_per_example*.npz` |
| Defenses | tabular models: `python -m fraudadv.defenses.run_defenses <d> --defenses none,pgd_at,trades,adv_boost,monotone,squeeze`; added later at eps 0.25, each model trained once and attacked by the constrained and (IEEE-CIS, Sparkov, PaySim) the consistent attacker: `python -m fraudadv.defenses.run_defenses_consistent <d> --families ivis --defenses adv_aug --eps 0.25 --tag _ivis_adv_aug --both` and, for each of hgnn, hgt, chronos2, timesfm and each of none, pgd_at, trades, squeeze, `... --families <f> --defenses <x> --eps 0.25 --tag _<f>_<x> --both` | `runs/<d>/defenses/results*.csv`, `runs_consistent/<d>/defenses/results_consistent_*.csv` |
| Explanations | `python -m fraudadv.explain.attributions <d> <families>`; `python -m fraudadv.explain.shap_values <d>` | `runs/<d>/explain/` |
| Foundation models without the encoding | `python -m fraudadv.hpo.fm_ablation <d> chronos2,timesfm` | `runs/<d>/<family>/ablation_noemb.json` |
| Ablations | `python -m fraudadv.causal.ablate <d> --variants ...` | `runs/<d>/causal/` |
| Attack views | `python -m fraudadv.report.attack_views compute <d>` | `runs/<d>/causal/views.pkl` |
| Consistent attacker | `python -m fraudadv.attacks.run_attacks <d> --modes consistent --full-bb --milp-budget 3600`; `fraudadv.attacks.transfer <d> --modes consistent`; `fraudadv.attacks.robust_auc <d> <families> --modes consistent --eps 0.25`; `fraudadv.attacks.history_consistent <d> <families>`; `fraudadv.defenses.run_defenses_consistent <d>` (all in `scripts/reproduce.sh`, stage `consistent`) | `runs_consistent/<d>/` |
| ivis variants | `scripts/analysis/ivis_variants.py tune/train/attack/adaptive/diag/report` | `runs_ivisvar/` |
| LoRA / zero-shot FMs | `python -m fraudadv.hpo.train_context <d> --families chronos2_lora,timesfm_lora,chronos2_zs,timesfm_zs` (models in `models/foundation_adapt.py`), then attacked by the grid | `runs/<d>/<family>/` (kept apart in `study-artifacts/fm_adapt/`) |
