# Main study code: adversarial robustness of fraud detection models

This folder holds the analysis code of *Adversarial Robustness of Fraud Detection Models: A Survey and Benchmark from
Gradient-Boosted Trees to Graph Transformers and Time-Series Foundation Models* (S. Ellington): the runs, and the
code that turns their outputs into every result table (as CSV) and figure. It holds no data, trained models or
results.

**Start with `RESULTS.md`.** It maps every table, figure and number of the study and its supplement to the CSV or
figure that holds it, the code that makes it and the run that made its inputs.

## What is where

```
RESULTS.md              table/figure/number -> output file -> code -> inputs
notebooks/              Jupyter notebooks that run the analysis interactively and show each result
src/fraudadv/           the library
  paths.py              where everything is read and written (data/, runs/, runs_consistent/, reports/, logs/)
  data/                 raw files -> features (out-of-time splits, past-only history features), loading
  models/               model families: zoo.py (XGBoost, CatBoost, MLP, FT-Transformer, ivis), graph.py (HGNN,
                        HGT), foundation.py (Chronos-2, TimesFM), foundation_adapt.py (LoRA, zero-shot),
                        ivis_variants.py
  hpo/                  hyperparameter search and training (train_all: tabular; train_context: graph and
                        foundation models; topup*: second tuning passes)
  attacks/              the three threat models (constraints.py: unconstrained, constrained, consistent; the
                        consistent repair in consistent.py and the realizability check in realizable.py);
                        APGD, Square, HopSkipJump (blackwhite.py); AATS for trees (trees.py) and the exact MILP
                        (milp.py); the attack grid (run_attacks.py); transfer, certification, ranking metrics
                        (robust_auc.py), attack-strength checks (budget_check, hsj_starts), history injection
                        (history.py, history_consistent.py)
  defenses/             adversarial training, TRADES, adversarial boosting, monotone constraints, squeezing and ivis's
                        adversarial augmentation, each attacked adaptively (run_defenses.py; run_defenses_consistent.py
                        for the consistent attacker and, with --both, the graph and foundation-model defenses)
  boundary/             decision-boundary thickness
  explain/              SHAP and Integrated Gradients attributions
  causal/               controlled architecture ablations and their analysis
  stats/                paired tests, mixed models, Friedman tests
  eval/                 metrics (AUC, KS, thresholds, dollar weighting)
  report/               summary CSVs and figures (make_report), the result tables that need more computation
                        (tables), the three-attacker tables (consistent_tables, realizable_share), the LoRA /
                        zero-shot table (fm_adapt) and the other figures (attack_views, attack_figures,
                        shap_figures, pareto)
  byom.py               attack your own model with the study's protocol
scripts/
  reproduce.sh          the runs for one dataset
  make_results.sh       every result table (CSV) and figure from saved results
  check_claims.py       prints the numbers behind the results-section statements (RQ1, RQ2)
  check_shared_repair.py  checks the consistent attacker against the copy shared with the project's other studies
  analysis/             ivis variants (retrained ivis, neighbour vote), their paired tests and probes
  ieee_cis_release.py, make_bundle.py
                        release: IEEE-CIS export without source values, DOI bundle
vendor/decoydef/        the shared copy of the consistent repair and realizability check (pinned by hash)
tests/                  unit tests (pytest)
environment/            pinned package versions
```

## Install

Python 3.13 reproduces the published numbers with the exact environment:

```bash
python3.13 -m venv .venv && . .venv/bin/activate
pip install -r environment/requirements-lock.txt   # if annoy fails: pip install --use-pep517 annoy==1.17.3
pip install -e . --no-deps
pip install jupyterlab                              # for the notebooks
pytest -q
```

## Run

- **Notebooks** (`jupyter lab`, then open `notebooks/`): each one produces a group of the study's results from the
  saved results, shows them inline and checks the files it writes against a previous build of `reports/` when
  `FRAUDADV_REFERENCE` points at a folder holding one. `00_build_all_results.ipynb` makes everything in one go;
  `01` to `11` follow the results section by section. Each also has a small live computation on a few frauds, and
  a cell (off by default) that reruns that step's runs from scratch. They need the saved results (`runs/`,
  `runs_consistent/`, `runs_ivisvar/` and `data/processed/` in this folder: the release bundle, `DATA.md`).

- Data: put the raw files under `data/raw/` (sources in `DATA.md`).
- Every run for one dataset: `bash scripts/reproduce.sh <dataset> [--models released|train] [--stages ...]`.
  The stages are `data`, `models`, `attacks` (unconstrained and constrained attackers), `consistent` (the
  consistent attacker) and `extras` (explanations, ablations, attack views). The ivis variants and the LoRA /
  zero-shot foundation models are separate (`RESULTS.md`, "What each run step is").
- Every result table and figure from saved results: `bash scripts/make_results.sh`. It needs `runs/`,
  `runs_consistent/` and `runs_ivisvar/` in this folder (copies or links) and `data/processed/` (or
  `FRAUDADV_PROC`), and writes CSVs and figures to `reports/`: the summary files of `make_report`, and
  `reports/tables/`, `reports/causal/`, `reports/attack_views/`, `reports/shap/` and
  `reports/analysis/consistent/`; the ivis-variant tables go to `runs_ivisvar/report/`. Set `FM_ADAPT_RUNS` to the
  LoRA / zero-shot runs to add Table S-XXVIII and their points in the Pareto figure.

Paths default to `data/raw`, `data/processed`, `runs`, `runs_consistent`, `reports` and `logs` in this folder;
`FRAUDADV_RAW`, `FRAUDADV_PROC`, `FRAUDADV_RUNS`, `FRAUDADV_RUNS_CONSISTENT`, `FRAUDADV_REPORTS` and `FRAUDADV_LOGS`
override them for the runs; `FRAUDADV_RL_RESULTS` names the RL-attacker results that Table S-XXVIII sets beside the
LoRA / zero-shot models. Every other option is a command-line flag.

## Versions

This is version 2 of the study's code: the same computations as the code that produced the results, rewritten to
be plain and easy to follow, with the consistent attacker built into the library and every option a command-line
flag. The result tables are written as CSV files.
