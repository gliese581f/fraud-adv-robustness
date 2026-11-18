#!/bin/bash
# Rebuild every result table (CSV) and figure from saved results.
#
#   bash scripts/make_results.sh
#
# Needs runs/, runs_consistent/ and runs_ivisvar/ (its report/ summaries included) in this folder (the release
# bundle, or links to them), and data/processed/ for the dataset metadata. Writes reports/: the summary CSVs and
# figures of make_report, the result tables of reports/tables/, reports/causal/, reports/attack_views/ and
# reports/analysis/consistent/. Optional: FM_ADAPT_RUNS points at the LoRA / zero-shot foundation-model runs, for
# Table S-XXVIII and their points in the Pareto figure (with FRAUDADV_RL_RESULTS for the RL attackers' columns).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src TF_USE_LEGACY_KERAS=1 TF_CPP_MIN_LOG_LEVEL=3
export FRAUDADV_PROC=${FRAUDADV_PROC:-$PWD/data/processed} FRAUDADV_RUNS=${FRAUDADV_RUNS:-$PWD/runs}
step() { echo "== $* ($(date -u +%FT%TZ))"; }
mkdir -p reports/tables

if [ -n "${FM_ADAPT_RUNS:-}" ]; then
  step "LoRA and zero-shot foundation models: Table S-XXVIII and their Pareto points"
  python3 -m fraudadv.report.fm_adapt --adapt-runs "$FM_ADAPT_RUNS"
fi
step "summary CSVs and figures (reports/)"
python3 -m fraudadv.report.make_report
step "controlled ablations: Table VI, Tables S-XXXIV to S-XXXVI, Figures S13-S14 (reports/causal/)"
python3 -m fraudadv.causal.analyze
step "attack views: Figure 2, Figures S15-S18, Table S-XXXVII (reports/attack_views/)"
python3 -m fraudadv.report.attack_views figures
step "SHAP and attribution figures: Figures S8-S12"
python3 -m fraudadv.report.shap_figures
step "result tables (reports/tables/)"
python3 -m fraudadv.report.tables
step "ivis variants: Holm-adjusted paired comparisons, Table S-XXVI"
python3 scripts/analysis/ivis_variants_paired.py
step "consistent attacker: Table III, Tables S-XIII to S-XX, Table S-XXIX, Figure S4"
python3 -m fraudadv.report.realizable_share
python3 -m fraudadv.report.consistent_tables
step "done"
