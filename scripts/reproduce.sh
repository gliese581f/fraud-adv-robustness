#!/bin/bash
# Reproduce the study's results for one dataset, end to end, with the options of the paper's runs:
#
#   bash scripts/reproduce.sh <dataset> [--models released|train] [--stages data,models,attacks,consistent,extras]
#
#   dataset   ulb | ieee_cis | sparkov | tabformer | paysim | baf
#   --models  released (default): use the trained models of the release bundle, unpacked under runs/<dataset>/
#             (see DATA.md); train: tune and train every family from scratch (days of CPU on the large datasets)
#   --stages  any subset, run in this order (default: all of them)
#               data        raw files -> data/processed/<dataset>
#               models      tune and train, or check the released models are there
#               attacks     unconstrained and constrained attackers: grid, transfer, certification, ranking metrics,
#                           attack-strength checks, history injection, defenses        -> runs/<dataset>/
#               consistent  the consistent attacker (IEEE-CIS, Sparkov, PaySim only; elsewhere it equals the
#                           constrained one): grid, transfer, ranking metrics, history injection, defenses
#                                                                                       -> runs_consistent/<dataset>/
#               extras      explanations, foundation models without the encoding, controlled ablations,
#                           attack views                                                -> runs/<dataset>/
# Then `bash scripts/make_results.sh` turns the saved results into the result tables (CSV) and figures.
#
# Every step skips work whose outputs exist and resumes from checkpoints, so the command can be rerun after an
# interruption. The paper's runs split these same steps across machines (version 1 of this folder,
# scripts/shards/run.sh, records exactly how); the results do not depend on the split.
set -euo pipefail
cd "$(dirname "$0")/.."
d=${1:?usage: bash scripts/reproduce.sh <dataset> [--models released|train] [--stages ...]}
shift
MODELS=released
STAGES=data,models,attacks,consistent,extras
while [ $# -gt 0 ]; do
  case $1 in
    --models) MODELS=$2; shift 2 ;;
    --stages) STAGES=$2; shift 2 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done
export PYTHONPATH=$PWD/src TF_USE_LEGACY_KERAS=1 TF_CPP_MIN_LOG_LEVEL=3 HF_HUB_DISABLE_XET=1
export FRAUDADV_PROC=${FRAUDADV_PROC:-$PWD/data/processed} FRAUDADV_RUNS=${FRAUDADV_RUNS:-$PWD/runs}
has() { [[ ",$STAGES," == *",$1,"* ]]; }
step() { echo "== $d: $* ($(date -u +%FT%TZ))"; }
py() { python3 -m "$@"; }

# ---- what runs on this dataset
TAB=xgboost,catboost,mlp,ft_transformer,ivis                  # tabular families (every dataset)
case $d in
  ulb|baf) CTX=""; FMS="" ;;                                  # no card/account sequences: no graph or foundation models
  sparkov|tabformer|ieee_cis|paysim) CTX=hgnn,hgt; FMS=chronos2,timesfm ;;
  *) echo "unknown dataset $d"; exit 2 ;;
esac
ALL=$TAB${CTX:+,$CTX}
HISTORY=0; case $d in sparkov|tabformer|ieee_cis) HISTORY=1 ;; esac      # history injection needs card histories
CONSISTENT=0; case $d in ieee_cis|sparkov|paysim) CONSISTENT=1 ;; esac   # elsewhere no constrained query is impossible
# the paper's grid options: black-box attacks on the whole fraud sample (--full-bb); one hour of exact tree attack
# per model and threat model, enough to reach every detected fraud of the largest ensemble (IEEE-CIS XGBoost);
# foundation models on 240 random frauds (all of Sparkov's), as their queries are slow
GRID="--full-bb --milp-budget 3600"
FM_N=240; [ $d = sparkov ] && FM_N=100000
TAB_DEFENSES=none,pgd_at,trades,adv_boost,monotone,squeeze    # the first defense runs (ivis's augmentation came later)
CAUSAL_VARIANTS=xgb,xgb_d2,xgb_d6,xgb_d10,xgb_mono,xgb_from_mlp,mlp,mlp_l1,mlp_l4,mlp_relu,mlp_sn,mlp_wd,mlp_from_xgb

if has data; then
  step "prepare features from data/raw/$d"
  [ -f "$FRAUDADV_PROC/$d/arrays.npz" ] || py fraudadv.data.prepare $d ${CTX:+--context}
fi

if has models; then
  if [ "$MODELS" = train ]; then
    step "tune and train tabular families"
    py fraudadv.hpo.train_all $d --families $TAB
    if [ -n "$CTX" ]; then
      step "tune and train graph and foundation models"
      py fraudadv.hpo.train_context $d --families $CTX,$FMS
    fi
  else
    for f in ${ALL//,/ } ${FMS//,/ }; do
      [ -f "$FRAUDADV_RUNS/$d/$f/seed0.pkl" ] || [ -d "$FRAUDADV_RUNS/$d/$f/seed0" ] || {
        echo "missing $FRAUDADV_RUNS/$d/$f: unpack the release bundle's runs/$d (DATA.md) or use --models train"; exit 1; }
    done
  fi
fi

if has attacks; then
  step "attack grid (white-box, Square, HopSkipJump, worst case), transfer examples and boundary thickness"
  py fraudadv.attacks.run_attacks $d --families $ALL --modes unconstrained,constrained $GRID --tag _full_all
  [ -n "$FMS" ] && py fraudadv.attacks.run_attacks $d --families $FMS --modes unconstrained,constrained $GRID \
    --fm-full-n $FM_N --tag _full_fm
  step "transfer matrices"
  py fraudadv.attacks.transfer $d
  step "boundary thickness between fraud and legitimate pairs"
  py fraudadv.boundary.run_thickness $d $ALL${FMS:+,$FMS}
  step "certification of the tree attack (exact MILP), both threat models"
  for mode in constrained unconstrained; do py fraudadv.attacks.certify $d --mode $mode; done
  step "ranking metrics under a score-minimising attacker"
  py fraudadv.attacks.robust_auc $d $ALL --tag _all
  [ -n "$FMS" ] && py fraudadv.attacks.robust_auc $d $FMS --tag _fm
  step "attack-strength check and five-start HopSkipJump"
  py fraudadv.attacks.budget_check $d $ALL${FMS:+,$FMS}
  py fraudadv.attacks.hsj_starts $d $ALL
  if [ $HISTORY = 1 ]; then
    step "history injection"
    # the paper's run wrote history_pe.csv and history_per_example_pe.npz (the per-fraud outcomes that the
    # history-group table reads) and copied history_pe.csv to history.csv, which the other tables read
    py fraudadv.attacks.history $d $ALL --tag _pe
    cp $FRAUDADV_RUNS/$d/attacks/history_pe.csv $FRAUDADV_RUNS/$d/attacks/history.csv
    for f in ${FMS//,/ }; do py fraudadv.attacks.history $d $f --tag _fm_$f; done
  fi
  step "defenses of the tabular models, attacked adaptively"
  py fraudadv.defenses.run_defenses $d --defenses $TAB_DEFENSES
  step "defenses added later, eps 0.25: ivis's adversarial augmentation and the graph and foundation models; each model
       trained once and attacked by the study's attacker and, on IEEE-CIS, Sparkov and PaySim, the consistent one"
  py fraudadv.defenses.run_defenses_consistent $d --families ivis --defenses adv_aug --eps 0.25 --tag _ivis_adv_aug --both
  for f in ${CTX//,/ } ${FMS//,/ }; do
    for x in none pgd_at trades squeeze; do
      py fraudadv.defenses.run_defenses_consistent $d --families $f --defenses $x --eps 0.25 --tag _${f}_$x --both
    done
  done
fi

if has consistent && [ $CONSISTENT = 1 ]; then
  step "consistent attacker: attack grid"
  py fraudadv.attacks.run_attacks $d --families $ALL --modes consistent $GRID --tag _consistent_all
  [ -n "$FMS" ] && py fraudadv.attacks.run_attacks $d --families $FMS --modes consistent $GRID \
    --fm-full-n $FM_N --tag _consistent_fm
  step "consistent attacker: transfer matrix"
  py fraudadv.attacks.transfer $d --modes consistent
  step "consistent attacker: ranking metrics at eps 0.25"
  py fraudadv.attacks.robust_auc $d $ALL --modes consistent --eps 0.25 --tag _consistent_all
  [ -n "$FMS" ] && py fraudadv.attacks.robust_auc $d $FMS --modes consistent --eps 0.25 --tag _consistent_fm
  if [ $HISTORY = 1 ]; then
    step "consistent attacker: history injection with repaired warm-up records"
    py fraudadv.attacks.history_consistent $d $ALL
    for f in ${FMS//,/ }; do py fraudadv.attacks.history_consistent $d $f --tag _fm_$f; done
  fi
  step "consistent attacker: defenses of the tabular models at eps 0.25"
  py fraudadv.defenses.run_defenses_consistent $d --defenses $TAB_DEFENSES
fi

if has extras; then
  step "explanations: SHAP and Integrated Gradients"
  py fraudadv.explain.shap_values $d
  py fraudadv.explain.attributions $d $ALL${FMS:+,$FMS}
  if [ -n "$FMS" ]; then
    step "foundation models: the same head trained without the encoding"
    py fraudadv.hpo.fm_ablation $d $FMS
  fi
  case $d in
    sparkov|ieee_cis|tabformer|baf)
      step "controlled architecture ablations, seeds 0-2"
      py fraudadv.causal.ablate $d --variants $CAUSAL_VARIANTS ;;
  esac
  case $d in
    sparkov|tabformer|ieee_cis|baf)
      step "attack views (perturbation profiles, distances, neighbourhoods)"
      py fraudadv.report.attack_views compute $d ;;
  esac
fi
step "done"
