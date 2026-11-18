"""Second hyperparameter search for the history-aware families (HGNN, HGT, Chronos-2, TimesFM 2.5).

The first search (fraudadv.hpo.train_context) gave each of these families a fixed CPU time of 5 to 20
minutes, which bought 1 to 9 trials, against 10 to 23 for the tabular families after their second search
(fraudadv.hpo.topup). This search closes that gap with the same rule as the tabular one:

  * the family's existing Optuna study is resumed (same search space, sampler and Hyperband pruner). A
    family whose first search left no study.db gets a new study whose first trial is its best configuration.
  * it runs to MIN_TRIALS finished trials (pruned trials count) or TIME_CAP more seconds
  * if the best trial changed, the old seed models are archived under runs/<dataset>/archive_hpo1/<family>
    and the new configuration is retrained with the usual three seeds; otherwise the models are kept

The search and retrain are train_context.run itself, called with a larger budget and a before_retrain hook.
A search can also run on another machine and be imported here (import_delegated).

Inputs:  runs/<dataset>/<family>/ from train_context (study.db, best_params.json, models).
Outputs: the same folder, updated; hpo_v2.json when the family is done (hpo_v2_wip.json while it runs, so an
         interrupted run resumes); archived first-search files under runs/<dataset>/archive_hpo1/<family>/.

Results: the final graph and foundation models of every result; Table V, Section VI-F,
Table S-II and Table S-IV.

usage: python -m fraudadv.hpo.topup_context <dataset> [--families hgnn,hgt,chronos2,timesfm]
Prints "RETRAINED <dataset> <families>" when any family got new models (including families retrained by an
earlier, interrupted run of this script, and those listed in runs/<dataset>/reattack.txt), so the pipeline
attacks them again.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import optuna

from fraudadv.hpo import train_context
from fraudadv.hpo.train_all import RUNS

MIN_TRIALS = {"hgnn": 10, "hgt": 10, "chronos2": 10, "timesfm": 10}
TIME_CAP = {"hgnn": 3000, "hgt": 4200, "chronos2": 2400, "timesfm": 2400}     # extra search seconds
# the searched hyperparameters (Optuna parameter names), to replay a best configuration as a trial
RAW_KEYS = {"hgnn": ["hidden", "dropout", "lr", "weight_decay", "pos_weight", "K1"],
            "chronos2": ["hidden", "proj", "dropout", "lr", "weight_decay", "pos_weight"]}
RAW_KEYS["hgt"], RAW_KEYS["timesfm"] = RAW_KEYS["hgnn"], RAW_KEYS["chronos2"]
SEED_FILES = ("metrics.json", "metrics_seed*.json", "seed*.pkl", "test_scores_seed*.npy")   # one family's models


def open_study(out, dataset, fam):
    """(study, fresh): the first search's study, or (fresh=True) a new one whose first queued trial is the
    first search's best configuration."""
    db = out / "study.db"
    kw = dict(study_name=f"{dataset}-{fam}", storage=f"sqlite:///{db}")
    if db.exists():
        return optuna.load_study(**kw), False
    st = optuna.create_study(direction="maximize", load_if_exists=True, **kw,
                             sampler=optuna.samplers.TPESampler(multivariate=True, group=True, seed=0, n_startup_trials=4),
                             pruner=optuna.pruners.HyperbandPruner(min_resource=1, max_resource=40, reduction_factor=3))
    best = json.loads((out / "best_params.json").read_text())["params"]
    st.enqueue_trial({k: best[k] for k in RAW_KEYS[fam] if k in best})
    return st, True


def finished(st):
    """Finished trials of study `st`: complete or pruned, failed trials excluded."""
    return sum(t.state.is_finished() and t.state != optuna.trial.TrialState.FAIL for t in st.trials)


def archive(out, dataset, fam, move):
    """Copy (move=False) or move the family's models and best_params.json to runs/<dataset>/archive_hpo1/<family>."""
    dst = RUNS / dataset / "archive_hpo1" / fam
    dst.mkdir(parents=True, exist_ok=True)
    for pat in SEED_FILES + ("best_params.json",):
        for f in out.glob(pat):
            (shutil.move if move else shutil.copy2)(str(f), str(dst / f.name))


def import_delegated(dataset, families, poll=60):
    """Import a second search that ran on another machine (the graph_hpo_<dataset> shards of version 1).

    runs/<dataset>/delegated_topup.json = {"source": <shared folder>/runs/<dataset>, "deadline": unix time}
    says that the families' tuned models will appear under source once source/../../DONE.json exists.
    This waits for them, archives the local first-pass models, and copies the tuned ones in (hpo_v2.json
    last, so an interrupted copy is repeated). After the deadline the marker is dropped and the pass runs
    here as usual. Without the marker this does nothing."""
    mk = RUNS / dataset / "delegated_topup.json"
    if not mk.exists():
        return
    d = json.loads(mk.read_text())
    src = Path(d["source"])
    todo = [f for f in families if not (RUNS / dataset / f / "hpo_v2.json").exists()]
    while todo:
        if (src.parent.parent / "DONE.json").exists() and all((src / f / "hpo_v2.json").exists() for f in todo):
            for fam in todo:
                out = RUNS / dataset / fam
                if not (RUNS / dataset / "archive_hpo1" / fam).exists():
                    archive(out, dataset, fam, move=False)
                names = sorted(p.name for p in (src / fam).iterdir() if p.is_file() and p.name != "hpo_v2.json")
                for n in names + ["hpo_v2.json"]:
                    shutil.copyfile(src / fam / n, out / n)
                (out / "hpo_v2_wip.json").unlink(missing_ok=True)
                print(dataset, fam, "second pass imported from", src, flush=True)
            return
        if time.time() > d["deadline"]:
            print(dataset, "delegated second pass did not arrive by the deadline; tuning here", flush=True)
            mk.rename(mk.with_suffix(".expired"))
            return
        time.sleep(poll)


def run(dataset, families):
    """Run the second search for each family of `families` that has no hpo_v2.json yet (module docstring)."""
    import_delegated(dataset, families)
    retrained = []
    for fam in families:
        out = RUNS / dataset / fam
        v2, wip = out / "hpo_v2.json", out / "hpo_v2_wip.json"
        if v2.exists():
            continue
        if wip.exists():                      # an interrupted run of this script: resume it
            w = json.loads(wip.read_text())
            fresh, old_best, n1, spent = w["fresh"], w["old_best"], w["n1"], w["spent"]
        else:
            if not (out / "best_params.json").exists():     # not trained by the first search
                continue
            st, fresh = open_study(out, dataset, fam)
            n1 = json.loads((out / "best_params.json").read_text()).get("n_trials", finished(st))
            if not fresh and finished(st) >= MIN_TRIALS[fam]:
                v2.write_text(json.dumps({"kept_first_pass": True, "n_trials": finished(st)}))
                print(dataset, fam, "kept: first pass finished", finished(st), "trials", flush=True)
                continue
            old_best = None if fresh else st.best_trial.number
            spent = sum((t.duration.total_seconds() if t.duration else 0.0) for t in st.trials)
            # train_context.run resumes the study when best_params.json is absent
            archive(out, dataset, fam, move=False)
            # models trained before per-seed metrics files existed: write them, so an unchanged best
            # configuration keeps the attacked models instead of retraining them
            if (out / "metrics.json").exists():
                for m in json.loads((out / "metrics.json").read_text()):
                    ms = out / f"metrics_seed{m['seed']}.json"
                    if not ms.exists() and (out / f"seed{m['seed']}.pkl").exists():
                        ms.write_text(json.dumps(m))
            wip.write_text(json.dumps({"fresh": fresh, "old_best": old_best, "n1": n1, "spent": spent}))
            (out / "best_params.json").unlink()
            # metrics.json stays until the retrain rewrites it, so the pipeline still sees a trained family
            # if this run is interrupted (train_context.run ignores it when called with before_retrain).
        # train_context.run counts the study's earlier trials and seconds against its budget, so these totals
        # give MIN_TRIALS finished trials and TIME_CAP more seconds
        train_context.MAX_TRIALS[fam] = MIN_TRIALS[fam] + (1 if fresh else 0)  # a new study re-runs the old best first
        train_context.BUDGET_S[fam] = spent + TIME_CAP[fam]
        state = {}

        def before_retrain(out_, study, fam=fam, old_best=old_best, fresh=fresh):
            # called by train_context.run between the search and the seed retrains
            if study is None:                 # resumed after the search finished
                study = optuna.load_study(study_name=f"{dataset}-{fam}", storage=f"sqlite:///{out_ / 'study.db'}")
            new = study.best_trial.number
            # new study: trial 0 is the first search's configuration evaluated again
            changed = new != (0 if fresh else old_best)
            state["changed"], state["n"] = changed, finished(study)
            if changed:
                # drop the first search's seed models only: seeds already retrained with the new configuration
                # (written after its best_params.json, by an interrupted run of this script or by the
                # first-search pipeline lane finishing the retrain) are kept
                bp = out_ / "best_params.json"
                cut = bp.stat().st_mtime if bp.exists() else float("inf")
                for pat in SEED_FILES[1:]:            # metrics.json is rewritten after the seed loop
                    for f in out_.glob(pat):
                        if f.stat().st_mtime < cut:
                            f.unlink()
            print(dataset, fam, f"v2 search: {finished(study)} trials, best val AUPRC {study.best_value:.4f};",
                  "retraining" if changed else "best configuration unchanged, models kept", flush=True)

        t0 = time.time()
        train_context.run(dataset, [fam], before_retrain=before_retrain)
        if "changed" not in state:            # the models were completed elsewhere: decide from the study
            st = optuna.load_study(study_name=f"{dataset}-{fam}", storage=f"sqlite:///{out / 'study.db'}")
            state["changed"] = st.best_trial.number != (0 if fresh else old_best)
            state["n"] = finished(st)
            print(dataset, fam, f"v2 search: {state['n']} trials (models completed by another run);",
                  "retrained" if state["changed"] else "best configuration unchanged", flush=True)
        v2.write_text(json.dumps({"kept_first_pass": not state.get("changed", False), "n_trials_first_pass": n1,
                                  "n_trials": state.get("n"), "seconds": round(time.time() - t0)}))
        wip.unlink(missing_ok=True)
        if state.get("changed"):
            retrained.append(fam)
    # report retrains from an earlier, interrupted run too
    for fam in families:
        v2 = RUNS / dataset / fam / "hpo_v2.json"
        if fam not in retrained and v2.exists() and not json.loads(v2.read_text()).get("kept_first_pass", True):
            retrained.append(fam)
    # families whose attacks must be redone although their models are kept (runs/<dataset>/reattack.txt,
    # written by hand when a code fix changes how a kept model is attacked, such as a gradient fix); the
    # pipeline re-attacks every family listed on this line
    ra = RUNS / dataset / "reattack.txt"
    if ra.exists():
        for fam in ra.read_text().strip().split(","):
            if fam in families and fam not in retrained:
                retrained.append(fam)
                print(dataset, fam, "queued for re-attack (reattack.txt)", flush=True)
    if retrained:
        print("RETRAINED", dataset, ",".join(retrained), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--families", default="hgnn,hgt")
    args = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fams = args.families.split(",")
    if any(f in ("chronos2", "timesfm") for f in fams):
        import chronos  # noqa: F401  must be imported before TensorFlow (LLVM option clash)
    run(args.dataset, fams)
