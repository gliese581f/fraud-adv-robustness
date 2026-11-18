#!/usr/bin/env python3
"""The archived release bundle (DOI): processed arrays, trained models, attack outputs and results, one archive per
dataset and part, each under its source's terms, with Croissant metadata.

  plan   python scripts/make_bundle.py plan [--out DIR]
         lists what goes in each archive with sizes and writes DIR/bundle_plan.json (default reports/bundle/)
  pack   python scripts/make_bundle.py pack --out DIR [--datasets ulb,baf,...]
         writes DIR/<dataset>_data.tar, DIR/<dataset>_runs.tar, croissant.json and SHA256SUMS; IEEE-CIS ships no
         data archive and its runs archive holds scripts/ieee_cis_release.py's export instead of its .npz files

Archives unpack into the repository root (data/processed/<dataset>/..., runs/<dataset>/...). Left out: stale
copies (archive_*, backup_*, *_v1 files), checkpoints and partial files, markers and logs. Nothing here uploads:
the bundle goes public only after the release clearance (DATA.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROC = Path(os.environ.get("FRAUDADV_PROC", REPO / "data" / "processed"))
RUNS = Path(os.environ.get("FRAUDADV_RUNS", REPO / "runs"))
DATASETS = ["ulb", "ieee_cis", "sparkov", "tabformer", "paysim", "baf"]
TERMS = {   # what each archive is shared under; Kaggle-page terms still to confirm (DATA.md)
    "ulb": "terms of the ULB credit-card fraud data (commonly ODbL 1.0 / DbCL 1.0; to confirm)",
    "ieee_cis": "no source values (IEEE-CIS competition data is not redistributed); outputs CC BY 4.0",
    "sparkov": "terms of the Sparkov simulated data (commonly CC0; to confirm)",
    "tabformer": "Apache-2.0 (IBM TabFormer)",
    "paysim": "terms of the PaySim data (commonly CC BY-SA 4.0; to confirm)",
    "baf": "terms of the Bank Account Fraud data (commonly CC BY-NC-SA 4.0, non-commercial; to confirm)",
}
SKIP_PARTS = ("archive_", "backup_", "history_v1", "hpo1")
SKIP_NAMES = ("ckpt", "partial", ".done", ".merged", ".delegated", ".archived", "delegated_topup", "reattack.txt")


def keep(rel: Path) -> bool:
    if any(p.startswith(SKIP_PARTS) for p in rel.parts):
        return False
    n = rel.name
    return not (n.startswith(SKIP_NAMES) or n.endswith(("_v1.npz", "_v1.csv", "_before_repair.csv", ".log", ".tmp", ".new"))
                or n in SKIP_NAMES)


def members(d: str):
    """(archive name, [(source path, path inside the archive)]) per archive of dataset d."""
    out = {}
    if d != "ieee_cis":
        src = PROC / d
        out[f"{d}_data.tar"] = [(p, Path("data/processed") / d / p.relative_to(src)) for p in sorted(src.rglob("*"))
                                if p.is_file() and keep(p.relative_to(src)) and not p.name.endswith(".part00")]
    src = RUNS / d
    files = [(p, Path("runs") / d / p.relative_to(src)) for p in sorted(src.rglob("*")) if p.is_file() and keep(p.relative_to(src))]
    if d == "ieee_cis":
        files = [(p, a) for p, a in files if p.suffix != ".npz"]   # replaced by the export (no source values)
    out[f"{d}_runs.tar"] = files
    return out


def plan(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    P = {}
    for d in DATASETS:
        for name, files in members(d).items():
            P[name] = dict(terms=TERMS[d], files=len(files), bytes=sum(p.stat().st_size for p, _ in files))
    P["ieee_cis_runs.tar"]["note"] = "plus scripts/ieee_cis_release.py export (re-encoded .npz files, about 8 MB)"
    (out / "bundle_plan.json").write_text(json.dumps(P, indent=1))
    tot = sum(v["bytes"] for v in P.values())
    for k, v in P.items():
        print(f"{k:22s} {v['files']:6d} files {v['bytes'] / 1e6:9.1f} MB  {v['terms']}")
    print(f"{'total':22s} {sum(v['files'] for v in P.values()):6d} files {tot / 1e9:9.2f} GB  -> {out / 'bundle_plan.json'}")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def croissant(out: Path, archives: dict):
    """Croissant 1.0 metadata: the dataset and one FileObject per archive (contentUrl filled in at upload)."""
    meta = {
        "@context": {"@language": "en", "@vocab": "https://schema.org/", "sc": "https://schema.org/",
                     "cr": "http://mlcommons.org/croissant/", "dct": "http://purl.org/dc/terms/"},
        "@type": "sc:Dataset", "dct:conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": "fraudadv-benchmark",
        "description": "Processed arrays, trained models (nine families), fraud samples, adversarial examples and results "
                       "of the fraud-detection adversarial robustness benchmark on six public datasets. IEEE-CIS ships "
                       "without source values; scripts/ieee_cis_release.py rebuilds it from the official files.",
        "license": sorted(set(TERMS.values())), "url": "TBD (DOI on release)", "version": "1.0.0",
        "creator": {"@type": "sc:Person", "name": "Sidney Ellington"},
        "distribution": [{"@type": "cr:FileObject", "@id": n, "name": n, "contentUrl": f"TBD/{n}",
                          "encodingFormat": "application/x-tar", "sha256": h, "description": TERMS[n.split("_")[0] if not n.startswith("ieee") else "ieee_cis"]}
                         for n, h in archives.items()],
    }
    (out / "croissant.json").write_text(json.dumps(meta, indent=1))


def pack(out: Path, datasets):
    out.mkdir(parents=True, exist_ok=True)
    sums = {}
    for d in datasets:
        extra = []
        if d == "ieee_cis":
            exp = out / "_ieee_export"
            subprocess.run([sys.executable, str(REPO / "scripts" / "ieee_cis_release.py"), "export", "--out", str(exp)], check=True)
            base = exp / "ieee_cis"
            extra = [(p, Path("release") / "ieee_cis" / p.relative_to(base)) for p in sorted(base.rglob("*")) if p.is_file()]
        for name, files in members(d).items():
            t = out / name
            with tarfile.open(t, "w") as tf:
                for p, a in files + (extra if name.endswith("_runs.tar") else []):
                    tf.add(p, arcname=str(a))
            sums[name] = sha256(t)
            print(f"{name}: {t.stat().st_size / 1e6:.1f} MB", flush=True)
    (out / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sorted(sums.items())))
    croissant(out, sums)
    print("bundle in", out, "(not uploaded)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan").add_argument("--out", default=str(REPO / "reports" / "bundle"))
    pk = sub.add_parser("pack")
    pk.add_argument("--out", required=True)
    pk.add_argument("--datasets", default=",".join(DATASETS))
    a = ap.parse_args()
    plan(Path(a.out)) if a.cmd == "plan" else pack(Path(a.out), a.datasets.split(","))
