#!/usr/bin/env python3
"""IEEE-CIS without its data: export every output file with the source values replaced by references, and
restore them bit for bit after rebuilding the arrays from the official competition files.

IEEE-CIS is Kaggle competition data (Vesta), and its rules do not let us redistribute it. The release therefore
holds no IEEE-CIS feature row, amount, timestamp, entity id or label, and no TransactionID:
  ieee_cis/arrays_sha256.json   sha256 of every array of data/processed/ieee_cis/arrays.npz and of the
                                TransactionIDs in array order (row_ids.npz), to check a rebuild
  ieee_cis/runs/<path>.npz      every .npz under runs/ieee_cis (archives excluded), encoded:
                                - an array of feature rows that are rows of a split -> their indices;
                                - any other array as wide as the features (adversarial examples, explanations)
                                  -> per row, the nearest such row (anchor) and the changed entries as
                                  (row, column, float64 difference), exact for float32;
                                - an array or column equal to a source column (amt, y, t, ent) at the rows of
                                  the fraud sample, of a row array of the file, or of the file's first column
                                  -> a reference;
                                the encoding is described in the file's __spec__ entry (JSON).
Everything else under runs/ieee_cis (models, scores, CSV results, JSON) holds no source values and ships as is.
Indices point into arrays whose order and content only a rebuild gives, so they reveal nothing on their own.

  export   python scripts/ieee_cis_release.py export --out <release dir> [--with-other]
           (maintainers; reads data/processed/ieee_cis and runs/ieee_cis; checks that every file restores
           bit for bit and that no source value is left; --with-other also copies the non-npz files)
  rebuild  python scripts/ieee_cis_release.py rebuild --release <release dir>
           1. Accept the competition rules and download train_transaction.csv and train_identity.csv from
              https://www.kaggle.com/competitions/ieee-fraud-detection/data into $FRAUDADV_RAW/ieee/
              (default data/raw/ieee/). Unpack the release's other runs/ieee_cis files into runs/ieee_cis.
           2. This command runs the study's preparation (python -m fraudadv.data.prepare ieee_cis, then with
              --context for the graph and foundation models), checks the sha256 of every array and of the
              TransactionIDs, and writes every runs/ieee_cis/<path>.npz back as the study saved it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PROC = Path(os.environ.get("FRAUDADV_PROC", REPO / "data" / "processed")) / "ieee_cis"
RUNS = Path(os.environ.get("FRAUDADV_RUNS", REPO / "runs")) / "ieee_cis"
SPLITS = ("test", "val", "train")
SOURCE_COLS = ("amt", "y", "t", "ent")        # per-row source values besides the features
SKIP_DIRS = ("archive_", "backup_")             # stale copies; not part of the release


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def hashes():
    with np.load(PROC / "arrays.npz") as z:
        h = {k: sha(z[k]) for k in sorted(z.files)}
    with np.load(PROC / "row_ids.npz") as z:
        h.update({f"row_ids:{k}": sha(z[k]) for k in sorted(z.files)})
    return h


class Data:
    """Lazy view of the processed arrays, with row lookup."""

    def __init__(self):
        self.z, self.c, self.h = np.load(PROC / "arrays.npz"), {}, {}
        self.d = self["X_test"].shape[1]
        self.S = np.load(RUNS / "attacks" / "sample.npz")["S"].astype(np.int64)

    def __getitem__(self, k):
        if k not in self.c:
            self.c[k] = self.z[k]
        return self.c[k]

    def locate(self, R, hints):
        """(split, indices) with X_split[indices] == R exactly, or None."""
        for s in SPLITS:
            X = self[f"X_{s}"]
            for I in hints:
                if len(I) == len(R) and I.min() >= 0 and I.max() < len(X) and np.array_equal(X[I], R):
                    return s, I
        for s in SPLITS:
            if s not in self.h:
                self.h[s] = {r.tobytes(): i for i, r in enumerate(np.ascontiguousarray(self[f"X_{s}"]))}
            I = np.array([self.h[s].get(r.tobytes(), -1) for r in np.ascontiguousarray(R)], np.int64)
            if len(I) and I.min() >= 0:
                return s, I
        return None


def _nearest(V, B):
    """Per row of V, the row of B with fewest differing entries (its own position on ties)."""
    a = np.empty(len(V), np.int64)
    for i0 in range(0, len(V), 100):
        cnt = (V[i0:i0 + 100, None, :] != B[None, :, :]).sum(2)
        best = cnt.argmin(1)
        pos = np.arange(i0, i0 + len(cnt))
        own = (pos < len(B)) & (cnt[np.arange(len(cnt)), np.minimum(pos, len(B) - 1)] == cnt.min(1))
        a[i0:i0 + len(cnt)] = np.where(own, pos, best)
    return a


def _int_index(x, n):
    return x.size and np.all(np.isfinite(x)) and np.all(x == np.round(x)) and x.min() >= 0 and x.max() < n


def pack(z, D):
    arr = {k: z[k] for k in z.files}
    hints = [v.astype(np.int64) for v in arr.values() if v.ndim == 1 and v.dtype.kind in "iu" and v.size] + [D.S]
    wide = [k for k, v in arr.items() if v.ndim == 2 and v.shape[1] == D.d and v.dtype.kind == "f"]
    rows = {k: loc for k in wide if arr[k].dtype == np.float32 and (loc := D.locate(arr[k], hints))}
    spec, out = {}, {}
    for k in wide:
        if k in rows:
            spec[k] = dict(kind="rows", split=rows[k][0])
            out[f"{k}@idx"] = rows[k][1]
            continue
        if not rows:
            raise SystemExit(f"{k}: feature-wide array with no row array to anchor it to")
        V, best = arr[k], None
        for b, (s, I) in rows.items():
            B = D[f"X_{s}"][I]
            a = _nearest(V, B)
            nnz = int((V != B[a]).sum())
            if best is None or nnz < best[0]:
                best = (nnz, b, a, B[a])
        _, b, a, B = best
        r, c = np.nonzero(V != B)
        spec[k] = dict(kind="delta", base=b, dtype=str(V.dtype))
        out.update({f"{k}@anchor": a.astype(np.int32), f"{k}@r": r.astype(np.int32), f"{k}@c": c.astype(np.int16),
                    f"{k}@v": V[r, c].astype(np.float64) - B[r, c].astype(np.float64)})
    anchors = {f"rows:{k}": loc for k, loc in rows.items()}
    anchors["sample"] = ("test", D.S)
    for k, v in arr.items():
        if k in wide:
            continue
        if v.ndim not in (1, 2) or v.dtype.kind not in "fiub" or v.size == 0:
            out[k] = v
            continue
        local = dict(anchors)
        if v.ndim == 2 and v.shape[1] <= 16 and _int_index(v[:, 0], len(D["X_test"])):
            local["col0"] = ("test", v[:, 0].astype(np.int64))
        refs = []
        for j in ([None] if v.ndim == 1 else range(v.shape[1])):
            x = v if j is None else v[:, j]
            hit = next(((an, cn) for an, (s, I) in local.items() if len(I) == len(x) and not (an == "col0" and j == 0)
                        for cn in SOURCE_COLS if np.array_equal(D[f"{cn}_{s}"][I].astype(v.dtype), x)), None)
            if hit:
                refs.append([j, hit[1], hit[0]])
        if refs:
            v = v.copy()
            for j, _, _ in refs:
                if j is None:
                    v = np.zeros_like(v)
                else:
                    v[:, j] = 0
            spec[k] = dict(kind="cols", refs=refs)
        out[k] = v
    out["__spec__"] = np.array(json.dumps(spec))
    return out


def unpack(p, D):
    z = dict(np.load(p))
    spec = json.loads(str(z.pop("__spec__")))
    rows = {}
    for k, s in spec.items():
        if s["kind"] == "rows":
            I = z.pop(f"{k}@idx")
            rows[k] = (s["split"], I)
            z[k] = D[f"X_{s['split']}"][I]
    for k, s in spec.items():
        if s["kind"] == "delta":
            bs, bI = rows[s["base"]]
            B = D[f"X_{bs}"][bI][z.pop(f"{k}@anchor")].astype(np.float64)
            r, c, dv = z.pop(f"{k}@r"), z.pop(f"{k}@c"), z.pop(f"{k}@v")
            B[r, c] += dv
            z[k] = B.astype(s["dtype"])
    for k, s in spec.items():
        if s["kind"] == "cols":
            v = z[k].copy()
            for j, cn, an in s["refs"]:
                sp, I = (("test", D.S) if an == "sample" else ("test", v[:, 0].astype(np.int64)) if an == "col0"
                         else rows[an[len("rows:"):]])
                x = D[f"{cn}_{sp}"][I].astype(v.dtype)
                if j is None:
                    v = x
                else:
                    v[:, j] = x
            z[k] = v
    return z


def release_files():
    for p in sorted(RUNS.rglob("*")):
        rel = p.relative_to(RUNS)
        if p.is_file() and not any(part.startswith(SKIP_DIRS) for part in rel.parts):
            yield p, rel


def leftover(packed, D):
    """Feature-wide float arrays or exact source columns of the fraud sample that survived packing."""
    bad = [k for k, v in packed.items() if v.ndim == 2 and v.shape[1] == D.d and v.dtype.kind == "f"]
    amt = D["amt_test"][D.S]
    for k, v in packed.items():
        if v.dtype.kind == "f" and v.ndim in (1, 2) and v.shape[0] == len(D.S):
            cols = [v] if v.ndim == 1 else list(v.T)
            bad += [k for x in cols if np.array_equal(x.astype(np.float32), amt)]
    return bad


def _same(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and (
        np.array_equal(a, b, equal_nan=True) if a.dtype.kind in "fc" else np.array_equal(a, b))


def export(out, with_other=False):
    out = Path(out) / "ieee_cis"
    if not (PROC / "row_ids.npz").exists():
        raise SystemExit("row_ids.npz missing: rerun python -m fraudadv.data.prepare ieee_cis (it now writes the ids)")
    out.mkdir(parents=True, exist_ok=True)
    (out / "arrays_sha256.json").write_text(json.dumps(hashes(), indent=1))
    D, n, other, enc = Data(), 0, 0, 0
    for p, rel in release_files():
        q = out / "runs" / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix != ".npz":
            if with_other:
                shutil.copy2(p, q)
                other += 1
            continue
        with np.load(p) as z:
            packed = pack(z, D)
            bad = leftover(packed, D)
            if bad:
                raise SystemExit(f"{rel}: source values left in {bad}")
            np.savez_compressed(q, **packed)
            back = unpack(q, D)
            if set(back) != set(z.files) or not all(_same(back[k], z[k]) for k in z.files):
                raise SystemExit(f"{rel}: does not restore bit for bit")
            n += 1
            enc += json.loads(str(packed["__spec__"])) != {}
    print(f"exported hashes and {n} .npz files ({enc} with source values replaced)"
          + (f" and {other} other files" if with_other else "") + f" to {out}")


def rebuild(release):
    rel = Path(release) / "ieee_cis"
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    for extra in ([], ["--context"]):
        subprocess.run([sys.executable, "-m", "fraudadv.data.prepare", "ieee_cis", *extra], check=True, env=env)
    want, got = json.loads((rel / "arrays_sha256.json").read_text()), hashes()
    bad = [k for k, h in want.items() if got.get(k) != h]
    if bad:
        raise SystemExit(f"rebuilt IEEE-CIS differs from the study's in {bad}: check the source files and versions")
    src = rel / "runs"
    (RUNS / "attacks").mkdir(parents=True, exist_ok=True)
    z = dict(np.load(src / "attacks" / "sample.npz"))   # the fraud sample (indices only) decodes the others
    if json.loads(str(z.pop("__spec__"))):
        raise SystemExit("sample.npz unexpectedly holds source values")
    np.savez_compressed(RUNS / "attacks" / "sample.npz", **z)
    D = Data()
    for q in sorted(src.rglob("*.npz")):
        dst = RUNS / q.relative_to(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, **unpack(q, D))
    print("IEEE-CIS rebuilt and verified; output files restored under", RUNS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", required=True)
    e.add_argument("--with-other", action="store_true")
    sub.add_parser("rebuild").add_argument("--release", required=True)
    a = ap.parse_args()
    export(a.out, a.with_other) if a.cmd == "export" else rebuild(a.release)
