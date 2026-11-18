"""Turn each raw dataset into the study's common, attack-ready representation.

Run: `python -m fraudadv.data.prepare <dataset> [<dataset> ...] [--context]` (no dataset: all six).

Input: the raw files under data/raw/ (FRAUDADV_RAW overrides; fraudadv.data.loaders lists them).
Output, in data/processed/<dataset>/ (FRAUDADV_PROC overrides):
  arrays.npz    per split (train, val, test):
                  X_<split>    features, float32, standardized with training means and standard deviations
                  y_<split>    label, 1 = fraud
                  w_<split>    sampling weight (1 for frauds, 1 / sampling rate for legitimate rows)
                  amt_<split>  dollar amount, original units
                  t_<split>    time
                  ent_<split>  entity id (card, client or account), -1 when the dataset has none
  meta.json     feature names, the features the attacker controls ("mutable"), integer features, the valid
                values of each categorical feature ("cat_values"), training bounds ("lower", "upper"), the
                scaler ("mu", "sd"), the name of the derived-feature rule ("recompute", applied by
                fraudadv.attacks.constraints), notes, and split sizes
  row_ids.npz   IEEE-CIS only: the source TransactionID of every row, so a release can ship ids instead of rows
  ctx.npz       with --context only (instead of the three files above): the full standardized matrix of all
                rows, times, log-amounts and the indices of each row's most recent earlier transactions, for
                the graph and foundation models (models/graph.py, models/foundation.py)

Used by every result of the paper. The dataset table (Table II) reads meta.json and arrays.npz; the feature
schema (the feature-schema appendix: who controls which feature) is the "mutable" list of meta.json, set per dataset below.

The same rules apply to every dataset:
  * out-of-time split 70/15/15 of the full, time-sorted data (BAF: months 0-5 / 6 / 7)
  * history aggregates use strictly earlier transactions of the same entity
  * categorical features become one numeric column each, by smoothed target encoding fit on the training
    split; they keep a finite set of valid values, so constrained attacks can snap to a real category
  * missing values: training median, plus a missing indicator for each column with more than 5% missing
  * legitimate rows are subsampled uniformly per split (every fraud is kept), and w = 1 / sampling rate, so
    weighted metrics estimate their population values
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fraudadv.paths import PROC as OUT, RAW  # noqa: E402  raw files, one folder per dataset; arrays for load()
CAPS = {"train": 300_000, "val": 100_000, "test": 150_000}   # max legit rows kept per split


@dataclass
class Spec:
    """One dataset, after its own feature engineering, ready for the common steps of `finalize`."""
    name: str
    df: pd.DataFrame                    # all rows; feature columns plus label, time, amount and entity
    numeric: list[str]
    categorical: list[str]
    mutable: list[str]                  # attacker-controllable features
    recompute: str | None = None        # name of the derived-feature rule in fraudadv.attacks.constraints (DERIVED)
    split_col: str | None = None        # column holding precomputed split labels (BAF)
    rel2: str | None = None             # second relation key for graph models (e.g. merchant)
    notes: dict = field(default_factory=dict)


# ---------------------------------------------------------------- history features
def history_features(df: pd.DataFrame, ent: str, t: str, amt: str, prefix: str = "") -> pd.DataFrame:
    """Per-entity aggregates of earlier transactions only. df must be sorted by [ent, t].

    "Earlier" means earlier in that sort order, so a transaction never sees itself or a later one. Columns
    (each with `prefix` in front):
      cnt_prev            log(1 + number of earlier transactions of the entity)
      log_amt_mean_prev   mean log(1 + |amount|) of those transactions (NaN for the first one)
      log_ratio           log(1 + |amount|) of this transaction minus that mean; it is linear in the current
                          log-amount, so attacks recompute it after changing the amount (recompute rule "log_ratio")
      log_dt_prev         log(1 + seconds since the entity's previous transaction)
      cnt_24h             log(1 + number of earlier transactions of the entity in the past 24 hours)
      logamt_sum_24h      sum of their log(1 + |amount|)
    """
    entity = df[ent].to_numpy()
    time = df[t].to_numpy().astype(np.float64)
    log_amt = np.log1p(np.abs(df[amt].to_numpy().astype(np.float64)))
    first_of_entity = np.r_[True, entity[1:] != entity[:-1]]
    group = np.cumsum(first_of_entity) - 1
    group_start = np.flatnonzero(first_of_entity)[group]       # row of each entity's first transaction
    row = np.arange(len(df))
    n_prev = row - group_start
    cum = np.cumsum(log_amt)
    cum_before = cum - log_amt                                 # sum over all earlier rows of the file
    cum_before_group = np.where(group_start > 0, cum[group_start - 1], 0.0)
    mean_prev = np.where(n_prev > 0, (cum_before - cum_before_group) / np.maximum(n_prev, 1), np.nan)
    dt_prev = np.where(n_prev > 0, time - np.r_[np.nan, time[:-1]], np.nan)
    # 24-hour window, earlier rows only. One sorted key for all entities: the entity number times a span longer
    # than the whole time range, plus the time, so a window never reaches into the previous entity.
    span = float(time.max() - time.min()) + 2 * 86400.0
    key = group.astype(np.float64) * span + (time - time.min())   # < 2**53 for all datasets used
    window_start = np.searchsorted(key, key - 86400.0, side="left")
    window_start = np.maximum(window_start, group_start)
    n_24h = row - window_start
    sum_24h = cum_before - np.where(window_start > 0, cum[window_start - 1], 0.0)
    out = pd.DataFrame(index=df.index)
    out[prefix + "cnt_prev"] = np.log1p(n_prev)
    out[prefix + "log_amt_mean_prev"] = mean_prev
    out[prefix + "log_ratio"] = log_amt - mean_prev            # derived: linear in log_amt
    out[prefix + "log_dt_prev"] = np.log1p(np.maximum(dt_prev, 0))
    out[prefix + "cnt_24h"] = np.log1p(n_24h)
    out[prefix + "logamt_sum_24h"] = sum_24h
    return out


def sort_by(df, cols):
    """Stable sort by `cols` (rows with equal keys keep their file order), with a fresh 0..n-1 index."""
    return df.sort_values(cols, kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------- datasets
# One builder per dataset: read the raw file(s), engineer the features, and say which ones the attacker controls.
def ulb() -> Spec:
    """ULB credit cards: 28 PCA components plus amount and hour; no card id, so no history features."""
    df = pd.read_csv(RAW / "ulb/creditcard.csv")
    df["log_amt"] = np.log1p(df["Amount"])
    df["hour"] = (df["Time"] // 3600) % 24
    df = df.rename(columns={"Class": "label", "Time": "time", "Amount": "amount"})
    df["entity"] = -1
    num = [f"V{i}" for i in range(1, 29)] + ["log_amt", "hour"]
    return Spec("ulb", df, num, [], mutable=["log_amt", "hour"],
                notes={"constrained": "V1-V28 are PCA of undisclosed raw features, so only amount and time are attacker-controllable"})


def ieee() -> Spec:
    """IEEE-CIS: transactions joined with two identity columns; the client id is rebuilt from card1, addr1 and
    the account start day (day - D1); at most 30 weakly correlated V columns are kept."""
    tr = pd.read_csv(RAW / "ieee/train_transaction.csv")
    idt = pd.read_csv(RAW / "ieee/train_identity.csv", usecols=["TransactionID", "DeviceType", "id_31"])
    df = tr.merge(idt, on="TransactionID", how="left")
    del tr
    df["day"] = df["TransactionDT"] // 86400
    df["uid"] = (df["card1"].astype(str) + "_" + df["addr1"].astype(str) + "_" + (df["day"] - df["D1"].fillna(0)).astype(str))
    df["uid"] = df["uid"].astype("category").cat.codes
    df = sort_by(df, ["uid", "TransactionDT"])
    hist = history_features(df, "uid", "TransactionDT", "TransactionAmt")
    df = pd.concat([df, hist], axis=1)
    df["log_amt"] = np.log1p(df["TransactionAmt"])
    df["hour"] = (df["TransactionDT"] // 3600) % 24
    df["browser"] = df["id_31"].astype(str).str.split(" ").str[0]
    # V columns: keep those with under 15% missing; in order of decreasing spread, keep a column only if its
    # absolute correlation with every kept one is under 0.75, up to 30 columns
    vcols = [c for c in df.columns if c.startswith("V") and df[c].isna().mean() < 0.15]
    keep: list[str] = []
    sample = df[vcols].sample(100_000, random_state=0).fillna(-1)
    corr = sample.corr().abs()
    for c in sorted(vcols, key=lambda c: -sample[c].std()):
        if all(corr.loc[c, k] < 0.75 for k in keep):
            keep.append(c)
        if len(keep) == 30:
            break
    num = (["log_amt", "hour", "card1", "card2", "card3", "card5", "addr1", "addr2", "dist1"]
           + [f"C{i}" for i in range(1, 15)] + ["D1", "D2", "D3", "D4", "D10", "D15"] + keep + list(hist.columns))
    cat = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "browser"] + [f"M{i}" for i in range(1, 10)]
    df = df.rename(columns={"isFraud": "label", "TransactionDT": "time", "TransactionAmt": "amount", "uid": "entity"})
    mut = ["log_amt", "hour", "addr1", "addr2", "dist1", "ProductCD", "P_emaildomain", "R_emaildomain", "DeviceType", "browser", "log_ratio"]
    return Spec("ieee_cis", df, num, cat, mut, recompute="log_ratio", rel2="card1",
                notes={"uid": "card1+addr1+(day-D1) client id, a standard IEEE-CIS reconstruction"})


def sparkov() -> Spec:
    """Sparkov (simulated card transactions): per-card history, time of day and week, customer age and the
    customer-merchant distance."""
    df = pd.read_csv(RAW / "sparkov/credit_card_fraud_transactions.csv")
    df = sort_by(df, ["cc_num", "unix_time"])
    hist = history_features(df, "cc_num", "unix_time", "amt")
    df = pd.concat([df, hist], axis=1)
    ts = pd.to_datetime(df["unix_time"], unit="s")
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["age"] = (ts - pd.to_datetime(df["dob"], format="%m/%d/%y")).dt.days / 365.25
    df.loc[df["age"] < 0, "age"] += 100          # two-digit birth years read as 20xx belong to 19xx
    df["log_amt"] = np.log1p(df["amt"])
    df["log_city_pop"] = np.log1p(df["city_pop"])
    df["gender_m"] = (df["gender"] == "M").astype(float)
    df["dist"] = np.hypot(df["lat"] - df["merch_lat"], (df["long"] - df["merch_long"]) * np.cos(np.radians(df["lat"])))
    df["entity"] = df["cc_num"].astype("category").cat.codes
    df = df.rename(columns={"is_fraud": "label", "unix_time": "time", "amt": "amount"})
    num = ["log_amt", "hour", "dow", "age", "gender_m", "log_city_pop", "lat", "long", "merch_lat", "merch_long", "dist"] + list(hist.columns)
    mut = ["log_amt", "hour", "dow", "merch_lat", "merch_long", "dist", "category", "log_ratio"]
    return Spec("sparkov", df, num, ["category", "state"], mut, recompute="log_ratio", rel2="merchant",
                notes={"rows": "HF mirror holds the first 1,048,575 rows of fraudTrain.csv (spreadsheet row cap)"})


def tabformer(years=(2015, 2019)) -> Spec:
    """IBM TabFormer: the years 2015-2019 of all users. History features are computed on each card's full
    record before the years are cut, so a 2015 transaction still sees the card's earlier transactions."""
    cols = ["User", "Card", "Year", "Month", "Day", "Time", "Amount", "Use Chip", "Merchant Name",
            "Merchant City", "MCC", "Errors?", "Is Fraud?"]
    df = pd.read_csv(RAW / "tabformer/card_transaction.v1.csv", usecols=cols,
                     dtype={"User": "int32", "Card": "int8", "Year": "int16", "Month": "int8", "Day": "int8", "MCC": "int32"})
    df["amount"] = df["Amount"].str.replace("$", "", regex=False).astype("float32")
    df.drop(columns=["Amount"], inplace=True)
    hm = df["Time"].str.split(":", expand=True).astype("int16")
    df["time"] = (pd.to_datetime(dict(year=df.Year, month=df.Month, day=df.Day)).astype("int64") // 10**9
                  + hm[0].astype("int64") * 3600 + hm[1].astype("int64") * 60)
    df["hour"] = hm[0]
    del hm
    df["entity"] = df["User"].astype("int64") * 10 + df["Card"]     # one entity per card of a user
    df = sort_by(df, ["entity", "time"])
    hist = history_features(df, "entity", "time", "amount")
    df = pd.concat([df, hist], axis=1)
    # merchant seen before by this card (past only)
    df["merch_seen"] = df.duplicated(["entity", "Merchant Name"]).astype("float32")
    df = df[(df["Year"] >= years[0]) & (df["Year"] <= years[1])].reset_index(drop=True)
    df["dow"] = pd.to_datetime(df["time"], unit="s").dt.dayofweek
    df["log_amt"] = np.log1p(df["amount"].abs())
    df["is_refund"] = (df["amount"] < 0).astype("float32")
    df["online"] = (df["Merchant City"] == "ONLINE").astype("float32")
    df["label"] = (df["Is Fraud?"] == "Yes").astype("int8")
    df["use_chip"] = df["Use Chip"].astype(str)
    df["errors"] = df["Errors?"].fillna("none").astype(str)
    df["mcc"] = df["MCC"].astype(str)
    df["amount"] = df["amount"].abs()
    num = ["log_amt", "is_refund", "online", "hour", "dow", "merch_seen"] + list(hist.columns)
    cat = ["use_chip", "mcc", "errors"]
    df["merch"] = df["Merchant Name"]
    df = df[num + cat + ["label", "time", "amount", "entity", "merch"]]
    mut = ["log_amt", "online", "hour", "dow", "merch_seen", "use_chip", "mcc", "log_ratio"]
    return Spec("tabformer", df, num, cat, mut, recompute="log_ratio", rel2="merch",
                notes={"subset": f"years {years[0]}-{years[1]}, all 2,000 users; history from full record"})


def paysim() -> Spec:
    """PaySim (simulated mobile money): TRANSFER and CASH_OUT only; history per destination account, log
    balances and the balance-error terms."""
    df = pd.read_csv(RAW / "paysim/paysim.csv")
    df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].reset_index(drop=True)
    df["t"] = df["step"] * 3600
    df["dest"] = df["nameDest"].astype("category").cat.codes
    df = sort_by(df, ["dest", "t"])
    hist = history_features(df, "dest", "t", "amount", prefix="dest_")
    df = pd.concat([df, hist], axis=1)
    df["log_amt"] = np.log1p(df["amount"])
    for c in ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
        df["log_" + c] = np.log1p(df[c])
    # signed log of how far each account's balance change is from the amount moved
    df["err_orig"] = np.sign(e := df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]) * np.log1p(np.abs(e))
    df["err_dest"] = np.sign(e := df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]) * np.log1p(np.abs(e))
    df["is_transfer"] = (df["type"] == "TRANSFER").astype(float)
    df["hour"] = df["step"] % 24
    df = df.rename(columns={"isFraud": "label", "t": "time", "dest": "entity"})
    num = ["log_amt", "is_transfer", "hour", "log_oldbalanceOrg", "log_newbalanceOrig", "log_oldbalanceDest",
           "log_newbalanceDest", "err_orig", "err_dest"] + list(hist.columns)
    mut = ["log_amt", "is_transfer", "hour", "log_oldbalanceDest", "log_newbalanceDest", "err_orig", "err_dest",
           "log_newbalanceOrig", "dest_cnt_prev", "dest_log_amt_mean_prev", "dest_log_ratio", "dest_log_dt_prev",
           "dest_cnt_24h", "dest_logamt_sum_24h"]
    return Spec("paysim", df, num, [], mut, recompute="paysim",
                notes={"types": "TRANSFER and CASH_OUT only (the only types with fraud)",
                       "constrained": "attacker picks amount, type, timing and the mule (destination) account; origin balance and error terms are recomputed"})


def baf() -> Spec:
    """Bank Account Fraud (account applications, Base variant): the month column gives the split; the
    proposed credit limit stands in for the dollar amount; no entity."""
    df = pd.read_parquet(RAW / "baf/baf.parquet")
    df = df.drop(columns=["dataset", "user_id", "seen_user"])
    df["time"] = df["ts"]
    df["amount"] = df["proposed_credit_limit"]
    df["entity"] = -1
    cat = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
    num = [c for c in df.columns if c not in cat + ["split", "ts", "label", "time", "amount", "entity"]]
    split = df["ts"].map(lambda m: "train" if m <= 5 else ("val" if m == 6 else "test"))
    df["split"] = split
    mut = ["income", "name_email_similarity", "prev_address_months_count", "current_address_months_count",
           "customer_age", "days_since_request", "intended_balcon_amount", "payment_type", "employment_status",
           "housing_status", "email_is_free", "phone_home_valid", "phone_mobile_valid", "has_other_cards",
           "proposed_credit_limit", "foreign_request", "source", "session_length_in_minutes", "device_os",
           "keep_alive_session"]
    return Spec("baf", df, num, cat, mut, split_col="split",
                notes={"variant": "Base (1M applications), months 0-5 train / 6 val / 7 test",
                       "amount": "proposed_credit_limit used as dollar exposure"})


BUILDERS = {"ulb": ulb, "ieee_cis": ieee, "sparkov": sparkov, "tabformer": tabformer, "paysim": paysim, "baf": baf}


# ---------------------------------------------------------------- common pipeline
def prev_neighbors(key: np.ndarray, t: np.ndarray, K: int) -> np.ndarray:
    """Row indices of the K most recent strictly-earlier rows sharing `key` (-1 = none)."""
    order = np.lexsort((t, key))
    ks, ts = key[order], t[order]
    nb = np.full((len(key), K), -1, dtype=np.int32)
    for j in range(1, K + 1):
        ok = np.zeros(len(order), bool)
        ok[j:] = (ks[j:] == ks[:-j]) & (ts[:-j] < ts[j:])
        nb[order[ok], j - 1] = order[np.flatnonzero(ok) - j]
    # compact: same-timestamp ties leave holes; keep the valid slots first, most recent first
    o = np.argsort(nb == -1, axis=1, kind="stable")
    return np.take_along_axis(nb, o, axis=1)


def split_labels(spec: Spec) -> np.ndarray:
    """The split of every row ("train", "val" or "test"): the dataset's own split column if it has one (BAF),
    else the first 70% of rows in time order, the next 15% and the last 15%."""
    df = spec.df
    if spec.split_col:
        return df[spec.split_col].to_numpy()
    order = np.argsort(df["time"].to_numpy(), kind="mergesort")
    q = np.empty(len(df), dtype=object)
    n = len(df)
    q[order[: int(0.70 * n)]] = "train"
    q[order[int(0.70 * n): int(0.85 * n)]] = "val"
    q[order[int(0.85 * n):]] = "test"
    return q


def encode_features(spec: Spec, y: np.ndarray, tr: np.ndarray):
    """Numeric feature matrix of all rows, standardized; every statistic is fit on the training rows `tr`.

    Steps: target-encode each categorical column, smoothed towards the training fraud rate with weight 20
    (unseen values get that rate); add a missing indicator isna_<c> for each numeric column with more than 5%
    missing in training; fill missing numeric values with the training median (0 when the median is
    missing too); standardize every column by its training mean and standard deviation.
    Returns (Z, feats, mu, sd, cat_values): Z the float32 frame with columns `feats` (numeric, then
    categorical, then the indicators), mu and sd the scaler, cat_values the sorted standardized values each
    categorical column takes in training (rounded to 6 decimals)."""
    df = spec.df
    feats = spec.numeric + spec.categorical
    F = pd.DataFrame(index=df.index)
    prior = y[tr].mean()
    cat_values = {}
    for c in spec.categorical:
        s = df[c].astype(str).fillna("nan")
        stats = pd.DataFrame({"c": s[tr], "y": y[tr]}).groupby("c")["y"].agg(["sum", "count"])
        enc = (stats["sum"] + 20 * prior) / (stats["count"] + 20)
        F[c] = s.map(enc).fillna(prior).astype(np.float64)
    for c in spec.numeric:
        F[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float64)
    miss = [c for c in spec.numeric if F.loc[tr, c].isna().mean() > 0.05]
    med = F.loc[tr, spec.numeric].median()
    for c in miss:
        F["isna_" + c] = F[c].isna().astype(np.float64)
    F[spec.numeric] = F[spec.numeric].fillna(med).fillna(0.0)
    feats = feats + ["isna_" + c for c in miss]
    mu = F.loc[tr, feats].mean()
    sd = F.loc[tr, feats].std().replace(0, 1.0).clip(lower=1e-6)
    Z = ((F[feats] - mu) / sd).astype(np.float32)
    for c in spec.categorical:
        cat_values[c] = sorted(set(np.round(Z.loc[tr, c].unique(), 6).tolist()))
    return Z, feats, mu, sd, cat_values


def subsample_splits(df: pd.DataFrame, Z: pd.DataFrame, y: np.ndarray, split: np.ndarray, rng):
    """The rows kept in each split, and their arrays.

    Every fraud is kept. Legitimate rows are kept at random, at most CAPS[split] of them (drawn without
    replacement from `rng`, in the order train, val, test); each kept legitimate row gets the weight
    1 / sampling rate. Returns (arrays, keeps): arrays as described in the module docstring, keeps the sorted
    kept row numbers per split."""
    arrays, keeps = {}, {}
    for part in ["train", "val", "test"]:
        idx = np.flatnonzero(split == part)
        legit = idx[y[idx] == 0]
        fraud = idx[y[idx] == 1]
        cap = CAPS[part]
        rate = min(1.0, cap / max(len(legit), 1))
        if rate < 1.0:
            legit = rng.choice(legit, size=cap, replace=False)
        keep = np.sort(np.r_[legit, fraud])
        keeps[part] = keep
        w = np.where(y[keep] == 1, 1.0, 1.0 / rate)
        arrays[f"X_{part}"] = Z.to_numpy()[keep]
        arrays[f"y_{part}"] = y[keep]
        arrays[f"w_{part}"] = w.astype(np.float32)
        arrays[f"amt_{part}"] = df["amount"].to_numpy()[keep].astype(np.float32)
        arrays[f"t_{part}"] = df["time"].to_numpy()[keep].astype(np.float64)
        arrays[f"ent_{part}"] = df["entity"].to_numpy()[keep].astype(np.int64)
    return arrays, keeps


def save_context(spec: Spec, Z: pd.DataFrame, keeps: dict, out) -> None:
    """Write ctx.npz: every row's standardized features ("Z"), time and log(1 + |amount|), the 32 most recent
    earlier rows of the same entity ("nbr1") and the 8 most recent earlier rows sharing the second relation key
    ("nbr2", all -1 when the dataset has none), and the kept row numbers of each split ("keep_<split>")."""
    df = spec.df
    t = df["time"].to_numpy().astype(np.float64)
    ent = df["entity"].to_numpy()
    ctx = {"Z": Z.to_numpy(), "time": t, "log_amt": np.log1p(np.abs(df["amount"].to_numpy().astype(np.float64))).astype(np.float32),
           "nbr1": prev_neighbors(ent, t, 32)}
    if spec.rel2:
        k2 = pd.factorize(df[spec.rel2].astype(str))[0]
        ctx["nbr2"] = prev_neighbors(k2, t, 8)
    else:
        ctx["nbr2"] = np.full((len(df), 8), -1, np.int32)
    for part in ["train", "val", "test"]:
        ctx[f"keep_{part}"] = keeps[part].astype(np.int64)
    np.savez(out / "ctx.npz", **ctx)
    print(spec.name, "context saved", ctx["Z"].shape, flush=True)


def finalize(spec: Spec, seed: int = 0, context: bool = False) -> None:
    """Apply the common steps to one dataset and write its files to data/processed/<name>/: split, encode,
    subsample, then either arrays.npz and meta.json (and row_ids.npz for IEEE-CIS) or, with context=True,
    only ctx.npz. `seed` drives the subsampling of legitimate rows."""
    rng = np.random.default_rng(seed)
    df = spec.df
    split = split_labels(spec)
    y = df["label"].to_numpy().astype(np.int8)
    tr = split == "train"
    Z, feats, mu, sd, cat_values = encode_features(spec, y, tr)
    arrays, keeps = subsample_splits(df, Z, y, split, rng)
    Xtr = arrays["X_train"]
    meta = {
        "name": spec.name, "features": feats, "categorical": spec.categorical,
        "mutable": [f for f in feats if f in spec.mutable], "cat_values": cat_values,
        "lower": Xtr.min(0).tolist(), "upper": Xtr.max(0).tolist(),
        "integer": [f for f in feats if f in ("hour", "dow") or f.startswith("isna_")],
        "mu": mu.tolist(), "sd": sd.tolist(), "recompute": spec.recompute, "notes": spec.notes,
        "sizes": {p: {"rows": int(len(arrays[f"y_{p}"])), "fraud": int(arrays[f"y_{p}"].sum()),
                      "pop_rows": int((split == p).sum()), "pop_fraud": int(y[split == p].sum())} for p in ["train", "val", "test"]},
    }
    out = OUT / spec.name
    out.mkdir(parents=True, exist_ok=True)
    if context:
        save_context(spec, Z, keeps, out)
        return
    np.savez_compressed(out / "arrays.npz", **arrays)
    if "TransactionID" in df.columns:        # IEEE-CIS: source row id of every array row, so a release can ship ids,
        np.savez(out / "row_ids.npz", **{f"TransactionID_{p}": df["TransactionID"].to_numpy()[keeps[p]]   # not rows
                                          for p in keeps})
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(spec.name, json.dumps(meta["sizes"]), len(feats), "features,", len(meta["mutable"]), "mutable")


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for name in args or list(BUILDERS):
        finalize(BUILDERS[name](), context="--context" in sys.argv)
