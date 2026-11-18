# Data: sources, licenses, preparation

The study uses six public fraud datasets. This file records where each raw file actually came from, what each
source says about its license, how the files become the study's arrays, and what the release ships. The license
column is what the source states. Rows marked *to confirm* could not be read from our machines (Kaggle pages
need a login) and must be checked in the release clearance.

## Sources as used

| Dataset | File(s) read by `fraudadv.data.prepare` | Where our copy came from | Original source | License stated |
|---|---|---|---|---|
| ULB | `ulb/creditcard.csv` (284,807 rows) | TensorFlow's public copy, `https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv` | Machine Learning Group, ULB, via Kaggle `mlg-ulb/creditcardfraud` [Dal Pozzolo et al. 2015] | Kaggle page: *to confirm* (commonly listed as ODbL 1.0 / DbCL 1.0) |
| IEEE-CIS | `ieee/train_transaction.csv`, `ieee/train_identity.csv` | Hugging Face mirror `aliceczr/ieee-fraud-detection` (card states no license, no source) | Vesta, Kaggle competition `ieee-fraud-detection` | Competition rules: *to confirm*; competition data generally may not be redistributed, so the release ships no IEEE-CIS rows (below) |
| Sparkov | `sparkov/credit_card_fraud_transactions.csv` (first 1,048,575 rows of the training file) | Hugging Face mirror `santosh3110/credit_card_fraud_transactions` (card: `apache-2.0`, no source named) | Sparkov simulator (B. Harris), data on Kaggle `kartik2112/fraud-detection` | Kaggle page: *to confirm* (commonly listed as CC0) |
| TabFormer | `tabformer/card_transaction.v1.csv` | IBM's GitHub repository, git-lfs file `data/credit_card/transactions.tgz` | IBM, `github.com/IBM/TabFormer` [Altman et al. 2021; Padhi et al. 2021] | Repository: Apache-2.0; the README sets no separate data terms |
| PaySim | `paysim/paysim.csv` | Hugging Face mirror `theman10/paysim` (card: `mit`, no source named) | E. Lopez-Rojas, Kaggle `ealaxi/paysim1` [Lopez-Rojas et al. 2016] | Kaggle page: *to confirm* (commonly listed as CC BY-SA 4.0) |
| BAF | `baf/baf.parquet` (Base variant) | Hugging Face mirror `mkbadeniyi/fraud-detection-benchmark` (card: `other`; lists Xente, IEEE-CIS, PaySim, MoMTSim, BAF without links) | Feedzai, `github.com/feedzai/bank-account-fraud`, data on Kaggle `sgpjesus/bank-account-fraud-dataset-neurips-2022` [Jesus et al. 2022] | Kaggle page: *to confirm* (commonly listed as CC BY-NC-SA 4.0, non-commercial) |

Mirrors were read on 2026-11-08; licenses on the Hugging Face cards and GitHub were read on 2026-11-14. The
mirrors are third-party copies whose cards do not cite the original, so the release points users to the original
sources, and `fraudadv.data.prepare` reads the original files under the same names. Before release, check that
each original file gives the same arrays (the sha256 files below make this a one-line check), and check every
non-commercial term against the author's affiliation.

## From raw files to arrays

`python -m fraudadv.data.prepare <dataset> [--context]` (raw files under `$FRAUDADV_RAW/<folder>/`, default
`data/raw`; output under `$FRAUDADV_PROC/<dataset>/`, default `data/processed`) applies one recipe to every
dataset:

- out-of-time split 70/15/15 (BAF by month: 0–5, 6, 7);
- history features from strictly earlier transactions of the same entity;
- target encoding of categoricals (smoothed, fit on training data);
- median imputation with missing indicators;
- standardization with training statistics;
- legitimate rows subsampled per split with inverse-rate weights (all frauds kept).

`--context` also writes the graph and sequence context for the graph and foundation models. The preparation is
deterministic: rerunning it on the same files gives arrays identical to the bit. For IEEE-CIS this was verified
on 2026-11-14 by rebuilding from the raw files and comparing the sha256 of every array.

## What the release ships

- **ULB, Sparkov, TabFormer, PaySim, BAF:** processed arrays (`arrays.npz`, `meta.json`, `ctx.npz`), trained
  models, fraud samples, adversarial examples and all results, under each source's license terms.
- **IEEE-CIS:** no source values: no feature row, amount, timestamp, entity id, label or TransactionID.
  `scripts/ieee_cis_release.py export` writes the sha256 of every array and of the TransactionIDs in array
  order, and re-encodes every `.npz` under `runs/ieee_cis`: rows of a split become their indices, adversarial
  examples and explanations become the changed entries against their nearest such row, and amount or label
  columns become references. The export checks that every file restores bit for bit and that no source value
  is left (117 files, 94 of them re-encoded, 7.6 MB on the first-run outputs). Indices point into arrays that
  only a rebuild gives, so they say nothing on their own. `scripts/ieee_cis_release.py rebuild --release <dir>`
  then:
  - rebuilds the arrays from the official competition files (download them after accepting the rules);
  - checks the hashes;
  - writes every output file back as the study saved it.

  Models, scores and CSV results hold no source values and ship as for the other datasets; whether models
  trained on competition data may ship is part of the clearance.

Everything goes public only after the author's employer clears the release. The DOI bundle (data, models,
attack outputs, with Croissant metadata) is prepared but not uploaded.
