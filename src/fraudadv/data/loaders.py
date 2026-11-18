"""Where the study's raw files came from (DATA.md has the licenses and the release plan).

Input: none (the table below), and the raw folder $FRAUDADV_RAW (default data/raw) to check.
Output: `python -m fraudadv.data.loaders` prints, per dataset, the file(s) that `fraudadv.data.prepare` reads
under the raw folder, whether they are present, the copy the study downloaded, and the original source.

The paper's dataset table (Table II) and data section cite these sources; no number is computed here.
"""
SOURCES = {
    "ulb": dict(files=["ulb/creditcard.csv"],
                copy="https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv",
                original="Machine Learning Group, ULB (Dal Pozzolo et al. 2015)"),
    "ieee_cis": dict(files=["ieee/train_transaction.csv", "ieee/train_identity.csv"],
                     copy="Hugging Face dataset aliceczr/ieee-fraud-detection",
                     original="Vesta, IEEE-CIS Fraud Detection competition (rebuild: scripts/ieee_cis_release.py)"),
    "sparkov": dict(files=["sparkov/credit_card_fraud_transactions.csv"],
                    copy="Hugging Face dataset santosh3110/credit_card_fraud_transactions",
                    original="Sparkov simulator (B. Harris)"),
    "tabformer": dict(files=["tabformer/card_transaction.v1.csv"],
                      copy="github.com/IBM/TabFormer, git-lfs file data/credit_card/transactions.tgz",
                      original="IBM (Altman et al. 2021; Padhi et al. 2021)"),
    "paysim": dict(files=["paysim/paysim.csv"], copy="Hugging Face dataset theman10/paysim",
                   original="E. Lopez-Rojas (Lopez-Rojas et al. 2016)"),
    "baf": dict(files=["baf/baf.parquet"], copy="Hugging Face dataset mkbadeniyi/fraud-detection-benchmark (BAF Base)",
                original="Feedzai, github.com/feedzai/bank-account-fraud (Jesus et al. 2022)"),
}

if __name__ == "__main__":
    from fraudadv.paths import RAW

    for d, s in SOURCES.items():
        have = all((RAW / f).exists() for f in s["files"])
        print(f"{d:10s} {'ok     ' if have else 'missing'} {', '.join(s['files'])}\n{'':18s}copy: {s['copy']}\n"
              f"{'':18s}original: {s['original']}")
