"""Load one dataset's processed arrays and metadata, as written by `python -m fraudadv.data.prepare <dataset>`.

Input: data/processed/<dataset>/arrays.npz and meta.json (FRAUDADV_PROC overrides data/processed).
Output: `load(dataset)` returns (arrays, meta):
  arrays  dict of numpy arrays per split (train, val, test): X_ (standardized features), y_ (1 = fraud),
          w_ (sampling weights), amt_ (dollar amount), t_ (time), ent_ (entity id or -1)
  meta    dict: feature names, which features the attacker controls, bounds, category values, scaler
          (see the docstring of fraudadv.data.prepare)

Every model, attack and analysis of the paper reads its data through this function.
"""
import json

import numpy as np

from fraudadv.paths import PROC  # noqa: E402  (FRAUDADV_PROC overrides)
DATASETS = ["ulb", "ieee_cis", "sparkov", "tabformer", "paysim", "baf"]


def load(name):
    """The arrays (dict) and metadata (dict) of dataset `name`."""
    a = dict(np.load(PROC / name / "arrays.npz"))
    meta = json.loads((PROC / name / "meta.json").read_text())
    return a, meta
