import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from config import resolve

RAW_DATA = os.path.join(REPO, "raw_data")
FETCH_CACHE = os.path.join(RAW_DATA, "hellaswag_fetch_cache")


def main():
    mapping = np.load(os.path.join(RAW_DATA, "hellaswag_index_map.npy"))
    n_items = len(mapping)
    assert sorted(mapping.tolist()) == list(range(n_items)), "not a permutation"

    rows, names = [], []
    for f in sorted(glob.glob(os.path.join(FETCH_CACHE, "*.npy"))):
        v = np.load(f)
        if len(v) != n_items:
            print("skip (length)", os.path.basename(f), len(v))
            continue
        ordered = np.empty(n_items, dtype=float)
        ordered[mapping] = v
        rows.append(ordered)
        names.append(os.path.basename(f)[:-4].replace("__", "/", 1))

    mat = pd.DataFrame(rows, index=names)
    mat.index.name = "Model_Name"
    print("response matrix:", mat.shape)
    print("mean accuracy: %.4f  (min %.4f, max %.4f)"
          % (mat.values.mean(), mat.values.mean(1).min(), mat.values.mean(1).max()))

    ab = pd.read_csv(resolve("mmlu_model_abilities.csv"))
    ab = ab.set_index("model_name")["theta_score"]
    hit = [n for n in mat.index if n in ab.index]
    print("matched to MMLU theta: %d of %d" % (len(hit), len(mat)))
    th = ab.loc[hit]
    print("theta span: [%.3f, %.3f]  median %.3f"
          % (th.min(), th.max(), th.median()))

    mat.to_parquet(os.path.join(RAW_DATA, "hellaswag_responses.parquet"))
    th.to_csv(os.path.join(RAW_DATA, "hellaswag_models.csv"))
    print("wrote hellaswag_responses.parquet and hellaswag_models.csv")


if __name__ == "__main__":
    main()
