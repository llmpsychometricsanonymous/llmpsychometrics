import os
import sys
import urllib.parse

import numpy as np
import pandas as pd
from datasets import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import hfget
from hs_text import queries_from_dataset

MODELS = ["01-ai/Yi-1.5-34B", "Qwen/Qwen2-72B",
          "meta-llama/Meta-Llama-3-70B-Instruct"]


def examples_for(model):
    repo = hfget.detail_repo(model)
    path = hfget.pick(hfget.list_files(repo), "hellaswag")
    url = hfget.BASE.format(repo=repo, path=urllib.parse.quote(path))
    return pd.read_parquet(url, columns=["example"])["example"].tolist()


def main():
    ds = load_dataset("Rowan/hellaswag", split="validation")
    mine = queries_from_dataset(ds)

    orders = {}
    for m in MODELS:
        try:
            orders[m] = examples_for(m)
            print("fetched example order for", m, flush=True)
        except Exception as e:
            print("failed", m, type(e).__name__, flush=True)

    keys = list(orders)
    base = orders[keys[0]]
    for k in keys[1:]:
        same = (orders[k] == base)
        print(f"order identical to {keys[0]}? {k}: {same}")

    print("\nset of texts matches dataset:", set(base) == set(mine))
    print("duplicate texts in dataset:", len(mine) - len(set(mine)))

    pos = {}
    for i, t in enumerate(mine):
        pos.setdefault(t, []).append(i)

    mapping = []
    used = {}
    for t in base:
        cands = pos.get(t)
        if not cands:
            mapping.append(-1)
            continue
        n = used.get(t, 0)
        mapping.append(cands[n] if n < len(cands) else cands[-1])
        used[t] = n + 1

    mapping = np.array(mapping)
    print("unmapped rows:", int((mapping < 0).sum()))
    print("mapping is a permutation:",
          sorted(mapping.tolist()) == list(range(len(mine))))

    out_path = os.path.join(REPO, "raw_data", "hellaswag_index_map.npy")
    np.save(out_path, mapping)
    print("saved", out_path)


if __name__ == "__main__":
    main()
