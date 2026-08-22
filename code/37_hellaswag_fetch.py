import os
import sys
import threading
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import hfget
from config import resolve

RAW_DATA = os.path.join(REPO, "raw_data")
FETCH_CACHE = os.path.join(RAW_DATA, "hellaswag_fetch_cache")
N_TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 250
WORKERS = 16
METRIC = "acc_norm"

lock = threading.Lock()
done, failed = {}, []


def stratified_models():
    ab = pd.read_csv(resolve("mmlu_model_abilities.csv"))
    ab = ab.sort_values("theta_score").reset_index(drop=True)
    idx = np.linspace(0, len(ab) - 1, N_TARGET).round().astype(int)
    idx = sorted(set(idx.tolist()))
    return ab.iloc[idx][["model_name", "theta_score"]].reset_index(drop=True)


def one(model):
    repo = hfget.detail_repo(model)
    files = hfget.list_files(repo)
    path = hfget.pick(files, "hellaswag")
    if path is None:
        raise RuntimeError("no hellaswag parquet")
    url = hfget.BASE.format(repo=repo, path=urllib.parse.quote(path))
    d = pd.read_parquet(url, columns=["metrics"])
    vals = d["metrics"].apply(lambda m: float(m[METRIC])).to_numpy()
    return vals


os.makedirs(FETCH_CACHE, exist_ok=True)


def cache_path(model):
    return os.path.join(FETCH_CACHE, model.replace("/", "__") + ".npy")


def worker(row):
    model = row.model_name
    cp = cache_path(model)
    if os.path.exists(cp):
        with lock:
            done[model] = np.load(cp)
        return
    miss = cp + ".miss"
    if os.path.exists(miss):
        with lock:
            failed.append((model, "cached-miss"))
        return
    try:
        v = one(model)
        np.save(cp, v)
        with lock:
            done[model] = v
            n = len(done)
        print(f"[{n:>4}] ok   {model}  n={len(v)}", flush=True)
    except Exception as e:
        open(miss, "w").close()
        with lock:
            failed.append((model, type(e).__name__))
        print(f"       MISS {model}  ({type(e).__name__})", flush=True)


def main():
    sel = stratified_models()
    sel = sel.sample(frac=1.0, random_state=1729).reset_index(drop=True)
    print(f"requesting {len(sel)} models spanning theta "
          f"[{sel.theta_score.min():.3f}, {sel.theta_score.max():.3f}]",
          flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(as_completed([ex.submit(worker, r)
                           for r in sel.itertuples(index=False)]))

    if not done:
        print("nothing downloaded")
        return
    lengths = {len(v) for v in done.values()}
    print("distinct item counts across models:", lengths)
    n_items = max(lengths, key=lambda L: sum(len(v) == L for v in done.values()))
    keep = {m: v for m, v in done.items() if len(v) == n_items}
    print(f"kept {len(keep)} models at {n_items} items "
          f"(dropped {len(done) - len(keep)} with mismatched length)")

    mat = pd.DataFrame(keep).T
    mat.index.name = "Model_Name"
    mat.to_parquet(os.path.join(RAW_DATA, "hellaswag_responses.parquet"))
    theta = sel.set_index("model_name")["theta_score"]
    theta.loc[theta.index.intersection(mat.index)].to_csv(
        os.path.join(RAW_DATA, "hellaswag_models.csv"))
    print("wrote hellaswag_responses.parquet", mat.shape)
    print("failures:", len(failed))


if __name__ == "__main__":
    main()
