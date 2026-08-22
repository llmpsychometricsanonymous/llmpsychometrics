import os
import sys
import time
import pickle
import importlib.util
import numpy as np
import pandas as pd

N_SUBSET = int(sys.argv[1]) if len(sys.argv) > 1 else 1500

code_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(code_dir)

spec = importlib.util.spec_from_file_location(
    "sens", os.path.join(code_dir, "15_wscg_sensitivity.py"))
sens = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sens)

res_dir = os.path.join(base_dir, "results")
if not os.path.exists(os.path.join(res_dir, "mmlu_parse_cache.pkl")):
    res_dir = os.path.join(base_dir, "results_precomputed")

with open(os.path.join(res_dir, "mmlu_parse_cache.pkl"), "rb") as f:
    cache = pickle.load(f)
df = pd.read_csv(os.path.join(res_dir, "mmlu_dimension1_CoT_reasoning.csv"))

n = min(N_SUBSET, len(cache)) if N_SUBSET > 0 else len(cache)
zero = {t: 0.0 for t in sens.TIERS}

t0 = time.time()
out = [sens.wscg_from_parse(cache[i], zero) for i in range(n)]
elapsed = time.time() - t0

nodes = np.array([o[0] for o in out], dtype=float)
depth = np.array([o[1] for o in out], dtype=float)
o_nodes = df['WSCG_Nodes'].values[:n]
o_depth = df['WSCG_Depth'].values[:n]

d_exact = int((np.abs(depth - o_depth) < 1e-6).sum())
n_exact = int((nodes == o_nodes).sum())

print(f"items compared:      {n}")
print(f"elapsed:             {elapsed:.1f}s  "
      f"(full {len(cache)}-item pass ~ {elapsed*len(cache)/n:.0f}s)")
print(f"nodes exact match:   {n_exact}/{n}  ({100*n_exact/n:.2f}%)")
print(f"depth exact match:   {d_exact}/{n}  ({100*d_exact/n:.2f}%)")
print(f"r(depth):            {np.corrcoef(depth, o_depth)[0,1]:.6f}")
print(f"max |delta depth|:   {np.abs(depth - o_depth).max():.4f}")

if d_exact == n and n_exact == n:
    print("\nPASS - re-implementation reproduces the published WSCG exactly.")
else:
    bad = np.where(np.abs(depth - o_depth) >= 1e-6)[0][:5]
    print(f"\nMISMATCH on {n - d_exact} item(s); first few: {bad.tolist()}")
    for i in bad:
        print(f"  idx {i}: recomputed=({nodes[i]:.0f}, {depth[i]:.4f}) "
              f"published=({o_nodes[i]:.0f}, {o_depth[i]:.4f})")
