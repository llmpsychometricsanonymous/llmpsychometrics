import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

import jax
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import SVI, Trace_ELBO, autoguide

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from config import resolve, output_path

RAW_DATA = os.path.join(REPO, "raw_data")
PRECOMPUTED = os.path.join(REPO, "results_precomputed")

IND8 = ['WSCG_Depth', 'WSCG_Nodes', 'Syntactic_MDD', 'Syntactic_Depth',
        'Knowledge_Zipf_Rarity', 'Knowledge_NER_Density',
        'Semantic_Concreteness', 'Adversarial_Score']
SEED = 1729
N_PERM = 2000


def irt_2pl(responses):
    n_models, n_items = responses.shape
    theta = numpyro.sample("theta", dist.Normal(0., 1.).expand([n_models]))
    b = numpyro.sample("b", dist.Normal(0., 1.).expand([n_items]))
    a = numpyro.sample("a", dist.HalfNormal(1.).expand([n_items]))
    logits = a * (theta[:, None] - b)
    with numpyro.plate("models", n_models, dim=-2):
        with numpyro.plate("items", n_items, dim=-1):
            numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=responses)


def calibrate(responses, seed=SEED, steps=5000):
    guide = autoguide.AutoDiagonalNormal(irt_2pl)
    svi = SVI(irt_2pl, guide, numpyro.optim.Adam(step_size=0.01),
              loss=Trace_ELBO())
    res = svi.run(random.PRNGKey(seed), num_steps=steps,
                  responses=responses, progress_bar=False)
    med = guide.median(res.params)
    return np.asarray(med['b']), np.asarray(med['theta'])


def split_half_reliability(responses, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(responses.shape[0])
    h1, h2 = idx[:len(idx) // 2], idx[len(idx) // 2:]
    b1, _ = calibrate(responses[h1], seed, steps=2500)
    b2, _ = calibrate(responses[h2], seed + 1, steps=2500)
    r = float(np.corrcoef(b1, b2)[0, 1])
    return r, 2 * r / (1 + r)


def stratified_fit(y, X, mask, indicators):
    d = sm.add_constant(X)
    pooled = sm.OLS(y, d).fit(cov_type='HC3')
    ma = sm.OLS(y[mask], d[mask]).fit(cov_type='HC3')
    mb = sm.OLS(y[~mask], d[~mask]).fit(cov_type='HC3')

    delta = ma.params[1:] - mb.params[1:]
    V = (np.asarray(ma.cov_params())[1:, 1:]
         + np.asarray(mb.cov_params())[1:, 1:])
    W = float(delta @ np.linalg.solve(V, delta))
    k = len(indicators)
    se = np.sqrt(ma.bse[1:] ** 2 + mb.bse[1:] ** 2)
    z = delta / se

    def cv(Xs, ys):
        if len(ys) < 50:
            return float('nan')
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        return float(np.mean(cross_val_score(LinearRegression(), Xs, ys,
                                             cv=kf, scoring='r2')))

    return {
        'r2_pooled': pooled.rsquared,
        'r2_a': ma.rsquared, 'r2_b': mb.rsquared,
        'gap': ma.rsquared - mb.rsquared,
        'cv_a': cv(X[mask], y[mask]), 'cv_b': cv(X[~mask], y[~mask]),
        'cv_pooled': cv(X, y),
        'W': W, 'df': k, 'p_W': float(stats.chi2.sf(W, k)),
        'z': z, 'delta': delta, 'se': se,
        'n_a': int(mask.sum()), 'n_b': int((~mask).sum()),
        'b_a': ma.params[1:], 'b_b': mb.params[1:],
    }


def permutation_gap(y, X, mask, n_perm=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    d = sm.add_constant(X)
    n, na = len(y), int(mask.sum())

    def gap_of(m):
        ra = sm.OLS(y[m], d[m]).fit().rsquared
        rb = sm.OLS(y[~m], d[~m]).fit().rsquared
        return ra - rb

    obs = gap_of(mask)
    null = np.empty(n_perm)
    base = np.zeros(n, bool)
    for i in range(n_perm):
        m = base.copy()
        m[rng.choice(n, na, replace=False)] = True
        null[i] = gap_of(m)
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, null, p


def report(name, res, indicators, out):
    out.append("")
    out.append(f"--- {name} ---")
    out.append(f"  group sizes: {res['n_a']} vs {res['n_b']}")
    out.append(f"  R2: {res['r2_a']:.4f} vs {res['r2_b']:.4f}   "
               f"gap {res['gap']:+.4f}   pooled {res['r2_pooled']:.4f}")
    out.append(f"  cross-validated R2: {res['cv_a']:.4f} vs {res['cv_b']:.4f}"
               f"   pooled {res['cv_pooled']:.4f}")
    out.append(f"  joint Wald W({res['df']}) = {res['W']:.2f}, "
               f"p = {res['p_W']:.3e}")
    out.append(f"  {'indicator':<24} {'b(A)':>10} {'b(B)':>10} {'Z':>8}")
    for i, ind in enumerate(indicators):
        out.append(f"  {ind:<24} {res['b_a'][i]:>10.4f} "
                   f"{res['b_b'][i]:>10.4f} {res['z'][i]:>8.2f}")


def main():
    out = ["HELLASWAG TRANSFER TEST", ""]
    print("JAX backend:", jax.devices()[0].platform.upper())

    resp = pd.read_parquet(os.path.join(RAW_DATA, "hellaswag_responses.parquet"))
    ind = pd.read_csv(os.path.join(PRECOMPUTED, "hellaswag_indicators.csv"))
    print("responses:", resp.shape, "indicators:", ind.shape)
    assert resp.shape[1] == len(ind), "item count mismatch"

    models = list(resp.index)
    Y_hs = resp.to_numpy().astype(float)

    out.append(f"Models with HellaSwag per-item responses: {len(models)}")
    out.append(f"Items: {Y_hs.shape[1]}")
    out.append(f"Partition: {(ind.domain_group == 'activitynet').sum()} "
               f"ActivityNet vs {(ind.domain_group == 'wikihow').sum()} WikiHow")

    print("calibrating HellaSwag 2PL ...")
    b_hs, th_hs = calibrate(Y_hs)
    r_hh, sb = split_half_reliability(Y_hs)
    out.append(f"Split-half difficulty reliability at this model count: "
               f"r = {r_hh:.4f} (Spearman-Brown {sb:.4f})")
    out.append(f"Difficulty range [{b_hs.min():.3f}, {b_hs.max():.3f}], "
               f"mean {b_hs.mean():.3f}")

    keep = ind.domain_group.isin(['activitynet', 'wikihow']).values
    X_hs = ind.loc[keep, IND8].values
    y_hs = b_hs[keep]
    mask_hs = (ind.loc[keep, 'domain_group'] == 'activitynet').values

    res_hs = stratified_fit(y_hs, X_hs, mask_hs, IND8)
    report("HellaSwag: ActivityNet (A) vs WikiHow (B)", res_hs, IND8, out)

    obs, null, p = permutation_gap(y_hs, X_hs, mask_hs)
    out.append(f"  permutation null over {N_PERM} draws: observed gap "
               f"{obs:+.4f}, null mean {null.mean():+.4f}, "
               f"null max |gap| {np.abs(null).max():.4f}, p = {p:.4f}")

    print("calibrating MMLU 2PL on the same models ...")
    mm = pd.read_csv(resolve("mmlu_dimension5_adversarial.csv"))
    aligned = pd.read_csv(resolve("mmlu_IRT_calibrated.csv"))
    shared = [m for m in models if m in aligned.columns]
    out.append("")
    out.append(f"MMLU control refitted on the same {len(shared)} models, "
               f"same eight indicators.")
    Y_mm = aligned[shared].to_numpy().T.astype(float)
    b_mm, _ = calibrate(Y_mm)

    mm = mm.copy()
    mm['b_matched'] = b_mm
    mm = mm.dropna(subset=IND8 + ['b_matched', 'domain_group'])
    X_mm = mm[IND8].values
    y_mm = mm['b_matched'].values
    mask_mm = (mm['domain_group'] == 'STEM').values

    res_mm = stratified_fit(y_mm, X_mm, mask_mm, IND8)
    report("MMLU control: STEM (A) vs non-STEM (B)", res_mm, IND8, out)
    obs_m, null_m, p_m = permutation_gap(y_mm, X_mm, mask_mm)
    out.append(f"  permutation null over {N_PERM} draws: observed gap "
               f"{obs_m:+.4f}, null max |gap| {np.abs(null_m).max():.4f}, "
               f"p = {p_m:.4f}")

    out.append("")
    out.append("SURFACE BASELINE (word count only), cross-validated R2")
    for nm, frame, ycol, msk in (("HellaSwag", ind[keep], y_hs, mask_hs),
                                 ("MMLU", mm, y_mm, mask_mm)):
        wc = frame['word_count'].values.reshape(-1, 1) \
            if 'word_count' in frame else None
        if wc is None:
            out.append(f"  {nm}: word_count unavailable")
            continue
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        r2 = float(np.mean(cross_val_score(LinearRegression(), wc, ycol,
                                           cv=kf, scoring='r2')))
        out.append(f"  {nm}: {r2:.4f}")

    text = "\n".join(out)
    print(text)
    report_path = output_path("results_hellaswag_transfer.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    np.save(output_path("hellaswag_permutation_null.npy"), null)
    print("\nwrote", report_path)


if __name__ == "__main__":
    main()
