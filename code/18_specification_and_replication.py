import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import jax
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import SVI, Trace_ELBO, autoguide

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (INDICATORS, SEED, SPLIT_HALF_SEED, output_path, resolve,
                    revision_report)

SVI_STEPS = 5000
METADATA_COLS = ['question_id', 'item_id', 'subject', 'question_text',
                 'clean_text', 'choices', 'ground_truth', 'domain_group',
                 'difficulty_score', 'discrimination_score']

def rasch_model(responses):
    n_models, n_items = responses.shape
    theta = numpyro.sample("theta", dist.Normal(0.0, 1.0).expand([n_models]))
    b = numpyro.sample("b", dist.Normal(0.0, 1.0).expand([n_items]))
    logits = theta[:, None] - b
    with numpyro.plate("models", n_models, dim=-2):
        with numpyro.plate("items", n_items, dim=-1):
            numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=responses)

def fit_rasch(responses, seed=SEED):
    guide = autoguide.AutoDiagonalNormal(rasch_model)
    svi = SVI(rasch_model, guide, numpyro.optim.Adam(step_size=0.01),
              loss=Trace_ELBO())
    result = svi.run(random.PRNGKey(seed), num_steps=SVI_STEPS,
                     responses=responses, progress_bar=True)
    medians = guide.median(result.params)
    return np.asarray(medians['b']), np.asarray(medians['theta'])

def domain_analysis(y, X, stem):
    design = sm.add_constant(X)
    pooled = sm.OLS(y, design).fit()
    m_s = sm.OLS(y[stem], design[stem]).fit(cov_type='HC3')
    m_n = sm.OLS(y[~stem], design[~stem]).fit(cov_type='HC3')
    m_s_ols = sm.OLS(y[stem], design[stem]).fit()
    m_n_ols = sm.OLS(y[~stem], design[~stem]).fit()

    lam = 2.0 * (m_s_ols.llf + m_n_ols.llf - pooled.llf)
    df_lrt = len(INDICATORS) + 2

    d = m_s.params[1:] - m_n.params[1:]
    se = np.sqrt(m_s.bse[1:] ** 2 + m_n.bse[1:] ** 2)
    z = d / se
    V = (np.asarray(m_s.cov_params())[1:, 1:]
         + np.asarray(m_n.cov_params())[1:, 1:])
    wald = float(d @ np.linalg.solve(V, d))

    mdd = INDICATORS.index('Syntactic_MDD')
    return {
        'r2_stem': m_s.rsquared, 'r2_nonstem': m_n.rsquared,
        'gap': m_s.rsquared - m_n.rsquared,
        'lambda': lam, 'p_lambda': float(stats.chi2.sf(lam, df_lrt)),
        'wald': wald,
        'p_wald': float(stats.chi2.sf(wald, len(INDICATORS))),
        'z_mdd': z[mdd],
        'z_ner': z[INDICATORS.index('Knowledge_NER_Density')],
        'n_stem': int(stem.sum()), 'n_nonstem': int((~stem).sum()),
    }

def format_block(title, res):
    return [
        title,
        f"  N: STEM {res['n_stem']}, non-STEM {res['n_nonstem']}",
        f"  STEM R2 = {res['r2_stem']:.6f}, "
        f"non-STEM R2 = {res['r2_nonstem']:.6f}, "
        f"gap = {res['gap']:.6f}",
        f"  Wald (HC3, 9 df) = {res['wald']:.4f}, p = {res['p_wald']:.3e}",
        f"  Lambda (11 df)   = {res['lambda']:.4f}, "
        f"p = {res['p_lambda']:.3e}",
        f"  Z(Syntactic MDD) = {res['z_mdd']:.4f}   "
        f"Z(Entity Density) = {res['z_ner']:.4f}",
    ]

def main():
    print(f"Specification and replication checks (JAX backend: "
          f"{jax.devices()[0].platform.upper()})")

    df = pd.read_csv(resolve("mmlu_dimension5_adversarial.csv"))
    df = df.dropna(subset=INDICATORS + ['difficulty_score', 'domain_group'])
    df = df.reset_index(drop=True)

    X = df[INDICATORS].values
    y_2pl = df['difficulty_score'].values
    stem = (df['domain_group'] == 'STEM').values

    out = ["IRT SPECIFICATION AND REPLICATION CHECKS", ""]

    rasch_path = output_path("mmlu_rasch_difficulty.csv")
    cached_rasch = resolve("mmlu_rasch_difficulty.csv")
    if os.path.exists(cached_rasch):
        print("Reusing cached Rasch estimates.")
        rasch = pd.read_csv(cached_rasch)
    else:
        aligned = pd.read_csv(resolve("mmlu_IRT_calibrated.csv"))
        model_cols = [c for c in aligned.columns if c not in METADATA_COLS]
        responses = aligned[model_cols].values.T.astype(float)
        print(f"Fitting 1PL on {responses.shape[0]} models "
              f"x {responses.shape[1]} items.")
        b_rasch, theta_rasch = fit_rasch(responses)
        rasch = pd.DataFrame({'question_id': aligned['question_id'],
                              'difficulty_rasch': b_rasch})
        rasch.to_csv(rasch_path, index=False)
        pd.DataFrame({'model_name': model_cols,
                      'theta_rasch': theta_rasch}).to_csv(
            output_path("mmlu_rasch_abilities.csv"), index=False)

    merged = df.merge(rasch, on='question_id', how='left')
    y_rasch = merged['difficulty_rasch'].values
    keep = ~np.isnan(y_rasch)

    r_pearson, p_pearson = stats.pearsonr(y_2pl[keep], y_rasch[keep])
    r_spearman, _ = stats.spearmanr(y_2pl[keep], y_rasch[keep])

    out.append("1PL (RASCH) VERSUS 2PL DIFFICULTY")
    out.append(f"  Items compared: {int(keep.sum())}")
    out.append(f"  Pearson  r(b_2PL, b_1PL) = {r_pearson:.6f} "
               f"(p = {p_pearson:.3e})")
    out.append(f"  Spearman rho             = {r_spearman:.6f}")
    out.append(f"  b_1PL range: [{y_rasch[keep].min():.4f}, "
               f"{y_rasch[keep].max():.4f}], mean {y_rasch[keep].mean():.4f}, "
               f"SD {y_rasch[keep].std(ddof=1):.4f}")
    out.append("")
    out += format_block("Domain analysis on 2PL difficulty (published):",
                        domain_analysis(y_2pl, X, stem))
    out.append("")
    out += format_block("Domain analysis on 1PL difficulty:",
                        domain_analysis(y_rasch[keep], X[keep], stem[keep]))

    rng = np.random.default_rng(SPLIT_HALF_SEED)
    order = rng.permutation(len(df))
    half_a = np.zeros(len(df), bool)
    half_a[order[:len(df) // 2]] = True

    out.append("")
    out.append("RANDOM SPLIT-HALF REPLICATION OF THE DOMAIN ANALYSIS")
    out.append("Items are split at random into two disjoint halves; the whole "
               "domain-stratified analysis is re-run on each independently.")
    out.append("")
    res_a = domain_analysis(y_2pl[half_a], X[half_a], stem[half_a])
    res_b = domain_analysis(y_2pl[~half_a], X[~half_a], stem[~half_a])
    out += format_block("Half A:", res_a)
    out.append("")
    out += format_block("Half B:", res_b)

    replicated = (res_a['gap'] > 0 and res_b['gap'] > 0
                  and res_a['p_wald'] < 0.001 and res_b['p_wald'] < 0.001)
    out.append("")
    out.append(f"Both halves show STEM R2 > non-STEM R2 and reject "
               f"coefficient equality at p < 0.001: {replicated}")

    reps = []
    for r in range(20):
        rr = np.random.default_rng(SPLIT_HALF_SEED + 100 + r)
        o = rr.permutation(len(df))
        m = np.zeros(len(df), bool)
        m[o[:len(df) // 2]] = True
        for mask in (m, ~m):
            reps.append(domain_analysis(y_2pl[mask], X[mask], stem[mask]))
    df_reps = pd.DataFrame(reps)
    out.append("")
    out.append("20 independent random splits (40 half-samples):")
    out.append(f"  STEM R2:     mean {df_reps['r2_stem'].mean():.4f}, "
               f"range [{df_reps['r2_stem'].min():.4f}, "
               f"{df_reps['r2_stem'].max():.4f}]")
    out.append(f"  non-STEM R2: mean {df_reps['r2_nonstem'].mean():.4f}, "
               f"range [{df_reps['r2_nonstem'].min():.4f}, "
               f"{df_reps['r2_nonstem'].max():.4f}]")
    out.append(f"  Half-samples with STEM R2 > non-STEM R2: "
               f"{int((df_reps['gap'] > 0).sum())}/{len(df_reps)}")
    out.append(f"  Half-samples rejecting coefficient equality at p < 0.001: "
               f"{int((df_reps['p_wald'] < 0.001).sum())}/{len(df_reps)}")
    out.append(f"  Z(Syntactic MDD) range: [{df_reps['z_mdd'].min():.3f}, "
               f"{df_reps['z_mdd'].max():.3f}]")
    out.append(f"  Z(Entity Density) range: [{df_reps['z_ner'].min():.3f}, "
               f"{df_reps['z_ner'].max():.3f}]")

    df_reps.to_csv(output_path("mmlu_split_half_replication.csv"), index=False)
    text = "\n".join(out)
    print(text)
    path = revision_report("results_specification_replication.txt", out)
    print(f"Written: {os.path.basename(path)}, "
          "mmlu_split_half_replication.csv")

if __name__ == '__main__':
    main()
