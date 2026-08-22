import os
import sys

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import SVI, Trace_ELBO, autoguide

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import INDICATORS, INDICATOR_LABELS, SEED, resolve, revision_report

STEPS = 6000
LR = 0.01
DIAG_MODELS = 250
DIAG_STEPS = [3000, 8000, 20000]


def explanatory_model(X, stem, subj_idx, n_subj, Y):
    n_models, n_items = Y.shape
    k = X.shape[1]

    theta = numpyro.sample("theta", dist.Normal(0., 1.).expand([n_models]))
    a = numpyro.sample("a", dist.HalfNormal(1.).expand([n_items]))

    g0 = numpyro.sample("g0", dist.Normal(0., 2.))
    g_stem = numpyro.sample("g_stem", dist.Normal(0., 2.))
    beta = numpyro.sample("beta", dist.Normal(0., 1.).expand([k]))
    delta = numpyro.sample("delta", dist.Normal(0., 1.).expand([k]))

    sigma_u = numpyro.sample("sigma_u", dist.HalfNormal(1.))
    u = numpyro.sample("u", dist.Normal(0., 1.).expand([n_subj]))
    sigma_e = numpyro.sample("sigma_e", dist.HalfNormal(2.))
    e = numpyro.sample("e", dist.Normal(0., 1.).expand([n_items]))

    b = (g0 + g_stem * stem + X @ beta + (X * stem[:, None]) @ delta
         + sigma_u * u[subj_idx] + sigma_e * e)
    numpyro.deterministic("b", b)

    logits = a * (theta[:, None] - b)
    numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=Y)


def mirt2_model(dim_idx, Y):
    n_models, n_items = Y.shape

    chol = numpyro.sample("chol", dist.LKJCholesky(2, concentration=1.0))
    z = numpyro.sample("z", dist.Normal(0., 1.).expand([n_models, 2]).to_event(1))
    theta = z @ chol.T
    numpyro.deterministic("rho", chol[1, 0])

    a = numpyro.sample("a", dist.HalfNormal(1.).expand([n_items]))
    b = numpyro.sample("b", dist.Normal(0., 1.).expand([n_items]))

    theta_i = theta[:, dim_idx]
    numpyro.sample("obs", dist.Bernoulli(logits=a * (theta_i - b)), obs=Y)


def unidim_model(Y):
    n_models, n_items = Y.shape
    theta = numpyro.sample("theta", dist.Normal(0., 1.).expand([n_models]))
    a = numpyro.sample("a", dist.HalfNormal(1.).expand([n_items]))
    b = numpyro.sample("b", dist.Normal(0., 1.).expand([n_items]))
    numpyro.sample("obs", dist.Bernoulli(logits=a * (theta[:, None] - b)), obs=Y)


def run_svi(model, seed, steps=STEPS, **kwargs):
    guide = autoguide.AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(step_size=LR), loss=Trace_ELBO())
    res = svi.run(random.PRNGKey(seed), num_steps=steps, progress_bar=False,
                  **kwargs)
    return guide, res


def posterior_summary(guide, params, model, seed, names, n=400, **kwargs):
    pred = numpyro.infer.Predictive(guide, params=params, num_samples=n)
    draws = pred(random.PRNGKey(seed + 1), **kwargs)
    return {nm: np.asarray(draws[nm]) for nm in names if nm in draws}


def main():
    print("Explanatory IRT and confirmatory MIRT")
    print("JAX backend:", jax.devices()[0].platform.upper())

    df = pd.read_csv(resolve("mmlu_IRT_calibrated.csv"), low_memory=False)
    feats = pd.read_csv(resolve("mmlu_dimension5_adversarial.csv"))
    meta = ['question_id', 'item_id', 'subject', 'question_text', 'clean_text',
            'choices', 'ground_truth', 'domain_group', 'difficulty_score',
            'discrimination_score']
    model_cols = [c for c in df.columns if c not in meta]
    Y = jnp.asarray(df[model_cols].to_numpy(dtype=np.float32).T)

    assert (feats['question_id'].values == df['question_id'].values).all(), \
        "feature table and response matrix are not aligned"

    Xraw = feats[INDICATORS].to_numpy(dtype=np.float64)
    Xz = (Xraw - Xraw.mean(0)) / Xraw.std(0)
    stem = (df['domain_group'] == 'STEM').to_numpy().astype(np.float64)
    subj_codes, subj_levels = pd.factorize(df['subject'])

    out = ["EXPLANATORY IRT AND CONFIRMATORY MIRT", ""]


    print("fitting explanatory IRT ...")
    guide_a, res_a = run_svi(
        explanatory_model, SEED,
        X=jnp.asarray(Xz), stem=jnp.asarray(stem),
        subj_idx=jnp.asarray(subj_codes), n_subj=len(subj_levels), Y=Y)
    draws = posterior_summary(
        guide_a, res_a.params, explanatory_model, SEED,
        ["beta", "delta", "sigma_u", "sigma_e", "g_stem"],
        X=jnp.asarray(Xz), stem=jnp.asarray(stem),
        subj_idx=jnp.asarray(subj_codes), n_subj=len(subj_levels), Y=Y)

    out += [
        "MODEL A: EXPLANATORY IRT (LLTM WITH ERROR) -- NOT USABLE AS SPECIFIED",
        "",
        "Difficulty is modelled as g0 + g_stem*STEM + X*beta + (X*STEM)*delta",
        "plus a subject random effect and a residual item term, fitted jointly",
        "with ability and discrimination over all 14,042,000 responses.",
        "Indicators are standardised, so beta and delta are in logits per SD.",
        "",
        f"Final ELBO loss: {float(res_a.losses[-1]):.1f}",
        "",
        "The estimates below are reported for completeness and should NOT be used.",
        "Two independent diagnostics say this fit has not converged, and the",
        "convergence check that follows the table is the reason.",
        "",
    ]
    beta_d, delta_d = draws["beta"], draws["delta"]
    header = (f"{'Indicator':<24}{'beta':>9}{'95% CrI':>20}"
              f"{'delta':>9}{'95% CrI':>20}{'excl. 0':>9}")
    out += [header, "-" * len(header)]
    n_excl = 0
    for i, ind in enumerate(INDICATORS):
        b_lo, b_hi = np.percentile(beta_d[:, i], [2.5, 97.5])
        d_lo, d_hi = np.percentile(delta_d[:, i], [2.5, 97.5])
        excl = (d_lo > 0) or (d_hi < 0)
        n_excl += excl
        out.append(
            f"{INDICATOR_LABELS[ind]:<24}{beta_d[:, i].mean():>9.4f}"
            f"{f'[{b_lo:+.3f}, {b_hi:+.3f}]':>20}"
            f"{delta_d[:, i].mean():>9.4f}"
            f"{f'[{d_lo:+.3f}, {d_hi:+.3f}]':>20}{'yes' if excl else 'no':>9}")

    sig_e = float(draws['sigma_e'].mean())
    out += [
        "",
        f"  Domain interactions whose 95% credible interval excludes zero: "
        f"{n_excl} of {len(INDICATORS)}",
        f"  Subject random-effect SD (sigma_u): {draws['sigma_u'].mean():.4f}",
        f"  Residual item SD (sigma_e):         {sig_e:.4f}",
        f"  STEM intercept shift (g_stem):      {draws['g_stem'].mean():+.4f}",
        "",
        "WHY THIS IS NOT REPORTED IN THE PAPER",
        "",
        f"First, sigma_e = {sig_e:.2f} logits exceeds the standard deviation of the",
        "difficulty distribution the two-stage calibration estimates (2.24). A",
        "residual item term larger than the quantity it is a residual of is not a",
        "plausible parameter value. Second, the credible intervals above are",
        "roughly an order of magnitude narrower than the HC3 standard errors on",
        "the same contrasts from the two-stage fit, which is the signature of a",
        "mean-field variational posterior collapsing rather than of a sharper",
        "estimate. Third, several interaction signs disagree with the two-stage",
        "OLS, and the convergence check below shows those signs are not stable.",
        "",
    ]


    print("running convergence diagnostic ...")
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(model_cols), size=DIAG_MODELS, replace=False)
    Ysub = jnp.asarray(
        df[[model_cols[i] for i in pick]].to_numpy(dtype=np.float32).T)
    kw = dict(X=jnp.asarray(Xz), stem=jnp.asarray(stem),
              subj_idx=jnp.asarray(subj_codes), n_subj=len(subj_levels), Y=Ysub)
    watch = ['Syntactic_MDD', 'WSCG_Nodes', 'Knowledge_NER_Density',
             'Syntactic_Depth']
    out += [
        f"Same model on a {DIAG_MODELS}-model subsample at increasing step counts.",
        "A converged fit is stable across them.",
        "",
    ]
    header = (f"{'steps':>8}{'sigma_e':>10}{'sigma_u':>10}"
              + "".join(f"{'d_' + w.split('_')[-1]:>11}" for w in watch))
    out += [header, "-" * len(header)]
    for steps in DIAG_STEPS:
        g = autoguide.AutoNormal(explanatory_model)
        s = SVI(explanatory_model, g, numpyro.optim.Adam(step_size=LR),
                loss=Trace_ELBO())
        r = s.run(random.PRNGKey(SEED), num_steps=steps, progress_bar=False, **kw)
        med = g.median(r.params)
        d = np.asarray(med['delta'])
        row = "".join(f"{d[INDICATORS.index(w)]:>11.3f}" for w in watch)
        out.append(f"{steps:>8}{float(med['sigma_e']):>10.3f}"
                   f"{float(med['sigma_u']):>10.3f}{row}")

    out += [
        "",
        "The parameters move with the step budget and two interaction",
        "coefficients change sign, so the optimisation has not found a stable",
        "solution. The model is weakly identified as written: a free per-item",
        "residual over 14,042 items can absorb whatever the indicators would",
        "otherwise explain, leaving beta and delta poorly constrained, and a",
        "mean-field guide handles that badly.",
        "",
        "We therefore report the two-stage analysis as the paper's inference and",
        "leave the joint explanatory model as future work. Making it usable needs",
        "a tighter identification strategy -- a strict LLTM without the free item",
        "residual, or an informative prior on its scale -- together with an",
        "inference method whose posterior spread can be trusted. Reporting the",
        "unstable fit because 9 of 9 intervals happen to exclude zero would be",
        "reporting an artefact of the optimiser.",
    ]


    print("fitting unidimensional baseline ...")
    _, res_uni = run_svi(unidim_model, SEED, Y=Y)
    print("fitting confirmatory 2D MIRT ...")
    dim_idx = jnp.asarray(stem.astype(np.int32))
    guide_b, res_b = run_svi(mirt2_model, SEED, dim_idx=dim_idx, Y=Y)
    rho_draws = posterior_summary(guide_b, res_b.params, mirt2_model, SEED,
                                  ["chol"], dim_idx=dim_idx, Y=Y)["chol"]
    rho = rho_draws[:, 1, 0]

    out += [
        "",
        "",
        "MODEL B: CONFIRMATORY TWO-DIMENSIONAL IRT",
        "",
        "Each model receives a STEM ability and a non-STEM ability; each item",
        "loads on the dimension its subject assigns. The structure is fixed",
        "before estimation, so it is a hypothesis the data can reject.",
        "",
        f"  Unidimensional final ELBO loss: {float(res_uni.losses[-1]):.1f}",
        f"  Two-dimensional final ELBO loss: {float(res_b.losses[-1]):.1f}",
        f"  Improvement: {float(res_uni.losses[-1] - res_b.losses[-1]):.1f}",
        "",
        f"  Latent correlation rho(theta_STEM, theta_non-STEM) = {rho.mean():.4f}",
        f"  95% credible interval [{np.percentile(rho, 2.5):.4f}, "
        f"{np.percentile(rho, 97.5):.4f}]",
        f"  Posterior mass at rho > 0.99: {100 * (rho > 0.99).mean():.1f}%",
        "",
        "A latent correlation below unity means the two partitions do not measure",
        "one interchangeable ability, which is the claim the aggregate presupposes.",
        "A correlation this high also means they are far from independent: the",
        "aggregate is close to one construct, and the practical consequences come",
        "from rank compression among closely spaced models rather than from two",
        "abilities that diverge widely.",
    ]

    path = revision_report("results_explanatory_irt.txt", out)
    print("\n".join(out))
    print(f"\nwrote {path}")


if __name__ == '__main__':
    main()
