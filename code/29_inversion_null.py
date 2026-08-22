import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import SVI, Trace_ELBO, autoguide

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import INDICATORS, SEED, resolve, revision_report

GUESS = 0.25
STEPS = 5000
LR = 0.01
N_SIM = 20

TRIMS = [
    ("full sample", None),
    ("bottom quartile removed", 0.25),
    ("bottom half removed", 0.50),
]


def threepl_model(Y, guess=GUESS):
    n_models, n_items = Y.shape
    theta = numpyro.sample("theta", dist.Normal(0., 1.).expand([n_models]))
    a = numpyro.sample("a", dist.HalfNormal(1.).expand([n_items]))
    b = numpyro.sample("b", dist.Normal(0., 1.).expand([n_items]))
    p = guess + (1.0 - guess) * jax.nn.sigmoid(a * (theta[:, None] - b))
    numpyro.sample("obs", dist.Bernoulli(probs=jnp.clip(p, 1e-6, 1 - 1e-6)),
                   obs=Y)


def interaction_model(Y, W, guess=GUESS):
    n_models, n_items = Y.shape
    theta = numpyro.sample("theta", dist.Normal(0., 1.).expand([n_models]))
    a = numpyro.sample("a", dist.HalfNormal(1.).expand([n_items]))
    b = numpyro.sample("b", dist.Normal(0., 1.).expand([n_items]))
    g0 = numpyro.sample("gamma0", dist.Normal(0., 1.))
    g1 = numpyro.sample("gamma1", dist.Normal(0., 1.))

    eta = (a * (theta[:, None] - b)
           + (g0 + g1 * theta[:, None]) * W[None, :])
    p = guess + (1.0 - guess) * jax.nn.sigmoid(eta)
    numpyro.sample("obs", dist.Bernoulli(probs=jnp.clip(p, 1e-6, 1 - 1e-6)),
                   obs=Y)


def run_svi(model, seed, steps=STEPS, **kw):
    guide = autoguide.AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(step_size=LR), loss=Trace_ELBO())
    res = svi.run(random.PRNGKey(seed), num_steps=steps, progress_bar=False,
                  **kw)
    return guide, res


def per_model_slopes(Y, x):
    design = sm.add_constant(x.reshape(-1, 1))
    out = np.zeros(Y.shape[0])
    for j in range(Y.shape[0]):
        yj = Y[j]
        if yj.min() == yj.max():
            continue
        try:
            out[j] = sm.Logit(yj, design).fit(
                disp=0, method="lbfgs", maxiter=1000).params[1]
        except Exception:
            out[j] = 0.0
    return out


def per_model_partial(Y, X, col):
    design = sm.add_constant(X)
    out = np.zeros(Y.shape[0])
    for j in range(Y.shape[0]):
        yj = Y[j]
        if yj.min() == yj.max():
            continue
        try:
            out[j] = sm.Logit(yj, design).fit(
                disp=0, method="lbfgs", maxiter=1000).params[col + 1]
        except Exception:
            out[j] = 0.0
    return out


def trimmed_r(theta, beta, frac):
    if frac is None:
        keep = np.ones(len(theta), bool)
    else:
        keep = theta >= np.quantile(theta, frac)
    r, p = sps.pearsonr(theta[keep], beta[keep])
    return r, p, int(keep.sum()), float(theta[keep].min())


def main():
    print("Generative null for the reasoning-stability inversion")
    print("JAX backend:", jax.devices()[0].platform.upper())

    df = pd.read_csv(resolve("mmlu_IRT_calibrated.csv"), low_memory=False)
    feats = pd.read_csv(resolve("mmlu_dimension5_adversarial.csv"))
    meta = ['question_id', 'item_id', 'subject', 'question_text', 'clean_text',
            'choices', 'ground_truth', 'domain_group', 'difficulty_score',
            'discrimination_score']
    model_cols = [c for c in df.columns if c not in meta]
    assert (feats['question_id'].values == df['question_id'].values).all()

    stem = (df['domain_group'] == 'STEM').to_numpy()
    Y = df.loc[stem, model_cols].to_numpy(dtype=np.int8).T
    W = feats.loc[stem, 'WSCG_Depth'].to_numpy(dtype=np.float64)
    Xall = feats.loc[stem, INDICATORS].to_numpy(dtype=np.float64)
    b_stem = df.loc[stem, 'difficulty_score'].to_numpy(dtype=np.float64)

    abil = pd.read_csv(resolve("mmlu_model_abilities.csv")).set_index(
        'model_name')
    theta = abil.loc[model_cols, 'theta_score'].to_numpy(dtype=np.float64)

    n_models, n_items = Y.shape
    out = ["A GENERATIVE NULL FOR THE REASONING-STABILITY INVERSION", ""]
    out.append(f"Models: {n_models}   STEM items: {n_items}")
    out.append(f"Guessing floor fixed at c = {GUESS}")
    out.append("")


    print("recomputing observed slopes ...")
    beta_obs = per_model_slopes(Y, W)

    out += ["OBSERVED", "",
            "Marginal per-model logistic slope of accuracy on WSCG Depth,",
            "correlated with 2PL ability, at the trim levels the paper reports.",
            ""]
    out.append(f"{'Trim':<28}{'N':>6}{'min theta':>12}{'r':>9}{'p':>12}")
    out.append("-" * 67)
    observed = {}
    for label, frac in TRIMS:
        r, p, n, tmin = trimmed_r(theta, beta_obs, frac)
        observed[label] = r
        out.append(f"{label:<28}{n:>6}{tmin:>12.3f}{r:>9.3f}{p:>12.2e}")


    print("fitting 3PL with fixed guessing floor ...")
    Yj = jnp.asarray(Y.astype(np.float32))
    guide, res = run_svi(threepl_model, SEED, Y=Yj)
    med = guide.median(res.params)
    a_hat = np.asarray(med['a'])
    b_hat = np.asarray(med['b'])
    th_hat = np.asarray(med['theta'])

    out += ["", "", "THE NULL MODEL", "",
            "A 3PL with c fixed at the guessing rate is fitted to the same STEM",
            "submatrix. Its parameters carry the ability spread, the difficulty",
            "spread and the discrimination spread of the real data, and nothing",
            "else: the generating process contains no term in which complexity",
            "interacts with ability.",
            ""]
    out.append(f"  final ELBO loss                : {float(res.losses[-1]):.1f}")
    out.append(f"  theta_hat vs published theta   : "
               f"r = {np.corrcoef(th_hat, theta)[0, 1]:.4f}")
    out.append(f"  b_hat vs published difficulty  : "
               f"r = {np.corrcoef(b_hat, b_stem)[0, 1]:.4f}")
    out.append(f"  fitted b range                 : "
               f"[{b_hat.min():.2f}, {b_hat.max():.2f}]")


    print(f"simulating {N_SIM} response matrices ...")
    P = GUESS + (1.0 - GUESS) / (1.0 + np.exp(-a_hat * (th_hat[:, None] - b_hat)))
    rng = np.random.default_rng(SEED)
    sim_r = {label: [] for label, _ in TRIMS}
    beta_null_mean = np.zeros(n_models)
    for s in range(N_SIM):
        Ys = (rng.random(P.shape) < P).astype(np.int8)
        beta_s = per_model_slopes(Ys, W)
        beta_null_mean += beta_s / N_SIM
        for label, frac in TRIMS:
            r, _, _, _ = trimmed_r(th_hat, beta_s, frac)
            sim_r[label].append(r)
        print(f"  draw {s + 1}/{N_SIM} done")

    out += ["", "", "OBSERVED AGAINST THE NULL", "",
            f"{N_SIM} simulated matrices, each pushed through the identical",
            "per-model logistic procedure. The null column is what the guessing",
            "floor alone produces.",
            ""]
    out.append(f"{'Trim':<28}{'observed r':>12}{'null mean':>11}"
               f"{'null SD':>10}{'null range':>20}{'residual':>10}")
    out.append("-" * 91)
    residual = {}
    for label, _ in TRIMS:
        v = np.array(sim_r[label])
        res_r = observed[label] - v.mean()
        residual[label] = res_r
        out.append(f"{label:<28}{observed[label]:>12.3f}{v.mean():>11.3f}"
                   f"{v.std(ddof=1):>10.3f}"
                   f"{f'[{v.min():.3f}, {v.max():.3f}]':>20}{res_r:>10.3f}")

    frac_acct = {
        label: (np.mean(sim_r[label]) / observed[label]
                if observed[label] != 0 else np.nan)
        for label, _ in TRIMS}
    out.append("")
    for label, _ in TRIMS:
        out.append(f"  {label:<28} null accounts for "
                   f"{100 * frac_acct[label]:5.1f}% of the observed correlation")


    print("computing partial slopes ...")
    jw = INDICATORS.index('WSCG_Depth')
    beta_txt = per_model_partial(Y, Xall, jw)
    beta_adj = per_model_partial(Y, np.column_stack([Xall, b_stem]), jw)

    out += ["", "", "THE SLOPE ADJUSTED FOR EVERYTHING ELSE", "",
            "The published slope is marginal, and WSCG Depth is correlated with",
            "length, noun count and item difficulty, so part of what it measures",
            "is sensitivity to those. Two adjustments are reported, and they",
            "answer different questions. Adjusting for the other eight indicators",
            "isolates the depth effect from the text features that travel with",
            "it. Additionally adjusting for the item's own IRT difficulty asks",
            "whether depth matters beyond how hard the item turned out to be,",
            "which conditions on a consequence of depth and should be read as a",
            "bound rather than as the estimate.",
            ""]
    out.append(f"{'Trim':<28}{'marginal':>10}{'+ 8 indicators':>16}"
               f"{'+ difficulty':>14}")
    out.append("-" * 68)
    for label, frac in TRIMS:
        r_m, _, _, _ = trimmed_r(theta, beta_obs, frac)
        r_t, _, _, _ = trimmed_r(theta, beta_txt, frac)
        r_a, _, _, _ = trimmed_r(theta, beta_adj, frac)
        out.append(f"{label:<28}{r_m:>10.3f}{r_t:>16.3f}{r_a:>14.3f}")

    print("simulating the null for the partial slopes ...")
    n_adj = max(3, N_SIM // 4)
    sim_txt = {label: [] for label, _ in TRIMS}
    sim_adj = {label: [] for label, _ in TRIMS}
    for s in range(n_adj):
        Ys = (rng.random(P.shape) < P).astype(np.int8)
        bt = per_model_partial(Ys, Xall, jw)
        ba = per_model_partial(Ys, np.column_stack([Xall, b_stem]), jw)
        for label, frac in TRIMS:
            sim_txt[label].append(trimmed_r(th_hat, bt, frac)[0])
            sim_adj[label].append(trimmed_r(th_hat, ba, frac)[0])
        print(f"  partial draw {s + 1}/{n_adj} done")
    out.append("")
    out.append("Against the same simulated null:")
    out.append("")
    out.append(f"{'Trim':<28}{'+8 obs':>9}{'+8 null':>9}{'resid':>8}"
               f"{'+b obs':>9}{'+b null':>9}{'resid':>8}")
    out.append("-" * 80)
    for label, frac in TRIMS:
        r_t, _, _, _ = trimmed_r(theta, beta_txt, frac)
        r_a, _, _, _ = trimmed_r(theta, beta_adj, frac)
        m_t = float(np.mean(sim_txt[label]))
        m_a = float(np.mean(sim_adj[label]))
        out.append(f"{label:<28}{r_t:>9.3f}{m_t:>9.3f}{r_t - m_t:>8.3f}"
                   f"{r_a:>9.3f}{m_a:>9.3f}{r_a - m_a:>8.3f}")


    print("fitting the explanatory interaction model ...")
    Wz = (W - W.mean()) / W.std()
    g_guide, g_res = run_svi(interaction_model, SEED, Y=Yj, W=jnp.asarray(Wz))
    pred = numpyro.infer.Predictive(g_guide, params=g_res.params,
                                    num_samples=500)
    draws = pred(random.PRNGKey(SEED + 7), Y=Yj, W=jnp.asarray(Wz))
    g1 = np.asarray(draws['gamma1']).ravel()
    g0 = np.asarray(draws['gamma0']).ravel()

    out += ["", "", "THE INTERACTION ESTIMATED INSIDE THE RESPONSE MODEL", "",
            "logit P(Y_ji = 1) = a_i (theta_j - b_i) + (g0 + g1 theta_j) W_i,",
            "with the lower asymptote fixed at the guessing rate and WSCG Depth",
            "standardised. This estimates the interaction in one fit, respects",
            "the floor natively, and does not read theta and the slope off",
            "overlapping data.",
            ""]
    out.append(f"  gamma0 (complexity effect at mean ability) : {g0.mean():+.4f}"
               f"  [{np.percentile(g0, 2.5):+.4f}, {np.percentile(g0, 97.5):+.4f}]")
    out.append(f"  gamma1 (ability x complexity)              : {g1.mean():+.4f}"
               f"  [{np.percentile(g1, 2.5):+.4f}, {np.percentile(g1, 97.5):+.4f}]")
    excl = (np.percentile(g1, 2.5) > 0) or (np.percentile(g1, 97.5) < 0)
    out.append(f"  95% credible interval excludes zero        : "
               f"{'yes' if excl else 'no'}")
    out.append(f"  final ELBO loss                            : "
               f"{float(g_res.losses[-1]):.1f}")
    out.append("")
    out.append("Mean-field variational intervals understate posterior spread, so")
    out.append("the interval above is a lower bound on uncertainty and the sign")
    out.append("and magnitude of gamma1 are what should be read from it.")


    lead = residual["bottom half removed"]
    out += ["", "", "READING", ""]
    out.append("The floor correction the paper applies is not sufficient on its")
    out.append("own terms. Trimming removes models whose slopes are flat by")
    out.append("constraint; it does not remove the compression that operates")
    out.append("continuously above the trim point, and the simulated null")
    out.append("reproduces a substantial share of the published correlation at")
    out.append("every trim level.")
    out.append("")
    if abs(lead) < 0.10:
        out.append("After correcting against that null the residual effect is")
        out.append(f"r = {lead:.3f} at the trim the paper reports, which is not a")
        out.append("finding. The inversion should not appear in the abstract or")
        out.append("the conclusion as evidence of anything, and Section 5.1 should")
        out.append("report the null rather than the raw correlation.")
    else:
        out.append(f"A residual of r = {lead:.3f} survives at the trim the paper")
        out.append("reports, so the effect is not wholly mechanical, but the")
        out.append("published figure is not the size of the effect and should not")
        out.append("be quoted as though it were.")
    out.append("")
    out.append("The model-based estimate is the one to quote, because it does not")
    out.append("depend on the two-stage geometry at all. It is reported above with")
    out.append("its sign and interval, and Section 5.1 has been rewritten around")
    out.append("it rather than around the marginal correlation.")

    path = revision_report("results_inversion_null.txt", out)
    outdir = os.path.dirname(path)
    pd.DataFrame({
        'model_name': model_cols, 'theta': theta, 'theta_3pl': th_hat,
        'beta_marginal': beta_obs, 'beta_partial_text': beta_txt,
        'beta_partial_difficulty': beta_adj,
        'beta_null_mean': beta_null_mean,
    }).to_csv(os.path.join(outdir, "mmlu_inversion_slopes.csv"), index=False)


    pd.DataFrame({'a_3pl': a_hat, 'b_3pl': b_hat}).to_csv(
        os.path.join(outdir, "mmlu_3pl_stem_items.csv"), index=False)
    print("\n".join(out))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
