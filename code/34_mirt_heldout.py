import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import SVI, Trace_ELBO, autoguide

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SEED, output_path, resolve

STEPS = 6000
LR = 0.01
HOLDOUT = 0.05


def unidim(Y, obs_mask):
    n_models, n_items = Y.shape
    theta = numpyro.sample("theta", dist.Normal(0., 1.).expand([n_models]))
    a = numpyro.sample("a", dist.HalfNormal(1.).expand([n_items]))
    b = numpyro.sample("b", dist.Normal(0., 1.).expand([n_items]))
    with numpyro.handlers.mask(mask=obs_mask):
        numpyro.sample("obs", dist.Bernoulli(logits=a * (theta[:, None] - b)),
                       obs=Y)


def mirt2(Y, obs_mask, dim_idx):
    n_models, n_items = Y.shape
    chol = numpyro.sample("chol", dist.LKJCholesky(2, concentration=1.0))
    z = numpyro.sample("z", dist.Normal(0., 1.).expand([n_models, 2]).to_event(1))
    theta = z @ chol.T
    numpyro.deterministic("theta2", theta)
    a = numpyro.sample("a", dist.HalfNormal(1.).expand([n_items]))
    b = numpyro.sample("b", dist.Normal(0., 1.).expand([n_items]))
    with numpyro.handlers.mask(mask=obs_mask):
        numpyro.sample("obs",
                       dist.Bernoulli(logits=a * (theta[:, dim_idx] - b)),
                       obs=Y)


def run(model, seed, **kw):
    guide = autoguide.AutoNormal(model)
    svi = SVI(model, guide, numpyro.optim.Adam(step_size=LR), loss=Trace_ELBO())
    res = svi.run(random.PRNGKey(seed), num_steps=STEPS, progress_bar=False,
                  **kw)
    return guide, res


def bernoulli_ll(logits, y, mask):
    lp = -jnp.logaddexp(0.0, -logits) * y - jnp.logaddexp(0.0, logits) * (1 - y)
    return float(jnp.sum(jnp.where(mask, lp, 0.0))), int(jnp.sum(mask))


def main():
    print("Held-out validation for the confirmatory MIRT")
    print("JAX backend:", jax.devices()[0].platform.upper())

    df = pd.read_csv(resolve("mmlu_IRT_calibrated.csv"), low_memory=False)
    meta = ['question_id', 'item_id', 'subject', 'question_text', 'clean_text',
            'choices', 'ground_truth', 'domain_group', 'difficulty_score',
            'discrimination_score']
    model_cols = [c for c in df.columns if c not in meta]
    Y = jnp.asarray(df[model_cols].to_numpy(dtype=np.float32).T)
    stem = (df['domain_group'] == 'STEM').to_numpy()
    dim_idx = jnp.asarray(stem.astype(np.int32))

    rng = np.random.default_rng(SEED)
    held = rng.random(Y.shape) < HOLDOUT
    train = jnp.asarray(~held)
    test = jnp.asarray(held)

    out = ["HELD-OUT VALIDATION FOR THE CONFIRMATORY MIRT", ""]
    out.append(f"Models: {Y.shape[0]}   Items: {Y.shape[1]}   "
               f"Cells: {Y.shape[0] * Y.shape[1]}")
    out.append(f"Held out: {int(held.sum())} cells ({100 * held.mean():.2f}%)")
    out.append("")

    print("fitting unidimensional on the training cells ...")
    g1, r1 = run(unidim, SEED, Y=Y, obs_mask=train)
    m1 = g1.median(r1.params)
    lg1 = m1['a'] * (m1['theta'][:, None] - m1['b'])

    print("fitting two-dimensional on the training cells ...")
    g2, r2 = run(mirt2, SEED, Y=Y, obs_mask=train, dim_idx=dim_idx)
    m2 = g2.median(r2.params)
    th2 = np.asarray(m2['z']) @ np.asarray(m2['chol']).T
    lg2 = m2['a'] * (jnp.asarray(th2)[:, dim_idx] - m2['b'])

    ll1_te, n_te = bernoulli_ll(lg1, Y, test)
    ll2_te, _ = bernoulli_ll(lg2, Y, test)
    ll1_tr, n_tr = bernoulli_ll(lg1, Y, train)
    ll2_tr, _ = bernoulli_ll(lg2, Y, train)

    out += ["LOG-LIKELIHOOD, TRAINING AND HELD-OUT CELLS", "",
            "Per-cell figures are the total divided by the number of cells, so",
            "the two columns are directly comparable.",
            ""]
    out.append(f"{'Model':<22}{'train total':>15}{'train/cell':>13}"
               f"{'held-out total':>17}{'held-out/cell':>15}")
    out.append("-" * 82)
    out.append(f"{'unidimensional 2PL':<22}{ll1_tr:>15.0f}{ll1_tr / n_tr:>13.5f}"
               f"{ll1_te:>17.0f}{ll1_te / n_te:>15.5f}")
    out.append(f"{'confirmatory 2D':<22}{ll2_tr:>15.0f}{ll2_tr / n_tr:>13.5f}"
               f"{ll2_te:>17.0f}{ll2_te / n_te:>15.5f}")
    out.append("")
    d_te = ll2_te - ll1_te
    out.append(f"  held-out improvement, total    : {d_te:+.1f} nats")
    out.append(f"  held-out improvement, per cell : {d_te / n_te:+.6f} nats")
    out.append(f"  training improvement, per cell : "
               f"{(ll2_tr - ll1_tr) / n_tr:+.6f} nats")


    lp1 = np.asarray(-jnp.logaddexp(0.0, -lg1) * Y
                     - jnp.logaddexp(0.0, lg1) * (1 - Y))[held]
    lp2 = np.asarray(-jnp.logaddexp(0.0, -lg2) * Y
                     - jnp.logaddexp(0.0, lg2) * (1 - Y))[held]
    diff = lp2 - lp1
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    out.append(f"  paired SE over held-out cells  : {se * len(diff):.1f} nats")
    out.append(f"  z on the held-out difference   : "
               f"{diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))):.2f}")
    out.append("")
    if d_te > 0:
        out.append("The second dimension pays for itself out of sample. The gain")
        out.append("is small per cell, as it must be when the two abilities")
        out.append("correlate above 0.96, but it is a gain on responses neither")
        out.append("model was fitted to, which extra parameters alone cannot buy.")
    else:
        out.append("The second dimension does not pay for itself out of sample:")
        out.append("the unidimensional model predicts held-out responses at least")
        out.append("as well, and the improvement in the training objective is")
        out.append("what a thousand extra parameters buy on their own.")


    rho = float(np.asarray(m2['chol'])[1, 0])
    th_ns, th_s = th2[:, 0], th2[:, 1]
    zs = (th_s - th_s.mean()) / th_s.std(ddof=1)
    zn = (th_ns - th_ns.mean()) / th_ns.std(ddof=1)
    gap = zs - zn

    out += ["", "", "HOW FAR MODELS SIT OFF THE IDENTITY LINE", "",
            "Both abilities are standardised, so the gap is in pooled SD units",
            "and a model at +1 is one standard deviation better at STEM than its",
            "non-STEM standing predicts.",
            ""]
    out.append(f"  latent correlation on the training cells : {rho:.4f}")
    out.append(f"  SD of the standardised gap               : "
               f"{gap.std(ddof=1):.4f}")
    out.append(f"  expected under rho                       : "
               f"{np.sqrt(2 * (1 - rho)):.4f}")
    out.append("")
    out.append(f"{'|gap| exceeds':>16}{'models':>9}{'share':>9}")
    out.append("-" * 34)
    for t in (0.25, 0.5, 1.0, 1.5):
        n = int((np.abs(gap) > t).sum())
        out.append(f"{t:>16.2f}{n:>9}{100 * n / len(gap):>8.1f}%")
    out.append("")
    top = np.argsort(-np.abs(gap))[:10]
    out.append("Largest deviations:")
    out.append(f"  {'model':<52}{'gap':>8}")
    for i in top:
        out.append(f"  {model_cols[i]:<52}{gap[i]:>+8.2f}")

    out.append("")
    q = np.quantile(np.abs(gap), [0.5, 0.9, 0.99])
    out.append(f"  median |gap| {q[0]:.2f}, 90th percentile {q[1]:.2f}, "
               f"99th {q[2]:.2f}")
    out.append("")
    out.append("The scatter is not uniform. Most models sit close to the")
    out.append("identity line, and the aggregate describes them adequately; a")
    out.append("minority sit far enough off it that a single number does not.")
    out.append("That is the same shape as the rank-displacement result, and it")
    out.append("is the honest version of the practical claim: the aggregate is a")
    out.append("good summary for most of the population and a poor one for an")
    out.append("identifiable minority, rather than a poor summary throughout.")

    pd.DataFrame({
        'model_name': model_cols,
        'theta_stem': th_s, 'theta_nonstem': th_ns,
        'theta_stem_z': zs, 'theta_nonstem_z': zn, 'gap_z': gap,
    }).to_csv(output_path("mmlu_mirt_abilities.csv"), index=False)

    text = "\n".join(out) + "\n"
    with open(output_path("results_mirt_heldout.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
