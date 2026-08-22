import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (INDICATORS, N_PERMUTATION, PERMUTATION_SEED, output_path,
                    resolve)


def fast_ols(X1, y):
    beta, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    return beta, rss, 1.0 - rss / tss


def gaussian_loglik(rss, n):
    return -0.5 * n * (np.log(2 * np.pi) + np.log(rss / n) + 1.0)


def lrt_from_designs(X1, y, stem):
    _, rss_p, _ = fast_ols(X1, y)
    _, rss_s, r2_s = fast_ols(X1[stem], y[stem])
    _, rss_n, r2_n = fast_ols(X1[~stem], y[~stem])
    n, ns, nn = len(y), int(stem.sum()), int((~stem).sum())
    lam = 2.0 * ((gaussian_loglik(rss_s, ns) + gaussian_loglik(rss_n, nn))
                 - gaussian_loglik(rss_p, n))
    return lam, r2_s, r2_n


def wald_equality(X, y, stem, groups=None):
    design = sm.add_constant(X)

    def fit(mask):
        if groups is None:
            return sm.OLS(y[mask], design[mask]).fit(cov_type='HC3')
        return sm.OLS(y[mask], design[mask]).fit(
            cov_type='cluster', cov_kwds={'groups': groups[mask]})

    m_s, m_n = fit(stem), fit(~stem)
    d = np.asarray(m_s.params)[1:] - np.asarray(m_n.params)[1:]
    V = (np.asarray(m_s.cov_params())[1:, 1:]
         + np.asarray(m_n.cov_params())[1:, 1:])
    W = float(d @ np.linalg.solve(V, d))
    se = np.sqrt(np.diag(V))
    return W, d, se, m_s, m_n


def main():
    print("Subject-clustered inference")
    df = pd.read_csv(resolve("mmlu_dimension5_adversarial.csv"))
    df = df.dropna(subset=INDICATORS + ['difficulty_score', 'domain_group',
                                        'subject'])

    y = df['difficulty_score'].values
    X = df[INDICATORS].values
    X1 = sm.add_constant(X)
    stem = (df['domain_group'] == 'STEM').values
    subjects = df['subject'].values
    subj_codes = pd.Categorical(subjects).codes
    uniq = np.unique(subj_codes)
    stem_subjects = np.unique(subj_codes[stem])
    k = len(INDICATORS)

    from scipy import stats as sps

    out = ["SUBJECT-CLUSTERED INFERENCE FOR THE INVARIANCE TEST", ""]
    out.append(f"Items: {len(y)}   Subjects: {len(uniq)}   "
               f"STEM subjects: {len(stem_subjects)}")
    out.append(f"STEM items: {int(stem.sum())}   "
               f"non-STEM items: {int((~stem).sum())}")
    out.append("")


    W_hc3, d_hc3, se_hc3, _, _ = wald_equality(X, y, stem)
    W_cr1, d_cr1, se_cr1, m_s, m_n = wald_equality(X, y, stem, subj_codes)
    p_hc3 = sps.chi2.sf(W_hc3, k)
    p_cr1 = sps.chi2.sf(W_cr1, k)

    out.append("JOINT WALD TEST OF SLOPE EQUALITY")
    out.append("")
    out.append("Covariance          W(9)            p     median SE inflation")
    out.append("-" * 68)
    out.append(f"HC3 (published)  {W_hc3:8.2f}   {p_hc3:.3e}"
               f"                    ---")
    infl = float(np.median(se_cr1 / se_hc3))
    out.append(f"CR1 by subject   {W_cr1:8.2f}   {p_cr1:.3e}"
               f"                  {infl:5.2f}x")
    out.append("")
    out.append("Per-indicator contrasts under both covariances:")
    out.append("")
    z_hc3 = d_hc3 / se_hc3
    z_cr1 = d_cr1 / se_cr1
    p_raw_hc3 = 2.0 * sps.norm.sf(np.abs(z_hc3))
    p_raw_cr1 = 2.0 * sps.norm.sf(np.abs(z_cr1))
    holm_hc3 = multipletests(p_raw_hc3, method='holm')[1]
    holm_cr1 = multipletests(p_raw_cr1, method='holm')[1]

    out.append("Indicator                    b(S)-b(nS)   SE HC3   SE CR1"
               "    z HC3   z CR1   Holm HC3   Holm CR1")
    out.append("-" * 96)
    for i, name in enumerate(INDICATORS):
        star_h = "*" if holm_hc3[i] < 0.05 else " "
        star_c = "*" if holm_cr1[i] < 0.05 else " "
        out.append(f"{name:<28}{d_cr1[i]:>9.4f}{se_hc3[i]:>9.4f}"
                   f"{se_cr1[i]:>9.4f}{z_hc3[i]:>9.2f}{z_cr1[i]:>8.2f}"
                   f"{holm_hc3[i]:>11.2e}{star_h}"
                   f"{holm_cr1[i]:>10.2e}{star_c}")
    n_hc3 = int((holm_hc3 < 0.05).sum())
    n_cr1 = int((holm_cr1 < 0.05).sum())
    out.append("")
    out.append(f"Surviving Holm at 0.05: {n_hc3} of {k} under HC3, "
               f"{n_cr1} of {k} under CR1.")
    kept = [INDICATORS[i] for i in range(k) if holm_cr1[i] < 0.05]
    dropped = [INDICATORS[i] for i in range(k)
               if holm_hc3[i] < 0.05 and holm_cr1[i] >= 0.05]
    out.append(f"  survive clustering : {', '.join(kept) if kept else 'none'}")
    out.append(f"  lost to clustering : "
               f"{', '.join(dropped) if dropped else 'none'}")


    lam_obs, r2_s_obs, r2_n_obs = lrt_from_designs(X1, y, stem)
    obs_gap = r2_s_obs - r2_n_obs
    n_stem_items = int(stem.sum())
    n_stem_subj = len(stem_subjects)

    prng = np.random.default_rng(PERMUTATION_SEED)
    item_gap = np.empty(N_PERMUTATION)
    item_lam = np.empty(N_PERMUTATION)
    base = np.zeros(len(y), bool)
    for b in range(N_PERMUTATION):
        m = base.copy()
        m[prng.choice(len(y), n_stem_items, replace=False)] = True
        item_lam[b], r2s, r2n = lrt_from_designs(X1, y, m)
        item_gap[b] = r2s - r2n

    prng = np.random.default_rng(PERMUTATION_SEED)
    blk_gap = np.empty(N_PERMUTATION)
    blk_lam = np.empty(N_PERMUTATION)
    blk_n = np.empty(N_PERMUTATION, int)
    for b in range(N_PERMUTATION):
        pick = prng.choice(uniq, n_stem_subj, replace=False)
        m = np.isin(subj_codes, pick)
        blk_lam[b], r2s, r2n = lrt_from_designs(X1, y, m)
        blk_gap[b] = r2s - r2n
        blk_n[b] = int(m.sum())

    def pval(null, obs, two_sided=True):
        if two_sided:
            hits = int(np.sum(np.abs(null) >= abs(obs)))
        else:
            hits = int(np.sum(null >= obs))
        return (hits + 1) / (len(null) + 1), hits

    p_item_gap, hit_item_gap = pval(item_gap, obs_gap)
    p_blk_gap, hit_blk_gap = pval(blk_gap, obs_gap)
    p_item_lam, hit_item_lam = pval(item_lam, lam_obs, two_sided=False)
    p_blk_lam, hit_blk_lam = pval(blk_lam, lam_obs, two_sided=False)

    out.append("")
    out.append(f"PERMUTATION NULLS ({N_PERMUTATION} draws each)")
    out.append("")
    out.append(f"Observed in-sample R2 gap (STEM - non-STEM): {obs_gap:.6f}")
    out.append(f"  STEM R2 {r2_s_obs:.6f}   non-STEM R2 {r2_n_obs:.6f}")
    out.append(f"Observed LRT statistic: {lam_obs:.4f}")
    out.append("")
    out.append("Null over the R-squared gap:")
    out.append("")
    out.append("Null design        mean       SD    95th |gap|   max |gap|"
               "        p    exceed")
    out.append("-" * 82)
    out.append(f"item-level      {item_gap.mean():8.5f} {item_gap.std(ddof=1):8.5f}"
               f"   {np.percentile(np.abs(item_gap), 95):9.5f}"
               f"   {np.abs(item_gap).max():9.5f}  {p_item_gap:8.4f}"
               f"  {hit_item_gap:6d}")
    out.append(f"subject-block   {blk_gap.mean():8.5f} {blk_gap.std(ddof=1):8.5f}"
               f"   {np.percentile(np.abs(blk_gap), 95):9.5f}"
               f"   {np.abs(blk_gap).max():9.5f}  {p_blk_gap:8.4f}"
               f"  {hit_blk_gap:6d}")
    out.append("")
    out.append("Null over the LRT statistic:")
    out.append("")
    out.append("Null design        mean       SD          max        p    exceed")
    out.append("-" * 68)
    out.append(f"item-level      {item_lam.mean():8.2f} {item_lam.std(ddof=1):8.2f}"
               f"  {item_lam.max():11.2f} {p_item_lam:8.4f}  {hit_item_lam:6d}")
    out.append(f"subject-block   {blk_lam.mean():8.2f} {blk_lam.std(ddof=1):8.2f}"
               f"  {blk_lam.max():11.2f} {p_blk_lam:8.4f}  {hit_blk_lam:6d}")
    out.append("")
    out.append(f"Subject-block draws hold the STEM group at {n_stem_subj} "
               f"subjects; item counts")
    out.append(f"therefore vary (mean {blk_n.mean():.0f}, "
               f"range {blk_n.min()}-{blk_n.max()}) against the observed "
               f"{n_stem_items}.")


    sd_ratio = blk_gap.std(ddof=1) / item_gap.std(ddof=1)
    out.append("")
    out.append("READING")
    out.append("")
    out.append("The two nulls answer different questions and give different")
    out.append("answers.")
    out.append("")
    if p_blk_lam < 0.05:
        out.append(f"Slope non-invariance survives. The LRT statistic is beyond")
        out.append(f"the subject-block null at p = {p_blk_lam:.4f}, and the")
        out.append(f"clustered Wald test rejects equality at W(9) = {W_cr1:.2f},")
        out.append(f"p = {p_cr1:.2e}. Clustering inflates the slope standard")
        out.append(f"errors by about {infl:.2f}x and the statistic falls from")
        out.append(f"{W_hc3:.2f} to {W_cr1:.2f}, but the conclusion does not")
        out.append("depend on treating items as independent.")
    else:
        out.append(f"Slope non-invariance does NOT survive the subject-block")
        out.append(f"null (p = {p_blk_lam:.4f}).")
    out.append("")
    if p_blk_gap >= 0.05:
        out.append("The R-squared gap does not survive. Against the item-level")
        out.append(f"null the gap is extreme (p = {p_item_gap:.4f}); against the")
        out.append(f"subject-block null it is ordinary (p = {p_blk_gap:.4f}).")
        out.append(f"The block null is {sd_ratio:.1f}x wider, because a random")
        out.append("union of whole subjects inherits the between-subject")
        out.append("variation in how predictable difficulty is, and the")
        out.append("item-level null destroys exactly that variation. A gap of")
        out.append(f"{obs_gap:.4f} is what any {n_stem_subj}-subject grouping of")
        out.append("MMLU tends to produce. It is therefore not evidence that")
        out.append("the STEM / non-STEM boundary specifically marks a change in")
        out.append("how well structural complexity predicts difficulty.")
        out.append("")
        out.append("What the gap does remain is a description of these two")
        out.append("partitions, which is what a practitioner reading the")
        out.append("leaderboard actually faces. It is not a rejected null.")
    else:
        out.append(f"The R-squared gap survives the subject-block null at "
                   f"p = {p_blk_gap:.4f}.")

    np.save(output_path("mmlu_subject_block_null.npy"), blk_gap)
    pd.DataFrame({
        'indicator': INDICATORS, 'diff': d_cr1,
        'se_hc3': se_hc3, 'se_cr1': se_cr1,
        'z_hc3': z_hc3, 'z_cr1': z_cr1,
        'holm_hc3': holm_hc3, 'holm_cr1': holm_cr1,
    }).to_csv(output_path("mmlu_clustered_contrasts.csv"), index=False)

    text = "\n".join(out) + "\n"
    with open(output_path("results_subject_clustered.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
