import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (BOOTSTRAP_SEED, INDICATORS, N_BOOTSTRAP, N_PERMUTATION,
                    PERMUTATION_SEED, output_path, resolve, revision_report)

def fit(X, y, robust=True):
    design = sm.add_constant(X)
    if robust:
        return sm.OLS(y, design).fit(cov_type='HC3')
    return sm.OLS(y, design).fit()

def fast_ols(X1, y):
    beta, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    return beta, rss, 1.0 - rss / tss

def gaussian_loglik(rss, n):
    return -0.5 * n * (np.log(2 * np.pi) + np.log(rss / n) + 1.0)

def lrt_from_designs(X1_all, y_all, stem):
    _, rss_p, _ = fast_ols(X1_all, y_all)
    _, rss_s, r2_s = fast_ols(X1_all[stem], y_all[stem])
    _, rss_n, r2_n = fast_ols(X1_all[~stem], y_all[~stem])
    n, ns, nn = len(y_all), int(stem.sum()), int((~stem).sum())
    lam = 2.0 * ((gaussian_loglik(rss_s, ns) + gaussian_loglik(rss_n, nn))
                 - gaussian_loglik(rss_p, n))
    return lam, r2_s, r2_n

def cohens_q(r2_a, r2_b):
    return float(np.arctanh(np.sqrt(max(r2_a, 0.0)))
                 - np.arctanh(np.sqrt(max(r2_b, 0.0))))

def main():
    print("Robust invariance testing")
    df = pd.read_csv(resolve("mmlu_dimension5_adversarial.csv"))
    df = df.dropna(subset=INDICATORS + ['difficulty_score', 'domain_group'])

    y = df['difficulty_score'].values
    X = df[INDICATORS].values
    X1 = sm.add_constant(X)
    stem = (df['domain_group'] == 'STEM').values
    k = len(INDICATORS)

    m_pooled = fit(X, y)
    m_stem = fit(X[stem], y[stem])
    m_nonstem = fit(X[~stem], y[~stem])

    out = ["ROBUST INVARIANCE TESTING", ""]

    delta = m_stem.params[1:] - m_nonstem.params[1:]
    se_delta = np.sqrt(m_stem.bse[1:] ** 2 + m_nonstem.bse[1:] ** 2)
    z = delta / se_delta
    p_raw = 2.0 * stats.norm.sf(np.abs(z))
    reject_holm, p_holm, _, _ = multipletests(p_raw, alpha=0.05, method='holm')
    reject_bh, p_bh, _, _ = multipletests(p_raw, alpha=0.05,
                                          method='fdr_bh')

    out.append("CROSS-GROUP COEFFICIENT TESTS WITH MULTIPLICITY CONTROL")
    out.append("Nine simultaneous Clogg et al. (1995) Z-tests; Holm-Bonferroni "
               "and Benjamini-Hochberg applied to the family.")
    out.append("")
    head = (f"{'Indicator':<24} | {'delta b':>9} | {'SE':>8} | {'Z':>7} | "
            f"{'p (raw)':>10} | {'p (Holm)':>10} | {'p (BH)':>10} | Holm |")
    out.append(head)
    out.append("-" * len(head))
    for i, ind in enumerate(INDICATORS):
        out.append(f"{ind:<24} | {delta[i]:>9.4f} | {se_delta[i]:>8.4f} | "
                   f"{z[i]:>7.3f} | {p_raw[i]:>10.3e} | {p_holm[i]:>10.3e} | "
                   f"{p_bh[i]:>10.3e} | "
                   f"{'yes' if reject_holm[i] else 'no':>4} |")
    out.append("")
    out.append(f"Surviving Holm at alpha = 0.05: "
               f"{', '.join(np.array(INDICATORS)[reject_holm]) or 'none'}")
    out.append(f"Surviving Benjamini-Hochberg at q = 0.05: "
               f"{', '.join(np.array(INDICATORS)[reject_bh]) or 'none'}")

    V_stem = np.asarray(m_stem.cov_params())[1:, 1:]
    V_nonstem = np.asarray(m_nonstem.cov_params())[1:, 1:]
    V_delta = V_stem + V_nonstem
    wald = float(delta @ np.linalg.solve(V_delta, delta))
    p_wald = float(stats.chi2.sf(wald, k))

    d_full = m_stem.params - m_nonstem.params
    V_full = (np.asarray(m_stem.cov_params())
              + np.asarray(m_nonstem.cov_params()))
    wald_full = float(d_full @ np.linalg.solve(V_full, d_full))
    p_wald_full = float(stats.chi2.sf(wald_full, k + 1))

    out.append("")
    out.append("JOINT WALD TEST OF COEFFICIENT EQUALITY (HC3 THROUGHOUT)")
    out.append("W = (b_STEM - b_nonSTEM)' [V_STEM + V_nonSTEM]^-1 "
               "(b_STEM - b_nonSTEM), with V the HC3 covariance of each "
               "partition. The partitions are disjoint, so the two covariance "
               "matrices add.")
    out.append(f"  Slopes only:            W = {wald:.4f}, df = {k}, "
               f"p = {p_wald:.3e}")
    out.append(f"  Slopes plus intercept:  W = {wald_full:.4f}, "
               f"df = {k + 1}, p = {p_wald_full:.3e}")

    lam_obs, r2_s_obs, r2_n_obs = lrt_from_designs(X1, y, stem)
    df_lrt = k + 2
    p_lrt = float(stats.chi2.sf(lam_obs, df_lrt))
    out.append("")
    out.append("NORMAL-THEORY LIKELIHOOD RATIO TEST (reference)")
    out.append(f"  Lambda = {lam_obs:.4f}, df = {df_lrt}, p = {p_lrt:.3e}")
    out.append(f"  STEM R2 = {r2_s_obs:.6f}, non-STEM R2 = {r2_n_obs:.6f}, "
               f"gap = {r2_s_obs - r2_n_obs:.6f}")
    out.append(f"  Cohen's q (domain level) = "
               f"{cohens_q(r2_s_obs, r2_n_obs):.4f}")

    out.append("")
    out.append("OLS RESIDUAL DIAGNOSTICS")
    diag_head = (f"{'Partition':<10} | {'skew':>7} | {'ex.kurt':>8} | "
                 f"{'Jarque-Bera':>12} | {'JB p':>10} | "
                 f"{'Breusch-Pagan':>13} | {'BP p':>10} | {'White p':>10} |")
    out.append(diag_head)
    out.append("-" * len(diag_head))
    diagnostics = {}
    for name, mask in (('Pooled', np.ones(len(y), bool)), ('STEM', stem),
                       ('non-STEM', ~stem)):
        m = fit(X[mask], y[mask], robust=False)
        r = m.resid
        jb, jb_p = stats.jarque_bera(r)[:2]
        bp = het_breuschpagan(r, m.model.exog)
        wh = het_white(r, m.model.exog)
        diagnostics[name] = {
            'skew': float(stats.skew(r)),
            'kurtosis': float(stats.kurtosis(r)),
            'jb': float(jb), 'jb_p': float(jb_p),
            'bp': float(bp[0]), 'bp_p': float(bp[1]),
            'white_p': float(wh[1]),
        }
        d = diagnostics[name]
        out.append(f"{name:<10} | {d['skew']:>7.4f} | {d['kurtosis']:>8.4f} | "
                   f"{d['jb']:>12.2f} | {d['jb_p']:>10.3e} | "
                   f"{d['bp']:>13.2f} | {d['bp_p']:>10.3e} | "
                   f"{d['white_p']:>10.3e} |")
    out.append("Residuals are right-skewed and leptokurtic relative to the "
               "normal, and Breusch-Pagan and White both reject "
               "homoskedasticity in every partition. The normal-theory "
               "likelihood the LRT rests on is therefore misspecified, which "
               "is exactly why the HC3 Wald test above, the bootstrap below "
               "and the permutation null are the load-bearing inference and "
               "the LRT is retained only for comparability with the "
               "multi-group SEM literature.")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx_stem = np.flatnonzero(stem)
    idx_nonstem = np.flatnonzero(~stem)
    boot = {'lambda': [], 'gap': [], 'r2_stem': [], 'r2_nonstem': [], 'q': []}
    for _ in range(N_BOOTSTRAP):

        take = np.concatenate([rng.choice(idx_stem, idx_stem.size, True),
                               rng.choice(idx_nonstem, idx_nonstem.size, True)])
        mask_b = np.zeros(take.size, bool)
        mask_b[:idx_stem.size] = True
        lam_b, r2s_b, r2n_b = lrt_from_designs(X1[take], y[take], mask_b)
        boot['lambda'].append(lam_b)
        boot['r2_stem'].append(r2s_b)
        boot['r2_nonstem'].append(r2n_b)
        boot['gap'].append(r2s_b - r2n_b)
        boot['q'].append(cohens_q(r2s_b, r2n_b))
    boot = {key: np.asarray(val) for key, val in boot.items()}

    out.append("")
    out.append(f"NONPARAMETRIC BOOTSTRAP ({N_BOOTSTRAP} stratified item "
               f"resamples)")
    bhead = (f"{'Quantity':<22} | {'observed':>10} | {'boot mean':>10} | "
             f"{'boot SD':>10} | {'2.5%':>10} | {'97.5%':>10} |")
    out.append(bhead)
    out.append("-" * len(bhead))
    for label, key, obs in (('Lambda (LRT)', 'lambda', lam_obs),
                            ('STEM R2', 'r2_stem', r2_s_obs),
                            ('non-STEM R2', 'r2_nonstem', r2_n_obs),
                            ('R2 gap', 'gap', r2_s_obs - r2_n_obs),
                            ("Cohen's q", 'q',
                             cohens_q(r2_s_obs, r2_n_obs))):
        v = boot[key]
        out.append(f"{label:<22} | {obs:>10.4f} | {v.mean():>10.4f} | "
                   f"{v.std(ddof=1):>10.4f} | "
                   f"{np.percentile(v, 2.5):>10.4f} | "
                   f"{np.percentile(v, 97.5):>10.4f} |")
    frac_sig = float(np.mean(stats.chi2.sf(boot['lambda'], df_lrt) < 1e-3))
    out.append(f"Bootstrap replicates with LRT p < 0.001: {frac_sig:.3%}")
    out.append(f"Bootstrap replicates with STEM R2 > non-STEM R2: "
               f"{np.mean(boot['gap'] > 0):.3%}")

    prng = np.random.default_rng(PERMUTATION_SEED)
    n_stem = int(stem.sum())
    n = len(y)
    perm_gap = np.empty(N_PERMUTATION)
    perm_lambda = np.empty(N_PERMUTATION)
    base = np.zeros(n, bool)
    for b in range(N_PERMUTATION):
        pick = prng.choice(n, n_stem, replace=False)
        m = base.copy()
        m[pick] = True
        lam_p, r2s_p, r2n_p = lrt_from_designs(X1, y, m)
        perm_gap[b] = r2s_p - r2n_p
        perm_lambda[b] = lam_p

    obs_gap = r2_s_obs - r2_n_obs

    p_perm_gap = (np.sum(np.abs(perm_gap) >= abs(obs_gap)) + 1) / (
        N_PERMUTATION + 1)
    p_perm_lambda = (np.sum(perm_lambda >= lam_obs) + 1) / (N_PERMUTATION + 1)

    out.append("")
    out.append(f"LABEL-PERMUTATION TEST ({N_PERMUTATION} permutations of the "
               f"STEM/non-STEM assignment)")
    out.append(f"  Observed R2 gap:      {obs_gap:.6f}")
    out.append(f"  Permutation null gap: mean {perm_gap.mean():.6f}, "
               f"SD {perm_gap.std(ddof=1):.6f}, "
               f"95th pct of |gap| {np.percentile(np.abs(perm_gap), 95):.6f}, "
               f"max |gap| {np.abs(perm_gap).max():.6f}")
    n_exceed = int(np.sum(np.abs(perm_gap) >= abs(obs_gap)))
    out.append(f"  Two-sided p (gap):    {p_perm_gap:.3e} "
               f"({n_exceed} of {N_PERMUTATION} permutations matched or "
               f"exceeded the observed gap)")
    out.append(f"  Observed Lambda:      {lam_obs:.4f}")
    out.append(f"  Permutation null Lambda: mean {perm_lambda.mean():.4f}, "
               f"SD {perm_lambda.std(ddof=1):.4f}, "
               f"max {perm_lambda.max():.4f}")
    out.append(f"  One-sided p (Lambda): {p_perm_lambda:.3e}")

    out.append("")
    out.append("SAMPLE SIZE AND POWER")
    for name, n_grp, r2_grp in (('STEM', int(stem.sum()), r2_s_obs),
                                ('non-STEM', int((~stem).sum()), r2_n_obs),
                                ('Pooled', n, m_pooled.rsquared)):
        f2 = r2_grp / (1 - r2_grp)
        ncp = f2 * (n_grp - k - 1)
        crit = stats.f.ppf(0.95, k, n_grp - k - 1)
        power = float(stats.ncf.sf(crit, k, n_grp - k - 1, ncp))

        lo, hi = 1e-6, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            p_mid = stats.ncf.sf(crit, k, n_grp - k - 1,
                                 mid * (n_grp - k - 1))
            if p_mid < 0.80:
                lo = mid
            else:
                hi = mid
        out.append(f"  {name:<9}: N = {n_grp:>5}, f2 = {f2:.4f}, "
                   f"power at alpha = 0.05 is {power:.4f}; minimum "
                   f"detectable f2 at 80% power is {hi:.5f} "
                   f"(R2 = {hi / (1 + hi):.5f})")

    text = "\n".join(out)
    print(text)
    path = revision_report("results_robust_invariance.txt", out)

    pd.DataFrame({
        'indicator': INDICATORS, 'delta_b': delta, 'se_delta': se_delta,
        'z': z, 'p_raw': p_raw, 'p_holm': p_holm, 'p_bh': p_bh,
        'holm_reject': reject_holm,
    }).to_csv(output_path("mmlu_crossgroup_tests.csv"), index=False)
    pd.DataFrame(boot).to_csv(output_path("mmlu_bootstrap_invariance.csv"),
                              index=False)
    np.save(output_path("mmlu_permutation_gap.npy"), perm_gap)
    print(f"Written: {os.path.basename(path)}, mmlu_crossgroup_tests.csv, "
          "mmlu_bootstrap_invariance.csv, mmlu_permutation_gap.npy")

if __name__ == '__main__':
    main()
