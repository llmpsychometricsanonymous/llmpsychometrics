import pandas as pd
import numpy as np
import os
from scipy.stats import chi2, norm
import statsmodels.api as sm
from config import RESULTS_DIR, resolve

os.makedirs(RESULTS_DIR, exist_ok=True)

def fit_ols(X, y):
    n = len(y)
    X_design = sm.add_constant(X)
    p = X_design.shape[1]

    model = sm.OLS(y, X_design).fit(cov_type='HC3')
    beta = model.params
    se = model.bse
    r2 = model.rsquared

    rss = model.ssr
    tss = model.centered_tss

    std_X = np.std(X, axis=0, ddof=1)
    std_y = np.std(y, ddof=1)
    beta_std = beta[1:] * (std_X / std_y)

    log_lik = model.llf

    return {
        'beta': beta,
        'se': se,
        'beta_std': beta_std,
        'rss': rss,
        'tss': tss,
        'r2': r2,
        'n': n,
        'p': p,
        'log_lik': log_lik
    }

def main():
    print("Initiating Domain Invariance Testing")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    final_path = resolve("mmlu_dimension5_adversarial.csv")

    df = pd.read_csv(final_path)

    indicators = [
        'WSCG_Depth',
        'WSCG_Nodes',
        'Syntactic_MDD',
        'Syntactic_Depth',
        'Knowledge_Zipf_Rarity',
        'Knowledge_NER_Density',
        'Semantic_Concreteness',
        'AMR_Depth',
        'Adversarial_Score'
    ]

    df = df.dropna(subset=indicators + ['difficulty_score', 'domain_group'])

    df_stem = df[df['domain_group'] == 'STEM'].copy()
    df_nonstem = df[df['domain_group'] != 'STEM'].copy()

    y_pooled = df['difficulty_score'].values
    X_pooled = df[indicators].values
    model_pooled = fit_ols(X_pooled, y_pooled)

    y_stem = df_stem['difficulty_score'].values
    X_stem = df_stem[indicators].values
    model_stem = fit_ols(X_stem, y_stem)

    y_nonstem = df_nonstem['difficulty_score'].values
    X_nonstem = df_nonstem[indicators].values
    model_nonstem = fit_ols(X_nonstem, y_nonstem)

    log_lik_restricted = model_pooled['log_lik']

    log_lik_unrestricted = model_stem['log_lik'] + model_nonstem['log_lik']

    lambda_lrt = 2.0 * (log_lik_unrestricted - log_lik_restricted)

    df_lrt = model_stem['p'] + model_nonstem['p'] - model_pooled['p'] + 1
    p_val_lrt = chi2.sf(lambda_lrt, df=df_lrt)

    z_scores = []
    p_vals_diff = []

    for j in range(1, len(indicators) + 1):
        beta_s = model_stem['beta'][j]
        beta_ns = model_nonstem['beta'][j]
        se_s = model_stem['se'][j]
        se_ns = model_nonstem['se'][j]
        z = (beta_s - beta_ns) / np.sqrt(se_s**2 + se_ns**2)
        p_val = norm.sf(np.abs(z)) * 2.0

        z_scores.append(z)
        p_vals_diff.append(p_val)

    report_lines = []
    report_lines.append("")
    report_lines.append(f"STEM Group Size (N1):        {len(df_stem)}")
    report_lines.append(f"Non-STEM Group Size (N2):    {len(df_nonstem)}")
    report_lines.append(f"STEM R-squared:              {model_stem['r2']:.6f}")
    report_lines.append(
        f"Non-STEM R-squared:          {model_nonstem['r2']:.6f}")
    report_lines.append(
        f"Restricted Log-Likelihood:   {log_lik_restricted:.4f} (Pooled Model)")
    report_lines.append(
        f"Unrestricted Log-Likelihood: {log_lik_unrestricted:.4f} (STEM + Non-STEM Models)")
    report_lines.append(
        f"Likelihood Ratio Test (LRT): Chi2 = {
            lambda_lrt:.4f} on {df_lrt} DF")
    report_lines.append(f"LRT p-value:                 {p_val_lrt:.3e}")

    report_lines.append("")
    report_lines.append(
        f"{'Indicator':<28} | {'STEM b':<10} | {'STEM Beta*':>10} | "
        f"{'non-STEM b':<12} | {'NS Beta*':>10} | {'Z-stat':>14} |"
    )
    report_lines.append("-" * 94)

    for idx, f_name in enumerate(indicators):
        b_s = model_stem['beta'][idx + 1]
        bs_std = model_stem['beta_std'][idx]
        b_ns = model_nonstem['beta'][idx + 1]
        bns_std = model_nonstem['beta_std'][idx]
        z_val = z_scores[idx]
        p_diff = p_vals_diff[idx]

        sig = ""
        if p_diff < 0.001:
            sig = "***"
        elif p_diff < 0.01:
            sig = "**"
        elif p_diff < 0.05:
            sig = "*"

        z_str = f"{z_val:.4f} {sig}".strip()

        report_lines.append(
            f"{f_name:<28} | {b_s:<10.4f} | {bs_std:>10.4f} | "
            f"{b_ns:<12.4f} | {bns_std:>10.4f} | {z_str:>14} |"
        )

    report_lines.append("")
    report_lines.append(
        "Significance codes: *** p < 0.001, ** p < 0.01, * p < 0.05")

    report_lines.append("")

    report_text = "\n".join(report_lines)

    results_path = os.path.join(RESULTS_DIR, "results_summary.txt")
    with open(results_path, "a", encoding="utf-8") as f:
        f.write("\n\nDOMAIN INVARIANCE (STEM vs Non-STEM)\n\n")
        f.write(report_text + "\n")

    print(
        f"Domain Invariance Testing completed. Results saved to {results_path}")

if __name__ == '__main__':
    main()
