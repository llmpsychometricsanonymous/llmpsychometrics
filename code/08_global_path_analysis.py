import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from config import RESULTS_DIR, resolve

os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    print("Initiating Global Path Analysis")
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

    df = df.dropna(subset=indicators + ['difficulty_score'])
    y = df['difficulty_score'].values
    X = df[indicators].values

    X_design = sm.add_constant(X)

    model = sm.OLS(y, X_design).fit(cov_type='HC3')

    beta = model.params
    se = model.bse
    t_stats = model.tvalues
    p_values = model.pvalues

    std_X = np.std(X, axis=0, ddof=1)
    std_y = np.std(y, ddof=1)
    beta_std = beta[1:] * (std_X / std_y)

    r2 = model.rsquared
    adj_r2 = model.rsquared_adj
    df_reg = model.df_model
    df_res = model.df_resid
    f_stat = model.fvalue
    f_p_val = model.f_pvalue
    rse = np.sqrt(model.mse_resid)
    cohens_f2 = r2 / (1.0 - r2)

    report_lines = []
    report_lines.append("")
    report_lines.append("GLOBAL PATH ANALYSIS RESULTS")
    report_lines.append("")
    report_lines.append("")
    report_lines.append(f"Residual Standard Error (RSE):   {rse:.6f}")
    report_lines.append(f"R-squared:                       {r2:.6f}")
    report_lines.append(f"Adjusted R-squared:              {adj_r2:.6f}")
    report_lines.append(
        f"F-statistic:                     F({int(df_reg)}, {int(df_res)}) = {f_stat:.4f} (p = {f_p_val:.3e})")
    report_lines.append(f"Cohen's f2:                      {cohens_f2:.6f}")
    report_lines.append("")
    report_lines.append(
        f"{'Variable':<30} | {'b (unstd.)':<12} | {'SE':>10} | {'Beta*':>8} | {'t':>10} | {'p':>12} |")
    report_lines.append("-" * 95)

    report_lines.append(
        f"{'(Intercept)':<30} | {beta[0]:<12.6f} | {se[0]:>10.6f} | {'---':>8} | "
        f"{t_stats[0]:>10.4f} | {p_values[0]:>12.3e} |"
    )

    for idx in range(len(indicators)):
        j = idx + 1
        sig = ""
        if p_values[j] < 0.001:
            sig = " ***"
        elif p_values[j] < 0.01:
            sig = " **"
        elif p_values[j] < 0.05:
            sig = " *"

        report_lines.append(
            f"{indicators[idx]:<30} | {beta[j]:<12.6f} | {se[j]:>10.6f} | {beta_std[idx]:>8.4f} | "
            f"{t_stats[j]:>10.4f} | {p_values[j]:>12.3e} |{sig}"
        )

    report_lines.append("")
    report_lines.append(
        "Significance codes: *** p < 0.001, ** p < 0.01, * p < 0.05")
    report_lines.append("")

    report_text = "\n".join(report_lines)

    results_path = os.path.join(RESULTS_DIR, "results_summary.txt")

    print(f"Global Path Analysis completed. Results saved to {results_path}")

    with open(results_path, "a", encoding="utf-8") as f:
        f.write("\n\n")
        f.write(report_text.strip() + "\n")

if __name__ == '__main__':
    main()
