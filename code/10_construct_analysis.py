import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr, norm
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from config import RESULTS_DIR, resolve

os.makedirs(RESULTS_DIR, exist_ok=True)

def steiger_z(r12, r13, r23, N):

    z12 = np.arctanh(r12)
    z13 = np.arctanh(r13)

    num = r23 * (1.0 - r12**2 - r13**2) - 0.5 * r12 * \
        r13 * (1.0 - r12**2 - r13**2 - r23**2)
    den = (1.0 - r12**2) * (1.0 - r13**2)
    cov = num / den

    z = (z12 - z13) * np.sqrt((N - 3.0) / (2.0 * (1.0 - cov)))
    p_val = norm.sf(np.abs(z)) * 2.0
    return z, p_val

def main():
    print("Initiating Construct Analysis (Reasoning Inversion)")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    aligned_path = resolve("mmlu_aligned.csv")
    dimensions_path = resolve("mmlu_dimension5_adversarial.csv")
    abilities_path = resolve("mmlu_model_abilities.csv")

    out_txt_path = os.path.join(RESULTS_DIR, "construct_analysis_results.txt")
    out_csv_path = os.path.join(RESULTS_DIR, "mmlu_model_profiles.csv")

    df_aligned = pd.read_csv(aligned_path)
    df_dimensions = pd.read_csv(dimensions_path)
    df_abilities = pd.read_csv(abilities_path)

    models = df_abilities['model_name'].tolist()

    df_aligned = df_aligned.set_index('question_id')
    df_dimensions = df_dimensions.set_index('question_id')

    common_idx = df_aligned.index.intersection(df_dimensions.index)
    df_aligned = df_aligned.loc[common_idx]
    df_dimensions = df_dimensions.loc[common_idx]

    stem_mask = df_dimensions['domain_group'] == 'STEM'
    nonstem_mask = df_dimensions['domain_group'] != 'STEM'

    model_profiles = []

    for model_name in models:

        responses = df_aligned[model_name].values
        acc_overall = np.mean(responses)
        acc_stem = np.mean(responses[stem_mask])
        acc_nonstem = np.mean(responses[nonstem_mask])

        X_reason = df_dimensions.loc[stem_mask,
                                     'WSCG_Depth'].values.reshape(-1, 1)
        responses_stem = df_aligned.loc[stem_mask, model_name].values

        unique_classes = np.unique(responses_stem)
        if len(unique_classes) > 1:
            try:
                X_reason_sm = sm.add_constant(X_reason)
                reg_reason = sm.Logit(responses_stem, X_reason_sm).fit(
                    disp=0, method='lbfgs', maxiter=1000)
                beta_reasoning = reg_reason.params[1]
            except Exception:

                beta_reasoning = 0.0
        else:
            beta_reasoning = 0.0

        X_semantic = df_dimensions.loc[stem_mask,
                                       'AMR_Depth'].values.reshape(-1, 1)
        if len(unique_classes) > 1:
            try:
                X_semantic_sm = sm.add_constant(X_semantic)
                reg_sem = sm.Logit(responses_stem, X_semantic_sm).fit(
                    disp=0, method='lbfgs', maxiter=1000)
                beta_semantic = reg_sem.params[1]
            except Exception:
                beta_semantic = 0.0
        else:
            beta_semantic = 0.0

        theta = df_abilities[df_abilities['model_name']
                             == model_name]['theta_score'].values[0]

        model_profiles.append({
            'model_name': model_name,
            'theta_score': theta,
            'accuracy_overall': round(acc_overall, 6),
            'accuracy_stem': round(acc_stem, 6),
            'accuracy_nonstem': round(acc_nonstem, 6),
            'reasoning_sensitivity': round(beta_reasoning, 6),
            'semantic_sensitivity': round(beta_semantic, 6)
        })

    df_profiles = pd.DataFrame(model_profiles)

    r12, _ = pearsonr(
        df_profiles['accuracy_overall'], df_profiles['accuracy_nonstem'])
    r13, _ = pearsonr(
        df_profiles['accuracy_overall'], df_profiles['accuracy_stem'])
    r23, _ = pearsonr(
        df_profiles['accuracy_nonstem'], df_profiles['accuracy_stem'])

    r12_clip = np.clip(r12, -0.999999, 0.999999)
    r13_clip = np.clip(r13, -0.999999, 0.999999)
    r23_clip = np.clip(r23, -0.999999, 0.999999)

    N = len(df_profiles)
    z_steiger, p_steiger = steiger_z(r12_clip, r13_clip, r23_clip, N)

    r_inversion, p_inversion = pearsonr(
        df_profiles['theta_score'], df_profiles['reasoning_sensitivity'])

    df_trim_750 = df_profiles[df_profiles['theta_score'] >= -0.652]
    r_trim_750, p_trim_750 = pearsonr(
        df_trim_750['theta_score'], df_trim_750['reasoning_sensitivity'])

    df_trim_500 = df_profiles[df_profiles['theta_score'] >= -0.072]
    r_trim_500, p_trim_500 = pearsonr(
        df_trim_500['theta_score'], df_trim_500['reasoning_sensitivity'])

    report_lines = []
    report_lines.append("")

    report_lines.append(
        "STEIGER'S DEPENDENT CORRELATION TEST (Raw Accuracy %)")
    report_lines.append(
        f" Correlation: Overall ~ Non-STEM Accuracy (r12): {r12:.6f}")
    report_lines.append(
        f" Correlation: Overall ~ STEM Accuracy (r13):     {r13:.6f}")
    report_lines.append(
        f" Cross-correlation: Non-STEM ~ STEM (r23):       {r23:.6f}")
    report_lines.append(f" Steiger's Z-statistic:  {z_steiger:.4f}")
    report_lines.append(f" Steiger's p-value:      {p_steiger:.3e}")

    report_lines.append("")
    report_lines.append("REASONING STABILITY INVERSION")
    report_lines.append(
        f" Pearson r(theta, reasoning_sensitivity):  r = {r_inversion:.6f}")
    report_lines.append(
        f" p-value:                                  p = {p_inversion:.3e}")

    report_lines.append("")
    report_lines.append("SYSTEMATIC TRIMMING ANALYSIS (FLOOR EFFECT CONTROL)")
    report_lines.append(
        f" Top 75% Models (theta >= -0.652, N={len(df_trim_750)}): r = {r_trim_750:.6f}, p = {p_trim_750:.3e}")
    report_lines.append(
        f" Top 50% Models (theta >= -0.072, N={len(df_trim_500)}): r = {r_trim_500:.6f}, p = {p_trim_500:.3e}")
    report_text = "\n".join(report_lines)

    results_path = os.path.join(RESULTS_DIR, "results_summary.txt")
    with open(results_path, "a", encoding="utf-8") as f:
        f.write("\n\nRETRIEVAL-REASONING CONSTRUCT ANALYSIS\n\n")
        f.write(report_text + "\n")

    df_profiles.to_csv(out_csv_path, index=False)

    print(f"Construct Analysis completed. Results saved to {results_path}")

if __name__ == '__main__':
    main()
