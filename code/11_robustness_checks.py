import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr, weightedtau
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, autoguide
import random as python_random
import warnings
from config import RESULTS_DIR, resolve

os.makedirs(RESULTS_DIR, exist_ok=True)

def irt_2pl_model(responses):
    num_models, num_items = responses.shape
    theta = numpyro.sample("theta", dist.Normal(0.0, 1.0).expand([num_models]))
    b = numpyro.sample("b", dist.Normal(0.0, 1.0).expand([num_items]))
    a = numpyro.sample("a", dist.HalfNormal(1.0).expand([num_items]))

    logits = a * (theta[:, None] - b)

    with numpyro.plate("models", num_models, dim=-2):
        with numpyro.plate("items", num_items, dim=-1):
            numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=responses)

def get_b_est(responses, prng_seed):
    guide = autoguide.AutoDiagonalNormal(irt_2pl_model)
    optimizer = numpyro.optim.Adam(step_size=0.01)
    svi = SVI(irt_2pl_model, guide, optimizer, loss=Trace_ELBO())

    svi_result = svi.run(random.PRNGKey(prng_seed),
                         num_steps=5000, responses=responses)
    medians = guide.median(svi_result.params)
    return np.array(medians['b'])

def main():
    warnings.filterwarnings('ignore')
    print("Initiating Robustness Checks (Collinearity, IRT & Rank Invariance)")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_file_path = os.path.join(RESULTS_DIR, "results_summary.txt")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    report_lines = []
    report_lines.append(
        "\n\nROBUSTNESS CHECKS (IRT INVARIANCE, COLLINEARITY & RANK INVARIANCE)\n\n")

    final_path = resolve("mmlu_dimension5_adversarial.csv")
    df = pd.read_csv(final_path)

    indicators = [
        'WSCG_Depth', 'WSCG_Nodes', 'Syntactic_MDD', 'Syntactic_Depth',
        'Knowledge_Zipf_Rarity', 'Knowledge_NER_Density', 'Semantic_Concreteness',
        'AMR_Depth', 'Adversarial_Score'
    ]

    print("\nFEATURE COLLINEARITY & VIF\n")
    report_lines.append("\nFEATURE COLLINEARITY & VIF\n")
    X_features = df[indicators].dropna()

    X_with_const = sm.add_constant(X_features)
    vif_data = pd.DataFrame()
    vif_data["Indicator"] = X_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(
        X_with_const.values, i) for i in range(X_with_const.shape[1])]
    vif_data = vif_data[vif_data["Indicator"] != "const"]

    vif_str = "\nVariance Inflation Factors (VIF):\n" + \
        vif_data.round(3).to_string()
    print(vif_str)
    report_lines.append(vif_str)

    print("\nIRT PARAMETER INVARIANCE & CROSS-COHORT STABILITY\n")
    report_lines.append(
        "\nIRT PARAMETER INVARIANCE & CROSS-COHORT STABILITY\n")
    metadata_cols = ['question_id', 'item_id', 'subject', 'question_text',
                     'clean_text', 'choices', 'ground_truth', 'domain_group']
    aligned_path = resolve("mmlu_aligned.csv")
    df_aligned = pd.read_csv(aligned_path)

    model_cols = [
        col for col in df_aligned.columns if col not in metadata_cols and col not in [
            'difficulty_score',
            'discrimination_score']]

    np.random.seed(1729)
    python_random.seed(1729)

    models_half_1 = np.random.choice(
        model_cols, size=len(model_cols) // 2, replace=False)
    models_half_2 = [m for m in model_cols if m not in models_half_1]

    resp_1 = df_aligned[models_half_1].values.T.astype(float)
    resp_2 = df_aligned[models_half_2].values.T.astype(float)

    b_est_1 = get_b_est(resp_1, 1729)
    b_est_2 = get_b_est(resp_2, 1729)

    r_invariance, p_val = pearsonr(b_est_1, b_est_2)

    inv_str = f"\nPearson Correlation of Item Difficulty (b) between Cohort 1 ({len(models_half_1)} models) and Cohort 2 ({len(models_half_2)} models): {r_invariance:.4f} (p = {p_val:.2e})\n"
    print(inv_str)
    report_lines.append(inv_str)

    print("\nCTT RANK INVARIANCE (WEIGHTED KENDALL'S TAU)\n")
    report_lines.append("\nCTT RANK INVARIANCE (WEIGHTED KENDALL'S TAU)\n")

    profiles_path = resolve("mmlu_model_profiles.csv")
    df_profiles = pd.read_csv(profiles_path)

    df_profiles['rank_overall'] = df_profiles['accuracy_overall'].rank(
        ascending=False, method='min')
    df_profiles['rank_stem'] = df_profiles['accuracy_stem'].rank(
        ascending=False, method='min')
    df_profiles['rank_nonstem'] = df_profiles['accuracy_nonstem'].rank(
        ascending=False, method='min')
    df_profiles = df_profiles.sort_values('rank_overall')

    wtau_o_ns = weightedtau(
        df_profiles['rank_overall'], df_profiles['rank_nonstem']).correlation
    wtau_o_s = weightedtau(
        df_profiles['rank_overall'], df_profiles['rank_stem']).correlation

    tau_str = f"\nOverall vs Non-STEM: Weighted Kendall tau_w = {wtau_o_ns:.4f}"
    tau_str += f"\nOverall vs STEM:     Weighted Kendall tau_w = {wtau_o_s:.4f}\n"
    print(tau_str)
    report_lines.append(tau_str)

    print("\nTOP-K MODEL SELECTION OVERLAP\n")
    report_lines.append("\nTOP-K MODEL SELECTION OVERLAP\n")

    top_10_overall = set(df_profiles.nsmallest(
        10, 'rank_overall')['model_name'])
    top_10_stem = set(df_profiles.nsmallest(10, 'rank_stem')['model_name'])
    top_10_nonstem = set(df_profiles.nsmallest(
        10, 'rank_nonstem')['model_name'])

    overlap_10_stem = len(top_10_overall.intersection(top_10_stem))
    overlap_10_nonstem = len(top_10_overall.intersection(top_10_nonstem))

    top10_str = "\nTop 10 Models:"
    top10_str += f"\n  - Overlap with Top 10 Non-STEM: {overlap_10_nonstem/10*100:.1f}% ({overlap_10_nonstem}/10)"
    top10_str += f"\n  - Overlap with Top 10 STEM:     {overlap_10_stem/10*100:.1f}% ({overlap_10_stem}/10) [Missed {10 - overlap_10_stem} model(s)]"
    print(top10_str)
    report_lines.append(top10_str)

    top_50_overall = set(df_profiles.nsmallest(
        50, 'rank_overall')['model_name'])
    top_50_stem = set(df_profiles.nsmallest(50, 'rank_stem')['model_name'])
    top_50_nonstem = set(df_profiles.nsmallest(
        50, 'rank_nonstem')['model_name'])

    overlap_50_stem = len(top_50_overall.intersection(top_50_stem))
    overlap_50_nonstem = len(top_50_overall.intersection(top_50_nonstem))

    top50_str = "Top 50 Models:"
    top50_str += f"\n  - Overlap with Top 50 Non-STEM: {overlap_50_nonstem/50*100:.1f}% ({overlap_50_nonstem}/50)"
    top50_str += f"\n  - Overlap with Top 50 STEM:     {overlap_50_stem/50*100:.1f}% ({overlap_50_stem}/50) [Yielding a {100 - (overlap_50_stem/50*100):.1f}% false-positive rate]"
    print(top50_str)
    report_lines.append(top50_str)

    print("\nUNINTERRUPTED RANK DISPLACEMENT (TOP-40 CANDIDATE POOL)\n")
    report_lines.append("\nUNINTERRUPTED RANK DISPLACEMENT (TOP-40 CANDIDATE POOL)\n")

    df_profiles['displacement'] = df_profiles['rank_overall'] - df_profiles['rank_stem']
    table_df = df_profiles[df_profiles['rank_overall'] <= 40].sort_values('rank_overall').copy()
    table_df['ctt_gap'] = (-table_df['accuracy_overall'].diff() * 100).fillna(0.0)

    header_str = f"{'Model Name':<45} | {'Overall':<7} | {'Non-STEM':<8} | {'STEM':<6} | {'Displacement':<12} | {'CTT Gap (%)':<11}\n" + "-"*106
    table_lines = [header_str]
    for _, row in table_df.iterrows():
        disp_val = int(row['displacement'])
        disp_str = f"{disp_val:>+12d}"
        gap_val = row['ctt_gap']
        gap_str = "-" if gap_val < 0.005 else f"{gap_val:>10.2f}%"
        line = f"{row['model_name'][:45]:<45} | {int(row['rank_overall']):>7} | {int(row['rank_nonstem']):>8} | {int(row['rank_stem']):>6} | {disp_str} | {gap_str}"
        table_lines.append(line)

    table_full_str = "\n".join(table_lines)
    print(table_full_str)
    report_lines.append(table_full_str)

    with open(output_file_path, "a", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(
        f"\nRobustness Checks completed. Results saved to {output_file_path}")

if __name__ == '__main__':
    main()
