import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from config import RESULTS_DIR, resolve

os.makedirs(RESULTS_DIR, exist_ok=True)

INDICATORS = [
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

CONTRAST_HIGH = 'college_chemistry'
CONTRAST_LOW = 'professional_law'

MIN_ITEMS = 10

def cohens_q(r2_a, r2_b):

    r_a = np.sqrt(r2_a)
    r_b = np.sqrt(r2_b)
    return np.arctanh(r_a) - np.arctanh(r_b), r_a, r_b

def interpret_q(q):

    q_abs = abs(q)
    if q_abs < 0.10:
        return "negligible"
    if q_abs < 0.30:
        return "small"
    if q_abs < 0.50:
        return "medium"
    return "large"

def main():
    print("Initiating Subject-Level Effect Size Analysis")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    final_path = resolve("mmlu_dimension5_adversarial.csv")
    results_path = os.path.join(RESULTS_DIR, "results_summary.txt")

    df = pd.read_csv(final_path)
    df = df.dropna(subset=INDICATORS + ['difficulty_score', 'domain_group'])

    global_r2 = LinearRegression().fit(
        df[INDICATORS], df['difficulty_score']).score(
        df[INDICATORS], df['difficulty_score'])

    rows = []
    for subject, group in df.groupby('subject'):
        if len(group) < MIN_ITEMS:
            continue
        X = group[INDICATORS]
        y = group['difficulty_score']
        r2 = LinearRegression().fit(X, y).score(X, y)
        rows.append({
            'subject': subject,
            'domain_group': group['domain_group'].iloc[0],
            'n_items': len(group),
            'R2': r2
        })

    df_r2 = pd.DataFrame(rows).sort_values(
        by='R2', ascending=False).reset_index(drop=True)

    out_csv = os.path.join(
        os.path.dirname(results_path), "mmlu_subject_r2.csv")
    df_r2.to_csv(out_csv, index=False)

    r2_high = df_r2.loc[df_r2['subject'] == CONTRAST_HIGH, 'R2'].values[0]
    r2_low = df_r2.loc[df_r2['subject'] == CONTRAST_LOW, 'R2'].values[0]
    q_contrast, r_high, r_low = cohens_q(r2_high, r2_low)

    top = df_r2.iloc[0]
    bottom = df_r2.iloc[-1]
    q_extremes, r_top, r_bottom = cohens_q(top['R2'], bottom['R2'])

    report_lines = []
    report_lines.append("")
    report_lines.append(f"Global Pooled R-squared:  {global_r2:.6f}")
    report_lines.append(f"Subjects Analyzed:        {len(df_r2)}")
    report_lines.append("")
    report_lines.append(
        f"{'Subject':<38} | {'Domain':<8} | {'N':>5} | {'R2':>8} |")
    report_lines.append("-" * 70)

    for _, row in df_r2.iterrows():
        report_lines.append(
            f"{row['subject']:<38} | {row['domain_group']:<8} | "
            f"{int(row['n_items']):>5} | {row['R2']:>8.4f} |"
        )

    report_lines.append("")
    report_lines.append("BETWEEN-DOMAIN EFFECT SIZE (COHEN'S q)")
    report_lines.append("")
    report_lines.append(
        "Cohen's q = arctanh(r_1) - arctanh(r_2), where r = sqrt(R-squared).")
    report_lines.append(
        "Conventions (Cohen, 1988): 0.10 small, 0.30 medium, 0.50 large.")
    report_lines.append("")
    report_lines.append(
        f" Exemplar contrast: {CONTRAST_HIGH} vs {CONTRAST_LOW}")
    report_lines.append(
        f"    R2: {r2_high:.4f} vs {r2_low:.4f}  |  r: {r_high:.4f} vs {r_low:.4f}")
    report_lines.append(
        f"    Cohen's q = {q_contrast:.4f} ({interpret_q(q_contrast)})")
    report_lines.append("")
    report_lines.append(
        f" Observed extremes: {top['subject']} vs {bottom['subject']}")
    report_lines.append(
        f"    R2: {top['R2']:.4f} vs {bottom['R2']:.4f}  |  r: {r_top:.4f} vs {r_bottom:.4f}")
    report_lines.append(
        f"    Cohen's q = {q_extremes:.4f} ({interpret_q(q_extremes)})")
    report_lines.append("")

    report_text = "\n".join(report_lines)
    print(report_text)

    with open(results_path, "a", encoding="utf-8") as f:
        f.write("\n\nSUBJECT-LEVEL STRUCTURAL SENSITIVITY & EFFECT SIZE\n\n")
        f.write(report_text + "\n")

    print(f"Subject Effect Size Analysis completed. Table saved to {out_csv}")

if __name__ == '__main__':
    main()
