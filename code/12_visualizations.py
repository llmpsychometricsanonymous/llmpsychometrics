import glob
import matplotlib.font_manager as fm
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from scipy.stats import pearsonr
import matplotlib.patches as mpatches
from config import RESULTS_DIR, resolve

os.makedirs(RESULTS_DIR, exist_ok=True)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_dir = os.path.join(RESULTS_DIR, 'plots', 'superseded')
os.makedirs(out_dir, exist_ok=True)
font_dir = os.path.join(base_dir, 'resources', 'fonts')
if os.path.exists(font_dir):
    for _f in glob.glob(os.path.join(font_dir, '*.ttf')):
        fm.fontManager.addfont(_f)
plt.rcParams.update({'axes.unicode_minus': False, 'font.family': 'Noto Serif', 'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16, 'axes.titleweight': 'bold', 'figure.dpi': 600,
                    'savefig.dpi': 600, 'savefig.bbox': 'tight', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Noto Serif', 'mathtext.it': 'Noto Serif:italic', 'mathtext.bf': 'Noto Serif:bold',
                     'pdf.fonttype': 42, 'ps.fonttype': 42})

def generate_bifurcation_plot():
    print('\nGenerating Figure 1: Subject Bifurcation Plot')
    df_path = resolve('mmlu_dimension5_adversarial.csv')
    df = pd.read_csv(df_path)
    indicators = ['WSCG_Depth', 'WSCG_Nodes', 'Syntactic_MDD', 'Syntactic_Depth', 'Knowledge_Zipf_Rarity',
                  'Knowledge_NER_Density', 'Semantic_Concreteness', 'AMR_Depth', 'Adversarial_Score']
    df = df.dropna(subset=indicators + ['difficulty_score', 'domain_group'])
    results = []
    for subject, group in df.groupby('subject'):
        if len(group) < 10:
            continue
        X = group[indicators]
        y = group['difficulty_score']
        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)
        domain = group['domain_group'].iloc[0]
        results.append({'subject': subject, 'R2': r2, 'domain': domain})
    res_df = pd.DataFrame(results).sort_values(by='R2', ascending=True)
    res_df['subject'] = res_df['subject'].str.replace('_', ' ').str.title()
    res_df['subject'] = res_df['subject'].str.replace(
        '\\bUs\\b', 'US', regex=True)
    res_df = res_df.reset_index(drop=True)
    X_glob = df[indicators]
    y_glob = df['difficulty_score']
    glob_r2 = LinearRegression().fit(X_glob, y_glob).score(X_glob, y_glob)
    fig, ax = plt.subplots(figsize=(12, 16))
    for i in range(len(res_df)):
        if i % 2 == 1:
            ax.axhspan(i - 0.5, i + 0.5, color='#f4f6f7', zorder=0)
    max_stem = res_df[res_df['domain'] == 'STEM']['R2'].max()
    max_hum = res_df[res_df['domain'] != 'STEM']['R2'].max()

    def interpolate_color(c1, c2, frac):
        rgb1 = np.array(mcolors.to_rgb(c1))
        rgb2 = np.array(mcolors.to_rgb(c2))
        return rgb1 + frac * (rgb2 - rgb1)
    colors = []
    for _, row in res_df.iterrows():
        val = row['R2']
        if row['domain'] == 'STEM':
            c = interpolate_color('#e99293', '#d62728', val / max_stem)
        else:
            c = interpolate_color('#7fb2d5', '#1f77b4', val / max_hum)
        colors.append(c)
    bars = ax.barh(res_df.index, res_df['R2'],
                   color=colors, height=0.7, zorder=2)
    for i, row in res_df.iterrows():
        val = row['R2']
        if val >= glob_r2:
            ax.text(val + 0.003, i, f'{val:.4f}', va='center',
                    fontsize=10, weight='normal', color='#2c3e50')
    ax.set_yticks(res_df.index)
    ax.set_yticklabels(res_df['subject'], fontsize=12)
    ax.set_ylim(-0.5, len(res_df) + 1)
    ax.spines['top'].set_visible(True)
    ax.spines['top'].set_color('#cccccc')
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    ax.tick_params(axis='y', length=4, width=1, color='#2c3e50', pad=5)
    ax.grid(True, axis='x', color='#eaeaea', linestyle='-', zorder=1)
    ax.axvline(glob_r2, color='#7f8c8d', linestyle='--', linewidth=1, zorder=3)
    y_idx = res_df[res_df['R2'] < glob_r2].index.max()
    ax.text(glob_r2 + 0.0006, y_idx - 6.0,
            f'Global Pooled R² ({glob_r2:.3f})', va='center', fontsize=12, color='#34495e')
    ax.set_xlabel('OLS Multiple Coefficient of Determination (R²)',
                  fontweight='bold', labelpad=10)
    n_stem = len(res_df[res_df['domain'] == 'STEM'])
    n_hum = len(res_df) - n_stem
    stem_patch = mpatches.Patch(
        color='#d62728', label=f'STEM Domains (N = {n_stem})')
    hum_patch = mpatches.Patch(
        color='#1f77b4', label=f'Humanities & Social Sciences (N = {n_hum})')
    glob_line = plt.Line2D([0], [0], color='#7f8c8d',
                           linestyle='--', linewidth=1, label=f'Global Pooled R²')
    legend = ax.legend(handles=[stem_patch, hum_patch, glob_line], loc='lower right', bbox_to_anchor=(
        0.96, 0.03), frameon=True, shadow=False, fontsize=12, borderpad=1)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    ax.set_xlim(0, res_df['R2'].max() + 0.05)
    plt.tight_layout()
    out_path_pdf = os.path.join(out_dir, '1_subject_bifurcation.pdf')
    plt.savefig(out_path_pdf)
    out_path_svg = os.path.join(out_dir, '1_subject_bifurcation.svg')
    plt.savefig(out_path_svg)
    out_path_png = os.path.join(out_dir, '1_subject_bifurcation.png')
    plt.savefig(out_path_png, dpi=300)
    print(f'Saved 1_subject_bifurcation to PDF, SVG and PNG')
    plt.close(fig)


def generate_stability_inversion_plot():
    print('\nGenerating Figure 3: Stability Inversion Plot')
    df_path = resolve('mmlu_model_profiles.csv')
    df_profiles = pd.read_csv(df_path)
    df_profiles = df_profiles.dropna(
        subset=['theta_score', 'reasoning_sensitivity', 'accuracy_overall'])
    theta_std = (df_profiles['theta_score'] -
                 df_profiles['theta_score'].mean()) / df_profiles['theta_score'].std()
    sens_std = (df_profiles['reasoning_sensitivity'] -
                df_profiles['reasoning_sensitivity'].mean()) / df_profiles['reasoning_sensitivity'].std()
    fig, ax = plt.subplots(figsize=(13, 8))
    acc_min = df_profiles['accuracy_overall'].min()
    acc_max = df_profiles['accuracy_overall'].max()
    norm = plt.Normalize(acc_min + 0.01, acc_max - 0.01)
    sc = ax.scatter(theta_std, sens_std, c=df_profiles['accuracy_overall'],
                    cmap='coolwarm', norm=norm, s=82, edgecolors='#0e141a', linewidths=0.7, zorder=3)
    cmap = plt.get_cmap('coolwarm')
    face_colors = cmap(norm(df_profiles['accuracy_overall'].values))
    face_colors[:, 3] = 0.92
    sc.set_facecolor(face_colors)
    sc.set_edgecolor('#0e141a')
    grid_std = np.linspace(theta_std.min(), theta_std.max(), 300)
    coef = np.polyfit(theta_std, sens_std, 1)
    pred_std = np.polyval(coef, grid_std)
    ax.plot(grid_std, pred_std, color='#2c3e50', linestyle='-',
            linewidth=2.8, zorder=1, label='OLS Linear Trend')
    cbar = fig.colorbar(sc, ax=ax)
    cbar.solids.set_edgecolor('face')
    cbar.solids.set_linewidth(0.5)
    cbar.set_label('Empirical MMLU Overall Accuracy',
                   rotation=270, labelpad=15, weight='bold')
    TEXT_COLOR = '#2d3436'
    ARROW_COLOR = '#95a5a6'

    def get_std_coords(model_name):
        row = df_profiles[df_profiles['model_name'] == model_name]
        if row.empty:
            return (None, None)
        t = (row['theta_score'].values[0] - df_profiles['theta_score'].mean()
             ) / df_profiles['theta_score'].std()
        s = (row['reasoning_sensitivity'].values[0] - df_profiles['reasoning_sensitivity'].mean()
             ) / df_profiles['reasoning_sensitivity'].std()
        return (t, s)
    ANNOTATIONS = [
        dict(model='Qwen/Qwen2-72B', label='Qwen2-72B',
             dx=-0.2, dy=0.7, ha='right', va='bottom'),
        dict(model='adamo1139/Yi-34B-200K-HESOYAM-0905',
             label='Yi-34B HESOYAM', dx=-0.03, dy=-1.0, ha='right', va='top'),
        dict(model='tokyotech-llm/Swallow-70b-instruct-hf',
             label='Swallow-70B', dx=0.4, dy=0.5, ha='left', va='bottom'),
        dict(model='OpenBuddy/openbuddy-mixtral-7bx8-v17.2-32k',
             label='OpenBuddy Mixtral', dx=0.7, dy=0.5, ha='left', va='bottom'),
        dict(model='Panchovix/airoboros-33b-gpt4-1.2-SuperHOT-8k',
             label='Airoboros-33B', dx=0.8, dy=0.3, ha='left', va='bottom'),
        dict(model='Open-Orca/Mixtral-SlimOrca-8x7B',
             label='Mixtral-SlimOrca-8×7B', dx=0.0, dy=-0.8, ha='left', va='top')
    ]
    arrow_kw = dict(arrowstyle='->', color=ARROW_COLOR,
                    lw=0.85, connectionstyle='arc3,rad=0.0')
    for ann in ANNOTATIONS:
        tx, ty = get_std_coords(ann['model'])
        if tx is None:
            continue
        ax.annotate(ann['label'], xy=(tx, ty), xytext=(tx + ann['dx'], ty + ann['dy']), xycoords='data', textcoords='data',
                    fontsize=8.5, color=TEXT_COLOR, ha=ann['ha'], va=ann['va'], style='italic', arrowprops=arrow_kw, zorder=5)
    r_val, p_val = pearsonr(theta_std, sens_std)
    p_sci = f'{p_val:.3e}'
    base, exp = p_sci.split('e')

    def to_superscript(num_str):
        sups = {'-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³',
                '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'}
        return ''.join(sups.get(c, c) for c in num_str)
    exp_str = to_superscript(str(int(exp)))
    r_str = f'-{abs(r_val):.4f}' if r_val < 0 else f'{r_val:.4f}'
    stats_text = (
        r'$\mathbf{Correlation\ Analysis}$' + '\n' +
        f'Pearson $r$ = {r_str}\n' +
        f'$p$ = {base} \u00d7 10{exp_str}\n' +
        r'$\mathbf{(Extreme\ Inversion)}$'
    )
    props = dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa',
                 alpha=0.95, edgecolor='#bdc3c7', linewidth=1.5)
    ax.text(0.03, 0.03, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='left', bbox=props, linespacing=1.6)
    ax.set_xlabel('Standardized Model Latent Ability ($\\theta$)',
                  fontweight='bold')
    ax.set_ylabel(
        'Standardized Reasoning Sensitivity Slope ($\\beta_{rj}$)', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.grid(True, linestyle='--', alpha=0.35, color='#888888', zorder=0)
    legend = ax.legend(loc='upper right', frameon=True,
                       shadow=False, fontsize=12, borderpad=1)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    out_path_pdf = os.path.join(out_dir, '3_stability_inversion.pdf')
    plt.savefig(out_path_pdf)
    out_path_svg = os.path.join(out_dir, '3_stability_inversion.svg')
    plt.savefig(out_path_svg)
    out_path_png = os.path.join(out_dir, '3_stability_inversion.png')
    plt.savefig(out_path_png, dpi=300)
    print('Saved 3_stability_inversion to PDF, SVG and PNG')
    plt.close(fig)

if __name__ == '__main__':
    generate_bifurcation_plot()
    generate_stability_inversion_plot()
    print('\nVisualizations successfully saved to the plots directory!')
