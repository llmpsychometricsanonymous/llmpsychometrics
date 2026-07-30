import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BASE_DIR, INDICATORS, RESULTS_DIR, resolve

OUT_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(OUT_DIR, exist_ok=True)

for _font in glob.glob(os.path.join(BASE_DIR, 'resources', 'fonts', '*.ttf')):
    fm.fontManager.addfont(_font)

plt.rcParams.update({
    'axes.unicode_minus': False, 'font.family': 'Noto Serif', 'font.size': 12,
    'axes.labelsize': 14, 'axes.titlesize': 15, 'axes.titleweight': 'bold',
    'figure.dpi': 600, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Noto Serif',
    'mathtext.it': 'Noto Serif:italic', 'mathtext.bf': 'Noto Serif:bold',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

STEM_RED = '#d62728'
STEM_RED_LIGHT = '#e99293'
HUM_BLUE = '#1f77b4'
HUM_BLUE_LIGHT = '#7fb2d5'

def save(fig, stem):
    for ext, kw in (('pdf', {}), ('svg', {}), ('png', {'dpi': 300})):
        fig.savefig(os.path.join(OUT_DIR, f'{stem}.{ext}'), **kw)
    plt.close(fig)
    print(f'  saved {stem} (pdf/svg/png)')

def prettify(name):
    label = name.replace('_', ' ').title()
    return label.replace('Us ', 'US ').replace(' Us', ' US')

def load_frames():
    df = pd.read_csv(resolve('mmlu_dimension5_adversarial.csv'))
    df = df.dropna(subset=INDICATORS + ['difficulty_score', 'domain_group'])
    subj = pd.read_csv(resolve('mmlu_subject_r2_crossvalidated.csv'))
    return df, subj

def figure1_bifurcation(df, subj):
    print('Figure 1: subject bifurcation (adjusted R-squared)')
    res = subj[subj['reported']].copy()
    res = res.sort_values('R2_adjusted').reset_index(drop=True)
    res['label'] = res['subject'].map(prettify)

    y = df['difficulty_score'].values
    X = sm.add_constant(df[INDICATORS].values)
    stem = (df['domain_group'] == 'STEM').values
    pooled_adj = sm.OLS(y, X).fit().rsquared_adj
    stem_adj = sm.OLS(y[stem], X[stem]).fit().rsquared_adj
    ns_adj = sm.OLS(y[~stem], X[~stem]).fit().rsquared_adj

    fig, ax = plt.subplots(figsize=(12, 16))
    for i in range(len(res)):
        if i % 2 == 1:
            ax.axhspan(i - 0.5, i + 0.5, color='#e3e8ea', zorder=0)

    is_stem = res['domain_group'].eq('STEM')
    span_stem = max(res.loc[is_stem, 'R2_adjusted'].max(), 1e-9)
    span_hum = max(res.loc[~is_stem, 'R2_adjusted'].max(), 1e-9)

    def blend(c1, c2, frac):
        import matplotlib.colors as mcolors
        a, b = np.array(mcolors.to_rgb(c1)), np.array(mcolors.to_rgb(c2))
        return a + np.clip(frac, 0, 1) * (b - a)

    colors = [
        blend(STEM_RED_LIGHT, STEM_RED, row.R2_adjusted / span_stem)
        if row.domain_group == 'STEM'
        else blend(HUM_BLUE_LIGHT, HUM_BLUE, row.R2_adjusted / span_hum)
        for row in res.itertuples()
    ]
    ax.barh(res.index, res['R2_adjusted'], color=colors, height=0.7, zorder=2)

    for i, row in res.iterrows():
        if row['R2_adjusted'] >= pooled_adj:
            ax.text(row['R2_adjusted'] + 0.003, i, f"{row['R2_adjusted']:.4f}",
                    va='center', fontsize=10, color='#2c3e50')

    ax.set_yticks(res.index)
    ax.set_yticklabels(res['label'], fontsize=12)
    ax.set_ylim(-0.5, len(res) + 2)
    ax.set_xlim(min(0.0, res['R2_adjusted'].min() - 0.01),
                res['R2_adjusted'].max() + 0.05)
    ax.grid(True, axis='x', color='#eaeaea', linestyle='-', zorder=1)
    for side, visible in (('top', True), ('right', False), ('left', False),
                          ('bottom', True)):
        ax.spines[side].set_visible(visible)
        ax.spines[side].set_color('#cccccc')
    ax.tick_params(axis='y', length=4, width=1, color='#2c3e50', pad=5)

    for value, colour, style, text, offset in (
            (stem_adj, STEM_RED, ':', f'STEM partition ({stem_adj:.3f})', 1.4),
            (ns_adj, HUM_BLUE, ':',
             f'non-STEM partition ({ns_adj:.3f})', 0.55),
            (pooled_adj, '#7f8c8d', '--', f'pooled ({pooled_adj:.3f})',
             -0.3)):
        ax.axvline(value, color=colour, linestyle=style, linewidth=1.4,
                   zorder=3)
        ax.annotate(text, xy=(value, len(res) + offset),
                    xytext=(6, 0), textcoords='offset points', fontsize=10,
                    color=colour, va='center', ha='left')

    n_stem = int(is_stem.sum())
    handles = [
        mpatches.Patch(color=STEM_RED, label=f'STEM subjects (N = {n_stem})'),
        mpatches.Patch(color=HUM_BLUE,
                       label=f'Humanities & social sciences '
                             f'(N = {len(res) - n_stem})'),
        plt.Line2D([0], [0], color='#7f8c8d', linestyle='--',
                   label='Pooled adjusted $R^2$'),
        plt.Line2D([0], [0], color=STEM_RED, linestyle=':',
                   label='Partition-level adjusted $R^2$'),
    ]
    legend = ax.legend(handles=handles, loc='lower right',
                       bbox_to_anchor=(0.97, 0.03), frameon=True, fontsize=11,
                       borderpad=1)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    ax.set_xlabel('Adjusted multiple coefficient of determination '
                  r'($R^2_{\mathrm{adj}}$)', fontweight='bold', labelpad=10)
    fig.tight_layout()
    save(fig, '1_subject_bifurcation')

def superscript(num_str):
    table = {'-': '⁻', '0': '⁰', '1': '¹', '2': '²',
             '3': '³', '4': '⁴', '5': '⁵', '6': '⁶',
             '7': '⁷', '8': '⁸', '9': '⁹'}
    return ''.join(table.get(ch, ch) for ch in num_str)


def stats_caption(r, p, verdict):
    base, exponent = f'{p:.3e}'.split('e')
    return (r'$\mathbf{Correlation\ Analysis}$' + '\n'
            + f'Pearson $r$ = {r:.4f}\n'
            + f'$p$ = {base} × 10{superscript(str(int(exponent)))}\n'
            + r'$\mathbf{(' + verdict + r')}$')


def frame_legend(legend):
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')
    return legend


def figure3_inversion(profiles):
    print('Figure 3: stability inversion, full sample and floor-corrected')
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.9), sharey=True,
                             gridspec_kw={'wspace': 0.055})

    acc = profiles['accuracy_overall']
    norm = plt.Normalize(acc.min() + 0.01, acc.max() - 0.01)
    cmap = plt.get_cmap('coolwarm')
    cutoff = profiles['theta_score'].median()
    trimmed = profiles[profiles['theta_score'] >= cutoff]

    # The fitted line is extrapolated to the extreme theta values, so the
    # y-limits have to cover it as well as the points, or it exits the panel.
    extremes = [profiles['reasoning_sensitivity'].min(),
                profiles['reasoning_sensitivity'].max()]
    for frame in (profiles, trimmed):
        th = frame['theta_score'].values
        fitted = np.polyval(
            np.polyfit(th, frame['reasoning_sensitivity'].values, 1),
            [th.min(), th.max()])
        extremes += list(fitted)
    lo, hi = min(extremes), max(extremes)
    pad = 0.06 * (hi - lo)
    ylim = (lo - pad, hi + pad)

    panels = [
        (axes[0], profiles, '(a)  All 1,000 models',
         'Extreme\\ Inversion', 'lower left', (0.03, 0.04)),
        (axes[1], trimmed,
         r'(b)  Floor-corrected: $\theta_j \geq$ ' + f'{cutoff:.3f}',
         'Floor\\ Corrected', 'upper left', (0.03, 0.96)),
    ]

    trend_handle = []
    for ax, data, title, verdict, box_loc, box_xy in panels:
        theta = data['theta_score'].values
        slope = data['reasoning_sensitivity'].values
        r, p = stats.pearsonr(theta, slope)

        faces = cmap(norm(data['accuracy_overall'].values))
        faces[:, 3] = 0.88
        ax.scatter(theta, slope, s=34, facecolors=faces,
                   edgecolors='#0e141a', linewidths=0.4, zorder=3)

        grid = np.linspace(theta.min(), theta.max(), 300)
        trend, = ax.plot(grid, np.polyval(np.polyfit(theta, slope, 1), grid),
                         color='#2c3e50', linestyle='-', linewidth=3.0,
                         zorder=4, solid_capstyle='round',
                         label='OLS Linear Trend')
        trend_handle.append(trend)

        ax.set_title(title, fontsize=17, pad=28, loc='left')
        ax.set_xlabel(r'Model Latent Ability ($\theta_j$)', fontweight='bold',
                      labelpad=8)
        ax.set_ylim(*ylim)
        ax.margins(x=0.06)
        ax.grid(True, linestyle='--', alpha=0.30, color='#9aa4ab', zorder=0)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#cccccc')
        ax.tick_params(labelsize=13)

        box = dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', alpha=0.96,
                   edgecolor='#bdc3c7', linewidth=1.2)
        ax.text(box_xy[0], box_xy[1],
                stats_caption(r, p, verdict) + f'\n$N$ = {len(data):,}',
                transform=ax.transAxes, fontsize=12.5,
                verticalalignment=box_loc.split()[0].replace('lower', 'bottom')
                .replace('upper', 'top'),
                horizontalalignment='left', bbox=box, linespacing=1.55,
                zorder=6)

    # One shared legend above the panels, outside the axes, so that it cannot
    # come into contact with the point cloud at any canvas size.
    frame_legend(fig.legend(handles=trend_handle[:1], loc='lower center',
                            bbox_to_anchor=(0.46, 0.985), frameon=True,
                            fontsize=13, borderpad=0.6, handlelength=1.9))

    axes[0].set_ylabel(r'Reasoning Sensitivity Slope ($\beta_{sj}$)',
                       fontweight='bold', fontsize=16, labelpad=8)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=axes, fraction=0.028, pad=0.015, aspect=32)
    cbar.solids.set_edgecolor('face')
    cbar.outline.set_edgecolor('#cccccc')
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Empirical MMLU Overall Accuracy', rotation=270,
                   labelpad=20, weight='bold', fontsize=14)
    save(fig, '3_stability_inversion')

def figureA1_shrinkage(subj):
    print('Figure A1: in-sample versus cross-validated subject fit')
    res = subj[subj['reported']]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    ax = axes[0]
    for label, colour in (('STEM', STEM_RED), ('Non-STEM', HUM_BLUE)):
        sub = res[res['domain_group'] == label]
        ax.scatter(sub['R2_insample'], sub['R2_cv'], s=48, color=colour,
                   alpha=0.8, edgecolors='#0e141a', linewidths=0.4,
                   label=f'{label} (N = {len(sub)})', zorder=3)
    lim = [min(res['R2_cv'].min(), 0) - 0.05, res['R2_insample'].max() + 0.05]
    ax.plot(lim, lim, color='#7f8c8d', linestyle='--', linewidth=1.2,
            label='No shrinkage', zorder=2)
    ax.axhline(0, color='#2c3e50', linewidth=1.0, zorder=2)
    ax.set_xlabel('In-sample $R^2$', fontweight='bold')
    ax.set_ylabel('Cross-validated $R^2$', fontweight='bold')
    ax.set_title('Per-subject shrinkage', fontsize=13)
    frame_legend(ax.legend(fontsize=10, frameon=True))

    ax = axes[1]
    for label, colour in (('STEM', STEM_RED), ('Non-STEM', HUM_BLUE)):
        sub = res[res['domain_group'] == label]
        ax.scatter(sub['n_items'], sub['R2_insample'] - sub['R2_cv'], s=48,
                   color=colour, alpha=0.8, edgecolors='#0e141a',
                   linewidths=0.4, label=label, zorder=3)
    ax.set_xscale('log')
    ax.set_xlabel('Items in subject (log scale)', fontweight='bold')
    ax.set_ylabel('In-sample $-$ cross-validated $R^2$', fontweight='bold')
    ax.set_title('Optimism scales with the predictor-to-item ratio',
                 fontsize=13)
    frame_legend(ax.legend(fontsize=10, frameon=True))

    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.35, color='#888888', zorder=0)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#cccccc')
    fig.tight_layout()
    save(fig, 'A1_subject_shrinkage')

def figureA2_residuals(df):
    print('Figure A2: residual diagnostics')
    y = df['difficulty_score'].values
    X = sm.add_constant(df[INDICATORS].values)
    stem = (df['domain_group'] == 'STEM').values

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    fits = (('Pooled', np.ones(len(y), bool), '#34495e'),
            ('STEM', stem, STEM_RED), ('non-STEM', ~stem, HUM_BLUE))

    ax = axes[0]
    for name, mask, colour in fits:
        m = sm.OLS(y[mask], X[mask]).fit()
        r = (m.resid - m.resid.mean()) / m.resid.std(ddof=1)
        q = stats.probplot(r, dist='norm', fit=False)
        ax.plot(q[0], q[1], '.', markersize=2.5, color=colour, alpha=0.6,
                label=name)
    span = [-4.5, 4.5]
    ax.plot(span, span, color='#2c3e50', linewidth=1.2, linestyle='--')
    ax.set_xlim(span)
    ax.set_xlabel('Theoretical normal quantile', fontweight='bold')
    ax.set_ylabel('Standardized residual quantile', fontweight='bold')
    ax.set_title('Normal Q-Q', fontsize=13)
    frame_legend(ax.legend(fontsize=10, markerscale=4, frameon=True))

    m_pooled = sm.OLS(y, X).fit()
    ax = axes[1]
    ax.scatter(m_pooled.fittedvalues, m_pooled.resid, s=3, alpha=0.25,
               color='#34495e')
    ax.axhline(0, color='#d62728', linewidth=1.2)
    ax.set_xlabel('Fitted difficulty', fontweight='bold')
    ax.set_ylabel('Residual', fontweight='bold')
    ax.set_title('Residual versus fitted (pooled)', fontsize=13)

    ax = axes[2]
    for name, mask, colour in fits:
        m = sm.OLS(y[mask], X[mask]).fit()
        r = (m.resid - m.resid.mean()) / m.resid.std(ddof=1)
        ax.hist(r, bins=70, density=True, histtype='step', linewidth=1.6,
                color=colour, label=name)
    grid = np.linspace(-4.5, 4.5, 400)
    ax.plot(grid, stats.norm.pdf(grid), color='#2c3e50', linestyle='--',
            linewidth=1.4, label='N(0, 1)')
    ax.set_xlabel('Standardized residual', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Residual distribution', fontsize=13)
    frame_legend(ax.legend(fontsize=10, frameon=True))

    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.3, color='#888888', zorder=0)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#cccccc')
    fig.tight_layout()
    save(fig, 'A2_residual_diagnostics')

def figureA3_permutation():
    path = resolve('mmlu_permutation_gap.npy')
    if not os.path.exists(path):
        print('Figure A3 skipped: run 17_robust_invariance.py first')
        return
    print('Figure A3: permutation null for the domain gap')
    perm = np.load(path)
    boot = pd.read_csv(resolve('mmlu_bootstrap_invariance.csv'))
    observed = float(boot['gap'].mean())

    df = pd.read_csv(resolve('mmlu_dimension5_adversarial.csv'))
    df = df.dropna(subset=INDICATORS + ['difficulty_score', 'domain_group'])
    y = df['difficulty_score'].values
    X = sm.add_constant(df[INDICATORS].values)
    stem = (df['domain_group'] == 'STEM').values
    observed = (sm.OLS(y[stem], X[stem]).fit().rsquared
                - sm.OLS(y[~stem], X[~stem]).fit().rsquared)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    ax.hist(perm, bins=60, color='#95a5a6', edgecolor='white', linewidth=0.4)
    ax.axvline(observed, color=STEM_RED, linewidth=2.4,
               label=f'Observed gap = {observed:.4f}')
    ax.axvline(np.percentile(np.abs(perm), 95), color='#2c3e50',
               linestyle='--', linewidth=1.4,
               label='95th percentile of $|$null$|$')
    ax.set_xlabel(r'$R^2_{\mathrm{STEM}} - R^2_{\mathrm{non\text{-}STEM}}$ '
                  'under label permutation', fontweight='bold')
    ax.set_ylabel('Permutations', fontweight='bold')
    ax.set_title(f'Permutation null ({len(perm):,} draws)', fontsize=13)
    frame_legend(ax.legend(fontsize=10, frameon=True))

    ax = axes[1]
    ax.hist(boot['gap'], bins=50, color='#7fb2d5', edgecolor='white',
            linewidth=0.4)
    lo, hi = np.percentile(boot['gap'], [2.5, 97.5])
    ax.axvline(observed, color=STEM_RED, linewidth=2.4, label='Observed gap')
    ax.axvline(lo, color='#2c3e50', linestyle='--', linewidth=1.4,
               label=f'95% CI [{lo:.4f}, {hi:.4f}]')
    ax.axvline(hi, color='#2c3e50', linestyle='--', linewidth=1.4)
    ax.axvline(0, color='#2c3e50', linewidth=1.0)
    ax.set_xlabel('Bootstrap $R^2$ gap', fontweight='bold')
    ax.set_ylabel('Resamples', fontweight='bold')
    ax.set_title(f'Bootstrap distribution ({len(boot):,} resamples)',
                 fontsize=13)
    frame_legend(ax.legend(fontsize=10, frameon=True))

    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.3, color='#888888', zorder=0)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#cccccc')
    fig.tight_layout()
    save(fig, 'A3_permutation_bootstrap')

def main():
    df, subj = load_frames()
    figure1_bifurcation(df, subj)
    figure_bifurcation_compact(df, subj)
    figure3_inversion(pd.read_csv(resolve('mmlu_model_profiles.csv')))
    figureA1_shrinkage(subj)
    figureA2_residuals(df)
    figureA3_permutation()
    print(f'Figures written to {OUT_DIR}')



def figure_bifurcation_compact(df, subj):
    print('Figure 1 (compact): subject-level bifurcation strip plot')
    res = subj[subj['reported']].copy()
    is_stem = res['domain_group'].eq('STEM')

    y = df['difficulty_score'].values
    X = sm.add_constant(df[INDICATORS].values)
    stem = (df['domain_group'] == 'STEM').values
    stem_adj = sm.OLS(y[stem], X[stem]).fit().rsquared_adj
    ns_adj = sm.OLS(y[~stem], X[~stem]).fit().rsquared_adj

    rng = np.random.default_rng(1729)
    fig, ax = plt.subplots(figsize=(7.6, 2.15))

    for row_y, mask, colour, label in ((1.0, is_stem, STEM_RED, 'STEM'),
                                       (0.0, ~is_stem, HUM_BLUE, 'non-STEM')):
        vals = res.loc[mask, 'R2_adjusted'].values
        jitter = row_y + rng.uniform(-0.16, 0.16, len(vals))
        ax.scatter(vals, jitter, s=42, facecolor=colour, alpha=0.75,
                   edgecolors='#0e141a', linewidths=0.45, zorder=3,
                   label=f'{label} ($N$ = {len(vals)})')

    for value, colour, style in ((stem_adj, STEM_RED, ':'),
                                 (ns_adj, HUM_BLUE, ':')):
        ax.axvline(value, color=colour, linestyle=style, linewidth=1.6,
                   zorder=2)
    ax.axvline(0.0, color='#7f8c8d', linewidth=1.0, zorder=1)

    ax.annotate(f'partition fit {stem_adj:.3f}', xy=(stem_adj, 1.42),
                xytext=(4, 0), textcoords='offset points', fontsize=9,
                color=STEM_RED, va='center')
    ax.annotate(f'{ns_adj:.3f}', xy=(ns_adj, -0.42), xytext=(4, 0),
                textcoords='offset points', fontsize=9, color=HUM_BLUE,
                va='center')

    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(['non-STEM', 'STEM'], fontsize=11, fontweight='bold')
    ax.set_ylim(-0.6, 1.75)
    ax.set_xlabel(r'Per-subject adjusted $R^{2}$', fontweight='bold',
                  fontsize=11)
    ax.tick_params(labelsize=10)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3, color='#9aa4ab',
            zorder=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    frame_legend(fig.legend(loc='lower center', bbox_to_anchor=(0.55, 0.965),
                            ncol=2, frameon=True, fontsize=9, borderpad=0.5,
                            handletextpad=0.4, columnspacing=1.4))
    fig.tight_layout()
    save(fig, '1_subject_bifurcation_compact')


if __name__ == '__main__':
    main()
