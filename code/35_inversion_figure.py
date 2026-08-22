import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BASE_DIR, RESULTS_DIR, SEED, resolve

_inv = __import__("29_inversion_null")
GUESS = _inv.GUESS
N_DRAW = 20

OUT_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(OUT_DIR, exist_ok=True)

for _font in glob.glob(os.path.join(BASE_DIR, 'resources', 'fonts', '*.ttf')):
    fm.fontManager.addfont(_font)

plt.rcParams.update({
    'axes.unicode_minus': False, 'font.family': 'Noto Serif', 'font.size': 8.0,
    'axes.labelsize': 9.5, 'axes.titlesize': 9.5, 'axes.titleweight': 'bold',
    'axes.linewidth': 0.8, 'figure.dpi': 600, 'savefig.dpi': 600,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'xtick.labelsize': 8.0, 'ytick.labelsize': 8.0,
    'xtick.major.size': 3.0, 'ytick.major.size': 3.0,
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'xtick.major.pad': 3.0, 'ytick.major.pad': 3.0,
    'lines.solid_capstyle': 'round',
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Noto Serif',
    'mathtext.it': 'Noto Serif:italic', 'mathtext.bf': 'Noto Serif:bold',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

NULL_GREY = '#8e9aa1'
NULL_FILL = '#dde3e7'
OBS_INK = '#2b3a47'
MARK_EDGE = '#16202a'
RULE = '#aab3b9'
CASE = '#1a2028'
GRID = '#aeb7bd'
CALLOUT = '#5b666e'

# Exemplar models annotated in panel (a); display names are
# shortened forms of the Hugging Face repository identifiers.
CALLOUTS = [
    ('monology/mixtral-soup', 'Mixtral Soup', (-0.85, 4.30), 'left'),
    ('OpenBuddy/openbuddy-mixtral-7bx8-v17.2-32k', 'OpenBuddy Mixtral',
     (-0.55, 0.90), 'left'),
    ('upstage/SOLAR-10.7B-v1.0', 'SOLAR-10.7B', (-0.30, -4.30), 'left'),
    ('Qwen/Qwen2-72B', 'Qwen2-72B', (1.05, -6.40), 'left'),
]


def save(fig, stem):
    for ext, kw in (('pdf', {}), ('svg', {}), ('png', {'dpi': 300})):
        fig.savefig(os.path.join(OUT_DIR, f'{stem}.{ext}'), **kw)
    plt.close(fig)
    print(f'  saved {stem} (pdf/svg/png)')


def first_draw(theta_hat, a_hat, b_hat, W, seed):
    P = GUESS + (1.0 - GUESS) / (1.0 + np.exp(
        -a_hat * (theta_hat[:, None] - b_hat)))
    rng = np.random.default_rng(seed)
    Ys = (rng.random(P.shape) < P).astype(np.int8)
    return _inv.per_model_slopes(Ys, W)


def main():
    print("Revised reasoning-sensitivity figure")
    slopes_path = resolve("mmlu_inversion_slopes.csv")
    if not os.path.exists(slopes_path):
        raise SystemExit("run stage 29 first: mmlu_inversion_slopes.csv missing")
    sl = pd.read_csv(slopes_path)

    df = pd.read_csv(resolve("mmlu_IRT_calibrated.csv"), low_memory=False)
    feats = pd.read_csv(resolve("mmlu_dimension5_adversarial.csv"))
    meta = ['question_id', 'item_id', 'subject', 'question_text',
            'clean_text', 'choices', 'ground_truth', 'domain_group',
            'difficulty_score', 'discrimination_score']
    model_cols = [c for c in df.columns if c not in meta]
    stem = (df['domain_group'] == 'STEM').to_numpy()
    W = feats.loc[stem, 'WSCG_Depth'].to_numpy(dtype=np.float64)

    cache = resolve("mmlu_3pl_stem_items.csv")
    if os.path.exists(cache):
        par = pd.read_csv(cache)
        a_hat, b_hat = par['a_3pl'].to_numpy(), par['b_3pl'].to_numpy()
        print("  using stage 29's cached 3PL fit")
    else:
        print("  no cached 3PL fit; refitting")
        import jax.numpy as jnp
        Y = df.loc[stem, model_cols].to_numpy(dtype=np.int8).T
        guide, res = _inv.run_svi(_inv.threepl_model, SEED,
                                  Y=jnp.asarray(Y.astype(np.float32)))
        med = guide.median(res.params)
        a_hat, b_hat = np.asarray(med['a']), np.asarray(med['b'])

    beta_null = first_draw(sl['theta_3pl'].to_numpy(), a_hat, b_hat, W, SEED)

    theta = sl['theta'].to_numpy()
    theta_null = sl['theta_3pl'].to_numpy()
    beta = sl['beta_marginal'].to_numpy()


    def z(v):
        return (v - v.mean()) / v.std(ddof=1)

    zx, zx_null = z(theta), z(theta_null)

    prof = pd.read_csv(resolve("mmlu_model_profiles.csv")).set_index(
        'model_name')
    acc = prof.loc[sl['model_name'], 'accuracy_overall'].to_numpy()

    cutoff = float(np.median(theta))
    names = sl['model_name'].to_numpy()

    # Slopes are reported on a 10^-2 scale so the axis carries integer ticks.
    SC = 100.0
    beta, beta_null = beta * SC, beta_null * SC

    panels = [
        (np.ones(len(theta), bool), '(a)  All 1,000 models'),
        (theta >= cutoff,
         r'(b)  Above the ability median, $\theta_j \geq$ ' + f'{cutoff:.3f}'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(6.30, 3.15), sharey=True,
                             gridspec_kw={'wspace': 0.06})
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.165, top=0.905)
    norm = plt.Normalize(acc.min() + 0.01, acc.max() - 0.01)
    cmap = plt.get_cmap('coolwarm')

    lo = min(beta.min(), beta_null.min())
    hi = max(beta.max(), beta_null.max())
    pad = 0.085 * (hi - lo)

    handles = None
    for idx, (ax, (mask, title)) in enumerate(zip(axes, panels)):
        mask_null = (np.ones(len(theta_null), bool) if mask.all()
                     else theta_null >= np.quantile(theta_null, 0.5))
        r_obs = stats.pearsonr(theta[mask], beta[mask])[0]
        r_null = stats.pearsonr(theta_null[mask_null], beta_null[mask_null])[0]
        th, bt = zx[mask], beta[mask]
        thn, bn = zx_null[mask_null], beta_null[mask_null]

        ax.set_axisbelow(True)
        ax.grid(True, linestyle=(0, (2.6, 2.6)), linewidth=0.55, alpha=0.55,
                color=GRID)

        # Null cloud first, so the observed models read as the foreground.
        ax.scatter(thn, bn, s=4.5, facecolors=NULL_FILL, edgecolors='none',
                   alpha=0.62, zorder=2)
        faces = cmap(norm(acc[mask]))
        faces[:, 3] = 0.92
        ax.scatter(th, bt, s=11.0, facecolors=faces, edgecolors=MARK_EDGE,
                   linewidths=0.30, zorder=3)

        gridn = np.linspace(thn.min(), thn.max(), 200)
        null_line, = ax.plot(
            gridn, np.polyval(np.polyfit(thn, bn, 1), gridn), color=NULL_GREY,
            linewidth=1.45, linestyle=(0, (3.6, 2.2)), zorder=2.4,
            dash_capstyle='round', label='3PL null')
        grid = np.linspace(th.min(), th.max(), 200)
        obs_line, = ax.plot(
            grid, np.polyval(np.polyfit(th, bt, 1), grid), color=OBS_INK,
            linewidth=2.0, zorder=2.6, label='Observed')
        handles = [obs_line, null_line]

        ax.set_title(title, fontsize=8.0, pad=4.5, loc='left', color='#1d262e')
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(min(th.min(), thn.min()) - 0.14,
                    max(th.max(), thn.max()) + 0.14)
        ax.xaxis.set_major_locator(plt.MaxNLocator(6, prune='both'))
        ax.yaxis.set_major_locator(plt.MultipleLocator(5))
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(RULE)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors='#4d585f', labelcolor='#1d262e')

        box = dict(boxstyle='round,pad=0.34,rounding_size=0.22',
                   facecolor='#f7f8f9', alpha=0.94, edgecolor='#c6ccd1',
                   linewidth=0.55)
        # Panel (a) is dense at the top, panel (b) at the bottom, so the
        # summary sits in whichever corner each panel leaves empty.
        anchor = (0.028, 0.030, 'bottom') if idx == 0 else (0.028, 0.972, 'top')
        ax.text(anchor[0], anchor[1],
                r'$\mathbf{Observed}$  $r$ = ' + f'{r_obs:.3f}\n'
                + r'$\mathbf{Null}$  $r$ = ' + f'{r_null:.3f}\n'
                + r'$\mathbf{Residual}$  ' + f'{r_obs - r_null:+.3f}\n'
                + f'$N$ = {int(mask.sum()):,}',
                transform=ax.transAxes, fontsize=6.3, color='#1d262e',
                verticalalignment=anchor[2], horizontalalignment='left',
                bbox=box, linespacing=1.4, zorder=7)

        if idx == 0:
            for model, label, (tx, ty), ha in CALLOUTS:
                hit = np.flatnonzero(names[mask] == model)
                if not len(hit):
                    continue
                k = hit[0]
                ax.annotate(
                    label, xy=(th[k], bt[k]), xytext=(tx, ty),
                    textcoords='data', fontsize=6.0, style='italic',
                    color='#2b3a47', ha=ha, va='center', zorder=8,
                    arrowprops=dict(arrowstyle='-', color=CALLOUT,
                                    linewidth=0.7, shrinkA=2.0, shrinkB=2.5,
                                    connectionstyle='arc3,rad=0.0'))
        else:
            leg = ax.legend(handles=handles, loc='upper right', frameon=True,
                            fontsize=6.3, borderpad=0.42,
                            labelspacing=0.38, handlelength=1.7,
                            handletextpad=0.5, borderaxespad=0.5)
            leg.set_zorder(9)
            leg.get_frame().set_facecolor('#ffffff')
            leg.get_frame().set_edgecolor('#c6ccd1')
            leg.get_frame().set_linewidth(0.55)
            leg.get_frame().set_alpha(0.94)
            for txt in leg.get_texts():
                txt.set_color('#1d262e')

    fig.supxlabel(r'Model Latent Ability ($\theta_j$, standardised)',
                  fontweight='bold', fontsize=8.0, y=0.018)
    axes[0].set_ylabel('Reasoning Sensitivity Slope' + chr(10) +
                       r'($\beta_{sj}$, $\times 10^{-2}$)',
                       fontweight='bold', fontsize=8.0, labelpad=3.5,
                       linespacing=1.35)
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                        fraction=0.030, pad=0.016, aspect=24)
    cbar.solids.set_edgecolor('face')
    cbar.outline.set_edgecolor(CASE)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(labelsize=8.0, width=0.6, size=2.8, pad=2.5,
                        colors=CASE, labelcolor='#1d262e')
    cbar.set_label('Empirical MMLU Overall Accuracy', rotation=270,
                   labelpad=10.0, weight='bold', fontsize=8.0)
    save(fig, '3_stability_inversion')


if __name__ == '__main__':
    main()
