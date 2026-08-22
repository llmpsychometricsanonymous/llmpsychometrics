import os
import re
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (CV_SEED, INDICATORS, MIN_ITEMS_REPORTED, N_FOLDS,
                    output_path, resolve, revision_report)

_VOWEL_RUN = re.compile(r'[aeiouy]+')
_SENTENCE_END = re.compile(r'[.!?]+(?:\s|$)')
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")

def count_syllables(word):
    word = word.lower().strip("'-")
    if not word:
        return 0
    runs = len(_VOWEL_RUN.findall(word))
    if word.endswith('e') and not word.endswith(('le', 'ee', 'ye')) and runs > 1:
        runs -= 1
    return max(runs, 1)

def surface_features(text):
    text = '' if not isinstance(text, str) else text
    words = _WORD.findall(text)
    n_words = len(words)
    n_sentences = max(len(_SENTENCE_END.findall(text)), 1)
    n_syllables = sum(count_syllables(w) for w in words)
    if n_words == 0:
        return 0, n_sentences, 0.0
    fk = (0.39 * (n_words / n_sentences)
          + 11.8 * (n_syllables / n_words) - 15.59)
    return n_words, n_sentences, fk

def ols_r2(X, y):
    if X.shape[0] <= X.shape[1] + 1:
        return float('nan')
    return LinearRegression().fit(X, y).score(X, y)

def cv_r2(X, y, n_splits=N_FOLDS, seed=CV_SEED):
    n = X.shape[0]
    k = min(n_splits, n)
    if k < 2 or n <= X.shape[1] + 2:
        return float('nan'), float('nan')
    splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
    scores = cross_val_score(LinearRegression(), X, y,
                             cv=splitter, scoring='r2')
    return float(np.mean(scores)), float(np.std(scores, ddof=1))

def repeated_cv_r2(X, y, n_splits=N_FOLDS, n_repeats=20, seed=CV_SEED):
    n = X.shape[0]
    if n <= X.shape[1] + 2:
        return float('nan'), float('nan')
    splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                             random_state=seed)
    scores = cross_val_score(LinearRegression(), X, y,
                             cv=splitter, scoring='r2')
    return float(np.mean(scores)), float(np.std(scores, ddof=1))

def adjusted_r2(r2, n, k):
    return 1.0 - (1.0 - r2) * (n - 1) / (n - k - 1)

def nested_f_test(rss_small, rss_big, n, p_small, p_big):
    df1 = p_big - p_small
    df2 = n - p_big - 1
    f = ((rss_small - rss_big) / df1) / (rss_big / df2)
    from scipy.stats import f as f_dist
    return f, df1, df2, float(f_dist.sf(f, df1, df2))

def rss_of(X, y):
    model = LinearRegression().fit(X, y)
    resid = y - model.predict(X)
    return float(np.sum(resid ** 2))

def main():
    print("Cross-validated subject fit and baseline model comparison")

    df = pd.read_csv(resolve("mmlu_dimension5_adversarial.csv"))
    df = df.dropna(subset=INDICATORS + ['difficulty_score', 'domain_group'])

    surface = df['clean_text'].apply(surface_features)
    df['n_words'] = [s[0] for s in surface]
    df['n_sentences'] = [s[1] for s in surface]
    df['flesch_kincaid'] = [s[2] for s in surface]

    y_all = df['difficulty_score'].values
    stem_mask = (df['domain_group'] == 'STEM').values

    baselines = [
        ("Word count only", ['n_words']),
        ("+ sentence count", ['n_words', 'n_sentences']),
        ("+ Flesch-Kincaid", ['n_words', 'n_sentences', 'flesch_kincaid']),
        ("Nine-indicator framework", INDICATORS),
        ("Framework + surface features",
         INDICATORS + ['n_words', 'n_sentences', 'flesch_kincaid']),
    ]

    baseline_rows = []
    for label, cols in baselines:
        X = df[cols].values
        row = {'model': label, 'k': len(cols)}
        for name, mask in (('global', np.ones(len(df), dtype=bool)),
                           ('stem', stem_mask),
                           ('nonstem', ~stem_mask)):
            Xm, ym = X[mask], y_all[mask]
            row[f'{name}_r2'] = ols_r2(Xm, ym)
            row[f'{name}_cv_r2'], row[f'{name}_cv_sd'] = cv_r2(Xm, ym)
        baseline_rows.append(row)
    df_base = pd.DataFrame(baseline_rows)

    increments = {}
    for name, mask in (('global', np.ones(len(df), dtype=bool)),
                       ('stem', stem_mask),
                       ('nonstem', ~stem_mask)):
        ym = y_all[mask]
        X_read = df.loc[mask, ['n_words', 'n_sentences',
                               'flesch_kincaid']].values
        X_full = df.loc[mask, INDICATORS +
                        ['n_words', 'n_sentences', 'flesch_kincaid']].values
        f, df1, df2, p = nested_f_test(rss_of(X_read, ym), rss_of(X_full, ym),
                                       len(ym), X_read.shape[1],
                                       X_full.shape[1])
        increments[name] = (f, df1, df2, p)

    subject_rows = []
    for subject, group in df.groupby('subject'):
        X = group[INDICATORS].values
        y = group['difficulty_score'].values
        cv_mean, cv_sd = repeated_cv_r2(X, y)
        r2_in = ols_r2(X, y)
        subject_rows.append({
            'subject': subject,
            'domain_group': group['domain_group'].iloc[0],
            'n_items': len(group),
            'R2_insample': r2_in,
            'R2_adjusted': adjusted_r2(r2_in, len(group), len(INDICATORS)),
            'R2_cv': cv_mean,
            'R2_cv_sd': cv_sd,
            'reported': len(group) >= MIN_ITEMS_REPORTED,
        })
    df_subj = pd.DataFrame(subject_rows).sort_values(
        'R2_cv', ascending=False).reset_index(drop=True)
    df_subj.to_csv(output_path("mmlu_subject_r2_crossvalidated.csv"),
                   index=False)

    reported = df_subj[df_subj['reported']]
    excluded = df_subj[~df_subj['reported']]

    dom_cv = {}
    for name, mask in (('global', np.ones(len(df), dtype=bool)),
                       ('stem', stem_mask),
                       ('nonstem', ~stem_mask)):
        dom_cv[name] = cv_r2(df.loc[mask, INDICATORS].values, y_all[mask])

    fit_summaries = {}
    for name, mask in (('Pooled', np.ones(len(df), dtype=bool)),
                       ('STEM', stem_mask),
                       ('non-STEM', ~stem_mask)):
        X = sm.add_constant(df.loc[mask, INDICATORS].values)
        model = sm.OLS(y_all[mask], X).fit(cov_type='HC3')
        ci = model.conf_int(alpha=0.05)
        fit_summaries[name] = {
            'r2': model.rsquared,
            'f2': model.rsquared / (1.0 - model.rsquared),
            'params': model.params,
            'ci': ci,
            'n': int(mask.sum()),
        }

    out = []
    out.append("")
    out.append("BASELINE MODEL COMPARISON (nested predictor sets)")
    out.append("")
    out.append("R2 is in-sample; CV R2 is the mean of 5-fold out-of-sample R2 "
               "(SD in parentheses).")
    out.append("")
    header = (f"{'Predictor set':<32} | {'k':>2} | {'Global R2':>9} | "
              f"{'Global CV':>16} | {'STEM R2':>8} | {'STEM CV':>16} | "
              f"{'nSTEM R2':>8} | {'nSTEM CV':>16} |")
    out.append(header)
    out.append("-" * len(header))
    for _, r in df_base.iterrows():
        out.append(
            f"{r['model']:<32} | {int(r['k']):>2} | {r['global_r2']:>9.4f} | "
            f"{r['global_cv_r2']:>9.4f} ({r['global_cv_sd']:.4f}) | "
            f"{r['stem_r2']:>8.4f} | "
            f"{r['stem_cv_r2']:>9.4f} ({r['stem_cv_sd']:.4f}) | "
            f"{r['nonstem_r2']:>8.4f} | "
            f"{r['nonstem_cv_r2']:>9.4f} ({r['nonstem_cv_sd']:.4f}) |")
    out.append("")
    out.append("Incremental F-test, nine indicators added to the readability "
               "baseline:")
    for name in ('global', 'stem', 'nonstem'):
        f, df1, df2, p = increments[name]
        out.append(f"  {name:<8}: F({df1}, {df2}) = {f:.3f}, p = {p:.3e}")

    out.append("")
    out.append("DOMAIN-LEVEL CROSS-VALIDATED R2 (nine indicators)")
    for name in ('global', 'stem', 'nonstem'):
        m, s = dom_cv[name]
        out.append(f"  {name:<8}: CV R2 = {m:.4f} (SD {s:.4f})")

    out.append("")
    out.append("MODEL FIT AND EFFECT SIZE BY PARTITION")
    out.append(f"{'Partition':<10} | {'N':>6} | {'R2':>8} | {'Cohen f2':>9} |")
    out.append("-" * 45)
    for name, s in fit_summaries.items():
        out.append(f"{name:<10} | {s['n']:>6} | {s['r2']:>8.4f} | "
                   f"{s['f2']:>9.4f} |")

    out.append("")
    out.append("95% HC3 CONFIDENCE INTERVALS FOR PATH COEFFICIENTS")
    out.append(f"{'Indicator':<24} | {'Pooled b [95% CI]':>30} | "
               f"{'STEM b [95% CI]':>30} | {'non-STEM b [95% CI]':>30} |")
    out.append("-" * 124)
    for idx, ind in enumerate(INDICATORS):
        cells = []
        for name in ('Pooled', 'STEM', 'non-STEM'):
            s = fit_summaries[name]
            b = s['params'][idx + 1]
            lo, hi = s['ci'][idx + 1]
            cells.append(f"{b:>8.4f} [{lo:>7.4f}, {hi:>7.4f}]")
        out.append(f"{ind:<24} | {cells[0]:>30} | {cells[1]:>30} | "
                   f"{cells[2]:>30} |")

    out.append("")
    out.append("PER-SUBJECT CROSS-VALIDATED R2")
    out.append(f"Subjects with N >= {MIN_ITEMS_REPORTED} items: "
               f"{len(reported)} of {len(df_subj)}")
    if len(excluded):
        out.append("Excluded from per-subject reporting (too few items): "
                   + ", ".join(f"{r['subject']} (N={int(r['n_items'])})"
                               for _, r in excluded.iterrows()))
    out.append("")
    out.append("CV column is the mean of 20 repeats of 5-fold CV; the SD in "
               "parentheses is across all 100 folds.")
    out.append("")
    out.append(f"{'Subject':<38} | {'Domain':<8} | {'N':>5} | "
               f"{'R2 (in-sample)':>14} | {'R2 (adj.)':>10} | "
               f"{'R2 (repeated CV)':>18} |")
    out.append("-" * 108)
    for _, r in reported.iterrows():
        out.append(
            f"{r['subject']:<38} | {r['domain_group']:<8} | "
            f"{int(r['n_items']):>5} | {r['R2_insample']:>14.4f} | "
            f"{r['R2_adjusted']:>10.4f} | "
            f"{r['R2_cv']:>10.4f} ({r['R2_cv_sd']:.4f}) |")

    shrink = reported['R2_insample'] - reported['R2_cv']
    n_positive = int((reported['R2_cv'] > 0).sum())
    out.append("")
    out.append(f"Mean in-sample minus cross-validated R2 across reported "
               f"subjects: {shrink.mean():.4f} (max {shrink.max():.4f})")
    out.append(f"Subjects with positive cross-validated R2: "
               f"{n_positive} of {len(reported)}")
    out.append("")
    out.append("SUBJECT-LEVEL DOMAIN CONTRAST")
    out.append("Treating each subject as one observation. Adjusted R2 is the "
               "defensible per-subject summary: it is a closed-form penalty "
               "for the 9:N ratio rather than a resampling estimate, so it "
               "does not inherit the fold-to-fold variance that makes the CV "
               "column uninterpretable subject by subject.")
    from scipy import stats as _st
    for metric in ('R2_insample', 'R2_adjusted', 'R2_cv'):
        s = reported.loc[reported['domain_group'] == 'STEM', metric]
        ns = reported.loc[reported['domain_group'] != 'STEM', metric]
        t, p_t = _st.ttest_ind(s, ns, equal_var=False)
        u, p_u = _st.mannwhitneyu(s, ns, alternative='two-sided')
        pooled_sd = np.sqrt(((len(s) - 1) * s.var(ddof=1)
                             + (len(ns) - 1) * ns.var(ddof=1))
                            / (len(s) + len(ns) - 2))
        d = (s.mean() - ns.mean()) / pooled_sd
        out.append(f"  {metric:<12}: STEM mean {s.mean():>8.4f} "
                   f"(median {s.median():>8.4f}), non-STEM mean "
                   f"{ns.mean():>8.4f} (median {ns.median():>8.4f}); "
                   f"Welch t({len(s) + len(ns) - 2}) = {t:.3f}, p = {p_t:.4f}; "
                   f"Mann-Whitney U = {u:.1f}, p = {p_u:.4f}; "
                   f"Cohen's d = {d:.3f}")
    out.append("")

    text = "\n".join(out)
    print(text)
    report = revision_report("results_crossvalidation_and_baselines.txt",
                             ["CROSS-VALIDATION AND BASELINE COMPARISON"] + out)

    df_base.to_csv(output_path("mmlu_baseline_models.csv"), index=False)
    print(f"Written: {os.path.basename(report)}, "
          "mmlu_subject_r2_crossvalidated.csv, mmlu_baseline_models.csv")

if __name__ == '__main__':
    main()
