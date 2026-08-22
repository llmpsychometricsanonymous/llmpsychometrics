import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import resolve, revision_report

REFERENCE = 'llama'
MIN_FAMILY = 25
N_STRATA = 10
MIN_STRATUM = 8
MIN_STRATA_USED = 6

FAMILY_PATTERNS = [
    ('mixtral', r'mixtral'),
    ('llama', r'llama'),
    ('qwen', r'qwen'),
    ('yi', r'\byi[-_. ]'),
    ('solar', r'solar'),
    ('falcon', r'falcon'),
    ('pythia', r'pythia'),
    ('deepseek', r'deepseek'),
    ('gemma', r'gemma'),
    ('phi', r'\bphi[-_ ]?\d'),
    ('mistral', r'mistral'),
]


def family_of(name):
    s = name.lower()
    for label, pat in FAMILY_PATTERNS:
        if re.search(pat, s):
            return label
    return 'other'


def mh_dif(Y_ref, Y_foc, theta_ref, theta_foc):
    theta_all = np.concatenate([theta_ref, theta_foc])
    edges = np.quantile(theta_all, np.linspace(0, 1, N_STRATA + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    s_ref = np.digitize(theta_ref, edges[1:-1])
    s_foc = np.digitize(theta_foc, edges[1:-1])

    n_items = Y_ref.shape[1]
    num = np.zeros(n_items)
    den = np.zeros(n_items)
    e_sum = np.zeros(n_items)
    v_sum = np.zeros(n_items)
    a_sum = np.zeros(n_items)
    used = 0

    for k in range(N_STRATA):
        r = Y_ref[s_ref == k]
        f = Y_foc[s_foc == k]
        n_r, n_f = len(r), len(f)
        if n_r + n_f < MIN_STRATUM or n_r == 0 or n_f == 0:
            continue
        used += 1
        A = f.sum(axis=0)
        B = n_f - A
        C = r.sum(axis=0)
        D = n_r - C
        N = float(n_r + n_f)


        num += (A + 0.5) * (D + 0.5) / (N + 2.0)
        den += (B + 0.5) * (C + 0.5) / (N + 2.0)
        n_correct = A + C
        e_sum += n_f * n_correct / N
        v_sum += (n_f * n_r * n_correct * (N - n_correct)) / (N * N * (N - 1.0))
        a_sum += A

    with np.errstate(divide='ignore', invalid='ignore'):
        alpha = np.where(den > 0, num / den, np.nan)
        delta = -2.35 * np.log(alpha)
        chi2 = (np.abs(a_sum - e_sum) - 0.5) ** 2 / v_sum
    chi2 = np.where(v_sum > 0, chi2, np.nan)
    return alpha, delta, chi2, used


def classify(delta, chi2, p):
    out = np.full(len(delta), 'A', dtype='<U1')
    sig = (p < 0.05) & np.isfinite(delta)
    ad = np.abs(delta)
    out[sig & (ad >= 1.0) & (ad < 1.5)] = 'B'
    out[sig & (ad >= 1.5)] = 'C'
    return out


def main():
    print("Differential item functioning across model families")
    df = pd.read_csv(resolve("mmlu_IRT_calibrated.csv"), low_memory=False)
    meta = ['question_id', 'item_id', 'subject', 'question_text', 'clean_text',
            'choices', 'ground_truth', 'domain_group', 'difficulty_score',
            'discrimination_score']
    model_cols = [c for c in df.columns if c not in meta]
    R = df[model_cols].to_numpy(dtype=np.int8).T
    stem = (df['domain_group'] == 'STEM').to_numpy()

    ab = pd.read_csv(resolve("mmlu_model_abilities.csv"))
    theta_map = dict(zip(ab['model_name'], ab['theta_score']))
    theta = np.array([theta_map[m] for m in model_cols])
    fam = np.array([family_of(m) for m in model_cols])

    counts = pd.Series(fam).value_counts()
    out = ["DIFFERENTIAL ITEM FUNCTIONING ACROSS MODEL FAMILIES", ""]
    out += [
        "Measurement invariance in Meredith's sense concerns groups of",
        "respondents. Here the respondents are language models and the groups are",
        "architecture families. Mantel-Haenszel conditions on estimated ability, so",
        "a family that is uniformly weaker does not register as DIF; only an item",
        "that is differentially hard for equally able models does.",
        "",
        f"Reference family: {REFERENCE} (n = {counts.get(REFERENCE, 0)})",
        f"Ability strata: {N_STRATA} quantiles of the joint theta distribution",
        "",
        "Family sizes:",
    ]
    for f, n in counts.items():
        out.append(f"  {f:<12}{n:>5}")

    ref_mask = fam == REFERENCE
    focal = [f for f, n in counts.items()
             if f != REFERENCE and f != 'other' and n >= MIN_FAMILY]

    out += ["", "", "DIF PREVALENCE BY FAMILY", "",
            "Items are classified on the ETS scale; B and C additionally require",
            "a significant MH chi-square at alpha = 0.05.", ""]
    header = (f"{'Focal family':<14}{'n':>5}{'strata':>8}{'A':>8}{'B':>7}{'C':>7}"
              f"{'B+C %':>9}{'mean|d|':>9}")
    out += [header, "-" * len(header)]

    per_family, weak = {}, []
    for f in focal:
        foc_mask = fam == f
        alpha, delta, chi2, used = mh_dif(R[ref_mask], R[foc_mask],
                                          theta[ref_mask], theta[foc_mask])
        p = stats.chi2.sf(chi2, 1)
        cls = classify(delta, chi2, p)
        if used < MIN_STRATA_USED:
            weak.append(f)
        per_family[f] = (delta, cls, used)
        nA = int((cls == 'A').sum())
        nB = int((cls == 'B').sum())
        nC = int((cls == 'C').sum())
        pct = 100.0 * (nB + nC) / len(cls)
        mark = ' *' if used < MIN_STRATA_USED else ''
        out.append(f"{f:<14}{int(foc_mask.sum()):>5}{used:>8}{nA:>8}{nB:>7}"
                   f"{nC:>7}{pct:>8.1f}%{np.nanmean(np.abs(delta)):>9.3f}{mark}")

    if weak:
        out += ["",
                f"* {', '.join(weak)}: fewer than {MIN_STRATA_USED} usable ability",
                "  strata, because the family's ability range barely overlaps the",
                "  reference. Conditioning is weak there and these rows should be",
                "  read as indicative only."]


    clean = np.ones(R.shape[1], bool)
    for f in focal:
        if f in weak:
            continue
        clean &= (per_family[f][1] == 'A')
    out += ["", "", "PURIFIED SECOND PASS", "",
            f"Matching on proportion correct over the {int(clean.sum())} items "
            "classified A for", "every well-conditioned focal family, rather "
            "than on the full-pool theta.", ""]

    if clean.sum() < 200:
        out.append("Too few clean items remain for a purified pass.")
        purified = {}
    else:
        theta_pure = R[:, clean].mean(axis=1)
        header = (f"{'Focal family':<14}{'strata':>8}{'B+C %':>9}{'STEM %':>9}"
                  f"{'non-STEM %':>13}{'diff':>8}{'z':>8}{'p':>11}")
        out += [header, "-" * len(header)]
        purified = {}
        for f in focal:
            if f in weak:
                continue
            fm = fam == f
            _, d2, c2, used2 = mh_dif(R[ref_mask], R[fm],
                                      theta_pure[ref_mask], theta_pure[fm])
            p2 = stats.chi2.sf(c2, 1)
            cls2 = classify(d2, c2, p2)
            purified[f] = (d2, cls2)
            flag = np.isin(cls2, ['B', 'C'])
            p1m, p2m = flag[stem].mean(), flag[~stem].mean()
            n1, n2 = int(stem.sum()), int((~stem).sum())
            pp = flag.sum() / (n1 + n2)
            se = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
            z = (p1m - p2m) / se if se > 0 else np.nan
            out.append(f"{f:<14}{used2:>8}{100 * flag.mean():>8.1f}%"
                       f"{100 * p1m:>8.1f}%{100 * p2m:>12.1f}%"
                       f"{100 * (p1m - p2m):>7.1f}%{z:>8.2f}"
                       f"{2 * stats.norm.sf(abs(z)):>11.2e}")


    out += ["", "", "IS DIF CONCENTRATED IN ONE DOMAIN?", "",
            "Proportion of items flagged B or C, by partition, with a two-",
            "proportion z-test on the difference.", ""]
    header = (f"{'Focal family':<14}{'STEM %':>9}{'non-STEM %':>13}"
              f"{'diff':>9}{'z':>8}{'p':>11}")
    out += [header, "-" * len(header)]
    for f in focal:
        delta, cls, _ = per_family[f]
        flag = np.isin(cls, ['B', 'C'])
        p1 = flag[stem].mean()
        p2 = flag[~stem].mean()
        n1, n2 = int(stem.sum()), int((~stem).sum())
        pp = (flag[stem].sum() + flag[~stem].sum()) / (n1 + n2)
        se = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
        z = (p1 - p2) / se if se > 0 else np.nan
        out.append(f"{f:<14}{100 * p1:>8.1f}%{100 * p2:>12.1f}%"
                   f"{100 * (p1 - p2):>8.1f}%{z:>8.2f}"
                   f"{2 * stats.norm.sf(abs(z)):>11.2e}")


    out += ["", "", "SIGNED DIF BY DOMAIN (positive = item favours the focal family)", ""]
    header = (f"{'Focal family':<14}{'STEM mean d':>13}{'nSTEM mean d':>14}"
              f"{'t':>8}{'p':>11}")
    out += [header, "-" * len(header)]
    for f in focal:
        delta, _, _ = per_family[f]
        a = delta[stem & np.isfinite(delta)]
        b = delta[(~stem) & np.isfinite(delta)]
        t, pv = stats.ttest_ind(a, b, equal_var=False)
        out.append(f"{f:<14}{a.mean():>13.3f}{b.mean():>14.3f}{t:>8.2f}{pv:>11.2e}")

    out += [
        "",
        "",
        "READING",
        "",
        "This is the invariance test the psychometric literature the paper cites is",
        "actually about, and it is a different question from the one the rest of the",
        "paper asks. Item-level DIF here means that two models of equal estimated",
        "ability but different architecture lineage have systematically different",
        "odds of answering particular items correctly. Where that is widespread, a",
        "single ability estimate is not comparable across families, and leaderboard",
        "positions that mix families are comparing scores that the measurement model",
        "does not licence comparing.",
        "",
        "Two qualifications. Mantel-Haenszel matches on a single ability estimate,",
        "so if the latent space is genuinely multidimensional -- which is what the",
        "rest of this paper argues -- some of what is flagged here is that",
        "multidimensionality reappearing as DIF rather than an independent defect.",
        "The two findings are symptoms of one cause, and should be reported as",
        "such rather than as separate evidence. Second, family labels are read off",
        "model names, and a large part of the population consists of merges and",
        "fine-tunes whose lineage is mixed; the reference and focal groups are",
        "therefore noisier than examinee groups in an educational setting.",
    ]

    path = revision_report("results_family_dif.txt", out)
    print("\n".join(out))
    print(f"\nwrote {path}")


if __name__ == '__main__':
    main()
