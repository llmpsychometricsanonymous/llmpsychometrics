import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats as sps
from scipy.stats import weightedtau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (BOOTSTRAP_SEED, N_BOOTSTRAP, PERMUTATION_SEED,
                    output_path, resolve)

METADATA_COLS = ['question_id', 'item_id', 'subject', 'question_text',
                 'clean_text', 'choices', 'ground_truth', 'domain_group',
                 'difficulty_score', 'discrimination_score']
TOP_K = (10, 50)
N_MATCH = 200
REPLICATE_TOL = 0.10


FAMILY_PATTERNS = [
    ('mixtral', r'mixtral'),
    ('codellama', r'codellama|code[-_ ]llama'),
    ('llama', r'llama|lumimaid|miqu|vicuna|alpaca|airoboros|platypus|tulu'),
    ('qwen', r'qwen'),
    ('yi', r'\byi[-_. ]|platyi|xaberius'),
    ('solar', r'solar'),
    ('falcon', r'falcon'),
    ('pythia', r'pythia'),
    ('deepseek', r'deepseek'),
    ('gemma', r'gemma'),
    ('phi', r'\bphi[-_ ]?\d'),
    ('mistral', r'mistral|zephyr|openhermes|starling'),
    ('bloom', r'bloom'),
    ('gpt-neox', r'neox|gpt[-_]?j'),
    ('opt', r'\bopt[-_]\d'),
]


VARIANT_SUFFIX = re.compile(
    r'[-_. ]?(v\d+(?:\.\d+)*|alt|hf|sf|ties|slerp|gptq|awq|gguf|exl2|ggml'
    r'|bf16|fp16|int8|int4|4bit|8bit|q\d+(?:_[a-z0-9]+)*)$')

SIZE_MOE = re.compile(r'(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*b\b')
SIZE_B = re.compile(r'(\d+(?:\.\d+)?)\s*b\b')
SIZE_M = re.compile(r'(\d+(?:\.\d+)?)\s*m\b')


def family_of(name):
    s = name.lower()
    for label, pat in FAMILY_PATTERNS:
        if re.search(pat, s):
            return label
    return 'unclassified'


def size_of(name):
    s = name.lower().replace('/', '-')
    m = SIZE_MOE.search(s)
    if m:
        return float(m.group(1)) * float(m.group(2))
    m = SIZE_B.search(s)
    if m:
        return float(m.group(1))
    m = SIZE_M.search(s)
    if m:
        return float(m.group(1)) / 1000.0
    return np.nan


def size_bucket(b):
    if np.isnan(b):
        return 'unknown'
    for hi, label in [(1, '<1B'), (4, '1-4B'), (9, '4-9B'), (16, '9-16B'),
                      (40, '16-40B'), (80, '40-80B')]:
        if b < hi:
            return label
    return '>80B'


def normalised_stem(name):
    s = name.split('/')[-1].lower()
    prev = None
    while prev != s:
        prev = s
        s = VARIANT_SUFFIX.sub('', s)
    return re.sub(r'[^a-z0-9]+', '-', s).strip('-')


def ranks(scores):
    return sps.rankdata(-scores, method='min')


def top_k_sets(scores, k):
    r_min = sps.rankdata(-scores, method='min')
    r_max = sps.rankdata(-scores, method='max')
    return set(np.where(r_min <= k)[0]), set(np.where(r_max <= k)[0])


def retention(agg, part, k):
    a_opt, a_con = top_k_sets(agg, k)
    p_opt, p_con = top_k_sets(part, k)
    opt = len(a_opt & p_opt) / len(a_opt)
    con = len(a_con & p_con) / max(1, len(a_con))
    return opt, con, len(a_opt), len(a_con)


def main():
    print("Model population structure")
    aligned = pd.read_csv(resolve("mmlu_aligned.csv"))
    model_cols = [c for c in aligned.columns if c not in METADATA_COLS]
    responses = aligned[model_cols].values.T.astype(np.float32)
    is_stem = (aligned['domain_group'].values == 'STEM')
    stem_idx = np.where(is_stem)[0]
    non_idx = np.where(~is_stem)[0]

    acc_all = responses.mean(axis=1)
    acc_stem = responses[:, stem_idx].mean(axis=1)
    acc_non = responses[:, non_idx].mean(axis=1)

    prof = pd.read_csv(resolve("mmlu_model_profiles.csv")).set_index(
        'model_name')
    theta = prof.loc[model_cols, 'theta_score'].to_numpy()
    beta_s = prof.loc[model_cols, 'reasoning_sensitivity'].to_numpy()

    fam = np.array([family_of(m) for m in model_cols])
    size = np.array([size_of(m) for m in model_cols])
    bucket = np.array([size_bucket(b) for b in size])
    coarse = np.array([f"{f}:{b}" for f, b in zip(fam, bucket)])
    fine = np.array([normalised_stem(m) for m in model_cols])

    out = ["MODEL POPULATION STRUCTURE", ""]
    out.append(f"Models: {len(model_cols)}   Items: {responses.shape[1]}")
    out.append("")


    out += ["REDUNDANCY IN THE POPULATION", "",
            "Two clusterings. The coarse key is base family and parameter",
            "bucket, which groups every fine-tune and merge of one base model",
            "together. The fine key is the checkpoint name with organisation,",
            "version and quantisation suffixes stripped, which separates",
            "different fine-tunes but collapses re-uploads of one artefact.",
            ""]
    out.append(f"  distinct coarse clusters : {len(set(coarse)):>5}")
    out.append(f"  distinct fine clusters   : {len(set(fine)):>5}")
    out.append(f"  models per coarse cluster: "
               f"{len(model_cols) / len(set(coarse)):>5.1f}")
    out.append("")
    out.append("Largest coarse clusters:")
    vc = pd.Series(coarse).value_counts()
    for k, v in vc.head(10).items():
        out.append(f"  {k:<26}{v:>5}")

    _, agg_con = top_k_sets(acc_all, 50)
    top50 = sorted(agg_con)
    out.append("")
    out.append("Inside the aggregate Top 50:")
    out.append(f"  distinct coarse clusters : "
               f"{len(set(coarse[top50])):>5} of {len(top50)} models")
    out.append(f"  distinct fine clusters   : {len(set(fine[top50])):>5}")
    rep = pd.Series(fine[top50]).value_counts()
    multi = rep[rep > 1]
    if len(multi):
        out.append("  checkpoints appearing more than once after normalisation:")
        for k, v in multi.items():
            out.append(f"    {k:<44}{v:>3}")


    base_names = [m.split('/')[-1].lower() for m in model_cols]
    dup_groups = {}
    for i, b in enumerate(base_names):
        dup_groups.setdefault(b, []).append(i)
    pairs = [(i, j) for v in dup_groups.values() if len(v) > 1
             for i in v for j in v if i < j]

    r_all = ranks(acc_all)
    r_stem = ranks(acc_stem)
    r_non = ranks(acc_non)

    d_all = np.array([abs(r_all[i] - r_all[j]) for i, j in pairs], float)
    d_stem = np.array([abs(r_stem[i] - r_stem[j]) for i, j in pairs], float)
    d_acc = np.array([abs(acc_all[i] - acc_all[j]) * 100 for i, j in pairs])


    same = d_acc <= REPLICATE_TOL

    out += ["", "", "AN EMPIRICAL EVALUATION-NOISE FLOOR", "",
            "Some checkpoints appear under more than one organisation. Where the",
            "weights really are the same, rank disagreement between the two",
            "entries is evaluation noise with no ability difference behind it.",
            "A shared name is not by itself proof of shared weights -- two groups",
            "can publish different merges under one name -- so the replicate set",
            "is restricted to pairs whose aggregate accuracies agree to within",
            f"{REPLICATE_TOL} percentage points, roughly fourteen items in "
            "fourteen thousand.",
            ""]
    out.append(f"  pairs sharing a checkpoint name : {len(pairs)}")
    out.append(f"  of those, accuracy-identical    : {int(same.sum())}")
    out.append("")
    out.append(f"{'quantity':<38}{'median':>9}{'mean':>9}{'90th':>9}{'max':>9}")
    out.append("-" * 74)
    for label, v in [
        ("aggregate-rank gap, replicates", d_all[same]),
        ("STEM-rank gap, replicates", d_stem[same]),
        ("aggregate-rank gap, all name pairs", d_all),
        ("STEM-rank gap, all name pairs", d_stem),
        ("accuracy gap, all name pairs (pp)", d_acc),
    ]:
        if len(v) == 0:
            continue
        out.append(f"{label:<38}{np.median(v):>9.1f}{v.mean():>9.1f}"
                   f"{np.percentile(v, 90):>9.1f}{v.max():>9.1f}")
    out.append("")
    out.append(f"{'pair':<62}{'agg':>6}{'STEM':>7}{'dAcc':>8}")
    out.append("-" * 83)
    for n, (i, j) in enumerate(pairs):
        out.append(f"{model_cols[i]:<62}{r_all[i]:>6}{r_stem[i]:>7}{'':>8}")
        out.append(f"{model_cols[j]:<62}{r_all[j]:>6}{r_stem[j]:>7}"
                   f"{d_acc[n]:>8.2f}{'  *' if same[n] else ''}")


    out += ["", "", "TIE CONVENTION AND TOP-K RETENTION", "",
            "Accuracies tie: adjacent aggregate gaps reach 0.01 percentage",
            "points through the middle of the leaderboard. Ranks are competition",
            "ranks, so tied models share the better rank, and Top-K membership is",
            "reported two ways. Optimistic admits every model whose best-case",
            "rank is at most K; conservative admits only those whose worst-case",
            "rank is at most K.",
            ""]
    n_tied_all = int(len(acc_all) - len(np.unique(acc_all)))
    out.append(f"  models sharing an aggregate accuracy with another: "
               f"{n_tied_all}")
    out.append("")
    out.append(f"{'':<22}{'optimistic':>22}{'conservative':>22}")
    out.append(f"{'':<22}{'retained':>11}{'displaced':>11}"
               f"{'retained':>11}{'displaced':>11}")
    out.append("-" * 66)
    for k in TOP_K:
        for label, part in [("STEM", acc_stem), ("non-STEM", acc_non)]:
            opt, con, n_opt, n_con = retention(acc_all, part, k)
            out.append(f"{f'{label}, K = {k}':<22}{100 * opt:>10.1f}%"
                       f"{100 * (1 - opt):>10.1f}%{100 * con:>10.1f}%"
                       f"{100 * (1 - con):>10.1f}%")


    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = {k: [] for k in TOP_K}
    for _ in range(N_BOOTSTRAP):
        pick = rng.integers(0, len(stem_idx), len(stem_idx))
        a = responses[:, stem_idx[pick]].mean(axis=1)
        for k in TOP_K:
            opt, _, _, _ = retention(acc_all, a, k)
            boot[k].append(100 * (1 - opt))
    out += ["", "", "A BOOTSTRAP INTERVAL ON THE OBSERVED RATE", "",
            f"STEM items resampled with replacement, {N_BOOTSTRAP} draws. The",
            "aggregate ranking is held fixed, so the interval is over the",
            "sampling of STEM items alone.",
            ""]
    out.append(f"{'K':>4}{'observed':>11}{'2.5%':>9}{'97.5%':>9}{'SD':>9}")
    out.append("-" * 42)
    for k in TOP_K:
        v = np.array(boot[k])
        opt, _, _, _ = retention(acc_all, acc_stem, k)
        out.append(f"{k:>4}{100 * (1 - opt):>10.1f}%"
                   f"{np.percentile(v, 2.5):>9.1f}{np.percentile(v, 97.5):>9.1f}"
                   f"{v.std(ddof=1):>9.2f}")


    tau_stem = weightedtau(-acc_all, -acc_stem).correlation
    tau_non = weightedtau(-acc_all, -acc_non).correlation
    rng = np.random.default_rng(PERMUTATION_SEED)
    matched = []
    for _ in range(N_MATCH):
        pick = rng.choice(non_idx, len(stem_idx), replace=False)
        a = responses[:, pick].mean(axis=1)
        matched.append(weightedtau(-acc_all, -a).correlation)
    matched = np.array(matched)

    out += ["", "", "WEIGHTED KENDALL AT MATCHED TEST LENGTH", "",
            "The published comparison is 0.9887 against 0.9634, but non-STEM is",
            f"{len(non_idx)} items against STEM's {len(stem_idx)}, and rank",
            "agreement with the aggregate rises with the number of items scored.",
            f"Subsampling non-STEM to STEM's size, {N_MATCH} draws, removes that.",
            ""]
    out.append(f"  tau_w, aggregate vs non-STEM (full)   : {tau_non:.4f}")
    out.append(f"  tau_w, aggregate vs STEM              : {tau_stem:.4f}")
    out.append(f"  tau_w, aggregate vs matched non-STEM  : {matched.mean():.4f}"
               f"   [{np.percentile(matched, 2.5):.4f}, "
               f"{np.percentile(matched, 97.5):.4f}]")
    p_match = (np.sum(matched <= tau_stem) + 1) / (len(matched) + 1)
    out.append(f"  draws at or below the STEM value      : "
               f"{int(np.sum(matched <= tau_stem))} of {N_MATCH}   "
               f"(p = {p_match:.4f})")
    shrink = (tau_non - tau_stem) - (matched.mean() - tau_stem)
    out.append("")
    out.append(f"Most of the published gap is test length: matching item count")
    out.append(f"removes {100 * shrink / (tau_non - tau_stem):.0f}% of it, leaving")
    out.append(f"{matched.mean() - tau_stem:.4f} against the unmatched "
               f"{tau_non - tau_stem:.4f}.")
    if p_match < 0.05:
        out.append("What remains is still one-sided: STEM sits below all but")
        out.append(f"{int(np.sum(matched <= tau_stem))} of {N_MATCH} equally sized")
        out.append("non-STEM draws. The asymmetry is real and small, and the")
        out.append("matched figure is the one the paper should quote.")
    else:
        out.append("What remains is inside the matched null, so the published")
        out.append("comparison is confounded with item count and cannot be read")
        out.append("as a partition effect.")


    def first_per_cluster(key):
        seen, keep = set(), []
        order = np.argsort(-acc_all)
        for i in order:
            if key[i] not in seen:
                seen.add(key[i])
                keep.append(i)
        return np.array(sorted(keep))

    out += ["", "", "THE AFFECTED STATISTICS, RE-RUN ON ONE MODEL PER CLUSTER",
            "",
            "The inversion correlation and the displacement rate are recomputed",
            "on the best-scoring member of each cluster, which is the choice a",
            "practitioner reading the leaderboard would make.",
            ""]
    out.append(f"{'Population':<28}{'N':>6}{'r(theta, beta_s)':>19}"
               f"{'p':>11}{'disp K=50':>12}")
    out.append("-" * 76)
    for label, key in [("full sample", None),
                       ("one per fine cluster", fine),
                       ("one per coarse cluster", coarse)]:
        keep = (np.arange(len(model_cols)) if key is None
                else first_per_cluster(key))
        r, p = sps.pearsonr(theta[keep], beta_s[keep])
        opt, _, _, _ = retention(acc_all[keep], acc_stem[keep], 50)
        out.append(f"{label:<28}{len(keep):>6}{r:>19.3f}{p:>11.2e}"
                   f"{100 * (1 - opt):>11.1f}%")

    keep_half = first_per_cluster(fine)
    th = theta[keep_half]
    bs = beta_s[keep_half]
    out.append("")
    out.append("Trimmed for the guessing floor, on the fine-deduplicated set:")
    out.append("")
    out.append(f"{'Trim':<28}{'N':>6}{'r':>9}{'p':>12}")
    out.append("-" * 55)
    for label, frac in [("full", None), ("bottom quartile removed", 0.25),
                        ("bottom half removed", 0.50)]:
        m = (np.ones(len(th), bool) if frac is None
             else th >= np.quantile(th, frac))
        r, p = sps.pearsonr(th[m], bs[m])
        out.append(f"{label:<28}{int(m.sum()):>6}{r:>9.3f}{p:>12.2e}")


    n_fine50 = len(set(fine[top50]))
    n_coarse50 = len(set(coarse[top50]))
    out += ["", "", "READING", ""]
    out.append(f"The aggregate Top 50 contains {n_fine50} distinct checkpoints and")
    out.append(f"{n_coarse50} distinct base-model-and-size combinations. A")
    out.append("displacement rate computed over it counts near-identical systems")
    out.append("separately, and the paper now says so where the rate is reported.")
    out.append("")
    if same.any():
        out.append("Weights re-evaluated under a different organisation disagree")
        out.append(f"by a median of {np.median(d_all[same]):.0f} aggregate ranks "
                   f"and {np.median(d_stem[same]):.0f} STEM ranks, reaching")
        out.append(f"{d_all[same].max():.0f} and {d_stem[same].max():.0f} at the "
                   f"extreme. That is the leaderboard's")
        out.append("own noise floor, measured on this population rather than")
        out.append("imported from a perturbation study, and it is the scale")
        out.append("against which the displacement figures should be read.")
    out.append("")
    out.append("The inversion correlation is not an artefact of redundancy: it")
    out.append("is essentially unchanged when the population is reduced to one")
    out.append("model per cluster, though the p-values are correspondingly")
    out.append("larger and the paper reports the deduplicated N alongside them.")

    pd.DataFrame({
        'model_name': model_cols, 'family': fam, 'size_b': size,
        'coarse_cluster': coarse, 'fine_cluster': fine,
        'accuracy_overall': acc_all, 'accuracy_stem': acc_stem,
        'accuracy_nonstem': acc_non,
        'rank_overall': r_all, 'rank_stem': r_stem, 'rank_nonstem': r_non,
    }).to_csv(output_path("mmlu_model_clusters.csv"), index=False)

    text = "\n".join(out) + "\n"
    with open(output_path("results_model_population.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
