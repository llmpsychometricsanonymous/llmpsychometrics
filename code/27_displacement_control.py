import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import N_PERMUTATION, PERMUTATION_SEED, output_path, resolve

TOP_K = (10, 50)
BLOCK = 250
METADATA_COLS = ['question_id', 'item_id', 'subject', 'question_text',
                 'clean_text', 'choices', 'ground_truth', 'domain_group',
                 'difficulty_score', 'discrimination_score']


def top_set(scores, k):
    return np.argpartition(-scores, k - 1)[:k]


def overlap_counts(acc_matrix, ref_sets, ks):
    out = {k: np.empty(acc_matrix.shape[1], int) for k in ks}
    for j in range(acc_matrix.shape[1]):
        col = acc_matrix[:, j]
        for k in ks:
            out[k][j] = len(np.intersect1d(top_set(col, k), ref_sets[k],
                                           assume_unique=True))
    return out


def run_null(responses, pool_builder, n_draws, ref_sets, rng):
    got = {k: [] for k in TOP_K}
    sizes = []
    done = 0
    n_items = responses.shape[1]
    while done < n_draws:
        b = min(BLOCK, n_draws - done)
        M = np.zeros((n_items, b), dtype=np.float32)
        for j in range(b):
            idx = pool_builder(rng)
            M[idx, j] = 1.0
            sizes.append(len(idx))
        counts = M.sum(axis=0)
        acc = (responses @ M) / counts[None, :]
        block = overlap_counts(acc, ref_sets, TOP_K)
        for k in TOP_K:
            got[k].append(block[k])
        done += b
    return ({k: np.concatenate(v) for k, v in got.items()},
            np.asarray(sizes))


def summarise(name, counts, k, observed_overlap, out):
    disp = (k - counts) / k * 100.0
    obs = (k - observed_overlap) / k * 100.0
    hits = int(np.sum(disp >= obs))
    p = (hits + 1) / (len(disp) + 1)
    out.append(f"  {name:<22}{disp.mean():8.2f}{disp.std(ddof=1):9.2f}"
               f"{np.percentile(disp, 95):10.2f}{disp.max():9.2f}"
               f"{p:10.4f}{hits:8d}")
    return p, disp


def main():
    print("Displacement control")
    aligned = pd.read_csv(resolve("mmlu_aligned.csv"))
    model_cols = [c for c in aligned.columns if c not in METADATA_COLS]
    responses = aligned[model_cols].values.T.astype(np.float32)
    subjects = pd.Categorical(aligned['subject'].values)
    subj_codes = subjects.codes
    uniq = np.unique(subj_codes)
    is_stem = (aligned['domain_group'].values == 'STEM')
    stem_idx = np.where(is_stem)[0]
    non_idx = np.where(~is_stem)[0]
    n_stem_subj = len(np.unique(subj_codes[is_stem]))

    acc_all = responses.mean(axis=1)
    acc_stem = responses[:, stem_idx].mean(axis=1)
    acc_non = responses[:, non_idx].mean(axis=1)

    ref_sets = {k: top_set(acc_all, k) for k in TOP_K}
    obs_stem = {k: len(np.intersect1d(top_set(acc_stem, k), ref_sets[k],
                                      assume_unique=True)) for k in TOP_K}
    obs_non = {k: len(np.intersect1d(top_set(acc_non, k), ref_sets[k],
                                     assume_unique=True)) for k in TOP_K}

    out = ["SIZE-MATCHED CONTROL FOR TOP-K DISPLACEMENT", ""]
    out.append(f"Models: {responses.shape[0]}   Items: {responses.shape[1]}")
    out.append(f"STEM items: {len(stem_idx)} in {n_stem_subj} subjects   "
               f"non-STEM items: {len(non_idx)}")
    out.append(f"Null draws: {N_PERMUTATION}")
    out.append("")
    out.append("OBSERVED")
    out.append("")
    out.append("Partition        K    overlap   displacement")
    out.append("-" * 48)
    for k in TOP_K:
        out.append(f"STEM          {k:>4}    {obs_stem[k]:>2}/{k:<3}"
                   f"      {(k - obs_stem[k]) / k * 100:6.1f}%")
    for k in TOP_K:
        out.append(f"non-STEM      {k:>4}    {obs_non[k]:>2}/{k:<3}"
                   f"      {(k - obs_non[k]) / k * 100:6.1f}%")

    n_stem = len(stem_idx)
    rng = np.random.default_rng(PERMUTATION_SEED)
    rand_counts, rand_sizes = run_null(
        responses, lambda r: r.choice(responses.shape[1], n_stem,
                                      replace=False),
        N_PERMUTATION, ref_sets, rng)

    rng = np.random.default_rng(PERMUTATION_SEED)

    def subject_pool(r):
        pick = r.choice(uniq, n_stem_subj, replace=False)
        return np.where(np.isin(subj_codes, pick))[0]

    blk_counts, blk_sizes = run_null(responses, subject_pool, N_PERMUTATION,
                                     ref_sets, rng)

    results = {}
    for k in TOP_K:
        out.append("")
        out.append(f"NULL DISTRIBUTION OF DISPLACEMENT, K = {k}")
        out.append(f"  observed STEM displacement: "
                   f"{(k - obs_stem[k]) / k * 100:.1f}%")
        out.append("")
        out.append("  Null design               mean       SD      95th"
                   "      max         p  exceed")
        out.append("  " + "-" * 74)
        p_rand, _ = summarise("random items", rand_counts[k], k,
                              obs_stem[k], out)
        p_blk, _ = summarise("subject blocks", blk_counts[k], k,
                             obs_stem[k], out)
        results[k] = (p_rand, p_blk)

    out.append("")
    out.append(f"Random-item draws hold size at {n_stem} items. Subject-block")
    out.append(f"draws hold subject count at {n_stem_subj}, so item counts vary")
    out.append(f"(mean {blk_sizes.mean():.0f}, range {blk_sizes.min()}"
               f"-{blk_sizes.max()}).")

    out.append("")
    out.append("READING")
    out.append("")
    p50_rand, p50_blk = results[50]
    if p50_rand >= 0.05:
        out.append(f"The Top-50 displacement rate of "
                   f"{(50 - obs_stem[50]) / 50 * 100:.0f}% is not distinguishable")
        out.append(f"from what a random subset of the same size produces "
                   f"(p = {p50_rand:.3f}),")
        out.append(f"nor from a random union of {n_stem_subj} subjects "
                   f"(p = {p50_blk:.3f}). Scoring")
        out.append("1,000 models on a quarter of the benchmark reshuffles the")
        out.append("Top 50 by about this much whatever the items are about.")
        out.append("")
        out.append("The displacement is therefore real as a cost -- a")
        out.append("practitioner selecting on the aggregate does miss roughly")
        out.append("this fraction -- but it is a cost of estimating rank from a")
        out.append("smaller item sample, not evidence that STEM ranks models")
        out.append("differently. The rank-based claim cannot carry the")
        out.append("construct argument on its own; the accuracy-level")
        out.append("asymmetry and the subscale correlation are the load-bearing")
        out.append("evidence there.")
    else:
        out.append(f"The Top-50 displacement exceeds the size-matched null "
                   f"(p = {p50_rand:.4f}).")

    pd.DataFrame({
        'k': np.repeat(TOP_K, N_PERMUTATION),
        'design_random_overlap': np.concatenate([rand_counts[k]
                                                 for k in TOP_K]),
        'design_block_overlap': np.concatenate([blk_counts[k] for k in TOP_K]),
    }).to_csv(output_path("mmlu_displacement_null.csv"), index=False)

    text = "\n".join(out) + "\n"
    with open(output_path("results_displacement_control.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
