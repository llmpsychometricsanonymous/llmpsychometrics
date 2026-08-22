import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SEED, output_path, resolve

N_REPLICATES = 200
METADATA_COLS = ['question_id', 'item_id', 'subject', 'question_text',
                 'clean_text', 'choices', 'ground_truth', 'domain_group',
                 'difficulty_score', 'discrimination_score']


def spearman_brown(r_half, k=2.0):
    return k * r_half / (1.0 + (k - 1.0) * r_half)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def score(responses, idx):
    return responses[:, idx].mean(axis=1)


def random_halves(rng, pool, length):
    picked = rng.permutation(pool)[:2 * length]
    return picked[:length], picked[length:]


def subject_halves(rng, pool, subjects):
    subs = rng.permutation(np.unique(subjects[pool]))
    side = {0: [], 1: []}
    total = {0: 0, 1: 0}
    for s in subs:
        members = pool[subjects[pool] == s]
        target = 0 if total[0] <= total[1] else 1
        side[target].append(members)
        total[target] += len(members)
    return (rng.permutation(np.concatenate(side[0])),
            rng.permutation(np.concatenate(side[1])))


def replicate(rng, responses, stem_pool, non_pool, subjects, length, design):
    if design == 'random':
        s_halves = random_halves(rng, stem_pool, length)
        n_halves = random_halves(rng, non_pool, length)
        used = length
    else:
        s_halves = subject_halves(rng, stem_pool, subjects)
        n_halves = subject_halves(rng, non_pool, subjects)


        used = min(min(len(h) for h in s_halves),
                   min(len(h) for h in n_halves))
        s_halves = tuple(h[:used] for h in s_halves)
        n_halves = tuple(h[:used] for h in n_halves)

    s1, s2 = (score(responses, h) for h in s_halves)
    n1, n2 = (score(responses, h) for h in n_halves)

    rel_s = spearman_brown(corr(s1, s2))
    rel_n = spearman_brown(corr(n1, n2))


    half_rel_s = corr(s1, s2)
    half_rel_n = corr(n1, n2)
    cross = np.mean([corr(a, b) for a in (s1, s2) for b in (n1, n2)])
    disattenuated = cross / np.sqrt(half_rel_s * half_rel_n)

    return dict(raw=cross, rel_stem=rel_s, rel_non=rel_n,
                half_rel_stem=half_rel_s, half_rel_non=half_rel_n,
                disattenuated=disattenuated, length=used)


def summarise(rows, label, out):
    frame = pd.DataFrame(rows)
    d = frame['disattenuated'].values
    lo, hi = np.percentile(d, [2.5, 97.5])
    out.append(f"\n--- {label} ---")
    out.append(f"  replicates                     : {len(frame)}")
    out.append(f"  matched length per side        : "
               f"{frame['length'].mean():.0f} items "
               f"(min {frame['length'].min()}, max {frame['length'].max()})")
    out.append(f"  raw cross-correlation (half len): "
               f"{frame['raw'].mean():.4f}")
    out.append(f"  STEM reliability (full len, SB) : "
               f"{frame['rel_stem'].mean():.4f}")
    out.append(f"  non-STEM reliability (full len) : "
               f"{frame['rel_non'].mean():.4f}")
    out.append(f"  disattenuated correlation       : {d.mean():.4f}")
    out.append(f"  percentile interval             : [{lo:.4f}, {hi:.4f}]")
    out.append(f"  replicates reaching 1.0         : "
               f"{int((d >= 1.0).sum())} of {len(d)}")
    return frame, d.mean(), lo, hi


def main():
    aligned = pd.read_csv(resolve('mmlu_aligned.csv'))
    model_cols = [c for c in aligned.columns if c not in METADATA_COLS]
    responses = aligned[model_cols].values.T.astype(float)
    subjects = aligned['subject'].values
    stem_pool = np.where(aligned['domain_group'].values == 'STEM')[0]
    non_pool = np.where(aligned['domain_group'].values != 'STEM')[0]
    length = len(stem_pool) // 2

    out = ["SUBSCALE CORRELATION: MATCHED-LENGTH DISATTENUATED ESTIMATE", ""]
    out.append(f"Models: {responses.shape[0]}   Items: {responses.shape[1]}")
    out.append(f"STEM items: {len(stem_pool)}   "
               f"non-STEM items: {len(non_pool)}")
    out.append(f"Matched half length: {length} items per side")
    out.append(f"Replicates: {N_REPLICATES}   seed: {SEED}")
    out.append("")
    out.append("Scores are proportion correct. Reliabilities are Spearman-Brown")
    out.append("stepped up from the within-partition half correlation; the")
    out.append("cross term is disattenuated with the half-length reliabilities")
    out.append("that actually apply to it.")

    frames = {}
    for design in ('random', 'subject-disjoint'):
        rng = np.random.default_rng(SEED)
        rows = []
        for _ in range(N_REPLICATES):
            r = replicate(rng, responses, stem_pool, non_pool, subjects,
                          length, design)
            if r is not None:
                rows.append(r)
        label = ('random item splits' if design == 'random'
                 else 'subject-disjoint splits')
        frame, mean, lo, hi = summarise(rows, label, out)
        frame['design'] = design
        frames[design] = frame

    out.append("")
    out.append("Both designs are reported because the subject-disjoint one is")
    out.append("the conservative reading: removing shared subject membership")
    out.append("lowers the within-partition reliabilities in the denominator,")
    out.append("which pushes the disattenuated estimate up. An estimate that")
    out.append("stays below 1.0 under the design that inflates it most is the")
    out.append("one worth quoting against the MIRT figure.")

    combined = pd.concat(frames.values(), ignore_index=True)
    combined.to_csv(output_path("mmlu_subscale_correlation.csv"), index=False)

    text = "\n".join(out) + "\n"
    with open(output_path("results_subscale_correlation.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
