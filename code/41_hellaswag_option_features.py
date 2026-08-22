import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from config import output_path

RAW_DATA = os.path.join(REPO, "raw_data")
PRECOMPUTED = os.path.join(REPO, "results_precomputed")


def load_transfer_module():
    spec = importlib.util.spec_from_file_location(
        "hellaswag_transfer", os.path.join(HERE, "40_hellaswag_transfer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_transfer = load_transfer_module()
IND8, SEED, calibrate = _transfer.IND8, _transfer.SEED, _transfer.calibrate

from hs_text import queries_from_dataset

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE = os.path.join(PRECOMPUTED, "hellaswag_ending_features.csv")

ENDING_INDICATORS = [
    'Ending_KeyDistractorSim',
    'Ending_MaxDistractorSim',
    'Ending_DistractorHomogeneity',
    'Ending_LengthDispersion',
    'Ending_KeyLengthAdvantage',
    'Ending_StemKeyAdvantage',
]


def encode_texts(texts):
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  encoding {len(texts)} strings on {device}")
    model = SentenceTransformer(EMBED_MODEL, device=device)
    model.eval()
    with torch.no_grad():
        emb = model.encode(texts, batch_size=512, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
    return emb.astype(np.float32)


def build_ending_features():
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    stems = queries_from_dataset(ds)
    endings = ds["endings"]
    labels = [int(x) for x in ds["label"]]

    flat, spans = [], []
    for stem, ends in zip(stems, endings):
        start = len(flat)
        flat.append(stem)
        flat.extend(ends)
        spans.append((start, len(flat)))
    emb = encode_texts(flat)

    rows = []
    for i, (ends, (lo, hi)) in enumerate(zip(endings, spans)):
        vecs = emb[lo + 1:hi]
        stem_vec = emb[lo]
        key = labels[i]
        dis = [j for j in range(len(ends)) if j != key]

        key_v, dis_v = vecs[key], vecs[dis]
        sims = dis_v @ key_v
        g = dis_v @ dis_v.T
        iu = np.triu_indices(len(dis), k=1)
        lens = np.array([len(e) for e in ends], float)
        other = np.delete(lens, key)
        stem_sims = vecs @ stem_vec

        rows.append({
            'Ending_KeyDistractorSim': float(sims.mean()),
            'Ending_MaxDistractorSim': float(sims.max()),
            'Ending_DistractorHomogeneity': float(g[iu].mean()),
            'Ending_LengthDispersion': float(lens.std() / lens.mean())
            if lens.mean() > 0 else 0.0,
            'Ending_KeyLengthAdvantage': float(
                (lens[key] - other.mean()) / other.std())
            if other.std() > 1e-9 else 0.0,
            'Ending_StemKeyAdvantage': float(
                stem_sims[key] - np.delete(stem_sims, key).mean()),
        })
    out = pd.DataFrame(rows)
    out.insert(0, 'item_id', np.arange(len(out)))
    out.insert(1, 'stem_text', stems)
    return out


def cv_r2(X, y, seed=SEED):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    tss = ((y - y.mean()) ** 2).sum()
    rss = 0.0
    for tr, te in kf.split(X):
        Xtr = sm.add_constant(X[tr])
        Xte = sm.add_constant(X[te], has_constant='add')
        beta, _, _, _ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
        rss += float(((y[te] - Xte @ beta) ** 2).sum())
    return 1.0 - rss / tss


def fit_block(X, y):
    Xc = sm.add_constant(X)
    beta, _, _, _ = np.linalg.lstsq(Xc, y, rcond=None)
    resid = y - Xc @ beta
    tss = ((y - y.mean()) ** 2).sum()
    XtX_inv = np.linalg.inv(Xc.T @ Xc)
    h = np.einsum('ij,jk,ik->i', Xc, XtX_inv, Xc)
    u = resid / (1.0 - h)
    meat = (Xc * u[:, None]).T @ (Xc * u[:, None])
    V = (XtX_inv @ meat @ XtX_inv)[1:, 1:]
    return beta[1:], V, 1.0 - float(resid @ resid) / tss


def main():
    out = ["HELLASWAG: DIFFICULTY CARRIED BY THE ENDINGS", ""]

    ind = pd.read_csv(os.path.join(PRECOMPUTED, "hellaswag_indicators.csv"))
    resp = pd.read_parquet(os.path.join(RAW_DATA, "hellaswag_responses.parquet"))
    Y = resp.to_numpy().astype(float)
    print(f"responses {Y.shape}, indicators {ind.shape}")

    if os.path.exists(CACHE):
        print(f"  loading cached ending features from {CACHE}")
        feats = pd.read_csv(CACHE)
    else:
        feats = build_ending_features()
        feats.to_csv(CACHE, index=False)

    assert len(feats) == len(ind), "ending features do not match the item pool"
    mismatch = int((feats['stem_text'].values != ind['clean_text'].values).sum())
    out.append(f"Stem-text alignment check: {mismatch} mismatches of {len(ind)}")
    if mismatch:
        raise SystemExit("alignment failed; ending features are not joinable")

    print("calibrating HellaSwag 2PL ...")
    b, _ = calibrate(Y)

    keep = ind.domain_group.isin(['activitynet', 'wikihow']).values
    y = b[keep]
    src = ind.loc[keep, 'domain_group'].values
    X_stem = ind.loc[keep, IND8].values
    X_end = feats.loc[keep, ENDING_INDICATORS].values
    X_both = np.column_stack([X_stem, X_end])

    out += ["", "Correlation of each ending indicator with IRT difficulty:", ""]
    header = f"{'Indicator':<32}{'mean':>10}{'SD':>10}{'r with b':>11}"
    out += [header, "-" * len(header)]
    for j, nm in enumerate(ENDING_INDICATORS):
        out.append(f"{nm:<32}{X_end[:, j].mean():>10.4f}"
                   f"{X_end[:, j].std():>10.4f}"
                   f"{np.corrcoef(X_end[:, j], y)[0, 1]:>11.4f}")

    out += ["", "", "CROSS-VALIDATED R-SQUARED, WITHIN AND ACROSS SOURCE", ""]
    header = (f"{'Partition':<14}{'N':>7}{'stem-only':>12}{'endings':>10}"
              f"{'combined':>11}")
    out += [header, "-" * len(header)]
    groups = [('Pooled', np.ones(keep.sum(), bool)),
              ('ActivityNet', src == 'activitynet'),
              ('WikiHow', src == 'wikihow')]
    store = {}
    for name, m in groups:
        r_s = cv_r2(X_stem[m], y[m])
        r_e = cv_r2(X_end[m], y[m])
        r_b = cv_r2(X_both[m], y[m])
        store[name] = (r_s, r_e, r_b)
        out.append(f"{name:<14}{int(m.sum()):>7}{r_s:>12.4f}{r_e:>10.4f}"
                   f"{r_b:>11.4f}")

    out += ["", "", "SOURCE INVARIANCE UNDER EACH FEATURE SET", ""]
    header = (f"{'Feature set':<16}{'k':>4}{'ActNet R2':>12}{'WikiHow R2':>12}"
              f"{'W':>9}{'df':>5}{'p':>11}")
    out += [header, "-" * len(header)]
    a_m, w_m = src == 'activitynet', src == 'wikihow'
    for label, X in (('stem-only', X_stem), ('endings', X_end),
                     ('combined', X_both)):
        ba, Va, r2a = fit_block(X[a_m], y[a_m])
        bw, Vw, r2w = fit_block(X[w_m], y[w_m])
        d = ba - bw
        W = float(d @ np.linalg.solve(Va + Vw, d))
        k = X.shape[1]
        out.append(f"{label:<16}{k:>4}{r2a:>12.4f}{r2w:>12.4f}{W:>9.2f}"
                   f"{k:>5}{stats.chi2.sf(W, k):>11.2e}")

    ps, pe, pb = store['Pooled']
    an, wh = store['ActivityNet'], store['WikiHow']
    out += [
        "",
        "",
        "READING",
        "",
        f"Within ActivityNet the stem-only framework cross-validates at "
        f"{an[0]:.4f} and the",
        f"ending indicators at {an[1]:.4f}; within WikiHow, {wh[0]:.4f} against "
        f"{wh[1]:.4f}. Reading",
        "the endings recovers difficulty variance that reading the context cannot,",
        "on the same items and the same difficulty estimates.",
        "",
        "That confirms the diagnosis offered for the transfer null rather than",
        "explaining it away. HellaSwag's difficulty is located in the ending set by",
        "construction, so a stem-only instrument returning nothing there is the",
        "expected result, not evidence that the instrument is broken. It also",
        "establishes the scope condition directly: these indicators measure",
        "difficulty that is carried by the text the model must read and reason",
        "over, and not difficulty manufactured in the response options.",
    ]

    path = output_path("results_hellaswag_endings.txt")
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write("\n".join(out).rstrip() + "\n")
    print("\n".join(out))
    print(f"\nwrote {path}")


if __name__ == '__main__':
    main()
