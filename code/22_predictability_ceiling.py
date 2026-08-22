import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CV_SEED, INDICATORS, N_FOLDS, output_path, resolve, revision_report

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMB_CACHE = "mmlu_item_embeddings.npy"
OPTION_CACHE = "mmlu_option_features.csv"
ALPHAS = np.logspace(-1, 4, 24)

OPTION_INDICATORS = [
    'Option_KeyDistractorSim', 'Option_MaxDistractorSim',
    'Option_DistractorHomogeneity', 'Option_LengthDispersion',
    'Option_KeyLengthAdvantage', 'Option_NumericProximity',
    'Option_MetaOption', 'Option_StemKeyAdvantage',
]


def full_item_text(row):
    try:
        import ast
        opts = ast.literal_eval(row['choices'])
        opts = [str(o) for o in opts]
    except (ValueError, SyntaxError, TypeError):
        opts = []
    return str(row['clean_text']) + " " + " ".join(opts)


def encode(texts):
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  encoding {len(texts)} items on {device}")
    model = SentenceTransformer(EMBED_MODEL, device=device)
    model.eval()
    with torch.no_grad():
        return model.encode(texts, batch_size=256, convert_to_numpy=True,
                            normalize_embeddings=True,
                            show_progress_bar=False).astype(np.float32)


def cv_r2(X, y, seed=CV_SEED, ridge=False):
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    tss = ((y - y.mean()) ** 2).sum()
    rss = 0.0
    for tr, te in kf.split(X):
        if ridge:
            m = RidgeCV(alphas=ALPHAS).fit(X[tr], y[tr])
            pred = m.predict(X[te])
        else:
            Xtr = np.column_stack([np.ones(len(tr)), X[tr]])
            Xte = np.column_stack([np.ones(len(te)), X[te]])
            beta, _, _, _ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
            pred = Xte @ beta
        rss += float(((y[te] - pred) ** 2).sum())
    return 1.0 - rss / tss


def main():
    print("Predictability ceiling")
    df = pd.read_csv(resolve("mmlu_dimension5_adversarial.csv"))
    df = df.dropna(subset=INDICATORS + ['difficulty_score', 'domain_group'])
    y = df['difficulty_score'].values

    emb_path = resolve(EMB_CACHE)
    if os.path.exists(emb_path):
        print(f"  loading cached embeddings from {emb_path}")
        E = np.load(emb_path)
        if len(E) != len(df):
            raise SystemExit("cached embeddings do not match the item pool")
    else:
        E = encode([full_item_text(r) for _, r in df.iterrows()])
        np.save(output_path(EMB_CACHE), E)

    opt_path = resolve(OPTION_CACHE)
    have_options = os.path.exists(opt_path)
    if have_options:
        opt = pd.read_csv(opt_path)
        for c in OPTION_INDICATORS:
            df[c] = opt[c].values
    else:
        print("  option features not found; run stage 21 first for the full table")


    S = pd.get_dummies(df['subject'], drop_first=True).values.astype(float)

    X_ind = df[INDICATORS].values
    sets = [("Nine indicators", X_ind, False)]
    if have_options:
        X_opt = df[OPTION_INDICATORS].values
        sets.append(("+ option indicators", np.column_stack([X_ind, X_opt]), False))
    sets += [
        ("Subject identity only", S, True),
        ("Indicators + subject", np.column_stack([X_ind, S]), True),
        ("Embeddings (ceiling)", E, True),
        ("Embeddings + subject", np.column_stack([E, S]), True),
    ]
    if have_options:
        sets.append(("Embeddings + all indicators",
                     np.column_stack([E, X_ind, df[OPTION_INDICATORS].values]), True))

    stem_mask = (df['domain_group'] == 'STEM').values
    partitions = [('Pooled', np.ones(len(df), bool)),
                  ('STEM', stem_mask), ('non-STEM', ~stem_mask)]

    out = ["PREDICTABILITY CEILING", ""]
    out += [
        "Cross-validated R-squared under a common 5-fold split. The encoder is",
        f"{EMBED_MODEL} applied to the stem and options together, which is what",
        "the model was scored on. Ridge penalty selected within each training",
        "fold, so no test information enters the fit.",
        "",
    ]
    header = f"{'Predictor set':<30}{'Pooled':>10}{'STEM':>10}{'non-STEM':>11}"
    out += [header, "-" * len(header)]
    table = {}
    for name, X, ridge in sets:
        row = []
        for _, mask in partitions:
            row.append(cv_r2(X[mask], y[mask], ridge=ridge))
        table[name] = row
        out.append(f"{name:<30}{row[0]:>10.4f}{row[1]:>10.4f}{row[2]:>11.4f}")

    ceil = table["Embeddings (ceiling)"]
    ceil_hi = table["Embeddings + subject"]
    ind_row = table["Nine indicators"]
    out += [
        "", "", "THE FRAMEWORK AS A FRACTION OF THE CEILING", "",
        "Two denominators are reported. The encoder ceiling is the best purely",
        "text-based predictor; the encoder-plus-subject ceiling adds topic",
        "membership, which is recoverable from the text in principle and is the",
        "more conservative comparison. Percentages against the smaller denominator",
        "should be read with the caveat that ridge over 384 dimensions is estimated",
        "on fewer items inside a partition than pooled, so the ceiling is itself",
        "less precisely determined in STEM than it is overall.",
        "",
    ]
    header = (f"{'Partition':<14}{'Framework':>11}{'Ceiling':>10}{'% ceil':>9}"
              f"{'+subject':>11}{'% ceil+s':>11}")
    rows = [("Nine indicators", ind_row)]
    if have_options:
        rows.append(("+ option indicators", table["+ option indicators"]))
    for label, vals in rows:
        out += ["", f"{label}:", "", header, "-" * len(header)]
        for i, (name, _) in enumerate(partitions):
            f1 = 100 * vals[i] / ceil[i] if ceil[i] > 0 else float('nan')
            f2 = 100 * vals[i] / ceil_hi[i] if ceil_hi[i] > 0 else float('nan')
            out.append(f"{name:<14}{vals[i]:>11.4f}{ceil[i]:>10.4f}{f1:>8.1f}%"
                       f"{ceil_hi[i]:>11.4f}{f2:>10.1f}%")

    subj = table["Subject identity only"]
    out += [
        "",
        "",
        "READING",
        "",
        f"A dense encoder reading the whole item reaches {ceil[0]:.4f} pooled,",
        f"{ceil[1]:.4f} in STEM. Most difficulty is therefore not recoverable from",
        "the item text by any method: the ceiling is low in absolute terms, which",
        "is the honest context for the framework's own share and cannot be read off",
        "an R-squared against total variance.",
        "",
        f"Subject identity alone reaches {subj[0]:.4f} pooled. A large part of what",
        "an encoder knows about an MMLU item is which subject it came from, which",
        "is topic membership rather than structural complexity. The nine indicators",
        "are not a proxy for topic: they are computed from text structure and carry",
        "no subject label.",
        "",
        "The comparison worth reporting is the framework against the ceiling, not",
        "against 1.0. Against a ceiling estimated the same way on the same folds,",
        f"the nine indicators recover {100 * ind_row[1] / ceil[1]:.0f}% of "
        f"text-predictable difficulty in STEM and",
        f"{100 * ind_row[2] / ceil[2]:.0f}% outside it, while remaining "
        "interpretable and theory-grounded, which",
        "a 384-dimensional embedding is not.",
    ]
    if have_options:
        both = table["+ option indicators"]
        out += [
            "",
            f"Adding the option indicators takes that to "
            f"{100 * both[1] / ceil[1]:.0f}% of the encoder ceiling in STEM",
            f"({100 * both[1] / ceil_hi[1]:.0f}% against the conservative "
            f"encoder-plus-subject ceiling). Seventeen",
            "interpretable features recover most of what a dense representation of",
            "the same item can extract. The residual gap is the part of difficulty",
            "that is neither structural nor topical, and on this evidence it is",
            "small relative to the part that is simply not in the text at all.",
        ]

    path = revision_report("results_predictability_ceiling.txt", out)
    print("\n".join(out))
    print(f"\nwrote {path}")


if __name__ == '__main__':
    main()
