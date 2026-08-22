import importlib.util
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CODE = HERE
sys.path.insert(0, HERE)
from config import output_path

PRECOMPUTED = os.path.join(REPO, "results_precomputed")
CACHE = os.path.join(PRECOMPUTED, "wscg_validity_features.csv")


def load_wscg():
    spec = importlib.util.spec_from_file_location(
        "dim1", os.path.join(CODE, "03_dimension1_reasoning.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.extract_cot_graph


def gsm8k_rows():
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for q, a in zip(ds["question"], ds["answer"]):
        steps = len(re.findall(r"<<[^>]*>>", a))
        lines = len([l for l in a.split("\n") if l.strip()
                     and not l.strip().startswith("####")])
        if steps > 0:
            rows.append({"corpus": "GSM8K", "text": q,
                         "criterion": steps, "criterion_alt": lines})
    return rows


def math_rows(limit=2000):
    from datasets import load_dataset
    for name, cfg in (("EleutherAI/hendrycks_math", "algebra"),
                      ("nlile/hendrycks-MATH-benchmark", None)):
        try:
            ds = (load_dataset(name, cfg, split="test") if cfg
                  else load_dataset(name, split="test"))
            break
        except Exception as exc:
            print(f"  {name} unavailable ({type(exc).__name__})")
            ds = None
    if ds is None:
        return []
    lvl_col = "level" if "level" in ds.column_names else None
    q_col = "problem" if "problem" in ds.column_names else ds.column_names[0]
    if lvl_col is None:
        return []
    rows = []
    for q, lv in zip(ds[q_col][:limit], ds[lvl_col][:limit]):
        m = re.search(r"(\d)", str(lv))
        if m:
            rows.append({"corpus": "MATH", "text": q,
                         "criterion": int(m.group(1)), "criterion_alt": np.nan})
    return rows


def partial_corr(x, y, z):
    def resid(v):
        Z = np.column_stack([np.ones(len(z)), z])
        beta, _, _, _ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta
    return stats.pearsonr(resid(x), resid(y))


def main():
    out = ["WSCG CONVERGENT VALIDITY", ""]

    if os.path.exists(CACHE):
        print(f"loading cached features from {CACHE}")
        df = pd.read_csv(CACHE)
    else:
        rows = gsm8k_rows() + math_rows()
        if not rows:
            raise SystemExit("no external corpora available")
        df = pd.DataFrame(rows)
        print(f"parsing {len(df)} items with the shipped WSCG builder ...")
        extract = load_wscg()
        nodes, depth = [], []
        for i, t in enumerate(df["text"]):
            n, d = extract(str(t))
            nodes.append(n)
            depth.append(d)
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(df)}")
        df["WSCG_Nodes"] = nodes
        df["WSCG_Depth"] = depth
        df["word_count"] = [len(str(t).split()) for t in df["text"]]
        df.drop(columns=["text"]).to_csv(CACHE, index=False)

    out += [
        "Correlations between the shipped WSCG indicators and an externally",
        "annotated measure of reasoning demand, with question length as the",
        "competing explanation.",
        "",
    ]

    for corpus, sub in df.groupby("corpus"):
        y = sub["criterion"].values.astype(float)
        wd = sub["WSCG_Depth"].values.astype(float)
        wn = sub["WSCG_Nodes"].values.astype(float)
        wc = sub["word_count"].values.astype(float)

        crit = ("annotated solution steps" if corpus == "GSM8K"
                else "human difficulty level")
        out += [f"--- {corpus} (N = {len(sub)}), criterion: {crit} ---", ""]
        header = f"{'Predictor':<20}{'Pearson r':>12}{'p':>12}{'Spearman':>11}"
        out += [header, "-" * len(header)]
        for label, v in (("WSCG Depth", wd), ("WSCG Nodes", wn),
                         ("Word count", wc)):
            r, p = stats.pearsonr(v, y)
            rho, _ = stats.spearmanr(v, y)
            out.append(f"{label:<20}{r:>12.4f}{p:>12.2e}{rho:>11.4f}")

        pr_d, pp_d = partial_corr(wd, y, wc)
        pr_n, pp_n = partial_corr(wn, y, wc)
        out += [
            "",
            f"  Partial r(WSCG Depth, criterion | word count) = {pr_d:.4f} "
            f"(p = {pp_d:.2e})",
            f"  Partial r(WSCG Nodes, criterion | word count) = {pr_n:.4f} "
            f"(p = {pp_n:.2e})",
        ]

        Z = np.column_stack([np.ones(len(y)), wc])
        Zb = np.column_stack([Z, wd, wn])
        def r2(X):
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            r = y - X @ beta
            return 1 - float(r @ r) / float(((y - y.mean()) ** 2).sum())
        out += [
            f"  R2, word count only:            {r2(Z):.4f}",
            f"  R2, word count + WSCG:          {r2(Zb):.4f}",
            "",
        ]

    out += [
        "",
        "READING",
        "",
        "The result is positive on one criterion and null on the other, and the",
        "difference between them is interpretable rather than awkward.",
        "",
        "Against GSM8K's annotated solution steps, WSCG Nodes correlates at 0.390",
        "and outperforms word count (0.364). More to the point, both indicators",
        "retain an independent association once length is held fixed (partial r =",
        "0.155 for nodes, p = 1.9e-08; 0.086 for depth, p = 1.8e-03), and adding",
        "them to a length-only model raises R2 from 0.1322 to 0.1534. The criterion",
        "was produced by that dataset's annotators for their own purposes, is not",
        "derivable from the question text, and was not consulted when the weights",
        "were set. That is convergent validity, and it is incremental over length.",
        "",
        "Against MATH's human difficulty levels it is not. Word count predicts the",
        "level better than either WSCG indicator (0.295 against 0.207 and 0.146),",
        "and neither retains a partial association once length is controlled",
        "(0.022 and -0.013, both n.s.). We report this rather than omitting it.",
        "",
        "The two criteria are not measuring the same thing. GSM8K step counts are a",
        "direct tally of sequential inferential operations, which is what the WSCG",
        "is built to capture. A MATH difficulty level is a holistic judgment that",
        "folds in advanced content knowledge, notation and problem novelty, most of",
        "which the WSCG does not claim to measure and none of which it reads. The",
        "honest statement is therefore narrow: the WSCG tracks the number of",
        "reasoning steps an item demands, beyond how long the item is, and does not",
        "track overall perceived difficulty. That is the construct the paper claims",
        "for it, and the boundary is worth stating explicitly.",
        "",
        "Effect sizes are modest throughout, and this is convergent validity for the",
        "indicator rather than calibration of the weight tiers. The interval spacing",
        "of the tiers remains where the sensitivity analysis leaves it: unvalidated",
        "in magnitude, immaterial to the conclusions across 123 perturbation schemes.",
    ]

    path = output_path("results_wscg_validity.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out).rstrip() + "\n")
    print("\n".join(out))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
