import pandas as pd
import numpy as np
import os
import stanza
from tqdm import tqdm
import random as python_random
from config import RESULTS_DIR, resolve

os.makedirs(RESULTS_DIR, exist_ok=True)

try:
    nlp = stanza.Pipeline(
        'en', processors='tokenize,pos,lemma,depparse', use_gpu=True, seed=1729, download_method=None)
except Exception:
    stanza.download('en', processors='tokenize,pos,lemma,depparse')
    nlp = stanza.Pipeline(
        'en', processors='tokenize,pos,lemma,depparse', use_gpu=True, seed=1729, download_method=None)
python_random.seed(1729)
np.random.seed(1729)

def syntactic_features(text):
    if not isinstance(text, str) or text.strip() == "":
        return 0.0, 0
    try:
        doc = nlp(text)
        mdd_sum = 0
        mdd_count = 0
        max_tree_depth = 0

        for sentence in doc.sentences:
            sent_nodes = {w.id: w for w in sentence.words}

            paths = {}
            for w in sentence.words:
                depth = 0
                current = w
                while current.head != 0:
                    depth += 1
                    if current.head in sent_nodes:
                        current = sent_nodes[current.head]
                    else:
                        break
                paths[w.id] = depth
            if paths:
                max_tree_depth = max(max_tree_depth, max(paths.values()))

            for word in sentence.words:
                if word.head != 0:
                    mdd_sum += abs(word.id - word.head)
                    mdd_count += 1

        mdd = round(mdd_sum / mdd_count, 4) if mdd_count > 0 else 0.0
        return mdd, max_tree_depth
    except Exception as e:
        return 0.0, 0

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_path = resolve("mmlu_dimension1_CoT_reasoning.csv")
    out_path = os.path.join(RESULTS_DIR, "mmlu_dimension2_syntax.csv")
    cached_out = resolve("mmlu_dimension2_syntax.csv")

    print("Extracting Dimension 2 Syntactic indicators...")

    df_data = pd.read_csv(in_path)

    cache_loaded = False
    if os.path.exists(cached_out):
        try:
            df_out = pd.read_csv(cached_out)
            if len(df_out) == len(df_data) and 'Syntactic_MDD' in df_out.columns and 'Syntactic_Depth' in df_out.columns:
                df_data['Syntactic_MDD'] = df_out['Syntactic_MDD']
                df_data['Syntactic_Depth'] = df_out['Syntactic_Depth']
                if not df_data['Syntactic_MDD'].isna().any() and not df_data['Syntactic_Depth'].isna().any():
                    print("Found fully computed Dimension 2 indicators in output cache. Skipping extraction.")
                    cache_loaded = True
        except Exception:
            pass

    if not cache_loaded:
        df_data['Syntactic_MDD'] = np.nan
        df_data['Syntactic_Depth'] = np.nan
        extraction_indices = df_data.index

        if len(extraction_indices) > 0:
            for idx in tqdm(
                    extraction_indices,
                    desc="Extracting Dimension 2 Syntactic Indicators"):
                row = df_data.loc[idx]
                text = row['clean_text']

                mdd, tree_h = syntactic_features(text)
                df_data.at[idx, 'Syntactic_MDD'] = mdd
                df_data.at[idx, 'Syntactic_Depth'] = tree_h

    keep_cols = [
        'question_id', 'item_id', 'subject', 'question_text', 'clean_text',
        'choices', 'ground_truth', 'domain_group', 'difficulty_score',
        'discrimination_score', 'WSCG_Nodes', 'WSCG_Depth',
        'Syntactic_MDD', 'Syntactic_Depth'
    ]
    df_data_stripped = df_data[keep_cols]
    df_data_stripped.to_csv(out_path, index=False)

    print(f"Completed Dimension 2. All associated files saved to {out_path}")

    results_path = os.path.join(RESULTS_DIR, "results_summary.txt")
    with open(results_path, "a", encoding="utf-8") as f:
        f.write("\n\nDIMENSION 2 - SYNTACTIC COMPLEXITY\n\n")
        for indicator in ['Syntactic_MDD', 'Syntactic_Depth']:
            val_max = df_data_stripped[indicator].max()
            val_min = df_data_stripped[indicator].min()
            val_sd = df_data_stripped[indicator].std()
            f.write(f" {indicator}\n")
            f.write(f"    Max: {val_max:>6.4f} | Min: {val_min:>6.4f}\n")
            f.write(f"    SD:  {val_sd:>6.4f}\n\n")

if __name__ == '__main__':
    main()
