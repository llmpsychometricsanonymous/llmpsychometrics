import pandas as pd
import numpy as np
import os
import string
import spacy
import wordfreq
from tqdm import tqdm
import random as python_random
from config import RESULTS_DIR, resolve

os.makedirs(RESULTS_DIR, exist_ok=True)

spacy.util.fix_random_seed(1729)
python_random.seed(1729)
np.random.seed(1729)
SPACY_DISABLE = ["tagger", "parser", "attribute_ruler", "lemmatizer"]

try:
    nlp = spacy.load("en_core_web_sm", disable=SPACY_DISABLE)
except OSError:
    from spacy.cli import download as spacy_download
    spacy_download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=SPACY_DISABLE)

def extract_knowledge_features(text):
    if not isinstance(text, str) or text.strip() == "":
        return 0.0, np.nan

    doc = nlp(text)
    total_tokens = len(doc)

    ner_tokens = sum(len(ent) for ent in doc.ents)
    ner_density = (ner_tokens / total_tokens *
                   100.0) if total_tokens > 0 else 0.0

    translator = str.maketrans('', '', string.punctuation)
    words = text.translate(translator).split()

    rarities = []
    for w in words:
        w_clean = w.lower().strip()
        if not w_clean or w_clean.isdigit() or w_clean in nlp.Defaults.stop_words:
            continue
        freq = wordfreq.zipf_frequency(w_clean, 'en')
        if freq > 0:

            rarities.append(8.0 - freq)

    zipf_rarity = np.mean(rarities) if rarities else np.nan
    return round(ner_density, 4), round(zipf_rarity, 4) if not np.isnan(zipf_rarity) else np.nan

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_path = resolve("mmlu_dimension2_syntax.csv")
    out_path = os.path.join(RESULTS_DIR, "mmlu_dimension3_knowledge.csv")
    cached_out = resolve("mmlu_dimension3_knowledge.csv")

    print("Extracting Dimension 3 Knowledge indicators...")
    df_data = pd.read_csv(in_path)

    cache_loaded = False
    if os.path.exists(cached_out):
        try:
            df_out = pd.read_csv(cached_out)
            if len(df_out) == len(df_data) and 'Knowledge_NER_Density' in df_out.columns and 'Knowledge_Zipf_Rarity' in df_out.columns:
                df_data['Knowledge_NER_Density'] = df_out['Knowledge_NER_Density']
                df_data['Knowledge_Zipf_Rarity'] = df_out['Knowledge_Zipf_Rarity']
                if not df_data['Knowledge_NER_Density'].isna().any() and not df_data['Knowledge_Zipf_Rarity'].isna().any():
                    print("Found fully computed Dimension 3 indicators in output cache. Skipping extraction.")
                    cache_loaded = True
        except Exception:
            pass

    if not cache_loaded:
        df_data['Knowledge_NER_Density'] = np.nan
        df_data['Knowledge_Zipf_Rarity'] = np.nan
        extraction_indices = df_data.index

        if len(extraction_indices) > 0:
            for idx in tqdm(
                    extraction_indices,
                    desc="Extracting Dimension 3 Knowledge Indicators"):
                row = df_data.loc[idx]
                text = row['clean_text']

                ner_d, zipf_r = extract_knowledge_features(text)
                df_data.at[idx, 'Knowledge_NER_Density'] = ner_d
                df_data.at[idx, 'Knowledge_Zipf_Rarity'] = zipf_r

    null_rarity = df_data['Knowledge_Zipf_Rarity'].isna().sum()
    null_ner = df_data['Knowledge_NER_Density'].isna().sum()

    if null_rarity > 0 or null_ner > 0:
        df_data['Knowledge_Zipf_Rarity'] = df_data['Knowledge_Zipf_Rarity'].fillna(
            df_data['Knowledge_Zipf_Rarity'].mean())
        df_data['Knowledge_NER_Density'] = df_data['Knowledge_NER_Density'].fillna(
            0.0)

    keep_cols = [
        'question_id', 'item_id', 'subject', 'question_text', 'clean_text',
        'choices', 'ground_truth', 'domain_group', 'difficulty_score',
        'discrimination_score', 'WSCG_Nodes', 'WSCG_Depth',
        'Syntactic_MDD', 'Syntactic_Depth',
        'Knowledge_NER_Density', 'Knowledge_Zipf_Rarity'
    ]
    df_data_stripped = df_data[keep_cols]
    df_data_stripped.to_csv(out_path, index=False)

    print(f"Completed Dimension 3. All associated files saved to {out_path}")

    results_path = os.path.join(RESULTS_DIR, "results_summary.txt")
    with open(results_path, "a", encoding="utf-8") as f:
        f.write("\n\nDIMENSION 3 - KNOWLEDGE SPECIFICITY\n\n")
        for indicator in ['Knowledge_NER_Density', 'Knowledge_Zipf_Rarity']:
            val_max = df_data_stripped[indicator].max()
            val_min = df_data_stripped[indicator].min()
            val_mean = df_data_stripped[indicator].mean()
            f.write(f" {indicator}\n")
            f.write(f"    Max: {val_max:>6.4f} | Min: {val_min:>6.4f}\n")
            f.write(f"    Mean: {val_mean:>6.4f}\n\n")

if __name__ == '__main__':
    main()
