import pandas as pd
import numpy as np
import os
import stanza
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from config import RESULTS_DIR, resolve

os.makedirs(RESULTS_DIR, exist_ok=True)

stanza_config = {'lang': 'en', 'processors': 'tokenize,pos,lemma,depparse',
                 'use_gpu': True, 'seed': 1729, 'download_method': None}

try:
    nlp = stanza.Pipeline(**stanza_config)
except Exception:
    stanza.download('en', processors='tokenize,pos,lemma,depparse')
    nlp = stanza.Pipeline(**stanza_config)

NEGATION_LEXICON = {
    'not', 'no', 'never', 'none', 'neither', 'nor', 'cannot', "n't", 'cant', 'dont', 'doesnt', 'didnt',
    'wont', 'wouldnt', 'shouldnt', 'couldnt', 'isnt', 'arent', 'wasnt', 'werent', 'hasnt', 'havent',
    'hadnt', 'aint', 'except', 'unless', 'excluding', 'without', 'barring', 'irrespective', 'regardless', 'exempt',
    'negate', 'refute', 'contradict', 'disprove', 'falsify', 'invalidate', 'nullify', 'preclude', 'prohibit',
    'deny', 'reject', 'impossible', 'invalid', 'incorrect', 'false', 'untrue', 'inconsistent', 'incompatible',
    'irrational', 'illogical', 'erroneous', 'inaccurate', 'rarely', 'scarcely', 'hardly', 'seldom', 'barely'
}

def extract_negation_and_word_count(text):
    if not isinstance(text, str) or text.strip() == "":
        return 0, 0
    try:
        doc = nlp(text)
        neg_scope = 0
        word_count = 0

        for sentence in doc.sentences:
            word_count += len(sentence.words)
            sent_nodes = {w.id: w for w in sentence.words}
            scope_tokens = set()

            for word in sentence.words:
                if word.text.lower() in NEGATION_LEXICON or (word.lemma and word.lemma.lower() in NEGATION_LEXICON):
                    head_id = word.head
                    scope_tokens.add(word.id)
                    if head_id != 0 and head_id in sent_nodes:
                        scope_tokens.add(head_id)
                        for w in sentence.words:
                            if w.head == head_id:
                                scope_tokens.add(w.id)

            neg_scope += len(scope_tokens)

        return neg_scope, word_count
    except Exception as e:
        return 0, 0

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_path = resolve("mmlu_dimension4_semantics.csv")
    out_path = os.path.join(RESULTS_DIR, "mmlu_dimension5_adversarial.csv")
    cached_out = resolve("mmlu_dimension5_adversarial.csv")

    df_data = pd.read_csv(in_path)
    print("Extracting Dimension 5 Adversarial indicators...")

    cache_path = resolve("mmlu_dimension5_cache.csv")
    if not os.path.exists(cache_path):
        cache_path = resolve(".mmlu_dimension5_cache.csv")
    cache_loaded = False
    if os.path.exists(cached_out) and os.path.exists(cache_path):
        try:
            df_cache = pd.read_csv(cache_path)
            if len(df_cache) == len(df_data) and 'adversarial_raw_count' in df_cache.columns and 'word_count' in df_cache.columns:
                df_data['adversarial_raw_count'] = df_cache['adversarial_raw_count']
                df_data['word_count'] = df_cache['word_count']
                if not df_data['adversarial_raw_count'].isna().any() and not df_data['word_count'].isna().any():
                    print("Found fully computed Dimension 5 indicators in sidecar cache. Skipping extraction.")
                    cache_loaded = True
        except Exception:
            pass

    if not cache_loaded:
        df_data['adversarial_raw_count'] = np.nan
        df_data['word_count'] = np.nan
        extraction_indices = df_data.index

        if len(extraction_indices) > 0:
            for idx in tqdm(
                    extraction_indices,
                    desc="Extracting Comprehensive Dimension 5 Negations"):
                row = df_data.loc[idx]
                text = row['clean_text']

                neg_raw, word_n = extract_negation_and_word_count(text)
                df_data.at[idx, 'adversarial_raw_count'] = neg_raw
                df_data.at[idx, 'word_count'] = word_n

    try:
        df_data[['question_id', 'adversarial_raw_count', 'word_count']].to_csv(cache_path, index=False)
    except Exception:
        pass

    X = df_data[['word_count']].values
    y = df_data['adversarial_raw_count'].values

    ols = LinearRegression().fit(X, y)
    beta = ols.coef_[0]
    alpha = ols.intercept_
    predictions = ols.predict(X)

    df_data['Adversarial_Score'] = y - predictions

    r, p_val = pearsonr(
        df_data['Adversarial_Score'], df_data['word_count'])
    print(f"Orthogonality Check (Adversarial Score vs Word Count): r={r:.4f}, p={p_val:.3e}")

    keep_cols = [
        'question_id', 'item_id', 'subject', 'question_text', 'clean_text',
        'choices', 'ground_truth', 'domain_group', 'difficulty_score',
        'discrimination_score', 'WSCG_Nodes', 'WSCG_Depth',
        'Syntactic_MDD', 'Syntactic_Depth',
        'Knowledge_NER_Density', 'Knowledge_Zipf_Rarity',
        'Semantic_Concreteness', 'AMR_Depth',
        'Adversarial_Score'
    ]
    df_data_stripped = df_data[keep_cols]
    df_data_stripped.to_csv(out_path, index=False)
    print(f"Completed Dimension 5. All associated files saved to {out_path}")

    results_path = os.path.join(RESULTS_DIR, "results_summary.txt")
    with open(results_path, "a", encoding="utf-8") as f:
        f.write("\n\nDIMENSION 5 - ADVERSARIAL NEGATIONS\n\n")
        for indicator in ['word_count', 'Adversarial_Score']:
            val_max = df_data[indicator].max()
            val_min = df_data[indicator].min()
            val_mean = df_data[indicator].mean()
            f.write(f" {indicator}\n")
            f.write(f"    Max: {val_max:>6.4f} | Min: {val_min:>6.4f}\n")
            f.write(f"    Mean: {val_mean:>6.4f}\n\n")

if __name__ == '__main__':
    main()
