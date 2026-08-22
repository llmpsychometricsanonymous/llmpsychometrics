import pandas as pd
import numpy as np
import os
import pickle
import stanza
from tqdm import tqdm
from config import RESULTS_DIR, resolve

os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH = 64

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    in_path = resolve("mmlu_dimension1_CoT_reasoning.csv")
    existing_cache = resolve("mmlu_parse_cache.pkl")
    cache_path = os.path.join(RESULTS_DIR, "mmlu_parse_cache.pkl")

    df = pd.read_csv(in_path)
    print(f"Loaded {len(df)} items from {os.path.basename(in_path)}")

    if os.path.exists(existing_cache):
        with open(existing_cache, "rb") as f:
            cache = pickle.load(f)
        if len(cache) == len(df):
            print("Parse cache already complete. Nothing to do.")
            return
        print(f"Found partial cache ({len(cache)}), rebuilding.")

    try:
        nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse',
                              use_gpu=True, seed=1729, download_method=None)
    except Exception:
        stanza.download('en', processors='tokenize,pos,lemma,depparse')
        nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse',
                              use_gpu=True, seed=1729, download_method=None)

    texts = df['clean_text'].fillna("").astype(str).tolist()
    cache = []

    for start in tqdm(range(0, len(texts), BATCH), desc="Parsing"):
        chunk = texts[start:start + BATCH]
        try:
            docs = nlp.bulk_process(chunk)
        except Exception:
            docs = []
            for t in chunk:
                try:
                    docs.append(nlp(t))
                except Exception:
                    docs.append(None)

        for doc in docs:
            if doc is None:
                cache.append(None)
                continue
            item = []
            for sentence in doc.sentences:
                sent = [(w.text.lower(), (w.lemma or "").lower(), w.upos,
                         w.id, w.head) for w in sentence.words]
                item.append(sent)
            cache.append(item)

    assert len(cache) == len(df), f"cache {len(cache)} != rows {len(df)}"

    with open(cache_path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(cache_path) / 1e6
    print(f"Parse cache written to {cache_path} ({size_mb:.1f} MB)")

if __name__ == '__main__':
    main()
