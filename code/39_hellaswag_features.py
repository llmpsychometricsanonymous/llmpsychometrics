import importlib.util
import os
import pickle
import re
import string
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CODE = HERE
OUT = os.path.join(REPO, "results_precomputed")

sys.path.insert(0, CODE)
sys.path.insert(0, HERE)

INDICATORS_8 = [
    'WSCG_Depth', 'WSCG_Nodes', 'Syntactic_MDD', 'Syntactic_Depth',
    'Knowledge_Zipf_Rarity', 'Knowledge_NER_Density', 'Semantic_Concreteness',
    'Adversarial_Score',
]


def load_published(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def negation_lexicon():
    src = open(os.path.join(CODE, "07_dimension5_adversarial.py"),
               encoding="utf-8").read()
    m = re.search(r"^NEGATION_LEXICON\s*=\s*\{", src, re.M)
    i = src.index("{", m.start())
    depth = 0
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                break
    ns = {}
    exec(src[m.start():j + 1], ns)
    return ns["NEGATION_LEXICON"]


def hellaswag_items():
    from datasets import load_dataset

    from hs_text import queries_from_dataset

    ds = load_dataset("Rowan/hellaswag", split="validation")
    text = queries_from_dataset(ds)
    src = ["wikihow" if "wikihow" in s else
           ("activitynet" if "activitynet" in s else "other")
           for s in ds["source_id"]]
    return pd.DataFrame({"item_id": list(range(len(text))),
                         "clean_text": text,
                         "domain_group": src})


def build_parse_cache(texts, path, batch=64, checkpoint_every=5):
    if os.path.exists(path):
        with open(path, "rb") as f:
            cache = pickle.load(f)
        if len(cache) == len(texts):
            print("parse cache reused")
            return cache
    else:
        cache = []

    part = path + ".part"
    if not cache and os.path.exists(part):
        with open(part, "rb") as f:
            cache = pickle.load(f)
        print(f"resuming parse from {len(cache)} items")

    if len(cache) >= len(texts):
        return cache[:len(texts)]

    import stanza
    from tqdm import tqdm
    nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,depparse",
                          use_gpu=False, seed=1729, download_method=None,
                          verbose=False)

    starts = list(range(len(cache), len(texts), batch))
    for n, s in enumerate(tqdm(starts, desc="parsing"), 1):
        chunk = texts[s:s + batch]
        try:
            docs = nlp.bulk_process(chunk)
        except Exception:
            docs = [nlp(t) for t in chunk]
        for doc in docs:
            cache.append([[(w.text.lower(), (w.lemma or "").lower(), w.upos,
                            w.id, w.head) for w in sent.words]
                          for sent in doc.sentences])
        if n % checkpoint_every == 0:
            with open(part, "wb") as f:
                pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    if os.path.exists(part):
        os.remove(part)
    return cache


def syntactic_from_parse(item):
    mdd_sum = mdd_n = max_depth = 0
    for sent in item:
        nodes = {wid: (wid, head) for (_, _, _, wid, head) in sent}
        for (_, _, _, wid, head) in sent:
            d, cur = 0, head
            while cur != 0 and cur in nodes:
                d += 1
                cur = nodes[cur][1]
            max_depth = max(max_depth, d)
            if head != 0:
                mdd_sum += abs(wid - head)
                mdd_n += 1
    return (round(mdd_sum / mdd_n, 4) if mdd_n else 0.0), max_depth


def negation_from_parse(item, lexicon):
    scope_total = words = 0
    for sent in item:
        words += len(sent)
        ids = {wid for (_, _, _, wid, _) in sent}
        scope = set()
        for (text, lemma, _, wid, head) in sent:
            if text in lexicon or (lemma and lemma in lexicon):
                scope.add(wid)
                if head != 0 and head in ids:
                    scope.add(head)
                    for (_, _, _, w2, h2) in sent:
                        if h2 == head:
                            scope.add(w2)
        scope_total += len(scope)
    return scope_total, words


def main():
    df = hellaswag_items()
    print("HellaSwag validation items:", len(df))
    print(df.domain_group.value_counts().to_string())

    cache = build_parse_cache(df["clean_text"].tolist(),
                              os.path.join(OUT, "hellaswag_parse_cache.pkl"))
    assert len(cache) == len(df)

    sens = load_published("sens", os.path.join(CODE, "15_wscg_sensitivity.py"))
    zero = {t: 0.0 for t in sens.TIERS}

    nodes, depth, mdd, sdep = [], [], [], []
    for item in cache:
        n, d = sens.wscg_from_parse(item, zero)
        nodes.append(n)
        depth.append(d)
        m, h = syntactic_from_parse(item)
        mdd.append(m)
        sdep.append(h)
    df["WSCG_Nodes"] = nodes
    df["WSCG_Depth"] = depth
    df["Syntactic_MDD"] = mdd
    df["Syntactic_Depth"] = sdep
    print("WSCG / syntax done")

    import spacy
    import wordfreq
    spacy.util.fix_random_seed(1729)
    try:
        nlp = spacy.load("en_core_web_sm",
                         disable=["tagger", "parser", "attribute_ruler",
                                  "lemmatizer"])
    except OSError:
        from spacy.cli import download as dl
        dl("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm",
                         disable=["tagger", "parser", "attribute_ruler",
                                  "lemmatizer"])
    stops = nlp.Defaults.stop_words
    translator = str.maketrans('', '', string.punctuation)
    ner_d, zipf_r = [], []
    for doc, raw in zip(nlp.pipe(df["clean_text"].tolist(), batch_size=256),
                        df["clean_text"]):
        total = len(doc)
        ent = sum(len(e) for e in doc.ents)
        ner_d.append(round(ent / total * 100.0, 4) if total else 0.0)
        rar = []
        for w in str(raw).translate(translator).split():
            wc = w.lower().strip()
            if not wc or wc.isdigit() or wc in stops:
                continue
            f = wordfreq.zipf_frequency(wc, 'en')
            if f > 0:
                rar.append(8.0 - f)
        zipf_r.append(round(float(np.mean(rar)), 4) if rar else np.nan)
    df["Knowledge_NER_Density"] = ner_d
    df["Knowledge_Zipf_Rarity"] = zipf_r
    df["Knowledge_Zipf_Rarity"] = df["Knowledge_Zipf_Rarity"].fillna(
        df["Knowledge_Zipf_Rarity"].mean())
    print("entity density / Zipf rarity done")

    brys = pd.read_excel(os.path.join(REPO, "raw_data",
                                      "Concreteness_ratings_Brysbaert_et_al_BRM.xlsx"))
    bd = dict(zip(brys["Word"].astype(str).str.lower(),
                  brys["Conc.M"].astype(float)))
    conc = []
    for t in df["clean_text"]:
        ws = [w.lower() for w in re.findall(r"\b\w+\b", str(t))]
        cs = [bd[w] for w in ws if w in bd]
        conc.append(round(float(np.mean(cs)) if cs else 3.0, 4))
    df["Semantic_Concreteness"] = conc
    print("concreteness done")

    lex = negation_lexicon()
    raw_scope, wc = [], []
    for item in cache:
        s, w = negation_from_parse(item, lex)
        raw_scope.append(s)
        wc.append(w)
    df["adversarial_raw_count"] = raw_scope
    df["word_count"] = wc
    from sklearn.linear_model import LinearRegression
    X = df[["word_count"]].values
    y = df["adversarial_raw_count"].values
    df["Adversarial_Score"] = y - LinearRegression().fit(X, y).predict(X)
    print("adversarial negation done")

    df.to_csv(os.path.join(OUT, "hellaswag_indicators.csv"), index=False)
    print("wrote hellaswag_indicators.csv", df.shape)
    print(df[INDICATORS_8].describe().T[["mean", "min", "max"]].to_string())


if __name__ == "__main__":
    main()
