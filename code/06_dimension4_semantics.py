import pandas as pd
import numpy as np
import re
import os
import networkx as nx
import amrlib
import penman
import torch
from tqdm import tqdm
import random as python_random
from config import RESULTS_DIR, resolve

os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(1729)
python_random.seed(1729)
np.random.seed(1729)

device = "cuda" if torch.cuda.is_available() else "cpu"

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, "resources", "model_stog")

weights_present = os.path.exists(os.path.join(model_path, "pytorch_model.bin"))
_stog = None

def get_stog():
    global _stog
    if _stog is None:
        if weights_present:
            _stog = amrlib.load_stog_model(model_dir=model_path)
        else:
            _stog = amrlib.load_stog_model()
    return _stog

def amr_dag_depth(text):
    if not isinstance(text, str) or text.strip() == "":
        return 0
    try:
        clean_text = re.sub(r'\s+', ' ', text).strip()
        penman_strings = get_stog().parse_sents([clean_text])
        if not penman_strings:
            return 0

        g_penman = penman.decode(penman_strings[0])
        G = nx.DiGraph()

        for source, relation, target in g_penman.triples:
            if relation.startswith(':'):
                G.add_edge(source, target)

        if G.number_of_edges() > 0:
            if not nx.is_directed_acyclic_graph(G):
                G_dag = nx.DiGraph(G)
                while not nx.is_directed_acyclic_graph(G_dag):
                    cycle = nx.find_cycle(G_dag)
                    G_dag.remove_edge(*cycle[0])
                depth_w = nx.dag_longest_path_length(G_dag) + 1
            else:
                depth_w = nx.dag_longest_path_length(G) + 1
        else:
            depth_w = 1 if G.number_of_nodes() > 0 else 0

        return depth_w
    except Exception as e:
        return 0

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_path = resolve("mmlu_dimension3_knowledge.csv")
    out_path = os.path.join(RESULTS_DIR, "mmlu_dimension4_semantics.csv")
    cached_out = resolve("mmlu_dimension4_semantics.csv")
    brysbaert_path = os.path.join(
        base_dir, "raw_data", "Concreteness_ratings_Brysbaert_et_al_BRM.xlsx")

    local_cache_path = os.path.join(
        base_dir, "raw_data", "mmlu_amr_precomputed.csv")

    if os.path.exists(local_cache_path):
        old_data_path = local_cache_path
        force_recompute = False
    else:
        old_data_path = None
        force_recompute = True

    if not os.path.exists(brysbaert_path):
        return

    df_data = pd.read_csv(in_path)

    df_brys = pd.read_excel(brysbaert_path)
    brys_dict = dict(zip(df_brys['Word'].astype(
        str).str.lower(), df_brys['Conc.M'].astype(float)))

    if force_recompute:
        df_data['Semantic_Concreteness'] = np.nan
        df_data['AMR_Depth'] = np.nan
        extraction_indices = df_data.index
    else:

        if os.path.exists(cached_out):
            df_old = pd.read_csv(cached_out)
        else:
            df_old = pd.read_csv(old_data_path)

        df_old_subset = df_old[['question_id', 'AMR_Depth']].copy()

        df_data = pd.merge(df_data, df_old_subset,
                           on='question_id', how='left')

        missing_mask = df_data['AMR_Depth'].isna()
        extraction_indices = df_data[missing_mask].index

    print("Extracting Dimension 4 Semantic indicators...")

    for idx in tqdm(df_data.index, desc="Extracting Brysbaert Concreteness (Real-time)"):
        row = df_data.loc[idx]
        text = row['clean_text']
        words = [w.lower() for w in re.findall(r'\b\w+\b', str(text))]
        concs = [brys_dict[w] for w in words if w in brys_dict]
        mean_conc = np.mean(concs) if concs else 3.0
        df_data.at[idx, 'Semantic_Concreteness'] = round(mean_conc, 4)

    if len(extraction_indices) > 0:
        for idx in tqdm(
                extraction_indices,
                desc="Extracting Missing AMR Depth"):
            row = df_data.loc[idx]
            text = row['clean_text']
            amr_d = amr_dag_depth(text)
            df_data.at[idx, 'AMR_Depth'] = amr_d

    keep_cols = [
        'question_id', 'item_id', 'subject', 'question_text', 'clean_text',
        'choices', 'ground_truth', 'domain_group', 'difficulty_score',
        'discrimination_score', 'WSCG_Nodes', 'WSCG_Depth',
        'Syntactic_MDD', 'Syntactic_Depth',
        'Knowledge_NER_Density', 'Knowledge_Zipf_Rarity',
        'Semantic_Concreteness', 'AMR_Depth'
    ]
    df_data_stripped = df_data[keep_cols]
    df_data_stripped.to_csv(out_path, index=False)

    print(f"Completed Dimension 4. All associated files saved to {out_path}")

    results_path = os.path.join(RESULTS_DIR, "results_summary.txt")
    with open(results_path, "a", encoding="utf-8") as f:
        f.write("\n\nDIMENSION 4 - SEMANTIC COMPLEXITY\n\n")
        for indicator in ['Semantic_Concreteness', 'AMR_Depth']:
            val_max = df_data_stripped[indicator].max()
            val_min = df_data_stripped[indicator].min()
            val_mean = df_data_stripped[indicator].mean()
            f.write(f" {indicator}\n")
            f.write(f"    Max: {val_max:>6.4f} | Min: {val_min:>6.4f}\n")
            f.write(f"    Mean: {val_mean:>6.4f}\n\n")

if __name__ == '__main__':
    main()
