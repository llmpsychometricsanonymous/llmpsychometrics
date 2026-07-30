import pandas as pd
import numpy as np
import os
import re
import pickle
import networkx as nx
import statsmodels.api as sm
from scipy.stats import chi2, norm
from tqdm import tqdm
from config import RESULTS_DIR, resolve

os.makedirs(RESULTS_DIR, exist_ok=True)

INDICATORS = [
    'WSCG_Depth', 'WSCG_Nodes', 'Syntactic_MDD', 'Syntactic_Depth',
    'Knowledge_Zipf_Rarity', 'Knowledge_NER_Density', 'Semantic_Concreteness',
    'AMR_Depth', 'Adversarial_Score'
]

TIERS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
N_MONTE_CARLO = 100
SEED = 1729

def load_weight_tables():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "03_dimension1_reasoning.py")
    src = open(path, encoding="utf-8").read()
    ns = {}

    for name in ["COT_CATEGORIES", "LATEX_OPERATORS", "MATH_OPERATORS"]:
        m = re.search(rf"^{name}\s*=\s*\{{", src, re.M)
        if m is None:
            raise RuntimeError(f"could not locate {name}")
        i = src.index("{", m.start())
        depth = 0
        for j in range(i, len(src)):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    break
        else:
            raise RuntimeError(f"unbalanced braces in {name}")
        exec(src[m.start():j + 1], ns)
    lexicon = {}
    for cat, data in ns["COT_CATEGORIES"].items():
        for term in data['terms']:
            lexicon[term] = data['weight']
    return lexicon, ns["LATEX_OPERATORS"], ns["MATH_OPERATORS"]

WEIGHTED_LEXICON, LATEX_OPERATORS, MATH_OPERATORS = load_weight_tables()
_LATEX_C = [(re.compile(p), w) for p, w in LATEX_OPERATORS.items()]
_MATH_C = [(re.compile(p), w) for p, w in MATH_OPERATORS.items()]

def base_weight(text, lemma, pos):

    if lemma in WEIGHTED_LEXICON:
        return WEIGHTED_LEXICON[lemma]
    for rx, w in _LATEX_C:
        if rx.search(text):
            return w
    for rx, w in _MATH_C:
        if rx.search(text):
            return w
    if pos == 'NUM':
        return 1.0
    if pos in ('NOUN', 'PROPN'):
        return 0.5
    return 0.0

def wscg_from_parse(item, deltas):
    if item is None:
        return 0, 0.0

    G = nx.DiGraph()
    memory = {}
    node_weights = {}
    node_id = 0

    for sent in item:
        sent_nodes = {}
        for (text, lemma, upos, wid, head) in sent:
            node_id += 1
            sent_nodes[wid] = node_id

            base = base_weight(text, lemma, upos)
            if base == 0.0:
                weight = 0.0
            else:
                weight = min(5.0, max(0.0, base + deltas.get(base, 0.0)))

            if weight > 0:
                G.add_node(node_id, lemma=lemma, weight=weight)
                node_weights[node_id] = weight
                if upos in ('NOUN', 'PROPN', 'NUM') or weight >= 2.0:
                    if lemma in memory:
                        G.add_edge(memory[lemma], node_id, weight=weight)
                    memory[lemma] = node_id

        for (text, lemma, upos, wid, head) in sent:
            if head != 0:
                head_id = sent_nodes.get(head)
                dep_id = sent_nodes.get(wid)
                if head_id in G.nodes and dep_id in G.nodes:
                    G.add_edge(head_id, dep_id, weight=node_weights[dep_id])

    nodes_count = G.number_of_nodes()

    G_dag = G
    if G_dag.number_of_edges() > 0 and not nx.is_directed_acyclic_graph(G_dag):
        G_dag = nx.DiGraph(G)
        while not nx.is_directed_acyclic_graph(G_dag):
            cycle = nx.find_cycle(G_dag)
            min_edge = min(cycle,
                           key=lambda e: G_dag[e[0]][e[1]].get('weight', 0.0))
            G_dag.remove_edge(*min_edge)

    if G_dag.number_of_edges() > 0:
        depth_w = nx.dag_longest_path_length(G_dag, weight='weight')
    elif node_weights:
        depth_w = max(node_weights.values())
    else:
        return 0, 0.0

    return nodes_count, round(depth_w, 4)

def fit_ols(X, y):
    model = sm.OLS(y, sm.add_constant(X)).fit(cov_type='HC3')
    return {'beta': model.params, 'se': model.bse, 'r2': model.rsquared,
            'p': sm.add_constant(X).shape[1], 'log_lik': model.llf}

def evaluate(df):
    d = df.dropna(subset=INDICATORS + ['difficulty_score', 'domain_group'])
    stem = d[d['domain_group'] == 'STEM']
    nons = d[d['domain_group'] != 'STEM']

    m_pool = fit_ols(d[INDICATORS].values, d['difficulty_score'].values)
    m_stem = fit_ols(stem[INDICATORS].values, stem['difficulty_score'].values)
    m_nons = fit_ols(nons[INDICATORS].values, nons['difficulty_score'].values)

    lam = 2.0 * ((m_stem['log_lik'] + m_nons['log_lik']) - m_pool['log_lik'])
    df_lrt = m_stem['p'] + m_nons['p'] - m_pool['p'] + 1
    p_lrt = chi2.sf(lam, df=df_lrt)

    j = INDICATORS.index('Syntactic_MDD') + 1
    z_mdd = ((m_stem['beta'][j] - m_nons['beta'][j]) /
             np.sqrt(m_stem['se'][j]**2 + m_nons['se'][j]**2))

    return {'r2_stem': m_stem['r2'], 'r2_nonstem': m_nons['r2'],
            'lrt': lam, 'lrt_df': df_lrt, 'p_lrt': p_lrt,
            'z_mdd': z_mdd, 'p_mdd': norm.sf(abs(z_mdd)) * 2.0}

def main():
    print("Initiating WSCG Cognitive Weight Sensitivity Analysis")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    res_dir = RESULTS_DIR
    cache_path = resolve("mmlu_parse_cache.pkl")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    df = pd.read_csv(resolve("mmlu_dimension5_adversarial.csv"))
    assert len(cache) == len(df), f"cache {len(cache)} != data {len(df)}"
    print(f"Loaded {len(df)} items and matching parse cache")

    def recompute(deltas, desc):
        nodes = np.empty(len(cache))
        depth = np.empty(len(cache))
        for i, item in enumerate(tqdm(cache, desc=desc, leave=False)):
            n, d = wscg_from_parse(item, deltas)
            nodes[i] = n
            depth[i] = d
        out = df.copy()
        out['WSCG_Nodes'] = nodes
        out['WSCG_Depth'] = depth
        return out

    zero = {t: 0.0 for t in TIERS}
    base_df = recompute(zero, "baseline")
    r_depth = np.corrcoef(base_df['WSCG_Depth'], df['WSCG_Depth'])[0, 1]
    r_nodes = np.corrcoef(base_df['WSCG_Nodes'], df['WSCG_Nodes'])[0, 1]
    max_abs = np.max(np.abs(base_df['WSCG_Depth'] - df['WSCG_Depth']))
    print(f"\nFidelity vs shipped WSCG: r(depth)={r_depth:.6f} "
          f"r(nodes)={r_nodes:.6f} max abs delta depth={max_abs:.4f}")

    baseline = evaluate(base_df)
    rows = [dict(scheme="Baseline (published weights)", **baseline)]

    for d in (+1.0, -1.0):
        rows.append(dict(scheme=f"Uniform {d:+.1f} (all tiers)",
                         **evaluate(recompute({t: d for t in TIERS},
                                              f"uniform {d:+.1f}"))))

    for t in TIERS:
        for d in (+1.0, -1.0):
            deltas = dict(zero)
            deltas[t] = d
            rows.append(dict(scheme=f"Tier {t:.1f} {d:+.1f}",
                             **evaluate(recompute(deltas, f"tier {t}{d:+.1f}"))))

    rng = np.random.default_rng(SEED)
    mc = []
    for k in range(N_MONTE_CARLO):
        deltas = {t: float(rng.uniform(-1.0, 1.0)) for t in TIERS}
        mc.append(evaluate(recompute(deltas, f"MC {k+1}/{N_MONTE_CARLO}")))
    mc_df = pd.DataFrame(mc)

    tbl = pd.DataFrame(rows)
    out_csv = os.path.join(res_dir, "mmlu_wscg_sensitivity.csv")
    pd.concat([tbl, mc_df.assign(scheme=[f"MC draw {i+1}" for i in range(len(mc_df))])],
              ignore_index=True).to_csv(out_csv, index=False)

    L = []
    L.append("")
    L.append(f"Parse cache: {len(cache)} items | tiers perturbed: {TIERS}")
    L.append(f"Fidelity of re-derived baseline vs shipped WSCG: "
             f"r(depth) = {r_depth:.6f}, r(nodes) = {r_nodes:.6f}, "
             f"max|delta depth| = {max_abs:.4f}")
    L.append("")
    L.append(f"{'Scheme':<30} | {'STEM R2':>8} | {'nSTEM R2':>9} | "
             f"{'LRT chi2':>10} | {'LRT p':>11} | {'Z(MDD)':>7} |")
    L.append("-" * 90)
    for r in rows:
        L.append(f"{r['scheme']:<30} | {r['r2_stem']:>8.4f} | {r['r2_nonstem']:>9.4f} | "
                 f"{r['lrt']:>10.2f} | {r['p_lrt']:>11.2e} | {r['z_mdd']:>7.2f} |")

    L.append("")
    L.append(f"MONTE CARLO ({N_MONTE_CARLO} draws, each tier ~ U(-1, +1))")
    for col, lab in [('r2_stem', 'STEM R2'), ('r2_nonstem', 'non-STEM R2'),
                     ('lrt', 'LRT chi2'), ('z_mdd', 'Z(MDD)')]:
        L.append(f"  {lab:<12} min={mc_df[col].min():>10.4f}  "
                 f"median={mc_df[col].median():>10.4f}  max={mc_df[col].max():>10.4f}")
    L.append(f"  LRT p           max={mc_df['p_lrt'].max():.3e}")

    allr = pd.concat([tbl, mc_df], ignore_index=True)
    L.append("")
    L.append("PRIMARY CONCLUSIONS ACROSS ALL "
             f"{len(allr)} SCHEMES")
    L.append(f"  LRT rejects homogeneity at p < 0.001:      "
             f"{int((allr['p_lrt'] < 0.001).sum())}/{len(allr)}")
    L.append(f"  STEM R2 exceeds non-STEM R2:               "
             f"{int((allr['r2_stem'] > allr['r2_nonstem']).sum())}/{len(allr)}")
    L.append(f"  Syntactic MDD Z positive and p < 0.01:     "
             f"{int(((allr['z_mdd'] > 0) & (allr['p_mdd'] < 0.01)).sum())}/{len(allr)}")
    L.append("")

    text = "\n".join(L)
    print(text)

    with open(os.path.join(res_dir, "results_summary.txt"), "a", encoding="utf-8") as f:
        f.write("\n\nWSCG COGNITIVE WEIGHT SENSITIVITY ANALYSIS\n\n")
        f.write(text + "\n")

    print(f"Sensitivity analysis completed. Per-scheme results saved to {out_csv}")

if __name__ == '__main__':
    main()
