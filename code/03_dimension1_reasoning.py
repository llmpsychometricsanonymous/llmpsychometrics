import pandas as pd
import numpy as np
import re
import os
import stanza
import networkx as nx
from tqdm import tqdm
import random as random
from config import RESULTS_DIR, resolve

os.makedirs(RESULTS_DIR, exist_ok=True)

try:
    nlp = stanza.Pipeline(
        'en', processors='tokenize,pos,lemma,depparse', use_gpu=True, seed=1729, download_method=None)
except Exception:
    stanza.download('en', processors='tokenize,pos,lemma,depparse')
    nlp = stanza.Pipeline(
        'en', processors='tokenize,pos,lemma,depparse', use_gpu=True, seed=1729, download_method=None)
random.seed(1729)
np.random.seed(1729)

COT_CATEGORIES = {
    'exception': {'weight': 4.5, 'terms': {
        'except', 'exception', 'unless', 'but', 'however', 'although', 'though',
        'despite', 'whereas', 'while', 'alternatively', 'instead', 'excluding',
        'without', 'notwithstanding', 'regardless', 'absent', 'omitting',
        'nevertheless', 'nonetheless', 'albeit', 'conversely', 'contrary',
        'lest', 'barring', 'apart', 'aside', 'yet'
    }},
    'conditional': {'weight': 4.0, 'terms': {
        'if', 'then', 'else', 'otherwise', 'provided', 'assuming', 'assume',
        'suppose', 'supposing', 'given', 'whenever', 'case', 'condition',
        'scenario', 'whether', 'hypothesis', 'postulate', 'premise', 'axiom',
        'contingent', 'must', 'require', 'requires', 'required'
    }},
    'heavy_task': {'weight': 4.0, 'terms': {
        'prove', 'show', 'demonstrate', 'verify', 'validate', 'justify', 'explain',
        'conclude', 'infer', 'deduce', 'establish', 'confirm', 'refute', 'disprove',
        'derive', 'integrate', 'differentiate', 'factor', 'maximize', 'minimize',
        'optimize', 'synthesize', 'extrapolate', 'interpolate', 'transform',
        'invert', 'model', 'formulate', 'theorize', 'forecast',
        'analyze', 'analyse', 'decompose', 'reconstruct', 'quantify',
        'simulate', 'critique', 'assess', 'classify', 'categorize',
        'parameterize', 'normalize', 'linearize'
    }},
    'probability': {'weight': 3.5, 'terms': {
        'probability', 'expected', 'variance', 'likely', 'chance', 'distribution',
        'random', 'odds', 'deviation', 'guess', 'predict', 'anticipate',
        'possibly', 'probably', 'stochastic', 'likelihood', 'prior', 'posterior',
        'marginal', 'joint', 'independence', 'independent', 'significance', 'confidence',
        'interval', 'null', 'uncertainty', 'risk'
    }},
    'causal': {'weight': 3.0, 'terms': {
        'because', 'since', 'therefore', 'thus', 'hence', 'consequently',
        'accordingly', 'resulting', 'cause', 'caused', 'due',
        'effect', 'leads', 'yields', 'produces', 'generates',
        'implies', 'entails', 'trigger', 'inhibit', 'prevent', 'enable', 'mediate',
        'suppress', 'promote', 'induce', 'regulate', 'modulate'
    }},
    'quantifier': {'weight': 3.0, 'terms': {
        'all', 'every', 'each', 'any', 'some', 'none', 'no', 'exists', 'exist',
        'exactly', 'least', 'most', 'maximum', 'minimum', 'majority', 'minority',
        'multiple', 'several', 'few', 'only', 'solely', 'exclusively',
        'finite', 'infinite', 'unique', 'arbitrary', 'always', 'never', 'precisely'
    }},
    'physics_system': {'weight': 3.0, 'terms': {
        'accelerate', 'decelerate', 'heat', 'cool', 'pressurize', 'expand',
        'contract', 'ignite', 'burn', 'melt', 'freeze', 'boil', 'condense',
        'evaporate', 'decay', 'equate', 'balance', 'oscillate', 'emit', 'absorb',
        'react', 'dissolve', 'diffuse', 'refract', 'ionize', 'oxidize', 'collide',
        'radiate', 'scatter', 'catalyze', 'polymerize', 'hydrolyze', 'denature'
    }},
    'spatial': {'weight': 2.5, 'terms': {
        'parallel', 'orthogonal', 'perpendicular', 'adjacent', 'tangent',
        'intersect', 'bisect', 'rotate', 'translate', 'plot', 'graph', 'sketch',
        'draw', 'trace', 'project', 'reflect', 'measure', 'scale', 'align',
        'symmetric', 'diagonal', 'angle', 'vector', 'diverge', 'converge',
        'convex', 'concave', 'horizontal', 'vertical', 'inscribe', 'circumscribe'
    }},
    'comparative': {'weight': 2.0, 'terms': {
        'equal', 'equals', 'equivalent', 'identical', 'same', 'similar',
        'different', 'greater', 'less', 'larger', 'smaller', 'higher', 'lower',
        'exceeds', 'surpasses', 'matches', 'corresponds', 'proportional',
        'correlated', 'relative', 'compare', 'contrast', 'monotonic', 'versus',
        'dominant', 'negligible', 'asymptotic', 'unlike'
    }},
    'sequential': {'weight': 2.0, 'terms': {
        'before', 'after', 'during', 'while', 'until', 'subsequently', 'previously',
        'initially', 'finally', 'ultimately', 'first', 'second', 'third', 'last',
        'next', 'following', 'preceding', 'simultaneously', 'concurrently',
        'order', 'rank', 'sort', 'step', 'phase', 'cycle', 'iterate', 'recursive',
        'successive', 'consecutive'
    }},
    'light_task': {'weight': 1.5, 'terms': {
        'calculate', 'compute', 'solve', 'evaluate', 'add', 'subtract', 'multiply',
        'divide', 'sum', 'average', 'estimate', 'approximate', 'round', 'simplify',
        'reduce', 'isolate', 'combine', 'substitute', 'convert', 'find', 'determine',
        'identify', 'state', 'name', 'list', 'specify', 'select', 'choose', 'pick',
        'extract', 'locate', 'describe', 'define', 'label', 'recall', 'denote', 'what',
        'which', 'who', 'where', 'when', 'how'
    }},
    'boolean': {'weight': 1.0, 'terms': {
        'and', 'or', 'xor', 'not', 'both', 'neither', 'nor', 'iff', 'true', 'false',
        'correct', 'incorrect', 'valid', 'invalid'
    }},
    'coreference': {'weight': 1.0, 'terms': {
        'it', 'they', 'former', 'latter', 'respectively', 'itself', 'themselves',
        'their', 'them'
    }}
}

LATEX_OPERATORS = {
    r'\\int': 5.0,
    r'\\partial': 5.0,
    r'\\nabla': 5.0,
    r'\\oint': 5.0,
    r'\\iint': 5.0,
    r'\\iiint': 5.0,
    r'\\sum': 4.0,
    r'\\prod': 4.0,
    r'\\lim': 4.0,
    r'\\matrix': 4.0,
    r'\\forall': 4.0,
    r'\\exists': 4.0,
    r'\\nexists': 4.0,
    r'\\Leftrightarrow': 4.0,
    r'\\Rightarrow': 3.5,
    r'\\binom': 3.5,
    r'\\det': 3.5,
    r'\\log': 3.0,
    r'\\ln': 3.0,
    r'\\exp': 3.0,
    r'\\arcsin': 3.0,
    r'\\arccos': 3.0,
    r'\\arctan': 3.0,
    r'\\wedge': 3.0,
    r'\\vee': 3.0,
    r'\\neg': 3.0,
    r'\\in': 3.0,
    r'\\notin': 3.0,
    r'\\subset': 3.0,
    r'\\cup': 3.0,
    r'\\cap': 3.0,
    r'\\infty': 3.0,
    r'\\sqrt': 3.0,
    r'\\sin': 2.5,
    r'\\cos': 2.5,
    r'\\tan': 2.5,
    r'\\propto': 2.5,
    r'\\times': 2.5,
    r'\\rightarrow': 2.5,
    r'\\frac': 2.0,
    r'\\theta': 2.0,
    r'\\pi': 2.0,
    r'\\alpha': 2.0,
    r'\\beta': 2.0,
    r'\\gamma': 2.0,
    r'\\lambda': 2.0,
    r'\\sigma': 2.0,
    r'\\Delta': 2.0,
    r'\\Omega': 2.0,
    r'\\approx': 2.0,
    r'\\vec': 2.0,
    r'\\cdot': 2.0,
    r'\\pm': 2.0,
    r'\\phi': 2.0,
    r'\\psi': 2.0,
    r'\\mu': 2.0,
    r'\\epsilon': 2.0,
    r'\\tau': 2.0,
    r'\\kappa': 2.0,
    r'\\rho': 2.0,
    r'\\chi': 2.0,
    r'\\zeta': 2.0,
    r'\\nu': 2.0,
    r'\\xi': 2.0,
    r'\\eta': 2.0}

MATH_OPERATORS = {
    r'\+': 2.0,
    r'-': 2.0,
    r'\*': 2.0,
    r'/': 2.0,
    r'=': 2.0,
    r'<': 2.0,
    r'>': 2.0,
    r'\^': 3.0,
    r'\%': 2.0,
    r'!': 3.0,
    r'\\le': 2.0,
    r'\\ge': 2.0,
    r'\\neq': 2.5,
    r'\|': 2.0,
    r':': 1.5}

WEIGHTED_LEXICON = {}
for cat, data in COT_CATEGORIES.items():
    for term in data['terms']:
        WEIGHTED_LEXICON[term] = data['weight']

def node_weight(text, lemma, pos):
    if lemma in WEIGHTED_LEXICON:
        return WEIGHTED_LEXICON[lemma]
    for latex, w in LATEX_OPERATORS.items():
        if re.search(latex, text):
            return w
    for m_op, w in MATH_OPERATORS.items():
        if re.search(m_op, text):
            return w
    if pos == 'NUM':
        return 1.0
    if pos in ['NOUN', 'PROPN']:
        return 0.5
    return 0.0

def extract_cot_graph(text):
    try:
        doc = nlp(text)
        G = nx.DiGraph()
        memory = {}
        node_weights = {}
        node_id = 0

        for sentence in doc.sentences:
            sent_nodes = {}
            for word in sentence.words:
                node_id += 1
                sent_nodes[word.id] = node_id

                weight = node_weight(
                    word.text.lower(), word.lemma.lower(), word.upos)
                if weight > 0:
                    G.add_node(node_id, lemma=word.lemma.lower(),
                               weight=weight)
                    node_weights[node_id] = weight

                    if word.upos in ['NOUN', 'PROPN', 'NUM'] or weight >= 2.0:
                        lemma = word.lemma.lower()
                        if lemma in memory:
                            past_id = memory[lemma]
                            G.add_edge(past_id, node_id, weight=weight)
                        memory[lemma] = node_id

            for word in sentence.words:
                if word.head != 0:
                    head_id = sent_nodes[word.head]
                    dep_id = sent_nodes[word.id]
                    if head_id in G.nodes and dep_id in G.nodes:
                        G.add_edge(head_id, dep_id,
                                   weight=node_weights[dep_id])

        nodes_count = G.number_of_nodes()

        G_dag = G
        if G_dag.number_of_edges() > 0 and not nx.is_directed_acyclic_graph(G_dag):
            G_dag = nx.DiGraph(G)
            while not nx.is_directed_acyclic_graph(G_dag):
                cycle = nx.find_cycle(G_dag)
                min_edge = min(
                    cycle, key=lambda e: G_dag[e[0]][e[1]].get('weight', 0.0))
                G_dag.remove_edge(*min_edge)

        if G_dag.number_of_edges() > 0:
            depth_w = nx.dag_longest_path_length(G_dag, weight='weight')
        elif node_weights:
            depth_w = max(node_weights.values())
        else:
            return 0, 0

        depth_w = round(depth_w, 4)

        return nodes_count, depth_w

    except Exception:
        return 0, 0

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_path = resolve("mmlu_IRT_calibrated.csv")
    out_path = os.path.join(RESULTS_DIR, "mmlu_dimension1_CoT_reasoning.csv")
    cached_out = resolve("mmlu_dimension1_CoT_reasoning.csv")

    print("Extracting Dimension 1 Reasoning indicators...")

    df_data = pd.read_csv(in_path)

    cache_loaded = False
    if os.path.exists(cached_out):
        try:
            df_out = pd.read_csv(cached_out)
            if len(df_out) == len(df_data) and 'WSCG_Nodes' in df_out.columns and 'WSCG_Depth' in df_out.columns:
                df_data['WSCG_Nodes'] = df_out['WSCG_Nodes']
                df_data['WSCG_Depth'] = df_out['WSCG_Depth']
                if not df_data['WSCG_Nodes'].isna().any() and not df_data['WSCG_Depth'].isna().any():
                    print("Found fully computed Dimension 1 indicators in output cache. Skipping extraction.")
                    cache_loaded = True
        except Exception:
            pass

    if not cache_loaded:
        df_data['WSCG_Nodes'] = np.nan
        df_data['WSCG_Depth'] = np.nan
        extraction_indices = df_data.index

        if len(extraction_indices) > 0:
            for idx in tqdm(extraction_indices, desc="Extracting Dimension 1 Indicators"):
                row = df_data.loc[idx]
                text = row['clean_text']

                nodes_count, depth_w = extract_cot_graph(text)

                df_data.at[idx, 'WSCG_Nodes'] = nodes_count
                df_data.at[idx, 'WSCG_Depth'] = depth_w

    keep_cols = [
        'question_id', 'item_id', 'subject', 'question_text', 'clean_text',
        'choices', 'ground_truth', 'domain_group', 'difficulty_score',
        'discrimination_score', 'WSCG_Nodes', 'WSCG_Depth'
    ]
    df_data_stripped = df_data[keep_cols]

    df_data_stripped.to_csv(out_path, index=False)

    print(f"Completed Dimension 1. All associated files saved to {out_path}")

    results_path = os.path.join(RESULTS_DIR, "results_summary.txt")
    with open(results_path, "a", encoding="utf-8") as f:
        f.write("\n\nDIMENSION 1 - REASONING COMPLEXITY\n\n")
        for indicator in ['WSCG_Nodes', 'WSCG_Depth']:
            val_max = df_data_stripped[indicator].max()
            val_min = df_data_stripped[indicator].min()
            val_mean = df_data_stripped[indicator].mean()
            f.write(f" {indicator}\n")
            f.write(f"    Max: {val_max:>6.4f} | Min: {val_min:>6.4f}\n")
            f.write(f"    Mean: {val_mean:>6.4f}\n\n")

if __name__ == '__main__':
    main()
