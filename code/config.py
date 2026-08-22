import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(BASE_DIR, "results")
PRECOMPUTED_DIR = os.path.join(BASE_DIR, "results_precomputed")

def resolve(filename):
    local = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(local):
        return local
    return os.path.join(PRECOMPUTED_DIR, filename)

def output_path(filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, filename)

def results_summary_path():
    return output_path("results_summary.txt")

def revision_report(filename, lines):
    path = output_path(filename)
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write("\n".join(lines).rstrip() + "\n")
    return path

INDICATORS = [
    'WSCG_Depth',
    'WSCG_Nodes',
    'Syntactic_MDD',
    'Syntactic_Depth',
    'Knowledge_Zipf_Rarity',
    'Knowledge_NER_Density',
    'Semantic_Concreteness',
    'AMR_Depth',
    'Adversarial_Score',
]

INDICATOR_LABELS = {
    'WSCG_Depth': 'WSCG Depth',
    'WSCG_Nodes': 'WSCG Nodes',
    'Syntactic_MDD': 'Syntactic MDD',
    'Syntactic_Depth': 'Syntactic Depth',
    'Knowledge_Zipf_Rarity': 'Zipf Rarity',
    'Knowledge_NER_Density': 'Entity Density',
    'Semantic_Concreteness': 'Lexical Concreteness',
    'AMR_Depth': 'AMR Depth',
    'Adversarial_Score': 'Adversarial Negation',
}

SEED = 1729
BOOTSTRAP_SEED = SEED + 1
PERMUTATION_SEED = SEED + 2
CV_SEED = SEED + 3
SPLIT_HALF_SEED = SEED + 4

N_BOOTSTRAP = 1000
N_PERMUTATION = 10000
N_FOLDS = 5

MIN_ITEMS_REPORTED = 50
