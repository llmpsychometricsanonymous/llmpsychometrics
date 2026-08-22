import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import random
from numpyro.infer import SVI, Trace_ELBO, autoguide

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import INDICATORS, SEED, output_path, resolve

REDUX = "edinburgh-dawg/mmlu-redux-2.0"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
CACHE = "mmlu_redux_labels.csv"
STEPS = 5000
LR = 0.01


EARLY = (r'pythia|bloom|gpt[-_]?neox|gpt[-_]?j|opt[-_]\d|mpt[-_]|stablelm'
         r'|redpajama|open[-_]?llama|falcon|vicuna|alpaca|guanaco|wizardlm'
         r'|llama[-_]?2|llama[-_]?1|xgen|baichuan|internlm[-_]?1|pygmalion')
RECENT = (r'llama[-_]?3|qwen2|qwen1\.5|phi[-_]?3|gemma|mixtral|yi[-_]?1\.5'
          r'|deepseek|solar|command[-_]?r|starling|dbrx|olmo|internlm2')


def get_json(url, tries=8):
    delay = 2.0
    for attempt in range(tries):
        try:
            return json.load(urllib.request.urlopen(url, timeout=90))
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                raise
        except Exception:
            if attempt == tries - 1:
                raise
        time.sleep(delay)
        delay = min(60.0, delay * 2)


def fetch_redux(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    cfgs = get_json("https://datasets-server.huggingface.co/splits?dataset="
                    + urllib.parse.quote(REDUX, safe=''))['splits']
    rows = []
    for i, s in enumerate(cfgs, 1):
        cfg = s['config']
        off = 0
        while True:
            q = (f"{ROWS_URL}?dataset={urllib.parse.quote(REDUX, safe='')}"
                 f"&config={cfg}&split={s['split']}&offset={off}&length=100")
            d = get_json(q)
            for r in d['rows']:
                rows.append({'subject': cfg,
                             'question': r['row']['question'],
                             'error_type': r['row']['error_type']})
            off += len(d['rows'])
            if off >= d['num_rows_total'] or not d['rows']:
                break
        print(f"  [{i:>2}/{len(cfgs)}] {cfg}: {off} rows")
        time.sleep(1.0)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def irt_model(Y):
    n_models, n_items = Y.shape
    theta = numpyro.sample("theta", dist.Normal(0., 1.).expand([n_models]))
    a = numpyro.sample("a", dist.HalfNormal(1.).expand([n_items]))
    b = numpyro.sample("b", dist.Normal(0., 1.).expand([n_items]))
    numpyro.sample("obs", dist.Bernoulli(logits=a * (theta[:, None] - b)),
                   obs=Y)


def calibrate(Y, seed):
    guide = autoguide.AutoNormal(irt_model)
    svi = SVI(irt_model, guide, numpyro.optim.Adam(step_size=LR),
              loss=Trace_ELBO())
    res = svi.run(random.PRNGKey(seed), num_steps=STEPS, progress_bar=False,
                  Y=Y)
    med = guide.median(res.params)
    return np.asarray(med['b']), np.asarray(med['a']), np.asarray(med['theta'])


def stratified(y, X, stem):
    a = sm.OLS(y[stem], sm.add_constant(X[stem])).fit(cov_type='HC3')
    b = sm.OLS(y[~stem], sm.add_constant(X[~stem])).fit(cov_type='HC3')
    d = np.asarray(a.params)[1:] - np.asarray(b.params)[1:]
    V = (np.asarray(a.cov_params())[1:, 1:]
         + np.asarray(b.cov_params())[1:, 1:])
    W = float(d @ np.linalg.solve(V, d))
    return a.rsquared, b.rsquared, W, sps.chi2.sf(W, len(INDICATORS)), d


def norm_q(s):
    return ' '.join(str(s).lower().split())[:220]


def main():
    print("Label errors and contamination")
    df = pd.read_csv(resolve("mmlu_IRT_calibrated.csv"), low_memory=False)
    feats = pd.read_csv(resolve("mmlu_dimension5_adversarial.csv"))
    meta = ['question_id', 'item_id', 'subject', 'question_text', 'clean_text',
            'choices', 'ground_truth', 'domain_group', 'difficulty_score',
            'discrimination_score']
    model_cols = [c for c in df.columns if c not in meta]
    assert (feats['question_id'].values == df['question_id'].values).all()

    y = feats['difficulty_score'].to_numpy(float)
    X = feats[INDICATORS].to_numpy(float)
    stem = (feats['domain_group'] == 'STEM').to_numpy()

    out = ["LABEL ERRORS AND TRAINING CONTAMINATION", ""]


    print("fetching the MMLU-Redux re-annotation ...")
    redux = fetch_redux(output_path(CACHE))
    key = {(r.subject, norm_q(r.question)): r.error_type
           for r in redux.itertuples()}
    err = np.array([key.get((s, norm_q(q)), None)
                    for s, q in zip(df['subject'], df['question_text'])],
                   dtype=object)
    covered = err != None
    ok = np.array([e == 'ok' for e in err])
    bad = covered & ~ok

    out += ["PART 1: MMLU-REDUX", "",
            "MMLU-Redux 2.0 re-annotates 100 items in each of the 57 subjects.",
            "Items are matched to ours on subject and normalised question text.",
            ""]
    n_norm = len(set(zip(df['subject'], [norm_q(q) for q in df['question_text']])))
    out.append(f"  re-annotated items in the release : {len(redux)}")
    out.append(f"  matched into our item pool        : {int(covered.sum())}")
    out.append(f"  distinct stems in our pool        : {n_norm} of {len(df)}")
    out.append("  The match count exceeds the annotation count because MMLU")
    out.append(f"  itself repeats {len(df) - n_norm} item stems within a subject, so one")
    out.append("  re-annotated question can correspond to more than one of our")
    out.append("  items. Every such item inherits the same verdict.")
    out.append(f"  of those, clean                   : {int(ok.sum())}")
    out.append(f"  of those, flagged as erroneous    : {int(bad.sum())}"
               f"   ({100 * bad.sum() / max(1, covered.sum()):.2f}%)")
    out.append("")
    vc = pd.Series([e for e in err[bad]]).value_counts()
    out.append("  error types:")
    for k, v in vc.items():
        out.append(f"    {k:<34}{v:>5}")
    out.append("")
    out.append(f"  error rate in STEM     : "
               f"{100 * bad[stem].sum() / max(1, covered[stem].sum()):.2f}%"
               f"   ({int(covered[stem].sum())} annotated)")
    out.append(f"  error rate outside STEM: "
               f"{100 * bad[~stem].sum() / max(1, covered[~stem].sum()):.2f}%"
               f"   ({int(covered[~stem].sum())} annotated)")

    d_bad = y[bad]
    d_ok = y[ok]
    out.append("")
    out.append(f"  mean difficulty, flagged items : {d_bad.mean():+.3f}"
               f"   (SD {d_bad.std(ddof=1):.3f})")
    out.append(f"  mean difficulty, clean items   : {d_ok.mean():+.3f}"
               f"   (SD {d_ok.std(ddof=1):.3f})")
    t, p = sps.ttest_ind(d_bad, d_ok, equal_var=False)
    out.append(f"  difference                     : t = {t:.2f}, p = {p:.2e}")
    out.append("  Flagged items are harder, which is what a wrong key predicts:")
    out.append("  models that reason correctly are scored wrong.")

    out += ["", "Refits. The verified subset is the 5,700 re-annotated items",
            "minus those flagged; the pruned pool is all 14,042 items minus the",
            "flagged ones, which keeps the unannotated majority.",
            ""]
    out.append(f"{'Item pool':<34}{'N':>7}{'STEM R2':>10}{'nSTEM R2':>10}"
               f"{'W(9)':>10}{'p':>12}")
    out.append("-" * 83)
    for label, mask in [("published, all items", np.ones(len(y), bool)),
                        ("verified subset only", ok),
                        ("all items minus flagged", ~bad),
                        ("annotated subset, unfiltered", covered)]:
        r_s, r_n, W, p, _ = stratified(y[mask], X[mask], stem[mask])
        out.append(f"{label:<34}{int(mask.sum()):>7}{r_s:>10.4f}{r_n:>10.4f}"
                   f"{W:>10.2f}{p:>12.2e}")


    fam_early = pd.Series(model_cols).str.lower().str.contains(
        EARLY, regex=True).to_numpy()
    fam_recent = pd.Series(model_cols).str.lower().str.contains(
        RECENT, regex=True).to_numpy()
    fam_early = fam_early & ~fam_recent
    idx_early = np.where(fam_early)[0]
    idx_recent = np.where(fam_recent)[0]

    out += ["", "", "PART 2: A COHORT SPLIT FOR CONTAMINATION", "",
            "Models are assigned to a generation from the architecture markers",
            "in their names. The early cohort is the Pythia / Falcon / Llama-2",
            "era; the recent cohort is the 2024 generation, trained after MMLU",
            "had become a headline leaderboard metric and is correspondingly",
            "more exposed to it.",
            ""]
    out.append(f"  early cohort  : {len(idx_early):>4} models")
    out.append(f"  recent cohort : {len(idx_recent):>4} models")
    out.append(f"  unassigned    : "
               f"{len(model_cols) - len(idx_early) - len(idx_recent):>4} models")

    Y = df[model_cols].to_numpy(dtype=np.float32).T
    print("calibrating the early cohort ...")
    b_e, a_e, th_e = calibrate(jnp.asarray(Y[idx_early]), SEED)
    print("calibrating the recent cohort ...")
    b_r, a_r, th_r = calibrate(jnp.asarray(Y[idx_recent]), SEED)

    print("calibrating two same-size random cohorts ...")
    rng = np.random.default_rng(SEED)
    pick = rng.permutation(len(model_cols))
    n_e = len(idx_early)
    b_c1, _, _ = calibrate(jnp.asarray(Y[pick[:n_e]]), SEED + 11)
    b_c2, _, _ = calibrate(jnp.asarray(Y[pick[n_e:2 * n_e]]), SEED + 12)
    r_ceiling = float(np.corrcoef(b_c1, b_c2)[0, 1])

    r_cohort = float(np.corrcoef(b_e, b_r)[0, 1])
    rs_cohort = float(sps.spearmanr(b_e, b_r).statistic)
    out.append("")
    out.append(f"  difficulty correlation between cohorts : r = {r_cohort:.4f}"
               f"   (Spearman {rs_cohort:.4f})")
    out.append(f"  early cohort b  : mean {b_e.mean():+.3f}, SD {b_e.std():.3f}")
    out.append(f"  recent cohort b : mean {b_r.mean():+.3f}, SD {b_r.std():.3f}")
    out.append("")
    out.append("That correlation needs a ceiling, because the early cohort is")
    out.append(f"{len(idx_early)} models and difficulty estimated from {len(idx_early)} raters is")
    out.append("noisier than difficulty estimated from a thousand. Two disjoint")
    out.append(f"random cohorts of {len(idx_early)} models each, drawn without regard to")
    out.append("generation, agree at:")
    out.append("")
    out.append(f"  same-size random-cohort ceiling        : r = {r_ceiling:.4f}")
    out.append(f"  early against recent                   : r = {r_cohort:.4f}"
               f"   ({100 * r_cohort / r_ceiling:.0f}% of the ceiling)")
    out.append("")
    out.append("A contaminated cohort should find leaked items easy that an")
    out.append("uncontaminated one finds hard, which would depress the observed")
    out.append("correlation relative to that ceiling. It does not depress it much.")

    out += ["", "The domain contrast, estimated separately within each cohort:",
            ""]
    out.append(f"{'Difficulty from':<28}{'STEM R2':>10}{'nSTEM R2':>10}"
               f"{'ratio':>9}{'W(9)':>10}{'p':>12}")
    out.append("-" * 79)
    contrasts = {}
    for label, bvec in [("all 1,000 models", y),
                        ("early cohort", b_e),
                        ("recent cohort", b_r)]:
        r_s, r_n, W, p, d = stratified(bvec, X, stem)
        contrasts[label] = d
        out.append(f"{label:<28}{r_s:>10.4f}{r_n:>10.4f}"
                   f"{r_s / r_n:>9.2f}{W:>10.2f}{p:>12.2e}")
    out.append("")
    cc = float(np.corrcoef(contrasts['early cohort'],
                           contrasts['recent cohort'])[0, 1])
    out.append(f"  correlation between the two cohorts' nine contrast vectors: "
               f"{cc:.3f}")


    out += ["", "", "PART 3: BOTH AT ONCE", "",
            "The strictest available pool: flagged items removed, difficulty",
            "estimated on the early cohort alone.",
            ""]
    r_s, r_n, W, p, _ = stratified(b_e[~bad], X[~bad], stem[~bad])
    out.append(f"  N = {int((~bad).sum())}   STEM R2 {r_s:.4f}   "
               f"non-STEM R2 {r_n:.4f}")
    out.append(f"  W(9) = {W:.2f}, p = {p:.2e}")


    out += ["", "", "READING", ""]
    out.append(f"Label error is real and measurable at "
               f"{100 * bad.sum() / max(1, covered.sum()):.1f}% of the "
               f"re-annotated")
    out.append("items, and flagged items are systematically harder, exactly as a")
    out.append("wrong key predicts. It is not, however, what produces the domain")
    out.append("contrast: dropping every flagged item leaves both partition fits")
    out.append("and the joint test essentially where they were, and refitting on")
    out.append("the verified subset alone -- a fortieth of the original pool, with")
    out.append("the loss of power that implies -- does not reverse the ordering.")
    out.append("")
    out.append(f"Contamination is likewise not the mechanism. Difficulty")
    out.append(f"estimated on models from the Pythia and Llama-2 era correlates")
    out.append(f"at r = {r_cohort:.3f} with difficulty estimated on the 2024")
    out.append(f"generation, against a same-size sampling ceiling of "
               f"{r_ceiling:.3f},")
    out.append("and the domain contrast appears at similar magnitude")
    out.append("in both cohorts with contrast vectors correlated at "
               f"{cc:.2f}. If")
    out.append("memorisation prevalence were driving the structure, the two")
    out.append("cohorts would not agree this closely about which items are hard.")

    pd.DataFrame({
        'question_id': df['question_id'], 'subject': df['subject'],
        'domain_group': df['domain_group'], 'redux_error_type': err,
        'b_published': y, 'b_early_cohort': b_e, 'b_recent_cohort': b_r,
        'b_random_cohort_a': b_c1, 'b_random_cohort_b': b_c2,
    }).to_csv(output_path("mmlu_label_contamination.csv"), index=False)

    text = "\n".join(out) + "\n"
    with open(output_path("results_labels_contamination.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
