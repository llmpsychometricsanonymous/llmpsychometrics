# What Does MMLU Actually Measure?

Code and data for the psychometric audit of MMLU described in the accompanying
paper. Item difficulty is calibrated with a 2PL IRT model over 1,000
open-weights models and all 14,042 MMLU test items, then explained by nine
text-extractable complexity indicators, and the resulting mapping is tested for
structural homogeneity across MMLU's own STEM / non-STEM split. A confirmatory
two-dimensional IRT model tests the same question from the response side, and
a transfer test on a second benchmark (HellaSwag) checks whether the
non-invariance finding is specific to MMLU.

Everything here runs on free, open resources. No API keys, no paid services.

## Layout

    code/                  42 numbered pipeline stages plus a driver
    raw_data/               MMLU items, model responses, concreteness norms,
                            HellaSwag responses
    resources/              fonts and the AMR parser checkpoint
    results_precomputed/    every artifact the paper reports, plus figures

`results_precomputed/` is not decoration. The two MMLU parsing stages take
about ten hours between them, so the parses, the IRT estimates and the
extracted indicators are shipped. Every analysis script looks in `results/`
first and falls back to `results_precomputed/`, which means you can reproduce
any number in the paper without re-parsing anything.

## Running it

Python 3.9 or newer.

    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    python code/run_framework.py --list
    python code/run_framework.py --from 8 --to 42

The SpaCy model is a separate download because it is not distributed on PyPI;
stage 5 will fetch it itself if you skip the second line.

Stages 1--7 build the data (alignment, IRT calibration, the five dimension
extractors). Stages 8--15 are the analyses reported in the original submission.
Stages 16--19 are the first round of revision analyses: cross-validation and
baseline models, the robust invariance tests, the Rasch and split-half
replication checks, and the figures. Stages 20--35 are the second round, and
several of them change claims rather than extending them; the table below says
which. Stages 36--42 are the HellaSwag transfer test and the WSCG
convergent-validity check: whether the non-invariance finding generalises
beyond MMLU, and whether the WSCG indicators track an externally annotated
measure of reasoning depth.

Individual stages run standalone:

    python code/31_interaction_and_scale.py

`--only 16 17` runs a subset, `--dry-run` prints the plan without executing it.

Stage 33 reaches the network once, for the MMLU-Redux re-annotation, and caches
it to `results/mmlu_redux_labels.csv`; every later run reads the cache. Stages
36--38 reach the network to build the HellaSwag response matrix (leaderboard
detail parquets and the `Rowan/hellaswag` dataset); the assembled matrix is
shipped in `raw_data/hellaswag_responses.parquet`, so a rerun from the shipped
release does not need them. Stage 42 reaches the network once for GSM8K and
MATH, and caches its features to
`results_precomputed/wscg_validity_features.csv`.

## Where the numbers live

Every stage that produces a written report appends or rewrites its own section
of the single `results_precomputed/results_summary.txt`, in stage order:
IRT calibration validity, path analysis, VIF, rank displacement, per-subject
cross-validation, robust invariance, Rasch and split-half replication, and the
sixteen second-round and extension analyses (measurement robustness through
WSCG convergent validity). The precomputed CSV, `.npy` and `.pkl` artifacts
each stage produces (indicators, IRT parameters, sensitivity grids, parse
caches) sit alongside it in `results_precomputed/` under self-describing
names; `mmlu_wscg_sensitivity.csv` is the 123-scheme weight perturbation grid.

## Stages that changed a claim

Six stages are corrections rather than extensions, and the paper was revised to
match each of them. They are listed here so a reader can go straight to the
evidence.

| Stage | What it changed |
| --- | --- |
| 26 | Items are not independent: clustering by subject and permuting whole subjects leaves the slope contrasts standing but puts the between-domain R-squared gap inside its own null (p = 0.08). The gap is now reported descriptively. |
| 27 | The Top-50 rank displacement of 22% is inside a size-matched random-subset control (p = 0.09). It is now reported as a practical cost rather than as evidence for the construct claim. |
| 29 | A 3PL generative null reproduces the whole of the published full-sample reasoning-sensitivity correlation and most of the trimmed one. Section 5.1 was rewritten around a model-based interaction estimate instead. |
| 30 | The 1,000 leaderboard entries are 869 distinct checkpoints. Seven pairs are the same weights evaluated twice, and their rank disagreement is the paper's empirical evaluation-noise floor. The weighted Kendall comparison is now reported at matched test length. |
| 31 | With 57 clusters the chi-square reference for the joint Wald test is anti-conservative. Under a wild cluster bootstrap the joint test is marginal (p = 0.060) and only the Entity Density contrast survives Holm correction. |
| 32 | Two of the three focal contrasts are monotone in STEM and non-monotone outside it rather than simply differing in slope; the Lexical Concreteness sign reversal is a reversal of a linear summary of a curved relationship and was downgraded accordingly. |

Stages 28, 33 and 34 are checks that the conclusions passed: the symbolic tier
is inert because MMLU carries no integral notation rather than because
preprocessing removed it, and WSCG Depth is not a noun counter (28); the domain
contrast survives dropping every item MMLU-Redux flags as erroneous and appears
in both a pre-2024 and a 2024 model cohort (33); the second latent dimension
improves held-out log-likelihood on cells the model never saw (34).

Stages 36--42 extend the construct rather than correcting it: invariance does
not reject on HellaSwag's designer-fixed partition (36--40), the ending-set
indicators recover the within-source variance the stem-only framework misses
there (41), and the WSCG shows convergent validity against GSM8K's annotated
solution-step counts but not against MATH's holistic difficulty levels (42).
See the HellaSwag and WSCG sections of `results_summary.txt` for the full
numbers.

## Compute notes

Stages 21, 22 and 41 need a sentence encoder
(`sentence-transformers/all-MiniLM-L6-v2`), for option similarity, the
predictability ceiling, and the HellaSwag ending-set indicators. All three
cache their encoded output to `results_precomputed/`, so a rerun from the
shipped release needs neither the model nor a GPU.

Four stages fit variational IRT models over the full 14M-response matrix and are
the exception to the runtime table below: stage 24 (three SVI fits), stage 29
(a 3PL plus an interaction model on the STEM submatrix, and 20 simulated
replications of a 1,000-model logistic sweep), stage 33 (four cohort
calibrations) and stage 34 (two masked fits). Each takes tens of minutes to a
few hours on the JAX CPU backend. Stages 40 and 41 fit the same 2PL model on
the HellaSwag response matrix and are similarly excluded from the table.

## Seeds

The master seed (1729) is set once in `code/config.py`. The bootstrap,
permutation, cross-validation and split-half routines derive separate streams
from it, so rerunning one script cannot shift another script's numbers. Nothing
in the pipeline is stochastic without a seed.

## Runtime

Measured wall-clock for one complete run of stages 1--19 starting from the
shipped caches, on a 6-core / 12-thread Intel Core i5-11260H laptop with 8 GB of
RAM and no GPU. JAX runs on its CPU backend.

The two MMLU parsing stages are not timed here. They were run once, on other
hardware, to produce the caches that ship with this repository, and we have no
measurement of them worth quoting. Stages 20--42 are not timed here either:
several depend on the network or on hardware not held constant across runs (see
Compute notes above).

| Stage | Time |
| --- | --- |
| Alignment and preprocessing | 11 s |
| 2PL IRT calibration | 10 s |
| Dimension 1, reasoning graph | 25 s |
| Dimension 2, syntax | 19 s |
| Dimension 3, knowledge | 8 s |
| Dimension 4, semantics | 19 s |
| Dimension 5, adversarial | 24 s |
| Global path analysis | 4 s |
| Domain invariance test | 2 s |
| Construct analysis | 14 s |
| Robustness checks | 13.3 min |
| Original figures | 8 s |
| Subject effect sizes | 2 s |
| Parse cache check | 16 s |
| WSCG sensitivity grid | 18.6 min |
| Cross-validation and baselines | 13 s |
| Robust invariance | 45 s |
| Rasch refit and split-half | 6 s |
| Figures | 14 s |
| **Total** | **35.9 min** |

## Data

MMLU items come from the public test set. Model responses come from the Hugging
Face Open LLM Leaderboard. Concreteness norms are Brysbaert et al. (2014), Zipf
frequencies come from `wordfreq`. The re-annotation used in stage 33 is
MMLU-Redux 2.0 (`edinburgh-dawg/mmlu-redux-2.0`), fetched once and cached. The
AMR checkpoint under `resources/model_stog/` is excluded from version control by
size; the pipeline falls back to `raw_data/mmlu_amr_precomputed.csv` if it is
absent.

HellaSwag items and their ActivityNet / WikiHow partition come from
`Rowan/hellaswag`; per-item model responses come from the same Open LLM
Leaderboard v1 detail parquets as the MMLU responses, for a stratified subset
of the MMLU model population. GSM8K and MATH, used for the WSCG
convergent-validity check, come from `openai/gsm8k` and the MATH benchmark
mirrors listed in `code/42_wscg_convergent_validity.py`.

## Two notes on reading the outputs

Per-subject regressions fit nine predictors to pools as small as 100 items.
Their in-sample R-squared values are badly optimistic --- cross-validated fit is
negative for 54 of 57 subjects --- and the paper reports them as descriptive
only. If you build on this code, use the partition-level fits. They are the ones
that survive out-of-sample validation.

Item-level p-values in stages 8--19 treat 14,042 items as independent
observations. They are not: MMLU was written in 57 subject blocks, and the
effective sample size for the invariance test is 57, not 14,042. Stages 26 and
31 report the same tests under that reading, and those are the numbers the paper
now quotes.
