# What Does MMLU Actually Measure?

Code and data for the psychometric audit of MMLU described in the accompanying
paper. Item difficulty is calibrated with a 2PL IRT model over 1,000
open-weights models and all 14,042 MMLU test items, then regressed on nine
text-extractable complexity indicators, and the resulting mapping is tested for
invariance across MMLU's own STEM / non-STEM split.

Everything here runs on free, open resources. No API keys, no paid services.

## Layout

    code/                  19 numbered pipeline stages plus a driver
    raw_data/              MMLU items, model responses, concreteness norms
    resources/             fonts and the AMR parser checkpoint
    results_precomputed/   every artifact the paper reports, plus figures

`results_precomputed/` is not decoration. The two parsing stages take about ten
hours between them, so the parses, the IRT estimates and the extracted
indicators are shipped. Every analysis script looks in `results/` first and
falls back to `results_precomputed/`, which means you can reproduce any number
in the paper without re-parsing anything.

## Running it

    pip install -r requirements.txt
    python code/run_framework.py --list
    python code/run_framework.py --from 8 --to 19

Stages 1--7 build the data (alignment, IRT calibration, the five dimension
extractors). Stages 8--15 are the analyses reported in the original submission.
Stages 16--19 are the revision analyses: cross-validation and baseline models,
the robust invariance tests, the Rasch and split-half replication checks, and
the figures.

Individual stages run standalone:

    python code/17_robust_invariance.py

`--only 16 17` runs a subset, `--dry-run` prints the plan without executing it.

## Where the numbers live

| File | Contents |
| --- | --- |
| `results_summary.txt` | IRT calibration, path analysis, VIF, rank displacement |
| `results_crossvalidation_and_baselines.txt` | per-subject CV fit, nested baselines, coefficient intervals |
| `results_robust_invariance.txt` | Holm correction, HC3 Wald test, bootstrap, permutation, diagnostics, power |
| `results_specification_replication.txt` | Rasch refit, split-half replication |
| `mmlu_wscg_sensitivity.csv` | the 123-scheme weight perturbation grid |

Stages 1--15 append to the shared `results_summary.txt`; stages 16--18 each
rewrite their own file, so rerunning one of them will not duplicate a section.

## Seeds

The master seed (1729) is set once in `code/config.py`. The bootstrap,
permutation, cross-validation and split-half routines derive separate streams
from it, so rerunning one script cannot shift another script's numbers. Nothing
in the pipeline is stochastic without a seed.

## Runtime

Measured wall-clock for one complete 19-stage run starting from the
shipped caches, on a 6-core / 12-thread Intel Core i5-11260H laptop
with 8 GB of RAM and no GPU. JAX runs on its CPU backend.

The two parsing stages are not timed here. They were run once, on
other hardware, to produce the caches that ship with this repository,
and we have no measurement of them worth quoting.

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
frequencies come from `wordfreq`. The AMR checkpoint under
`resources/model_stog/` is excluded from version control by size; the pipeline
falls back to `raw_data/mmlu_amr_precomputed.csv` if it is absent.

## A note on the per-subject numbers

Per-subject regressions fit nine predictors to pools as small as 100 items.
Their in-sample R-squared values are badly optimistic --- cross-validated fit is
negative for 54 of 57 subjects --- and the paper reports them as descriptive
only. If you build on this code, use the partition-level fits. They are the ones
that survive out-of-sample validation.
