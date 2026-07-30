import argparse
import os
import subprocess
import sys

STAGES = [
    "01_align_and_preprocess.py",
    "02_irt_calibration.py",
    "03_dimension1_reasoning.py",
    "04_dimension2_syntax.py",
    "05_dimension3_knowledge.py",
    "06_dimension4_semantics.py",
    "07_dimension5_adversarial.py",
    "08_global_path_analysis.py",
    "09_domain_invariance_testing.py",
    "10_construct_analysis.py",
    "11_robustness_checks.py",
    "12_visualizations.py",
    "13_subject_effect_sizes.py",
    "14_parse_cache.py",
    "15_wscg_sensitivity.py",
    "16_crossvalidation_and_baselines.py",
    "17_robust_invariance.py",
    "18_specification_and_replication.py",
    "19_revised_figures.py",
]

def parse_args():
    p = argparse.ArgumentParser(
        description="Run the MMLU psychometric analysis pipeline.")
    p.add_argument("--from", dest="start", type=int, default=1,
                   help="first stage to run (1-%d)" % len(STAGES))
    p.add_argument("--to", dest="end", type=int, default=len(STAGES),
                   help="last stage to run (1-%d)" % len(STAGES))
    p.add_argument("--only", type=int, nargs="+",
                   help="run just these stage numbers")
    p.add_argument("--list", action="store_true",
                   help="print the stage list and exit")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would run without running it")
    return p.parse_args()

def selected(args):
    if args.only:
        return [(n, STAGES[n - 1]) for n in args.only
                if 1 <= n <= len(STAGES)]
    return [(i + 1, s) for i, s in enumerate(STAGES)
            if args.start <= i + 1 <= args.end]

def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.list:
        for i, script in enumerate(STAGES, 1):
            print(f"{i:>2}  {script}")
        return

    plan = selected(args)
    if not plan:
        print("No stages selected.")
        sys.exit(1)

    print(f"Running {len(plan)} of {len(STAGES)} stages.")

    for number, script in plan:
        path = os.path.join(base_dir, script)
        if not os.path.exists(path):
            print(f"[ERROR] missing stage {number}: {path}")
            sys.exit(1)
        if args.dry_run:
            print(f"  [{number:>2}] {script}")
            continue

        print(f"\n[{number:>2}/{len(STAGES)}] {script}")
        try:
            subprocess.run([sys.executable, path], check=True)
        except subprocess.CalledProcessError as exc:
            print(f"\n[ERROR] {script} exited with code {exc.returncode}")
            sys.exit(exc.returncode)

    if not args.dry_run:
        print("\nPipeline finished. Artifacts are in the results folder.")

if __name__ == "__main__":
    main()
