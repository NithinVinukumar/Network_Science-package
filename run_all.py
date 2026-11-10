"""
run_all.py - run all eight steps and write results/summary.md
=============================================================
    python run_all.py              full quality (about 6-8 minutes)
    python run_all.py --quick      fast test run (about 1 minute; noisier, do not report those numbers)
    python run_all.py --only 5 6   run only some steps (6 needs 5 to have run once)

Steps
   1 network structure              5 robustness of diagnosis to missing / misleading symptoms
   2 similarity network, twins      6 diagnosability map: which diseases are hard, and why
   3 communities                    7 link prediction: recover missing symptoms
   4 percolation                    8 Joubert syndrome case study
   9 ranking algorithms compared: IC-cosine vs Jaccard vs random walk with restart (personalised PageRank)
  10 adaptive questioning: which symptoms to ask, in what order (greedy test-cover / '20 questions' on the network)
  11 noise-robust adaptive questioning: the same game when answers can be wrong (Bayesian posterior + information gain)
  12 bipartite modules (Barber's modularity, BRIM): syndromic modules of diseases AND symptoms, and triage
Every step prints its numbers AND a plain-language "what this means" block, saves figures to figures/ and numbers to results/*.json.
"""
import argparse
import time
import traceback

import hpo_data
import reporting
import step1_network
import step2_similarity
import step3_communities
import step4_percolation
import step5_robustness
import step6_diagnosability
import step7_link_prediction
import step8_joubert
import step9_algorithms
import step10_questions
import step11_robust_questions
import step12_modules
from config import Settings

STEPS = [(1, step1_network), (2, step2_similarity), (3, step3_communities), (4, step4_percolation),
         (5, step5_robustness), (6, step6_diagnosability), (7, step7_link_prediction), (8, step8_joubert), (9, step9_algorithms), (10, step10_questions), (11, step11_robust_questions), (12, step12_modules)]


def main():
    ap = argparse.ArgumentParser(description="Rare-disease diagnosability network")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", type=int, nargs="+")
    args = ap.parse_args()
    cfg = Settings.quick() if args.quick else Settings()
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)
    cfg.res_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = hpo_data.load(cfg)
    print("=" * 78)
    print(f"Rare-disease network: {data.n} diseases, {data.t} symptom terms, {int(data.Ddir.nnz)} links (HPO)")
    print("=" * 78)
    failures = []
    for num, mod in STEPS:
        if args.only and num not in args.only:
            continue
        t = time.time()
        try:
            mod.run(data, cfg)
        except Exception:
            failures.append(f"step{num}")
            print(f"!! step {num} failed:")
            traceback.print_exc()
        print(f"    (step {num}: {time.time() - t:.0f}s)\n")
    reporting.write_summary(cfg, failures, args.quick)
    print(f"Done in {time.time() - t0:.0f}s. Failed steps: {failures or 'none'}")


if __name__ == "__main__":
    main()
