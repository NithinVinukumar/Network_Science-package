"""
step9_algorithms.py - STEP 9: which ranking ALGORITHM finds the right disease best? (algorithm benchmark)
==========================================================================================================
Concepts: similarity (projection) ranking vs DIFFUSION on the network (random walk with restart = personalised PageRank), information
          content weighting, benchmark with simulated patients.

Three algorithms rank all diseases for the same simulated patient (25% of the symptoms known, 2 wrong ones added):
  1. IC-weighted cosine similarity  (our method: rare symptoms count more)
  2. Jaccard overlap                (plain overlap: every symptom counts the same)
  3. Random walk with restart       (a walker starts on the patient's symptoms and walks disease <-> symptom; the diseases it visits
                                     most are the answer; hub symptoms spread the walker thinly)
Scenarios: a random 25% of symptoms, only the most common 25%, only the most specific 25%.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix

import network_tools as nt
from utils import explain, get_rng, savefig, write_json

ALGOS = {"cosine": "IC-weighted cosine (ours)", "jaccard": "Jaccard overlap", "rwr": "random walk with restart"}
SCEN = [("random", "random 25%"), ("common", "only common 25%"), ("specific", "only specific 25%")]


def _rank(scores, rows):
    true = scores[np.arange(len(rows)), rows]
    return 1 + (scores > true[:, None]).sum(1)


def run(data, cfg):
    print("[Step 9] Ranking algorithms compared")
    rng = get_rng(cfg.seed + 9)
    rows = np.sort(rng.choice(data.n, min(cfg.algo_diseases, data.n), replace=False))
    res = {"patients": int(len(rows)), "rwr_continue": cfg.rwr_continue, "rwr_iters": cfg.rwr_iters, "scenarios": {}}
    for mode, label in SCEN:
        Qw = nt.make_patients(data, rows, 0.25, mode, cfg.noise_symptoms, np.random.default_rng(cfg.seed + 90))
        Qb = Qw.copy()                      # a real copy: csr_matrix(Qw) would share memory and overwrite the IC weights of Qw
        Qb.data[:] = 1.0
        ranks = {"cosine": _rank((Qw @ data.Dw.T).toarray(), rows),
                 "jaccard": _rank(nt.jaccard_scores(data, Qb), rows),
                 "rwr": _rank(nt.rwr_scores(data, Qb, cfg.rwr_continue, cfg.rwr_iters).toarray(), rows)}
        res["scenarios"][mode] = {a: {"top1": float((r <= 1).mean()), "top10": float((r <= 10).mean()), "median_rank": float(np.median(r))}
                                  for a, r in ranks.items()}
        print(f"    {label:18s} top-10: " + "  ".join(f"{ALGOS[a]} {100 * res['scenarios'][mode][a]['top10']:.0f}%" for a in ALGOS))
    write_json(cfg.res_dir / "step9.json", res)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    w = 0.26
    for k, a in enumerate(ALGOS):
        ax.bar(np.arange(len(SCEN)) + (k - 1) * w, [100 * res["scenarios"][m][a]["top10"] for m, _ in SCEN], w, label=ALGOS[a],
               color=["#E45756", "#7F7F7F", "#4C78A8"][k])
    ax.set_xticks(range(len(SCEN)), [l for _, l in SCEN])
    ax.set(ylabel="% of diseases found in the top 10", title="Which ranking algorithm finds the disease best?")
    ax.legend(fontsize=8)
    savefig(fig, cfg.fig_dir / "9_algorithms.png")

    r = res["scenarios"]["random"]
    explain([
        f"The same {len(rows)} simulated patients (25% of symptoms known + 2 wrong ones) were ranked by three algorithms. With a random 25% the true disease is in the top 10 for "
        f"{100 * r['cosine']['top10']:.0f}% (IC-weighted cosine, our method), {100 * r['jaccard']['top10']:.0f}% (Jaccard overlap) and {100 * r['rwr']['top10']:.0f}% (random walk with restart).",
        "Weighting symptoms by rarity helps when rare symptoms are present: with a random 25% or only the specific 25%, plain overlap loses because it treats a hub symptom like a rare one. "
        f"The one exception is when only the most common symptoms are known: there plain overlap is a little better ({100 * res['scenarios']['common']['jaccard']['top10']:.0f}% vs "
        f"{100 * res['scenarios']['common']['cosine']['top10']:.0f}%), because weighting deliberately discounts exactly those symptoms. The random walk is weakest everywhere: "
        "hub symptoms spread the walker thinly, but random walks also tend to favour well-connected diseases.",
        f"Whatever the algorithm, the kind of symptom matters most: only specific symptoms give {100 * res['scenarios']['specific']['cosine']['top10']:.0f}% (cosine), "
        f"only common symptoms {100 * res['scenarios']['common']['cosine']['top10']:.0f}%.",
    ])
    return res
