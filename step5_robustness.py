"""
step5_robustness.py - STEP 5: how robust is diagnosis to missing and misleading symptoms?
==========================================================================================
Concepts: robustness / node removal (random vs targeted), hubs vs periphery, information content, sensitivity analysis.

Simulated patient (a "doctor with an incomplete picture")
  take a disease, keep only a FRACTION of its symptoms, add a few WRONG symptoms, then rank all diseases by similarity to the patient.
  Success = the true disease is in the top 10 of the ranking (top-1 = it is ranked first).

Three ways of losing symptoms
  random     the doctor knows a random subset (random failure)
  common     the doctor only knows the most COMMON symptoms (the rare, informative ones are lost first)
  specific   the doctor only knows the most SPECIFIC symptoms (the hubs are lost first: a targeted attack on hubs)

Warning (said in the paper too): patients are built from each disease's own annotation, so the results are optimistic.
"""
import matplotlib.pyplot as plt
import numpy as np

import hpo_data as hd
import network_tools as nt
from utils import explain, get_rng, savefig, write_json

MODES = {"random": "random subset of symptoms", "common": "only the most common symptoms", "specific": "only the most specific symptoms"}


def run(data, cfg):
    print("[Step 5] Robustness of diagnosis")
    rng = get_rng(cfg.seed + 5)
    N = data.n
    rows = np.arange(N) if cfg.main_diseases is None else np.sort(rng.choice(N, cfg.main_diseases, replace=False))
    res = {"main": {}, "n_main": int(len(rows))}
    scenarios = [("random", k) for k in cfg.keep_fractions] + [("common", 0.25), ("specific", 0.25)]
    for mode, keep in scenarios:
        Q = nt.make_patients(data, rows, keep, mode, cfg.noise_symptoms, rng)
        ranks = nt.rank_of_true(data, Q, rows)
        res["main"][f"{mode}_{int(round(keep * 100))}"] = nt.hit_rates(ranks)
        if mode == "random" and abs(keep - 0.25) < 1e-9:
            np.savez(cfg.res_dir / "_cache_ranks25.npz", rows=rows, ranks=ranks)
        h = res["main"][f"{mode}_{int(round(keep * 100))}"]
        print(f"    {mode:8s} {int(keep * 100):3d}% known: top-1 {100 * h['top1']['rate']:.0f}%  top-10 {100 * h['top10']['rate']:.0f}%  top-50 {100 * h['top50']['rate']:.0f}%")

    # ---- sweeps on a subset: top-10 hit rate vs fraction of symptoms known, for the three loss modes
    sub = np.sort(rng.choice(N, min(cfg.sweep_diseases, N), replace=False))
    sweep = {m: [] for m in MODES}
    for m in MODES:
        for f in cfg.sweep_fractions:
            Q = nt.make_patients(data, sub, f, m, cfg.noise_symptoms, rng)
            h = nt.hit_rates(nt.rank_of_true(data, Q, sub))
            sweep[m].append({"fraction": f, "top10": h["top10"], "top1": h["top1"]})
    res["sweep"] = sweep
    # ---- noise sensitivity (wrong symptoms added) at 25% known, random
    noise = {}
    for nz in (0, 2, 5):
        Q = nt.make_patients(data, sub, 0.25, "random", nz, rng)
        noise[str(nz)] = nt.hit_rates(nt.rank_of_true(data, Q, sub))
    res["noise"] = noise
    # ---- sensitivity: one database only (OMIM), to see whether duplicates between databases change the picture
    om = hd.load(cfg, sources=["OMIM"])
    orow = np.arange(om.n) if cfg.main_diseases is None else np.sort(rng.choice(om.n, min(cfg.main_diseases, om.n), replace=False))
    ho = nt.hit_rates(nt.rank_of_true(om, nt.make_patients(om, orow, 0.25, "random", cfg.noise_symptoms, rng), orow))
    res["omim_only_25"] = {"diseases": om.n, **ho}
    write_json(cfg.res_dir / "step5.json", res)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    cols = {"random": "#4C78A8", "common": "#E45756", "specific": "#54A24B"}
    for m in MODES:
        x = [p["fraction"] for p in sweep[m]]
        y = np.array([p["top10"]["rate"] for p in sweep[m]])
        lo = np.array([p["top10"]["low"] for p in sweep[m]])
        hi = np.array([p["top10"]["high"] for p in sweep[m]])
        ax[0].errorbar(x, 100 * y, yerr=[100 * (y - lo), 100 * (hi - y)], fmt="-o", ms=4, color=cols[m], label=MODES[m], capsize=2)
    ax[0].set(xlabel="fraction of the disease's symptoms known", ylabel="% of diseases found in the top 10",
              title="Robustness: random loss vs losing hubs vs losing rare symptoms")
    ax[0].legend(fontsize=7)
    names = [f"{m}\n{int(k * 100)}%" for m, k in scenarios]
    vals = [res["main"][f"{m}_{int(round(k * 100))}"]["top10"]["rate"] for m, k in scenarios]
    ax[1].bar(range(len(names)), [100 * v for v in vals], color=[cols[m] for m, _ in scenarios])
    ax[1].set_xticks(range(len(names)), names, fontsize=8)
    ax[1].set(ylabel="% in the top 10 (all diseases)", title="Main scenarios")
    ax[2].bar(list(noise), [100 * noise[k]["top10"]["rate"] for k in noise], color="#B279A2")
    ax[2].set(xlabel="number of WRONG symptoms added", ylabel="% in the top 10 (25% known)", title="Sensitivity to misleading symptoms")
    savefig(fig, cfg.fig_dir / "5_robustness.png")

    r25, c25, s25 = res["main"]["random_25"]["top10"]["rate"], res["main"]["common_25"]["top10"]["rate"], res["main"]["specific_25"]["top10"]["rate"]
    explain([
        f"With only 25% of a disease's symptoms known (plus {cfg.noise_symptoms} wrong ones) the true disease is in the top 10 for {100 * r25:.0f}% of diseases "
        f"(top-1: {100 * res['main']['random_25']['top1']['rate']:.0f}%).",
        f"The kind of symptom matters far more than the number: knowing only the most COMMON 25% gives {100 * c25:.0f}%, knowing only the most SPECIFIC 25% gives {100 * s25:.0f}%. "
        "Losing the hubs (common symptoms) costs almost nothing; losing the rare ones is devastating. This is the opposite of an airline network, where hubs are what matters.",
        f"Wrong symptoms hurt a little: 0 / 2 / 5 wrong symptoms give {100 * noise['0']['top10']['rate']:.0f}% / {100 * noise['2']['top10']['rate']:.0f}% / {100 * noise['5']['top10']['rate']:.0f}%.",
        f"Sensitivity check: restricting to OMIM only (no OMIM-Orphanet duplicates) gives {100 * ho['top10']['rate']:.0f}% in the top 10.",
        "Caveat: patients are simulated from each disease's own annotation, so real-world accuracy would be lower.",
    ])
    return res
