"""
step6_diagnosability.py - STEP 6: which rare diseases are hard to diagnose, and why? (the "diagnosability map")
================================================================================================================
Concepts: network position as a predictor (supervised learning on network features), local density, twin structure,
          information content, community (organ system) effects.

For every disease we know from step 5 whether it was found in the top 10 when only 25% of its symptoms were known.
We ask which NETWORK features of the disease predict a failure:
  number of symptoms listed, mean information content of its symptoms, share of rare symptoms, similarity to its nearest look-alike
  (twin), number of look-alikes with similarity >= 0.5 (local density), number of known genes.
AUC 0.5 = a feature tells nothing; 1.0 = it separates hard from easy diseases perfectly.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

import network_tools as nt
from step2_similarity import get_neighbours
from utils import auc_score, explain, savefig, wilson_ci, write_json


def run(data, cfg):
    print("[Step 6] Diagnosability map")
    f = cfg.res_dir / "_cache_ranks25.npz"
    if not f.exists():
        raise FileNotFoundError("Run step 5 first (it saves the simulated-diagnosis ranks).")
    z = np.load(f)
    rows, ranks = z["rows"], z["ranks"]
    nb, sm, best, dens = get_neighbours(data, cfg)
    pop = nt.popularity(data)
    nsym = np.asarray(data.Ddir.sum(1)).ravel()
    mic = np.array([data.ic[data.direct_terms(i)].mean() for i in rows])
    spec = np.array([(pop[data.direct_terms(i)] <= 50).mean() for i in rows])
    ngen = np.array([len(data.genes.get(data.ids[i], ())) for i in rows])
    X = np.column_stack([nsym[rows], mic, spec, best[rows], dens[rows], ngen])
    names = ["number of symptoms listed", "mean information content of its symptoms", "share of rare (specific) symptoms",
             "similarity to nearest look-alike (twin)", "number of look-alikes (similarity >= 0.5)", "number of known genes"]
    fail = (ranks > 10).astype(int)
    uni = {}
    for k, nm in enumerate(names):
        a = auc_score(X[:, k], fail)
        uni[nm] = {"auc": max(a, 1 - a), "harder_when": "higher" if a >= 0.5 else "lower"}
    Xs = StandardScaler().fit_transform(X)
    Xs[:, 4] = StandardScaler().fit_transform(np.log1p(X[:, [4]])).ravel()
    p_lr = cross_val_predict(LogisticRegression(max_iter=1000), Xs, fail, cv=5, method="predict_proba")[:, 1]
    p_gb = cross_val_predict(HistGradientBoostingClassifier(max_depth=3, max_iter=150, random_state=cfg.seed), X, fail, cv=5, method="predict_proba")[:, 1]
    auc_lr, auc_gb = auc_score(p_lr, fail), auc_score(p_gb, fail)
    # by organ system and by database
    org = data.organ[rows]
    by_org = []
    for o in range(len(data.organ_names)):
        m = org == o
        if m.sum() >= 40:
            h = int((ranks[m] <= 10).sum())
            lo, hi = wilson_ci(h, int(m.sum()))
            by_org.append({"organ": data.organ_names[o], "n": int(m.sum()), "top10": h / m.sum(), "low": lo, "high": hi})
    by_org.sort(key=lambda r: r["top10"])
    src = data.source[rows]
    by_src = {s: {"n": int((src == s).sum()), "top10": float((ranks[src == s] <= 10).mean())} for s in np.unique(src)}
    worst = np.argsort(-ranks)[:12]
    hardest = [{"disease": data.disease_names[rows[i]], "rank": int(ranks[i]), "twin": data.disease_names[nb[rows[i], 0]],
                "twin_similarity": float(best[rows[i]])} for i in worst]
    twin_hard = {}
    for lo_, hi_ in ((0, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.01)):
        m = (best[rows] >= lo_) & (best[rows] < hi_)
        twin_hard[f"{lo_}-{min(hi_, 1.0)}"] = {"n": int(m.sum()), "top10": float((ranks[m] <= 10).mean()) if m.any() else None}
    nrow = nsym[rows]
    by_count = {}
    for lo_, hi_ in ((1, 3), (4, 6), (7, 10), (11, 20), (21, 100000)):
        m = (nrow >= lo_) & (nrow <= hi_)
        by_count[f"{lo_}-{hi_}" if hi_ < 100000 else f"{lo_}+"] = {"n": int(m.sum()), "top10": float((ranks[m] <= 10).mean()) if m.any() else None}
    res = {"n": int(len(rows)), "miss_rate": float(fail.mean()), "univariate": uni, "by_symptom_count": by_count,
           "median_symptoms_hardest200": float(np.median(nrow[np.argsort(-ranks)[:200]])), "median_symptoms_all": float(np.median(nrow)), "auc_logistic": auc_lr, "auc_boosting": auc_gb,
           "by_organ": by_org, "by_database": by_src, "hardest": hardest, "by_twin_similarity": twin_hard,
           "median_rank": float(np.median(ranks))}
    write_json(cfg.res_dir / "step6.json", res)

    fig, ax = plt.subplots(1, 4, figsize=(22, 4.8))
    order = sorted(uni, key=lambda k: uni[k]["auc"])
    short = {"number of symptoms listed": "symptoms listed", "mean information content of its symptoms": "mean information content",
             "share of rare (specific) symptoms": "share of rare symptoms", "similarity to nearest look-alike (twin)": "twin similarity",
             "number of look-alikes (similarity >= 0.5)": "look-alikes (sim >= 0.5)", "number of known genes": "known genes"}
    ax[0].barh([short.get(k, k) for k in order], [uni[k]["auc"] for k in order], color="#4C78A8")
    ax[0].axvline(0.5, color="k", lw=0.8)
    ax[0].set(xlim=(0.45, 0.75), xlabel="AUC for predicting a missed diagnosis", title=f"What predicts difficulty? (combined AUC {auc_lr:.2f})")
    ax[1].barh([r["organ"].replace("Abnormality of the ", "").replace("Abnormality of ", "")[:32] for r in by_org], [100 * r["top10"] for r in by_org], color="#54A24B")
    ax[1].set(xlabel="% found in the top 10 (25% of symptoms known)", title="Diagnosability by organ system")
    ks = list(twin_hard)
    ax[2].bar(ks, [100 * (twin_hard[k]["top10"] or 0) for k in ks], color="#E45756")
    ax[2].set(xlabel="similarity to the nearest look-alike", ylabel="% found in the top 10", title="Twins hide from diagnosis")
    kc = list(by_count)
    ax[3].bar(kc, [100 * (by_count[k]["top10"] or 0) for k in kc], color="#F58518")
    ax[3].set(xlabel="symptoms listed for the disease", ylabel="% found in the top 10", title="U-shape in symptom count")
    savefig(fig, cfg.fig_dir / "6_diagnosability.png")

    top3 = sorted(uni, key=lambda k: -uni[k]["auc"])[:3]
    print(f"    miss rate {100 * fail.mean():.0f}%; combined model AUC {auc_lr:.2f} (logistic), {auc_gb:.2f} (boosting)")
    explain([
        f"{100 * fail.mean():.0f}% of diseases are missed (not in the top 10) when only 25% of their symptoms are known. Which ones? Network features alone predict it with AUC {auc_lr:.2f} "
        f"(logistic) / {auc_gb:.2f} (gradient boosting), where 0.5 would be chance.",
        "Strongest single predictors: " + "; ".join(f"{k} ({uni[k]['auc']:.2f})" for k in top3) + " (the symptom-count effect is U-shaped, see below).",
        "The number of symptoms matters in a U-shaped way: success is " + ", ".join(f"{100 * v['top10']:.0f}% ({k} symptoms)" for k, v in by_count.items())
        + ". Because the simulated doctor always knows at least two symptoms, diseases with 1-3 listed symptoms are almost completely known; the hardest are mid-sized entries "
        f"(median {res['median_symptoms_hardest200']:.0f} symptoms for the 200 hardest vs {res['median_symptoms_all']:.0f} overall), where 25% means only two or three clues.",
        f"Twins hide: diseases whose nearest look-alike has similarity >= 0.8 are found in the top 10 {100 * twin_hard['0.8-1.0']['top10']:.0f}% of the time, "
        f"versus {100 * twin_hard['0-0.3']['top10']:.0f}% for diseases with no close look-alike.",
        f"Hardest organ systems (lowest success): {by_org[0]['organ']} ({100 * by_org[0]['top10']:.0f}%), {by_org[1]['organ']} ({100 * by_org[1]['top10']:.0f}%).",
    ])
    return res
