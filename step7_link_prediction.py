"""
step7_link_prediction.py - STEP 7: can the network predict MISSING links? (link prediction)
=============================================================================================
Concepts: link prediction, collaborative filtering on a bipartite network, neighbourhood-based scores, baseline comparison.

Task: for a disease, hide 20% of its listed symptoms. Using only the remaining symptoms, find the 50 most similar diseases and let them
"vote" for symptoms (a symptom scores higher if many similar diseases have it). Success = the hidden symptoms appear among the top 20
suggestions. Baseline: simply suggest the most popular symptoms (what a doctor would guess without a network).
The gene-layer link prediction (does similarity predict shared genes?) is in step 2.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from utils import explain, get_rng, savefig, wilson_ci, write_json


def run(data, cfg):
    print("[Step 7] Link prediction")
    rng = get_rng(cfg.seed + 7)
    nsym = np.asarray(data.Ddir.sum(1)).ravel()
    cand = np.flatnonzero(nsym >= 5)
    test = rng.choice(cand, min(cfg.lp_diseases, len(cand)), replace=False)
    pop_direct = np.asarray(data.Ddir.sum(0)).ravel().astype(float)
    rec_net, rec_base, rr_net = [], [], []
    for i in test:
        dt = data.direct_terms(i)
        k = max(1, int(round(cfg.lp_hide * len(dt))))
        hid = rng.choice(dt, k, replace=False)
        keep = np.setdiff1d(dt, hid)
        q = csr_matrix((np.ones(len(keep)), (np.zeros(len(keep), dtype=int), keep)), shape=(1, data.t))
        q = q @ data.Anc
        q.data[:] = 1.0
        qn = normalize(q.multiply(data.ic).tocsr()).astype(np.float32)
        sims = (data.Dw @ qn.T).toarray().ravel()
        sims[i] = 0.0
        nbr = np.argpartition(-sims, cfg.lp_neighbours)[:cfg.lp_neighbours]
        score = np.asarray(csr_matrix(sims[nbr]).dot(data.Ddir[nbr]).todense()).ravel()
        score[keep] = -1.0
        top = np.argpartition(-score, cfg.lp_top)[:cfg.lp_top]
        rec_net.append(len(set(top.tolist()) & set(hid.tolist())) / len(hid))
        b = pop_direct.copy()
        b[keep] = -1.0
        topb = np.argpartition(-b, cfg.lp_top)[:cfg.lp_top]
        rec_base.append(len(set(topb.tolist()) & set(hid.tolist())) / len(hid))
        order = np.argsort(-score)
        pos = np.array([np.flatnonzero(order == h)[0] + 1 for h in hid[:3]])
        rr_net.append(float(np.mean(1.0 / pos)))
    rec_net, rec_base = np.array(rec_net), np.array(rec_base)
    boot = [rec_net[ix].mean() - rec_base[ix].mean() for ix in (rng.integers(0, len(rec_net), len(rec_net)) for _ in range(200))]
    res = {"diseases": int(len(test)), "recall_network": float(rec_net.mean()), "recall_baseline": float(rec_base.mean()),
           "gain": float(rec_net.mean() / max(rec_base.mean(), 1e-9)),
           "difference_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
           "top_k": cfg.lp_top, "hidden_fraction": cfg.lp_hide, "mean_reciprocal_rank": float(np.mean(rr_net))}
    write_json(cfg.res_dir / "step7.json", res)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.bar(["popularity\nbaseline", "network\nmethod"], [100 * res["recall_baseline"], 100 * res["recall_network"]], color=["#7F7F7F", "#E45756"])
    ax.set(ylabel=f"% of hidden symptoms found in the top {cfg.lp_top} suggestions", title="Link prediction: recovering hidden symptoms")
    savefig(fig, cfg.fig_dir / "7_link_prediction.png")
    print(f"    recall@{cfg.lp_top}: network {rec_net.mean():.2f} vs baseline {rec_base.mean():.2f}")
    explain([
        f"For {len(test)} diseases we hid {int(cfg.lp_hide * 100)}% of their symptoms and asked the network to suggest them from look-alike diseases.",
        f"The network recovers {100 * rec_net.mean():.0f}% of the hidden symptoms in its top {cfg.lp_top} suggestions, versus {100 * rec_base.mean():.0f}% for simply suggesting the most popular symptoms "
        f"({res['gain']:.1f} times better; difference 95% interval {res['difference_ci'][0]:.2f}-{res['difference_ci'][1]:.2f}).",
        "Use: this points curators to symptoms that are probably missing from sparsely described diseases (annotation gaps).",
    ])
    return res
