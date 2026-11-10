"""
step2_similarity.py - STEP 2: which diseases look alike? (similarity network, twins, families, genes)
=====================================================================================================
Concepts: one-mode PROJECTION of the bipartite network (disease-disease similarity), information-weighted cosine
          similarity, nearest-neighbour graph, near-duplicate "twin" clusters, family cohesion, MULTILAYER validation with genes
          (link prediction: does symptom similarity predict that two diseases share a gene?).

Plain meaning
  * Two diseases are similar when they share many symptoms, counting rare shared symptoms much more than common ones.
  * Twin: a disease whose most similar other disease has similarity >= 0.8 (almost the same symptom profile).
  * Gene check: genes were NOT used to build the similarity. If similar diseases still tend to share a gene, the
    similarity captures real biology (AUC 0.5 = no relationship, 1.0 = perfect).
"""
import re

import matplotlib.pyplot as plt
import numpy as np

import network_tools as nt
from utils import auc_score, explain, get_rng, savefig, write_json

FAMILIES = {"Ehlers-Danlos": r"Ehlers-Danlos", "Marfan / Loeys-Dietz": r"Marfan|Loeys-Dietz",
            "RASopathies": r"Noonan syndrome|Costello|Cardiofaciocutaneous|LEOPARD|Leopard|Legius|Neurofibromatosis, type 1",
            "Joubert": r"Joubert", "Bardet-Biedl": r"Bardet-Biedl",
            "Lysosomal (Gaucher, Fabry, Pompe, MPS)": r"Gaucher|Fabry|Pompe|Mucopolysaccharidosis"}


def get_neighbours(data, cfg):
    """Nearest neighbours, cached on disk (computing them takes ~10 seconds)."""
    f = cfg.res_dir / "_cache_neighbours.npz"
    if f.exists():
        z = np.load(f)
        if z["nb"].shape == (data.n, cfg.knn):
            return z["nb"], z["sm"], z["best"], z["dens"]
    nb, sm, best, dens = nt.top_neighbours(data, cfg.knn)
    cfg.res_dir.mkdir(exist_ok=True)
    np.savez(f, nb=nb, sm=sm, best=best, dens=dens)
    return nb, sm, best, dens


def run(data, cfg):
    print("[Step 2] Similarity network, twins, families, genes")
    rng = get_rng(cfg.seed + 2)
    nb, sm, best, dens = get_neighbours(data, cfg)
    src = data.source
    res = {}
    # ---- twins
    tw = {}
    for th in cfg.twin_thresholds:
        m = best >= th
        cross = src[m] != src[nb[m, 0]]
        tw[str(th)] = {"diseases": int(m.sum()), "share": float(m.mean()), "cross_database_share": float(cross.mean()) if m.any() else 0.0,
                       "within_database": int((~cross).sum())}
    order = [i for i in np.argsort(-best) if src[i] == src[nb[i, 0]] and data.ids[i] < data.ids[nb[i, 0]]][:8]
    examples = [(data.disease_names[i], data.disease_names[nb[i, 0]], float(best[i])) for i in order]
    res["twins"] = {"by_threshold": tw, "examples": examples, "mean_best_similarity": float(best.mean()),
                    "median_best_similarity": float(np.median(best))}
    # ---- family cohesion
    fam = {}
    bg_idx = rng.choice(data.n, 2000, replace=False)
    for name, pat in FAMILIES.items():
        idx = [i for i, nm in enumerate(data.disease_names) if re.search(pat, nm)]
        if len(idx) < 3:
            continue
        S = (data.Dw[idx] @ data.Dw[idx].T).toarray()
        iu = np.triu_indices(len(idx), 1)
        within = S[iu]
        bg = (data.Dw[idx] @ data.Dw[bg_idx].T).toarray().ravel()
        fam[name] = {"n": len(idx), "within": float(within.mean()), "background": float(bg.mean()),
                     "ratio": float(within.mean() / bg.mean()), "auc": auc_score(np.concatenate([within, bg]),
                                                                                   np.concatenate([np.ones(len(within)), np.zeros(len(bg))]))}
    res["families"] = fam
    # ---- gene-sharing validation (a second layer of the network)
    withg = [i for i, d in enumerate(data.ids) if d in data.genes]
    by_gene = {}
    for i in withg:
        for g in data.genes[data.ids[i]]:
            by_gene.setdefault(g, []).append(i)
    pos = set()
    for members in by_gene.values():
        if len(members) <= 60:                                   # skip a few huge gene families to stay fast
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    pos.add((members[x], members[y]) if members[x] < members[y] else (members[y], members[x]))
    pos = sorted(pos)
    if len(pos) > cfg.gene_pairs:
        pos = [pos[k] for k in rng.choice(len(pos), cfg.gene_pairs, replace=False)]
    na = rng.choice(withg, 3 * len(pos))
    nbb = rng.choice(withg, 3 * len(pos))
    posset = set(pos)
    neg = [(min(x, y), max(x, y)) for x, y in zip(na, nbb) if x != y and (min(x, y), max(x, y)) not in posset][:len(pos)]
    a = np.array([p[0] for p in pos] + [p[0] for p in neg])
    b = np.array([p[1] for p in pos] + [p[1] for p in neg])
    share = np.array([True] * len(pos) + [False] * len(neg))
    sim = np.asarray(data.Dw[a].multiply(data.Dw[b]).sum(1)).ravel()
    auc = auc_score(sim, share)
    boots = [auc_score(sim[ix], share[ix]) for ix in (rng.integers(0, len(sim), len(sim)) for _ in range(30))]
    # does the single nearest look-alike share a gene?
    pairs = [(i, nb[i, 0]) for i in withg if data.ids[nb[i, 0]] in data.genes]
    nn_share = float(np.mean([len(data.genes[data.ids[i]] & data.genes[data.ids[j]]) > 0 for i, j in pairs]))
    n_all = len(withg) * (len(withg) - 1) / 2
    res["gene_validation"] = {"pairs": int(len(sim)), "pairs_sharing_gene": int(share.sum()), "base_rate": float(len(pos) / n_all),
                              "auc": auc, "auc_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                              "nearest_lookalike_shares_gene": nn_share, "diseases_with_genes": len(withg)}
    write_json(cfg.res_dir / "step2.json", res)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].hist(best, bins=40, color="#4C78A8")
    for th, col in zip(cfg.twin_thresholds, ("#E45756", "#B2182B")):
        ax[0].axvline(th, color=col, ls="--", label=f"twin if >= {th}")
    ax[0].set(xlabel="similarity to the most similar other disease", ylabel="diseases", title="How alike is each disease to its closest look-alike?")
    ax[0].legend(fontsize=8)
    names = list(fam)
    ax[1].barh(names[::-1], [fam[n]["ratio"] for n in names][::-1], color="#54A24B")
    ax[1].set(xlabel="within-family similarity / similarity to random diseases", title="Disease families are cohesive")
    ax[2].hist([sim[~share], sim[share]], bins=30, density=True, color=["#7F7F7F", "#E45756"], label=["no shared gene", "share a gene"])
    ax[2].set(xlabel="symptom similarity of the pair", ylabel="density", title=f"Similar symptoms predict shared genes (AUC {auc:.2f})", yscale="log")
    ax[2].legend(fontsize=8)
    savefig(fig, cfg.fig_dir / "2_similarity.png")

    t8, t9 = tw["0.8"], tw["0.9"]
    print(f"    twins (>=0.8): {t8['diseases']} diseases ({100 * t8['share']:.1f}%); gene-sharing AUC {auc:.2f}")
    explain([
        f"Each disease has a closest look-alike. {t8['diseases']} diseases ({100 * t8['share']:.1f}%) have a twin with similarity >= 0.8 and {t9['diseases']} ({100 * t9['share']:.1f}%) >= 0.9. "
        f"{100 * (1 - t8['cross_database_share']):.0f}% of the 0.8-twins sit inside the same database (genetic sub-types of one disease family, e.g. '{examples[0][0]}' and '{examples[0][1]}'); "
        f"the rest are OMIM-Orphanet duplicates of the same disease, which a careful analysis must merge.",
        "Families are cohesive: " + "; ".join(f"{k} {v['ratio']:.1f}x" for k, v in list(fam.items())[:4]) + " more alike than random diseases.",
        f"Genes were never used to build the similarity, yet it predicts shared genes (AUC {auc:.2f}, 95% interval {res['gene_validation']['auc_ci'][0]:.2f}-{res['gene_validation']['auc_ci'][1]:.2f}); "
        f"for {100 * nn_share:.0f}% of diseases their single nearest look-alike shares a gene (random pairs: {100 * res['gene_validation']['base_rate']:.2f}%).",
    ])
    return res
