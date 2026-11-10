"""
step3_communities.py - STEP 3: do diseases form families? (community structure)
=================================================================================
Concepts: nearest-neighbour graph, community detection (Louvain), modularity, normalised mutual information (NMI) against
          known categories (organ systems), community stability, participation coefficient (bridge diseases).

Plain meaning
  * Join every disease to its 10 most similar diseases. Communities are groups of diseases that are densely linked to each other.
  * Modularity Q measures how much more linked the communities are inside than chance predicts (above ~0.3 = real structure).
  * NMI compares the communities with the organ-system label of each disease (0 = unrelated, 1 = identical).
  * A bridge disease has look-alikes in several different communities (participation coefficient near 1).
"""
import collections

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

import network_tools as nt
from step2_similarity import get_neighbours
from utils import explain, savefig, write_json


def louvain_labels(nb, sm, seed):
    G = nx.Graph()
    for i in range(nb.shape[0]):
        for j, s in zip(nb[i], sm[i]):
            G.add_edge(int(i), int(j), weight=float(max(s, 1e-6)))
    comms = sorted(nx.community.louvain_communities(G, weight="weight", seed=seed), key=len, reverse=True)
    lab = np.zeros(nb.shape[0], dtype=int)
    for k, c in enumerate(comms):
        lab[list(c)] = k
    return lab, comms, nx.community.modularity(G, comms, weight="weight")


def run(data, cfg):
    print("[Step 3] Communities")
    nb, sm, best, dens = get_neighbours(data, cfg)
    lab, comms, Q = louvain_labels(nb, sm, cfg.seed)
    nmi = float(normalized_mutual_info_score(data.organ, lab))
    stab = [float(normalized_mutual_info_score(lab, louvain_labels(nb, sm, cfg.seed + k)[0])) for k in (1, 2, 3)]
    # purity: share of each community's diseases in its most common organ system
    top = []
    for k, c in enumerate(comms[:12]):
        cnt = collections.Counter(int(data.organ[i]) for i in c)
        o, n = cnt.most_common(1)[0]
        top.append({"community": k, "size": len(c), "main_organ": data.organ_names[o] if o >= 0 else "none", "purity": n / len(c)})
    P = nt.participation_coefficient(nb, lab)
    nsym = np.asarray(data.Ddir.sum(1)).ravel()
    P_ok = np.where(nsym >= 10, P, -1)            # sparse annotation makes neighbours scatter: only diseases with >= 10 symptoms
    bridge_order = np.argsort(-P_ok)[:10]
    bridges = [(data.disease_names[i], float(P[i])) for i in bridge_order]
    M = np.zeros((12, len(data.organ_names)))
    for k, c in enumerate(comms[:12]):
        for i in c:
            if data.organ[i] >= 0:
                M[k, data.organ[i]] += 1
        M[k] /= max(len(c), 1)
    keepcols = np.argsort(-M.sum(0))[:10]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    im = ax[0].imshow(M[:, keepcols], aspect="auto", cmap="viridis")
    ax[0].set_xticks(range(len(keepcols)), [data.organ_names[j].replace("Abnormality of the ", "").replace("Abnormality of ", "")[:22] for j in keepcols], rotation=60, ha="right", fontsize=8)
    ax[0].set_yticks(range(12), [f"community {k} (n={len(comms[k])})" for k in range(12)], fontsize=8)
    ax[0].set_title("Communities vs organ systems (share of each community)")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    ax[1].bar(range(len(top)), [t["purity"] for t in top], color="#4C78A8")
    ax[1].set(xlabel="community (largest first)", ylabel="share in its main organ system", title=f"Community purity (NMI overall {nmi:.2f})")
    savefig(fig, cfg.fig_dir / "3_communities.png")

    res = {"n_communities": len(comms), "modularity": Q, "nmi_with_organ_systems": nmi, "stability_nmi": stab,
           "largest_sizes": [len(c) for c in comms[:10]], "top_communities": top, "bridge_diseases": bridges,
           "mean_participation": float(P.mean())}
    write_json(cfg.res_dir / "step3.json", res)
    print(f"    {len(comms)} communities, modularity {Q:.2f}, NMI with organ systems {nmi:.2f}")
    explain([
        f"Linking each disease to its {cfg.knn} closest look-alikes and detecting communities (Louvain) gives {len(comms)} communities with modularity Q = {Q:.2f} "
        f"(above 0.3 is real structure). Rerunning with other random seeds gives almost the same groups (stability NMI {np.mean(stab):.2f}).",
        f"The communities follow organ systems but are not identical to them (NMI {nmi:.2f}): e.g. community 1 is {100 * top[1]['purity']:.0f}% '{top[1]['main_organ']}'.",
        f"Bridge diseases (well-annotated diseases whose look-alikes lie in many communities) include {bridges[0][0]} and {bridges[1][0]}: these are the diseases most easily confused across families.",
    ])
    return res
