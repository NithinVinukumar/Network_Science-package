"""
step1_network.py - STEP 1: what does the disease-symptom network look like?
===========================================================================
Concepts: bipartite network, degree distribution, heavy tail / power-law exponent, hubs, information content,
          k-core decomposition (core-periphery), degree assortativity, friendship paradox.

Plain meaning
  * The network has two kinds of dots: DISEASES and SYMPTOMS. A line = "this disease has this symptom".
  * "Degree" of a symptom = number of diseases that have it; of a disease = number of symptoms it has.
  * Heavy tail: a few symptoms are in thousands of diseases, most in only a few.
  * Information content (IC) = -ln(share of diseases with the symptom): rare symptoms carry more information.
  * k-core: repeatedly delete dots with fewer than k lines; what survives is the dense "core".
  * Assortativity: do well-connected dots attach to well-connected dots (positive) or to poorly connected ones (negative)?
  * Friendship paradox: the symptoms of a typical disease are, on average, MORE popular than a typical symptom.
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from utils import ccdf, explain, savefig, tail_exponent, write_json


def run(data, cfg):
    print("[Step 1] Network structure")
    sd = np.asarray(data.Ddir.sum(0)).ravel()            # diseases per symptom (direct annotations)
    dd = np.asarray(data.Ddir.sum(1)).ravel()            # symptoms per disease
    used = sd > 0
    gam = {k: tail_exponent(sd[used], k) for k in (10, 30, 100)}
    order = np.argsort(-sd)[:10]
    hubs = [(data.term_names[j], int(sd[j])) for j in order]
    pop = np.asarray(data.Dprop.sum(0)).ravel()

    r, c = data.Ddir.nonzero()
    BG = nx.Graph()
    BG.add_edges_from((("d", int(i)), ("s", int(j))) for i, j in zip(r, c))
    core = nx.core_number(BG)
    kmax = max(core.values())
    inner = [n for n, k in core.items() if k == kmax]
    inner_s = sorted((n[1] for n in inner if n[0] == "s"), key=lambda j: -sd[j])
    inner_d = [n[1] for n in inner if n[0] == "d"]
    assort = nx.degree_assortativity_coefficient(BG)
    nb = np.asarray(data.Ddir @ sd).ravel() / np.maximum(dd, 1)
    ic_hub = float(data.ic[order[0]])
    ic_single = float(np.max(data.ic[used]))

    # figures
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    x, p = ccdf(sd[used])
    ax[0].loglog(x, p, "o", ms=3, color="#4C78A8")
    ax[0].set(xlabel="diseases having the symptom (degree)", ylabel="share of symptoms with at least this degree",
              title="A few symptoms are huge hubs")
    ax[1].hist(dd, bins=np.arange(0, 100, 3), color="#54A24B")
    ax[1].set(xlabel="symptoms per disease", ylabel="diseases", title=f"Symptoms per disease (median {int(np.median(dd))})")
    ax[2].scatter(np.maximum(pop, 1), data.ic, s=3, alpha=0.4, color="#E45756")
    ax[2].set(xscale="log", xlabel="diseases having the symptom (propagated)", ylabel="information content (IC)",
              title="Popular symptoms carry little information")
    savefig(fig, cfg.fig_dir / "1_network_structure.png")

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    vals, counts = np.unique(list(core.values()), return_counts=True)
    ax.bar(vals, counts, color="#B279A2")
    ax.set(xlabel="k-core number (higher = deeper in the dense core)", ylabel="number of nodes (diseases + symptoms)",
           title=f"k-core: innermost core k={kmax} ({len(inner_d)} diseases, {len(inner_s)} symptoms)")
    savefig(fig, cfg.fig_dir / "1_kcore.png")

    res = {
        "diseases": data.n, "symptom_terms": data.t, "direct_links": int(data.Ddir.nnz),
        "sources": {k: int((data.source == k).sum()) for k in np.unique(data.source)},
        "symptoms_per_disease_median": float(np.median(dd)), "symptoms_per_disease_mean": float(dd.mean()),
        "symptoms_per_disease_max": int(dd.max()), "symptoms_used_directly": int(used.sum()),
        "share_symptoms_in_one_disease": float((sd[used] == 1).mean()),
        "tail_exponent": {str(k): {"gamma": v[0], "n": v[1]} for k, v in gam.items()},
        "top_hubs": hubs, "kcore_max": kmax, "kcore_diseases": len(inner_d), "kcore_symptoms": len(inner_s),
        "kcore_top_symptoms": [data.term_names[j] for j in inner_s[:8]],
        "assortativity": assort, "avg_diseases_per_symptom": float(sd[used].mean()),
        "avg_neighbour_degree_of_disease": float(nb.mean()), "friendship_ratio": float(nb.mean() / sd[used].mean()),
        "ic_of_top_hub": ic_hub, "ic_of_rarest_symptom": ic_single,
    }
    write_json(cfg.res_dir / "step1.json", res)
    print(f"    {data.n} diseases, {data.t} symptom terms, {int(data.Ddir.nnz)} links; hub: {hubs[0][0]} in {hubs[0][1]} diseases")
    explain([
        f"Two kinds of dots: {data.n} diseases and {data.t} symptom terms; {int(data.Ddir.nnz)} direct links. A typical disease lists {int(np.median(dd))} symptoms.",
        f"Heavy tail: the biggest hub, '{hubs[0][0]}', occurs in {hubs[0][1]} diseases, while {100 * res['share_symptoms_in_one_disease']:.0f}% of symptoms appear in only one disease. "
        f"Tail exponent about {gam[30][0]:.1f} (for degrees >= 30).",
        f"Information content (IC = -ln of the share of diseases with the symptom): the top hub scores {ic_hub:.1f}, a symptom of a single disease scores {ic_single:.1f}; "
        "so one rare symptom counts for several common ones when diseases are compared.",
        f"Dense core: the innermost k-core (k={kmax}) holds {len(inner_d)} diseases and {len(inner_s)} symptoms, led by {', '.join(res['kcore_top_symptoms'][:4])}.",
        f"Assortativity {assort:.2f} (negative): well-connected dots tend to link to poorly connected ones, a hub-and-spoke pattern. "
        f"Friendship paradox: the symptoms of a typical disease are linked to {res['avg_neighbour_degree_of_disease']:.0f} diseases on average, "
        f"{res['friendship_ratio']:.0f} times more than a typical symptom ({res['avg_diseases_per_symptom']:.0f}).",
    ])
    return res
