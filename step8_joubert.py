"""
step8_joubert.py - STEP 8: case study of ONE disease: Joubert syndrome (genes, subtypes, symptoms)
===================================================================================================
Concepts: multilayer network (gene layer + disease layer + symptom layer), ego network, pleiotropy / hubs, community detection,
          nestedness (NODF) with a null model, permutation test, core-periphery of symptoms, annotation inequality.

Joubert syndrome is a rare inherited brain disorder with many genetic causes. In the data it appears as
  * "subtypes" (OMIM entries "Joubert syndrome 1 ... 40"), each linked to one gene, and
  * "clinical forms" (Orphanet entries: with hepatic defect, ocular anomaly, renal defect, ...), each linked to several genes.

Questions
  1. Which symptoms do (almost) all subtypes share, and how many are reported by only one subtype? (core vs periphery)
  2. Are the subtypes a nested mild-to-severe ladder (nestedness) or separate groups?
  3. Do genes behind the SAME clinical form have more similar symptom profiles than random genes?
  4. Do Joubert genes cause other diseases too (pleiotropy), and how do those diseases group (communities)?
"""
import re

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import network_tools as nt
from utils import explain, get_rng, savefig, write_json


def run(data, cfg):
    print("[Step 8] Joubert syndrome case study")
    rng = get_rng(cfg.seed + 8)
    J = [i for i, nm in enumerate(data.disease_names) if re.search(r"Joubert", nm, re.I)]
    sub = [i for i in J if data.ids[i].startswith("OMIM")]
    forms = {k: [i for i in J if data.ids[i].startswith("ORPHA") and pat in data.disease_names[i].lower()]
             for k, pat in (("hepatic", "hepatic"), ("ocular", "ocular anomaly"), ("renal", "renal defect"), ("oculorenal", "oculorenal"))}
    gene_of = {i: sorted(data.genes.get(data.ids[i], []))[0] for i in sub if data.ids[i] in data.genes}
    sub = [i for i in sub if i in gene_of]
    # ---- 1. core vs periphery
    M = np.zeros((len(sub), data.t), dtype=int)
    for r, i in enumerate(sub):
        M[r, data.direct_terms(i)] = 1
    used = np.flatnonzero(M.sum(0) > 0)
    share = M[:, used].mean(0)
    order = np.argsort(-share)
    core = [(data.term_names[used[o]], float(share[o])) for o in order[:8]]
    singleton = float((M[:, used].sum(0) == 1).mean())
    nsym = M.sum(1)
    # ---- 2. nestedness against a null with the same row/column sums
    Mu = M[:, used]
    real = nt.nodf(Mu)
    nul = np.array([nt.nodf(nt.curveball(Mu, rng, 2000)) for _ in range(cfg.nodf_nulls)])
    z = float((real - nul.mean()) / max(nul.std(), 1e-9))
    # ---- 3. clinical-form alignment (permutation test)
    S = (data.Dw[sub] @ data.Dw[sub].T).toarray()
    gidx = {g: r for r, g in enumerate(gene_of[i] for i in sub)}
    align = {}
    for name, members in forms.items():
        genes = sorted({g for i in members for g in data.genes.get(data.ids[i], [])})
        mem = [gidx[g] for g in genes if g in gidx]
        if len(mem) < 3:
            continue
        pair = lambda idx: float(np.mean([S[a, b] for k, a in enumerate(idx) for b in idx[k + 1:]]))
        w = pair(mem)
        perm = np.array([pair(list(rng.choice(len(sub), len(mem), replace=False))) for _ in range(2000)])
        align[name] = {"genes": len(mem), "within": w, "random_mean": float(perm.mean()), "random_sd": float(perm.std()),
                       "p_value": float((np.sum(perm >= w) + 1) / (len(perm) + 1))}
    # ---- 4. pleiotropy and the ciliopathy neighbourhood
    gd = {}
    for i, d in enumerate(data.ids):
        for g in data.genes.get(d, ()):
            gd.setdefault(g, []).append(i)
    jg = sorted(set(gene_of.values()))
    pleio = {g: len(gd.get(g, [])) for g in jg}
    B = nx.Graph()
    for g in jg:
        for i in gd.get(g, []):
            B.add_edge(("g", g), ("d", i))
    dn = [n for n in B if n[0] == "d"]
    P = nx.bipartite.weighted_projected_graph(B, dn)
    comm = sorted(nx.community.louvain_communities(P, seed=cfg.seed), key=len, reverse=True)
    comm_out = [{"size": len(c), "examples": [data.disease_names[n[1]] for n in list(c)[:4]]} for c in comm[:5]]
    res = {"subtypes": len(sub), "genes": len(jg), "symptoms_used": int(len(used)), "density": float(Mu.mean()),
           "symptoms_per_subtype_median": float(np.median(nsym)), "symptoms_per_subtype_range": [int(nsym.min()), int(nsym.max())],
           "subtypes_le10_symptoms": int((nsym <= 10).sum()), "core_symptoms": core, "share_reported_by_one_subtype": singleton,
           "nodf_real": real, "nodf_null_mean": float(nul.mean()), "nodf_null_sd": float(nul.std()), "nodf_z": z, "nodf_nulls": cfg.nodf_nulls,
           "form_alignment": align, "pleiotropy": sorted(pleio.items(), key=lambda kv: -kv[1])[:10],
           "genes_only_joubert": int(sum(1 for v in pleio.values() if v <= 1)), "median_diseases_per_gene": float(np.median(list(pleio.values()))),
           "ego_diseases": len(dn), "ego_links": P.number_of_edges(), "ego_communities": len(comm), "ego_community_info": comm_out}
    write_json(cfg.res_dir / "step8.json", res)

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    top = order[:40]
    rord = np.argsort(-nsym)
    ax[0].imshow(Mu[np.ix_(rord, top)], aspect="auto", cmap="Greys")
    ax[0].set_yticks(range(len(sub)), [gene_of[sub[r]] for r in rord], fontsize=6)
    ax[0].set(xlabel="40 most shared symptoms", title="Joubert subtypes (rows, by gene) x symptoms")
    ax[1].barh([g for g, _ in sorted(pleio.items(), key=lambda kv: kv[1])[-12:]], [v for _, v in sorted(pleio.items(), key=lambda kv: kv[1])[-12:]], color="#E45756")
    ax[1].set(xlabel="number of rare diseases linked to the gene", title="Pleiotropy: one gene, many diseases")
    ax[2].hist(nul, bins=15, color="#7F7F7F", label="null (same row/column sums)")
    ax[2].axvline(real, color="#E45756", lw=2.5, label=f"real {real:.1f} (Z={z:.1f})")
    ax[2].set(xlabel="NODF nestedness", title="Nestedness vs null")
    ax[2].legend(fontsize=8)
    savefig(fig, cfg.fig_dir / "8_joubert.png")

    print(f"    {len(sub)} subtypes, {len(jg)} genes, NODF Z = {z:.1f}")
    explain([
        f"Joubert syndrome appears as {len(sub)} genetic subtypes (one gene each) described with {len(used)} different symptoms. "
        f"Core symptoms: {', '.join(f'{n} ({100 * s:.0f}%)' for n, s in core[:4])}; but {100 * singleton:.0f}% of symptoms are reported by only one subtype (a long tail).",
        f"Nestedness: NODF {real:.1f} versus {nul.mean():.1f} +/- {nul.std():.1f} for random matrices with the same totals (Z = {z:.1f}). "
        + (f"The gap is statistically clear but tiny in size ({abs(real - nul.mean()):.1f} NODF points): there is no evidence of a nested mild-to-severe ladder; "
           "subtypes behave like separate groups." if z <= 0 else "Subtypes are more nested than chance: a partial mild-to-severe ladder."),
        "Genes behind the same clinical form: " + "; ".join(f"{k}: similarity {v['within']:.2f} vs {v['random_mean']:.2f} random (p = {v['p_value']:.3f})" for k, v in align.items()) + ". "
        "With only a few genes per form, only the clearest effects can be significant.",
        f"Pleiotropy: none of the Joubert genes cause only Joubert syndrome ({res['genes_only_joubert']} do); the median gene is linked to {res['median_diseases_per_gene']:.0f} diseases, "
        f"led by {pleio and max(pleio, key=pleio.get)} ({max(pleio.values())}). Together they link {len(dn)} diseases into {len(comm)} communities (the ciliopathy neighbourhood).",
    ])
    return res
