"""
step12_modules.py - STEP 12: syndromic MODULES of diseases AND symptoms (Barber's bipartite modularity, the BRIM algorithm)
=========================================================================================================================
Concepts: bipartite modularity Q_B (Barber 2007), the BRIM algorithm (Bipartite, Recursively Induced Modules), co-clustering of the two
          node types, null model with the same degrees, module profiles, triage.

Louvain (step 3) works on the one-mode PROJECTION: it groups diseases only. BRIM works on the bipartite network itself and groups diseases
and symptoms TOGETHER, so every module comes with its own signature symptoms.
  Q_B = (1/m) sum_ij (A_ij - k_i d_j / m) [module(i) = module(j)]   (links inside modules minus the number expected by chance)
  BRIM: start with random symptom modules; every disease joins the module that raises Q_B most; every symptom does the same; repeat.
Evaluated on: bipartite modularity (against Louvain's communities and organ systems, symptoms assigned optimally), agreement with organ
systems (NMI), share-a-gene rate inside groups, and TRIAGE: can 25% of a patient's symptoms (+2 wrong ones) predict the patient's module?
Honest note: BRIM wins on bipartite modularity because it optimises exactly that; Louvain's groups are closer to organ systems and share
genes more often (they are smaller). The point of BRIM is the joint disease-and-symptom modules and the triage.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import normalized_mutual_info_score

import network_tools as nt
from step2_similarity import get_neighbours
from step3_communities import louvain_labels
from utils import explain, get_rng, savefig, write_json


def _compact(row, col):
    """Renumber modules 0..c'-1, largest module (most diseases) first."""
    order = np.argsort(-np.bincount(row, minlength=max(row.max(), col.max()) + 1))
    order = [m for m in order if (row == m).any() or (col == m).any()]
    remap = {int(m): i for i, m in enumerate(order)}
    return np.array([remap[int(x)] for x in row]), np.array([remap[int(x)] for x in col]), len(order)


def get_modules(data, cfg, c=None, restarts=None, use_cache=True):
    """BRIM modules (cached on disk). Returns (disease labels, symptom labels, number of modules, Q_B)."""
    f = cfg.res_dir / "_cache_modules.npz"
    if use_cache and f.exists():
        z = np.load(f)
        if z["row"].shape[0] == data.n:
            return z["row"], z["col"], int(z["c"]), float(z["q"])
    rng = get_rng(cfg.seed + 12)
    A = data.Ddir.tocsr()
    c = c or cfg.brim_c_main
    best = None
    for _ in range(restarts or cfg.brim_restarts):
        row, col, q = nt.brim(A, c, rng)
        if best is None or q > best[3]:
            best = (row, col, c, q)
    row, col, c = _compact(best[0], best[1])
    if use_cache:
        cfg.res_dir.mkdir(parents=True, exist_ok=True)
        np.savez(f, row=row, col=col, c=c, q=best[3])
    return row, col, c, float(best[3])


def describe(data, A, row, col, c, top=8, n_sig=6):
    """The largest modules with their main organ system and signature symptoms (share of the module's diseases listing the symptom x IC)."""
    cov = nt.module_coverage(A, row, c)
    sizes = np.bincount(row, minlength=c)
    out = []
    for mod in np.argsort(-sizes)[:top]:
        rows = np.flatnonzero(row == mod)
        cols = np.flatnonzero(col == mod)
        sc = cov[mod, cols] * data.ic[cols]
        sig = cols[np.argsort(-sc)[:n_sig]]
        org = np.bincount(data.organ[rows][data.organ[rows] >= 0], minlength=len(data.organ_names))
        o = int(org.argmax())
        out.append({"module": int(mod), "diseases": int(len(rows)), "symptoms": int(len(cols)),
                    "main_organ": data.organ_names[o].replace("Abnormality of the ", "").replace("Abnormality of ", ""),
                    "organ_share": float(org[o] / len(rows)), "signature": [data.term_names[j] for j in sig]})
    return out


def gene_rate(data, labels, rng, n):
    """Share of disease pairs from the SAME group (both with known genes) that share a gene."""
    groups = {}
    for i, l in enumerate(labels):
        groups.setdefault(int(l), []).append(i)
    big = [v for v in groups.values() if len(v) >= 2]
    w = np.array([len(v) * (len(v) - 1) / 2 for v in big], float)
    w /= w.sum()
    hit = tot = 0
    for _ in range(n):
        v = big[rng.choice(len(big), p=w)]
        a, b = rng.choice(v, 2, replace=False)
        ga, gb = data.genes.get(data.ids[a]), data.genes.get(data.ids[b])
        if ga and gb:
            tot += 1
            hit += bool(ga & gb)
    return hit / max(tot, 1), tot


def triage_accuracy(data, A, labels, c, n, seed):
    """Can a patient (random 25% of a disease's listed symptoms + 2 wrong ones) be assigned to the disease's own group?
    Score of a group = sum over the patient's symptoms of the log-lift of the symptom in the group (nt.module_scores)."""
    rng = np.random.default_rng(seed)
    prof = nt.module_profile(A, labels, c)
    ok = 0
    targets = rng.choice(data.n, min(n, data.n), replace=False)
    for t in targets:
        dt = A.indices[A.indptr[t]:A.indptr[t + 1]]
        k = min(len(dt), max(2, int(round(0.25 * len(dt)))))
        sel = np.concatenate([rng.choice(dt, k, replace=False), rng.integers(0, data.t, 2)])
        ok += int(np.argmax(nt.module_scores(prof, sel)) == labels[t])
    return ok / len(targets)


def run(data, cfg):
    print("[Step 12] Bipartite modules (BRIM)")
    rng = get_rng(cfg.seed + 12)
    A = data.Ddir.tocsr()
    scan = {}
    for cc in sorted(set(cfg.brim_c_values) | {cfg.brim_c_main}):
        scan[int(cc)] = float(max(nt.brim(A, cc, rng)[2] for _ in range(max(3, cfg.brim_restarts // 2))))
    row, col, c, q = get_modules(data, cfg, use_cache=False)
    (cfg.res_dir).mkdir(parents=True, exist_ok=True)
    np.savez(cfg.res_dir / "_cache_modules.npz", row=row, col=col, c=c, q=q)
    nb, sm, _, _ = get_neighbours(data, cfg)
    lou, _, _ = louvain_labels(nb, sm, cfg.seed)
    cL = int(lou.max()) + 1
    org = np.where(data.organ < 0, data.organ.max() + 1, data.organ)
    cO = int(org.max()) + 1
    rnd = rng.integers(0, c, data.n)
    parts = {"BRIM modules": (row, c), "Louvain communities": (lou, cL), "organ systems": (org, cO), "random groups": (rnd, c)}
    qb = {k: nt.bipartite_modularity(A, lab, nt.symptom_modules_for(A, lab, cc), cc) if k != "BRIM modules" else q for k, (lab, cc) in parts.items()}
    nmi = {k: float(normalized_mutual_info_score(data.organ, lab)) for k, (lab, _) in parts.items() if k in ("BRIM modules", "Louvain communities")}
    genes = {k: gene_rate(data, lab, rng, cfg.module_gene_pairs) for k, (lab, _) in parts.items()}
    tri = {"BRIM modules": triage_accuracy(data, A, row, c, cfg.module_triage_targets, cfg.seed + 120),
           "organ systems": triage_accuracy(data, A, org, cO, cfg.module_triage_targets, cfg.seed + 120)}
    sizes = np.bincount(row, minlength=c)
    mods = describe(data, A, row, col, c)
    res = {"modules": c, "modularity_by_c": scan, "Q_B": qb, "nmi_with_organ": nmi,
           "gene_share_rate": {k: v[0] for k, v in genes.items()}, "gene_pairs": {k: v[1] for k, v in genes.items()},
           "triage_accuracy": tri, "triage_chance": 1.0 / c, "triage_largest_module": float(sizes.max() / data.n),
           "triage_organ_largest": float(np.bincount(org).max() / data.n), "largest_modules": mods}
    write_json(cfg.res_dir / "step12.json", res)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    names = list(qb)
    ax[0].bar(names, [qb[k] for k in names], color=["#E45756", "#4C78A8", "#54A24B", "#7F7F7F"])
    ax[0].set(ylabel="bipartite modularity Q_B", title="Modules with symptoms assigned optimally")
    ax[0].tick_params(axis="x", labelrotation=20, labelsize=8)
    ax[1].bar(["BRIM modules\n(%d)" % c, "organ systems\n(%d)" % cO, "always the\nlargest group"], [100 * tri["BRIM modules"], 100 * tri["organ systems"], 100 * res["triage_largest_module"]],
              color=["#E45756", "#54A24B", "#7F7F7F"])
    ax[1].set(ylabel="% of patients assigned to the right group", title="Triage from 25% of the symptoms")
    top = mods[:8]
    ax[2].barh([f"module {m['module']}: {m['main_organ'][:16]}" for m in top][::-1], [m["diseases"] for m in top][::-1], color="#B279A2")
    ax[2].set(xlabel="diseases in the module", title="Largest syndromic modules")
    savefig(fig, cfg.fig_dir / "12_modules.png")

    print(f"    {c} modules, Q_B {q:.3f} (Louvain partition {qb['Louvain communities']:.3f}, organ systems {qb['organ systems']:.3f}); triage {100 * tri['BRIM modules']:.0f}% vs organ systems {100 * tri['organ systems']:.0f}%")
    explain([
        f"BRIM (Barber's bipartite modularity) groups diseases AND symptoms into {c} syndromic modules (Q_B is almost flat between {min(scan)} and {max(scan)} modules: "
        f"{min(scan.values()):.2f}-{max(scan.values()):.2f}; {c} modules are comparable to the {cO} organ systems). Their bipartite modularity is {q:.2f}, against {qb['Louvain communities']:.2f} for "
        f"Louvain's communities and {qb['organ systems']:.2f} for organ systems (symptoms assigned optimally to each), and {qb['random groups']:.2f} for random groups. BRIM wins because it optimises exactly this score.",
        "Every module comes with signature symptoms. For example: " + "; ".join(f"module {m['module']} ({m['main_organ']}, {100 * m['organ_share']:.0f}%): " + ", ".join(m['signature'][:3]) for m in mods[:3]) + ".",
        f"Honest comparison with step 3: Louvain's communities are closer to organ systems (NMI {nmi['Louvain communities']:.2f} vs {nmi['BRIM modules']:.2f}) and share a gene more often inside groups "
        f"({100 * res['gene_share_rate']['Louvain communities']:.2f}% vs {100 * res['gene_share_rate']['BRIM modules']:.2f}%; random groups {100 * res['gene_share_rate']['random groups']:.2f}%). BRIM is not better at everything; it gives something different.",
        f"Triage: from 25% of a patient's symptoms (plus 2 wrong ones), the module is predicted correctly for {100 * tri['BRIM modules']:.0f}% of diseases (chance {100 / c:.0f}%, always the largest module {100 * res['triage_largest_module']:.0f}%), "
        f"against {100 * tri['organ systems']:.0f}% when predicting the organ system the same way.",
    ])
    return res
