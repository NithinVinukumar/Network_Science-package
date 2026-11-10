"""
demo.py - LIVE USE CASE: "a doctor uses our network to work out a rare disease from a few findings"
===================================================================================================
    python demo.py                              pauses between scenes (use this when presenting)
    python demo.py --auto                       no pauses (to test it)
    python demo.py --disease "Marfan syndrome"  scene 3 uses another disease

Every scene prints the NETWORK CONCEPT it uses and the CODE that runs, so you can point at the screen.

Scene 1  The doctor enters 3 findings          bipartite network -> information-weighted similarity -> ranked diseases
Scene 2  The system suggests the next question neighbourhood of the top candidates (the link-prediction idea)
Scene 3  Stress test: what if only common symptoms are known?   robustness: hubs vs rare symptoms
Scene 4  Warning: twins                        similarity network, twin clusters, gene layer (multilayer)
Scene 5  Does it hold for ALL 12,866 diseases? results of steps 3, 4, 5, 7 (read from results/*.json)
Everything in scenes 1-4 is computed live from the data. Scene 5 reads the saved results of run_all.py.
This is a research demonstration on curated disease descriptions, not a diagnostic tool.
"""
import argparse
import json
import textwrap

import numpy as np

import hpo_data
import network_tools as nt
from config import Settings
from explore import query_vector

PATIENT = ["Molar tooth sign on MRI", "Ataxia", "Nystagmus"]
TWIN_PAIRS = [("Porokeratosis 1, multiple types", "Porokeratosis 7, multiple types"),
              ("Spermatogenic failure 66", "Spermatogenic failure 69")]


def banner(n, title, concept, code):
    print("\n" + "=" * 92)
    print(f"SCENE {n}: {title}")
    print(f"  NETWORK CONCEPT : {concept}")
    print(f"  CODE THAT RUNS  : {code}")
    print("=" * 92)


def pause(auto):
    if auto:
        return
    try:
        input("\n  [press Enter for the next scene] ")
    except EOFError:
        pass


def term_id(data, name):
    hits = [i for i in data.find_term(name, limit=100) if data.term_names[i].lower() == name.lower()]
    if not hits:
        raise ValueError(f"symptom '{name}' not found")
    return hits[0]


def rank_diseases(data, cols, top=10):
    """Rank all diseases for the given symptom columns. Returns (similarities, top indices, query indicator)."""
    Qn, qbin = query_vector(data, cols)
    sims = (data.Dw @ Qn.T).toarray().ravel()
    return sims, np.argsort(-sims)[:top], qbin


def next_question(data, order, qbin):
    """The symptom that splits the current top candidates best: present in some but not all, and informative."""
    frac = data.Dprop[order].toarray().mean(0)
    score = frac * (1 - frac) * data.ic
    score[qbin] = -1
    return np.argsort(-score)[:5], frac


def show_top(data, sims, order, n):
    for r, j in enumerate(order[:n], 1):
        print(f"    {r}. similarity {sims[j]:.2f}   {data.disease_names[j]}")


def pair_similarity(data, a, b):
    return float((data.Dw[a] @ data.Dw[b].T).toarray()[0, 0])


def scene1(data):
    banner(1, "USE CASE: a doctor enters three findings for a child",
           "bipartite disease-symptom network; information content; similarity (projection)",
           "explore.query_vector()  ->  data.Dw @ query  (hpo_data.py: Dprop, ic, Dw)")
    cols = [term_id(data, n) for n in PATIENT]
    pop = nt.popularity(data)
    print("  Step A - how the network sees each finding (hubs carry little information, rare symptoms a lot):")
    print(f"     {'finding':28s} {'diseases that have it':>22s} {'information content':>21s}")
    for c in cols:
        print(f"     {data.term_names[c]:28s} {int(pop[c]):>22d} {data.ic[c]:>21.2f}")
    sims, order, qbin = rank_diseases(data, cols, 10)
    print(f"\n  Step B - the findings are compared with all {data.n} diseases (cosine of IC-weighted symptom lists):")
    show_top(data, sims, order, 8)
    n_j = sum("Joubert" in data.disease_names[j] for j in order)
    print(f"\n  Step C - decision-support output: {n_j} of the top 10 are Joubert syndrome subtypes, so the FAMILY is clear.")
    a, b = int(order[0]), int(order[1])
    pair = pair_similarity(data, a, b)
    print(f"           Top candidate: {data.disease_names[a]} ({sims[a]:.2f}); runner-up {data.disease_names[b]} ({sims[b]:.2f}).")
    print(f"           The two candidates are {pair:.2f} similar to each other: " +
          ("TWIN WARNING, symptoms alone may not separate them." if pair >= 0.8 else "different enough that a well-chosen question can separate them."))
    nxt, frac = next_question(data, order, qbin)
    print("\n  What this shows: three findings, weighted by how rare they are, narrow 12,866 diseases to one family.")
    return {"top": [data.disease_names[j] for j in order], "n_joubert_top10": n_j, "next": [data.term_names[j] for j in nxt],
            "ic": [float(data.ic[c]) for c in cols], "pop": [int(pop[c]) for c in cols], "pair_similarity": pair}


def scene2(data):
    banner(2, "The system suggests what to ask next; suppose the patient answers YES",
           "neighbourhood of the top candidates: similar diseases 'vote' for symptoms (the idea behind link prediction, step 7)",
           "demo.next_question(): symptoms present in about half of the top-10 look-alikes, weighted by information content")
    cols = [term_id(data, n) for n in PATIENT]
    history = []
    for step in (1, 2):
        sims, order, qbin = rank_diseases(data, cols, 10)
        nxt, frac = next_question(data, order, qbin)
        j = int(nxt[0])
        cols.append(j)
        sims2, order2, _ = rank_diseases(data, cols, 10)
        print(f"\n  Question {step}: does the child have '{data.term_names[j]}' (present in {int(round(frac[j] * 10))} of the top 10)?  Suppose yes. New ranking:")
        show_top(data, sims2, order2, 4)
        history.append({"asked": data.term_names[j], "top": data.disease_names[order2[0]], "top_sim": float(sims2[order2[0]]),
                        "second_sim": float(sims2[order2[1]])})
    print("\n  What this shows: the network itself suggests which symptom to ask about. (The 'yes' answers are an example, not a real patient.)")
    return history


def scene3(data, disease="Joubert syndrome 6"):
    banner(3, "Stress test: the SAME disease, the SAME number of symptoms known, but different kinds of symptoms",
           "robustness: losing hubs (common symptoms) vs losing the periphery (rare symptoms)",
           "network_tools.popularity() ranks symptoms by number of diseases; same ranking as network_tools.make_patients() / step5_robustness.py")
    hits = data.find_disease(disease, limit=50)
    exact = [i for i in hits if data.disease_names[i].lower() == disease.lower()]
    if not hits:
        raise ValueError(f"disease '{disease}' not found")
    i = (exact or hits)[0]
    dt = data.direct_terms(i)
    pop = nt.popularity(data)
    k = max(2, int(round(0.25 * len(dt))))
    print(f"  Disease: {data.disease_names[i]} ({len(dt)} listed symptoms). Suppose the doctor knows only {k} of them (about 25%).")
    out = {}
    for mode, sel in (("only COMMON symptoms (the hubs)", dt[np.argsort(-pop[dt], kind='stable')[:k]]),
                      ("only SPECIFIC symptoms (the periphery)", dt[np.argsort(pop[dt], kind='stable')[:k]])):
        sims, order, _ = rank_diseases(data, list(sel), 10)
        rank = 1 + int((sims > sims[i]).sum())
        print(f"\n  Doctor knows {mode}:")
        for x in sel:
            print(f"     - {data.term_names[x]}   (in {int(pop[x])} diseases)")
        print(f"   -> the true disease ranks #{rank} of {data.n}")
        out[mode] = rank
    print("\n  What this shows: in this network, losing the hubs is harmless but losing the rare symptoms breaks diagnosis.")
    print("  (An airline network behaves the opposite way: hubs matter most.)")
    return {"common": out["only COMMON symptoms (the hubs)"], "specific": out["only SPECIFIC symptoms (the periphery)"]}


def scene4(data, cfg=None):
    banner(4, "Warning system: some diseases are twins that symptoms cannot separate",
           "similarity (projection) network; twin clusters; a second layer of the network = genes (multilayer)",
           "data.Dw rows (cosine similarity) and data.genes (genes_to_disease.txt); step2_similarity.py counts the twins")
    res = []
    for a, b in TWIN_PAIRS:
        ia, ib = data.find_disease(a, 5), data.find_disease(b, 5)
        if not ia or not ib:
            continue
        ia = ([i for i in ia if data.disease_names[i] == a] or ia)[0]
        ib = ([i for i in ib if data.disease_names[i] == b] or ib)[0]
        sim = pair_similarity(data, ia, ib)
        ga, gb = sorted(data.genes.get(data.ids[ia], [])), sorted(data.genes.get(data.ids[ib], []))
        sa = [data.term_names[j] for j in data.direct_terms(ia)]
        print(f"\n  {data.disease_names[ia]}   vs   {data.disease_names[ib]}")
        print(f"    symptom similarity {sim:.2f}   listed symptoms: {', '.join(sa)}")
        print(f"    gene layer: {', '.join(ga) or 'none'}   vs   {', '.join(gb) or 'none'}")
        res.append({"similarity": sim, "genes": [ga, gb]})
    cfg = cfg or Settings()
    f2 = cfg.res_dir / "step2.json"
    if f2.exists():
        t = json.loads(f2.read_text())["twins"]["by_threshold"]["0.8"]
        print(f"\n  Over ALL diseases (step 2): {t['diseases']} diseases ({100 * t['share']:.1f}%) have a twin with similarity of at least 0.8.")
    else:
        print("\n  About 1 in 8 diseases (11.9%) has a twin with similarity of at least 0.8 (run run_all.py to see it computed).")
    print("  What this shows: the symptom layer cannot separate these diseases; the gene layer can. The system flags such cases for a gene test.")
    print("  (Both entries list only two symptoms in the database, an honest limit of the data.)")
    return res


def scene5(cfg):
    banner(5, "Does it hold for ALL 12,866 diseases? (saved results of run_all.py)",
           "null model + percolation (step 4); robustness (step 5); communities (step 3); link prediction (step 7)",
           "run_all.py -> step3_communities.py, step4_percolation.py, step5_robustness.py, step7_link_prediction.py")
    f5, f4, f3, f7 = (cfg.res_dir / f"step{n}.json" for n in (5, 4, 3, 7))
    if not f5.exists():
        print("  Run `python run_all.py` first to create the results.")
        return None
    m = json.loads(f5.read_text())["main"]
    print("  Robustness (step 5): doctor knows 25% of the symptoms (+2 wrong ones). True disease in the top 10:")
    print(f"     random 25% .............. {100 * m['random_25']['top10']['rate']:.0f}% of diseases")
    print(f"     only COMMON 25% ......... {100 * m['common_25']['top10']['rate']:.0f}%")
    print(f"     only SPECIFIC 25% ....... {100 * m['specific_25']['top10']['rate']:.0f}%")
    if f4.exists():
        p = json.loads(f4.read_text())
        print(f"  Percolation vs null model (step 4): half of all diseases join one giant cluster at similarity "
              f"{p['giant_half_threshold_real']:.2f} (real) vs {p['giant_half_threshold_null']:.2f} (random null).")
    if f3.exists():
        c = json.loads(f3.read_text())
        print(f"  Communities (step 3): {c['n_communities']} communities, modularity {c['modularity']:.2f}.")
    if f7.exists():
        lp = json.loads(f7.read_text())
        print(f"  Link prediction (step 7): hidden symptoms recovered {100 * lp['recall_network']:.0f}% vs {100 * lp['recall_baseline']:.0f}% for guessing the popular ones.")
    print("\n  Figures to open now: figures/5_robustness.png and figures/4_percolation.png")
    return {"random": m["random_25"]["top10"]["rate"], "common": m["common_25"]["top10"]["rate"], "specific": m["specific_25"]["top10"]["rate"]}


def main():
    ap = argparse.ArgumentParser(description="Live use-case demo")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--disease", default="Joubert syndrome 6")
    args = ap.parse_args()
    cfg = Settings()
    data = hpo_data.load(cfg)
    print(f"Loaded {data.n} rare diseases and {data.t} symptom terms (Human Phenotype Ontology).")
    for fn in (lambda: scene1(data), lambda: scene2(data), lambda: scene3(data, args.disease), lambda: scene4(data, cfg), lambda: scene5(cfg)):
        fn()
        pause(args.auto)
    print("\nEnd of demo. Limits: simulated findings from curated disease descriptions; not a diagnostic tool.")


if __name__ == "__main__":
    main()
