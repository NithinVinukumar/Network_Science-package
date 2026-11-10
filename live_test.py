"""
live_test.py - LIVE TESTING: you type the inputs, the system answers (nothing is pre-scripted)
==============================================================================================
    python live_test.py

Menu
  1  Diagnose from findings YOU type  - type symptoms, see the ranked diseases, answer the system's questions (y / n / type another symptom)
  2  Stress test a disease YOU type    - same disease, same number of symptoms: common-only vs specific-only vs random
  3  Twin check: two diseases YOU type  - are they separable by symptoms? what do the genes say?
  4  Look up a symptom                  - how many diseases have it, and its information content (hub or rare?)
  5  Compare 3 ranking algorithms       - the same findings ranked by IC-cosine, Jaccard overlap and random walk with restart (PageRank-style)
  6  Play 20 questions                  - the system asks yes/no symptom questions (greedy best split) until ONE disease is left
  7  Noisy 20 questions                 - the same game, but the patient makes MISTAKES; the system keeps probabilities and recovers
  8  Which syndromic module?            - BRIM modules (diseases AND symptoms together): which module do your findings point to?
  0  Quit

Every option prints the NETWORK CONCEPT it uses and the CODE that runs. Press Enter to accept the value in [brackets].
This is a research demonstration on curated disease descriptions, not a diagnostic tool.
"""
import numpy as np

import hpo_data
import network_tools as nt
import step12_modules
from config import Settings
from explore import ask, pick_disease, query_vector

SEED = 7


def tag(concept, code):
    print("\n  NETWORK CONCEPT : " + concept)
    print("  CODE THAT RUNS  : " + code)


def resolve_symptom(data, text):
    """Turn typed text into one symptom column. Exact name wins; one partial match is used; several partial matches -> you choose."""
    text = text.strip()
    if not text:
        return None
    hits = data.find_term(text, limit=500)
    exact = [h for h in hits if data.term_names[h].lower() == text.lower()]
    if exact:
        return exact[0]
    if not hits:
        print(f"    (no symptom contains '{text}')")
        return None
    if len(hits) == 1:
        return hits[0]
    pop = nt.popularity(data)
    hits = sorted(hits, key=lambda h: -pop[h])[:8]
    print(f"    several symptoms contain '{text}':")
    for k, h in enumerate(hits, 1):
        print(f"      {k}. {data.term_names[h]}   (in {int(pop[h])} diseases)")
    n = ask("which one? (number)", 1, int)
    return hits[n - 1] if 1 <= n <= len(hits) else hits[0]


def rank_diseases(data, cols, top=10):
    Qn, qbin = query_vector(data, cols)
    sims = (data.Dw @ Qn.T).toarray().ravel()
    return sims, np.argsort(-sims)[:top], qbin


def rank_with_absent(data, cols, absent, top=10):
    """Rank diseases for the findings in `cols`. A symptom the patient does NOT have (answered 'n') counts as evidence AGAINST:
    every disease that has it loses its (weighted) share of that symptom, Dw[d, symptom]. Rare absent symptoms push down more."""
    Qn, qbin = query_vector(data, cols)
    sims = (data.Dw @ Qn.T).toarray().ravel()
    for a in absent:
        sims = sims - data.Dw[:, a].toarray().ravel()
    return sims, np.argsort(-sims)[:top], qbin


def next_questions(data, order, qbin, asked):
    """Symptoms present in some (not all) of the top candidates and informative: asking about them separates the candidates."""
    frac = data.Dprop[order].toarray().mean(0)
    score = frac * (1 - frac) * data.ic
    score[qbin] = -1
    for a in asked:
        score[a] = -1
    return np.argsort(-score)[:5], frac


def pair_similarity(data, a, b):
    return float((data.Dw[a] @ data.Dw[b].T).toarray()[0, 0])


def show_findings(data, cols):
    pop = nt.popularity(data)
    print(f"     {'finding':36s} {'diseases that have it':>22s} {'information content':>20s}")
    for c in cols:
        kind = "hub" if pop[c] > 1000 else ("rare" if pop[c] < 100 else "medium")
        print(f"     {data.term_names[c]:36s} {int(pop[c]):>22d} {data.ic[c]:>14.2f}  ({kind})")


def diagnose(data, cfg=None):
    tag("bipartite disease-symptom network; information content; similarity (projection of the network onto diseases)",
        "explore.query_vector() -> data.Dw @ query (hpo_data.py: Dprop, ic, Dw)")
    print("  Type the patient's findings as symptom names separated by commas (part of a name is fine).")
    raw = input("  findings: ")
    cols = []
    for part in [p for p in raw.split(",") if p.strip()]:
        c = resolve_symptom(data, part)
        if c is not None and c not in cols:
            cols.append(c)
    if not cols:
        print("  No symptoms recognised.")
        return None
    asked, absent, result, prev = [], [], None, None
    for _ in range(10):
        print("\n  How the network sees your findings:")
        show_findings(data, cols)
        sims, order, qbin = rank_with_absent(data, cols, absent, 10)
        if absent:
            print("\n  The patient does NOT have: " + "; ".join(data.term_names[a] for a in absent) + "  (diseases that have these are pushed down)")
        if prev is not None:
            delta = sims - prev
            moves = [("UP  ", j) for j in np.argsort(-delta)[:3] if delta[j] > 0.005] + [("DOWN", j) for j in np.argsort(delta)[:3] if delta[j] < -0.005]
            if moves:
                print("\n  Changes since your last answer (the biggest moves among all diseases):")
                for arrow, j in moves:
                    print(f"    {arrow}  {data.disease_names[j][:60]:60s} {prev[j]:.2f} -> {sims[j]:.2f}")
        prev = sims.copy()
        print(f"\n  Most likely diseases (similarity to all {data.n} diseases):")
        for r, j in enumerate(order[:8], 1):
            print(f"    {r}. {sims[j]:.2f}   {data.disease_names[j]}")
        a, b = int(order[0]), int(order[1])
        pair = pair_similarity(data, a, b)
        print(f"\n  Decision support: top candidate {data.disease_names[a]} ({sims[a]:.2f}); runner-up {data.disease_names[b]} ({sims[b]:.2f}).")
        print(f"                    the two are {pair:.2f} similar to each other: " +
              ("TWIN WARNING, symptoms alone may not separate them (gene test needed)." if pair >= 0.8 else "a well-chosen question can separate them."))
        nxt, frac = next_questions(data, order, qbin, asked)
        print("\n  Best symptoms to ask about next (neighbourhood of the top-10 candidates, weighted by information content):")
        for k, j in enumerate(nxt[:4], 1):
            print(f"    {k}. {data.term_names[j]}   (present in {int(round(frac[j] * 10))} of the top 10)")
        result = {"top": data.disease_names[a], "top_sim": float(sims[a]), "pair_similarity": pair, "n_findings": len(cols), "n_absent": len(absent),
                  "suggestions": [data.term_names[j] for j in nxt[:4]]}
        ans = input(f"\n  Does the patient have '{data.term_names[nxt[0]]}'?  y = yes (added as a finding), n = no (diseases that have it are pushed down), or type another symptom; Enter = finish: ").strip()
        if ans == "":
            break
        if ans.lower() == "y":
            cols.append(int(nxt[0]))
            asked.append(int(nxt[0]))
        elif ans.lower() == "n":
            asked.append(int(nxt[0]))
            absent.append(int(nxt[0]))
        else:
            for part in [p for p in ans.split(",") if p.strip()]:
                c = resolve_symptom(data, part)
                if c is not None and c not in cols:
                    cols.append(c)
    print("\n  What this shows: findings weighted by rarity narrow the diseases; the system tells you what to ask next and when symptoms cannot decide.")
    return result


def stress_test(data, cfg=None):
    tag("robustness: losing hubs (common symptoms) vs losing the periphery (rare symptoms)",
        "network_tools.popularity() + ranking (the same logic as network_tools.make_patients() and step5_robustness.py)")
    i = pick_disease(data, ask("disease name (part of the name is fine)", "Joubert syndrome 6"))
    if i is None:
        print("   No disease found with that text.")
        return None
    frac = ask("fraction of its symptoms the doctor knows (0.05-1)", 0.25, float)
    dt = data.direct_terms(i)
    pop = nt.popularity(data)
    k = min(len(dt), max(2, int(round(frac * len(dt)))))
    rng = np.random.default_rng(SEED)
    print(f"\n  {data.disease_names[i]} lists {len(dt)} symptoms; the doctor knows {k}.")
    out = {}
    for key, name, sel in (("random", "a RANDOM subset", rng.choice(dt, k, replace=False)),
                           ("common", "only the most COMMON symptoms (hubs)", dt[np.argsort(-pop[dt], kind="stable")[:k]]),
                           ("specific", "only the most SPECIFIC symptoms (periphery)", dt[np.argsort(pop[dt], kind="stable")[:k]])):
        sims, _, _ = rank_diseases(data, [int(x) for x in sel], 10)
        rank = 1 + int((sims > sims[i]).sum())
        print(f"\n  Doctor knows {name}:")
        for x in sel:
            print(f"     - {data.term_names[x]}   (in {int(pop[x])} diseases)")
        print(f"   -> the true disease ranks #{rank} of {data.n}")
        out[key] = rank
    print("\n  What this shows: the NUMBER of symptoms is the same; the KIND decides. Rare symptoms find the disease, hubs do not.")
    return out


def twin_check(data, cfg=None):
    tag("similarity (projection) network; twin clusters; a second layer of the network = genes (multilayer)",
        "data.Dw rows (cosine similarity) and data.genes (genes_to_disease.txt); step2_similarity.py counts the twins")
    a = pick_disease(data, ask("first disease", "Porokeratosis 1"))
    b = pick_disease(data, ask("second disease", "Porokeratosis 7"))
    if a is None or b is None:
        print("   Disease not found.")
        return None
    sim = pair_similarity(data, a, b)
    A, B = set(data.direct_terms(a)), set(data.direct_terms(b))
    ga, gb = sorted(data.genes.get(data.ids[a], [])), sorted(data.genes.get(data.ids[b], []))
    print(f"\n  {data.disease_names[a]}   vs   {data.disease_names[b]}")
    print(f"  symptom similarity: {sim:.2f}   ({len(A)} and {len(B)} listed symptoms, {len(A & B)} in common)")
    for title, s in (("only in the first", A - B), ("only in the second", B - A), ("shared", A & B)):
        names = [data.term_names[j] for j in sorted(s, key=lambda j: -data.ic[j])[:6]]
        print(f"  {title}: " + ("; ".join(names) if names else "-"))
    print(f"  gene layer: {', '.join(ga) or 'none listed'}   vs   {', '.join(gb) or 'none listed'}")
    print("\n  Verdict: " + ("TWINS. Symptoms cannot separate them; the gene layer can (a gene test is needed)." if sim >= 0.8
                             else "not twins: the symptom lists differ enough that a targeted question can separate them."))
    return {"similarity": sim, "genes": [ga, gb]}


def symptom_lookup(data, cfg=None):
    tag("degree and hubs; heavy-tailed degree distribution; information content",
        "network_tools.popularity() (diseases per symptom) and data.ic (hpo_data.py)")
    text = ask("symptom name (part of a name is fine)", "seizure")
    hits = data.find_term(text, limit=500)
    if not hits:
        print("   No symptom found.")
        return None
    pop = nt.popularity(data)
    hits = sorted(hits, key=lambda h: -pop[h])[:10]
    print(f"\n  {'symptom':44s} {'diseases':>9s} {'information content':>20s}")
    for h in hits:
        print(f"  {data.term_names[h][:44]:44s} {int(pop[h]):>9d} {data.ic[h]:>14.2f}")
    print("\n  What this shows: a symptom that many diseases have (a hub) has low information content; a rare one has high information content.")
    return {"n_found": len(hits)}


def compare_algorithms(data, cfg=None):
    tag("similarity (projection) ranking vs DIFFUSION on the bipartite network: random walk with restart (personalised PageRank)",
        "network_tools.rwr_scores() and network_tools.jaccard_scores() vs data.Dw @ query (step9_algorithms.py benchmarks them)")
    print("  Type the findings as symptom names separated by commas (part of a name is fine).")
    raw = input("  findings: ")
    cols = []
    for part in [p for p in raw.split(",") if p.strip()]:
        c = resolve_symptom(data, part)
        if c is not None and c not in cols:
            cols.append(c)
    if not cols:
        print("  No symptoms recognised.")
        return None
    Qn, _ = query_vector(data, cols)
    Qb = Qn.copy()
    Qb.data[:] = 1.0
    cfg = cfg or Settings()
    scores = {"IC-weighted cosine (ours)": (Qn @ data.Dw.T).toarray().ravel(),
              "Jaccard overlap": nt.jaccard_scores(data, Qb).ravel(),
              "random walk with restart": nt.rwr_scores(data, Qb, cfg.rwr_continue, cfg.rwr_iters).toarray().ravel()}
    tops = {}
    for name, sc in scores.items():
        order = np.argsort(-sc)[:5]
        tops[name] = [data.disease_names[j] for j in order]
        print(f"\n  {name}:")
        for r, j in enumerate(order, 1):
            print(f"    {r}. {sc[j]:.3f}   {data.disease_names[j]}")
    sets = [set(v) for v in tops.values()]
    print(f"\n  Diseases that appear in the top 5 of all three algorithms: {len(set.intersection(*sets))}")
    print("  What this shows: three different algorithms rank the same findings. In the step 9 benchmark IC-weighted cosine finds the true disease"
          " best (84% in the top 10 with 25% of symptoms known, vs 70% Jaccard and 52% random walk).")
    return {"top": tops, "agree": len(set.intersection(*sets))}


def twenty_questions(data, cfg=None):
    tag("greedy test-cover / identifying-code problem on the bipartite network: ask the symptom that splits the candidates best (binary search)",
        "network_tools.best_split_symptom() and adaptive_questions() (step10_questions.py benchmarks them)")
    i = pick_disease(data, ask("disease the system must find (part of a name; the answers come from the data)", "Joubert syndrome 6"))
    if i is None:
        print("   No disease found with that text.")
        return None
    manual = ask("who answers the questions? Enter = the system (truthfully, from the data), y = you type yes/no", "", str).strip().lower() == "y"
    Dp = data.Dprop.tocsr()
    pop = nt.popularity(data)
    cand = np.arange(data.n)
    n_q, stuck = 0, False
    print(f"\n  The system must find one of {data.n} diseases by asking yes/no questions about symptoms.")
    while len(cand) > 1 and n_q < 60:
        s = nt.best_split_symptom(Dp, cand, "greedy")
        if s is None:
            stuck = True
            break
        if manual:
            raw = input(f"  Q{n_q + 1}: does it have '{data.term_names[s]}' (in {int(pop[s])} diseases)? y / n, Enter = stop: ").strip().lower()
            if raw == "":
                break
            ans = raw == "y"
        else:
            ans = bool(Dp[i, s] > 0)
        has = np.asarray(Dp[cand][:, s].todense()).ravel() > 0
        cand = cand[has == ans]
        n_q += 1
        if not manual:
            print(f"  Q{n_q:<2d} Does it have '{data.term_names[s]}'?  (in {int(pop[s])} of {data.n} diseases)  -> {'yes' if ans else 'no':3s}   diseases left: {len(cand)}")
        else:
            print(f"        diseases left: {len(cand)}")
        if len(cand) == 0:
            break
    identified = len(cand) == 1
    if identified:
        print(f"\n  FOUND: {data.disease_names[int(cand[0])]} after {n_q} questions (binary-search limit for {data.n} diseases: {np.log2(data.n):.1f}).")
    elif len(cand) == 0:
        print("\n  No disease matches all of your answers (real descriptions are incomplete, so strict yes/no matching can be too harsh).")
    elif stuck:
        print(f"\n  STOPPED with {len(cand)} diseases that have IDENTICAL symptom sets: no question can separate them. These are exact twins; a gene test is needed.")
        for j in cand[:5]:
            print(f"    - {data.disease_names[int(j)]}   genes: {', '.join(sorted(data.genes.get(data.ids[int(j)], []))) or 'none listed'}")
    else:
        print(f"\n  STOPPED after {n_q} questions with {len(cand)} diseases still possible.")
    print("  What this shows: the best question is not a hub and not a rare symptom but one that splits the candidates about half and half (binary search).")
    return {"n_questions": n_q, "left": int(len(cand)), "identified": identified, "stuck": stuck, "target": data.disease_names[i]}


def noisy_questions(data, cfg=None):
    tag("noise-robust adaptive questioning: a probability for every disease of the bipartite network, updated by Bayes' rule, and the question with the largest information gain",
        "network_tools.robust_choose() and robust_update() (step11_robust_questions.py benchmarks them)")
    i = pick_disease(data, ask("disease the patient has (part of a name)", "Joubert syndrome 6"))
    if i is None:
        print("   No disease found with that text.")
        return None
    miss = min(max(ask("how often the patient FAILS to report a symptom the disease has (0 to 0.5)", 0.2, float), 0.0), 0.5)
    manual = ask("who answers? Enter = the simulated patient (with mistakes), y = you", "", str).strip().lower() == "y"
    eps, dl = miss, (0.02 if miss > 0 else 0.0)
    me, md = max(eps, 0.01), max(dl, 0.002)                 # the model always allows a small error rate
    Dp = data.Dprop.tocsr()
    Dc = Dp.tocsc()
    DpT = Dp.T.tocsr().astype(np.float32)
    pop = nt.popularity(data)
    rng = np.random.default_rng(SEED)
    w = np.full(data.n, 1.0 / data.n, dtype=np.float32)
    asked = np.zeros(data.t, dtype=bool)
    n_q = mistakes = 0
    print(f"\n  The system keeps a probability for each of the {data.n} diseases and asks about one symptom at a time.")
    for _ in range(30):
        s = nt.robust_choose(DpT, w, asked, me, md, "ig")
        asked[s] = True
        truth = bool(Dp[i, s] > 0)
        flag = ""
        if manual:
            raw = input(f"  Q{n_q + 1}: does the patient have '{data.term_names[s]}' (in {int(pop[s])} diseases)? y / n, Enter = stop: ").strip().lower()
            if raw == "":
                break
            ans = raw == "y"
        else:
            ans = bool(rng.random() > eps) if truth else bool(rng.random() < dl)
            if ans != truth:
                mistakes += 1
                flag = f"   <-- MISTAKE: the disease {'HAS' if truth else 'LACKS'} this symptom"
        w = nt.robust_update(w, Dc.getcol(s).toarray().ravel(), ans, me, md)
        n_q += 1
        top = np.argsort(-w)[:3]
        print(f"  Q{n_q:<2d} Does it have '{data.term_names[s]}'?  (in {int(pop[s])} diseases) -> patient says {'yes' if ans else 'no':3s}{flag}")
        print("       most probable: " + "; ".join(f"{data.disease_names[int(j)][:30]} {w[j]:.2f}" + (" (TRUE)" if (not manual and int(j) == i) else "") for j in top))
        if w[top[0]] >= 0.95:
            break
    top1 = int(np.argmax(w))
    true_rank = int((w >= w[i]).sum())
    print(f"\n  RESULT after {n_q} questions: the most probable disease is {data.disease_names[top1]} (probability {w[top1]:.2f}).")
    if not manual:
        print(f"  The patient made {mistakes} mistake(s). The true disease is {data.disease_names[i]}: " + ("CORRECT." if true_rank == 1 else f"ranked {true_rank}."))
    print("  What this shows: a wrong answer only lowers a disease's probability instead of removing it, so later answers can bring the true disease back.")
    return {"n_questions": n_q, "top": data.disease_names[top1], "top_p": float(w[top1]), "correct": (true_rank == 1) if not manual else None,
            "mistakes": mistakes, "true_rank": true_rank}


def which_module(data, cfg=None):
    tag("Barber's bipartite modularity and the BRIM algorithm: modules that group diseases AND symptoms together, each with signature symptoms",
        "step12_modules.get_modules() (BRIM) and network_tools.module_scores(); score = sum over your findings of the log-lift of the symptom in the module (how over-represented it is)")
    print("  Type the patient's findings as symptom names separated by commas (part of a name is fine).")
    raw = input("  findings: ")
    cols = []
    for part in [p for p in raw.split(",") if p.strip()]:
        c0 = resolve_symptom(data, part)
        if c0 is not None and c0 not in cols:
            cols.append(c0)
    if not cols:
        print("  No symptoms recognised.")
        return None
    cfg = cfg or Settings()
    if not (cfg.res_dir / "_cache_modules.npz").exists():
        print("  (computing the modules with BRIM: about 10 seconds, only the first time)")
    row, col, c, q = step12_modules.get_modules(data, cfg)
    A = data.Ddir.tocsr()
    scores = nt.module_scores(nt.module_profile(A, row, c), np.array(cols))
    info = {m["module"]: m for m in step12_modules.describe(data, A, row, col, c, top=c)}
    order = [int(m) for m in np.argsort(-scores)[:3]]
    print(f"\n  {c} syndromic modules (bipartite modularity {q:.2f}). The findings point to:")
    for r, m in enumerate(order, 1):
        mi = info[m]
        print(f"    {r}. module {m}  score {scores[m]:.2f}  ({mi['diseases']} diseases, {mi['symptoms']} symptoms; mostly {mi['main_organ']} {100 * mi['organ_share']:.0f}%)")
        print("       signature symptoms: " + "; ".join(mi["signature"][:6]))
    Qn, _ = query_vector(data, cols)
    sims = (data.Dw @ Qn.T).toarray().ravel()
    inside = np.flatnonzero(row == order[0])
    top = inside[np.argsort(-sims[inside])[:5]]
    print(f"\n  Best matching diseases INSIDE module {order[0]}:")
    for r, j in enumerate(top, 1):
        print(f"    {r}. {sims[j]:.2f}   {data.disease_names[int(j)]}")
    g = int(np.argmax(sims))
    print(f"\n  Best match over ALL diseases: {data.disease_names[g]} ({sims[g]:.2f}), which lies in module {int(row[g])}.")
    print("  What this shows: BRIM groups diseases and symptoms together, so a module has signature symptoms and gives a first triage step before ranking single diseases.")
    return {"module": order[0], "n_modules": c, "signature": info[order[0]]["signature"], "top_diseases": [data.disease_names[int(j)] for j in top],
            "global_top": data.disease_names[g], "global_top_module": int(row[g])}


MENU = """
================  LIVE TEST  (you type, the system answers)  ================
  1  Diagnose from findings YOU type
  2  Stress test a disease YOU type (common vs specific symptoms)
  3  Twin check: two diseases YOU type
  4  Look up a symptom (hub or rare?)
  5  Compare 3 ranking algorithms on findings YOU type
  6  Play 20 questions (the system finds a disease)
  7  Noisy 20 questions (the patient makes mistakes)
  8  Which syndromic module? (BRIM)
  0  Quit
"""
ACTIONS = {"1": diagnose, "2": stress_test, "3": twin_check, "4": symptom_lookup, "5": compare_algorithms, "6": twenty_questions, "7": noisy_questions, "8": which_module}


def main():
    cfg = Settings()
    data = hpo_data.load(cfg)
    print(f"Loaded {data.n} rare diseases and {data.t} symptom terms.")
    while True:
        print(MENU)
        try:
            choice = input("Choose a number: ").strip()
        except EOFError:
            break
        if choice in ("0", "q", "Q"):
            break
        fn = ACTIONS.get(choice)
        if fn is None:
            if choice:
                print("  Please type a number from the menu.")
            continue
        try:
            fn(data, cfg)
        except EOFError:
            break
    print("Bye.")


if __name__ == "__main__":
    main()
