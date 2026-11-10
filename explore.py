"""
explore.py - try your own inputs, with plain-language answers (made for explaining to your professor)
=====================================================================================================
    python explore.py

   1  Look up a disease              its symptoms (most informative first), nearest look-alikes, genes
   2  Simulate a patient             hide symptoms of a real disease, add wrong ones, see where the true disease ranks
   3  Diagnose from YOUR symptoms    type symptom names; get a ranked list and the best NEXT question to ask
   4  Compare two diseases           similarity, shared and unique symptoms
   0  Quit
Press Enter to accept the value in [brackets]. Same inputs + same seed give the same output.
"""
import textwrap

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

import hpo_data
import network_tools as nt
from config import Settings

SEED = 1


def ask(prompt, default, cast=str):
    try:
        raw = input(f"  {prompt} [{default}]: ").strip()
    except EOFError:
        raw = ""
    if raw == "":
        return default
    try:
        return cast(raw)
    except ValueError:
        print(f"    (not understood, using {default})")
        return default


def say(lines):
    print("\n  What this means:")
    for ln in lines:
        for i, part in enumerate(textwrap.wrap(ln, 88)):
            print(("   * " if i == 0 else "     ") + part)
    print()


def pick_disease(data, text):
    """Best match for a disease name: exact name first, otherwise the first partial match."""
    hits = data.find_disease(text, limit=50)
    if not hits:
        return None
    exact = [i for i in hits if data.disease_names[i].lower() == text.lower()]
    return (exact or hits)[0]


def query_vector(data, term_cols):
    """IC-weighted, length-1 query vector for a list of symptom columns (ancestors added)."""
    q = csr_matrix((np.ones(len(term_cols)), (np.zeros(len(term_cols), dtype=int), list(term_cols))), shape=(1, data.t)) @ data.Anc
    q.data[:] = 1.0
    return normalize(q.multiply(data.ic).tocsr()).astype(np.float32), np.asarray(q.todense()).ravel() > 0


def opt_lookup(data, cfg):
    text = ask("disease name (part of the name is fine)", "Joubert syndrome 1")
    i = pick_disease(data, text)
    if i is None:
        print("   No disease found with that text.")
        return
    dt = data.direct_terms(i)
    order = dt[np.argsort(-data.ic[dt])]
    sims = (data.Dw @ data.Dw[i].T).toarray().ravel()
    sims[i] = -1
    top = np.argsort(-sims)[:8]
    print(f"\n  {data.disease_names[i]}   [{data.ids[i]}]   {len(dt)} listed symptoms")
    print("  most informative symptoms first (information content in brackets):")
    for j in order[:12]:
        print(f"     - {data.term_names[j]}  ({data.ic[j]:.1f})")
    if len(order) > 12:
        print(f"     ... and {len(order) - 12} more")
    print("  nearest look-alikes:")
    for j in top:
        print(f"     {sims[j]:.2f}  {data.disease_names[j]}  [{data.ids[j]}]")
    g = sorted(data.genes.get(data.ids[i], []))
    print("  genes:", ", ".join(g) if g else "none listed")
    say([f"The look-alike list shows which diseases could be confused with this one when only some symptoms are known; a similarity near 1 means almost the same symptom profile.",
         "Rare symptoms (high information content) are the best clues; common ones such as developmental delay appear in thousands of diseases."])


def opt_simulate(data, cfg):
    text = ask("disease name", "Joubert syndrome 1")
    i = pick_disease(data, text)
    if i is None:
        print("   No disease found.")
        return
    keep = ask("fraction of its symptoms the doctor knows (0.05-1)", 0.25, float)
    mode = ask("which symptoms: random | common | specific", "random")
    if mode not in ("random", "common", "specific"):
        mode = "random"
    noise = ask("number of WRONG symptoms added", 2, int)
    rng = np.random.default_rng(SEED)
    Q = nt.make_patients(data, np.array([i]), keep, mode, noise, rng)
    sims = (data.Dw @ Q.T).toarray().ravel()
    rank = 1 + int((sims > sims[i]).sum())
    top = np.argsort(-sims)[:10]
    print(f"\n  True disease: {data.disease_names[i]}  ->  rank {rank} of {data.n}")
    for r, j in enumerate(top, 1):
        print(f"   {r:2d}. {sims[j]:.2f}  {data.disease_names[j]}" + ("   <-- the true disease" if j == i else ""))
    say([f"The doctor knew {keep:.0%} of the symptoms ({mode}) plus {noise} wrong ones. The true disease came out at rank {rank}.",
         "Try mode 'common' (only hub symptoms) and mode 'specific' (only rare symptoms) with the same fraction: rare symptoms give far better rankings."])


def opt_diagnose(data, cfg):
    raw = ask("your symptoms, separated by commas (e.g. molar tooth sign, ataxia, nystagmus)", "molar tooth sign, ataxia, nystagmus")
    cols, shown = [], []
    for part in [p.strip() for p in raw.split(",") if p.strip()]:
        hits = data.find_term(part, limit=50)
        exact = [h for h in hits if data.term_names[h].lower() == part.lower()]
        h = (exact or hits or [None])[0]
        if h is None:
            print(f"   (no symptom matches '{part}')")
            continue
        cols.append(h)
        shown.append(data.term_names[h])
    if not cols:
        print("   No symptoms recognised.")
        return
    print("  understood as:", "; ".join(shown))
    Qn, qbin = query_vector(data, cols)
    sims = (data.Dw @ Qn.T).toarray().ravel()
    top = np.argsort(-sims)[:10]
    print("\n  most likely diseases:")
    for r, j in enumerate(top, 1):
        print(f"   {r:2d}. {sims[j]:.2f}  {data.disease_names[j]}  [{data.ids[j]}]")
    Dp = data.Dprop[top].toarray()
    frac = Dp.mean(0)
    score = frac * (1 - frac) * data.ic
    score[qbin] = -1
    nxt = np.argsort(-score)[:5]
    print("\n  best symptoms to ask about NEXT (they split the current top 10 and are informative):")
    for j in nxt:
        print(f"     - {data.term_names[j]}  (present in {int(round(frac[j] * 10))} of the top 10)")
    say(["Diseases are ranked by information-weighted similarity to the symptoms you typed; rare symptoms count more than common ones.",
         "The 'next question' list picks symptoms that about half of the top candidates have, because asking about them separates the candidates best.",
         "This is a research demonstration on curated disease descriptions, not a diagnostic tool."])


def opt_compare(data, cfg):
    a = pick_disease(data, ask("first disease", "Joubert syndrome 1"))
    b = pick_disease(data, ask("second disease", "Joubert syndrome 6"))
    if a is None or b is None:
        print("   Disease not found.")
        return
    sim = float((data.Dw[a] @ data.Dw[b].T).toarray()[0, 0])
    A, B = set(data.direct_terms(a)), set(data.direct_terms(b))
    print(f"\n  {data.disease_names[a]}  vs  {data.disease_names[b]}\n  similarity {sim:.2f}; {len(A)} and {len(B)} listed symptoms; {len(A & B)} in common")
    for title, s in (("only in the first", A - B), ("only in the second", B - A), ("shared", A & B)):
        names = [data.term_names[j] for j in sorted(s, key=lambda j: -data.ic[j])[:6]]
        print(f"  {title}: " + ("; ".join(names) if names else "-"))
    say([f"Similarity {sim:.2f} (0 = nothing in common, 1 = identical). Diseases above about 0.8 are 'twins' that symptoms alone can hardly separate; "
         "the lists above show which symptoms still tell them apart."])


MENU = """
================  Rare-disease explorer  ================
  1  Look up a disease
  2  Simulate a patient (hide symptoms of a real disease)
  3  Diagnose from YOUR symptoms
  4  Compare two diseases
  0  Quit
"""
ACTIONS = {"1": opt_lookup, "2": opt_simulate, "3": opt_diagnose, "4": opt_compare}


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
        if choice == "":
            continue
        fn = ACTIONS.get(choice)
        if fn is None:
            print("  Please type a number from the menu.")
            continue
        fn(data, cfg)
    print("Bye.")


if __name__ == "__main__":
    main()
