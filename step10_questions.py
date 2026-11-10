"""
step10_questions.py - STEP 10: which symptoms should the doctor ask, and in what order? (adaptive questioning)
=============================================================================================================
Concepts: greedy test-cover / identifying-code problem on the bipartite network, neighbourhoods of symptom nodes, degree (hubs vs
          balanced questions), information theory (a yes/no question gives most information when about half of the candidates say yes),
          exact twins (identical neighbourhoods).

The game ("20 questions"): a disease is chosen. The doctor may ask yes/no questions of the form "does it have symptom X?". The answer is
read from the disease's symptom list. After every answer only the diseases consistent with ALL answers stay possible. How many questions
until ONE disease is left?
Strategies for choosing the next question:
  greedy  the symptom that splits the remaining candidates best (about half yes, half no) = binary search on the network
  hubs    always the most common useful symptom      rare   always the rarest useful symptom      random   any useful symptom
Diseases with an IDENTICAL symptom set can never be separated by any question: exact twins (a gene test is needed).
Idealisation: answers are perfect and complete (real patients and real annotations are not).
"""
import matplotlib.pyplot as plt
import numpy as np

import network_tools as nt
from utils import explain, get_rng, savefig, write_json

STRATS = ["greedy", "hubs", "rare", "random"]
LABEL = {"greedy": "greedy best split (ours)", "hubs": "most common symptom first", "rare": "rarest symptom first", "random": "random symptom"}


def run(data, cfg):
    print("[Step 10] Adaptive questioning")
    rng = get_rng(cfg.seed + 10)
    N = data.n
    Dp = data.Dprop.tocsr()
    Dc = Dp.tocsc()
    pop = nt.popularity(data)
    # exact twins: diseases whose full symptom set is identical to another disease's
    keys = [Dp.indices[Dp.indptr[i]:Dp.indptr[i + 1]].tobytes() for i in range(N)]
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    exact = int(sum(1 for k in keys if counts[k] > 1))
    res = {"diseases": N, "exact_twin_diseases": exact, "exact_twin_share": exact / N, "distinct_symptom_sets": len(counts),
           "min_questions_binary_search": float(np.log2(N))}
    cap_curve = 25
    curves, summary = {}, {}
    greedy_rows = {}
    for strat in STRATS:
        k = cfg.q_targets if strat == "greedy" else cfg.q_baseline_targets
        cap = cfg.q_cap if strat == "greedy" else cfg.q_cap_baseline
        targets = rng.choice(N, min(k, N), replace=False)
        out = [nt.adaptive_questions(data, int(t), strat, rng, cap, Dp, Dc) for t in targets]
        left_by_q = np.array([[o["steps"][min(q, len(o["steps"])) - 1][2] if q > 0 and o["steps"] else (N if q == 0 else o["left"])
                               if q <= len(o["steps"]) else o["left"] for q in range(cap_curve + 1)] for o in out])
        curves[strat] = np.median(left_by_q, axis=0)
        nq = np.array([o["n_questions"] for o in out])
        ident = np.array([o["identified"] for o in out])
        stuck = np.array([o["stuck"] for o in out])
        summary[strat] = {"targets": int(len(out)), "mean_questions": float(nq.mean()), "median_questions": float(np.median(nq)),
                          "identified_share": float(ident.mean()), "stuck_share": float(stuck.mean()), "cap": cap,
                          "median_left_after_10": float(curves[strat][10]), "median_left_after_25": float(curves[strat][25]),
                          "identified_within_10": float((ident & (nq <= 10)).mean()), "identified_within_15": float((ident & (nq <= 15)).mean()),
                          "identified_within_20": float((ident & (nq <= 20)).mean())}
        if strat == "greedy":
            greedy_rows = out
        print(f"    {LABEL[strat]:28s} mean {nq.mean():5.1f} questions | identified {100 * ident.mean():3.0f}% within {cap} | median candidates left after 10 questions: {curves[strat][10]:.0f}")
    g = greedy_rows
    nq_g = np.array([o["n_questions"] for o in g if o["identified"]])
    first_share = np.array([[pop[o["steps"][q][0]] / N for o in g if len(o["steps"]) > q] for q in range(5)], dtype=object)
    res["strategies"] = summary
    res["greedy_questions_histogram"] = np.bincount(nq_g, minlength=1).tolist()
    res["greedy_first_five_questions_share_of_diseases"] = [float(np.mean(x)) for x in first_share]
    i = data.find_disease("Joubert syndrome 6")[0]
    ex = nt.adaptive_questions(data, int(i), "greedy", rng, cfg.q_cap, Dp, Dc)
    res["example"] = {"disease": data.disease_names[i], "n_questions": ex["n_questions"], "candidates_left": [s[2] for s in ex["steps"]],
                      "questions": [(data.term_names[s], a) for s, a, _ in ex["steps"][:6]]}
    write_json(cfg.res_dir / "step10.json", res)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    cols = {"greedy": "#E45756", "hubs": "#4C78A8", "rare": "#54A24B", "random": "#7F7F7F"}
    for s in STRATS:
        ax[0].plot(range(cap_curve + 1), np.maximum(curves[s], 1), "-o", ms=3, color=cols[s], label=LABEL[s])
    ax[0].set(yscale="log", xlabel="questions asked", ylabel="diseases still possible (median, log scale)", title="The greedy question halves the candidates each time")
    ax[0].legend(fontsize=7)
    ax[1].hist(nq_g, bins=np.arange(0, nq_g.max() + 2) - 0.5, color="#E45756")
    ax[1].axvline(np.log2(N), color="k", ls="--", label=f"binary-search limit log2({N}) = {np.log2(N):.1f}")
    ax[1].set(xlabel="questions needed to find the disease (greedy)", ylabel="diseases", title=f"Mean {nq_g.mean():.1f} questions")
    ax[1].legend(fontsize=8)
    ax[2].bar([LABEL[s].split(" (")[0] for s in STRATS], [100 * summary[s]["identified_share"] for s in STRATS], color=[cols[s] for s in STRATS])
    ax[2].set(ylabel=f"% of diseases identified within the question cap", title="Found uniquely")
    ax[2].tick_params(axis="x", labelrotation=20, labelsize=8)
    savefig(fig, cfg.fig_dir / "10_questions.png")

    gs = summary["greedy"]
    explain([
        f"'20 questions' on the disease-symptom network: ask yes/no questions about symptoms and keep only the diseases consistent with the answers. The greedy strategy always asks "
        f"the symptom that splits the remaining candidates best. It identifies a disease uniquely in {gs['mean_questions']:.1f} questions on average (limit for binary search: {np.log2(N):.1f}); "
        f"{100 * gs['identified_share']:.0f}% of diseases are found within {gs['cap']} questions.",
        f"The best questions are not the hubs and not the rarest symptoms but BALANCED ones: the first five questions are symptoms that about "
        f"{100 * np.mean(res['greedy_first_five_questions_share_of_diseases']):.0f}% of diseases have. Always asking the most common symptom needs {summary['hubs']['mean_questions']:.1f} questions "
        f"(and finds {100 * summary['hubs']['identified_share']:.0f}% within {summary['hubs']['cap']}); asking the rarest or random symptoms almost never finds the disease in {summary['rare']['cap']} questions.",
        f"Exact twins: {exact} diseases ({100 * exact / N:.1f}%) have a symptom set identical to another disease's, so NO question can separate them; the remaining diseases can all be told apart in principle.",
        "Idealised: answers are assumed perfect and complete, which real patients and real annotations are not.",
    ])
    return res
