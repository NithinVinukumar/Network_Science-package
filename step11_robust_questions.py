"""
step11_robust_questions.py - STEP 11: what if the answers are WRONG? (noise-robust adaptive questioning)
========================================================================================================
Concepts: Bayesian posterior over the diseases of the bipartite network, expected information gain (entropy), robustness to noise,
          hard elimination vs soft probabilities, hubs vs balanced questions.

Weakness of step 10: it crosses a disease out as soon as one answer disagrees with it. Real patients forget or misreport symptoms
(and real annotations are incomplete), so ONE wrong answer removes the true disease for good.
Proposed method: keep a probability for EVERY disease. After each answer multiply by how likely that answer is for each disease
(a disease that has the symptom is answered 'yes' with probability 1 - eps, one that lacks it with probability delta), and ask next the
symptom with the largest expected information gain. A wrong answer only lowers the true disease; later answers can bring it back.

Simulated patient: answers 'no' to a symptom the disease HAS with probability eps = 20% (missed) and 'yes' to one it lacks with
probability delta = 2% (invented). Success = the true disease has the highest probability (ties count AGAINST the method).
Compared: hard elimination (step 10), robust information gain (proposed), robust 'most common symptom first', robust 'random symptom'.
"""
import matplotlib.pyplot as plt
import numpy as np

import network_tools as nt
from utils import explain, get_rng, savefig, write_json

NAMES = {"hard": "hard elimination (step 10)", "soft_ig": "robust information gain (proposed)",
         "soft_hubs": "robust, most common symptom first", "soft_random": "robust, random symptom"}


def run(data, cfg):
    print("[Step 11] Noise-robust adaptive questioning")
    rng = get_rng(cfg.seed + 11)
    N, T, Q = data.n, data.t, cfg.rq_questions
    Dp = data.Dprop.tocsr()
    Dc = Dp.tocsc()
    DpT = Dp.T.tocsr().astype(np.float32)
    popv = np.asarray(Dp.sum(0)).ravel().astype(float)
    targets = rng.choice(N, min(cfg.rq_targets, N), replace=False)

    def answer(t, s, eps, dl):
        return (rng.random() > eps) if Dp[t, s] > 0 else (rng.random() < dl)

    def run_hard(t, eps, dl):
        cand = np.arange(N)
        ranks = np.zeros(Q, dtype=np.int64)
        for q in range(Q):
            if len(cand) > 1:
                s = nt.best_split_symptom(Dp, cand, "greedy")
                if s is not None:
                    has = Dc.getcol(s).toarray().ravel()[cand] > 0
                    cand = cand[has == answer(t, s, eps, dl)]
            ranks[q] = len(cand) if t in cand else 10 ** 9
        return ranks

    def run_soft(t, strategy, eps, dl, me, md):
        w = np.full(N, 1.0 / N, dtype=np.float32)
        asked = np.zeros(T, dtype=bool)
        ranks = np.zeros(Q, dtype=np.int64)
        for q in range(Q):
            s = nt.robust_choose(DpT, w, asked, me, md, strategy, rng, popv)
            asked[s] = True
            w = nt.robust_update(w, Dc.getcol(s).toarray().ravel(), answer(t, s, eps, dl), me, md)
            ranks[q] = int((w >= w[t]).sum())              # pessimistic: ties count against us
        return ranks

    res = {"targets": int(len(targets)), "questions": Q, "miss": cfg.rq_miss, "invent": cfg.rq_invent, "conditions": {}}
    curves = {}
    for cname, (eps, dl) in {"perfect answers": (0.0, 0.0), "noisy answers": (cfg.rq_miss, cfg.rq_invent)}.items():
        me, md = max(eps, 0.01), max(dl, 0.002)            # the model always allows a small error rate
        res["conditions"][cname] = {}
        curves[cname] = {}
        for method in NAMES:
            R = np.array([run_hard(int(t), eps, dl) if method == "hard" else run_soft(int(t), method.split("_")[1], eps, dl, me, md) for t in targets])
            top1, top5 = (R <= 1).mean(0), (R <= 5).mean(0)
            curves[cname][method] = top1
            res["conditions"][cname][method] = {"top1_curve": top1.tolist(),
                                                "top1_at": {str(q): float(top1[q - 1]) for q in (10, 15, 20, 25, Q) if q <= Q},
                                                "top5_at": {str(q): float(top5[q - 1]) for q in (10, 15, 20, 25, Q) if q <= Q}}
        print(f"    {cname:16s} top-1 after {Q} questions: " + "  ".join(f"{NAMES[m].split(' (')[0]} {100 * curves[cname][m][-1]:.0f}%" for m in NAMES))
    write_json(cfg.res_dir / "step11.json", res)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    cols = {"hard": "#4C78A8", "soft_ig": "#E45756", "soft_hubs": "#54A24B", "soft_random": "#7F7F7F"}
    for k, cname in enumerate(curves):
        for m in NAMES:
            ax[k].plot(range(1, Q + 1), 100 * curves[cname][m], "-", color=cols[m], lw=2, label=NAMES[m])
        ax[k].set(xlabel="questions asked", ylabel="% of diseases ranked first (true disease)", title=cname, ylim=(-2, 102))
    ax[0].legend(fontsize=7, loc="upper left")
    savefig(fig, cfg.fig_dir / "11_robust_questions.png")

    n, p = res["conditions"]["noisy answers"], res["conditions"]["perfect answers"]
    explain([
        f"Step 10 crosses a disease out as soon as an answer disagrees with it. With perfect answers that works (hard elimination ranks the true disease first for "
        f"{100 * p['hard']['top1_at'][str(Q)]:.0f}% of diseases after {Q} questions), but real answers have errors.",
        f"When the simulated patient misses {100 * cfg.rq_miss:.0f}% of the symptoms the disease has and invents {100 * cfg.rq_invent:.0f}% of the others, hard elimination collapses to "
        f"{100 * n['hard']['top1_at'][str(Q)]:.0f}%, because one wrong answer removes the true disease for good.",
        f"The proposed method keeps a probability for every disease and asks the question with the largest expected information gain. With the same noisy answers it ranks the true disease "
        f"first for {100 * n['soft_ig']['top1_at'][str(Q)]:.0f}% of diseases (top 5: {100 * n['soft_ig']['top5_at'][str(Q)]:.0f}%) after {Q} questions, against "
        f"{100 * n['soft_hubs']['top1_at'][str(Q)]:.0f}% when it always asks the most common symptom and {100 * n['soft_random']['top1_at'][str(Q)]:.0f}% for random questions.",
        "Honest limits: the model is told the true error rates (a mismatch would cost accuracy), diseases with identical symptoms tie and count as failures, and the noise is simulated.",
    ])
    return res
