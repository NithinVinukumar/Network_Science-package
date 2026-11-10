"""
step4_percolation.py - STEP 4: when do look-alike diseases join into one giant cluster? (percolation)
=======================================================================================================
Concepts: percolation / phase transition, giant connected component, null model (configuration model of the bipartite network),
          critical threshold.

Plain meaning
  * Draw a link between two diseases if their similarity is at least a threshold. A HIGH threshold gives few links and many small groups.
    Lowering the threshold adds links; at some point the groups suddenly merge into one giant cluster: a phase transition.
  * Null model: a random disease-symptom network with exactly the same number of symptoms per disease and diseases per symptom.
    If the real network percolates EARLIER than the null, real disease profiles overlap in structured, clustered ways, beyond what the
    popularity of symptoms alone would produce.
  * Critical threshold: the threshold where the second-largest cluster is biggest (the moment of maximum competition between clusters).
"""
import matplotlib.pyplot as plt
import numpy as np

import network_tools as nt
from utils import explain, get_rng, savefig, write_json


def crossing(ths, giant, level=0.5):
    """Threshold at which the giant component first exceeds `level` (linear interpolation, thresholds descending)."""
    for a, b, ga, gb in zip(ths[:-1], ths[1:], giant[:-1], giant[1:]):
        if ga < level <= gb:
            return float(a + (level - ga) * (b - a) / (gb - ga))
    return None


def run(data, cfg):
    print("[Step 4] Percolation")
    rng = get_rng(cfg.seed + 4)
    ths = sorted(cfg.perc_thresholds, reverse=True)
    lo = min(ths)
    I, J, S = nt.similarity_edges(data.Dw, lo)
    real = nt.giant_stats(I, J, S, data.n, ths)
    nulls = []
    for k in range(cfg.n_null):
        Dn = nt.bipartite_null(data, rng)
        In, Jn, Sn = nt.similarity_edges(Dn, lo)
        nulls.append(nt.giant_stats(In, Jn, Sn, data.n, ths))
        print(f"    null {k + 1}/{cfg.n_null} done")
    g_real = np.array([r["giant"] for r in real])
    g_null = np.array([[r["giant"] for r in n] for n in nulls])
    s_real = np.array([r["second"] for r in real])
    s_null = np.array([[r["second"] for r in n] for n in nulls])
    iso_real = np.array([r["isolated"] for r in real])
    iso_null = np.array([[r["isolated"] for r in n] for n in nulls])
    crit_real = float(ths[int(np.argmax(s_real))])
    crit_null = float(ths[int(np.argmax(s_null.mean(0)))])
    half_real, half_null = crossing(ths, g_real), crossing(ths, g_null.mean(0))

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].plot(ths, 100 * g_real, "-o", color="#E45756", label="real network")
    ax[0].plot(ths, 100 * g_null.mean(0), "-o", color="#7F7F7F", label=f"random null (mean of {cfg.n_null})")
    ax[0].fill_between(ths, 100 * (g_null.mean(0) - g_null.std(0)), 100 * (g_null.mean(0) + g_null.std(0)), color="#7F7F7F", alpha=0.25)
    ax[0].set(xlabel="similarity threshold for a link", ylabel="% of diseases in the giant cluster", title="Percolation of look-alike diseases")
    ax[0].invert_xaxis()
    ax[0].legend(fontsize=8)
    ax[1].plot(ths, 100 * s_real, "-o", color="#E45756")
    ax[1].plot(ths, 100 * s_null.mean(0), "-o", color="#7F7F7F")
    ax[1].set(xlabel="similarity threshold", ylabel="% in the SECOND-largest cluster", title=f"Critical threshold (peak): real {crit_real}, null {crit_null}")
    ax[1].invert_xaxis()
    ax[2].plot(ths, 100 * iso_real, "-o", color="#E45756", label="real")
    ax[2].plot(ths, 100 * iso_null.mean(0), "-o", color="#7F7F7F", label="null")
    ax[2].set(xlabel="similarity threshold", ylabel="% of diseases with no look-alike at all", title="Isolated diseases")
    ax[2].invert_xaxis()
    ax[2].legend(fontsize=8)
    savefig(fig, cfg.fig_dir / "4_percolation.png")

    res = {"thresholds": ths, "real": real, "null_mean_giant": g_null.mean(0), "null_sd_giant": g_null.std(0),
           "null_mean_second": s_null.mean(0), "null_mean_isolated": iso_null.mean(0), "n_null": cfg.n_null,
           "critical_threshold_real": crit_real, "critical_threshold_null": crit_null,
           "giant_half_threshold_real": half_real, "giant_half_threshold_null": half_null}
    write_json(cfg.res_dir / "step4.json", res)
    i5 = ths.index(0.5) if 0.5 in ths else 0
    print(f"    giant cluster reaches 50% at similarity {half_real:.2f} (real) vs {half_null:.2f} (null)")
    explain([
        "Link two diseases when their similarity is at least a threshold, and lower the threshold step by step. At a high threshold diseases form small isolated groups; "
        "then, quite suddenly, the groups merge into one giant cluster. That sudden merging is a percolation phase transition.",
        f"In the real network the giant cluster holds {100 * g_real[i5]:.0f}% of diseases at threshold {ths[i5]}, against {100 * g_null.mean(0)[i5]:.0f}% (+/- {100 * g_null.std(0)[i5]:.0f}) in random networks "
        f"with the same symptom counts. It reaches half of all diseases at about {half_real:.2f} versus {half_null:.2f} for the null." if half_real and half_null else
        f"The giant cluster is {100 * g_real[i5]:.0f}% (real) vs {100 * g_null.mean(0)[i5]:.0f}% (null) at {ths[i5]}.",
        "So real disease profiles overlap in clustered, structured ways that popularity of symptoms alone cannot explain: related diseases share rare symptoms.",
    ])
    return res
