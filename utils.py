"""
utils.py - small helpers shared by every step
=============================================
get_rng        random generator from a seed
savefig        save a figure safely (creates folders, ignores harmless layout warnings)
wilson_ci      95% confidence interval for a proportion (e.g. "85% of diseases found in the top 10")
auc_score      area under the ROC curve (0.5 = no better than chance, 1.0 = perfect), via rank sums
ccdf           complementary cumulative distribution: share of items with value >= x
tail_exponent  maximum-likelihood power-law exponent of the tail k >= kmin
explain        print a plain-language "what this means" block
write_json     save a dictionary of results (numpy values converted)
"""
import json
import textwrap
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata


def get_rng(seed):
    return np.random.default_rng(seed)


def savefig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    saved {path.name}")


def wilson_ci(k, n, z=1.96):
    """95% Wilson confidence interval for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def auc_score(scores, labels):
    """AUC of `scores` for predicting binary `labels` (ties get average rank)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels).astype(bool)
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ccdf(values):
    v = np.sort(np.asarray(values, float))
    v = v[v > 0]
    x = np.unique(v)
    p = 1.0 - np.searchsorted(v, x, side="left") / len(v)
    return x, p


def tail_exponent(values, kmin):
    """gamma = 1 + n / sum ln(k / (kmin - 1/2)) for values >= kmin (Clauset et al. 2009). Returns (gamma, n)."""
    t = np.asarray(values, float)
    t = t[t >= kmin]
    return float(1.0 + len(t) / np.sum(np.log(t / (kmin - 0.5)))), int(len(t))


def explain(lines):
    print("    --- what this means ---")
    for ln in lines:
        for i, part in enumerate(textwrap.wrap(ln, 92) or [""]):
            print(("    * " if i == 0 else "      ") + part)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=1), encoding="utf-8")
