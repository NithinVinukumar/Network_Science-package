"""
network_tools.py - the reusable network operations behind the steps
===================================================================
similarity search      top_neighbours, similarity_edges
adaptive questioning   best_split_symptom, adaptive_questions   (greedy test-cover / identifying-code algorithm on the bipartite network)
bipartite modules      bipartite_modularity, brim, symptom_modules_for, module_coverage, module_profile, module_scores   (Barber's bipartite modularity and the BRIM algorithm)
robust questioning     robust_choose, robust_update   (Bayesian posterior over diseases + information-gain questions; tolerates wrong answers)
ranking algorithms     rwr_scores (random walk with restart = personalised PageRank), jaccard_scores
null model             bipartite_null  (random disease-symptom network with the SAME number of symptoms per disease and
                       the same number of diseases per symptom: "stub matching" configuration model)
patient simulation     make_patients, rank_of_true   (hide symptoms, add wrong ones, then rank all diseases)
percolation            giant_stats   (size of the largest connected group of look-alike diseases at each threshold)
bridging               participation_coefficient
nestedness             nodf, curveball
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import normalize

from hpo_data import propagate_matrices


# ----------------------------------------------------------------------------- similarity
def top_neighbours(data, k, block=900):
    """For every disease: its k most similar OTHER diseases. Returns (neighbour index [N,k], similarity [N,k],
    best similarity [N], number of diseases with similarity >= 0.5 [N])."""
    N = data.n
    nb, sm = np.zeros((N, k), dtype=np.int32), np.zeros((N, k), dtype=np.float32)
    best, dens = np.zeros(N, dtype=np.float32), np.zeros(N, dtype=np.int32)
    for s in range(0, N, block):
        rows = np.arange(s, min(s + block, N))
        C = (data.Dw @ data.Dw[rows].T).toarray()
        C[rows, np.arange(len(rows))] = -1.0                                   # exclude self
        idx = np.argpartition(-C, k, axis=0)[:k]                               # k best per column (unordered)
        vals = np.take_along_axis(C, idx, axis=0)
        order = np.argsort(-vals, axis=0)
        nb[rows] = np.take_along_axis(idx, order, axis=0).T
        sm[rows] = np.take_along_axis(vals, order, axis=0).T
        best[rows] = sm[rows, 0]
        dens[rows] = (C >= 0.5).sum(0)
    return nb, sm, best, dens


def similarity_edges(Dw, threshold, block=900):
    """All disease pairs (i < j) with similarity >= threshold. Returns arrays I, J, S."""
    N = Dw.shape[0]
    I, J, S = [], [], []
    for s in range(0, N, block):
        C = (Dw @ Dw[s:s + block].T).toarray()
        rr, cc = np.where(C >= threshold)
        keep = rr > (s + cc)
        I.append(rr[keep])
        J.append(s + cc[keep])
        S.append(C[rr[keep], cc[keep]])
    return np.concatenate(I), np.concatenate(J), np.concatenate(S)


def giant_stats(I, J, S, N, thresholds):
    """For each threshold: number of links, giant-component share, isolated share, size of the second-largest component."""
    out = []
    for th in thresholds:
        m = S >= th
        G = csr_matrix((np.ones(int(m.sum())), (I[m], J[m])), shape=(N, N))
        _, lab = connected_components(G, directed=False)
        sz = np.sort(np.bincount(lab))[::-1]
        out.append({"threshold": th, "links": int(m.sum()), "giant": sz[0] / N,
                    "second": (sz[1] / N) if len(sz) > 1 else 0.0, "isolated": float((sz == 1).sum() / N)})
    return out


# ----------------------------------------------------------------------------- null model
def bipartite_null(data, rng):
    """Random disease-symptom network keeping every disease's number of direct symptoms and every symptom's number of
    diseases (stub matching). Returns (Dw_null) built the same way as the real network."""
    rows, cols = data.Ddir.nonzero()
    M = csr_matrix((np.ones(len(rows)), (rows, rng.permutation(cols))), shape=data.Ddir.shape)
    M.data[:] = 1.0
    _, _, Dw = propagate_matrices(M, data.Anc, data.n)
    return Dw


# ----------------------------------------------------------------------------- simulated patients
def popularity(data):
    """Number of diseases that have each symptom (after propagation)."""
    return np.asarray(data.Dprop.sum(0)).ravel()


def make_patients(data, rows, keep, mode, noise, rng):
    """One simulated patient per disease in `rows`.
    keep   fraction of the disease's directly annotated symptoms the doctor knows (at least 2)
    mode   'random'   a random subset
           'common'   the most COMMON symptoms only (symptoms shared by many diseases)
           'specific' the most SPECIFIC symptoms only (symptoms of few diseases)
    noise  number of WRONG symptoms (random terms) added to every patient
    Returns the patients' IC-weighted, length-1 query vectors (len(rows) x T, sparse)."""
    pop = popularity(data)
    r, c = [], []
    for b, i in enumerate(rows):
        dt = data.direct_terms(i)
        k = min(len(dt), max(2, int(round(keep * len(dt)))))
        if mode == "random":
            sel = rng.choice(dt, k, replace=False)
        elif mode == "common":
            sel = dt[np.argsort(-pop[dt], kind="stable")[:k]]
        elif mode == "specific":
            sel = dt[np.argsort(pop[dt], kind="stable")[:k]]
        else:
            raise ValueError(mode)
        if noise:
            sel = np.concatenate([sel, rng.integers(0, data.t, noise)])
        r.extend([b] * len(sel))
        c.extend(sel)
    Q = csr_matrix((np.ones(len(r)), (r, c)), shape=(len(rows), data.t))
    Q.data[:] = 1.0
    Q = Q @ data.Anc                                  # add the general terms above each symptom
    Q.data[:] = 1.0
    return normalize(Q.multiply(data.ic).tocsr()).astype(np.float32)


def rank_of_true(data, Qn, rows, block=800):
    """Rank (1 = best) of the true disease among all diseases for each simulated patient."""
    ranks = np.zeros(len(rows), dtype=np.int64)
    rows = np.asarray(rows)
    for s in range(0, len(rows), block):
        C = (data.Dw @ Qn[s:s + block].T).toarray()                      # N x b
        true = C[rows[s:s + block], np.arange(C.shape[1])]
        ranks[s:s + block] = 1 + (C > true[None, :]).sum(0)
    return ranks


def hit_rates(ranks, ks=(1, 10, 50)):
    from utils import wilson_ci
    out = {}
    for k in ks:
        h = int((ranks <= k).sum())
        lo, hi = wilson_ci(h, len(ranks))
        out[f"top{k}"] = {"rate": h / len(ranks), "low": lo, "high": hi}
    out["median_rank"] = float(np.median(ranks))
    return out


# ----------------------------------------------------------------------------- bridging
def participation_coefficient(neigh, labels):
    """Participation coefficient of every node of a neighbour graph: P = 1 - sum_c (k_ic / k_i)^2.
    0 = all neighbours in one community; close to 1 = neighbours spread over many communities (a 'bridge')."""
    N, k = neigh.shape
    P = np.zeros(N)
    for i in range(N):
        _, cnt = np.unique(labels[neigh[i]], return_counts=True)
        P[i] = 1.0 - ((cnt / k) ** 2).sum()
    return P


# ----------------------------------------------------------------------------- nestedness
def nodf(M):
    """NODF nestedness (Almeida-Neto et al. 2008) of a 0/1 matrix, 0..100 (100 = perfectly nested)."""
    M = (np.asarray(M) > 0).astype(int)
    M = M[M.sum(1) > 0][:, M.sum(0) > 0]

    def part(A):
        r = A.sum(1)
        n = len(A)
        tot, pairs = 0.0, n * (n - 1) / 2
        for i in range(n):
            for j in range(i + 1, n):
                if r[i] == r[j]:
                    continue
                hi, lo = (i, j) if r[i] > r[j] else (j, i)
                tot += (A[hi] & A[lo]).sum() / r[lo]
        return tot / pairs * 100 if pairs else 0.0, pairs
    a, pa = part(M)
    b, pb = part(M.T)
    return float((a * pa + b * pb) / (pa + pb))


def curveball(M, rng, trades):
    """Random 0/1 matrix with exactly the same row sums and column sums (Strona et al. 2014)."""
    A = (np.asarray(M) > 0).astype(int).copy()
    n = len(A)
    for _ in range(trades):
        i, j = rng.choice(n, 2, replace=False)
        a, b = np.flatnonzero(A[i] & ~A[j]), np.flatnonzero(A[j] & ~A[i])
        if len(a) == 0 or len(b) == 0:
            continue
        k = int(rng.integers(1, min(len(a), len(b)) + 1))
        sa, sb = rng.choice(a, k, replace=False), rng.choice(b, k, replace=False)
        A[i, sa], A[i, sb], A[j, sb], A[j, sa] = 0, 1, 0, 1
    return A


# ----------------------------------------------------------------------------- other ranking algorithms (step 9)
def rwr_scores(data, Q, cont=0.5, iters=6):
    """Random walk with restart (personalised PageRank) on the bipartite disease-symptom graph.
    A walker starts on the patient's symptoms (restart distribution Q). At every step it either jumps back to the start (probability
    1 - cont) or walks symptom -> a random disease having it -> a random symptom of that disease (probability cont). A symptom that
    belongs to many diseases (a hub) spreads the walker thinly over them. After `iters` rounds the probability of standing on each
    disease is its score. Q: (b x T) sparse matrix of start weights. Returns a (b x N) sparse matrix whose rows sum to 1."""
    B = data.Dprop.tocsr()
    S2D = normalize(B.T.tocsr(), norm="l1").astype(np.float32).tocsr()
    D2S = normalize(B, norm="l1").astype(np.float32).tocsr()
    Q = normalize(csr_matrix(Q), norm="l1").astype(np.float32).tocsr()
    s = Q.copy()
    for _ in range(iters):
        s = (1 - cont) * Q + cont * ((s @ S2D) @ D2S)
    return s @ S2D


def jaccard_scores(data, Qb):
    """Plain overlap (no weighting by rarity): |patient AND disease| / |patient OR disease|. Qb: (b x T) 0/1 sparse. Returns dense (b x N)."""
    Qb = csr_matrix(Qb)
    inter = (Qb @ data.Dprop.T).toarray()
    dsz = np.asarray(data.Dprop.sum(1)).ravel()
    qs = np.asarray(Qb.sum(1)).ravel()[:, None]
    return inter / (dsz[None, :] + qs - inter)


# ----------------------------------------------------------------------------- adaptive questioning (step 10)
def best_split_symptom(Dp, cand, strategy="greedy", rng=None):
    """Choose the next symptom to ask about, given the diseases still possible (`cand`, an index array).
    Dp = diseases x symptoms 0/1 matrix (CSR). For every symptom let n_yes = number of candidates that have it.
    A symptom is USEFUL only if 0 < n_yes < n (it splits the candidates).
      greedy  minimise the expected number of candidates left after the answer, (n_yes^2 + n_no^2) / n
              = the best split; a symptom shared by about half of the candidates is best (binary search)
      hubs    the most common useful symptom         rare    the rarest useful symptom         random  any useful symptom
    Returns the symptom column, or None when no symptom can split the candidates (they have identical symptom sets)."""
    n = len(cand)
    ny = np.asarray(Dp[cand].sum(0)).ravel()
    useful = np.flatnonzero((ny > 0) & (ny < n))
    if len(useful) == 0:
        return None
    if strategy == "greedy":
        return int(useful[np.argmin((ny[useful] ** 2 + (n - ny[useful]) ** 2) / n)])
    if strategy == "hubs":
        return int(useful[np.argmax(ny[useful])])
    if strategy == "rare":
        return int(useful[np.argmin(ny[useful])])
    if strategy == "random":
        return int(rng.choice(useful))
    raise ValueError(strategy)


def adaptive_questions(data, target, strategy="greedy", rng=None, cap=60, Dp=None, Dc=None):
    """Play '20 questions' for the disease `target`: repeatedly ask the symptom chosen by `strategy`, get the TRUE answer from the
    target's symptom list, and keep only the diseases consistent with all answers so far.
    Returns {steps: [(symptom, answer, candidates_left), ...], n_questions, left, identified, stuck}.
    identified = exactly one disease left; stuck = several diseases left that no symptom can separate (identical symptom sets: twins)."""
    Dp = Dp if Dp is not None else data.Dprop.tocsr()
    Dc = Dc if Dc is not None else Dp.tocsc()
    cand = np.arange(data.n)
    steps = []
    while len(cand) > 1 and len(steps) < cap:
        s = best_split_symptom(Dp, cand, strategy, rng)
        if s is None:
            break
        ans = bool(Dp[target, s] > 0)
        has = Dc.getcol(s).toarray().ravel()[cand] > 0
        cand = cand[has == ans]
        steps.append((s, ans, int(len(cand))))
    return {"steps": steps, "n_questions": len(steps), "left": int(len(cand)), "identified": len(cand) == 1,
            "stuck": len(cand) > 1 and len(steps) < cap}


# ----------------------------------------------------------------------------- noise-robust adaptive questioning (step 11)
def _entropy(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def robust_choose(DpT, w, asked, eps, dl, strategy="ig", rng=None, popv=None):
    """Choose the next symptom to ask about when answers may be WRONG.
    w = current probability of every disease (sums to 1). DpT = symptoms x diseases 0/1 matrix.
    Answer model: a disease that HAS the symptom is answered 'yes' with probability 1 - eps (the patient may miss it);
                  a disease that LACKS the symptom is answered 'yes' with probability dl (the patient may invent it).
    m[s] = probability mass of the diseases that have symptom s.  Expected information gain of asking s (in bits):
        IG(s) = H(P(yes)) - m[s] H(eps) - (1 - m[s]) H(dl),   P(yes) = m (1 - eps) + (1 - m) dl      (H = binary entropy)
    strategy 'ig' = ask the symptom with the largest information gain (about half of the probability says yes);
             'hubs' = the most common symptom that still carries probability;  'random' = any such symptom."""
    m = DpT @ w
    if strategy == "ig":
        py = m * (1 - eps) + (1 - m) * dl
        ig = _entropy(py) - m * _entropy(eps) - (1 - m) * _entropy(dl)
        ig[asked] = -1.0
        return int(np.argmax(ig))
    if strategy == "hubs":
        sc = popv.copy()
        sc[asked] = -1.0
        sc[(m < 0.01) | (m > 0.99)] = -1.0
        return int(np.argmax(sc))
    ok = np.flatnonzero((~asked) & (m > 0.01) & (m < 0.99))
    return int(rng.choice(ok)) if len(ok) else int(rng.integers(len(m)))


def robust_update(w, col, ans, eps, dl):
    """Bayes' rule: multiply every disease's probability by the chance of the given answer, then renormalise.
    col = 0/1 vector over diseases: does the disease have the asked symptom? A wrong answer lowers the true disease
    but never sets it to zero, so later answers can bring it back (hard elimination cannot)."""
    lik = np.where(col > 0, (1 - eps) if ans else eps, dl if ans else 1 - dl).astype(np.float32)
    w = w * lik
    return w / w.sum()


# ----------------------------------------------------------------------------- bipartite modules: Barber's modularity and BRIM (step 12)
def _onehot(lab, c):
    return csr_matrix((np.ones(len(lab)), (np.arange(len(lab)), lab)), shape=(len(lab), c))


def bipartite_modularity(A, row, col, c):
    """Barber's bipartite modularity Q_B of a partition of the diseases (`row` labels) AND the symptoms (`col` labels) into c modules:
        Q_B = (1/m) * sum_{i,j} (A_ij - k_i d_j / m) * [module(i) == module(j)]
    where A is the disease x symptom 0/1 matrix, k_i = symptoms of disease i, d_j = diseases having symptom j, m = number of links.
    It compares the links INSIDE modules with the number expected if every disease linked to symptoms at random (same degrees).
    Two disjoint equal blocks give 0.5; a random partition gives about 0."""
    k = np.asarray(A.sum(1)).ravel()
    d = np.asarray(A.sum(0)).ravel()
    m = A.sum()
    X, Y = _onehot(row, c), _onehot(col, c)
    inside = (X.T @ A @ Y).diagonal().sum()
    null = ((X.T @ k) * (Y.T @ d)).sum() / m
    return float((inside - null) / m)


def symptom_modules_for(A, row, c):
    """Given the disease modules, put every symptom into the module that raises modularity most (half of one BRIM step)."""
    k = np.asarray(A.sum(1)).ravel()
    d = np.asarray(A.sum(0)).ravel()
    m = A.sum()
    X = _onehot(row, c)
    S = np.asarray((A.T @ X).todense()) - np.outer(d, (k @ X) / m)
    return S.argmax(1)


def brim(A, c, rng, iters=100):
    """BRIM = Bipartite, Recursively Induced Modules (Barber 2007). Start from random symptom modules, then alternate:
    (1) every disease moves to the module that maximises Q_B given the symptom modules, (2) every symptom moves to the module that
    maximises Q_B given the disease modules, until nothing changes. Returns (disease labels, symptom labels, Q_B)."""
    N, T = A.shape
    k = np.asarray(A.sum(1)).ravel()
    d = np.asarray(A.sum(0)).ravel()
    m = A.sum()
    col = rng.integers(0, c, T)
    row = None
    for _ in range(iters):
        Y = _onehot(col, c)
        S = np.asarray((A @ Y).todense()) - np.outer(k, (d @ Y) / m)
        new_row = S.argmax(1)
        new_col = symptom_modules_for(A, new_row, c)
        if row is not None and (new_row == row).all() and (new_col == col).all():
            break
        row, col = new_row, new_col
    return row, col, bipartite_modularity(A, row, col, c)


def module_coverage(A, row, c):
    """c x T matrix: share of the diseases of each module that list each symptom (the module's symptom profile)."""
    X = _onehot(row, c)
    counts = np.asarray((X.T @ A).todense())
    sizes = np.maximum(np.asarray(X.sum(0)).ravel(), 1)
    return counts / sizes[:, None]


def module_profile(A, row, c):
    """Counts of diseases per module listing each symptom, module sizes, and each symptom's overall prevalence (used by module_scores)."""
    X = _onehot(row, c)
    counts = np.asarray((X.T @ A).todense())
    sizes = np.maximum(np.asarray(X.sum(0)).ravel(), 1)
    prev = np.maximum(np.asarray(A.sum(0)).ravel(), 1) / A.shape[0]
    return counts, sizes, prev


def module_scores(profile, symptoms, a=0.5):
    """Score every module for a patient with the given symptoms: sum over the symptoms of log( how over-represented the symptom is in the module ),
    log[(n_module_with_symptom + a) / (module_size x overall_prevalence + a)]   (a naive-Bayes style log-lift).
    A rare symptom concentrated in one module gives that module a large score; a hub symptom (the same share in every module) adds about 0."""
    counts, sizes, prev = profile
    return np.log((counts[:, symptoms] + a) / (sizes[:, None] * prev[None, symptoms] + a)).sum(1)
