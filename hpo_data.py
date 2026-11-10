"""
hpo_data.py - read the HPO files and build the disease-symptom network
=======================================================================
What it builds (all stored in one `HPOData` object)

    Ddir   diseases x symptoms matrix of DIRECT annotations (1 = curators listed this symptom for this disease)
    Anc    symptom x symptom matrix: row t has 1 for t itself and for every ancestor (parent, grandparent ...) of t
    Dprop  Ddir after "propagation": if a disease has "Seizure" it also has the general terms above it
           ("Abnormality of the nervous system" ...). Dprop = (Ddir @ Anc) > 0
    ic     information content of each symptom: ic(t) = -ln( share of diseases having t ). A symptom of every disease
           has ic 0 (no information); a symptom of one disease has the largest ic.
    Dw     Dprop with every row weighted by ic and scaled to length 1 -> the dot product of two rows is the
           IC-weighted COSINE SIMILARITY of two diseases.

Only the "P" aspect (phenotypic abnormalities) and non-negated annotations are used. Obsolete terms are dropped and alternative
term ids are mapped to their primary id.

Also provided: genes of each disease (genes_to_disease.txt) and an "organ system" label for each disease (the top-level
branch of the symptom tree, one of the children of "Phenotypic abnormality", that the disease's symptoms fall in most often).
"""
import collections
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from config import Settings

ABNORMALITY_ROOT = "HP:0000118"


def parse_obo(path):
    """Return (names, parents, alt_to_primary, obsolete) from an OBO file."""
    names, parents, alt, obsolete = {}, collections.defaultdict(set), {}, set()
    cur = None
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line == "[Term]":
            cur = None
        elif line.startswith("id: HP:"):
            cur = line[4:]
        elif cur and line.startswith("name: "):
            names[cur] = line[6:]
        elif cur and line.startswith("is_obsolete: true"):
            obsolete.add(cur)
        elif cur and line.startswith("alt_id: HP:"):
            alt[line[8:]] = cur
        elif cur and line.startswith("is_a: HP:"):
            parents[cur].add(line[6:16])
    return names, parents, alt, obsolete


def propagate_matrices(Ddir, Anc, n_diseases):
    """From direct annotations to (Dprop, ic, Dw). Used for the real data and for random null networks."""
    Dp = Ddir @ Anc
    Dp.data[:] = 1.0
    Dp = Dp.tocsr()
    count = np.asarray(Dp.sum(0)).ravel()
    ic = -np.log(np.maximum(count, 1) / n_diseases)
    Dw = normalize(Dp.multiply(ic).tocsr()).astype(np.float32)
    return Dp, ic, Dw


@dataclass
class HPOData:
    ids: list                       # disease ids ("OMIM:130000", "ORPHA:475" ...), sorted
    disease_names: list             # parallel to ids
    terms: list                     # symptom ids ("HP:0001250"), sorted
    term_names: list                # parallel to terms
    Ddir: csr_matrix
    Anc: csr_matrix
    Dprop: csr_matrix
    ic: np.ndarray
    Dw: csr_matrix
    genes: dict                     # disease id -> set of gene symbols
    organ: np.ndarray               # organ-system index per disease (-1 = none)
    organ_names: list
    parents: dict = field(repr=False, default_factory=dict)
    term_index: dict = field(repr=False, default_factory=dict)
    disease_index: dict = field(repr=False, default_factory=dict)

    @property
    def n(self):
        return len(self.ids)

    @property
    def t(self):
        return len(self.terms)

    @property
    def source(self):
        return np.array([i.split(":")[0] for i in self.ids])

    def direct_terms(self, i):
        """Column indices of the directly annotated symptoms of disease i."""
        return self.Ddir.indices[self.Ddir.indptr[i]:self.Ddir.indptr[i + 1]]

    def find_disease(self, text, limit=10):
        """Diseases whose name contains `text` (case-insensitive)."""
        t = text.lower()
        return [i for i, nm in enumerate(self.disease_names) if t in nm.lower()][:limit]

    def find_term(self, text, limit=10):
        t = text.lower()
        return [i for i, nm in enumerate(self.term_names) if t in nm.lower()][:limit]


def load(cfg=None, sources=None):
    """Build the whole dataset. `sources` = e.g. ["OMIM"] to keep one database only (a sensitivity check)."""
    cfg = cfg or Settings()
    names, parents, alt, obsolete = parse_obo(cfg.data_dir / "hp.obo")
    df = pd.read_csv(cfg.data_dir / "phenotype.hpoa", sep="\t", comment="#", dtype=str).fillna("")
    df = df[(df.aspect == "P") & (df.qualifier != "NOT")].copy()
    df["hpo_id"] = df.hpo_id.map(lambda t: alt.get(t, t))
    df = df[df.hpo_id.isin(names) & ~df.hpo_id.isin(obsolete)]
    if sources:
        df = df[df.database_id.str.split(":").str[0].isin(sources)]
    dname = df.drop_duplicates("database_id").set_index("database_id")["disease_name"].to_dict()
    d2t = df.groupby("database_id")["hpo_id"].apply(set).to_dict()
    ids = sorted(d2t)

    cache = {}

    def anc(t):
        if t in cache:
            return cache[t]
        s = {t}
        for p in parents.get(t, ()):
            s |= anc(p)
        cache[t] = s
        return s

    used = set()
    for s in d2t.values():
        for t in s:
            used |= anc(t)
    terms = sorted(used)
    tx = {t: i for i, t in enumerate(terms)}
    T, N = len(terms), len(ids)

    r, c = [], []
    for i, d in enumerate(ids):
        for t in d2t[d]:
            r.append(i)
            c.append(tx[t])
    Ddir = csr_matrix((np.ones(len(r)), (r, c)), shape=(N, T))
    Ddir.data[:] = 1.0
    ar, ac = [], []
    for t in terms:
        for a in anc(t):
            ar.append(tx[t])
            ac.append(tx[a])
    Anc = csr_matrix((np.ones(len(ar)), (ar, ac)), shape=(T, T))
    Dprop, ic, Dw = propagate_matrices(Ddir, Anc, N)

    g = pd.read_csv(cfg.data_dir / "genes_to_disease.txt", sep="\t", dtype=str)
    gmap = g.groupby("disease_id")["gene_symbol"].apply(set).to_dict()
    genes = {d: gmap[d] for d in ids if d in gmap}

    top = sorted(t for t in terms if ABNORMALITY_ROOT in parents.get(t, ()))
    topx = {t: i for i, t in enumerate(top)}
    organ = np.full(N, -1)
    for i, d in enumerate(ids):
        cnt = collections.Counter()
        for t in sorted(d2t[d]):
            for a in anc(t):
                if a in topx:
                    cnt[a] += 1
        if cnt:
            # most symptoms in the branch; ties go to the branch listed first (fixed rule, so results never depend on set order)
            organ[i] = min(topx[a] for a in cnt if cnt[a] == max(cnt.values()))
    return HPOData(ids=ids, disease_names=[dname[d] for d in ids], terms=terms, term_names=[names[t] for t in terms],
                   Ddir=Ddir, Anc=Anc, Dprop=Dprop, ic=ic, Dw=Dw, genes=genes, organ=organ,
                   organ_names=[names[t] for t in top], parents=dict(parents), term_index=tx,
                   disease_index={d: i for i, d in enumerate(ids)})


def similarity_block(data, query_rows):
    """IC-weighted cosine similarity of every disease (rows) to the diseases `query_rows` (columns). Shape (N, b)."""
    return (data.Dw @ data.Dw[query_rows].T).toarray()


if __name__ == "__main__":
    d = load()
    print(f"{d.n} diseases, {d.t} symptom terms, {d.Ddir.nnz} direct links, {len(d.organ_names)} organ systems")
    print("sources:", dict(collections.Counter(d.source)))
    print("example:", d.ids[5], d.disease_names[5], "| direct symptoms", len(d.direct_terms(5)))
