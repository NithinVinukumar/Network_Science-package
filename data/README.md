# Data: Human Phenotype Ontology (HPO) rare-disease annotations

Three files from the HPO project (https://hpo.jax.org, releases at
https://github.com/obophenotype/human-phenotype-ontology/releases), release dated 2026-09-02:

| File | What it is |
|---|---|
| `hp.obo` | the symptom ("phenotype") ontology: every symptom term and its parent terms (a family tree, 20,482 terms) |
| `phenotype.hpoa` | rare-disease annotations: which symptoms were recorded for each disease (OMIM, Orphanet and DECIPHER diseases) |
| `genes_to_disease.txt` | which genes are linked to which diseases |

Please cite the HPO (for example Kohler et al., "The Human Phenotype Ontology in 2021", Nucleic Acids Research 49(D1):D1207-D1217; verify the reference) and
respect the HPO's licence and terms (see the HPO website). The annotations are curated from the literature: they describe diseases,
NOT individual patients, and they say nothing about how common a disease is.

To download fresh copies: `python fetch_data.py` (needs internet).
