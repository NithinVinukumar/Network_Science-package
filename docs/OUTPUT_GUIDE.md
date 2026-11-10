# Output guide: what each figure and number means

All numbers below are for seed 42, full quality (`python run_all.py`). Re-running gives the same values.

## Figures (folder `figures/`)

| File | What you see | What to say |
|---|---|---|
| `1_network_structure.png` | left: share of symptoms with at least this many diseases (log-log); middle: symptoms per disease; right: information content vs number of diseases | "A few symptoms are huge hubs and most are rare; the more diseases share a symptom, the less information it carries." |
| `1_kcore.png` | number of nodes at each k-core level | "The dense core (k = 36) holds 330 diseases and 196 general neurodevelopmental symptoms." |
| `2_similarity.png` | left: how alike each disease is to its closest look-alike (dashed lines = twin thresholds); middle: family cohesion; right: symptom similarity of gene-sharing vs other pairs (log scale) | "About 12% of diseases have a near-identical twin; disease families are 3-8 times more alike than random; similarity predicts shared genes (AUC 0.845)." |
| `3_communities.png` | left: heat map of communities vs organ systems; right: purity of each community | "32 communities (Q = 0.79) follow organ systems only partly (NMI 0.41): some are 95% nervous-system, others mix systems." |
| `4_percolation.png` | left: share of diseases in the giant cluster vs similarity threshold, real vs random null; middle: second-largest cluster; right: isolated diseases | "The real network merges into one giant cluster much earlier than the random null (half at 0.45 vs 0.34): related diseases share rare symptoms." |
| `5_robustness.png` | left: top-10 success vs fraction of symptoms known for random / common-only / specific-only; middle: main scenarios; right: effect of wrong symptoms | "The kind of symptom matters more than the amount: with 25% known, common-only 22%, specific-only 98%." |
| `6_diagnosability.png` | four panels: single-feature AUCs, success by organ system, success by twin similarity, success by number of listed symptoms | "Network features predict which diseases are missed (AUC 0.77-0.83); twins hide; difficulty is U-shaped in symptom count because the doctor always knows at least 2 symptoms." |
| `7_link_prediction.png` | recall of hidden symptoms: popularity baseline vs network method | "Look-alike diseases recover 38% of hidden symptoms vs 11% for the baseline." |
| `12_modules.png` | left: bipartite modularity of BRIM modules, Louvain communities, organ systems and random groups (symptoms assigned optimally to each); middle: triage accuracy of BRIM modules vs organ systems; right: sizes of the largest modules | "BRIM finds modules of diseases AND symptoms with higher bipartite modularity (0.41 vs 0.35), and from 25% of a patient's symptoms it finds the right module for 75% of diseases against 66% for organ systems." |
| `11_robust_questions.png` | % of diseases ranked first against the number of questions, with perfect answers (left) and noisy answers (right), for hard elimination, robust information gain, most-common-first and random questions | "With perfect answers the robust method matches hard elimination (97%); with noisy answers hard elimination collapses to 19% while the robust method reaches 79%." |
| `10_questions.png` | left: diseases still possible after each question (log scale) for four ways of choosing questions; middle: questions needed by the greedy strategy vs the binary-search limit; right: % of diseases found uniquely | "The best question splits the candidates in half, like binary search: 13.9 questions on average (limit 13.7), 96% found; the rest are exact twins. Asking the most common symptom needs 19.6; rare or random symptoms almost never succeed." |
| `9_algorithms.png` | grouped bars: % of diseases found in the top 10 by three algorithms (IC-weighted cosine, Jaccard, random walk with restart) for random / only-common / only-specific symptoms | "Our rarity-weighted similarity beats plain overlap and a PageRank-style random walk (84% vs 70% vs 52% with a random 25% of symptoms); with only common symptoms every method fails." |
| `8_joubert.png` | left: Joubert subtypes (rows, by gene) x most shared symptoms; middle: diseases per gene; right: nestedness vs random matrices | "No nested severity ladder (gap under half a NODF point); every Joubert gene also causes other diseases (TMEM67: 11)." |

## Numbers (file `results/summary.md`, raw values in `results/stepN.json`)

| Number | Meaning |
|---|---|
| 12,866 / 12,732 / 267,579 | diseases / symptom terms / direct links |
| 1,537 (11.9%) twins | diseases whose most similar other disease has similarity at least 0.8; 89% inside one database |
| AUC 0.845 | chance that a gene-sharing pair has higher similarity than a random pair |
| Q = 0.79, NMI = 0.41 | strength of community structure; agreement with organ systems |
| 0.45 vs 0.34 | similarity at which half of the diseases are in the giant cluster: real vs null |
| 85.2% (49.8% top-1) | diseases found in the top 10 with 25% of symptoms known |
| 21.7% vs 98.1% | only common vs only specific 25% of symptoms |
| 94.1 / 85.3 / 76.9% | top-10 success with 0 / 2 / 5 wrong symptoms |
| 86.2% | OMIM-only check (no OMIM-Orphanet duplicates) |
| AUC 0.77 / 0.83 | predicting a missed diagnosis from network features (logistic / boosting, 5-fold CV) |
| 89 / 78 / 60 / 82 / 96% | success for diseases with 1-3 / 4-6 / 7-10 / 11-20 / 21+ symptoms |
| 37.7% vs 10.7% | hidden symptoms recovered in the top 20: network vs popularity |
| Z = -4.5 (NODF 12.9 vs 13.3) | Joubert subtypes: statistically clear but tiny departure from chance nestedness |
| p = 0.013 / 0.187 / 0.003 / 0.014 | Joubert clinical forms hepatic / ocular / renal / oculorenal (4 tests: Bonferroni 0.0125) |

## Files in `results/`
- `summary.md`: everything above in words, one "Meaning" line per step.
- `step1.json` ... `step8.json`: raw numbers (what the paper quotes).
- `_cache_neighbours.npz`, `_cache_ranks25.npz`: speed-up caches (safe to delete).
