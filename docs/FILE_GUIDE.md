# File guide

## Data and settings
| File | What it is |
|---|---|
| `data/hp.obo` | the symptom ontology: 20,000+ symptom terms and their parent terms (a family tree) |
| `data/phenotype.hpoa` | which symptoms were recorded for each rare disease (OMIM, Orphanet, DECIPHER) |
| `data/genes_to_disease.txt` | which genes are linked to which diseases |
| `data/README.md` | source, licence note and citation for the data |
| `fetch_data.py` | re-downloads the three data files (needs internet) |
| `config.py` | every adjustable setting (seed, thresholds, number of nulls, sample sizes); `Settings.quick()` for fast tests |

## Building blocks (no output of their own)
| File | What it does |
|---|---|
| `hpo_data.py` | reads the files and builds the network: direct and propagated disease-symptom matrices, information content of every symptom, the similarity-ready matrix, genes, organ-system labels. `HPOData` is the object every step uses |
| `network_tools.py` | reusable operations: nearest look-alikes, similarity links above a threshold, the random null network, simulated patients and ranking, giant-component statistics, participation coefficient, nestedness (NODF) and the curveball random matrices |
| `utils.py` | small helpers: random generator, safe figure saving, confidence intervals (Wilson), AUC, tail exponent, plain-language explanation blocks, JSON writing |

## The twelve analysis steps (each writes figures to `figures/` and numbers to `results/stepN.json`)
| File | Step |
|---|---|
| `step1_network.py` | network structure: degrees, hubs, tail exponent, k-core, assortativity, friendship paradox |
| `step2_similarity.py` | similarity, near-twins, disease families, gene-sharing check; saves a cache of nearest neighbours |
| `step3_communities.py` | Louvain communities, modularity, agreement with organ systems, stability, bridge diseases |
| `step4_percolation.py` | similarity-threshold percolation against 5 random null networks |
| `step5_robustness.py` | simulated diagnosis with random / common-only / specific-only symptoms, wrong symptoms, OMIM-only check; saves ranks for step 6 |
| `step6_diagnosability.py` | which diseases are missed: single features, logistic regression and gradient boosting, organ systems, symptom-count bins |
| `step7_link_prediction.py` | hide 20% of symptoms and recover them from look-alike diseases |
| `step8_joubert.py` | Joubert syndrome case study: core symptoms, nestedness, clinical forms, pleiotropy, ciliopathy communities |
| `step12_modules.py` | BRIM (Barber's bipartite modularity): modules of diseases AND symptoms with signature symptoms; compared with Louvain and organ systems on modularity, organ agreement, gene sharing and triage |
| `step11_robust_questions.py` | noise-robust adaptive questioning: Bayesian posterior over the diseases + information-gain questions, benchmarked against hard elimination, most-common-first and random questions, with perfect and noisy answers |
| `step10_questions.py` | adaptive questioning: the greedy test-cover algorithm ('20 questions') compared with hubs-first, rarest-first and random questions; counts exact twins |
| `step9_algorithms.py` | algorithm benchmark: IC-weighted cosine vs Jaccard overlap vs random walk with restart (personalised PageRank) on the same simulated patients |

## Tools for you
| File | What it does |
|---|---|
| `run_all.py` | runs all eight steps (`--quick`, `--only N ...`), then writes the summary |
| `reporting.py` | builds `results/summary.md` from the saved JSON files |
| `live_test.py` | LIVE TESTING for presentations: you type findings / diseases / symptoms and the system answers; each option names the network concept and the code that runs |
| `demo.py` | guided 5-scene live demo for presentations (`--auto` to test, `--disease` to change scene 3) |
| `explore.py` | interactive menu: look up a disease, simulate a patient, diagnose from your own symptoms, compare two diseases |
| `.vscode/launch.json` | ready-made Run and Debug buttons in VS Code |

## Tests
| File | What it checks |
|---|---|
| `tests/test_core.py` | 29 tests: helpers (confidence interval, AUC, tail exponent), toy networks worked out by hand (information content, nestedness 100 for a perfect staircase, participation coefficient, giant-component sizes), real-data checks (12,866 diseases, similarity symmetric, a fully known patient is found, rare symptoms beat common ones, null keeps degrees), explorer input handling |
| `tests/test_live.py` | 17 tests: typed findings give the Joubert and Bardet-Biedl families, answering questions adds findings, unknown symptoms are handled, stress test, twin check, symptom lookup, menu |
| `tests/test_demo.py` | 6 tests: the demo gives the results the presentation relies on (family found, specific beats common, twins, results read) |
| `tests/test_steps.py` | 14 tests: runs all eight steps in quick mode and checks the results are sensible (real percolates earlier than null, specific beats common, link prediction beats baseline, ...) |

## Documents and outputs
| Folder | What is in it |
|---|---|
| `figures/` | 13 pictures made by the steps |
| `results/` | `summary.md` (all numbers), `stepN.json` (raw numbers), `_cache_*.npz` (speed-up caches; safe to delete) |
| `docs/` | this guide, `PRESENTATION_SPLIT.md` (two-speaker script), `CONCEPTS_SIMPLE.md`, `OUTPUT_GUIDE.md`, `VIVA_QA.md` |
| `paper/` | pointer to the paper draft and a copy of its text if exported |
