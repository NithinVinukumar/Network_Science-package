# Live testing script: YOU type, the system answers

Run: `python live_test.py`. Nothing is pre-filled or pre-printed. Every option prints two lines on screen,
`NETWORK CONCEPT` and `CODE THAT RUNS`, so you can point at them while you talk.
(`python demo.py` is an optional scripted version; for your presentation use `live_test.py`.)

## Menu
| Option | You type | The system shows | Concept | Code |
|---|---|---|---|---|
| 1 Diagnose | findings (symptom names, commas) | information content of each finding, ranked diseases, top-2 gap and twin warning, best next questions; you answer `y`, `n` or type another symptom | bipartite network, information content, similarity (projection), neighbourhood voting | `explore.query_vector`, `data.Dw @ query` (hpo_data.py) |
| 2 Stress test | a disease name and a fraction | the true disease's rank with random / only common / only specific symptoms | robustness: hubs vs periphery | `network_tools.popularity`, `make_patients`, step5_robustness.py |
| 3 Twin check | two disease names | similarity, shared and unique symptoms, genes, verdict | similarity network, twin clusters, gene layer (multilayer) | `data.Dw`, `data.genes`, step2_similarity.py |
| 8 Which syndromic module? | findings (symptom names, commas) | the 3 modules the findings point to, their signature symptoms, and the best diseases inside the best module | Barber's bipartite modularity and BRIM (modules of diseases AND symptoms) | `step12_modules.get_modules`, `network_tools.module_profile`, `module_scores` |
| 7 Noisy 20 questions | a disease name, how often the patient misses symptoms (default 0.2) | each question, the patient's answer (MISTAKES are flagged), the 3 most probable diseases with their probabilities | Bayesian posterior over the diseases + expected information gain | `network_tools.robust_choose`, `robust_update`, step11_robust_questions.py |
| 6 Play 20 questions | a disease name (and, if you want, you answer yes/no yourself) | the system's questions, the number of diseases left after each answer, and the disease it finds | greedy test-cover / identifying code on the bipartite network (binary search) | `network_tools.best_split_symptom`, `adaptive_questions`, step10_questions.py |
| 5 Compare algorithms | findings (symptom names, commas) | the same findings ranked by IC-cosine, Jaccard overlap and random walk with restart | similarity (projection) vs diffusion on the network (personalised PageRank) | `network_tools.rwr_scores`, `jaccard_scores`, step9_algorithms.py |
| 4 Symptom lookup | a symptom name | diseases that have it and its information content | degree, hubs, heavy tail, information content | `network_tools.popularity`, `data.ic` |

## Typed examples that work (real outputs)

**Option 1, example A.** Type: `molar tooth sign, ataxia, nystagmus`
- findings: Molar tooth sign (57 diseases, IC 5.42, rare); Ataxia (1445, 2.19); Nystagmus (1299, 2.29)
- top: Joubert syndrome 25 (0.57), 31 (0.53), 28 (0.46); the top two are 0.51 similar, so "a well-chosen question can separate them"
- next questions: Retinopathy, Polydactyly, Aplasia/Hypoplasia of the cerebellum.
- **What `y` and `n` do.** Each answer prints a panel "Changes since your last answer" with the biggest moves among all diseases.
  - `y` = the patient HAS it: it is added as a finding. Retinopathy yes: Bardet-Biedl syndrome 11 goes UP (0.01 -> 0.31), Joubert syndrome 25 goes DOWN (0.57 -> 0.47) because it does not have retinopathy.
  - `n` = the patient does NOT have it: diseases that have that symptom are pushed DOWN (rare symptoms push harder). Retinopathy no: Bardet-Biedl syndrome 11 0.01 -> -0.43; the Joubert diseases stay on top because they do not have retinopathy, so retina diseases are ruled out. Say no to Polydactyly (the second question): Joubert syndrome 25 drops 0.57 -> 0.23 and Joubert syndrome 31 0.53 -> 0.26, because they have polydactyly.
  - Scores can go below zero after a `n`: it is a score, not a probability.

**Option 1, example B.** Type: `retinal dystrophy, polydactyly, obesity`
- top: Bardet-Biedl syndrome 14 (0.72), 11 (0.62), 10 (0.61); next question: Rod-cone dystrophy.

**Option 1, example C (vague findings).** Type: `seizure, hypotonia, intellectual disability`
- all three are hubs (IC about 1.4 to 1.5); top 0.70 is an intellectual developmental disorder; the top two are 0.89 similar, so the system prints the TWIN WARNING. Say: "hub symptoms do not narrow anything down; compare with example A".

**Option 2.** Type `Joubert syndrome 6`, then Enter for 0.25: only common symptoms rank #85, only specific symptoms rank #1.
(Other diseases behave differently; for Bardet-Biedl syndrome 1 common-only ranks #40 and specific-only #1.)

**Option 3.** `Porokeratosis 1` and `Porokeratosis 7`: similarity 1.00, genes PMVK vs MVD, verdict TWINS.
`Joubert syndrome 6` and `Joubert syndrome 1`: similarity 0.44, genes TMEM67 vs INPP5E, verdict not twins.

**Option 4.** `seizure`: Seizure is in 3,129 diseases (IC 1.41); Motor seizure 579 (3.10); Bilateral tonic-clonic seizure 491 (3.27).

**Option 5.** Type `molar tooth sign, ataxia, nystagmus`: IC-cosine puts Joubert syndrome 25, 31, 28 on top; Jaccard overlap returns tiny diseases (Benign paroxysmal tonic upgaze of childhood with ataxia, Kuru ...); random walk with restart finds the ciliopathy family (Joubert syndrome 2, Orofaciodigital syndrome type 6, Arima syndrome, Meckel syndrome 1). Say: "plain overlap favours tiny diseases; the random walk finds the family but less precisely; rarity-weighted similarity wins in our benchmark (84% / 70% / 52%)."

**Option 6.** Type `6`, `Joubert syndrome 6`, Enter (the system answers from the data): 13 questions, diseases left 6286, 3136, 1498, 721, 355, 189, 96, 48, 23, 11, 6, 3, 1, then FOUND. Say: "each question halves the candidates, like binary search; 12,866 diseases need at least 13.7 questions". Type `6`, `Porokeratosis 1`: the system STOPS with 2 diseases that have IDENTICAL symptom sets (exact twins): "no question can separate them; a gene test is needed". Type `y` at the second prompt to answer the questions yourself.

**Option 7.** Type `7`, `Joubert syndrome 6`, Enter (patient misses 20%), Enter: watch the lines marked `<-- MISTAKE` and the probabilities of the three most probable diseases; a mistake lowers the true disease's probability but later answers bring it back. Say: "step 10 would have crossed the true disease out at the first mistake; here it only loses probability, and the benchmark shows 79% against 19%." With `0` instead of `0.2` there are no mistakes.

**Option 8.** Type `8` then, for example: `nyctalopia, rod-cone dystrophy` -> module 4 (eye, 57%) score 4.58, next module -0.23, best disease Retinitis pigmentosa 35; `nail dystrophy, palmoplantar keratoderma` -> module 5 (skin, 62%), score 5.11; `decreased circulating IgG, immunodeficiency` -> module 1 (immune), score 2.86; `hypertelorism, cleft palate, micrognathia` -> module 0 (craniofacial), score 3.51. Say: "BRIM groups diseases and symptoms together, so each module has signature symptoms; the findings point to one module before we rank single diseases." Avoid `molar tooth sign, ataxia, nystagmus` here: the Joubert diseases are spread over three modules (limit of step 12), so module 16 comes first and the best overall match, Joubert syndrome 25, lies in module 3.

## Let your professor type
Hand over the keyboard for option 1 or 4. Part of a name is enough; if several symptoms match, the program lists them and you pick a number.
Every typed symptom must be an HPO symptom name; if one is not found the program says so and carries on.

## Before presenting
Run `python live_test.py` once and try example A. No `run_all.py` is needed for live_test.py.
Limits to say: simulated findings from curated disease descriptions; not a diagnostic tool.
