# The project and every concept, in simple words

## The problem
There are thousands of rare diseases (this project uses 12,866 from a public database). A patient arrives with some symptoms; the doctor must work out which disease it might be. Often only a few symptoms are known, and some may be wrong.
**The question:** which rare diseases are easy to recognise from a few symptoms, which are hard, and why?

## The data
The Human Phenotype Ontology (HPO) is a free public database: one "page" per rare disease, listing the symptoms doctors have recorded for it. It also contains a family tree of symptoms ("Seizure" sits under "Abnormality of the nervous system").
Mini example (made up):

| Disease | Symptoms |
|---|---|
| A | developmental delay, floppy muscles, special brain-scan sign |
| B | developmental delay, seizures |
| C | developmental delay, floppy muscles, kidney problems |

## The network
Two kinds of dots: DISEASES and SYMPTOMS. A line means "this disease has this symptom". Real size: 12,866 diseases, 12,732 symptom terms, 267,579 lines.

## Three simple ideas that drive everything
1. **Popular symptoms are weak clues.** "Developmental delay" is in 2,583 diseases, so it tells you almost nothing. A rare symptom that appears in one disease tells you a lot. Like a library search: the word "the" is useless, a rare word finds the book. We measure this with **information content**: IC = -ln(share of diseases with the symptom). Top hub: 1.5. Rarest symptom: 9.5.
2. **Look-alike diseases.** Two diseases are alike when they share many *rare* symptoms. We compute this as an **information-weighted cosine similarity** (0 = nothing in common, 1 = identical). 11.9% of diseases have a "twin" with similarity 0.8 or more.
3. **Diagnosability.** How likely the true disease is to appear in the doctor's top-10 list when only part of its symptoms is known.

## The concepts, one by one

| Concept | Plain meaning | Where | Result to quote |
|---|---|---|---|
| **Bipartite network** | two kinds of dots; lines only between different kinds | step 1 | 12,866 diseases + 12,732 symptoms |
| **Projection** | turn it into a disease-to-disease "how alike" network | step 2 | similarity network |
| **Degree, hub** | number of lines of a dot; a hub has very many | step 1 | "global developmental delay" in 2,583 diseases |
| **Heavy tail, power-law exponent** | a few huge hubs, many tiny dots; the exponent says how fast it fades | step 1 | exponent 1.78 / 2.00 / 2.36 (cut-offs 10 / 30 / 100): heavy-tailed but not a clean power law |
| **Hierarchy (ontology)** | symptoms form a family tree; a specific symptom implies its general parents | data | "propagation" |
| **Information content** | rare symptom = more information | all steps | 1.5 vs 9.5 |
| **k-core** | keep removing dots with fewer than k lines; what survives is the dense core | step 1 | k = 36: 330 diseases, 196 symptoms (general neurodevelopmental ones) |
| **Core-periphery** | a dense core plus a long tail of rare symptoms | steps 1, 8 | 25% of symptoms occur in one disease only |
| **Assortativity** | do well-connected dots link to well-connected dots? | step 1 | -0.14 (hub-and-spoke pattern) |
| **Friendship paradox** | your neighbours are more popular than you | step 1 | symptoms of a typical disease occur in 310 diseases vs 23 for a typical symptom (13x) |
| **Community detection (Louvain), modularity** | find groups of densely linked diseases; Q says how strong (above 0.3 is real) | step 3 | 32 communities, Q = 0.79 |
| **NMI** | how well two groupings agree (0 none, 1 identical) | step 3 | 0.41 with organ systems |
| **Participation coefficient** | how many different communities a node's neighbours lie in (bridge nodes) | step 3 | Wolfram syndrome 0.86 |
| **Percolation, giant component** | add links step by step; at some point most nodes suddenly join one big cluster | step 4 | half of diseases in one cluster at similarity 0.45 |
| **Null model** | a random network with the same degrees; tells you if a pattern is special | step 4, 8 | real: half of diseases join one cluster at similarity 0.45; random null: 0.34 |
| **Robustness (random vs targeted loss)** | what breaks when symptoms are lost at random or in a chosen order | step 5 | 22% (common only) vs 98% (specific only) |
| **Link prediction** | guess missing links from the existing network | steps 2, 7 | gene sharing AUC 0.845; hidden symptoms recovered 38% vs 11% |
| **AUC** | 0.5 = no better than chance, 1.0 = perfect separation | steps 2, 6 | 0.77 (logistic), 0.83 (boosting) for hard diseases |
| **Multilayer network** | several kinds of links on the same nodes (symptom layer + gene layer) | steps 2, 8 | genes validate symptom similarity |
| **Pleiotropy, ego network** | one gene causing many diseases; the neighbourhood of a few nodes | step 8 | median Joubert gene links to 4 diseases; none cause Joubert only |
| **Nestedness (NODF)** | are small groups' symptom sets contained in bigger groups' sets (a ladder)? | step 8 | no evidence of a ladder |
| **Permutation test** | shuffle labels many times to see how often chance matches the result | step 8 | renal form p = 0.003 |
| **Random walk with restart (personalised PageRank)** | a walker starts on the patient's symptoms and keeps walking disease <-> symptom, sometimes jumping back to the start; the diseases it visits most are the answer | step 9 | 52% found in the top 10 (vs 84% for IC-cosine) |
| **Greedy test-cover / identifying code ('20 questions')** | choose symptom nodes whose neighbourhoods tell every disease apart; ask the symptom that splits the remaining diseases about half and half | step 10 | 13.9 questions on average (limit 13.7); 4.2% exact twins cannot be separated |
| **Bayesian posterior and expected information gain** | keep a probability for every disease, update it with Bayes' rule after each answer, and ask the symptom whose answer is expected to tell you the most (about 50/50) | step 11 | with noisy answers: 79% ranked first vs 19% for hard elimination |
| **Bipartite modularity and BRIM (Barber 2007)** | find groups (modules) that contain diseases AND symptoms together, so that more links fall inside the modules than chance predicts; the BRIM algorithm alternates: every disease joins its best module, then every symptom joins its best module | step 12 | 20 modules, Q_B 0.41 vs 0.35 for Louvain; triage 75% vs 66% for organ systems |
| **Confidence interval** | the plausible range of a measured percentage | step 5 | 85.2% (84.6-85.9%) |

## The simulated patient (step 5)
Take a disease, keep only some of its symptoms (at least 2), add 2 wrong symptoms, add the general parent terms, then rank all 12,866 diseases by similarity. If the true disease is among the top 10 it counts as found.
- Random 25% of symptoms known: found for 85.2% (top-1: 49.8%).
- Only the 25% most COMMON: 21.7%. Only the 25% most SPECIFIC: 98.1%.
So rare symptoms do almost all the work; hubs do almost none. (In an airline network it is the opposite: hubs are what matters.)

## What the paper says in one paragraph
Diagnosability depends on where a disease sits in the disease-symptom network. Rare symptoms carry almost all the information. Look-alike diseases merge into one giant cluster earlier than random networks with the same degrees, about one disease in eight has a near-identical twin, and network features predict which diseases are missed. Similarity also predicts shared genes, and Joubert syndrome shows separate subtypes with pervasive gene pleiotropy.

## What this is NOT
Not a clinical tool; patients are simulated from curated descriptions, so real-world accuracy would be lower.
