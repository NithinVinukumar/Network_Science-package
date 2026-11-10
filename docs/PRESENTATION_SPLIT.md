# Presentation split for two speakers (15 minutes)

**Speaker A** = the person who built the project. **Speaker B** = the teammate.
Speaker B takes the 5 easiest steps (1, 2, 5, 7, 9) and the 2 easiest live options (4 and 3).

## Timeline
| Time | Who | What |
|---|---|---|
| 0:00-1:30 | A | Introduction |
| 1:30-2:30 | A | The dataset |
| 2:30-5:00 | A | How the graph is built (bipartite, then projection) |
| 5:00-8:30 | **B** | Steps 1, 2, 5, 7, 9 (about 40 seconds each) |
| 8:30-10:00 | A | Steps 3, 4, 6, 8 (one line each; show the percolation figure) |
| 10:00-10:30 | **B** | LIVE option 4: symptom lookup |
| 10:30-11:40 | A | LIVE option 1: diagnose from typed findings |
| 11:40-12:20 | A | LIVE option 2: stress test |
| 12:20-13:00 | **B** | LIVE option 3: twin check |
| 13:00-13:30 | A | LIVE option 5: compare algorithms |
| 13:30-14:00 | A | Limits and conclusion |
| 14:00-15:00 | both | Questions |

Handover lines
- A to B (after the graph): "Now B will show what we found when we studied this network, starting with its shape."
- B to A (after step 9): "A will now explain the communities and the percolation test."
- A to B (live): "For the live demo, B starts with a symptom lookup."
- B to A: "A will now use the system like a doctor."
- A to B: "B will check twins."
- B to A: "And A finishes by comparing three ranking algorithms."

---
# SPEAKER B: the easy parts

## Steps (5:00-8:30). Show figures from the figures/ folder.

**Step 1: the shape of the network** (show 1_network_structure.png, left panel)
"Before using the network we looked at its shape. Some symptoms are huge hubs: global developmental delay is in 2,583 diseases and seizure in 2,544. But a quarter of all symptoms appear in only one disease. So a few symptoms are everywhere and most are rare. A symptom that is everywhere tells a doctor very little."

**Step 2: look-alikes and twins** (show 2_similarity.png)
"For each disease we found its closest look-alike. 1,537 diseases, which is 11.9%, have a twin with almost the same symptoms. Known disease families are 3 to 8 times more alike than random diseases. We never used genes to build the similarity, but diseases that share a gene still look more alike: the score is 0.845, where 0.5 would be luck."

**Step 5: the simulated doctor** (show 5_robustness.png)
"We pretended to be a doctor who knows only some of a patient's symptoms. We keep part of a disease's symptoms, add two wrong ones, rank all 12,866 diseases, and check if the true disease is in the top 10. With a random quarter of the symptoms, 85% are found. With only the most common symptoms, 22%. With only the most specific symptoms, 98%."

**Step 7: missing symptoms** (show 7_link_prediction.png)
"We hid 20% of a disease's symptoms and let its look-alike diseases suggest them. The network found 37.7% of the hidden symptoms, while guessing the most common symptoms found only 10.7%. So it could point to gaps in the database."

**Step 9: which ranking method is best** (show 9_algorithms.png)
"We compared three methods: our rarity-weighted similarity, plain overlap, and a PageRank-style random walk. With a random quarter of the symptoms, the true disease is in the top 10 for 84%, 70% and 52%. One exception: with only common symptoms, plain overlap is slightly better, but all methods fail."

## Live option 4: symptom lookup (10:00-10:30)
Say: "This shows whether a symptom is a hub or rare." Type: `4`, then `seizure`.
Result: Seizure is in 3,129 diseases (information content 1.41); Motor seizure 579; Generalized non-motor (absence) seizure 212 (4.11).
Say: "The more specific the symptom, the fewer diseases have it and the more it tells us."
Optional: type `4`, then `molar tooth`: Molar tooth sign on MRI is in only 57 diseases, information content 5.42.

## Live option 3: twin check (12:20-13:00)
Type: `3`, then press Enter twice (Porokeratosis 1 and 7).
Result: similarity 1.00, genes PMVK vs MVD, verdict TWINS.
Say: "These two diseases have identical symptoms but different genes, so only a gene test can separate them."
Then type `3`, `Joubert syndrome 6`, `Joubert syndrome 1`: similarity 0.44, genes TMEM67 vs INPP5E, "not twins".
Say: "Here the symptoms differ, so a targeted question can separate them. About 1 in 8 diseases has a twin."

## Numbers for B to remember
2,583 | 25% | 11.9% (1,537) | 3 to 8 times | 0.845 | 85% / 22% / 98% | 37.7% vs 10.7% | 84% / 70% / 52%

## Questions B can answer
Hubs and rare symptoms, twins, the doctor test, missing symptoms, the three methods.
For anything else say: "A built that part, so A will answer."
Pass to A: Louvain and modularity, percolation and the null model, k-core, AUC details, nestedness, how the graph is built.

---
# SPEAKER A: the rest

## Introduction (0:00-1:30)
"Our project is about rare diseases. There are thousands, each seen rarely, and a doctor usually knows only a few symptoms of a patient. Some symptoms help a lot, like a very unusual brain-scan sign; others help almost nothing, like developmental delay, which is in about 2,900 diseases. Our question: if a doctor knows only some symptoms, which diseases can still be found, and why? We built a network of 12,866 rare diseases and their symptoms. Our main result: rare symptoms find the disease 98% of the time, and common ones only 22%."

## Dataset (1:30-2:30)
"We used the Human Phenotype Ontology, a free public database. It has three files: the main table of which symptoms were recorded for each disease, a dictionary and family tree of symptoms, and which genes are linked to each disease. After cleaning we have 12,866 diseases, 12,732 symptom terms and 267,579 disease-symptom pairs. It is curated from the medical literature, so it is not patient data."

## How the graph is built (2:30-5:00)  (draw diseases on the left, symptoms on the right)
"Each disease is a dot on the left, each symptom a dot on the right, and a line means the disease has the symptom. This is a bipartite network. To compare diseases we project it onto the diseases: two diseases are linked by how many symptoms they share. But we weight each symptom by how rare it is: information content is minus the log of the share of diseases that have it, so developmental delay scores about 1.5 and a symptom in only one disease about 9.5. Multiplying the weighted table by itself gives the similarity between every pair. For example Joubert syndrome 6 and 4 come out at 0.67, mainly because they share very rare brain symptoms. We keep each disease's 10 closest neighbours, and that is the disease-to-disease network."

## Steps 3, 4, 6, 8 (8:30-10:00)
- Step 3 (communities): "Using the Louvain algorithm we found 32 families of diseases, with modularity 0.79, and some are 95% nervous-system diseases."
- Step 4 (percolation, show 4_percolation.png): "We lower the similarity bar and watch diseases merge into one giant cluster. In the real network half of all diseases join at similarity 0.45; in a random network with the same counts it takes 0.34. So related diseases share rare symptoms."
- Step 6 (hard diseases): "15% of diseases were missed, and the network predicts which ones with an AUC of 0.77 to 0.83. Diseases with a twin are found 80% of the time, against 92% without."
- Step 8 (Joubert case study): "Joubert syndrome has 35 subtypes. They are not a ladder from mild to severe, and every one of its genes also causes other diseases."

## Live (10:30-11:40, 11:40-12:20, 13:00-13:30)
**Option 1:** type `1`, then `molar tooth sign, ataxia, nystagmus`. Say: "Three findings narrow 12,866 diseases to the Joubert family; the program shows each finding's information content and suggests the next question." Type `y` (Retinopathy): point at the Changes panel. Then `n`. Press Enter.
**Option 2:** type `2`, `Joubert syndrome 6`, Enter. Say: "Same disease, same 5 symptoms: only common symptoms rank it 85th, only specific symptoms rank it first."
**Option 5:** type `5`, then `molar tooth sign, ataxia, nystagmus`. Say: "Ours finds the Joubert family; plain overlap returns tiny diseases; the random walk finds the related family but less precisely."

## Conclusion and limits (13:30-14:00)
"In a network of 12,866 rare diseases, rare symptoms carry almost all the diagnostic information and common symptoms almost none, and about 1 in 8 diseases has a twin that symptoms cannot separate. The patients are simulated from the database, so real accuracy would be lower, and this is a research analysis, not a diagnostic tool."
