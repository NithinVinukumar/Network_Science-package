# Results summary (auto-generated)

Generated 2026-10-28 16:30 | seed 42

## Step 1: network structure
- 12866 diseases (DECIPHER 47, OMIM 8464, ORPHA 4355), 12732 symptom terms, 267579 direct links; median 15 symptoms per disease.
- Hubs: Global developmental delay (2583), Seizure (2544), Intellectual disability (2480), Short stature (1875); 25% of used symptoms occur in one disease only; tail exponent 1.78 (k>=10), 2.00 (k>=30).
- k-core: innermost k=36 with 330 diseases and 196 symptoms (Global developmental delay, Seizure, Intellectual disability, Short stature, Hypotonia).
- Assortativity -0.141; friendship paradox: symptoms of an average disease are in 310 diseases vs 23 for an average symptom (x13).
- **Meaning:** a few symptoms are huge hubs and most are rare; the dense core is made of general neurodevelopmental symptoms.

## Step 2: similarity network, twins, genes
- Twins: 1537 diseases (11.9%) have a look-alike with similarity >= 0.8; 876 (6.8%) >= 0.9; 89% of the 0.8-twins are inside one database.
- Family cohesion (within / random): Ehlers-Danlos 6.0x (AUC 0.99); Marfan / Loeys-Dietz 6.8x (AUC 0.99); RASopathies 5.1x (AUC 0.97); Joubert 6.0x (AUC 0.98); Bardet-Biedl 7.5x (AUC 0.99); Lysosomal (Gaucher, Fabry, Pompe, MPS) 3.4x (AUC 0.86).
- Gene layer: similarity predicts shared genes with AUC 0.845 (95% 0.842-0.848; 19255 gene-sharing pairs vs equally many random pairs); nearest look-alike shares a gene for 30% of diseases (random pairs 0.05%).
- **Meaning:** similar symptoms mean similar genes, so the symptom network captures real biology; about one disease in eight has a near-identical twin.

## Step 3: communities
- 32 communities, modularity 0.79, NMI with organ systems 0.41, stability NMI 0.81.
- Largest communities: #0 n=1488 Abnormality of head or neck 49%; #1 n=1229 musculoskeletal system 74%; #2 n=1015 immune system 45%; #3 n=883 nervous system 95%; #4 n=771 nervous system 93%.
- Bridge diseases: Wolfram syndrome (0.86); Malouf syndrome (0.86); Arts syndrome (0.86); Encephalocraniocutaneous lipomatosis (0.86); Alpha-mannosidosis, adult form (0.86).
- **Meaning:** diseases form clear families that follow, but do not simply copy, organ systems.

## Step 4: percolation
- Giant cluster reaches half of all diseases at similarity 0.45 (real) vs 0.34 (random null, 5 networks); second-cluster peak at 0.6 (real) vs 0.55 (null).
- Giant share by threshold (real / null): 0.9: 0% / 0%; 0.8: 1% / 0%; 0.7: 1% / 0%; 0.6: 3% / 0%; 0.55: 12% / 2%; 0.5: 29% / 8%; 0.45: 51% / 17%; 0.4: 72% / 29%; 0.35: 90% / 46%; 0.3: 98% / 66%.
- **Meaning:** real disease profiles merge into one giant cluster earlier than random networks with the same symptom counts: related diseases share rare symptoms.

## Step 5: robustness of diagnosis
- (12866 diseases) top-10 hit rate with k% of symptoms known (+2 wrong): 25%: 85%; 50%: 96%.
- Only the most common 25%: 22%; only the most specific 25%: 98%. Top-1 at 25% random: 50%.
- Wrong symptoms (0/2/5): 94% / 85% / 77%. OMIM only: 86%.
- **Meaning:** rare symptoms carry almost all the diagnostic information; common (hub) symptoms almost none. Patients are simulated, so real accuracy is lower.

## Step 6: diagnosability map
- Miss rate 15%; combined AUC 0.772 (logistic), 0.828 (boosting).
- Single predictors (AUC): number of symptoms listed 0.67; mean information content of its symptoms 0.64; share of rare (specific) symptoms 0.62; similarity to nearest look-alike (twin) 0.60; number of look-alikes (similarity >= 0.5) 0.60; number of known genes 0.54.
- Top-10 success by number of listed symptoms (U-shaped; the simulation always keeps at least 2 symptoms): 1-3: 89% (n=1754); 4-6: 78% (n=1466); 7-10: 60% (n=1546); 11-20: 82% (n=3144); 21+: 96% (n=4956); median symptoms of the 200 hardest diseases 7 vs 15 overall.
- Top-10 success by twin similarity: 0-0.3: 92% (n=241); 0.3-0.5: 89% (n=5612); 0.5-0.8: 82% (n=5476); 0.8-1.0: 80% (n=1537).
- Success by organ system (lowest): Growth abnormality 76%; eye 77%; Abnormality of limbs 77%.
- **Meaning:** diagnostic difficulty is predictable from network structure (sparse annotation, rare-symptom share, look-alikes).

## Step 7: link prediction
- Recall of hidden symptoms in the top 20: network 38% vs popularity baseline 11% (x3.5); n=1500.
- **Meaning:** look-alike diseases can suggest missing symptoms, pointing curators to annotation gaps.

## Step 8: Joubert syndrome case study
- 35 subtypes, 35 genes, 276 symptoms; core: Molar tooth sign on MRI (89%), Global developmental delay (89%), Oculomotor apraxia (74%), Ataxia (54%); 60% of symptoms in one subtype only.
- Nestedness NODF 12.9 vs null 13.3 +/- 0.1 (Z = -4.5, 100 nulls).
- Same clinical form vs random genes: hepatic 0.40 vs 0.26 (p=0.013); ocular 0.30 vs 0.26 (p=0.187); renal 0.37 vs 0.26 (p=0.003); oculorenal 0.35 vs 0.26 (p=0.014).
- Pleiotropy: 0 genes cause only Joubert; median 4 diseases per gene; 97 diseases in 6 communities.
- **Meaning:** no evidence that Joubert subtypes form a nested severity ladder (the gap to the null is statistically clear but under half a NODF point); genes behind the same clinical form tend to look alike; every Joubert gene also causes other ciliopathies.

## Step 9: ranking algorithms compared
- (400 simulated patients, top-10 success) random 25%: IC-weighted cosine 84%, Jaccard 70%, random walk with restart 52%.
- Only specific 25%: 96% / 83% / 80%; only common 25%: 23% / 30% / 6%.
- **Meaning:** weighting symptoms by rarity beats plain overlap and a network diffusion (random walk with restart) when rare symptoms are known; with only common symptoms plain overlap is slightly better but every method fails; the kind of symptom matters more than the algorithm.

## Step 10: adaptive questioning (which symptoms to ask)
- Greedy best-split questions identify a disease uniquely in 13.9 questions on average (binary-search limit 13.7); 96% of 1000 diseases are identified, median 13 candidates left after 10 questions.
- Always asking the most common symptom: 19.6 questions, 70% identified within 25; rarest symptom first: 0%; random: 3% (within 25).
- Exact twins (identical symptom sets): 541 diseases (4.2%); no question can separate them.
- **Meaning:** the best question splits the candidates about half and half, not a hub and not a rare symptom; diseases can be found in about 14 questions unless they have an exact twin. Idealised: answers are assumed perfect and complete.

## Step 11: noise-robust adaptive questioning
- (150 diseases, 30 questions, patient misses 20% of the symptoms the disease has and invents 2% of the others) true disease ranked FIRST: robust information gain 79% (top 5: 88%); hard elimination (step 10) 19%; most common symptom first 29%; random 8%.
- With perfect answers: robust information gain 97%, hard elimination 97%.
- **Meaning:** one wrong answer removes the true disease under hard elimination; keeping a probability for every disease and asking the question with the largest information gain survives realistic errors. Ties (identical diseases) count as failures; the error rates are given to the model.

## Step 12: syndromic modules of diseases and symptoms (BRIM)
- 20 modules; bipartite modularity Q_B: BRIM 0.408, Louvain communities 0.349, organ systems 0.356, random 0.075 (Q_B by number of modules: 10: 0.407, 20: 0.401, 30: 0.405, 40: 0.411).
- NMI with organ systems: BRIM 0.33, Louvain 0.41; share-a-gene rate inside groups: BRIM 0.18%, Louvain 0.48%, organ systems 0.17%, random 0.06%.
- Triage (25% of the symptoms + 2 wrong): BRIM module 75% vs organ system 66% (chance 5%, always the largest module 20%).
- Largest modules: #0 musculoskeletal system 40% (Hypertelorism, Micrognathia, Cleft palate); #1 immune system 18% (Hepatomegaly, Splenomegaly, Fever); #2 nervous system 67% (Dysarthria, Babinski sign, Ataxia); #3 nervous system 72% (Global developmental delay, Seizure, Microcephaly).
- **Meaning:** BRIM groups diseases and symptoms together, so every module has signature symptoms, and it scores higher bipartite modularity (it optimises it); Louvain's communities are closer to organ systems and share genes more often; modules give a slightly better triage than organ systems.

## Honest-reporting checklist
- Annotations describe diseases (curated from the literature), not patients; they give no prevalence.
- Simulated patients are built from each disease's own annotation, so accuracy is optimistic.
- OMIM and Orphanet sometimes list the same disease twice, which inflates twin counts (see step 2 and the OMIM-only check in step 5).
- Gene-sharing pairs are enriched in related subtypes of one disease; the AUC measures agreement, not causation.
- Percolation null = random network with the same symptom counts per disease and diseases per symptom (stub matching).
- Not a clinical tool.