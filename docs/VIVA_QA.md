# Questions your professor may ask

**1. In one sentence, what is your project?**
A network of 12,866 rare diseases and their symptoms, used to study which diseases are hard to diagnose from an incomplete symptom list and why.

**2. Why is this a network problem?**
Diseases and symptoms are two kinds of nodes joined by links. Network position (hub symptoms, look-alike neighbours, communities) explains diagnostic difficulty.

**3. What is bipartite, and what is a projection?**
Bipartite: two node types, links only between types (disease-symptom). Projection: a one-type network (disease-disease) where two diseases are linked by how many symptoms they share, here weighted by information content.

**4. Why weight symptoms by information content?**
A symptom in 2,583 diseases says almost nothing; one in a single disease says a lot. IC = -ln(share of diseases). Without it, common symptoms would dominate similarity.

**5. Is the degree distribution a power law?**
Heavy-tailed, not a clean power law: the fitted exponent changes with the cut-off (1.78 for k>=10, 2.00 for k>=30, 2.36 for k>=100).

**6. What is percolation here, and why a null model?**
Link diseases whose similarity is at least a threshold and lower the threshold; a giant cluster appears. Against random networks with the same symptom counts, the real network percolates earlier (half at 0.45 vs 0.34), so overlaps are structured beyond symptom popularity.

**7. How is your null model built, and what are its limits?**
Stub matching: same number of symptoms per disease and diseases per symptom, random pairing, then the same propagation. It merges about 2% duplicate links and does not preserve the ontology beyond propagation; 5 replicates.

**8. How do you know the similarity is meaningful?**
Genes were not used to build it, yet similarity separates gene-sharing disease pairs from random pairs with AUC 0.845 (interval 0.842-0.848); the nearest look-alike shares a gene for 30% of diseases.

**9. What is your main finding?**
Rare symptoms carry almost all diagnostic information: with 25% of symptoms known, only the most common give 22% top-10 success, only the most specific give 98%.

**10. Isn't that just how TF-IDF or phenotype-matching tools already work?**
Yes, weighting by rarity is established; our contribution is quantifying it across the whole network, showing the asymmetric robustness, the twin structure (11.9%), the percolation behaviour, and a diagnosability map predictable from network features.

**11. How good is the 85% figure really?**
Optimistic: simulated patients come from each disease's own annotation. Only relative findings (common vs specific, twins, predictors) should be generalised. We also report an OMIM-only check (86.2%) and wrong-symptom sensitivity (94 / 85 / 77% for 0 / 2 / 5 wrong symptoms).

**12. Why is difficulty U-shaped in the number of symptoms?**
Part is a design artifact: the simulated doctor always knows at least 2 symptoms, so diseases with 1-3 listed symptoms are almost fully known (89% success). The hardest are mid-sized entries (7-10 symptoms: 60%).

**13. Duplicates between OMIM and Orphanet?**
Not merged, no cross-reference table is bundled. 11% of 0.8-twins are cross-database; the OMIM-only check shows the main result is stable.

**14. Joubert syndrome: what did you find, and is it significant?**
No nested severity ladder (NODF 12.9 vs 13.3, Z = -4.5, but under half a NODF point); genes of the same clinical form look alike (renal p = 0.003, hepatic 0.013, oculorenal 0.014, ocular 0.187). With 4 tests the Bonferroni threshold is 0.0125, so only renal stays significant; the sample is small.

**15. What are the limitations?**
Curated annotations, not patients; no prevalence; simulated patients; duplicates; coarse percolation grid (no finite-size scaling); small Joubert sample; not a clinical tool.

**16. What would you do next?**
Merge duplicate diseases, validate on real patient phenotypes, locate the percolation transition with finite-size analysis, and add a diffusion method on the bipartite network.

**17. Why not an epidemic or cascade model?**
They do not fit a disease-symptom network naturally; we use robustness (symptom loss), percolation and link prediction instead of forcing a spreading model.

**18. How reproducible is it?**
Seed 42, 66 automated tests, full run in about 3 minutes, data bundled.

**19. Did you use any real network algorithms, or is it just analysis?**
Yes, several, and we compared them. Network algorithms: Louvain community detection, k-core decomposition, connected components for percolation, a random-walk-with-restart (personalised PageRank) ranking, a configuration-model null network (stub matching), curveball randomisation, maximum-likelihood tail fitting, participation coefficient, NODF. Our own ranking uses an information-weighted similarity projection. In step 9 we benchmarked three ranking algorithms on the same simulated patients: IC-cosine 84%, Jaccard 70%, random walk with restart 52% (top 10, random 25% of symptoms).

**20. Why does the random walk do worse?**
Hub symptoms spread the walker thinly, which helps, but random walks also tend to favour well-connected diseases, and they do not weight a rare symptom more than a common one the way information content does. It finds the right neighbourhood (for example the ciliopathy family) but ranks the exact disease less precisely.

**21. Is IC-cosine always better?**
No. With only common symptoms known, plain overlap is slightly better (30% vs 23%), because information weighting deliberately discounts exactly those symptoms; but every method fails there. The kind of symptom matters more than the algorithm.

**22. What is the new algorithm in step 10?**
Greedy adaptive questioning, the "20 questions" game on the bipartite network. A disease is hidden; we ask yes/no questions "does it have symptom X?", read the answer from the disease's symptom list, and keep only the diseases consistent with all answers. The next question is the symptom that leaves the smallest expected number of candidates, i.e. the one that about half of the remaining diseases have. In graph terms it is the test-cover (identifying-code) problem: a set of symptom nodes whose neighbourhoods uniquely identify each disease. Finding the optimal set is NP-hard, so we use the standard greedy heuristic.

**23. How good is it?**
13.9 questions on average, against a binary-search limit of log2(12,866) = 13.7. It identifies 96% of diseases; the other 4.2% (541 diseases) have a symptom set identical to another disease's, so no question can separate them. Always asking the most common symptom needs 19.6 questions and finds 70% within 25; asking the rarest symptom or random symptoms almost never succeeds.

**24. This says rare symptoms are best, but the best questions are not rare. Contradiction?**
No. Two different things. As EVIDENCE a rare symptom is worth the most (it narrows the diseases a lot). As a QUESTION to ask, the most informative is one where about half of the candidates would say yes, because then either answer halves the list (information is largest when the answer is 50/50). Hubs are asked about by almost everyone, rare symptoms by almost no one, so both split the list badly.

**25. What are the limits of step 10?**
Answers are assumed perfect and complete; real patients and real annotations are not, so real strict yes/no matching can wrongly rule out the right disease. The first questions it picks are broad categories such as "Abnormality of the nervous system" (because they split well), which is a triage-style order, not a clinical protocol.

**26. What is new in step 11, and is it really a new algorithm?**
Step 10 assumes perfect answers and crosses a disease out as soon as one answer disagrees. Step 11 keeps a probability for every disease of the bipartite network, updates it with Bayes' rule after each answer (a disease that has the symptom is answered "yes" with probability 1 - eps, one that lacks it with probability delta), and asks the symptom with the largest expected information gain. It is a noise-robust adaptive questioning method for disease-symptom networks. The ingredients (Bayesian experimental design, noisy twenty-questions games) are known in other fields; what is ours is the formulation on this network and the evaluation. Say "we propose and evaluate"; do not say "we invented a new algorithm".

**27. How much better is it?**
With simulated noise (the patient misses 20% of the symptoms the disease has and invents 2% of the others), after 30 questions the true disease is ranked first for 79% of diseases (88% in the top 5), against 19% for hard elimination, 29% for always asking the most common symptom and 8% for random questions. With perfect answers both reach 97%.

**28. Does it use network diffusion?**
In a pilot we also tried spreading the probabilities over the disease-similarity graph before ranking: it gave no clear gain (80% vs 79%, within noise), so it is not part of the method.

**29. Limits of step 11?**
The model is told the true error rates (wrong rates would cost accuracy), the noise is simulated, diseases with identical symptoms tie and count as failures, and 30 questions is a lot for a real consultation.

**30. What is the new algorithm in step 12?**
BRIM (Bipartite, Recursively Induced Modules; Barber 2007) maximises bipartite modularity, which groups diseases AND symptoms into the same modules. Start with random symptom modules; every disease moves to the module that raises modularity most; then every symptom does the same; repeat until nothing changes. Louvain (step 3) works on the one-mode projection and groups diseases only. We did not invent BRIM: it is an existing, specialised bipartite-network algorithm applied to our problem.

**31. Is BRIM better than Louvain?**
Not at everything. BRIM gives higher bipartite modularity (0.41 vs 0.35) because it optimises exactly that. Louvain's communities are closer to organ systems (NMI 0.41 vs 0.33) and share a gene more often inside groups (0.48% vs 0.18%; random 0.06%). What BRIM adds is the joint disease-and-symptom modules with signature symptoms, and a triage step.

**32. How is the triage done and how good is it?**
For a patient with 25% of a disease's symptoms plus 2 wrong ones, every module gets a score: the sum over the patient's symptoms of the log of how over-represented the symptom is in the module (log-lift, naive-Bayes style). The highest score is the predicted module. It is right for 75% of diseases (chance 5%, always the largest module 20%), against 66% when the same method predicts the organ system. An earlier, simpler scoring rule gave 61%.

**33. Why 20 modules?**
Q_B is almost flat between 10 and 40 modules (0.40 to 0.41), so the number is a choice: 20 is comparable to the 22 organ systems (a fair comparison) and gives more specific modules.

**34. What are the limits of step 12?**
BRIM starts from random modules (we take the best of 12 starts), modules are broad syndromic groups and not clean disease families (the 41 Joubert diseases are spread over three modules: 19, 17 and 5), a few modules are odd small groups of sparsely described diseases, and the triage uses simulated patients.
