# Rare-disease diagnosability network

**One-line idea:** a network of 12,866 rare diseases and their symptoms, used to study which diseases are easy or hard to recognise from an incomplete symptom list, and why.

*(Data: Human Phenotype Ontology, release 2026-09-02. Research analysis on curated disease descriptions, not a clinical tool.)*

## Set up and run (3 steps)

```
pip install -r requirements.txt          # 1. install (Python 3.9+)
python -m unittest discover -s tests     # 2. check everything works (about 2 minutes, 66 tests, should end with OK)
python run_all.py                        # 3. run all twelve analyses (about 7 minutes)
```

Then open, in this order:
1. `results/summary.md` - every key number, each section ends with a "Meaning" line.
2. `figures/` - 13 pictures (explained in `docs/OUTPUT_GUIDE.md`).
3. The paper draft (see `paper/README.md`).

A fast test run: `python run_all.py --quick` (about 1 minute, noisier numbers - do not report them).
Only some steps: `python run_all.py --only 5 6` (step 6 needs step 5 to have been run once).

## Live testing for your presentation (you type, the system answers)

```
python live_test.py
```
Menu: 1 diagnose from findings you type (and answer the system's questions), 2 stress test a disease you type, 3 twin check for two diseases you type, 4 symptom lookup, 5 compare three ranking algorithms on your findings, 6 play 20 questions (the system finds a disease by asking yes/no symptom questions), 7 noisy 20 questions (the patient makes mistakes and the system recovers), 8 which syndromic module (BRIM) do your findings point to. Each option prints the NETWORK CONCEPT and the CODE THAT RUNS. Exact examples to type, with the real outputs: `docs/LIVE_DEMO.md`.
(`python demo.py` is an optional scripted 5-scene version.)

## Try your own inputs (audience questions)

```
python explore.py
```
| Menu | What it does |
|---|---|
| 1 Look up a disease | its symptoms (most informative first), nearest look-alikes, genes |
| 2 Simulate a patient | hide symptoms of a real disease, add wrong ones, see where the true disease ranks. Try `random`, `common`, `specific` |
| 3 Diagnose from YOUR symptoms | type symptom names, get a ranked list and the best NEXT question to ask |
| 4 Compare two diseases | similarity, shared and unique symptoms |

Press Enter to accept the value in [brackets]. In VS Code you can also use Run and Debug (`.vscode/launch.json`).

## What the twelve steps do

| Step | Question | Main concepts |
|---|---|---|
| 1 | What does the disease-symptom network look like? | bipartite network, heavy tail, hubs, k-core, assortativity, friendship paradox |
| 2 | Which diseases look alike? | projection, similarity network, twins, gene layer (link prediction) |
| 3 | Do diseases form families? | communities (Louvain), modularity, NMI, bridges |
| 4 | When do look-alike diseases merge into one cluster? | percolation, giant component, null model |
| 5 | How robust is diagnosis to missing and wrong symptoms? | random vs targeted removal, hubs vs periphery |
| 6 | Which diseases are hard to diagnose and why? | network features as predictors, twins, organ systems |
| 7 | Can the network predict missing symptoms? | link prediction |
| 8 | Case study: Joubert syndrome | multilayer network, pleiotropy, nestedness, permutation tests |
| 12 | Which syndromic modules group diseases AND symptoms? | Barber's bipartite modularity and the BRIM algorithm; module signature symptoms; triage compared with organ systems and Louvain |
| 11 | What if the answers are wrong? | noise-robust adaptive questioning: a probability for every disease (Bayes' rule) and the question with the largest expected information gain, compared with hard elimination (step 10) |
| 10 | Which symptoms should the doctor ask, and in what order? | greedy test-cover / identifying-code algorithm on the bipartite network ('20 questions'), compared with hubs-first, rarest-first and random |
| 9 | Which ranking algorithm finds the disease best? | similarity vs diffusion on the network: IC-weighted cosine vs Jaccard vs random walk with restart (personalised PageRank) |

## Documents
- `docs/PRESENTATION_SPLIT.md` - the 15-minute presentation split between two speakers, with a full script for each
- `docs/LIVE_DEMO.md` - the presentation script: exactly what to run, what appears, what to say
- `docs/CONCEPTS_SIMPLE.md` - every concept in plain words (start here if you are new to the project)
- `docs/FILE_GUIDE.md` - every file and what it does
- `docs/OUTPUT_GUIDE.md` - every figure and number, and what to say about it
- `docs/VIVA_QA.md` - questions your professor may ask, with answers
- `paper/README.md` - where the paper draft is

## Honest limits (say these before you are asked)
- Annotations describe diseases (curated from the literature), not patients, and give no prevalence.
- Simulated patients are built from each disease's own annotation, so accuracies are optimistic. Only relative findings should be generalised.
- OMIM and Orphanet sometimes list the same disease twice; duplicates are not merged (an OMIM-only check is included).
- The simulated doctor always knows at least 2 symptoms, which causes a U-shape in difficulty vs symptom count.
- The Joubert case study is small (35 subtypes); with multiple testing only the renal-form result stays significant.

## Troubleshooting
- `ModuleNotFoundError`: run `pip install -r requirements.txt`.
- First run of step 2 creates `results/_cache_neighbours.npz` (about 10 s); delete `results/` files to recompute.
- Data missing: `python fetch_data.py` (needs internet).
- Same seed gives the same numbers; change `seed` in `config.py` to see how much results move.
