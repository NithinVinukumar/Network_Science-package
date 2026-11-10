"""
reporting.py - turns the numbers saved by every step into results/summary.md
============================================================================
Reads results/step1.json ... step8.json (written by the steps), so numbers in the summary always match the figures and the paper.
Each section ends with a one-line "Meaning". Copy numbers from here into your report and slides.
"""
import datetime
import json


def _load(cfg, n):
    f = cfg.res_dir / f"step{n}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


def pct(x, nd=0):
    return "n/a" if x is None else f"{100 * x:.{nd}f}%"


def write_summary(cfg, failures, quick):
    L = []
    add = L.append
    add("# Results summary (auto-generated)\n")
    add(f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} | seed {cfg.seed}\n")
    if quick:
        add("> **QUICK MODE: reduced sizes. Numbers are noisier; re-run without --quick before reporting.**\n")
    if failures:
        add(f"**Steps that failed:** {', '.join(failures)}\n")

    s = _load(cfg, 1)
    if s:
        t = s["tail_exponent"]
        add("## Step 1: network structure")
        add(f"- {s['diseases']} diseases ({', '.join(f'{k} {v}' for k, v in s['sources'].items())}), {s['symptom_terms']} symptom terms, {s['direct_links']} direct links; median {s['symptoms_per_disease_median']:.0f} symptoms per disease.")
        add(f"- Hubs: {', '.join(f'{n} ({d})' for n, d in s['top_hubs'][:4])}; {pct(s['share_symptoms_in_one_disease'])} of used symptoms occur in one disease only; tail exponent {t['10']['gamma']:.2f} (k>=10), {t['30']['gamma']:.2f} (k>=30).")
        add(f"- k-core: innermost k={s['kcore_max']} with {s['kcore_diseases']} diseases and {s['kcore_symptoms']} symptoms ({', '.join(s['kcore_top_symptoms'][:5])}).")
        add(f"- Assortativity {s['assortativity']:.3f}; friendship paradox: symptoms of an average disease are in {s['avg_neighbour_degree_of_disease']:.0f} diseases vs {s['avg_diseases_per_symptom']:.0f} for an average symptom (x{s['friendship_ratio']:.0f}).")
        add("- **Meaning:** a few symptoms are huge hubs and most are rare; the dense core is made of general neurodevelopmental symptoms.\n")
    s = _load(cfg, 2)
    if s:
        tw, g = s["twins"]["by_threshold"], s["gene_validation"]
        add("## Step 2: similarity network, twins, genes")
        add(f"- Twins: {tw['0.8']['diseases']} diseases ({pct(tw['0.8']['share'], 1)}) have a look-alike with similarity >= 0.8; {tw['0.9']['diseases']} ({pct(tw['0.9']['share'], 1)}) >= 0.9; {pct(1 - tw['0.8']['cross_database_share'])} of the 0.8-twins are inside one database.")
        add("- Family cohesion (within / random): " + "; ".join(f"{k} {v['ratio']:.1f}x (AUC {v['auc']:.2f})" for k, v in s["families"].items()) + ".")
        add(f"- Gene layer: similarity predicts shared genes with AUC {g['auc']:.3f} (95% {g['auc_ci'][0]:.3f}-{g['auc_ci'][1]:.3f}; {g['pairs_sharing_gene']} gene-sharing pairs vs equally many random pairs); nearest look-alike shares a gene for {pct(g['nearest_lookalike_shares_gene'])} of diseases (random pairs {pct(g['base_rate'], 2)}).")
        add("- **Meaning:** similar symptoms mean similar genes, so the symptom network captures real biology; about one disease in eight has a near-identical twin.\n")
    s = _load(cfg, 3)
    if s:
        add("## Step 3: communities")
        add(f"- {s['n_communities']} communities, modularity {s['modularity']:.2f}, NMI with organ systems {s['nmi_with_organ_systems']:.2f}, stability NMI {sum(s['stability_nmi']) / len(s['stability_nmi']):.2f}.")
        add("- Largest communities: " + "; ".join(f"#{c['community']} n={c['size']} {c['main_organ'].replace('Abnormality of the ', '')} {pct(c['purity'])}" for c in s["top_communities"][:5]) + ".")
        add("- Bridge diseases: " + "; ".join(f"{n} ({p:.2f})" for n, p in s["bridge_diseases"][:5]) + ".")
        add("- **Meaning:** diseases form clear families that follow, but do not simply copy, organ systems.\n")
    s = _load(cfg, 4)
    if s:
        add("## Step 4: percolation")
        add(f"- Giant cluster reaches half of all diseases at similarity {s['giant_half_threshold_real']:.2f} (real) vs {s['giant_half_threshold_null']:.2f} (random null, {s['n_null']} networks); second-cluster peak at {s['critical_threshold_real']} (real) vs {s['critical_threshold_null']} (null).")
        add("- Giant share by threshold (real / null): " + "; ".join(f"{r['threshold']}: {pct(r['giant'])} / {pct(n)}" for r, n in zip(s["real"], s["null_mean_giant"])) + ".")
        add("- **Meaning:** real disease profiles merge into one giant cluster earlier than random networks with the same symptom counts: related diseases share rare symptoms.\n")
    s = _load(cfg, 5)
    if s:
        m = s["main"]
        add("## Step 5: robustness of diagnosis")
        add(f"- ({s['n_main']} diseases) top-10 hit rate with k% of symptoms known (+2 wrong): " + "; ".join(f"{k.replace('random_', '')}%: {pct(v['top10']['rate'])}" for k, v in m.items() if k.startswith("random")) + ".")
        add(f"- Only the most common 25%: {pct(m['common_25']['top10']['rate'])}; only the most specific 25%: {pct(m['specific_25']['top10']['rate'])}. Top-1 at 25% random: {pct(m['random_25']['top1']['rate'])}.")
        add(f"- Wrong symptoms (0/2/5): " + " / ".join(pct(s['noise'][k]['top10']['rate']) for k in ('0', '2', '5')) + f". OMIM only: {pct(s['omim_only_25']['top10']['rate'])}.")
        add("- **Meaning:** rare symptoms carry almost all the diagnostic information; common (hub) symptoms almost none. Patients are simulated, so real accuracy is lower.\n")
    s = _load(cfg, 6)
    if s:
        add("## Step 6: diagnosability map")
        add(f"- Miss rate {pct(s['miss_rate'])}; combined AUC {s['auc_logistic']:.3f} (logistic), {s['auc_boosting']:.3f} (boosting).")
        add("- Single predictors (AUC): " + "; ".join(f"{k} {v['auc']:.2f}" for k, v in s["univariate"].items()) + ".")
        add("- Top-10 success by number of listed symptoms (U-shaped; the simulation always keeps at least 2 symptoms): " + "; ".join(f"{k}: {pct(v['top10'])} (n={v['n']})" for k, v in s["by_symptom_count"].items()) + f"; median symptoms of the 200 hardest diseases {s['median_symptoms_hardest200']:.0f} vs {s['median_symptoms_all']:.0f} overall.")
        add("- Top-10 success by twin similarity: " + "; ".join(f"{k}: {pct(v['top10'])} (n={v['n']})" for k, v in s["by_twin_similarity"].items()) + ".")
        add("- Success by organ system (lowest): " + "; ".join(f"{r['organ'].replace('Abnormality of the ', '')} {pct(r['top10'])}" for r in s["by_organ"][:3]) + ".")
        add("- **Meaning:** diagnostic difficulty is predictable from network structure (sparse annotation, rare-symptom share, look-alikes).\n")
    s = _load(cfg, 7)
    if s:
        add("## Step 7: link prediction")
        add(f"- Recall of hidden symptoms in the top {s['top_k']}: network {pct(s['recall_network'])} vs popularity baseline {pct(s['recall_baseline'])} (x{s['gain']:.1f}); n={s['diseases']}.")
        add("- **Meaning:** look-alike diseases can suggest missing symptoms, pointing curators to annotation gaps.\n")
    s = _load(cfg, 8)
    if s:
        add("## Step 8: Joubert syndrome case study")
        add(f"- {s['subtypes']} subtypes, {s['genes']} genes, {s['symptoms_used']} symptoms; core: " + ", ".join(f"{n} ({pct(v)})" for n, v in s["core_symptoms"][:4]) + f"; {pct(s['share_reported_by_one_subtype'])} of symptoms in one subtype only.")
        add(f"- Nestedness NODF {s['nodf_real']:.1f} vs null {s['nodf_null_mean']:.1f} +/- {s['nodf_null_sd']:.1f} (Z = {s['nodf_z']:.1f}, {s['nodf_nulls']} nulls).")
        add("- Same clinical form vs random genes: " + "; ".join(f"{k} {v['within']:.2f} vs {v['random_mean']:.2f} (p={v['p_value']:.3f})" for k, v in s["form_alignment"].items()) + ".")
        add(f"- Pleiotropy: {s['genes_only_joubert']} genes cause only Joubert; median {s['median_diseases_per_gene']:.0f} diseases per gene; {s['ego_diseases']} diseases in {s['ego_communities']} communities.")
        add("- **Meaning:** no evidence that Joubert subtypes form a nested severity ladder (the gap to the null is statistically clear but under half a NODF point); genes behind the same clinical form tend to look alike; every Joubert gene also causes other ciliopathies.\n")
    s = _load(cfg, 9)
    if s:
        sc = s["scenarios"]
        add("## Step 9: ranking algorithms compared")
        add(f"- ({s['patients']} simulated patients, top-10 success) random 25%: IC-weighted cosine {pct(sc['random']['cosine']['top10'])}, Jaccard {pct(sc['random']['jaccard']['top10'])}, random walk with restart {pct(sc['random']['rwr']['top10'])}.")
        add(f"- Only specific 25%: {pct(sc['specific']['cosine']['top10'])} / {pct(sc['specific']['jaccard']['top10'])} / {pct(sc['specific']['rwr']['top10'])}; only common 25%: {pct(sc['common']['cosine']['top10'])} / {pct(sc['common']['jaccard']['top10'])} / {pct(sc['common']['rwr']['top10'])}.")
        add("- **Meaning:** weighting symptoms by rarity beats plain overlap and a network diffusion (random walk with restart) when rare symptoms are known; with only common symptoms plain overlap is slightly better but every method fails; the kind of symptom matters more than the algorithm.\n")
    s = _load(cfg, 10)
    if s:
        st_ = s["strategies"]
        add("## Step 10: adaptive questioning (which symptoms to ask)")
        add(f"- Greedy best-split questions identify a disease uniquely in {st_['greedy']['mean_questions']:.1f} questions on average (binary-search limit {s['min_questions_binary_search']:.1f}); {pct(st_['greedy']['identified_share'])} of {st_['greedy']['targets']} diseases are identified, median {st_['greedy']['median_left_after_10']:.0f} candidates left after 10 questions.")
        add(f"- Always asking the most common symptom: {st_['hubs']['mean_questions']:.1f} questions, {pct(st_['hubs']['identified_share'])} identified within {st_['hubs']['cap']}; rarest symptom first: {pct(st_['rare']['identified_share'])}; random: {pct(st_['random']['identified_share'])} (within {st_['rare']['cap']}).")
        add(f"- Exact twins (identical symptom sets): {s['exact_twin_diseases']} diseases ({pct(s['exact_twin_share'], 1)}); no question can separate them.")
        add("- **Meaning:** the best question splits the candidates about half and half, not a hub and not a rare symptom; diseases can be found in about 14 questions unless they have an exact twin. Idealised: answers are assumed perfect and complete.\n")
    s = _load(cfg, 11)
    if s:
        n, p, Q = s["conditions"]["noisy answers"], s["conditions"]["perfect answers"], str(s["questions"])
        add("## Step 11: noise-robust adaptive questioning")
        add(f"- ({s['targets']} diseases, {Q} questions, patient misses {pct(s['miss'])} of the symptoms the disease has and invents {pct(s['invent'])} of the others) true disease ranked FIRST: robust information gain {pct(n['soft_ig']['top1_at'][Q])} (top 5: {pct(n['soft_ig']['top5_at'][Q])}); hard elimination (step 10) {pct(n['hard']['top1_at'][Q])}; most common symptom first {pct(n['soft_hubs']['top1_at'][Q])}; random {pct(n['soft_random']['top1_at'][Q])}.")
        add(f"- With perfect answers: robust information gain {pct(p['soft_ig']['top1_at'][Q])}, hard elimination {pct(p['hard']['top1_at'][Q])}.")
        add("- **Meaning:** one wrong answer removes the true disease under hard elimination; keeping a probability for every disease and asking the question with the largest information gain survives realistic errors. Ties (identical diseases) count as failures; the error rates are given to the model.\n")
    s = _load(cfg, 12)
    if s:
        add("## Step 12: syndromic modules of diseases and symptoms (BRIM)")
        add(f"- {s['modules']} modules; bipartite modularity Q_B: BRIM {s['Q_B']['BRIM modules']:.3f}, Louvain communities {s['Q_B']['Louvain communities']:.3f}, organ systems {s['Q_B']['organ systems']:.3f}, random {s['Q_B']['random groups']:.3f} (Q_B by number of modules: " + ", ".join(f"{k}: {v:.3f}" for k, v in s["modularity_by_c"].items()) + ").")
        add(f"- NMI with organ systems: BRIM {s['nmi_with_organ']['BRIM modules']:.2f}, Louvain {s['nmi_with_organ']['Louvain communities']:.2f}; share-a-gene rate inside groups: BRIM {pct(s['gene_share_rate']['BRIM modules'], 2)}, Louvain {pct(s['gene_share_rate']['Louvain communities'], 2)}, organ systems {pct(s['gene_share_rate']['organ systems'], 2)}, random {pct(s['gene_share_rate']['random groups'], 2)}.")
        add(f"- Triage (25% of the symptoms + 2 wrong): BRIM module {pct(s['triage_accuracy']['BRIM modules'])} vs organ system {pct(s['triage_accuracy']['organ systems'])} (chance {pct(s['triage_chance'])}, always the largest module {pct(s['triage_largest_module'])}).")
        add("- Largest modules: " + "; ".join(f"#{m['module']} {m['main_organ']} {pct(m['organ_share'])} ({', '.join(m['signature'][:3])})" for m in s["largest_modules"][:4]) + ".")
        add("- **Meaning:** BRIM groups diseases and symptoms together, so every module has signature symptoms, and it scores higher bipartite modularity (it optimises it); Louvain's communities are closer to organ systems and share genes more often; modules give a slightly better triage than organ systems.\n")
    add("## Honest-reporting checklist")
    add("- Annotations describe diseases (curated from the literature), not patients; they give no prevalence.")
    add("- Simulated patients are built from each disease's own annotation, so accuracy is optimistic.")
    add("- OMIM and Orphanet sometimes list the same disease twice, which inflates twin counts (see step 2 and the OMIM-only check in step 5).")
    add("- Gene-sharing pairs are enriched in related subtypes of one disease; the AUC measures agreement, not causation.")
    add("- Percolation null = random network with the same symptom counts per disease and diseases per symptom (stub matching).")
    add("- Not a clinical tool.")
    (cfg.res_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nWrote {cfg.res_dir / 'summary.md'}")
