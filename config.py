"""
config.py - every adjustable setting in one place
=================================================
    cfg = Settings()          # full-quality run
    cfg = Settings.quick()    # fast run for testing (noisier; do not report those numbers)
"""
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent


@dataclass
class Settings:
    data_dir: Path = ROOT / "data"
    fig_dir: Path = ROOT / "figures"
    res_dir: Path = ROOT / "results"
    seed: int = 42                          # all randomness derives from this

    # ---- similarity network (step 2, 3) ---------------------------------------------
    twin_thresholds: Tuple[float, ...] = (0.8, 0.9)
    knn: int = 10                           # nearest look-alikes per disease used for communities
    gene_pairs: int = 150_000               # random disease pairs for the gene-sharing check

    # ---- percolation (step 4) ---------------------------------------------------------
    perc_thresholds: List[float] = field(default_factory=lambda: [0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3])
    n_null: int = 5

    # ---- diagnosis simulation (step 5, 6) ----------------------------------------------
    keep_fractions: Tuple[float, ...] = (0.25, 0.50)
    noise_symptoms: int = 2                 # wrong symptoms added to every simulated patient
    main_diseases: Optional[int] = None     # None = all diseases in the main diagnosis scenarios
    sweep_diseases: int = 2000              # diseases used for the removal sweeps
    sweep_fractions: Tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 1.0)

    # ---- ranking algorithms compared (step 9) ------------------------------------------
    algo_diseases: int = 400                # simulated patients per scenario
    rwr_continue: float = 0.5               # random walk with restart: probability of continuing the walk
    rwr_iters: int = 6

    # ---- adaptive questioning (step 10) -------------------------------------------------
    q_targets: int = 1000                   # diseases to identify with the greedy strategy
    q_baseline_targets: int = 150           # diseases for the weak strategies (hubs / rare / random)
    q_cap: int = 60
    q_cap_baseline: int = 25

    # ---- robust adaptive questioning (step 11) ------------------------------------------
    rq_targets: int = 150
    rq_questions: int = 30
    rq_miss: float = 0.20                   # the patient fails to report a symptom the disease has
    rq_invent: float = 0.02                 # the patient reports a symptom the disease does not have

    # ---- bipartite modules, BRIM (step 12) --------------------------------------------------
    brim_c_values: Tuple[int, ...] = (10, 20, 30, 40)   # Q_B is scanned over these numbers of modules (reported for transparency)
    brim_c_main: int = 20                                # the modules used everywhere else: comparable to the ~22 organ systems
    brim_restarts: int = 12
    module_triage_targets: int = 1500
    module_gene_pairs: int = 60000

    # ---- link prediction (step 7) -------------------------------------------------------
    lp_diseases: int = 1500
    lp_hide: float = 0.2
    lp_neighbours: int = 50
    lp_top: int = 20

    # ---- Joubert case study (step 8) ------------------------------------------------------
    nodf_nulls: int = 100

    @classmethod
    def quick(cls) -> "Settings":
        return replace(cls(), perc_thresholds=[0.8, 0.6, 0.5, 0.4, 0.3], n_null=2, keep_fractions=(0.25, 0.50),
                       main_diseases=1500, sweep_diseases=500, algo_diseases=100, q_targets=100, q_baseline_targets=30, rq_targets=30, rq_questions=20, brim_c_values=(20,), brim_restarts=3, module_triage_targets=200, module_gene_pairs=8000, sweep_fractions=(0.10, 0.25, 0.5), lp_diseases=200, nodf_nulls=10, gene_pairs=20_000)
