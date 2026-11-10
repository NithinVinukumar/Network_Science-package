"""
tests/test_core.py - checks of the building blocks
==================================================
Run all tests:   python -m unittest discover -s tests -v
Many tests use tiny examples whose right answer can be worked out by hand, so the code is checked against mathematics.
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import explore
import hpo_data
import network_tools as nt
from config import Settings
from utils import auc_score, ccdf, tail_exponent, wilson_ci


class TestUtils(unittest.TestCase):
    def test_wilson(self):
        lo, hi = wilson_ci(50, 100)
        self.assertTrue(0.40 < lo < 0.5 < hi < 0.60)
        self.assertEqual(wilson_ci(0, 0), (0.0, 0.0))

    def test_auc(self):
        self.assertEqual(auc_score([1, 2, 3, 4], [0, 0, 1, 1]), 1.0)
        self.assertEqual(auc_score([4, 3, 2, 1], [0, 0, 1, 1]), 0.0)
        rng = np.random.default_rng(0)
        self.assertAlmostEqual(auc_score(rng.random(4000), rng.integers(0, 2, 4000)), 0.5, delta=0.04)

    def test_tail_exponent_recovers_truth(self):
        rng = np.random.default_rng(1)
        k = np.floor(10 * (1 - rng.random(30000)) ** (-1 / 1.5))          # power law, exponent 2.5
        self.assertAlmostEqual(tail_exponent(k, 10)[0], 2.5, delta=0.2)

    def test_ccdf(self):
        x, p = ccdf([1, 1, 2, 4])
        self.assertEqual(list(x), [1, 2, 4])
        self.assertEqual(list(p), [1.0, 0.5, 0.25])


class TestToyNetworks(unittest.TestCase):
    def test_propagation_and_information_content(self):
        # ontology: term0 (root) <- term1 <- term2.  Disease A has term2, disease B has term1.
        Ddir = csr_matrix(np.array([[0, 0, 1], [0, 1, 0]], float))
        Anc = csr_matrix(np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], float))
        Dp, ic, Dw = hpo_data.propagate_matrices(Ddir, Anc, 2)
        self.assertEqual(Dp.toarray().tolist(), [[1, 1, 1], [1, 1, 0]])      # A inherits term1 and root
        self.assertAlmostEqual(ic[0], 0.0)                                   # root is in every disease: no information
        self.assertAlmostEqual(ic[2], np.log(2))                             # term2 is in 1 of 2 diseases
        self.assertAlmostEqual(float(np.linalg.norm(Dw[0].toarray())), 1.0, places=5)
        self.assertAlmostEqual(float((Dw[0] @ Dw[1].T).toarray()[0, 0]), 0.0, places=6)   # share only zero-information terms

    def test_nodf_and_curveball(self):
        tri = np.tril(np.ones((8, 8), int))
        self.assertAlmostEqual(nt.nodf(tri), 100.0)                          # perfectly nested staircase
        rng = np.random.default_rng(0)
        M = (rng.random((20, 30)) < 0.2).astype(int)
        R = nt.curveball(M, rng, 500)
        self.assertTrue(np.array_equal(R.sum(0), M.sum(0)) and np.array_equal(R.sum(1), M.sum(1)))
        self.assertFalse(np.array_equal(R, M))

    def test_participation_coefficient(self):
        neigh = np.array([[1, 2, 3, 4], [0, 2, 3, 4], [0, 1, 3, 4], [0, 1, 2, 4], [0, 1, 2, 3]])
        labels = np.array([0, 0, 0, 1, 1])
        P = nt.participation_coefficient(neigh, labels)
        self.assertAlmostEqual(P[0], 1 - (0.5 ** 2 + 0.5 ** 2))             # node 0: 2 neighbours in each community

    def test_giant_stats(self):
        # 5 diseases: chain 0-1-2 (similarity 0.9, 0.6) and pair 3-4 (0.7)
        I, J, S = np.array([0, 1, 3]), np.array([1, 2, 4]), np.array([0.9, 0.6, 0.7])
        out = nt.giant_stats(I, J, S, 5, [0.95, 0.8, 0.5])
        self.assertAlmostEqual(out[0]["giant"], 1 / 5)                       # no links: all isolated
        self.assertAlmostEqual(out[1]["giant"], 2 / 5)                       # pairs 0-1 and 3-4
        self.assertAlmostEqual(out[2]["giant"], 3 / 5)                       # chain of 3 + pair of 2
        self.assertAlmostEqual(out[2]["second"], 2 / 5)


class TestRankingAlgorithms(unittest.TestCase):
    def toy(self):
        # 3 diseases x 4 symptoms (a hub symptom 0 shared by all, symptom 1 only in disease 0, 2 in diseases 1,2, 3 only in disease 2)
        import types
        B = csr_matrix(np.array([[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 1, 1]], float))
        return types.SimpleNamespace(Dprop=B)

    def test_rwr_rows_sum_to_one_and_find_the_rare_symptom_owner(self):
        d = self.toy()
        Q = csr_matrix(np.array([[0, 1, 0, 0.0], [1, 0, 0, 0.0], [0, 0, 0, 1.0]]))
        S = nt.rwr_scores(d, Q, cont=0.5, iters=6).toarray()
        self.assertTrue(np.allclose(S.sum(1), 1.0, atol=1e-5))
        self.assertEqual(int(S[0].argmax()), 0)              # symptom only in disease 0 -> disease 0
        self.assertEqual(int(S[2].argmax()), 2)              # symptom only in disease 2 -> disease 2
        self.assertLess(S[1].max(), 0.5)                     # the hub symptom spreads the walker over all diseases

    def test_jaccard_hand_computed(self):
        d = self.toy()
        Qb = csr_matrix(np.array([[1, 1, 0, 0.0]]))           # patient has symptoms 0 and 1
        J = nt.jaccard_scores(d, Qb)[0]
        self.assertAlmostEqual(J[0], 1.0)                    # same set
        self.assertAlmostEqual(J[1], 1 / 3)                  # intersection {0}, union {0,1,2}
        self.assertAlmostEqual(J[2], 1 / 4)                  # intersection {0}, union {0,1,2,3}


class TestAdaptiveQuestions(unittest.TestCase):
    def data(self, M):
        import types
        M = csr_matrix(np.array(M, float))
        return types.SimpleNamespace(n=M.shape[0], Dprop=M), M

    def test_binary_codes_are_found_by_binary_search(self):
        # 8 diseases, 3 symptoms; disease i has symptom k when bit k of i is set -> every question halves the candidates
        d, M = self.data([[(i >> k) & 1 for k in range(3)] for i in range(8)])
        for target in range(8):
            r = nt.adaptive_questions(d, target, "greedy", Dp=M)
            self.assertTrue(r["identified"])
            self.assertEqual(r["n_questions"], 3)
            self.assertEqual([s[2] for s in r["steps"]], [4, 2, 1])

    def test_identical_symptom_sets_cannot_be_separated(self):
        d, M = self.data([[1, 1, 0], [1, 1, 0], [1, 0, 1]])
        r = nt.adaptive_questions(d, 0, "greedy", Dp=M)
        self.assertTrue(r["stuck"])
        self.assertEqual(r["left"], 2)
        self.assertIsNone(nt.best_split_symptom(M, np.array([0, 1])))

    def test_greedy_picks_the_balanced_symptom(self):
        # 6 diseases: symptom 0 in all (useless), symptom 1 in 3 of 6 (balanced), symptom 2 in 1 of 6 (rare)
        M = csr_matrix(np.array([[1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]], float))
        cand = np.arange(6)
        self.assertEqual(nt.best_split_symptom(M, cand, "greedy"), 1)
        self.assertEqual(nt.best_split_symptom(M, cand, "hubs"), 1)
        self.assertEqual(nt.best_split_symptom(M, cand, "rare"), 2)


class TestRobustQuestioning(unittest.TestCase):
    def test_a_wrong_answer_lowers_but_never_removes_the_true_disease(self):
        w = np.array([0.5, 0.5], dtype=np.float32)
        col = np.array([1.0, 0.0])                         # disease 0 has the symptom, disease 1 lacks it
        w2 = nt.robust_update(w, col, False, 0.2, 0.02)    # the patient wrongly says NO although disease 0 has it
        self.assertAlmostEqual(float(w2.sum()), 1.0, places=5)
        self.assertGreater(float(w2[0]), 0.1)              # still alive (hard elimination would give 0)
        self.assertLess(float(w2[0]), float(w2[1]))        # but now less likely than the disease that lacks it
        w3 = nt.robust_update(w2, col, True, 0.2, 0.02)    # a later correct YES brings it back up
        self.assertGreater(float(w3[0]), float(w2[0]))

    def test_information_gain_prefers_the_balanced_symptom(self):
        M = csr_matrix(np.array([[1, 1, 1], [1, 1, 0], [1, 1, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]], float))
        DpT = M.T.tocsr().astype(np.float32)
        w = np.full(6, 1 / 6, dtype=np.float32)
        asked = np.zeros(3, dtype=bool)
        self.assertEqual(nt.robust_choose(DpT, w, asked, 0.01, 0.002, "ig"), 1)      # in half of the diseases
        asked[1] = True
        self.assertEqual(nt.robust_choose(DpT, w, asked, 0.01, 0.002, "ig"), 2)      # next best, never the useless symptom 0


class TestBipartiteModules(unittest.TestCase):
    def blocks(self):
        # 6 diseases x 6 symptoms: diseases 0-2 list symptoms 0-2, diseases 3-5 list symptoms 3-5 (two perfect blocks)
        M = np.zeros((6, 6))
        M[:3, :3] = 1
        M[3:, 3:] = 1
        return csr_matrix(M)

    def test_modularity_of_two_perfect_blocks_is_one_half(self):
        A = self.blocks()
        self.assertAlmostEqual(nt.bipartite_modularity(A, np.array([0, 0, 0, 1, 1, 1]), np.array([0, 0, 0, 1, 1, 1]), 2), 0.5)
        self.assertAlmostEqual(nt.bipartite_modularity(A, np.array([0, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0]), 1), 0.0)   # one module: no structure
        self.assertLess(nt.bipartite_modularity(A, np.array([0, 1, 0, 1, 0, 1]), np.array([0, 0, 1, 1, 0, 1]), 2), 0.2)         # a scrambled partition

    def test_brim_recovers_the_blocks(self):
        A = self.blocks()
        rng = np.random.default_rng(0)
        best = max((nt.brim(A, 2, rng) for _ in range(8)), key=lambda x: x[2])
        row, col, q = best
        self.assertAlmostEqual(q, 0.5)
        self.assertEqual(len(set(row[:3])), 1)
        self.assertEqual(len(set(row[3:])), 1)
        self.assertNotEqual(row[0], row[3])
        self.assertEqual(col[0], row[0])                                    # symptoms follow their diseases
        self.assertEqual(col[5], row[5])

    def test_module_scores_point_to_the_right_module(self):
        A = self.blocks()
        prof = nt.module_profile(A, np.array([0, 0, 0, 1, 1, 1]), 2)
        self.assertEqual(int(np.argmax(nt.module_scores(prof, np.array([0, 1])))), 0)
        self.assertEqual(int(np.argmax(nt.module_scores(prof, np.array([4])))), 1)

    def test_module_coverage_and_symptom_assignment(self):
        A = self.blocks()
        row = np.array([0, 0, 0, 1, 1, 1])
        cov = nt.module_coverage(A, row, 2)
        self.assertEqual(cov.tolist(), [[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]])
        self.assertEqual(nt.symptom_modules_for(A, row, 2).tolist(), [0, 0, 0, 1, 1, 1])


class TestRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Settings()
        cls.d = hpo_data.load(cls.cfg)

    def test_size(self):
        d = self.d
        self.assertEqual(d.n, 12866)
        self.assertEqual(d.t, 12732)
        self.assertEqual(int(d.Ddir.nnz), 267579)
        self.assertTrue((d.ic >= 0).all())
        self.assertEqual(float(d.ic.min()), 0.0)

    def test_propagation_contains_direct(self):
        d = self.d
        self.assertTrue((d.Dprop.toarray()[:50] >= d.Ddir.toarray()[:50]).all())

    def test_similarity_properties(self):
        d = self.d
        S = (d.Dw[:60] @ d.Dw[:60].T).toarray()
        self.assertTrue(np.allclose(S, S.T, atol=1e-5))
        self.assertTrue(np.all(S <= 1.0001) and np.all(S >= -1e-6))

    def test_find_helpers(self):
        d = self.d
        i = d.find_disease("Joubert syndrome 6")[0]
        self.assertEqual(d.disease_names[i], "Joubert syndrome 6")
        self.assertIn("TMEM67", d.genes[d.ids[i]])
        self.assertTrue(any("Molar tooth" in d.term_names[j] for j in d.find_term("molar tooth")))

    def test_full_information_patient_is_found(self):
        d = self.d
        rng = np.random.default_rng(0)
        rows = np.sort(rng.choice(d.n, 300, replace=False))
        Q = nt.make_patients(d, rows, 1.0, "random", 0, rng)
        r = nt.rank_of_true(d, Q, rows)
        self.assertGreater((r <= 3).mean(), 0.9)

    def test_rare_symptoms_beat_common_symptoms(self):
        d = self.d
        rng = np.random.default_rng(1)
        rows = np.sort(rng.choice(d.n, 300, replace=False))
        top10 = {}
        for mode in ("common", "specific"):
            Q = nt.make_patients(d, rows, 0.25, mode, 2, rng)
            top10[mode] = (nt.rank_of_true(d, Q, rows) <= 10).mean()
        self.assertGreater(top10["specific"], top10["common"] + 0.3)

    def test_null_keeps_degrees(self):
        d = self.d
        rows, cols = d.Ddir.nonzero()
        rng = np.random.default_rng(2)
        from scipy.sparse import csr_matrix as cm
        M = cm((np.ones(len(rows)), (rows, rng.permutation(cols))), shape=d.Ddir.shape)
        M.data[:] = 1.0
        self.assertAlmostEqual(M.nnz / d.Ddir.nnz, 1.0, delta=0.04)          # about 2% duplicate links are merged
        diff = np.abs(np.asarray(M.sum(1)).ravel() - np.asarray(d.Ddir.sum(1)).ravel())
        self.assertLess(diff.mean(), 0.5)

    def test_omim_only_subset(self):
        om = hpo_data.load(self.cfg, sources=["OMIM"])
        self.assertTrue(all(i.startswith("OMIM") for i in om.ids))
        self.assertLess(om.n, self.d.n)


class TestExplorer(unittest.TestCase):
    def test_ask_defaults(self):
        with mock.patch("builtins.input", return_value=""):
            self.assertEqual(explore.ask("x", 5, int), 5)
        with mock.patch("builtins.input", side_effect=EOFError):
            self.assertEqual(explore.ask("x", 0.25, float), 0.25)
        with mock.patch("builtins.input", return_value="abc"):
            self.assertEqual(explore.ask("x", 7, int), 7)

    def test_pick_disease(self):
        d = hpo_data.load(Settings())
        self.assertEqual(d.disease_names[explore.pick_disease(d, "Joubert syndrome 6")], "Joubert syndrome 6")
        self.assertIsNone(explore.pick_disease(d, "zzzz not a disease"))


if __name__ == "__main__":
    unittest.main()
