"""
tests/test_steps.py - every analysis step runs end to end (quick mode) and gives sensible results
=================================================================================================
Writes into a temporary folder, so it never touches your real results.
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hpo_data
import reporting
import step1_network
import step2_similarity
import step3_communities
import step4_percolation
import step5_robustness
import step6_diagnosability
import step7_link_prediction
import step8_joubert
import step9_algorithms
import step10_questions
import step11_robust_questions
import step12_modules
from config import Settings


class TestStepsQuick(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cfg = Settings.quick()
        cls.cfg.fig_dir = Path(cls.tmp.name) / "fig"
        cls.cfg.res_dir = Path(cls.tmp.name) / "res"
        cls.cfg.fig_dir.mkdir()
        cls.cfg.res_dir.mkdir()
        cls.d = hpo_data.load(cls.cfg)
        cls.out = {}
        with mock.patch("builtins.print"):
            for n, m in enumerate([step1_network, step2_similarity, step3_communities, step4_percolation, step5_robustness,
                                   step6_diagnosability, step7_link_prediction, step8_joubert, step9_algorithms, step10_questions, step11_robust_questions, step12_modules], 1):
                cls.out[n] = m.run(cls.d, cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_step1(self):
        s = self.out[1]
        self.assertEqual(s["diseases"], 12866)
        self.assertEqual(s["kcore_max"], 36)
        self.assertLess(s["assortativity"], 0)
        self.assertGreater(s["friendship_ratio"], 5)

    def test_step2(self):
        s = self.out[2]
        self.assertGreater(s["twins"]["by_threshold"]["0.8"]["share"], 0.08)
        self.assertGreater(s["gene_validation"]["auc"], 0.7)

    def test_step3(self):
        s = self.out[3]
        self.assertGreater(s["modularity"], 0.5)
        self.assertGreater(s["nmi_with_organ_systems"], 0.2)

    def test_step4(self):
        s = self.out[4]
        self.assertGreater(s["giant_half_threshold_real"], s["giant_half_threshold_null"])      # real percolates earlier

    def test_step5(self):
        s = self.out[5]["main"]
        self.assertGreater(s["specific_25"]["top10"]["rate"], s["common_25"]["top10"]["rate"] + 0.3)
        self.assertGreater(s["random_50"]["top10"]["rate"], s["random_25"]["top10"]["rate"])

    def test_step6(self):
        self.assertGreater(self.out[6]["auc_logistic"], 0.65)

    def test_step7(self):
        s = self.out[7]
        self.assertGreater(s["recall_network"], 2 * s["recall_baseline"])

    def test_step8(self):
        s = self.out[8]
        self.assertEqual(s["subtypes"], 35)
        self.assertLess(s["nodf_z"], 0)

    def test_step9(self):
        s = self.out[9]["scenarios"]
        self.assertGreater(s["random"]["cosine"]["top10"], s["random"]["rwr"]["top10"] + 0.1)
        self.assertGreater(s["specific"]["cosine"]["top10"], s["specific"]["jaccard"]["top10"])
        self.assertGreater(s["specific"]["cosine"]["top10"], s["common"]["cosine"]["top10"] + 0.3)

    def test_step10(self):
        s = self.out[10]
        self.assertEqual(s["exact_twin_diseases"], 541)
        g, h = s["strategies"]["greedy"], s["strategies"]["hubs"]
        self.assertLess(g["mean_questions"], h["mean_questions"])
        self.assertGreater(g["identified_share"], 0.9)
        self.assertLess(g["mean_questions"], 16)

    def test_step11(self):
        c = self.out[11]["conditions"]
        Q = str(self.out[11]["questions"])
        self.assertGreater(c["noisy answers"]["soft_ig"]["top1_at"][Q], c["noisy answers"]["hard"]["top1_at"][Q] + 0.1)
        self.assertGreater(c["perfect answers"]["soft_ig"]["top1_at"][Q], 0.8)
        self.assertGreater(c["noisy answers"]["soft_ig"]["top1_at"][Q], c["noisy answers"]["soft_random"]["top1_at"][Q])

    def test_step12(self):
        s = self.out[12]
        self.assertEqual(s["modules"], 20)
        self.assertGreater(s["Q_B"]["BRIM modules"], s["Q_B"]["random groups"] + 0.2)
        self.assertGreater(s["Q_B"]["BRIM modules"], s["Q_B"]["Louvain communities"])
        self.assertGreater(s["triage_accuracy"]["BRIM modules"], 4 * s["triage_chance"])
        self.assertEqual(len(s["largest_modules"]), 8)

    def test_summary_written(self):
        reporting.write_summary(self.cfg, [], True)
        text = (self.cfg.res_dir / "summary.md").read_text(encoding="utf-8")
        self.assertIn("Step 8", text)
        self.assertIn("QUICK MODE", text)


if __name__ == "__main__":
    unittest.main()
