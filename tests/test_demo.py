"""tests/test_demo.py - the guided live demo gives the results the presentation relies on."""
import contextlib
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import demo
import hpo_data
from config import Settings


class TestDemo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Settings()
        cls.d = hpo_data.load(cls.cfg)

    def run_quiet(self, fn, *a):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*a)

    def test_scene1_family_found(self):
        r = self.run_quiet(demo.scene1, self.d)
        self.assertGreaterEqual(r["n_joubert_top10"], 6)
        self.assertTrue(len(r["next"]) >= 3)

    def test_scene2_question_changes_ranking(self):
        h = self.run_quiet(demo.scene2, self.d)
        self.assertEqual(len(h), 2)
        self.assertGreater(h[1]["top_sim"], h[1]["second_sim"])

    def test_scene3_specific_beats_common(self):
        r = self.run_quiet(demo.scene3, self.d, "Joubert syndrome 6")
        self.assertEqual(r["specific"], 1)
        self.assertGreater(r["common"], 20)

    def test_scene3_other_disease_and_error(self):
        r = self.run_quiet(demo.scene3, self.d, "Marfan syndrome")
        self.assertLess(r["specific"], r["common"])
        with self.assertRaises(ValueError):
            self.run_quiet(demo.scene3, self.d, "zzzz not a disease")

    def test_scene4_twins(self):
        r = self.run_quiet(demo.scene4, self.d)
        self.assertEqual(len(r), 2)
        for t in r:
            self.assertAlmostEqual(t["similarity"], 1.0, places=2)
            self.assertNotEqual(t["genes"][0], t["genes"][1])

    def test_scene5_reads_results(self):
        r = self.run_quiet(demo.scene5, self.cfg)
        self.assertGreater(r["specific"], r["common"] + 0.5)


if __name__ == "__main__":
    unittest.main()
