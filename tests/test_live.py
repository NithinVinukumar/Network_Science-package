"""tests/test_live.py - the typed live-test tool answers sensibly for typed inputs."""
import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hpo_data
import live_test
from config import Settings


class TestLiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = hpo_data.load(Settings())

    def run_quiet(self, fn, inputs):
        buf = io.StringIO()
        with mock.patch("builtins.input", side_effect=inputs), contextlib.redirect_stdout(buf):
            r = fn(self.d)
        return r, buf.getvalue()

    def test_typed_findings_give_joubert_family(self):
        r, out = self.run_quiet(live_test.diagnose, ["molar tooth sign, ataxia, nystagmus", ""])
        self.assertIn("Joubert syndrome", r["top"])
        self.assertIn("NETWORK CONCEPT", out)
        self.assertIn("CODE THAT RUNS", out)
        self.assertEqual(r["n_findings"], 3)

    def test_answering_questions_adds_findings(self):
        r, _ = self.run_quiet(live_test.diagnose, ["molar tooth sign, ataxia, nystagmus", "y", "y", ""])
        self.assertEqual(r["n_findings"], 5)
        r2, _ = self.run_quiet(live_test.diagnose, ["molar tooth sign, ataxia, nystagmus", "n", ""])
        self.assertEqual(r2["n_findings"], 3)

    def test_no_answer_changes_the_scores(self):
        r, out = self.run_quiet(live_test.diagnose, ["molar tooth sign, ataxia, nystagmus", "n", ""])
        self.assertEqual(r["n_absent"], 1)
        self.assertIn("does NOT have", out)
        self.assertIn("Changes since your last answer", out)
        self.assertIn("DOWN", out)
        r2, out2 = self.run_quiet(live_test.diagnose, ["molar tooth sign, ataxia, nystagmus", "y", ""])
        self.assertIn("UP", out2)

    def test_second_typed_case_gives_bardet_biedl(self):
        r, _ = self.run_quiet(live_test.diagnose, ["retinal dystrophy, polydactyly, obesity", ""])
        self.assertIn("Bardet-Biedl", r["top"])

    def test_unknown_symptom_is_handled(self):
        r, out = self.run_quiet(live_test.diagnose, ["zzzz not a symptom", ""])
        self.assertIsNone(r)
        self.assertIn("no symptom contains", out)

    def test_stress_test_typed_disease(self):
        r, _ = self.run_quiet(live_test.stress_test, ["Joubert syndrome 6", "0.25"])
        self.assertEqual(r["specific"], 1)
        self.assertGreater(r["common"], 20)

    def test_twin_check_typed_diseases(self):
        r, out = self.run_quiet(live_test.twin_check, ["Porokeratosis 1", "Porokeratosis 7"])
        self.assertAlmostEqual(r["similarity"], 1.0, places=2)
        self.assertIn("TWINS", out)
        r2, out2 = self.run_quiet(live_test.twin_check, ["Joubert syndrome 6", "Marfan syndrome"])
        self.assertLess(r2["similarity"], 0.5)
        self.assertIn("not twins", out2)

    def test_symptom_lookup(self):
        r, out = self.run_quiet(live_test.symptom_lookup, ["seizure"])
        self.assertGreater(r["n_found"], 3)
        self.assertIn("information content", out)

    def test_compare_algorithms(self):
        r, out = self.run_quiet(live_test.compare_algorithms, ["molar tooth sign, ataxia, nystagmus"])
        self.assertEqual(len(r["top"]), 3)
        self.assertIn("random walk with restart", out)
        self.assertTrue(any("Joubert" in x for x in r["top"]["IC-weighted cosine (ours)"]))

    def test_twenty_questions_finds_the_disease(self):
        r, out = self.run_quiet(live_test.twenty_questions, ["Joubert syndrome 6", ""])
        self.assertTrue(r["identified"])
        self.assertTrue(8 <= r["n_questions"] <= 22)
        self.assertIn("FOUND", out)
        self.assertIn("NETWORK CONCEPT", out)

    def test_twenty_questions_twins_cannot_be_separated(self):
        r, out = self.run_quiet(live_test.twenty_questions, ["Porokeratosis 1", ""])
        self.assertTrue(r["stuck"])
        self.assertGreaterEqual(r["left"], 2)
        self.assertIn("IDENTICAL", out)

    def test_twenty_questions_manual_and_unknown(self):
        r, _ = self.run_quiet(live_test.twenty_questions, ["Joubert syndrome 6", "y", "y", "n", ""])
        self.assertEqual(r["n_questions"], 2)
        self.assertGreater(r["left"], 1)
        r2, out2 = self.run_quiet(live_test.twenty_questions, ["zzzz not a disease"])
        self.assertIsNone(r2)

    def test_noisy_questions_without_noise_finds_the_disease(self):
        r, out = self.run_quiet(live_test.noisy_questions, ["Joubert syndrome 6", "0", ""])
        self.assertTrue(r["correct"])
        self.assertEqual(r["mistakes"], 0)
        self.assertIn("NETWORK CONCEPT", out)

    def test_noisy_questions_with_noise_runs_and_reports_mistakes(self):
        r, out = self.run_quiet(live_test.noisy_questions, ["Joubert syndrome 6", "0.2", ""])
        self.assertGreaterEqual(r["n_questions"], 10)
        self.assertGreaterEqual(r["mistakes"], 0)
        self.assertTrue(0.0 < r["top_p"] <= 1.0)
        self.assertIn("RESULT", out)

    def test_noisy_questions_manual_and_unknown(self):
        # inputs: disease, miss rate, "y" = I answer myself, then three answers (y, n, y), then Enter = stop
        r, _ = self.run_quiet(live_test.noisy_questions, ["Joubert syndrome 6", "0.2", "y", "y", "n", "y", ""])
        self.assertEqual(r["n_questions"], 3)
        self.assertIsNone(r["correct"])
        r2, _ = self.run_quiet(live_test.noisy_questions, ["zzzz not a disease"])
        self.assertIsNone(r2)

    def test_which_module(self):
        r, out = self.run_quiet(live_test.which_module, ["molar tooth sign, ataxia, nystagmus"])
        self.assertTrue(0 <= r["module"] < r["n_modules"])
        self.assertEqual(len(r["top_diseases"]), 5)
        self.assertTrue(len(r["signature"]) >= 3)
        self.assertIn("NETWORK CONCEPT", out)
        self.assertIn("signature symptoms", out)

    def test_which_module_unknown_symptom(self):
        r, out = self.run_quiet(live_test.which_module, ["zzzz not a symptom"])
        self.assertIsNone(r)

    def test_menu_quits_cleanly(self):
        with mock.patch("builtins.input", side_effect=["9", "0"]), contextlib.redirect_stdout(io.StringIO()) as buf, \
                mock.patch("hpo_data.load", return_value=self.d):
            live_test.main()
        self.assertIn("Bye.", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
