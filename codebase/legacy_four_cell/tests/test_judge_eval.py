"""Default literature-judge evaluation (offline, deterministic).

The gate default is `default_judge` (token overlap). Measured here on 8
labeled claim/passage pairs from the real corpus: 4/8. It is a weak
lexical filter — reliable for verbatim/paraphrase/unrelated, blind to
contradiction, negation, wrong-material, and true entailment. The LLM
judge scores 6/6 on the same battery (see DECISIONS.md D17); wire it via
`LiteratureCell(judge='llm')` when model access is available.

Easy cases are pinned strictly; the floor assertion guards regressions
while leaving room to improve the hard cases.
"""
import unittest

from perovagents.verify.evidence_checks import check_support, default_judge

P1 = ('Sn-doped TiO2 films were successfully deposited by low-temperature '
      'solution method as electron transport layer (ETL), whose influence '
      'on the performance of perovskite solar cell (PSC) was investigated.')
P2 = ('Perovskite films deposited onto a SnO2 electron transporting layer '
      '(ETL) using the antisolvent method after annealing with the FACl '
      'additive, producing small grains on the surface.')

EASY = [
    (P1, P1, True, "verbatim"),
    (P1, "Sn-doped TiO2 films work as electron transport layers made by "
         "low-temperature solution processing", True, "paraphrase"),
    (P1, "quantum dot lasers lase at room temperature", False, "unrelated"),
    (P2, "Perovskite films were deposited on SnO2 ETL by antisolvent "
         "method with FACl additive", True, "paraphrase2"),
]

# Known limitations of the token judge (pinned as documentation, NOT as
# required failures — improving them must not break this suite).
HARD = [
    (P1, "Sn-doped TiO2 films were deposited by high-temperature vacuum "
         "evaporation", False, "contradicted-method"),
    (P2, "Perovskite films were deposited on NiO hole transport layers "
         "by spin coating", False, "wrong-material"),
    (P2, "The FACl additive produces large grains on the perovskite "
         "surface", False, "negated-grains"),
    (P1, "Sn-doped TiO2 can serve as an electron transport layer in "
         "perovskite solar cells", True, "entailed"),
]


class TestDefaultJudge(unittest.TestCase):
    def test_easy_cases(self):
        for pas, claim, want, name in EASY:
            got, _ = default_judge(claim, pas)
            self.assertEqual(got, want, name)

    def test_accuracy_floor(self):
        ok = sum(1 for pas, claim, want, _ in EASY + HARD
                 if default_judge(claim, pas)[0] == want)
        total = len(EASY + HARD)
        self.assertGreaterEqual(ok / total, 0.5,
                                f"judge below floor: {ok}/{total}")

    def test_check_support_uses_injected_judge(self):
        fake = lambda c, p: (True, "injected")  # noqa: E731
        r = check_support("anything", "anything", judge=fake)
        self.assertTrue(r.passed)
        self.assertIn("injected", r.message)

    def test_cell_honors_injected_judge(self):
        from perovagents.cells.literature import LiteratureCell
        raw = {"claims": [{"text": "X works",
                           "citations": [{"paper_id": "10/x",
                                          "locator": "p1",
                                          "passage": "X works well",
                                          "context": {}}]}]}
        cell_yes = LiteratureCell(
            judge=lambda c, p: (True, "fake-yes"))
        self.assertTrue(cell_yes._gate(raw, {}).passed)
        cell_no = LiteratureCell(
            judge=lambda c, p: (False, "fake-no"))
        self.assertFalse(cell_no._gate(raw, {}).passed)


if __name__ == "__main__":
    unittest.main()
