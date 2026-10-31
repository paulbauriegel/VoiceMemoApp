import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('names', Path(__file__).resolve().parents[1] / 'compare-name-cleanup.py')
names = importlib.util.module_from_spec(spec)
spec.loader.exec_module(names)


class NameCleanupTest(unittest.TestCase):
    def test_boundaries_possessives_and_other_characters(self):
        text = 'Erik, Elnas! Erikson; Erik_s.\nEltern? ERIK und Elna.'
        fixed, edits = names.replace_names(text, {'Erik': 'Eric', 'Elna': 'Elena', 'Elnas': 'Elenas'})
        self.assertEqual(fixed, 'Eric, Elenas! Erikson; Erik_s.\nEltern? ERIK und Elena.')
        # Reverse edits using original offsets to establish preservation independently.
        expected = text
        for edit in reversed(edits):
            expected = expected[:edit['start']] + edit['after'] + expected[edit['end']:]
        self.assertEqual(fixed, expected)
        self.assertEqual(len(edits), 3)

    def test_does_not_cascade_aliases(self):
        self.assertEqual(names.replace_names('A B', {'A': 'B', 'B': 'C'})[0], 'B C')
