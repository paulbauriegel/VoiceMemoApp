from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/rewrite-whisper-layernorm-fp16-safe.py"
SPEC = importlib.util.spec_from_file_location("layernorm_rewrite", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
rewrite = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = rewrite
SPEC.loader.exec_module(rewrite)


class LayerNormRewriteTest(unittest.TestCase):
    def test_scaled_variance_is_algebraically_equivalent(self) -> None:
        values = [-295.0, -14.0, 0.25, 13.0, 238.375]
        mean = sum(values) / len(values)
        original = sum((value - mean) ** 2 for value in values) / len(values)
        scaled = (
            sum(((value - mean) * rewrite.SCALE) ** 2 for value in values)
            / len(values)
            * rewrite.RESTORE
        )
        self.assertTrue(math.isclose(original, scaled, rel_tol=0.0, abs_tol=1e-12))

    def test_observed_outlier_square_is_brought_below_fp16_limit(self) -> None:
        centered = 295.2745861828327
        self.assertGreater(centered * centered, 65_504)
        self.assertLess((centered * rewrite.SCALE) ** 2, 65_504)

    def test_scale_and_restore_are_exact_inverse_squares(self) -> None:
        self.assertEqual(rewrite.SCALE * rewrite.SCALE * rewrite.RESTORE, 1.0)


if __name__ == "__main__":
    unittest.main()
