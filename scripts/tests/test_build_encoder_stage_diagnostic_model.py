from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/build-encoder-stage-diagnostic-model.py"


def load_script():
    spec = importlib.util.spec_from_file_location("encoder_stage_diagnostic", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load encoder stage diagnostic builder")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class EncoderStageDiagnosticBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = load_script()

    def test_precision_localization_covers_encoder_blocks_16_through_31(self) -> None:
        self.assertEqual(list(range(16, 32)), list(self.script.ENCODER_BLOCKS))

    def test_block_16_input_is_previous_block_residual(self) -> None:
        self.assertEqual(
            "encoder/layers.15/add_1",
            self.script.stage_tensor_name(16, "input"),
        )

    def test_block_stage_name_uses_requested_block(self) -> None:
        self.assertEqual(
            "encoder/layers.23/self_attn/Softmax",
            self.script.stage_tensor_name(23, "softmax"),
        )


if __name__ == "__main__":
    unittest.main()
