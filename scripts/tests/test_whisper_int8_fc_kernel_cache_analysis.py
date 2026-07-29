import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "analyze-whisper-int8-fc-kernel-caches.py"
)
SPEC = importlib.util.spec_from_file_location("kernel_cache_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class KernelCacheAnalysisTest(unittest.TestCase):
    def test_groups_binding_families_without_private_class_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            variants = [
                "direct-int8-fc",
                "source-quantized-default",
                "source-quantized-buffer-storage",
            ]
            (root / "report.json").write_text(
                json.dumps(
                    {
                        "variants": [
                            {"name": name}
                            for name in variants
                        ]
                    }
                )
            )
            cache_strings = {
                "direct-int8-fc": [
                    "main_function",
                    "weights_convert_uint8_to_float32",
                    "src_tensor_image2d",
                ],
                "source-quantized-default": [
                    "main_function",
                    "weights_convert_uint8_to_int8",
                    "src_tensor_image2d",
                ],
                "source-quantized-buffer-storage": [
                    "main_function",
                    "weights_convert_uint8_to_int8",
                    "src_tensor_buffer",
                ],
            }
            for name, values in cache_strings.items():
                (root / f"{name}_mldrift_program_cache.bin").write_bytes(
                    b"\0".join(value.encode() for value in values)
                )

            with mock.patch.object(sys, "argv", [str(SCRIPT), str(root)]):
                self.assertEqual(MODULE.main(), 0)

            result = json.loads((root / "kernel-selection.json").read_text())
            families = {
                variant["variant"]: variant["binding_family"]
                for variant in result["variants"]
            }
            self.assertEqual(
                families["direct-int8-fc"],
                "direct-int8-fc-dequantized-to-float",
            )
            self.assertEqual(
                families["source-quantized-default"],
                "source-quantized-fc-conv-image2d-bindings",
            )
            self.assertEqual(
                families["source-quantized-buffer-storage"],
                "source-quantized-fc-conv-buffer-bindings",
            )
            self.assertTrue(
                all(
                    variant["implementation_class"] is None
                    for variant in result["variants"]
                )
            )


if __name__ == "__main__":
    unittest.main()
