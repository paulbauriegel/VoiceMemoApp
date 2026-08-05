import importlib.util
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/sample-android-thermal.py"
SPEC = importlib.util.spec_from_file_location("sample_android_thermal", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SAMPLER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SAMPLER
SPEC.loader.exec_module(SAMPLER)


THERMAL_DUMP = """\
Thermal Status: 1
Cached temperatures:
    Temperature{mValue=67.0, mType=0, mName=BIG, mStatus=1}
HAL Ready: true
Current temperatures from HAL:
    Temperature{mValue=31.0, mType=0, mName=BIG, mStatus=0}
    Temperature{mValue=29.5, mType=1, mName=G3D, mStatus=0}
    Temperature{mValue=26.25, mType=3, mName=VIRTUAL-SKIN, mStatus=0}
Current cooling devices from HAL:
    CoolingDevice{mValue=0, mType=2, mName=thermal-cpufreq-0}
"""


class AndroidThermalSamplerTest(unittest.TestCase):
    def test_parser_uses_current_hal_temperatures_not_cached_values(self):
        status, temperatures = SAMPLER.parse_thermalservice(THERMAL_DUMP)

        self.assertEqual(status, 1)
        self.assertEqual(
            [(item.name, item.sensor_type, item.value_celsius) for item in temperatures],
            [("BIG", 0, 31.0), ("G3D", 1, 29.5), ("VIRTUAL-SKIN", 3, 26.25)],
        )

    def test_parser_rejects_dump_without_live_hal_section(self):
        with self.assertRaisesRegex(SAMPLER.SamplingError, "live HAL"):
            SAMPLER.parse_thermalservice(
                "Thermal Status: 0\nCached temperatures:\n"
                "Temperature{mValue=67.0, mType=0, mName=BIG, mStatus=0}\n"
            )

    def test_battery_temperature_is_tenths_of_a_degree(self):
        battery = SAMPLER.parse_battery(
            "AC powered: false\nUSB powered: true\n"
            "Wireless powered: false\nlevel: 72\ntemperature: 243\n"
        )

        self.assertEqual(battery["temperature_celsius"], 24.3)
        self.assertEqual(battery["level_percent"], 72)
        self.assertFalse(battery["ac_powered"])
        self.assertTrue(battery["usb_powered"])

    def test_summary_records_first_status_transition_and_sensor_extrema(self):
        summary = {
            "sample_count": 0,
            "elapsed_seconds": 0.0,
            "thermal_status_first_seen_seconds": {},
            "sensors": {},
        }
        for elapsed, status, value in [(0.5, 0, 30.0), (5.0, 1, 37.0), (8.0, 1, 35.0)]:
            SAMPLER.update_summary(
                summary,
                {
                    "elapsed_seconds": elapsed,
                    "thermal_status": status,
                    "temperatures": [
                        {"name": "BIG", "type": 0, "value_celsius": value}
                    ],
                },
            )

        self.assertEqual(summary["thermal_status_first_seen_seconds"], {"0": 0.5, "1": 5.0})
        self.assertEqual(summary["sensors"]["BIG"]["minimum_celsius"], 30.0)
        self.assertEqual(summary["sensors"]["BIG"]["maximum_celsius"], 37.0)
        self.assertEqual(summary["sensors"]["BIG"]["last_celsius"], 35.0)


if __name__ == "__main__":
    unittest.main()
