import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_rhaister_signal.py"
SCRIPT_DIR = str(SCRIPT.parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
SPEC = importlib.util.spec_from_file_location("evaluate_rhaister_signal", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SignalPilotTests(unittest.TestCase):
    def test_parse_single_drug_treatment(self):
        result = MODULE.parse_treatment("[('Drug A', 0.5, 'uM')]")
        self.assertEqual(result, ("Drug A", 0.5, "uM"))

    def test_centered_cosine_is_pearson(self):
        query = np.array([[1, 2, 3]], dtype=np.float32)
        reference = np.array([[2, 4, 6], [3, 2, 1]], dtype=np.float32)
        result = MODULE.centered_cosine_similarity(query, reference)
        np.testing.assert_allclose(result, [[1.0, -1.0]], atol=1e-8)

    def test_line_template_removal_zeroes_line_means(self):
        cells = np.array(["a", "a", "b", "b"])
        treatments = np.array(["x", "y", "x", "y"])
        values = np.array([
            [1, 2], [3, 6],
            [2, 4], [8, 10],
        ], dtype=np.float32)
        result, indices = MODULE.remove_line_template(values, cells, treatments)
        cube = result.reshape(2, 2, 2)
        np.testing.assert_allclose(cube.mean(axis=1), 0.0, atol=1e-7)
        np.testing.assert_array_equal(indices, [0, 1, 2, 3])

    def test_clustered_bootstrap_keeps_finite_results(self):
        result = MODULE.clustered_bootstrap(
            np.array([0.2, 0.4, 0.8, 1.0]),
            np.array([0.1, 0.1, 0.2, 0.2]),
            np.array(["a", "a", "b", "b"]),
            n_bootstrap=100,
            seed=0,
        )
        self.assertEqual(result["n_drugs"], 2)
        self.assertTrue(all(np.isfinite(value) for value in result.values()))


if __name__ == "__main__":
    unittest.main()
