import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_all_rhaister.py"
SCRIPT_DIR = str(SCRIPT.parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
SPEC = importlib.util.spec_from_file_location("evaluate_all_rhaister", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class AllSplitEvaluationTests(unittest.TestCase):
    def test_first_pc_recovers_rank_one_axis(self):
        scores = np.array([-2, -1, 1, 2], dtype=np.float32)
        axis = np.array([1, 2, -1], dtype=np.float32)
        values = scores[:, None] * axis[None, :]
        mean, component, ratio = MODULE.fit_first_pc(values, seed=0, iterations=20)
        self.assertGreater(ratio, 0.999999)
        residual = MODULE.remove_pc(values, mean, component)
        np.testing.assert_allclose(residual, 0.0, atol=1e-5)

    def test_remove_pc_preserves_orthogonal_signal(self):
        mean = np.zeros(2, dtype=np.float32)
        component = np.array([1, 0], dtype=np.float32)
        values = np.array([[3, 4], [-2, 5]], dtype=np.float32)
        result = MODULE.remove_pc(values, mean, component)
        np.testing.assert_allclose(result, [[0, 4], [0, 5]])


if __name__ == "__main__":
    unittest.main()
