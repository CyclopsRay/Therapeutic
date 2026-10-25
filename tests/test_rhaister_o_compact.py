import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_rhaister_o_compact.py"
SPEC = importlib.util.spec_from_file_location("run_rhaister_o_compact", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BaselineUnitTests(unittest.TestCase):
    def test_rowwise_pearson_and_constant_truth(self):
        truth = np.array([[1, 2, 3], [2, 2, 2]], dtype=np.float32)
        prediction = np.array([[4, 8, 12], [1, 2, 3]], dtype=np.float32)
        metrics = MODULE.rowwise_metrics(truth, prediction)
        np.testing.assert_allclose(metrics["pearson"], [1.0, 0.0])

    def test_group_mean(self):
        values = np.array([[1, 3], [3, 5], [7, 9]], dtype=np.float32)
        result = MODULE.group_mean(values, np.array([0, 0, 1]), 2)
        np.testing.assert_allclose(result, [[2, 4], [7, 9]])

    def test_additive_als_shape_and_finite(self):
        values = np.array([[1, 2], [3, 4], [2, 3], [4, 5]], dtype=np.float32)
        cells = np.array(["a", "a", "b", "b"])
        treatments = np.array(["x", "y", "x", "y"])
        result = MODULE.additive_als(values, cells, treatments, np.array(["x", "y"]))
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(np.isfinite(result).all())

    def test_diagonal_equation_and_complete_coverage(self):
        train_cells = np.array(["a", "a", "b", "b"])
        train_treatments = np.array(["x", "y", "x", "y"])
        train_y = np.array([[1, 2], [3, 4], [2, 5], [5, 9]], dtype=np.float32)
        train_h = np.array([[1, 2], [1, 2], [3, 5], [3, 5]], dtype=np.float32)
        test_cells = np.array(["c", "c"])
        test_treatments = np.array(["x", "y"])
        test_h = np.array([[2, 3], [2, 3]], dtype=np.float32)
        additive = np.zeros((2, 2), dtype=np.float32)

        result, _, covered = MODULE.diagonal_predictions(
            train_y,
            train_cells,
            train_treatments,
            test_cells,
            test_treatments,
            train_h,
            test_h,
            additive,
            ridge_lambda=1.0,
            blend=1.0,
        )

        z = np.stack([np.log1p([1, 2]), np.log1p([3, 5])])
        z_mean = z.mean(axis=0)
        z_centered = z - z_mean
        z_test = np.log1p([2, 3]) - z_mean
        y_all = train_y.reshape(2, 2, 2).astype(np.float64)
        treatment_mean = y_all.mean(axis=0)
        residual = y_all - treatment_mean
        gamma = np.einsum("cg,ctg->tg", z_centered, residual)
        gamma /= (np.sum(z_centered**2, axis=0) + 1.0)[None, :]
        expected = treatment_mean + gamma * z_test[None, :]
        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(covered, [True, True])


if __name__ == "__main__":
    unittest.main()
