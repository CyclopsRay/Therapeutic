import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_depmap_expression_ridge.py"
SCRIPT_DIR = str(SCRIPT.parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
SPEC = importlib.util.spec_from_file_location("run_depmap_expression_ridge", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class DepMapRidgeTests(unittest.TestCase):
    def test_analytic_loo_matches_brute_force_ridge(self):
        x = np.array([-2.0, -1.0, 1.0, 2.0])[:, None]
        y = np.column_stack([2.0 + 3.0 * x[:, 0], -1.0 + 0.5 * x[:, 0]])
        response_gram = y @ y.T
        candidates = MODULE.build_candidates(
            x,
            response_gram,
            y.shape[1],
            component_grid=(1,),
            lambda_grid=(1.0,),
        )
        candidate = candidates[1]
        brute_predictions = []
        for heldout in range(len(x)):
            keep = np.arange(len(x)) != heldout
            design = np.column_stack([np.ones(keep.sum()), x[keep, 0]])
            penalty = np.diag([0.0, 1.0])
            coefficients = np.linalg.solve(
                design.T @ design + penalty,
                design.T @ y[keep],
            )
            brute_predictions.append(np.array([1.0, x[heldout, 0]]) @ coefficients)
        brute_predictions = np.stack(brute_predictions)
        brute_rmse = np.sqrt(np.mean(np.square(y - brute_predictions)))
        self.assertAlmostEqual(candidate.loo_rmse, brute_rmse, places=10)

    def test_target_weights_match_direct_coefficients(self):
        scores = np.array([[-1.0], [0.0], [1.0]])
        y = np.array([[1.0, 2.0], [2.0, 1.0], [4.0, 3.0]])
        candidates = MODULE.build_candidates(
            scores, y @ y.T, y.shape[1], component_grid=(1,), lambda_grid=(2.0,)
        )
        candidate = candidates[1]
        query = np.array([[0.5]])
        weighted = MODULE.target_weights(candidate, query) @ y
        design = candidate.design
        penalty = np.diag([0.0, 2.0])
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        direct = np.array([[1.0, 0.5]]) @ coefficients
        np.testing.assert_allclose(weighted, direct, atol=1e-12)

    def test_reference_only_pca_shapes(self):
        reference = np.array([
            [1, 2, 3, 4],
            [2, 1, 4, 3],
            [3, 4, 1, 2],
        ], dtype=float)
        query = np.array([[1.5, 2.5, 3.5, 4.5]])
        scores, query_scores, info = MODULE.reference_only_pca(reference, query)
        self.assertEqual(scores.shape[0], 3)
        self.assertEqual(query_scores.shape, (1, scores.shape[1]))
        self.assertEqual(info["n_input_genes"], 4)


if __name__ == "__main__":
    unittest.main()
