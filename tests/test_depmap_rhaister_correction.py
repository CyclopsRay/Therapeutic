import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_depmap_rhaister_correction import loo_rhaister_corrections
from run_rhaister_o_compact import diagonal_predictions


class RhaisterCorrectionTests(unittest.TestCase):
    def test_closed_form_matches_direct_leave_one_line_out(self):
        rng = np.random.default_rng(12)
        n_lines, n_treatments, n_genes = 6, 5, 4
        response = rng.normal(size=(n_lines, n_treatments, n_genes)).astype(np.float32)
        raw_hvg = rng.uniform(0.05, 1.5, size=(n_lines, n_genes))
        expected = loo_rhaister_corrections(response, raw_hvg)
        treatment_names = np.array([f"drug_{i}" for i in range(n_treatments)])

        for heldout in range(n_lines):
            training = [line for line in range(n_lines) if line != heldout]
            train_y = response[training].reshape(-1, n_genes)
            train_cells = np.repeat([f"c{i}" for i in training], n_treatments)
            train_treatments = np.tile(treatment_names, len(training))
            train_h = np.repeat(np.expm1(raw_hvg[training]), n_treatments, axis=0)
            test_cells = np.repeat(f"c{heldout}", n_treatments)
            test_h = np.repeat(np.expm1(raw_hvg[[heldout]]), n_treatments, axis=0)
            additive = response[training].mean(axis=0)
            prediction, _, covered = diagonal_predictions(
                train_y,
                train_cells,
                train_treatments,
                test_cells,
                treatment_names,
                train_h,
                test_h,
                additive,
                ridge_lambda=1.0,
                blend=0.95,
            )
            self.assertTrue(covered.all())
            np.testing.assert_allclose(
                expected[heldout], response[heldout] - prediction, rtol=2e-6, atol=2e-6
            )

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            loo_rhaister_corrections(np.zeros((3, 2, 4)), np.zeros((3, 3)))


if __name__ == "__main__":
    unittest.main()
