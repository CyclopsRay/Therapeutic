import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_lincs_tahoe_concordance import (
    cross_metrics,
    is_24_hour,
    parse_lincs_dose_um,
    rank_rows,
)


class LincsTahoeConcordanceTests(unittest.TestCase):
    def test_phase_one_dose_units(self):
        frame = pd.DataFrame({
            "pert_dose": [100.0, 100.0, 100.0],
            "pert_dose_unit": ["nM", "uM", "mM"],
        })
        np.testing.assert_allclose(parse_lincs_dose_um(frame), [0.1, 100.0, 100000.0])

    def test_phase_two_text_doses(self):
        frame = pd.DataFrame({"pert_idose": ["40 nM", "3.33 um", "0.01 mM"]})
        np.testing.assert_allclose(parse_lincs_dose_um(frame), [0.04, 3.33, 10.0])

    def test_time_parsing(self):
        phase_one = pd.DataFrame({"pert_time": [24, 6], "pert_time_unit": ["h", "h"]})
        phase_two = pd.DataFrame({"pert_itime": ["24 h", "6 h"]})
        np.testing.assert_array_equal(is_24_hour(phase_one), [True, False])
        np.testing.assert_array_equal(is_24_hour(phase_two), [True, False])

    def test_metrics_identical_and_reversed(self):
        truth = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        prediction = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
        metrics = cross_metrics(truth, prediction)
        np.testing.assert_allclose(metrics["pearson"], [1.0, -1.0])
        np.testing.assert_allclose(metrics["spearman"], [1.0, -1.0])
        np.testing.assert_allclose(metrics["sign_agreement"], [1.0, 0.0])

    def test_rank_rows(self):
        values = np.array([[3.0, 1.0, 2.0]])
        np.testing.assert_array_equal(rank_rows(values), [[2.0, 0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
