"""Phase 9H: focused tests for the multimodal-baseline ablation script.

Covers the temporal-alignment investigation's correlation method (with
synthetic positive/negative controls, independent of the real PCAP/CSV
data), the bootstrap CI helper, the metrics.csv writer, and a light
regression check against this run's own persisted report (skipped if not
present). No network access; no retraining triggered by these tests.
"""

from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS_DIR))

import numpy as np  # noqa: E402

from phase9h_multimodal_baseline import (  # noqa: E402
    CORRELATION_MIN_ABS_R,
    CORRELATION_MIN_MARGIN,
    OUTPUT_DIR,
    _pearson,
    bootstrap_ci,
    verify_cuda,
    write_metrics_csv,
)


class PearsonCorrelationMethodTests(unittest.TestCase):
    """The whole phase's central finding (STOP for B/C/D) rests on this
    correlation method correctly distinguishing real correlation from
    noise. These are positive/negative controls with synthetic data,
    independent of the real PCAP/CSV alignment question."""

    def test_perfect_positive_correlation(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [2.0, 4.0, 6.0, 8.0, 10.0]
        self.assertAlmostEqual(_pearson(xs, ys), 1.0)

    def test_perfect_negative_correlation(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [10.0, 8.0, 6.0, 4.0, 2.0]
        self.assertAlmostEqual(_pearson(xs, ys), -1.0)

    def test_constant_series_is_undefined(self) -> None:
        xs = [1.0, 2.0, 3.0]
        ys = [5.0, 5.0, 5.0]
        self.assertIsNone(_pearson(xs, ys))

    def test_too_few_points_is_undefined(self) -> None:
        self.assertIsNone(_pearson([1.0], [1.0]))
        self.assertIsNone(_pearson([], []))

    def test_random_uncorrelated_series_is_near_zero(self) -> None:
        rng = np.random.default_rng(7)
        xs = list(rng.normal(size=500))
        ys = list(rng.normal(size=500))
        r = _pearson(xs, ys)
        self.assertIsNotNone(r)
        self.assertLess(abs(r), 0.15)  # should NOT clear the phase's own 0.30 floor

    def test_method_correctly_flags_a_real_correlated_signal_as_alignable(self) -> None:
        # Positive control: verifies the METHOD (not the real data) would
        # correctly recognize a genuine correlation if one existed -- i.e.
        # the phase's negative finding is due to the data, not a broken check.
        rng = np.random.default_rng(3)
        base = rng.normal(size=300)
        xs = list(base)
        ys = list(base * 2.0 + rng.normal(scale=0.1, size=300))
        r = _pearson(xs, ys)
        self.assertGreater(abs(r), CORRELATION_MIN_ABS_R)


class BootstrapCITests(unittest.TestCase):
    def test_ci_bounds_contain_mean_and_are_ordered(self) -> None:
        rng = np.random.default_rng(1)
        n = 500
        labels = (rng.random(n) < 0.2).astype(np.int64)
        probabilities = np.clip(labels * 0.7 + rng.normal(scale=0.2, size=n) + 0.15, 0.0, 1.0)
        result = bootstrap_ci(labels, probabilities, threshold=0.5, n_iterations=300, seed=42)

        self.assertEqual(result["n_iterations_requested"], 300)
        self.assertGreater(result["n_iterations_used"], 0)
        for metric in ("pr_auc", "roc_auc", "f1", "recall"):
            entry = result[metric]
            self.assertIsNotNone(entry)
            self.assertLessEqual(entry["ci_lower_2.5pct"], entry["mean"])
            self.assertLessEqual(entry["mean"], entry["ci_upper_97.5pct"])

    def test_deterministic_with_fixed_seed(self) -> None:
        rng = np.random.default_rng(2)
        n = 200
        labels = (rng.random(n) < 0.3).astype(np.int64)
        probabilities = rng.random(n)
        first = bootstrap_ci(labels, probabilities, threshold=0.5, n_iterations=100, seed=99)
        second = bootstrap_ci(labels, probabilities, threshold=0.5, n_iterations=100, seed=99)
        self.assertEqual(first, second)

    def test_single_class_dataset_produces_no_usable_resamples(self) -> None:
        labels = np.zeros(50, dtype=np.int64)
        probabilities = np.random.default_rng(0).random(50)
        result = bootstrap_ci(labels, probabilities, threshold=0.5, n_iterations=50, seed=1)
        self.assertEqual(result["n_iterations_used"], 0)
        for metric in ("pr_auc", "roc_auc", "f1", "recall"):
            self.assertIsNone(result[metric])


class MetricsCsvWriterTests(unittest.TestCase):
    def test_writes_four_rows_with_correct_status(self) -> None:
        import tempfile

        fake_variant_a = {
            "parameter_count": 347806,
            "feature_dimensionality": 157,
            "n_train": 29315,
            "n_validation": 6195,
            "n_test": 6195,
            "best_epoch": 1,
            "training_duration_seconds": 25.1,
            "selected_threshold": 0.71,
            "validation_metrics": {
                "pr_auc": 0.75, "roc_auc": 0.77, "precision": 0.86, "recall": 0.42,
                "f1": 0.56, "false_positive_rate": 0.048,
            },
            "test_metrics": {
                "pr_auc": 0.87, "roc_auc": 0.88, "precision": 0.95, "recall": 0.76,
                "f1": 0.84, "false_positive_rate": 0.016,
            },
            "test_brier": 0.13,
            "test_ece": 0.16,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            write_metrics_csv(path, fake_variant_a)
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 4)
        by_variant = {row["variant"]: row for row in rows}
        self.assertEqual(by_variant["A_flow_only"]["status"], "TRAINED")
        self.assertEqual(by_variant["A_flow_only"]["parameter_count"], "347806")
        for variant in ("B_flow_plus_event", "C_flow_plus_graph", "D_flow_plus_event_plus_graph"):
            self.assertEqual(by_variant[variant]["status"], "NOT_CONSTRUCTED")
            self.assertEqual(by_variant[variant]["parameter_count"], "")


class CudaVerificationTests(unittest.TestCase):
    def test_verify_cuda_returns_expected_keys(self) -> None:
        result = verify_cuda()
        self.assertIn("cuda_available", result)
        if result["cuda_available"]:
            self.assertIn("gpu_name", result)
            self.assertIsInstance(result["gpu_name"], str)


class ConstantsSanityTests(unittest.TestCase):
    def test_correlation_thresholds_are_reasonable(self) -> None:
        self.assertGreater(CORRELATION_MIN_ABS_R, 0.0)
        self.assertLess(CORRELATION_MIN_ABS_R, 1.0)
        self.assertGreaterEqual(CORRELATION_MIN_MARGIN, 1.0)


class PersistedRunRegressionTests(unittest.TestCase):
    """Regression check against this phase's own persisted output, if present."""

    def test_persisted_report_is_internally_consistent(self) -> None:
        report_path = OUTPUT_DIR / "multimodal_baseline_report.json"
        if not report_path.exists():
            self.skipTest("Phase 9H has not been run yet in this environment")
        report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertIn(report["verdict"], {"GREEN", "YELLOW", "RED"})
        align = report["temporal_alignment_investigation"]
        self.assertIn(align["decision"], {"PROCEED", "STOP"})

        a = report["variant_a"]
        self.assertGreater(a["parameter_count"], 0)
        for split in ("validation_metrics", "test_metrics"):
            m = a[split]
            for key in ("precision", "recall", "f1", "pr_auc", "roc_auc", "false_positive_rate"):
                self.assertGreaterEqual(m[key], 0.0)
                self.assertLessEqual(m[key], 1.0)

        # If the alignment decision is STOP, the report must not claim B/C/D were trained.
        if align["decision"] == "STOP":
            self.assertIn("not be constructed", report["executive_summary"].lower())

    def test_metrics_csv_matches_json_for_variant_a(self) -> None:
        csv_path = OUTPUT_DIR / "metrics.csv"
        json_path = OUTPUT_DIR / "multimodal_baseline_report.json"
        if not csv_path.exists() or not json_path.exists():
            self.skipTest("Phase 9H has not been run yet in this environment")

        report = json.loads(json_path.read_text(encoding="utf-8"))
        with csv_path.open(newline="", encoding="utf-8") as f:
            rows = {row["variant"]: row for row in csv.DictReader(f)}

        a_json = report["variant_a"]
        a_csv = rows["A_flow_only"]
        self.assertEqual(int(a_csv["parameter_count"]), a_json["parameter_count"])
        self.assertAlmostEqual(float(a_csv["test_pr_auc"]), a_json["test_metrics"]["pr_auc"], places=9)


if __name__ == "__main__":
    unittest.main()
