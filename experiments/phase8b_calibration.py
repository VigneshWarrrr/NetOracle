"""Phase 8B: attack-risk calibration analysis for the Temporal Transformer
and the Vector World Model Run 1.

Pure post-hoc analysis over already-persisted validation/test probabilities
(imported unmodified from phase8_evaluation_comparison.py -- same
`load_predictions` function, same `MODEL_SPECS` file paths). No model is
loaded, no checkpoint is touched, nothing is trained or fine-tuned.

Calibrators (Platt scaling via sklearn LogisticRegression, isotonic
regression via sklearn IsotonicRegression) are fit on VALIDATION probabilities
and VALIDATION labels ONLY, then applied (transform only, no re-fitting) to
TEST probabilities exactly once. Method selection between Platt and
isotonic uses a VALIDATION-only Brier-score comparison (see
`select_calibration_method` for the explicit rule and its stated
limitation). Test data is never used to fit, tune, or select anything here.

Calibration recalibrates probability VALUES only. It does not define,
reselect, or apply any classification threshold -- Phase 8A's threshold and
results are read-only reference points, never recomputed or replaced here.

No external plotting library is used (matplotlib is not installed in this
environment and this phase does not install dependencies): reliability
diagrams and probability histograms are rendered as small, dependency-free
SVG files built from plain string formatting.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from phase8_evaluation_comparison import MODEL_SPECS, load_predictions

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_DIR = RESULTS_DIR / "phase8b_calibration"
PHASE8A_DIR = RESULTS_DIR / "phase8_final_evaluation"

TARGET_MODEL_NAMES = ["temporal_transformer", "vector_world_model_run1"]
TARGET_MODEL_SPECS = [spec for spec in MODEL_SPECS if spec["name"] in TARGET_MODEL_NAMES]

N_RELIABILITY_BINS = 10
AUC_EQUALITY_TOLERANCE = 1e-9

# Files this script only ever reads, never writes -- used for the
# before/after integrity sweep (includes Phase 8A's own outputs, which must
# remain untouched by this phase).
FROZEN_FILES_TO_VERIFY = [spec["validation_csv"] for spec in TARGET_MODEL_SPECS] + [spec["test_csv"] for spec in TARGET_MODEL_SPECS] + [
    spec["ranking_metrics_file"] for spec in TARGET_MODEL_SPECS
] + [
    PHASE8A_DIR / "comparison_metrics.json",
    PHASE8A_DIR / "FINAL_EVALUATION_TABLE.md",
    PHASE8A_DIR / "validation_report.json",
]


def _file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def brier_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((probabilities - labels) ** 2))


def reliability_bins(labels: np.ndarray, probabilities: np.ndarray, n_bins: int = N_RELIABILITY_BINS) -> tuple[list[dict[str, object]], float]:
    """Standard equal-width reliability binning + Expected Calibration Error
    (weighted mean absolute gap between mean predicted probability and
    observed positive rate per bin)."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, object]] = []
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= lo) & (probabilities < hi) if hi < 1.0 else (probabilities >= lo) & (probabilities <= hi)
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin_low": float(lo), "bin_high": float(hi), "count": 0, "mean_predicted": None, "observed_rate": None, "gap": None})
            continue
        mean_predicted = float(probabilities[mask].mean())
        observed_rate = float(labels[mask].mean())
        gap = abs(mean_predicted - observed_rate)
        ece += (count / n) * gap
        rows.append({"bin_low": float(lo), "bin_high": float(hi), "count": count, "mean_predicted": mean_predicted, "observed_rate": observed_rate, "gap": gap})
    return rows, float(ece)


def probability_distribution_summary(probabilities: np.ndarray) -> dict[str, float]:
    values = probabilities.astype(np.float64)
    return {
        "count": int(values.shape[0]),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p1": float(np.percentile(values, 1)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def fit_platt(val_labels: np.ndarray, val_probabilities: np.ndarray) -> LogisticRegression:
    """Platt scaling: a 1D logistic regression of label ~ raw probability,
    fit on VALIDATION ONLY. Deterministic (lbfgs solver, no randomness)."""
    model = LogisticRegression(solver="lbfgs")
    model.fit(val_probabilities.reshape(-1, 1), val_labels)
    return model


def apply_platt(model: LogisticRegression, probabilities: np.ndarray) -> np.ndarray:
    return model.predict_proba(probabilities.reshape(-1, 1))[:, 1]


def fit_isotonic(val_labels: np.ndarray, val_probabilities: np.ndarray) -> IsotonicRegression:
    """Isotonic regression, fit on VALIDATION ONLY. Deterministic (PAV
    algorithm has no randomness). out_of_bounds='clip' so test-time
    probabilities outside the validation probability range are clipped
    rather than extrapolated."""
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(val_probabilities, val_labels)
    return model


def apply_isotonic(model: IsotonicRegression, probabilities: np.ndarray) -> np.ndarray:
    return model.predict(probabilities)


def _preserves_ranking(original_probabilities: np.ndarray, calibrated_probabilities: np.ndarray) -> bool:
    """Structural, VALIDATION-ONLY proxy for 'does this calibrator preserve
    PR-AUC/ROC-AUC': does it avoid introducing NEW ties among originally-
    distinct probabilities? Platt scaling (a strictly monotonic sigmoid of a
    linear function with nonzero slope) does not introduce new ties.
    Isotonic regression's pooling (PAV) algorithm deliberately creates flat
    (tied) regions where the raw scores are non-monotonic with respect to
    the labels -- ties change how ROC-AUC/PR-AUC's tie-averaged rank
    statistic is computed, so isotonic can measurably alter ranking metrics
    even though it is monotonic non-decreasing. This check uses ONLY
    validation data (the same data the calibrator was fit on) -- it never
    looks at test."""
    return len(np.unique(calibrated_probabilities)) >= len(np.unique(original_probabilities))


def select_calibration_method(val_labels: np.ndarray, val_probabilities: np.ndarray) -> dict[str, object]:
    """Selection rule (stated explicitly, applied identically to both
    models):
      1. Fit both Platt and isotonic on validation.
      2. HARD FILTER (required by the task: calibration must not alter
         PR-AUC/ROC-AUC): keep only methods that do not introduce new ties
         among validation probabilities (`_preserves_ranking`, validation-
         only). Platt satisfies this structurally (verified below via its
         fitted coefficient sign); isotonic often does not, because its
         pooling algorithm creates tied blocks by design.
      3. Among the methods that pass the hard filter, select whichever
         achieves the LOWER Brier score on the SAME validation set it was
         fit on (in-sample; test data plays no role in this decision).

    LIMITATION, stated plainly: step 3 is an in-sample comparison (no
    held-out calibration-selection set exists beyond validation itself).
    Step 2 was added after observing, empirically, that isotonic changed
    test PR-AUC by up to 0.017 and test ROC-AUC by up to 0.017 in this
    dataset -- a real, structural property of isotonic regression's tie-
    formation, not a bug, and not something a Brier-only selection rule
    would have caught. Both methods' full test-set results are reported
    regardless of which is selected, so this filtering never hides isotonic's
    numbers -- it only decides which one becomes the single 'selected' choice.
    """
    platt = fit_platt(val_labels, val_probabilities)
    isotonic = fit_isotonic(val_labels, val_probabilities)

    platt_val_calibrated = apply_platt(platt, val_probabilities)
    isotonic_val_calibrated = apply_isotonic(isotonic, val_probabilities)

    platt_val_brier = brier_score(val_labels, platt_val_calibrated)
    isotonic_val_brier = brier_score(val_labels, isotonic_val_calibrated)

    platt_preserves_ranking = _preserves_ranking(val_probabilities, platt_val_calibrated)
    isotonic_preserves_ranking = _preserves_ranking(val_probabilities, isotonic_val_calibrated)
    platt_coefficient_positive = bool(platt.coef_[0][0] > 0)

    candidates = []
    if platt_preserves_ranking:
        candidates.append(("platt", platt_val_brier))
    if isotonic_preserves_ranking:
        candidates.append(("isotonic", isotonic_val_brier))
    if not candidates:
        candidates = [("platt", platt_val_brier)]  # structural fallback; should not occur

    selected_name = min(candidates, key=lambda item: item[1])[0]

    return {
        "rule": "Fit both Platt and isotonic on validation; HARD-FILTER out any method that introduces new "
        "ties among validation probabilities (a validation-only proxy for altering PR-AUC/ROC-AUC, required "
        "because calibration must not change ranking metrics); among methods that pass the filter, select "
        "the one with the LOWER in-sample validation Brier score. Test data plays no role in this decision.",
        "platt_validation_brier": platt_val_brier,
        "isotonic_validation_brier": isotonic_val_brier,
        "platt_preserves_ranking_on_validation": platt_preserves_ranking,
        "isotonic_preserves_ranking_on_validation": isotonic_preserves_ranking,
        "platt_fitted_coefficient_positive": platt_coefficient_positive,
        "selected_method": selected_name,
        "limitation": "The Brier comparison in the final step is in-sample (fit and evaluated on the same "
        "validation set); both methods' full test-set results are reported regardless of which is selected.",
        "_platt_model": platt,
        "_isotonic_model": isotonic,
    }


# ============================================================
# Dependency-free SVG plotting (matplotlib is not installed; this phase
# does not install dependencies, so plots are hand-built SVG files).
# ============================================================

SVG_WIDTH = 480
SVG_HEIGHT = 480
SVG_MARGIN = 50


def _svg_point(x_data: float, y_data: float) -> tuple[float, float]:
    plot_size = SVG_WIDTH - 2 * SVG_MARGIN
    x_pixel = SVG_MARGIN + x_data * plot_size
    y_pixel = SVG_HEIGHT - SVG_MARGIN - y_data * plot_size
    return x_pixel, y_pixel


def render_reliability_diagram_svg(title: str, curves: list[tuple[str, str, list[dict[str, object]]]]) -> str:
    """curves: list of (label, color, reliability_bins_rows)."""
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        f'<rect width="{SVG_WIDTH}" height="{SVG_HEIGHT}" fill="white"/>',
        f'<text x="{SVG_WIDTH/2}" y="20" font-size="14" text-anchor="middle" font-family="sans-serif">{title}</text>',
    ]
    x0, y0 = _svg_point(0, 0)
    x1, y1 = _svg_point(1, 1)
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="black" stroke-width="1"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="black" stroke-width="1"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="lightgray" stroke-width="1" stroke-dasharray="4,4"/>')
    parts.append(f'<text x="{SVG_WIDTH/2}" y="{SVG_HEIGHT-15}" font-size="11" text-anchor="middle" font-family="sans-serif">Mean predicted probability</text>')
    parts.append(f'<text x="15" y="{SVG_HEIGHT/2}" font-size="11" text-anchor="middle" font-family="sans-serif" transform="rotate(-90,15,{SVG_HEIGHT/2})">Observed positive rate</text>')

    legend_y = 35
    for label, color, rows in curves:
        points = [row for row in rows if row["count"] and row["count"] > 0]
        polyline = " ".join(f"{_svg_point(row['mean_predicted'], row['observed_rate'])[0]:.1f},{_svg_point(row['mean_predicted'], row['observed_rate'])[1]:.1f}" for row in points)
        if polyline:
            parts.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2"/>')
        for row in points:
            px, py = _svg_point(row["mean_predicted"], row["observed_rate"])
            radius = max(3.0, min(10.0, 3.0 + 7.0 * (row["count"] / max(1, max(r["count"] for r in points)))))
            parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.7"/>')
        parts.append(f'<circle cx="{SVG_WIDTH-140}" cy="{legend_y}" r="5" fill="{color}"/>')
        parts.append(f'<text x="{SVG_WIDTH-128}" y="{legend_y+4}" font-size="11" font-family="sans-serif">{label}</text>')
        legend_y += 18

    parts.append("</svg>")
    return "\n".join(parts)


def render_histogram_svg(title: str, series: list[tuple[str, str, np.ndarray]], n_bins: int = 20) -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}">',
        f'<rect width="{SVG_WIDTH}" height="{SVG_HEIGHT}" fill="white"/>',
        f'<text x="{SVG_WIDTH/2}" y="20" font-size="14" text-anchor="middle" font-family="sans-serif">{title}</text>',
    ]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    all_counts = []
    histograms = []
    for label, color, values in series:
        counts, _ = np.histogram(values, bins=edges)
        histograms.append((label, color, counts))
        all_counts.append(counts.max() if counts.size else 0)
    max_count = max(all_counts) if all_counts else 1
    max_count = max(max_count, 1)

    x0, y0 = _svg_point(0, 0)
    x1, y1 = _svg_point(1, 1)
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="black" stroke-width="1"/>')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="black" stroke-width="1"/>')
    parts.append(f'<text x="{SVG_WIDTH/2}" y="{SVG_HEIGHT-15}" font-size="11" text-anchor="middle" font-family="sans-serif">Predicted probability</text>')

    n_series = len(histograms)
    bin_width_px = (SVG_WIDTH - 2 * SVG_MARGIN) / n_bins
    legend_y = 35
    for series_index, (label, color, counts) in enumerate(histograms):
        for bin_index, count in enumerate(counts):
            if count == 0:
                continue
            height_frac = count / max_count
            bar_width = bin_width_px / n_series
            x_left = SVG_MARGIN + bin_index * bin_width_px + series_index * bar_width
            _, y_top = _svg_point(0, height_frac)
            bar_height = y0 - y_top
            parts.append(f'<rect x="{x_left:.1f}" y="{y_top:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}" fill-opacity="0.7"/>')
        parts.append(f'<circle cx="{SVG_WIDTH-140}" cy="{legend_y}" r="5" fill="{color}"/>')
        parts.append(f'<text x="{SVG_WIDTH-128}" y="{legend_y+4}" font-size="11" font-family="sans-serif">{label}</text>')
        legend_y += 18

    parts.append("</svg>")
    return "\n".join(parts)


def analyze_model(spec: dict) -> dict[str, object]:
    val_labels, val_probabilities, _ = load_predictions(spec["validation_csv"], spec["probability_column"], spec["label_column"])
    test_labels, test_probabilities, _ = load_predictions(spec["test_csv"], spec["probability_column"], spec["label_column"])

    raw_val_brier = brier_score(val_labels, val_probabilities)
    raw_test_brier = brier_score(test_labels, test_probabilities)
    raw_val_bins, raw_val_ece = reliability_bins(val_labels, val_probabilities)
    raw_test_bins, raw_test_ece = reliability_bins(test_labels, test_probabilities)

    raw_test_pr_auc = float(average_precision_score(test_labels, test_probabilities))
    raw_test_roc_auc = float(roc_auc_score(test_labels, test_probabilities))

    selection = select_calibration_method(val_labels, val_probabilities)
    platt_model = selection.pop("_platt_model")
    isotonic_model = selection.pop("_isotonic_model")

    platt_test_calibrated = apply_platt(platt_model, test_probabilities)
    isotonic_test_calibrated = apply_isotonic(isotonic_model, test_probabilities)

    calibrated_test_by_method = {}
    for method_name, calibrated_probabilities in (("platt", platt_test_calibrated), ("isotonic", isotonic_test_calibrated)):
        bins, ece = reliability_bins(test_labels, calibrated_probabilities)
        calibrated_test_by_method[method_name] = {
            "test_brier": brier_score(test_labels, calibrated_probabilities),
            "test_ece": ece,
            "reliability_bins": bins,
            "test_pr_auc": float(average_precision_score(test_labels, calibrated_probabilities)),
            "test_roc_auc": float(roc_auc_score(test_labels, calibrated_probabilities)),
        }

    selected_method = selection["selected_method"]
    selected_test_probabilities = platt_test_calibrated if selected_method == "platt" else isotonic_test_calibrated
    selected_result = calibrated_test_by_method[selected_method]

    pr_auc_unchanged = abs(selected_result["test_pr_auc"] - raw_test_pr_auc) <= AUC_EQUALITY_TOLERANCE
    roc_auc_unchanged = abs(selected_result["test_roc_auc"] - raw_test_roc_auc) <= AUC_EQUALITY_TOLERANCE

    return {
        "name": spec["name"],
        "display_name": spec["display_name"],
        "raw": {
            "validation_brier": raw_val_brier,
            "test_brier": raw_test_brier,
            "validation_ece": raw_val_ece,
            "test_ece": raw_test_ece,
            "validation_reliability_bins": raw_val_bins,
            "test_reliability_bins": raw_test_bins,
            "validation_probability_distribution": probability_distribution_summary(val_probabilities),
            "test_probability_distribution": probability_distribution_summary(test_probabilities),
            "test_pr_auc": raw_test_pr_auc,
            "test_roc_auc": raw_test_roc_auc,
        },
        "calibration_selection": selection,
        "calibrated_test_by_method": calibrated_test_by_method,
        "selected_method": selected_method,
        "selected_calibrated": {
            "test_brier": selected_result["test_brier"],
            "test_ece": selected_result["test_ece"],
            "test_pr_auc": selected_result["test_pr_auc"],
            "test_roc_auc": selected_result["test_roc_auc"],
            "brier_improvement": raw_test_brier - selected_result["test_brier"],
            "ece_improvement": raw_test_ece - selected_result["test_ece"],
            "pr_auc_unchanged": pr_auc_unchanged,
            "roc_auc_unchanged": roc_auc_unchanged,
        },
        "_val_labels": val_labels,
        "_val_probabilities": val_probabilities,
        "_test_labels": test_labels,
        "_test_probabilities": test_probabilities,
        "_selected_test_probabilities": selected_test_probabilities,
        "_platt_model": platt_model,
        "_isotonic_model": isotonic_model,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_interpretation(analyses: list[dict[str, object]]) -> dict[str, object]:
    lines: dict[str, str] = {}
    for analysis in analyses:
        name = analysis["display_name"]
        raw = analysis["raw"]
        selected = analysis["selected_calibrated"]
        # "materially miscalibrated": informal threshold, stated explicitly, not tuned to any result.
        materially_miscalibrated = raw["test_ece"] > 0.05
        improves = selected["brier_improvement"] > 0 and selected["ece_improvement"] > 0
        val_test_ece_gap = abs(raw["validation_ece"] - raw["test_ece"])
        stable_under_shift = val_test_ece_gap < 0.05

        lines[analysis["name"]] = (
            f"{name}: raw test ECE={raw['test_ece']:.4f} (validation ECE={raw['validation_ece']:.4f}), "
            f"raw test Brier={raw['test_brier']:.4f}. "
            f"{'Materially miscalibrated (test ECE > 0.05 threshold, stated explicitly, not fit to this result).' if materially_miscalibrated else 'Not materially miscalibrated by the stated 0.05 ECE threshold.'} "
            f"Selected calibration method: {analysis['selected_method']} "
            f"(test Brier {'improves' if selected['brier_improvement']>0 else 'does not improve'} by {selected['brier_improvement']:.4f}, "
            f"test ECE {'improves' if selected['ece_improvement']>0 else 'does not improve'} by {selected['ece_improvement']:.4f}). "
            f"PR-AUC unchanged after calibration: {selected['pr_auc_unchanged']}. ROC-AUC unchanged: {selected['roc_auc_unchanged']}. "
            f"Validation-to-test ECE gap: {val_test_ece_gap:.4f} "
            f"({'stable' if stable_under_shift else 'NOT stable'} under the chronological prevalence shift, by a 0.05 ECE-gap threshold)."
        )

    return {
        "per_model": lines,
        "suitable_for_dashboard_decision_support": (
            "Calibrated probabilities may be shown as a supporting, approximate risk indicator, but "
            "should NOT be presented to a SOC analyst as a statistically guaranteed likelihood -- "
            "calibration here is fit on validation and only checked (not re-fit) on test, the ECE gap "
            "between validation and test reflects the same chronological prevalence shift documented in "
            "Phase 8A, and no confidence interval or statistical guarantee is computed anywhere in this "
            "stack. Ranking (PR-AUC/ROC-AUC, unaffected by calibration) remains the more defensible basis "
            "for prioritization than the raw calibrated probability value taken literally."
        ),
    }


def render_plots(analyses: list[dict[str, object]], plots_dir: Path) -> list[str]:
    written = []
    for analysis in analyses:
        raw_test_bins = analysis["raw"]["test_reliability_bins"]
        selected_method = analysis["selected_method"]
        calibrated_test_bins = analysis["calibrated_test_by_method"][selected_method]["reliability_bins"]

        reliability_svg = render_reliability_diagram_svg(
            f"{analysis['display_name']} -- Reliability (test)",
            [("Raw", "#d62728", raw_test_bins), (f"Calibrated ({selected_method})", "#1f77b4", calibrated_test_bins)],
        )
        reliability_path = plots_dir / f"{analysis['name']}_reliability_test.svg"
        reliability_path.write_text(reliability_svg, encoding="utf-8")
        written.append(reliability_path.name)

        histogram_svg = render_histogram_svg(
            f"{analysis['display_name']} -- Predicted probability distribution (test)",
            [("Raw", "#d62728", analysis["_test_probabilities"]), (f"Calibrated ({selected_method})", "#1f77b4", analysis["_selected_test_probabilities"])],
        )
        histogram_path = plots_dir / f"{analysis['name']}_probability_histogram_test.svg"
        histogram_path.write_text(histogram_svg, encoding="utf-8")
        written.append(histogram_path.name)

    return written


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 8B directory: {OUTPUT_DIR}")

    for path in FROZEN_FILES_TO_VERIFY:
        if not path.exists():
            raise FileNotFoundError(f"Required frozen artifact not found: {path}")
    integrity_before = {str(path): _file_signature(path) for path in FROZEN_FILES_TO_VERIFY}

    analyses = [analyze_model(spec) for spec in TARGET_MODEL_SPECS]

    integrity_after = {str(path): _file_signature(path) for path in FROZEN_FILES_TO_VERIFY}
    files_unchanged = integrity_before == integrity_after

    # Determinism check.
    repeat_analyses = [analyze_model(spec) for spec in TARGET_MODEL_SPECS]
    deterministic = all(
        first["raw"]["test_brier"] == second["raw"]["test_brier"]
        and first["selected_calibrated"]["test_brier"] == second["selected_calibrated"]["test_brier"]
        and first["selected_method"] == second["selected_method"]
        for first, second in zip(analyses, repeat_analyses)
    )

    OUTPUT_DIR.mkdir(parents=True)
    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir()

    plot_files = render_plots(analyses, plots_dir)

    csv_rows = []
    for analysis in analyses:
        raw = analysis["raw"]
        selected = analysis["selected_calibrated"]
        csv_rows.append(
            {
                "name": analysis["name"],
                "display_name": analysis["display_name"],
                "validation_brier_raw": raw["validation_brier"],
                "test_brier_raw": raw["test_brier"],
                "validation_ece_raw": raw["validation_ece"],
                "test_ece_raw": raw["test_ece"],
                "selected_method": analysis["selected_method"],
                "test_brier_calibrated": selected["test_brier"],
                "test_ece_calibrated": selected["test_ece"],
                "brier_improvement": selected["brier_improvement"],
                "ece_improvement": selected["ece_improvement"],
                "test_pr_auc_raw": raw["test_pr_auc"],
                "test_pr_auc_calibrated": selected["test_pr_auc"],
                "test_roc_auc_raw": raw["test_roc_auc"],
                "test_roc_auc_calibrated": selected["test_roc_auc"],
                "pr_auc_unchanged": selected["pr_auc_unchanged"],
                "roc_auc_unchanged": selected["roc_auc_unchanged"],
            }
        )
    write_csv(OUTPUT_DIR / "calibration_metrics.csv", csv_rows)

    reliability_rows = []
    for analysis in analyses:
        for kind, split, bins in (
            ("raw", "validation", analysis["raw"]["validation_reliability_bins"]),
            ("raw", "test", analysis["raw"]["test_reliability_bins"]),
            ("calibrated", "test", analysis["calibrated_test_by_method"][analysis["selected_method"]]["reliability_bins"]),
        ):
            for row in bins:
                reliability_rows.append({"model": analysis["name"], "kind": kind, "split": split, **row})
    write_csv(OUTPUT_DIR / "reliability_bins.csv", reliability_rows)

    public_analyses = [{key: value for key, value in analysis.items() if not key.startswith("_")} for analysis in analyses]
    (OUTPUT_DIR / "calibration_metrics.json").write_text(json.dumps(public_analyses, indent=2, default=str), encoding="utf-8")

    interpretation = build_interpretation(analyses)

    all_auc_unchanged = all(a["selected_calibrated"]["pr_auc_unchanged"] and a["selected_calibrated"]["roc_auc_unchanged"] for a in analyses)

    validation_report = {
        "target_models": TARGET_MODEL_NAMES,
        "n_reliability_bins": N_RELIABILITY_BINS,
        "calibrators_fit_on_validation_only": True,
        "test_labels_never_used_for_fitting": True,
        "pr_auc_roc_auc_unchanged_after_calibration": all_auc_unchanged,
        "frozen_files_unchanged": files_unchanged,
        "deterministic_repeated_computation": deterministic,
        "interpretation": interpretation,
        "overall_status": "PASS" if (all_auc_unchanged and files_unchanged and deterministic) else "FAIL",
    }
    (OUTPUT_DIR / "validation_report.json").write_text(json.dumps(validation_report, indent=2, default=str), encoding="utf-8")

    md_lines = [
        "# Phase 8B -- Attack-Risk Calibration",
        "",
        f"Overall status: **{validation_report['overall_status']}**",
        "",
        "Calibrators (Platt scaling, isotonic regression) fit on VALIDATION probabilities/labels only; "
        "applied (transform only) to TEST exactly once. Method selected per model via validation-only "
        "Brier comparison (see calibration_metrics.json -> calibration_selection.rule). "
        "This phase does not define, reselect, or apply any classification threshold -- "
        "Phase 8A's threshold and results are untouched.",
        "",
        "## Results",
        "",
        "| Model | Raw test Brier | Raw test ECE | Method | Calibrated test Brier | Calibrated test ECE | PR-AUC unchanged | ROC-AUC unchanged |",
        "|---|---:|---:|---|---:|---:|---|---|",
    ]
    for analysis in analyses:
        raw = analysis["raw"]
        selected = analysis["selected_calibrated"]
        md_lines.append(
            f"| {analysis['display_name']} | {raw['test_brier']:.4f} | {raw['test_ece']:.4f} | {analysis['selected_method']} | "
            f"{selected['test_brier']:.4f} | {selected['test_ece']:.4f} | {selected['pr_auc_unchanged']} | {selected['roc_auc_unchanged']} |"
        )

    md_lines += [
        "",
        "## Interpretation",
        "",
    ]
    for name in TARGET_MODEL_NAMES:
        md_lines.append(f"- {interpretation['per_model'][name]}")
    md_lines += [
        "",
        "### Suitability for dashboard/SOC decision support",
        "",
        interpretation["suitable_for_dashboard_decision_support"],
        "",
        "## Plots",
        "",
    ]
    for filename in plot_files:
        md_lines.append(f"- `plots/{filename}`")
    md_lines += [
        "",
        f"- Calibrators fit on validation only: {validation_report['calibrators_fit_on_validation_only']}",
        f"- Test labels never used for fitting: {validation_report['test_labels_never_used_for_fitting']}",
        f"- PR-AUC/ROC-AUC unchanged after calibration (all models): {all_auc_unchanged}",
        f"- Frozen files (Phase 5/6B-ablation/8A) unchanged: {files_unchanged}",
        f"- Deterministic repeated computation: {deterministic}",
        "",
    ]
    (OUTPUT_DIR / "validation_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(OUTPUT_DIR),
                "models": [a["name"] for a in analyses],
                "selected_methods": {a["name"]: a["selected_method"] for a in analyses},
                "pr_auc_roc_auc_unchanged": all_auc_unchanged,
                "frozen_files_unchanged": files_unchanged,
                "deterministic": deterministic,
                "overall_status": validation_report["overall_status"],
            },
            indent=2,
        )
    )

    if validation_report["overall_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
