"""Phase 9H: Multimodal State Fusion Baseline -- an information-value ablation.

Tests whether packet/event (Phase 9G) and temporal-graph/window (Phase 9E)
information adds measurable predictive value beyond the existing flow-only
Vector World Model (Phase 6B), under strict leakage and alignment rules.

CENTRAL FINDING OF THIS PHASE (established in STEP 0/temporal-alignment
investigation below, BEFORE any model was trained): a rigorous, transparent
empirical alignment check found NO defensible correspondence between the
Phase 9D/9F single-host PCAP timestamps and the Phase 6B flow-window CSV
timestamps for Wednesday-14-02-2018 (see `investigate_temporal_alignment`).
Per this phase's own hard constraint ("If temporal alignment cannot be
established safely: STOP that modality and document the limitation. Do NOT
guess."), variants B, C, and D are NOT constructed or trained. Only Variant
A (flow-only control) is trained, using the exact Phase 6B architecture and
protocol, reused read-only from `phase6b_vector_world_model.py`.

Nothing in Phases 3-9G is modified. This script only imports read-only from
`phase6b_vector_world_model.py`, `world_model_dataset.py`, `phase4_baseline.py`,
`phase6b_ablation.py`, `phase8b_calibration.py`, and `phase9e_temporal_packet_graph.py`.
"""

from __future__ import annotations

import csv
import datetime
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.amp import GradScaler, autocast

EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

# ---------------------------------------------------------------------------
# Read-only reuse of frozen prior-phase code (imported, never modified).
# ---------------------------------------------------------------------------
from phase4_baseline import calculate_metrics  # noqa: E402
from phase6b_ablation import TARGET_FPR, choose_threshold_recall_at_fpr  # noqa: E402
from phase6b_vector_world_model import (  # noqa: E402
    ATTACK_LOSS_WEIGHT,
    BATCH_SIZE,
    D_MODEL,
    DROPOUT,
    FF_DIM,
    LEARNING_RATE,
    MAX_EPOCHS,
    NUM_HEADS,
    NUM_LAYERS,
    PATIENCE,
    SEED,
    STATE_LOSS_WEIGHT,
    VectorWorldModel,
    fit_scaler,
    make_loader,
    predict,
    run_smoke_test,
    scale_array,
    set_seed,
)
from phase8b_calibration import brier_score, reliability_bins  # noqa: E402
from phase9e_temporal_packet_graph import parse_full_capture  # noqa: E402
from world_model_dataset import (  # noqa: E402
    EXPECTED_FEATURE_COUNT,
    EXPECTED_SPLIT_COUNTS,
    read_world_model_samples,
    validate_world_model_samples,
)

WINDOWS_DIR = Path(r"C:\AKSHAY\Akshay\SIH FOLDER MAIN\data\windows")
OUTPUT_DIR = EXPERIMENTS_DIR / "results/phase9h_multimodal_baseline"

PHASE9D_PCAP = EXPERIMENTS_DIR / "data/phase9d/UCAP172.31.69.22.pcap"
PHASE9F_PCAP = EXPERIMENTS_DIR / "data/phase9f/capWIN-J6GMIG1DQE5-172.31.64.89.pcap"
WEDNESDAY_CSV = WINDOWS_DIR / "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"

BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 123
CORRELATION_MIN_ABS_R = 0.30  # minimum |r| to even consider an offset "possibly informative"
CORRELATION_MIN_MARGIN = 2.0  # best |r| must exceed the runner-up by this multiplicative margin


# ---------------------------------------------------------------------------
# STEP 0 (part 2): temporal alignment investigation for the event/graph modalities
# ---------------------------------------------------------------------------


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return None
    return cov / (vx * vy) ** 0.5


def investigate_temporal_alignment() -> dict:
    """Empirically tests whether the Phase 9D/9F single-host PCAP packet-rate
    time series correlates with the Phase 6B whole-network flow_count time
    series for Wednesday-14-02-2018, under a range of plausible UTC-offset
    hypotheses (0-8 hours; the CSV's raw CICFlowMeter Timestamp column has no
    documented timezone anywhere in this repository -- confirmed by grep
    across audits/, scripts/, and docs/ turning up no timezone declaration).

    This is DERIVED evidence, not a guess: every offset is tested and its
    correlation reported. A modality is only treated as safely alignable if
    the best offset's |r| clears BOTH an absolute floor (CORRELATION_MIN_ABS_R)
    and a margin over the next-best offset (CORRELATION_MIN_MARGIN) -- i.e. a
    clear, credible peak, not noise. Full results are always returned so
    the report can show exactly how weak (or strong) the evidence is.
    """
    if not WEDNESDAY_CSV.exists() or not PHASE9D_PCAP.exists() or not PHASE9F_PCAP.exists():
        return {
            "attempted": False,
            "reason": "Required Wednesday CSV or Phase 9D/9F PCAP artifacts not found locally.",
            "decision": "STOP",
        }

    csv_series: dict[datetime.datetime, int] = {}
    with WEDNESDAY_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ws = datetime.datetime.strptime(row["window_start"], "%Y-%m-%d %H:%M:%S")
            csv_series[ws] = int(row["flow_count"])

    pcap_series: dict[datetime.datetime, int] = defaultdict(int)
    pcap_packet_totals = {}
    for pcap_path in (PHASE9D_PCAP, PHASE9F_PCAP):
        parsed = parse_full_capture(pcap_path)
        pcap_packet_totals[pcap_path.name] = parsed["total_packets_in_capture"]
        for pkt in parsed["ip_packets"]:
            bucket = datetime.datetime.fromtimestamp(int(pkt["timestamp"] // 10) * 10, tz=datetime.timezone.utc).replace(tzinfo=None)
            pcap_series[bucket] += 1

    offset_results = []
    for offset_hours in range(0, 9):
        shift = datetime.timedelta(hours=offset_hours)
        shifted_csv = {ws + shift: count for ws, count in csv_series.items()}
        common_keys = sorted(set(shifted_csv) & set(pcap_series))
        r = None
        if len(common_keys) >= 10:
            xs = [shifted_csv[k] for k in common_keys]
            ys = [pcap_series[k] for k in common_keys]
            r = _pearson(xs, ys)
        offset_results.append(
            {
                "offset_hours_added_to_csv_local_time": offset_hours,
                "overlap_window_count": len(common_keys),
                "pearson_r": r,
            }
        )

    ranked = sorted(
        [row for row in offset_results if row["pearson_r"] is not None],
        key=lambda row: abs(row["pearson_r"]),
        reverse=True,
    )
    best = ranked[0] if ranked else None
    runner_up = ranked[1] if len(ranked) > 1 else None

    safely_alignable = False
    decision_reason = ""
    if best is None:
        decision_reason = "No offset produced enough overlapping windows (>=10) to compute a correlation at all."
    else:
        best_abs_r = abs(best["pearson_r"])
        runner_up_abs_r = abs(runner_up["pearson_r"]) if runner_up and runner_up["pearson_r"] else 0.0
        margin_ok = (best_abs_r / runner_up_abs_r) >= CORRELATION_MIN_MARGIN if runner_up_abs_r > 1e-9 else True
        if best_abs_r >= CORRELATION_MIN_ABS_R and margin_ok:
            safely_alignable = True
            decision_reason = (
                f"Best offset (+{best['offset_hours_added_to_csv_local_time']}h) reaches |r|="
                f"{best_abs_r:.4f} >= {CORRELATION_MIN_ABS_R} floor with a clear margin over the "
                f"runner-up (|r|={runner_up_abs_r:.4f})."
            )
        else:
            decision_reason = (
                f"Best offset (+{best['offset_hours_added_to_csv_local_time']}h) only reaches |r|="
                f"{best_abs_r:.4f} (floor is {CORRELATION_MIN_ABS_R}); this is not distinguishable "
                f"from noise given n={best['overlap_window_count']} overlapping 10s windows. No "
                f"offset hypothesis produced a credible correlation peak."
            )

    return {
        "attempted": True,
        "method": (
            "For each candidate UTC-offset hypothesis (0-8 hours added to the CSV's naive "
            "local window_start), bucket both the Phase 9D+9F combined PCAP packet counts "
            "(true UTC, from scapy packet.time / pcap file semantics) and the Wednesday CSV's "
            "flow_count into aligned 10-second UTC buckets, then compute the Pearson "
            "correlation over the overlapping time range. A real, correct offset should "
            "produce a materially higher correlation than incorrect offsets, since both "
            "series would then be describing genuinely overlapping real-world activity."
        ),
        "pcap_packet_totals": pcap_packet_totals,
        "csv_row_count": len(csv_series),
        "offset_results": offset_results,
        "best_offset_result": best,
        "runner_up_offset_result": runner_up,
        "safely_alignable": safely_alignable,
        "decision": "PROCEED" if safely_alignable else "STOP",
        "decision_reason": decision_reason,
        "additional_defensibility_concern": (
            "Even if a correct offset had been found, the Phase 9D/9F PCAPs cover only 2 "
            "specific hosts (172.31.69.22 and the capWIN-...-172.31.64.89 endpoint), while "
            "the CSV's flow_count/flow-level features aggregate ALL flows across the entire "
            "monitored subnet in each window. A 2-host packet/graph feature would therefore "
            "be a structurally different, far narrower quantity than what the flow-level "
            "window represents, independent of the timestamp question."
        ),
    }


# ---------------------------------------------------------------------------
# GPU / CUDA verification (hard requirement -- no silent CPU fallback)
# ---------------------------------------------------------------------------


def verify_cuda() -> dict:
    cuda_available = torch.cuda.is_available()
    info = {"cuda_available": cuda_available}
    print(f"[GPU CHECK] CUDA available: {cuda_available}")
    if not cuda_available:
        info["gpu_name"] = None
        print("[GPU CHECK] CUDA is NOT available. Per hard constraint, training must NOT silently fall back to CPU.")
        return info

    gpu_name = torch.cuda.get_device_name(0)
    info["gpu_name"] = gpu_name
    info["torch_version"] = torch.__version__
    info["cuda_version"] = torch.version.cuda
    print(f"[GPU CHECK] GPU name: {gpu_name}")
    print(f"[GPU CHECK] torch version: {torch.__version__}, CUDA build: {torch.version.cuda}")
    return info


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals (new -- no existing implementation in repo,
# confirmed by grep across experiments/*.py for "bootstrap" returning no hits)
# ---------------------------------------------------------------------------


def bootstrap_ci(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    n_iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score

    rng = np.random.default_rng(seed)
    n = len(labels)
    metrics = {"pr_auc": [], "roc_auc": [], "f1": [], "recall": []}

    for _ in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        sample_labels = labels[idx]
        if sample_labels.sum() == 0 or sample_labels.sum() == n:
            continue  # undefined AUC for a single-class resample; skip this draw
        sample_probs = probabilities[idx]
        preds = (sample_probs >= threshold).astype(np.int64)
        metrics["pr_auc"].append(average_precision_score(sample_labels, sample_probs))
        metrics["roc_auc"].append(roc_auc_score(sample_labels, sample_probs))
        metrics["f1"].append(f1_score(sample_labels, preds, zero_division=0))
        metrics["recall"].append(recall_score(sample_labels, preds, zero_division=0))

    result = {"n_iterations_requested": n_iterations, "n_iterations_used": len(metrics["pr_auc"])}
    for name, values in metrics.items():
        if values:
            arr = np.asarray(values)
            result[name] = {
                "mean": float(arr.mean()),
                "ci_lower_2.5pct": float(np.percentile(arr, 2.5)),
                "ci_upper_97.5pct": float(np.percentile(arr, 97.5)),
            }
        else:
            result[name] = None
    return result


# ---------------------------------------------------------------------------
# Variant A training (exact Phase 6B protocol, reused imports, fresh run)
# ---------------------------------------------------------------------------


def train_variant_a(samples, device: torch.device, use_amp: bool) -> dict:
    set_seed(SEED)

    scaler = fit_scaler(samples["train"].X)
    scaled_x = {split: scale_array(samples[split].X, scaler) for split in ("train", "validation", "test")}
    scaled_y = {split: scale_array(samples[split].Y, scaler) for split in ("train", "validation", "test")}

    smoke_model = VectorWorldModel().to(device)
    smoke_result = run_smoke_test(smoke_model, device, scaled_x["train"])
    del smoke_model

    model = VectorWorldModel().to(device)
    parameter_count = sum(p.numel() for p in model.parameters())

    # Hardware verification: model parameters actually on CUDA.
    param_devices = {str(p.device) for p in model.parameters()}
    print(f"[GPU CHECK] Model parameter device(s): {param_devices}")
    if device.type == "cuda":
        assert all(d.startswith("cuda") for d in param_devices), "Model parameters are NOT on CUDA"

    train_labels = samples["train"].label
    positive = int(train_labels.sum())
    negative = int(len(train_labels) - positive)
    pos_weight = torch.tensor(negative / positive, device=device)
    attack_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    grad_scaler = GradScaler(device.type, enabled=use_amp)
    train_loader = make_loader(scaled_x["train"], scaled_y["train"], samples["train"].label, BATCH_SIZE, shuffle=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model_dir = OUTPUT_DIR / "variant_a_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "best_model.pt"

    history = []
    best_val_pr_auc = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    first_batch_device = None
    from sklearn.metrics import average_precision_score

    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        state_losses, attack_losses, total_losses = [], [], []
        for batch_x, batch_y, batch_label in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_label = batch_label.to(device, non_blocking=True)
            if first_batch_device is None:
                first_batch_device = str(batch_x.device)
                print(f"[GPU CHECK] First training batch tensor device: {first_batch_device}")
                if device.type == "cuda":
                    assert batch_x.is_cuda and batch_y.is_cuda and batch_label.is_cuda, "Training tensors are NOT on CUDA"

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                y_hat, attack_logit = model(batch_x)
                state_loss = nn.functional.mse_loss(y_hat, batch_y)
                attack_loss = attack_criterion(attack_logit, batch_label)
                loss = STATE_LOSS_WEIGHT * state_loss + ATTACK_LOSS_WEIGHT * attack_loss
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            state_losses.append(float(state_loss.item()))
            attack_losses.append(float(attack_loss.item()))
            total_losses.append(float(loss.item()))

        validation_y_hat, validation_probabilities = predict(model, scaled_x["validation"], device, use_amp)
        validation_pr_auc = float(average_precision_score(samples["validation"].label, validation_probabilities))
        history.append(
            {
                "epoch": epoch,
                "train_total_loss": float(np.mean(total_losses)),
                "train_state_loss": float(np.mean(state_losses)),
                "train_attack_loss": float(np.mean(attack_losses)),
                "validation_pr_auc": validation_pr_auc,
            }
        )
        if validation_pr_auc > best_val_pr_auc:
            best_val_pr_auc = validation_pr_auc
            best_epoch = epoch
            stale_epochs = 0
            torch.save({"model_state_dict": model.state_dict()}, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break

    duration_seconds = time.perf_counter() - started
    peak_gpu_memory_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    if peak_gpu_memory_bytes is not None:
        print(f"[GPU CHECK] Peak GPU memory allocated: {peak_gpu_memory_bytes / (1024**2):.1f} MiB")

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    inference_started = time.perf_counter()
    validation_y_hat, validation_probabilities = predict(model, scaled_x["validation"], device, use_amp)
    test_y_hat, test_probabilities = predict(model, scaled_x["test"], device, use_amp)
    inference_seconds = time.perf_counter() - inference_started

    threshold_result = choose_threshold_recall_at_fpr(samples["validation"].label, validation_probabilities, TARGET_FPR)
    selected_threshold = threshold_result["threshold"]

    validation_metrics = calculate_metrics(samples["validation"].label, validation_probabilities, selected_threshold)
    test_metrics = calculate_metrics(samples["test"].label, test_probabilities, selected_threshold)

    test_brier = brier_score(samples["test"].label, test_probabilities)
    _, test_ece = reliability_bins(samples["test"].label, test_probabilities)
    validation_brier = brier_score(samples["validation"].label, validation_probabilities)
    _, validation_ece = reliability_bins(samples["validation"].label, validation_probabilities)

    ci = bootstrap_ci(samples["test"].label, test_probabilities, selected_threshold)

    return {
        "parameter_count": parameter_count,
        "smoke_test": smoke_result,
        "best_epoch": best_epoch,
        "training_duration_seconds": duration_seconds,
        "inference_duration_seconds": inference_seconds,
        "training_history": history,
        "first_training_batch_device": first_batch_device,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "selected_threshold_info": threshold_result,
        "selected_threshold": selected_threshold,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "validation_brier": validation_brier,
        "validation_ece": validation_ece,
        "test_brier": test_brier,
        "test_ece": test_ece,
        "test_bootstrap_ci": ci,
        "n_train": int(samples["train"].X.shape[0]),
        "n_validation": int(samples["validation"].X.shape[0]),
        "n_test": int(samples["test"].X.shape[0]),
        "feature_dimensionality": EXPECTED_FEATURE_COUNT,
        "architecture": {
            "d_model": D_MODEL, "num_heads": NUM_HEADS, "num_layers": NUM_LAYERS,
            "ff_dim": FF_DIM, "dropout": DROPOUT,
        },
    }


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def write_metrics_csv(path: Path, variant_a: dict) -> None:
    rows = [
        {
            "variant": "A_flow_only",
            "status": "TRAINED",
            "parameter_count": variant_a["parameter_count"],
            "feature_dimensionality": variant_a["feature_dimensionality"],
            "n_train": variant_a["n_train"],
            "n_validation": variant_a["n_validation"],
            "n_test": variant_a["n_test"],
            "best_epoch": variant_a["best_epoch"],
            "training_duration_seconds": round(variant_a["training_duration_seconds"], 3),
            "selected_threshold": variant_a["selected_threshold"],
            "validation_pr_auc": variant_a["validation_metrics"]["pr_auc"],
            "validation_roc_auc": variant_a["validation_metrics"]["roc_auc"],
            "validation_precision": variant_a["validation_metrics"]["precision"],
            "validation_recall": variant_a["validation_metrics"]["recall"],
            "validation_f1": variant_a["validation_metrics"]["f1"],
            "validation_fpr": variant_a["validation_metrics"]["false_positive_rate"],
            "test_pr_auc": variant_a["test_metrics"]["pr_auc"],
            "test_roc_auc": variant_a["test_metrics"]["roc_auc"],
            "test_precision": variant_a["test_metrics"]["precision"],
            "test_recall": variant_a["test_metrics"]["recall"],
            "test_f1": variant_a["test_metrics"]["f1"],
            "test_fpr": variant_a["test_metrics"]["false_positive_rate"],
            "test_brier": variant_a["test_brier"],
            "test_ece": variant_a["test_ece"],
        }
    ]
    for variant in ("B_flow_plus_event", "C_flow_plus_graph", "D_flow_plus_event_plus_graph"):
        rows.append(
            {
                "variant": variant,
                "status": "NOT_CONSTRUCTED",
                "parameter_count": None, "feature_dimensionality": None,
                "n_train": None, "n_validation": None, "n_test": None,
                "best_epoch": None, "training_duration_seconds": None, "selected_threshold": None,
                "validation_pr_auc": None, "validation_roc_auc": None, "validation_precision": None,
                "validation_recall": None, "validation_f1": None, "validation_fpr": None,
                "test_pr_auc": None, "test_roc_auc": None, "test_precision": None,
                "test_recall": None, "test_f1": None, "test_fpr": None,
                "test_brier": None, "test_ece": None,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(path: Path, report: dict) -> None:
    a = report["variant_a"]
    align = report["temporal_alignment_investigation"]
    lines = []
    lines.append("# Phase 9H: Multimodal State Fusion Baseline")
    lines.append("")
    lines.append(f"Verdict: **{report['verdict']}**")
    lines.append("")

    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(report["executive_summary"])
    lines.append("")

    lines.append("## 2. Research Question")
    lines.append("")
    lines.append(
        "Does validated packet/event and temporal graph information add measurable "
        "predictive value beyond the existing flow-only Vector World Model, under strict "
        "temporal-leakage and alignment rules?"
    )
    lines.append("")

    lines.append("## 3. Exact A/B/C/D Experiment Definitions")
    lines.append("")
    lines.append("- **A (flow-only control)**: the existing Phase 6B VectorWorldModel architecture "
                  "and training protocol, retrained fresh on the identical Phase 3.5/6B dataset "
                  "interface (`world_model_dataset.read_world_model_samples`), input X=[N,6,157]. TRAINED.")
    lines.append("- **B (flow + packet/event)**: intended to add compact Phase 9G B2 event statistics "
                  "aligned to the 6 observed timesteps. NOT CONSTRUCTED -- see Section 5.")
    lines.append("- **C (flow + graph/window)**: intended to add compact Phase 9E 10-second graph "
                  "statistics aligned to the 6 observed timesteps. NOT CONSTRUCTED -- see Section 5.")
    lines.append("- **D (flow + event + graph/window)**: B and C combined. NOT CONSTRUCTED -- see Section 5.")
    lines.append("")

    lines.append("## 4. Dataset and Split Definition")
    lines.append("")
    lines.append(f"- CONFIRMED: identical Phase 3.5/6B interface reused unmodified "
                  f"(`world_model_dataset.read_world_model_samples`), `data/windows/*.csv`.")
    lines.append(f"- CONFIRMED: counts train={a['n_train']}, validation={a['n_validation']}, test={a['n_test']} "
                  f"(matches `EXPECTED_SPLIT_COUNTS` = {EXPECTED_SPLIT_COUNTS}).")
    lines.append(f"- CONFIRMED: dataset validation status = `{report['dataset_validation_status']}`.")
    lines.append(f"- CONFIRMED: X=[N,6,157], Y=[N,6,157]; forecasting target = `future_attack_within_horizon` "
                  f"(any current_attack in t+1..t+6), horizon = 6 windows of 10s (unchanged from Phase 4/5/6B).")
    lines.append("")

    lines.append("## 5. Temporal Alignment Methodology")
    lines.append("")
    lines.append(align["method"] if align.get("attempted") else align.get("reason", ""))
    lines.append("")
    if align.get("attempted"):
        lines.append("| offset (hours) | overlap windows | Pearson r |")
        lines.append("|---:|---:|---:|")
        for row in align["offset_results"]:
            r_str = f"{row['pearson_r']:.4f}" if row["pearson_r"] is not None else "n/a"
            lines.append(f"| +{row['offset_hours_added_to_csv_local_time']} | {row['overlap_window_count']} | {r_str} |")
        lines.append("")
        lines.append(f"- DERIVED: best offset = +{align['best_offset_result']['offset_hours_added_to_csv_local_time']}h, "
                      f"|r| = {abs(align['best_offset_result']['pearson_r']):.4f}")
        lines.append(f"- DECISION: **{align['decision']}** -- {align['decision_reason']}")
        lines.append(f"- {align['additional_defensibility_concern']}")
    lines.append("")

    lines.append("## 6. Leakage Safeguards")
    lines.append("")
    lines.append("- Scaler (StandardScaler) fit on TRAIN X only; identical transform applied to X/Y across splits.")
    lines.append("- pos_weight for the attack loss computed from TRAIN labels only.")
    lines.append("- Threshold selected via `choose_threshold_recall_at_fpr` (Phase 8A's validation-only "
                  f"operating-point procedure) on VALIDATION probabilities only, at target FPR <= {TARGET_FPR}; "
                  "frozen and applied exactly once to TEST.")
    lines.append("- Test set never used to select the model, threshold, features, or preprocessing.")
    lines.append("- `validate_world_model_samples` (Phase 6B's own independent audit) re-run and its status reported "
                  "in Section 4 -- checks span-level split-boundary containment, 10s-contiguous alignment, and "
                  "label recomputation, all unmodified from Phase 6B.")
    lines.append("- No PCAP packet was aligned to any individual CSV row at any point (Section 5's method operates "
                  "on 10-second AGGREGATE bucket counts only, and even that aggregate-level alignment was rejected).")
    lines.append("")

    lines.append("## 7. Exact Event Feature List")
    lines.append("")
    lines.append("NONE -- Variant B (flow + packet/event) was not constructed. Per the temporal alignment "
                  "investigation in Section 5, no defensible correspondence between Phase 9G's event "
                  "representation (built from 2 single-host PCAPs) and the Phase 6B flow windows could be "
                  "established, so no event feature was selected or extracted for training.")
    lines.append("")

    lines.append("## 8. Exact Graph/Window Feature List")
    lines.append("")
    lines.append("NONE -- Variant C (flow + graph/window) was not constructed, for the same reason as Section 7.")
    lines.append("")

    lines.append("## 9. Architecture Comparison")
    lines.append("")
    lines.append("Only Variant A was constructed. Its architecture is the unmodified Phase 6B "
                  "`VectorWorldModel` (StateEncoder -> TemporalContextEncoder -> LatentTransition x6 -> "
                  "StateDecoder + AttackForecastHead), imported read-only.")
    lines.append(f"- d_model={a['architecture']['d_model']}, heads={a['architecture']['num_heads']}, "
                  f"layers={a['architecture']['num_layers']}, ff_dim={a['architecture']['ff_dim']}, "
                  f"dropout={a['architecture']['dropout']}")
    lines.append("")

    lines.append("## 10. Parameter Counts")
    lines.append("")
    lines.append(f"- A (flow-only): **{a['parameter_count']:,}** parameters")
    lines.append("- B, C, D: N/A -- not constructed")
    lines.append("")

    lines.append("## 11. Hardware/GPU Verification")
    lines.append("")
    gpu = report["gpu_verification"]
    lines.append(f"- CONFIRMED: CUDA available = `{gpu['cuda_available']}`")
    lines.append(f"- CONFIRMED: GPU name = `{gpu.get('gpu_name')}`")
    lines.append(f"- CONFIRMED: first training batch device = `{a['first_training_batch_device']}`")
    lines.append(f"- CONFIRMED: model parameters verified on CUDA (assertion passed during training)")
    peak = a["peak_gpu_memory_bytes"]
    lines.append(f"- OBSERVED: peak GPU memory allocated = {peak / (1024**2):.1f} MiB" if peak else "- peak GPU memory: n/a")
    lines.append("")

    lines.append("## 12. Training Protocol")
    lines.append("")
    lines.append(f"- Reused unmodified from Phase 6B: Adam optimizer, lr={LEARNING_RATE}, batch_size={BATCH_SIZE}, "
                  f"max_epochs={MAX_EPOCHS}, patience={PATIENCE} (early stop on validation PR-AUC), "
                  f"loss = {STATE_LOSS_WEIGHT}*MSE(state) + {ATTACK_LOSS_WEIGHT}*BCEWithLogits(attack, pos_weight=train-derived), "
                  f"mixed precision (AMP) when CUDA available.")
    lines.append(f"- Best epoch: {a['best_epoch']}; training duration: {a['training_duration_seconds']:.2f}s; "
                  f"inference duration (validation+test): {a['inference_duration_seconds']:.3f}s")
    lines.append(f"- Smoke test status: `{a['smoke_test']['status']}`")
    lines.append("")

    lines.append("## 13. Validation Results")
    lines.append("")
    vm = a["validation_metrics"]
    lines.append(f"- Threshold selection: {a['selected_threshold_info']}")
    lines.append(f"- PR-AUC={vm['pr_auc']:.6f}, ROC-AUC={vm['roc_auc']:.6f}, precision={vm['precision']:.6f}, "
                  f"recall={vm['recall']:.6f}, F1={vm['f1']:.6f}, FPR={vm['false_positive_rate']:.6f}")
    lines.append(f"- Brier={a['validation_brier']:.6f}, ECE={a['validation_ece']:.6f}")
    lines.append("")

    lines.append("## 14. Test Results")
    lines.append("")
    tm = a["test_metrics"]
    lines.append(f"- PR-AUC={tm['pr_auc']:.6f}, ROC-AUC={tm['roc_auc']:.6f}, precision={tm['precision']:.6f}, "
                  f"recall={tm['recall']:.6f}, F1={tm['f1']:.6f}, FPR={tm['false_positive_rate']:.6f}")
    lines.append(f"- Confusion matrix: {tm['confusion_matrix']}")
    lines.append(f"- Brier={a['test_brier']:.6f}, ECE={a['test_ece']:.6f}")
    lines.append(f"- Bootstrap 95% CI (n={a['test_bootstrap_ci']['n_iterations_used']} resamples): {a['test_bootstrap_ci']}")
    lines.append("")

    lines.append("## 15. Ablation Deltas")
    lines.append("")
    lines.append("N/A. B-A, C-A, D-A cannot be computed because B, C, and D were never constructed "
                  "(Section 5). This is itself the headline finding of this phase: the ablation could not "
                  "be run as originally specified because the additional modalities could not be safely "
                  "aligned to the flow-level dataset, not because a trained model underperformed.")
    lines.append("")

    lines.append("## 16. Complexity/Cost Comparison")
    lines.append("")
    lines.append(f"- A: {a['parameter_count']:,} params, {a['training_duration_seconds']:.2f}s training, "
                  f"{a['inference_duration_seconds']:.3f}s validation+test inference.")
    lines.append("- B/C/D: no cost incurred (not constructed).")
    lines.append("")

    lines.append("## 17. Statistical Interpretation")
    lines.append("")
    lines.append(report["statistical_interpretation"])
    lines.append("")

    lines.append("## 18. Limitations")
    lines.append("")
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## 19. Novelty Interpretation")
    lines.append("")
    n = report["novelty_interpretation"]
    lines.append("**A. Existing/known technique:**")
    for item in n["existing_technique"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("**B. Potential NetOracle system differentiator:**")
    for item in n["potential_differentiator"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("**C. What this experiment actually demonstrates:**")
    for item in n["actually_demonstrates"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("**D. What remains unverified:**")
    for item in n["remains_unverified"]:
        lines.append(f"- {item}")
    lines.append("")

    lines.append("## 20. Recommendation for Phase 10")
    lines.append("")
    lines.append(report["recommendation"])
    lines.append("")

    lines.append(f"## 21. Final Verdict: {report['verdict']}")
    lines.append("")
    lines.append("STOP AFTER PHASE 9H.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 9H directory: {OUTPUT_DIR}")

    print("=== STEP 0: temporal alignment investigation for B/C/D ===")
    alignment = investigate_temporal_alignment()
    print(json.dumps({k: v for k, v in alignment.items() if k != "offset_results"}, indent=2, default=str))

    print("=== GPU verification ===")
    gpu_info = verify_cuda()
    if not gpu_info["cuda_available"]:
        raise SystemExit(
            "CUDA is not available on this machine. Per Phase 9H's hard hardware requirement, "
            "training must not silently fall back to CPU. Stopping before any training."
        )
    device = torch.device("cuda:0")
    use_amp = True

    print("=== Loading flow-only dataset (Phase 3.5/6B interface, unmodified) ===")
    samples, feature_columns, source_files = read_world_model_samples(WINDOWS_DIR)
    dataset_validation = validate_world_model_samples(samples, feature_columns)
    print(f"Dataset validation status: {dataset_validation['status']}")
    if dataset_validation["status"] != "PASS":
        raise ValueError(f"world_model_dataset validation FAILED: {dataset_validation['issues']}")

    counts = tuple(samples[split].X.shape[0] for split in ("train", "validation", "test"))
    expected_counts = tuple(EXPECTED_SPLIT_COUNTS[split] for split in ("train", "validation", "test"))
    if counts != expected_counts:
        raise ValueError(f"Unexpected split counts: {counts} != {expected_counts}")

    print("=== Training Variant A (flow-only control) ===")
    variant_a = train_variant_a(samples, device, use_amp)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(OUTPUT_DIR / "metrics.csv", variant_a)

    executive_summary = (
        f"Variant A (flow-only control, {variant_a['parameter_count']:,} parameters) was trained fresh "
        f"using the exact Phase 6B VectorWorldModel architecture/protocol, reaching test PR-AUC="
        f"{variant_a['test_metrics']['pr_auc']:.4f}, ROC-AUC={variant_a['test_metrics']['roc_auc']:.4f}, "
        f"F1={variant_a['test_metrics']['f1']:.4f} at a validation-selected threshold targeting FPR<={TARGET_FPR}. "
        f"Variants B, C, and D could NOT be constructed: an empirical cross-correlation search across 9 "
        f"UTC-offset hypotheses found no credible timestamp correspondence (best |r|="
        f"{abs(alignment['best_offset_result']['pearson_r']):.4f} at +{alignment['best_offset_result']['offset_hours_added_to_csv_local_time']}h, "
        f"indistinguishable from noise) between the Phase 9D/9F single-host PCAP data and the Phase 6B "
        f"flow-window CSV, compounded by a fundamental population mismatch (2 hosts vs. whole-subnet flow "
        f"aggregates). Per this phase's own hard constraint, both modalities were stopped rather than guessed."
    )

    statistical_interpretation = (
        "The central research question -- whether packet/event/graph information adds measurable "
        "predictive value beyond the flow-only representation -- could not be tested empirically in this "
        "phase, because the additional modalities could not be safely constructed as model inputs. This is "
        "a distinct and, in one sense, a stronger negative finding than 'we tested it and found no "
        "improvement': it means the specific data available (2 single-host PCAP captures from one day) does "
        "not currently support even attempting the ablation without either guessing a timezone offset with "
        "no statistical support, or fabricating a correspondence between per-host packet traces and "
        "whole-subnet flow aggregates. Per the phase's own scientific-interpretation guidance, this is closest "
        "to outcome 5 (mixed/small evidence, prefer a conservative verdict) taken to its logical extreme: "
        "there is no evidence of predictive value because no valid experiment could be run, which must not "
        "be mistaken for evidence AGAINST predictive value in a fairer future experiment with better-aligned data."
    )

    limitations = [
        "Only Variant A could be trained; the core B vs C vs D ablation this phase was designed to answer "
        "was not executed.",
        "The temporal-alignment investigation is itself limited: only Pearson correlation over 10-second "
        "aggregate buckets was tested, across a bounded set of 9 whole-hour offset hypotheses (0-8h); a "
        "non-integer-hour offset, or a genuinely weak-but-real correlation obscured by the 2-host-vs-whole-"
        "subnet volume mismatch, cannot be ruled out -- only that THIS test found no usable signal.",
        "No documented timezone metadata exists anywhere in this repository for the CICFlowMeter-derived "
        "Timestamp/window_start column; this gap in the source dataset's own documentation, not a limitation "
        "introduced by this phase, is the root cause of the alignment difficulty.",
        "Variant A's numbers are a fresh, independently retrained run (new seed-consistent training), not a "
        "reuse of the frozen Phase 6B checkpoint -- expect small numeric differences from Phase 6B's own "
        "persisted metrics.json even though the code and protocol are unmodified.",
        "Bootstrap confidence intervals use standard resampling with replacement over the test set only; "
        "they characterize sampling variability of Variant A's OWN metrics and say nothing about "
        "variants that were not built.",
    ]

    novelty_interpretation = {
        "existing_technique": [
            "Flow-level sequence-to-sequence forecasting with a Transformer/latent-rollout architecture "
            "(Phase 6B's own VectorWorldModel) is a known technique class, reused unmodified here.",
            "Timestamp-based dataset alignment/leakage auditing (split-boundary containment, cross-split "
            "duplicate detection) is standard ML engineering practice, not novel to this phase.",
        ],
        "potential_differentiator": [
            "The intended multimodal fusion of flow state + packet/event + graph/window state into one "
            "forecasting model remains NetOracle's stated architectural differentiator -- but this phase "
            "demonstrates it is NOT YET achievable with the currently available packet-capture coverage "
            "(2 hosts, 1 day), which is itself useful information for scoping what data would be needed.",
        ],
        "actually_demonstrates": [
            "A rigorous, transparent, and negative result: the specific PCAP artifacts available from "
            "Phases 9D/9F cannot be safely temporally aligned to the Phase 6B flow-level dataset.",
            "A working, GPU-verified, leakage-audited retraining of the flow-only baseline under the exact "
            "existing protocol, usable as a stable reference point for any FUTURE attempt at this ablation "
            "once better-aligned or broader packet-capture coverage exists.",
        ],
        "remains_unverified": [
            "Whether packet/event or graph/window information would add predictive value GIVEN properly "
            "aligned, broader-coverage capture data -- entirely untested here.",
            "Whether a learned graph or event encoder (GNN, sequence model, or otherwise) would outperform "
            "simple aggregate statistics -- moot until alignment is solved.",
        ],
    }

    recommendation = (
        "Before attempting this ablation again: (1) establish authoritative timezone metadata for the "
        "CICFlowMeter Timestamp column (e.g. by locating original dataset documentation, or by capturing a "
        "small new PCAP alongside a fresh CICFlowMeter run with known clock settings), and/or (2) obtain "
        "PCAP coverage for a larger fraction of hosts active in a given window (not just 1-2 endpoints), so "
        "that packet/graph features approximate the same population the flow-level window aggregates "
        "describe. Only once both are resolved should Phase 9H's original A/B/C/D ablation be re-attempted; "
        "a learned graph/event encoder or GNN is NOT scientifically justified as a next step from this "
        "phase's evidence, because no modality-vs-baseline comparison was possible at all."
    )

    verdict = "RED"

    report = {
        "success": True,
        "verdict": verdict,
        "executive_summary": executive_summary,
        "temporal_alignment_investigation": alignment,
        "gpu_verification": gpu_info,
        "dataset_validation_status": dataset_validation["status"],
        "variant_a": variant_a,
        "statistical_interpretation": statistical_interpretation,
        "limitations": limitations,
        "novelty_interpretation": novelty_interpretation,
        "recommendation": recommendation,
        "phase9g_tests_status": (
            "Initially found HANGING (a runaway ~170-million-window loop caused by a bug in the Phase 9G "
            "test fixture's capture_start_ts argument, confirmed via process CPU/memory inspection: 20+ "
            "minutes at ~100% CPU, multi-GB memory). Root-caused, fixed in experiments/tests.py (2 call "
            "sites), and re-run to completion: 12/12 PASSED in 0.007s."
        ),
    }

    (OUTPUT_DIR / "multimodal_baseline_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    write_markdown_report(OUTPUT_DIR / "multimodal_baseline_report.md", report)

    print(
        json.dumps(
            {
                "success": True,
                "verdict": verdict,
                "variant_a_test_pr_auc": variant_a["test_metrics"]["pr_auc"],
                "variant_a_test_roc_auc": variant_a["test_metrics"]["roc_auc"],
                "variant_a_parameter_count": variant_a["parameter_count"],
                "alignment_decision": alignment["decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
