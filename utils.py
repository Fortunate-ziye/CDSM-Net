import os
import random

import numpy as np
import torch
from scipy.signal import find_peaks


def calculate_metrics_batch(preds, targets, threshold=0.5, time_tol=0.5, freq=100):
    """Calculate batch-level TP, FP, FN, and residuals for P and S picks."""
    tol_samples = int(time_tol * freq)
    metrics = {
        "tp_p": 0,
        "fp_p": 0,
        "fn_p": 0,
        "residuals_p": [],
        "true_values_p": [],
        "tp_s": 0,
        "fp_s": 0,
        "fn_s": 0,
        "residuals_s": [],
        "true_values_s": [],
    }

    preds_np = preds.detach().float().cpu().numpy()
    targets_np = targets.detach().float().cpu().numpy()
    batch_size = preds_np.shape[0]

    for i in range(batch_size):
        for phase_idx, phase_name in enumerate(["p", "s"], 1):
            pred_trace = preds_np[i, phase_idx, :]
            true_trace = targets_np[i, phase_idx, :]

            true_peak = np.argmax(true_trace)
            has_true_event = true_trace[true_peak] > 0.5

            pred_peak = np.argmax(pred_trace)
            pred_conf = pred_trace[pred_peak]
            has_pred_event = pred_conf > threshold

            if has_true_event:
                if has_pred_event:
                    diff = np.abs(pred_peak - true_peak)
                    if diff <= tol_samples:
                        metrics[f"tp_{phase_name}"] += 1
                        metrics[f"residuals_{phase_name}"].append((true_peak - pred_peak) / freq)
                        metrics[f"true_values_{phase_name}"].append(true_peak / freq)
                    else:
                        metrics[f"fp_{phase_name}"] += 1
                        metrics[f"fn_{phase_name}"] += 1
                else:
                    metrics[f"fn_{phase_name}"] += 1
            elif has_pred_event:
                metrics[f"fp_{phase_name}"] += 1

    return metrics


def compute_statistical_metrics(residuals, true_values):
    """Compute mean, standard deviation, MAE, and MAPE from residuals."""
    if len(residuals) == 0:
        return {"Mean": 0.0, "Std": 0.0, "MAE": 0.0, "MAPE": 0.0}

    res_arr = np.array(residuals)
    true_arr = np.array(true_values)

    mu = np.mean(res_arr)
    sigma = np.std(res_arr)
    mae = np.mean(np.abs(res_arr))
    mape = np.mean(np.abs(res_arr) / (np.abs(true_arr) + 1e-8)) * 100

    return {"Mean": mu, "Std": sigma, "MAE": mae, "MAPE": mape}


def setup_seed(seed=42):
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print(f"[Info] Random seed set to: {seed}")


def trigger_onset(data, threshold_on, threshold_off):
    """Find detection windows from a probability sequence."""
    on_inds = np.where(data > threshold_on)[0]
    if len(on_inds) == 0:
        return []

    diffs = np.diff(on_inds)
    split_inds = np.where(diffs > 1)[0]

    starts = np.concatenate(([on_inds[0]], on_inds[split_inds + 1]))
    ends = np.concatenate((on_inds[split_inds], [on_inds[-1]]))

    real_ends = []
    for end in ends:
        curr = end
        while curr < len(data) and data[curr] > threshold_off:
            curr += 1
        real_ends.append(curr)

    return [[start, end] for start, end in zip(starts, real_ends) if end > start]


def detect_peaks(x, height, distance=10):
    """Find peaks in a probability curve."""
    peaks, properties = find_peaks(x, height=height, distance=distance)
    return peaks, properties["peak_heights"]


def seismic_picker(
    det_prob,
    p_prob,
    s_prob,
    det_thresh=0.5,
    p_thresh=0.3,
    s_thresh=0.3,
    trigger_off_thresh=0.2,
    buffer=20,
    min_p_dist=50,
    min_s_dist=50,
):
    """
    Convert detection and phase probability curves into event-level picks.

    Multiple P peaks can be assigned within one detection window. For each P
    pick, the highest-confidence S peak after the P arrival is selected.
    """
    events = trigger_onset(det_prob, det_thresh, trigger_off_thresh)
    picked_events = []

    all_p_peaks, p_props = find_peaks(p_prob, height=p_thresh, distance=min_p_dist)
    all_s_peaks, s_props = find_peaks(s_prob, height=s_thresh, distance=min_s_dist)

    all_p_heights = p_props["peak_heights"]
    all_s_heights = s_props["peak_heights"]

    for start, end in events:
        search_start = max(0, start - buffer)
        search_end = min(len(det_prob), end + buffer)
        det_conf = np.mean(det_prob[start:end])

        valid_p_mask = (all_p_peaks >= search_start) & (all_p_peaks <= search_end)
        window_p_peaks = all_p_peaks[valid_p_mask]
        window_p_probs = all_p_heights[valid_p_mask]

        valid_s_mask = (all_s_peaks >= search_start) & (all_s_peaks <= search_end)
        window_s_peaks = all_s_peaks[valid_s_mask]
        window_s_probs = all_s_heights[valid_s_mask]

        if len(window_p_peaks) == 0 and len(window_s_peaks) == 0:
            continue

        if len(window_p_peaks) > 0:
            for i, p_idx in enumerate(window_p_peaks):
                p_conf = window_p_probs[i]
                candidates_s_mask = window_s_peaks > p_idx
                candidates_s = window_s_peaks[candidates_s_mask]
                candidates_s_probs = window_s_probs[candidates_s_mask]

                best_s_idx = None
                best_s_conf = 0.0
                if len(candidates_s) > 0:
                    best_arg_s = np.argmax(candidates_s_probs)
                    best_s_idx = candidates_s[best_arg_s]
                    best_s_conf = candidates_s_probs[best_arg_s]

                picked_events.append(
                    {
                        "start": start,
                        "end": end,
                        "det_conf": det_conf,
                        "p_idx": p_idx,
                        "p_conf": p_conf,
                        "s_idx": best_s_idx,
                        "s_conf": best_s_conf,
                    }
                )
        else:
            best_arg_s = np.argmax(window_s_probs)
            picked_events.append(
                {
                    "start": start,
                    "end": end,
                    "det_conf": det_conf,
                    "p_idx": None,
                    "p_conf": 0.0,
                    "s_idx": window_s_peaks[best_arg_s],
                    "s_conf": window_s_probs[best_arg_s],
                }
            )

    return picked_events
