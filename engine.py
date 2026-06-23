from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from plot import log_fig4_to_wandb, plot_error_distribution, plot_event_level_confusion_matrix
from utils import seismic_picker


def _autocast_context(device):
    if torch.device(device).type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _normalize_waveforms(waveforms):
    mean = torch.mean(waveforms, dim=2, keepdim=True)
    std = torch.std(waveforms, dim=2, keepdim=True)
    return (waveforms - mean) / (std + 1e-6)


def _split_outputs(outputs):
    if isinstance(outputs, dict):
        return torch.cat([outputs["det"], outputs["p"], outputs["s"]], dim=1)
    return outputs


def train_epoch(model, loader, criterion, optimizer, device, epoch, augmentor=None):
    """Run one training epoch."""
    model.train()
    if augmentor:
        augmentor.train()

    metrics = {"loss": 0.0, "d_loss": 0.0, "p_loss": 0.0, "s_loss": 0.0}
    valid_batches = 0

    with tqdm(loader, desc=f"Training Epoch {epoch}") as loop:
        for batch_idx, (waveforms, targets) in enumerate(loop):
            waveforms = waveforms.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if torch.isnan(waveforms).any():
                continue

            if augmentor:
                with torch.no_grad():
                    waveforms, targets = augmentor(waveforms, targets)
            else:
                waveforms = _normalize_waveforms(waveforms)

            optimizer.zero_grad(set_to_none=True)

            with _autocast_context(device):
                logits = model(waveforms)

                found_nan = False
                if isinstance(logits, dict):
                    found_nan = any(torch.isnan(value).any() for value in logits.values())
                else:
                    found_nan = torch.isnan(logits).any()

                if found_nan:
                    print(f"[Warning] Epoch {epoch} batch {batch_idx}: NaN output skipped.")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                loss = criterion(logits, targets)

                with torch.no_grad():
                    d_l, p_l, s_l = 0.0, 0.0, 0.0
                    if isinstance(logits, dict):
                        if "det" in logits:
                            d_l = F.binary_cross_entropy_with_logits(
                                logits["det"],
                                targets[:, 0:1, :],
                            ).item()
                        if "p" in logits:
                            p_l = F.binary_cross_entropy_with_logits(
                                logits["p"],
                                targets[:, 1:2, :],
                            ).item()
                        if "s" in logits:
                            s_l = F.binary_cross_entropy_with_logits(
                                logits["s"],
                                targets[:, 2:3, :],
                            ).item()
                    elif logits.shape[1] >= 3:
                        d_l = F.binary_cross_entropy_with_logits(
                            logits[:, 0, :],
                            targets[:, 0, :],
                        ).item()
                        p_l = F.binary_cross_entropy_with_logits(
                            logits[:, 1, :],
                            targets[:, 1, :],
                        ).item()
                        s_l = F.binary_cross_entropy_with_logits(
                            logits[:, 2, :],
                            targets[:, 2, :],
                        ).item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            metrics["loss"] += loss.item()
            metrics["d_loss"] += d_l
            metrics["p_loss"] += p_l
            metrics["s_loss"] += s_l
            valid_batches += 1

            loop.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Det": f"{d_l:.3f}",
                    "P": f"{p_l:.3f}",
                    "S": f"{s_l:.3f}",
                }
            )

    return metrics["loss"] / valid_batches if valid_batches > 0 else 0.0


def validate(
    model,
    loader,
    criterion,
    device,
    augmentor=None,
    freq=100,
    plot_mode="time_frequency",
    det_thresh=0.5,
    p_thresh=0.3,
    s_thresh=0.3,
):
    """
    Validate the model and return loss, metrics, and plotting data.

    A P or S pick is counted as correct when the predicted arrival is within
    0.5 s of the manual arrival.
    """
    model.eval()
    if augmentor:
        augmentor.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    metrics = {"loss": 0.0, "d_loss": 0.0, "p_loss": 0.0, "s_loss": 0.0}
    p_detect_true, p_detect_pred = [], []
    s_detect_true, s_detect_pred = [], []
    detection_true, detection_pred = [], []

    tp_p, fp_p, fn_p = 0, 0, 0
    tp_s, fp_s, fn_s = 0, 0, 0
    residuals_p, residuals_s = [], []

    tolerance_samples = int(0.5 * freq)
    has_plotted = False
    valid_batches = 0

    with torch.no_grad():
        with tqdm(loader, desc="Validating") as loop:
            for _, (waveforms, targets) in enumerate(loop):
                waveforms = waveforms.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                if torch.isnan(waveforms).any():
                    continue

                if augmentor:
                    waveforms, targets = augmentor(waveforms, targets)
                else:
                    waveforms = _normalize_waveforms(waveforms)

                with _autocast_context(device):
                    logits = model(waveforms)
                    loss = criterion(logits, targets)
                    final_logits = _split_outputs(logits)

                    if isinstance(logits, dict):
                        if "det" in logits:
                            metrics["d_loss"] += F.binary_cross_entropy_with_logits(
                                logits["det"],
                                targets[:, 0:1, :],
                            ).item()
                        if "p" in logits:
                            metrics["p_loss"] += F.binary_cross_entropy_with_logits(
                                logits["p"],
                                targets[:, 1:2, :],
                            ).item()
                        if "s" in logits:
                            metrics["s_loss"] += F.binary_cross_entropy_with_logits(
                                logits["s"],
                                targets[:, 2:3, :],
                            ).item()
                    elif logits.shape[1] >= 3:
                        metrics["d_loss"] += F.binary_cross_entropy_with_logits(
                            logits[:, 0, :],
                            targets[:, 0, :],
                        ).item()
                        metrics["p_loss"] += F.binary_cross_entropy_with_logits(
                            logits[:, 1, :],
                            targets[:, 1, :],
                        ).item()
                        metrics["s_loss"] += F.binary_cross_entropy_with_logits(
                            logits[:, 2, :],
                            targets[:, 2, :],
                        ).item()

                metrics["loss"] += loss.item()
                valid_batches += 1
                loop.set_postfix({"Val Loss": f"{loss.item():.4f}"})

                if not has_plotted and wandb.run is not None:
                    try:
                        log_fig4_to_wandb(
                            model,
                            waveforms,
                            targets,
                            device,
                            freq=freq,
                            plot_mode=plot_mode,
                        )
                        has_plotted = True
                    except Exception as exc:
                        print(f"[Warning] Plotting skipped: {exc}")

                probs = torch.sigmoid(final_logits).detach().float().cpu()
                targets_cpu = targets.detach().float().cpu()

                probs_np = probs.numpy()
                targets_np = targets_cpu.numpy()

                batch_true_max = targets_cpu.view(targets_cpu.size(0), -1).max(dim=1).values
                detection_true.extend((batch_true_max > 0.5).long().tolist())

                batch_pred_max = probs.view(probs.size(0), -1).max(dim=1).values
                detection_pred.extend((batch_pred_max > det_thresh).long().tolist())

                p_detect_true.extend(
                    (targets_cpu[:, 1, :].max(dim=1).values > 0.5).long().tolist()
                )
                s_detect_true.extend(
                    (targets_cpu[:, 2, :].max(dim=1).values > 0.5).long().tolist()
                )
                p_detect_pred.extend(
                    (probs[:, 1, :].max(dim=1).values > p_thresh).long().tolist()
                )
                s_detect_pred.extend(
                    (probs[:, 2, :].max(dim=1).values > s_thresh).long().tolist()
                )

                for i in range(probs_np.shape[0]):
                    picks = seismic_picker(
                        probs_np[i, 0],
                        probs_np[i, 1],
                        probs_np[i, 2],
                        det_thresh=det_thresh,
                        p_thresh=p_thresh,
                        s_thresh=s_thresh,
                        min_s_dist=50,
                        min_p_dist=50,
                    )

                    gt_p_trace = targets_np[i, 1]
                    gt_s_trace = targets_np[i, 2]
                    gt_p_idx = np.argmax(gt_p_trace) if gt_p_trace.max() > 0.5 else None
                    gt_s_idx = np.argmax(gt_s_trace) if gt_s_trace.max() > 0.5 else None

                    if gt_p_idx is not None:
                        cand = [pick["p_idx"] for pick in picks if pick["p_idx"]]
                        if cand:
                            diffs = np.abs(np.array(cand) - gt_p_idx)
                            idx = np.argmin(diffs)
                            if diffs[idx] <= tolerance_samples:
                                tp_p += 1
                                residuals_p.append((cand[idx] - gt_p_idx) / freq)
                            else:
                                fp_p += 1
                                fn_p += 1
                        else:
                            fn_p += 1
                    elif any(pick["p_idx"] for pick in picks):
                        fp_p += 1

                    if gt_s_idx is not None:
                        cand = [pick["s_idx"] for pick in picks if pick["s_idx"]]
                        if cand:
                            diffs = np.abs(np.array(cand) - gt_s_idx)
                            idx = np.argmin(diffs)
                            if diffs[idx] <= tolerance_samples:
                                tp_s += 1
                                residuals_s.append((cand[idx] - gt_s_idx) / freq)
                            else:
                                fp_s += 1
                                fn_s += 1
                        else:
                            fn_s += 1
                    elif any(pick["s_idx"] for pick in picks):
                        fp_s += 1

                del waveforms, logits, probs, targets, final_logits

    results = {}
    denom = max(valid_batches, 1)
    results["avg_loss"] = metrics["loss"] / denom
    results["avg_d_loss"] = metrics["d_loss"] / denom
    results["avg_p_loss"] = metrics["p_loss"] / denom
    results["avg_s_loss"] = metrics["s_loss"] / denom

    results["d_acc"] = accuracy_score(detection_true, detection_pred)
    results["d_precision"] = precision_score(detection_true, detection_pred, zero_division=0)
    results["d_recall"] = recall_score(detection_true, detection_pred, zero_division=0)
    results["d_f1"] = f1_score(detection_true, detection_pred, zero_division=0)

    results["p_acc"] = accuracy_score(p_detect_true, p_detect_pred)
    results["s_acc"] = accuracy_score(s_detect_true, s_detect_pred)

    def calc_metrics(tp, fp, fn):
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        return precision, recall, f1

    results["p_precision"], results["p_recall"], results["p_f1"] = calc_metrics(tp_p, fp_p, fn_p)
    results["s_precision"], results["s_recall"], results["s_f1"] = calc_metrics(tp_s, fp_s, fn_s)

    def safe_stats(res_list):
        if len(res_list) == 0:
            return 0.0, 0.0, 0.0
        return np.mean(res_list), np.mean(np.abs(res_list)), np.std(res_list)

    results["p_mean_error"], results["p_mae"], results["p_std_dev"] = safe_stats(residuals_p)
    results["s_mean_error"], results["s_mae"], results["s_std_dev"] = safe_stats(residuals_s)

    if wandb.run is not None:
        try:
            wandb.log(
                {
                    "Error Distribution": plot_error_distribution(residuals_p, residuals_s, results),
                    "Detection Confusion Matrix": plot_event_level_confusion_matrix(
                        detection_true,
                        detection_pred,
                    ),
                },
                commit=False,
            )
        except Exception as exc:
            print(f"[Warning] Summary plotting skipped: {exc}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        results["avg_loss"],
        results,
        (residuals_p, residuals_s, detection_true, detection_pred),
    )
