import argparse
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from dataset import STEADDataset
from engine import validate
from loss import Loss
from Model.model import CDSMNet
from plot import plot_error_distribution, plot_event_level_confusion_matrix


def load_model(checkpoint_path, device):
    """Load a full saved model or a checkpoint/state-dict file."""
    loaded_obj = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(loaded_obj, torch.nn.Module):
        print("[Info] Loaded a full PyTorch model object.")
        return loaded_obj.to(device).eval()

    if isinstance(loaded_obj, dict):
        model = CDSMNet().to(device)
        state_dict = loaded_obj.get("model_state_dict", loaded_obj)
        model.load_state_dict(state_dict)
        print("[Info] Loaded a model state dictionary.")
        return model.eval()

    raise TypeError(f"Unsupported checkpoint format: {type(loaded_obj)}")


def build_report(val_loss, metrics):
    avg_f1 = (metrics["p_f1"] + metrics["s_f1"]) / 2
    width_metric, width_val, width_desc = 12, 12, 28
    line = (
        "+"
        + "-" * width_metric
        + "+"
        + "-" * width_val
        + "+"
        + "-" * width_val
        + "+"
        + "-" * width_val
        + "+"
        + "-" * width_desc
        + "+"
    )

    rows = [
        f"\n[Evaluation Result] Val Loss: {val_loss:.4f} | Avg F1: {avg_f1:.4f}",
        line,
        f"|{'Metric':^{width_metric}}|{'Detection':^{width_val}}|{'P-Wave':^{width_val}}|{'S-Wave':^{width_val}}|{'Description':^{width_desc}}|",
        line,
        f"|{'Accuracy':<{width_metric}}|{metrics.get('d_acc', 0):<{width_val}.4f}|{metrics['p_acc']:<{width_val}.4f}|{metrics['s_acc']:<{width_val}.4f}|{'(TP + TN) / Total':<{width_desc}}|",
        f"|{'Precision':<{width_metric}}|{metrics.get('d_precision', 0):<{width_val}.4f}|{metrics['p_precision']:<{width_val}.4f}|{metrics['s_precision']:<{width_val}.4f}|{'TP / (TP + FP)':<{width_desc}}|",
        f"|{'Recall':<{width_metric}}|{metrics.get('d_recall', 0):<{width_val}.4f}|{metrics['p_recall']:<{width_val}.4f}|{metrics['s_recall']:<{width_val}.4f}|{'TP / (TP + FN)':<{width_desc}}|",
        f"|{'F1-Score':<{width_metric}}|{metrics.get('d_f1', 0):<{width_val}.4f}|{metrics['p_f1']:<{width_val}.4f}|{metrics['s_f1']:<{width_val}.4f}|{'Harmonic mean':<{width_desc}}|",
        f"|{'MAE (s)':<{width_metric}}|{'N/A':<{width_val}}|{metrics.get('p_mae', 0):<{width_val}.4f}|{metrics.get('s_mae', 0):<{width_val}.4f}|{'Mean abs error':<{width_desc}}|",
        line + "\n",
    ]
    return "\n".join(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CDSM-Net on a processed dataset.")
    parser.add_argument("--checkpoint", type=str, default="./Model/best_model_loss.pt")
    parser.add_argument("--val_csv", type=str, default="./datasets/test.csv")
    parser.add_argument("--hdf5", type=str, default="./datasets/merge.h5")
    parser.add_argument("--dataset", type=str, default="stead", choices=["stead", "instance"])
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="./eval_results")
    parser.add_argument("--Plabel_width", type=int, default=10)
    parser.add_argument("--Slabel_width", type=int, default=40)
    parser.add_argument("--label_type", type=str, default="gaussian")
    parser.add_argument("--det_thresh", type=float, default=0.5)
    parser.add_argument("--p_thresh", type=float, default=0.4)
    parser.add_argument("--s_thresh", type=float, default=0.35)
    parser.add_argument("--freq", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.save_dir, f"eval_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    print(f"[Info] Evaluation results will be saved to: {output_path}")

    val_ds = STEADDataset(
        args.val_csv,
        args.hdf5,
        label_type=args.label_type,
        Plabel_width=args.Plabel_width,
        Slabel_width=args.Slabel_width,
        dataset_format=args.dataset,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print(f"[Info] Loading model from {args.checkpoint} ...")
    model = load_model(args.checkpoint, device)
    criterion = Loss(task_weights=[0.05, 0.40, 0.55]).to(device)

    print("[Info] Starting inference...")
    val_loss, metrics, extra_data = validate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        freq=args.freq,
        det_thresh=args.det_thresh,
        p_thresh=args.p_thresh,
        s_thresh=args.s_thresh,
    )
    res_p, res_s, d_true, d_pred = extra_data

    dist_wandb = plot_error_distribution(res_p, res_s, metrics)
    dist_wandb.image.save(os.path.join(output_path, "evaluation_error_dist.png"))

    cm_wandb = plot_event_level_confusion_matrix(d_true, d_pred)
    cm_wandb.image.save(os.path.join(output_path, "evaluation_confusion_matrix.png"))

    final_report = build_report(val_loss, metrics)
    print(final_report)

    with open(os.path.join(output_path, "metrics_report.txt"), "w", encoding="utf-8") as file:
        file.write(final_report)

    print(f"[Info] Evaluation completed. Files saved to: {output_path}")


if __name__ == "__main__":
    main()
