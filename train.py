import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from augmentation import SeismicAugmentor
from dataset import STEADDataset
from engine import train_epoch, validate
from loss import Loss
from Model.model import CDSMNet
from utils import setup_seed


def build_parser():
    parser = argparse.ArgumentParser(description="Train CDSM-Net for seismic event detection and phase picking.")

    parser.add_argument("--train_csv", type=str, default="datasets/train.csv", help="Training metadata CSV.")
    parser.add_argument("--val_csv", type=str, default="datasets/val.csv", help="Validation metadata CSV.")
    parser.add_argument("--hdf5", type=str, default="datasets/merge.h5", help="Processed waveform HDF5 file.")
    parser.add_argument("--dataset_format", type=str, default="stead", choices=["stead", "instance"])
    parser.add_argument("--save_path", type=str, default=None, help="Root directory for runs and logs.")
    parser.add_argument("--project_name", type=str, default="CDSM-Net", help="WandB project name.")
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="disabled",
        choices=["online", "offline", "disabled"],
        help="WandB logging mode.",
    )

    parser.add_argument("--batch_size", type=int, default=450, help="Training batch size.")
    parser.add_argument("--base_lr", type=float, default=1.5e-3, help="Reference learning rate.")
    parser.add_argument("--weight_decay", type=float, default=5e-6, help="AdamW weight decay.")
    parser.add_argument("--epochs", type=int, default=60, help="Maximum number of training epochs.")
    parser.add_argument("--num_workers", type=int, default=12, help="DataLoader worker count.")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs.")
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience by validation loss.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save_freq", type=int, default=10, help="Periodic checkpoint interval; 0 disables it.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability.")

    parser.add_argument("--p_stack", type=float, default=0.6, help="Probability of stacking a secondary event.")
    parser.add_argument("--p_noise", type=float, default=0.6, help="Probability of adding Gaussian noise.")
    parser.add_argument("--p_shift", type=float, default=0.99, help="Probability of circular time shift.")
    parser.add_argument("--p_gap", type=float, default=0.2, help="Probability of inserting random gaps.")
    parser.add_argument("--p_drop", type=float, default=0.3, help="Probability of dropping waveform channels.")
    parser.add_argument("--max_shift", type=int, default=1000, help="Maximum circular shift in samples.")
    parser.add_argument("--max_gap_size", type=int, default=1000, help="Maximum random-gap length in samples.")
    parser.add_argument("--snr_min", type=float, default=1.0, help="Minimum injected-noise SNR in dB.")
    parser.add_argument("--snr_max", type=float, default=20.0, help="Maximum injected-noise SNR in dB.")

    parser.add_argument("--resume", type=str, default="", help="Checkpoint path for resuming training.")
    parser.add_argument("--freq", type=int, default=100, help="Sampling rate in Hz.")
    parser.add_argument(
        "--plot_mode",
        type=str,
        default="time",
        choices=["time", "frequency", "time_frequency"],
        help="Validation visualization mode.",
    )
    parser.add_argument(
        "--label_type",
        type=str,
        default="gaussian",
        choices=["gaussian", "triangular", "box"],
        help="Phase-label shape.",
    )
    parser.add_argument("--Plabel_width", type=int, default=10, help="P-label half width in samples.")
    parser.add_argument("--Slabel_width", type=int, default=40, help="S-label half width in samples.")
    parser.add_argument("--det_thresh", type=float, default=0.5, help="Detection threshold.")
    parser.add_argument("--p_thresh", type=float, default=0.4, help="P-pick threshold.")
    parser.add_argument("--s_thresh", type=float, default=0.35, help="S-pick threshold.")
    return parser


def make_loader(dataset, batch_size, shuffle, num_workers, device, prefetch_factor):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def print_metric_table(curr_epoch, total_epochs, train_loss, val_loss, avg_f1, metrics):
    d_acc = metrics.get("d_acc", 0.0)
    d_pre = metrics.get("d_precision", 0.0)
    d_rec = metrics.get("d_recall", 0.0)
    d_f1 = metrics.get("d_f1", 0.0)
    p_pre = metrics.get("p_precision", 0.0)
    p_rec = metrics.get("p_recall", 0.0)
    s_pre = metrics.get("s_precision", 0.0)
    s_rec = metrics.get("s_recall", 0.0)

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

    print(
        f"\n[Epoch {curr_epoch:03d}/{total_epochs:03d}] "
        f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Avg F1: {avg_f1:.4f}"
    )
    print(line)
    print(
        f"|{'Metric':^{width_metric}}|{'Detection':^{width_val}}|"
        f"{'P-Wave':^{width_val}}|{'S-Wave':^{width_val}}|{'Description':^{width_desc}}|"
    )
    print(line)
    print(
        f"|{'Accuracy':<{width_metric}}|{d_acc:<{width_val}.4f}|"
        f"{metrics['p_acc']:<{width_val}.4f}|{metrics['s_acc']:<{width_val}.4f}|"
        f"{'(TP + TN) / Total':<{width_desc}}|"
    )
    print(
        f"|{'Precision':<{width_metric}}|{d_pre:<{width_val}.4f}|"
        f"{p_pre:<{width_val}.4f}|{s_pre:<{width_val}.4f}|"
        f"{'TP / (TP + FP)':<{width_desc}}|"
    )
    print(
        f"|{'Recall':<{width_metric}}|{d_rec:<{width_val}.4f}|"
        f"{p_rec:<{width_val}.4f}|{s_rec:<{width_val}.4f}|"
        f"{'TP / (TP + FN)':<{width_desc}}|"
    )
    print(
        f"|{'F1-Score':<{width_metric}}|{d_f1:<{width_val}.4f}|"
        f"{metrics['p_f1']:<{width_val}.4f}|{metrics['s_f1']:<{width_val}.4f}|"
        f"{'Harmonic mean':<{width_desc}}|"
    )
    print(
        f"|{'MAE (s)':<{width_metric}}|{'N/A':<{width_val}}|"
        f"{metrics.get('p_mae', 0):<{width_val}.4f}|{metrics.get('s_mae', 0):<{width_val}.4f}|"
        f"{'Mean abs error':<{width_desc}}|"
    )
    print(line + "\n")


def main():
    args = build_parser().parse_args()

    ref_batch_size = 512
    scaled_lr = args.base_lr * (args.batch_size / ref_batch_size)

    print("\n" + "=" * 40)
    print("Auto learning-rate scaling")
    print("=" * 40)
    print(f"Reference batch size: {ref_batch_size}")
    print(f"Reference LR        : {args.base_lr:.2e}")
    print(f"Current batch size  : {args.batch_size}")
    print(f"Actual LR           : {scaled_lr:.2e}")
    print("=" * 40 + "\n")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    root = Path(__file__).resolve().parents[0]
    if not args.save_path:
        args.save_path = str(root)

    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_epoch = 0
    best_f1 = 0.0
    min_val_loss = float("inf")
    wandb_id = None

    if args.resume and os.path.isfile(args.resume):
        temp = torch.load(args.resume, map_location="cpu")
        run_dir = os.path.dirname(args.resume)
        wandb_id = temp.get("wandb_id", None)
        print(f"[Info] Resuming from: {run_dir}")
        del temp
    else:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(args.save_path, "runs", f"{current_time}_{args.project_name}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"[Info] New run created at: {run_dir}")

    config_dict = vars(args).copy()
    config_dict["scaled_lr"] = scaled_lr
    config_dict["device"] = str(device)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as file:
        json.dump(config_dict, file, indent=4)

    if wandb_id is None:
        wandb_id = wandb.util.generate_id()
    wandb.init(
        project=args.project_name,
        name=os.path.basename(run_dir),
        config=config_dict,
        id=wandb_id,
        resume="allow",
        dir=run_dir,
        mode=args.wandb_mode,
    )

    print("[Info] Loading datasets...")
    train_ds = STEADDataset(
        args.train_csv,
        args.hdf5,
        label_type=args.label_type,
        transform=None,
        Plabel_width=args.Plabel_width,
        Slabel_width=args.Slabel_width,
        dataset_format=args.dataset_format,
    )
    val_ds = STEADDataset(
        args.val_csv,
        args.hdf5,
        label_type=args.label_type,
        transform=None,
        Plabel_width=args.Plabel_width,
        Slabel_width=args.Slabel_width,
        dataset_format=args.dataset_format,
    )

    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, device, prefetch_factor=3)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers, device, prefetch_factor=2)

    model = CDSMNet(dropout=args.dropout).to(device)
    total_params = sum(param.numel() for param in model.parameters())
    param_size_mb = total_params * 4 / (1024**2)

    print("\n" + "=" * 40)
    print(f"Model Summary: {model.__class__.__name__}")
    print(f"Total Params : {total_params / 1e6:.2f} M")
    print(f"Model Size   : {param_size_mb:.2f} MB")
    print("=" * 40 + "\n")

    arch_path = os.path.join(run_dir, "model_arch.txt")
    with open(arch_path, "w", encoding="utf-8") as file:
        file.write(str(model))
    wandb.save(arch_path, base_path=run_dir)

    augmentor = SeismicAugmentor(
        p_stack=args.p_stack,
        p_noise=args.p_noise,
        p_shift=args.p_shift,
        p_gap=args.p_gap,
        p_drop=args.p_drop,
        max_shift=args.max_shift,
        max_gap_size=args.max_gap_size,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    criterion = Loss(task_weights=[0.05, 0.40, 0.55]).to(device)

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=max(args.warmup_epochs, 1),
    )
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs - args.warmup_epochs, 1),
        eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[args.warmup_epochs],
    )

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        best_f1 = ckpt.get("best_f1", 0.0)
        print(f"[Info] Resumed from epoch {start_epoch}; best F1 = {best_f1:.4f}")

    print(f"[Info] Start training with LR={scaled_lr:.2e}")
    no_improve_cnt = 0

    for epoch in range(start_epoch, args.epochs):
        curr_epoch = epoch + 1

        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            curr_epoch,
            augmentor=augmentor,
        )

        val_loss, metrics, _ = validate(
            model,
            val_loader,
            criterion,
            device,
            augmentor=augmentor,
            freq=args.freq,
            plot_mode=args.plot_mode,
            det_thresh=args.det_thresh,
            p_thresh=args.p_thresh,
            s_thresh=args.s_thresh,
        )

        avg_f1 = (metrics["p_f1"] + metrics["s_f1"]) / 2
        print_metric_table(curr_epoch, args.epochs, train_loss, val_loss, avg_f1, metrics)

        wandb.log(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": curr_epoch,
                "avg_f1": avg_f1,
                "metrics/p_f1": metrics["p_f1"],
                "metrics/p_pre": metrics.get("p_precision", 0.0),
                "metrics/p_rec": metrics.get("p_recall", 0.0),
                "metrics/p_mae": metrics.get("p_mae", 0.0),
                "metrics/s_f1": metrics["s_f1"],
                "metrics/s_pre": metrics.get("s_precision", 0.0),
                "metrics/s_rec": metrics.get("s_recall", 0.0),
                "metrics/s_mae": metrics.get("s_mae", 0.0),
                "val_d_loss": metrics.get("avg_d_loss", 0),
                "val_p_loss": metrics.get("avg_p_loss", 0),
                "val_s_loss": metrics.get("avg_s_loss", 0),
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        scheduler.step()

        ckpt = {
            "epoch": curr_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_f1": best_f1,
            "wandb_id": wandb_id,
        }
        torch.save(ckpt, os.path.join(run_dir, "last.pt"))

        if args.save_freq > 0 and curr_epoch % args.save_freq == 0:
            periodic_path = os.path.join(run_dir, f"epoch_{curr_epoch}.pt")
            torch.save(ckpt, periodic_path)
            print(f"[Info] Periodic checkpoint saved to: {periodic_path}")

        if avg_f1 > best_f1:
            best_f1 = avg_f1
            no_improve_cnt = 0
            best_f1_path = os.path.join(run_dir, "best_model.pt")
            wandb.run.summary["best_avg_f1"] = best_f1
            wandb.run.summary["best_epoch"] = curr_epoch
            for key, value in metrics.items():
                wandb.run.summary[f"best/{key}"] = value
            torch.save(model, best_f1_path)
            print(f"[Info] New best-F1 model saved to: {best_f1_path} (F1: {best_f1:.4f})")

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_loss_path = os.path.join(run_dir, "best_model_loss.pt")
            torch.save(model, best_loss_path)
            no_improve_cnt = 0
            print(f"[Info] New best-loss model saved to: {best_loss_path} (Loss: {min_val_loss:.4f})")
        else:
            no_improve_cnt += 1

        if no_improve_cnt >= args.patience:
            print(f"[Info] Early stopping at epoch {curr_epoch}; best validation loss = {min_val_loss:.4f}")
            break

    wandb.finish()


if __name__ == "__main__":
    main()
