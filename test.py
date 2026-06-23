import argparse
import os
from contextlib import nullcontext

import pandas as pd
import torch

from dataset import STEADDataset
from plot import plot_single_panel


def load_model(model_path, device):
    """Load either a full PyTorch model object or a checkpoint dictionary."""
    obj = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(obj, torch.nn.Module):
        return obj.to(device).eval()
    raise TypeError(
        "The supplied file is not a full model object. "
        "This inference script expects a model saved with torch.save(model, path)."
    )


def plot_trace_local(
    trace_name,
    model_path,
    csv_path,
    hdf5_path,
    save_dir="./inference_results",
    device="cuda",
    dataset_format="stead",
):
    """Run inference for one trace and save a local probability-panel figure."""
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")

    print(f"[Info] Searching trace: {trace_name}")
    df = pd.read_csv(csv_path)
    if "trace_name" not in df.columns:
        raise ValueError(f"CSV file does not contain a 'trace_name' column: {csv_path}")

    match_rows = df[df["trace_name"] == trace_name]
    if len(match_rows) == 0:
        print(f"[Warning] Trace not found in CSV: {trace_name}")
        return None

    dataset_idx = match_rows.index[0]
    dataset = STEADDataset(
        metadata_path=csv_path,
        hdf5_path=hdf5_path,
        label_type="gaussian",
        dataset_format=dataset_format,
    )
    waveform_tensor, target_tensor = dataset[dataset_idx]

    print(f"[Info] Loading model: {model_path}")
    model = load_model(model_path, device)

    print("[Info] Running inference...")
    with torch.no_grad():
        waveforms_batch = waveform_tensor.unsqueeze(0).to(device)
        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with autocast_ctx:
            logits = model(waveforms_batch)
            if isinstance(logits, dict):
                final_logits = torch.cat([logits["det"], logits["p"], logits["s"]], dim=1)
            else:
                final_logits = logits

        probs = torch.sigmoid(final_logits).squeeze(0).float().cpu().numpy()

    image = plot_single_panel(
        waveform=waveform_tensor.numpy(),
        target=target_tensor.numpy(),
        pred_prob=probs,
        title_suffix=f"Trace: {trace_name}",
        sample_rate=100,
        plot_mode="time",
        clean_mode=False,
    ).image

    safe_name = trace_name.replace("/", "_").replace("\\", "_").replace(".", "_")
    save_path = os.path.join(save_dir, f"{safe_name}_result.png")
    image.save(save_path)
    print(f"[Info] Figure saved to: {save_path}")
    return save_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run CDSM-Net inference for one waveform trace.")
    parser.add_argument("--trace_name", type=str, required=True, help="Trace name in the metadata CSV.")
    parser.add_argument("--model_path", type=str, default="Model/best_model_loss.pt")
    parser.add_argument("--csv_path", type=str, default="datasets/test.csv")
    parser.add_argument("--hdf5_path", type=str, default="datasets/merge.h5")
    parser.add_argument("--save_dir", type=str, default="./inference_results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset_format", type=str, default="stead", choices=["stead", "instance"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_trace_local(
        trace_name=args.trace_name,
        model_path=args.model_path,
        csv_path=args.csv_path,
        hdf5_path=args.hdf5_path,
        save_dir=args.save_dir,
        device=args.device,
        dataset_format=args.dataset_format,
    )
