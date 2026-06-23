import gc
import multiprocessing
import os
from functools import partial

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


ORIGINAL_CSV_PATH = "merge.csv"
ORIGINAL_HDF5_PATH = "merge.hdf5"
OUTPUT_DIR = "./datasets/"
PROCESSED_H5_NAME = "merge.h5"

TRACE_LEN = 6000
NOISE_RATIO = 0.1
BATCH_SIZE = 512
NUM_WORKERS = 4


def process_batch_worker(df_chunk, h5_path):
    """Read raw STEAD waveforms and return a batch in [N, 3, T] layout."""
    trace_names = df_chunk["trace_name"].values
    batch_size = len(df_chunk)

    waveforms = np.zeros((batch_size, 3, TRACE_LEN), dtype=np.float32)
    valid_names = []

    with h5py.File(h5_path, "r") as file:
        for i, name in enumerate(trace_names):
            try:
                dataset = file.get(f"data/{name}")
                if dataset is not None:
                    waveforms[i] = dataset[()].T
            except Exception:
                pass
            valid_names.append(name)

    del df_chunk
    return waveforms, np.array(valid_names, dtype="S")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("[Info] Building processed STEAD HDF5 and event-based splits.")

    df = pd.read_csv(ORIGINAL_CSV_PATH, low_memory=False)
    for col in ["p_arrival_sample", "s_arrival_sample", "coda_end_sample"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    eq_df = (
        df[df["trace_category"] == "earthquake_local"]
        .dropna(subset=["p_arrival_sample", "s_arrival_sample"])
        .copy()
    )
    noise_df = df[df["trace_category"] == "noise"].copy()

    n_noise = int(len(eq_df) * (NOISE_RATIO / (1 - NOISE_RATIO)))
    if len(noise_df) > n_noise:
        noise_df = noise_df.sample(n=n_noise, random_state=42)

    combined_df = pd.concat([eq_df, noise_df], axis=0).reset_index(drop=True)
    total_samples = len(combined_df)
    print(f"[Info] Total samples to process: {total_samples}")

    output_h5_path = os.path.join(OUTPUT_DIR, PROCESSED_H5_NAME)
    chunks = [combined_df[i : i + BATCH_SIZE] for i in range(0, total_samples, BATCH_SIZE)]

    with h5py.File(output_h5_path, "w") as file_out:
        print("[Info] Creating compressed HDF5 file.")
        dset_wave = file_out.create_dataset(
            "waveforms",
            (total_samples, 3, TRACE_LEN),
            dtype=np.float32,
            compression="gzip",
            chunks=(1, 3, TRACE_LEN),
        )
        dset_names = file_out.create_dataset(
            "trace_names",
            (total_samples,),
            dtype=h5py.special_dtype(vlen=str),
        )

        if NUM_WORKERS > 0:
            pool = multiprocessing.Pool(processes=NUM_WORKERS)
            mapper = pool.imap
        else:
            pool = None
            mapper = map

        func = partial(process_batch_worker, h5_path=ORIGINAL_HDF5_PATH)
        start_idx = 0
        for wave, names in tqdm(mapper(func, chunks), total=len(chunks)):
            end_idx = start_idx + len(wave)
            dset_wave[start_idx:end_idx] = wave
            dset_names[start_idx:end_idx] = names
            start_idx = end_idx
            gc.collect()

        if pool:
            pool.close()
            pool.join()

    print(f"[Info] HDF5 file saved to: {output_h5_path}")

    combined_df["h5_index"] = np.arange(total_samples)
    df_eq = combined_df[combined_df["trace_category"] == "earthquake_local"]
    df_noise = combined_df[combined_df["trace_category"] == "noise"]

    # Event-based splitting reduces leakage between train, validation, and test.
    unique_events = df_eq["source_id"].unique()
    train_ids, test_ids = train_test_split(unique_events, test_size=0.10, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.05, random_state=42)

    train_eq = df_eq[df_eq["source_id"].isin(train_ids)]
    val_eq = df_eq[df_eq["source_id"].isin(val_ids)]
    test_eq = df_eq[df_eq["source_id"].isin(test_ids)]

    train_noise, test_noise = train_test_split(df_noise, test_size=0.10, random_state=42)
    train_noise, val_noise = train_test_split(train_noise, test_size=0.05, random_state=42)

    train_df = pd.concat([train_eq, train_noise]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = pd.concat([val_eq, val_noise]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.concat([test_eq, test_noise]).sample(frac=1, random_state=42).reset_index(drop=True)

    keep_cols = [
        "trace_name",
        "trace_category",
        "source_id",
        "p_arrival_sample",
        "s_arrival_sample",
        "coda_end_sample",
        "h5_index",
    ]
    if "snr_db" in combined_df.columns:
        keep_cols.append("snr_db")

    train_df[keep_cols].to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val_df[keep_cols].to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    test_df[keep_cols].to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print(
        "[Info] Processing completed. "
        f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
