import h5py
import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset


class STEADDataset(Dataset):
    """Dataset wrapper for processed STEAD and INSTANCE waveform files."""

    def __init__(
        self,
        metadata_path,
        hdf5_path,
        label_type="gaussian",
        transform=None,
        Plabel_width=15,
        Slabel_width=15,
        dataset_format="stead",
    ):
        self.metadata_path = metadata_path
        self.hdf5_path = hdf5_path
        self.label_type = label_type
        self.transform = transform
        self.dataset_format = dataset_format.lower()
        self.Plabel_width = Plabel_width
        self.Slabel_width = Slabel_width
        self.trace_len = 6000

        sample_rate = 100.0
        nyquist = 0.5 * sample_rate
        low = 1.0 / nyquist
        high = 45.0 / nyquist
        self.b, self.a = butter(4, [low, high], btype="bandpass")

        print(f"Loading metadata from {metadata_path}...")
        cols = ["h5_index", "p_arrival_sample", "s_arrival_sample", "coda_end_sample"]
        df = pd.read_csv(metadata_path, usecols=cols)

        self.indices = df["h5_index"].values.astype(np.int64)
        self.p_arrivals = df["p_arrival_sample"].values.astype(np.float32)
        self.s_arrivals = df["s_arrival_sample"].values.astype(np.float32)

        if df["coda_end_sample"].dtype == "object":
            self.coda_ends = (
                df["coda_end_sample"]
                .astype(str)
                .str.replace("[", "", regex=False)
                .str.replace("]", "", regex=False)
                .astype(np.float32)
                .values
            )
        else:
            self.coda_ends = df["coda_end_sample"].values.astype(np.float32)

        self.x_grid = np.arange(self.trace_len)
        self.dtfl = None
        self.waveforms_ds = None

    def __len__(self):
        return len(self.indices)

    def __del__(self):
        if getattr(self, "dtfl", None) is not None:
            self.dtfl.close()

    def _open_hdf5(self):
        if self.dtfl is None:
            self.dtfl = h5py.File(self.hdf5_path, "r", rdcc_nbytes=10 * 1024 * 1024)
            self.waveforms_ds = self.dtfl["waveforms"]

    def _make_label(self, center, width):
        start = int(max(0, center - width))
        end = int(min(self.trace_len, center + width + 1))
        if start >= end:
            return start, end, np.array([])

        grid_slice = self.x_grid[start:end]

        if self.label_type == "triangular":
            dists = np.abs(grid_slice - center)
            label_shape = np.clip(1.0 - dists / width, 0, 1)
        elif self.label_type == "gaussian":
            sigma = width / 4.0
            label_shape = np.exp(-((grid_slice - center) ** 2) / (2 * sigma**2))
        elif self.label_type == "box":
            label_shape = np.ones_like(grid_slice, dtype=np.float32)
        else:
            raise ValueError(f"Unknown label type: {self.label_type}")

        return start, end, label_shape

    def _generate_label(self, p_idx, s_idx, coda_idx):
        target = np.zeros((3, self.trace_len), dtype=np.float32)
        if np.isnan(p_idx) or np.isnan(s_idx):
            return target

        if np.isnan(coda_idx):
            coda_idx = s_idx + 1.4 * (s_idx - p_idx)
        d_start = int(max(0, p_idx))
        d_end = int(min(self.trace_len, coda_idx))
        if d_end > d_start:
            target[0, d_start:d_end] = 1.0

        if 0 <= p_idx < self.trace_len:
            start, end, shape = self._make_label(p_idx, self.Plabel_width)
            if len(shape) > 0:
                target[1, start:end] = shape

        if 0 <= s_idx < self.trace_len:
            start, end, shape = self._make_label(s_idx, self.Slabel_width)
            if len(shape) > 0:
                target[2, start:end] = shape

        return np.clip(target, 0, 1)

    def __getitem__(self, idx):
        self._open_hdf5()
        h5_idx = self.indices[idx]

        try:
            waveform = self.waveforms_ds[h5_idx]
        except Exception:
            waveform = np.zeros((3, self.trace_len), dtype=np.float32)

        try:
            waveform = filtfilt(self.b, self.a, waveform, axis=-1)
        except Exception:
            pass

        if self.dataset_format == "instance":
            max_val = np.max(np.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val

        waveform_tensor = torch.from_numpy(waveform.copy()).float()
        target_tensor = torch.from_numpy(
            self._generate_label(
                self.p_arrivals[idx],
                self.s_arrivals[idx],
                self.coda_ends[idx],
            )
        )

        if self.transform:
            waveform_tensor, target_tensor = self.transform(waveform_tensor, target_tensor)

        return waveform_tensor, target_tensor
