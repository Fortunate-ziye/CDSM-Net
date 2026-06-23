# CDSM-Net

CDSM-Net is a deep-learning framework for seismic event detection and P/S
phase picking in continuous waveform data. The model combines convolutional
feature extraction, continuous dynamic modeling, and bidirectional state-space
sequence mixing for efficient waveform representation.



## Repository Contents

```text
CDSM-Net-code/
  Model/
    model.py                 # CDSM-Net architecture
    best_model_loss.pt       # Example pretrained model checkpoint
  processing_data/
    processing_data_stead.py # STEAD preprocessing script
  datasets/
    README.md                # Expected local data layout
  augmentation.py            # Online waveform augmentation
  dataset.py                 # Processed HDF5/CSV dataset loader
  engine.py                  # Training and validation loops
  loss.py                    # Multi-task loss functions
  plot.py                    # Diagnostic visualization utilities
  train.py                   # Training entry point
  val.py                     # Evaluation entry point
  test.py                    # Single-trace inference and plotting
  requirements.txt           # Python dependencies
  LICENSE                    # GPL-3.0 license for the code
```

Large waveform files are intentionally not stored in the GitHub repository.
See the data section below.

## License

The source code is released under the GNU General Public License v3.0
(GPL-3.0). See `LICENSE`.

The processed waveform data should be cited separately and follows the license
specified in the corresponding data repository. The original STEAD and INSTANCE
datasets must also be cited when those data are used.

## Installation

The code was developed with Python 3.10 and PyTorch. A CUDA-capable GPU is
recommended for training.

```bash
conda create -n cdsm python=3.10
conda activate cdsm

# Install PyTorch following the command recommended for your CUDA version.
# Example only:
pip install torch==2.7.0

pip install -r requirements.txt
```

CDSM-Net uses `mamba-ssm`, which may require CUDA build tools. If direct
installation fails, install `causal-conv1d` and `mamba-ssm` from their official
repositories with `--no-build-isolation`.

```bash
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm --no-build-isolation
```

## Dependencies

Core dependencies are listed in `requirements.txt`. Main packages include:

- Python 3.10
- PyTorch
- NumPy, SciPy, pandas, h5py
- scikit-learn
- matplotlib, seaborn, Pillow
- tqdm
- wandb
- ncps
- mamba-ssm and causal-conv1d

## Computing Requirements

Inference can be run on CPU for small examples, but GPU execution is strongly
recommended. Training the full model on the processed STEAD split requires a
CUDA-capable GPU. The default batch size in `train.py` was selected for a
high-memory GPU; reduce `--batch_size` if GPU memory is limited.

Typical recommendations:

- GPU: NVIDIA CUDA GPU with at least 12 GB memory for practical training.
- RAM: 32 GB or more for large HDF5 files.
- Storage: enough local disk space for processed waveform HDF5 files.

## Data

The code expects processed waveform data in HDF5 format and metadata in CSV
format:

```text
datasets/
  merge.h5
  train.csv
  val.csv
  test.csv
```

The processed files used for the CDSM-Net experiments are provided separately
on Hugging Face:

<https://huggingface.co/datasets/ziye0013/Stead>

The processed data are derived from the public STEAD and INSTANCE datasets.
They are provided for reproducibility and do not replace the official raw data
releases. Users should follow the licenses and citation requirements of the
original datasets.

If the processed HDF5 files are not available locally, download them from the
data repository and place them under `datasets/`, or edit the command-line
paths to point to your local data location.

## Training

Example training command:

```bash
python train.py \
  --train_csv datasets/train.csv \
  --val_csv datasets/val.csv \
  --hdf5 datasets/merge.h5 \
  --dataset_format stead \
  --batch_size 450 \
  --epochs 60 \
  --wandb_mode disabled
```

Training outputs are saved to `runs/<timestamp>_CDSM-Net/`, including:

- `config.json`: command-line configuration
- `model_arch.txt`: model architecture
- `last.pt`: latest training checkpoint
- `best_model.pt`: best model by average P/S F1-score
- `best_model_loss.pt`: best model by validation loss

## Evaluation

Evaluate a trained or pretrained model:

```bash
python val.py \
  --checkpoint Model/best_model_loss.pt \
  --val_csv datasets/test.csv \
  --hdf5 datasets/merge.h5 \
  --dataset stead \
  --batch_size 2048
```

Evaluation outputs are written to `eval_results/` and include a metrics report,
an error-distribution figure, and a detection confusion matrix.

The main reported metrics are:

- Detection accuracy, precision, recall, and F1-score
- P-wave precision, recall, F1-score, and MAE
- S-wave precision, recall, F1-score, and MAE

A phase pick is counted as correct when the predicted arrival is within 0.5 s
of the manual arrival.

## Single-Trace Inference

Generate a local visualization for one waveform trace:

```bash
python test.py \
  --trace_name "YOUR_TRACE_NAME" \
  --model_path Model/best_model_loss.pt \
  --csv_path datasets/test.csv \
  --hdf5_path datasets/merge.h5 \
  --dataset_format stead \
  --save_dir inference_results
```

The output image shows three-component waveforms, manual P/S arrivals, and the
predicted detection, P, and S probability curves.

## Reproducing the Main Results

1. Install the environment described above.
2. Download the processed STEAD and INSTANCE files from the data repository.
3. Place or link the processed STEAD files to `datasets/`.
4. Train with `train.py`, or evaluate the provided pretrained checkpoint with
   `val.py`.
5. Compare the generated metrics with the evaluation summary provided in the
   data repository.

For cross-domain INSTANCE evaluation, use the INSTANCE processed HDF5 and test
CSV files and set `--dataset instance`.

## Input and Output Summary

Inputs:

- HDF5 file with a `waveforms` dataset of shape `[N, 3, 6000]`.
- CSV metadata with at least `h5_index`, `p_arrival_sample`,
  `s_arrival_sample`, and `coda_end_sample`.

Outputs:

- Trained checkpoints in PyTorch format.
- Text metrics reports.
- Diagnostic PNG figures.
- Optional WandB logs when `--wandb_mode online` or `offline` is used.

## Citation

If you use this code, please cite the CDSM-Net manuscript and the processed data
repository. Please also cite the original STEAD and INSTANCE data sources when
their data are used.


