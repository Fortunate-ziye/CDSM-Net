# Dataset Directory


```text
datasets/
  merge.h5
  train.csv
  val.csv
  test.csv
```

The HDF5 file must contain a `waveforms` dataset with shape `[N, 3, 6000]`.
The CSV files must contain at least:

- `h5_index`
- `p_arrival_sample`
- `s_arrival_sample`
- `coda_end_sample`

Processed data used for the manuscript are available separately:

<https://huggingface.co/datasets/ziye0013/Stead>

For INSTANCE evaluation, place the corresponding processed INSTANCE HDF5 and
test CSV locally and pass their paths to `val.py` with `--dataset instance`.
