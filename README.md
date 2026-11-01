# CalvingDB

**CalvingDB** is a large-scale, multi-modal dataset for glacier calving front forecasting, collected from marine-terminating glaciers across Svalbard. It provides time series of satellite imagery, calving front positions, climate reanalysis data, and geophysical variables, supporting research at the intersection of glaciology, computer vision, and climate-aware machine learning.

---

## Table of Contents

- [Overview](#overview)
- [Dataset Statistics](#dataset-statistics)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Structure](#dataset-structure)
- [Benchmarks](#benchmarks)
- [Splits & Versioning](#splits--versioning)
- [Citation](#citation)
- [License](#license)

---

## Overview

Glacier calving is a major source of uncertainty in sea-level rise projections. Despite its importance, calving remains difficult to model due to the complex, non-linear interplay of climate forcing and glacier dynamics.

CalvingDB addresses three key gaps in existing datasets:

| Gap | CalvingDB Solution |
|-----|----------------------|
| Single-sensor datasets | **Multi-modal**: optical, shapefile, geophysical variables, atmosphere - ocean reanalysis |
| Short time horizons | **10 years** of coverage (2013–2023) |
| Limited spatial coverage | **123 glaciers with 17358 calving front traces** across Svalbard |
| Unclear benchmark protocols | **Standardised splits + metrics** for forecasting |

---

## Dataset Statistics

| Property | Value |
|----------|-------|
| Glaciers | 123 marine-terminating, Svalbard |
| Calving front traces | 17,358 |
| Temporal coverage | 2013–2023 |
| Spatial resolution | 30 m |
| Projection | EPSG:3995 (Arctic Polar Stereographic) |
| Optical bands | 5 (blue, green, red, NIR, SWIR1) |
| Geophysical rasters | 6 (bed topography, surface elevation, ice thickness, surface velocity, strain rate, ice/fjord mask) |
| Climate scalars | 2 per observation |
| Front representation | Signed distance transform (SDT), clipped to ±2000 m |

---

## Installation

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 *(optional, for deep learning benchmarks)*
- GDAL ≥ 3.4 *(optional, for GeoTIFF I/O)*

### From source

```bash
git clone https://github.com/icetianli/calvingdb.git
cd calvingdb
conda env create -f environment.yml
conda activate calvingdb
pip install -e .
```

---

## Quick Start

### 1. Build the processed dataset

The preprocessing scripts run in order and are each configured by the path constants at the top of the file — set these to your local raw data location before running.

```bash
python preprocessing/00_create_glacier_mask.py   
python preprocessing/01_inventory.py             
python preprocessing/02_process_observations.py  
python preprocessing/03_build_index.py           
python preprocessing/04_compute_stats.py         
```
Statistics are computed from the training split alone to avoid leakage into the normalisation parameters.

### 2. Train a model

Point `data.splits_dir` and `data.stats_path` in your config at the outputs of step 04, then:

```bash
python train.py  # uses configs/default.yaml
python train.py --config configs/simvp.yaml
python train.py --config configs/predrnn.yaml --resume checkpoints/epoch_0050.pt
```

One config per baseline is provided in [configs/](configs/): `convlstm`, `predrnn`, `simvp`, `phydnet`, `stgcn`, `earthformer`.

### 3. Evaluate

```bash
python evaluate.py --checkpoint experiments/exp10_simvp/checkpoints/best.pt --split test
```

Metrics are written to the directory given by `--log_dir`. See [notebooks/example_loading.ipynb](notebooks/example_loading.ipynb) for loading samples directly.

### Loading samples in your own code

```python
from data.dataset import CalvingFrontDataset, calving_collate_fn
from torch.utils.data import DataLoader

ds = CalvingFrontDataset(
    index_path = "splits/test.json",
    stats_path = "splits/normalisation_stats.json",
    modalities = ["sdt", "imagery", "geophys", "climate"],  
)
loader = DataLoader(ds, batch_size=4, collate_fn=calving_collate_fn)
```

---

## Dataset Structure

Processed data is stored as one [zarr](https://zarr.dev) group per glacier, with `T` observations each:

```
TrainingDataset_processed/
├── inventory.parquet              # flat table of every raw observation
├── zarr/
│   └── <glacier_id>.zarr
│       ├── trace        (T, H, W)      bool    calving front line
│       ├── sdt          (T, H, W)      float32 signed distance transform (m)
│       ├── imagery      (T, 5, H, W)   float32 optical bands
│       ├── geophys      (T, 6, H, W)   float32 geophysical rasters
│       ├── climate      (T, 2)         float32 per-observation climate scalars
│       ├── dates        (T,)           str     ISO acquisition dates
│       ├── sensors      (T,)           uint8   sensor code
│       └── valid_*      (T,)           bool    per-modality availability flags
└── splits/
    ├── train.json / val.json / test.json   # pre-resolved sample indices
    ├── sampling_params.json                # K, horizons, split ratios, seed
    └── normalisation_stats.json            # train-split mean/std per modality
```

---

## Benchmarks

### Task — Short-Term Forecasting (`CalvingDB`)

First benchmark dataset for data-driven annual calving front forecasting, covering 123 marine-terminating glaciers in Svalbard from 2013 to 2023.

Each benchmark sample asks: given five past calving front observations (spanning roughly four years), predict where the calving front will be 365 days in the future.

### Baselines

Six spatiotemporal forecasting architectures are implemented in [models/](models/): ConvLSTM, PredRNN, SimVP, PhyDNet, STGCN, and EarthFormer. 

### Metrics

All metrics are computed in metres within a 300 m band around the true front:

| Metric | Description |
|--------|-------------|
| `mae`| SDT error within the front band (m) |
| `mde` | Mean distance error between predicted and true front, symmetric (m) |
| `hausdorff` | Worst-case minimum distance in either direction (m) |
| `front_iou` | IoU of dilated front bands, [0, 1] |
| `skill_score` | 1 − (model MDE / persistence MDE); > 0 beats persistence |

Persistence — predicting no change from the most recent observation — is the reference baseline for `skill_score`.

---

## Splits & Versioning

Splits are **glacier-level**, not sample-level. This measures generalisation to unseen glaciers rather than interpolation within known ones.

| Split | Ratio |
|-------|-------|
| Train | 70% of glaciers |
| Val | 10% of glaciers |
| Test | 20% of glaciers |

The assignment is deterministic (seed `42`, see `SPLIT_SEED` in [preprocessing/03_build_index.py](preprocessing/03_build_index.py)). Sampling parameters — `K = 5` input slots, `horizon = 365` days, split ratios and seed — are recorded in `splits/sampling_params.json` alongside the split files.

Dataset version: **0.1.0**.

---

## Citation

```bibtex
@misc{Li2026CalvingDB,
  title        = {CalvingDB: A Multi-Modal Benchmark for Marine-Terminating
                  Glacier Calving Prediction in Svalbard},
  author       = {Li, Tian and Steidl, Viola and Xiong, Zhitong
                  and Zhu, Xiao Xiang and Mitcham, Tom and Cornford, Stephen
                  and Bamber, Jonathan},
  year         = {2026},
  note         = {Manuscript under review}
}
```


---

## License

- **Dataset** (imagery, labels, metadata): [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Code** (this repository): [MIT License](LICENSE)


