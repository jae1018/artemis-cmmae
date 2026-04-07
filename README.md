# artemis-cmmae

A pre-trained classifier for identifying plasma regions from ARTEMIS
spacecraft observations. Uses a Contrastive Multi-Modal Autoencoder
(C-MMAE) with supervised contrastive learning and Self-Organizing Map
(SOM) clustering. The default model achieves ~98.9% test accuracy across
four plasma regions (solar wind, magnetosheath, lobe, plasma sheet) on
held-out ARTEMIS observations.

## Classified Regions

| ID | Region |
|----|--------|
| -1 | Unknown |
| 0 | Solar Wind |
| 1 | Magnetosheath |
| 2 | Lobe |
| 3 | Plasma Sheet |

## Installation

```bash
pip install git+https://github.com/jae1018/artemis-cmmae.git
```

Or clone and install locally:

```bash
git clone https://github.com/jae1018/artemis-cmmae.git
cd artemis-cmmae
pip install .
```

### Dependencies

numpy, pandas, torch, joblib, scikit-learn, xpysom

## Quick Start

```python
from artemis_cmmae import PlasmaClassifier, load_sample_data

# Load bundled sample data (1 month of ARTEMIS observations)
df = load_sample_data()

# Initialize classifier
clf = PlasmaClassifier(alpha=1)

# Get predictions
predictions, label_map = clf.predict(df)
```

To classify your own data, pass any DataFrame with the required columns
(see [Input Data Format](#input-data-format) below).

## API Reference

### `PlasmaClassifier(alpha=1, weights_dir=None, device=None)`

**Parameters:**
- `alpha` (int): Map size penalty controlling SOM granularity. One of `1`, `5`, or `10`.
  - `alpha=1`: 16x20 = 320 nodes (finest resolution, highest accuracy, default)
  - `alpha=5`: 9x11 = 99 nodes (medium)
  - `alpha=10`: 7x9 = 63 nodes (most compact)
- `weights_dir` (Path, optional): Custom path to weights directory.
- `device` (str, optional): `"cuda"`, `"cpu"`, or `None` for auto-detect.

### `clf.predict(df, batch_size=2048)`

Returns `(predictions, label_map)` where `predictions` is an integer array of
label IDs and `label_map` is a dict mapping IDs to region names.

### `clf.predict_bmu(df, mode="1d", batch_size=2048)`

Returns Best Matching Unit (BMU) indices on the SOM grid.
- `mode="1d"`: flattened node index (int array)
- `mode="2d"`: list of `(i, j)` tuples

### `clf.predict_with_embeddings(df, batch_size=2048)`

Returns a dict with all intermediate results:
- `predictions`: label ID array
- `bmu_1d`, `bmu_2d`: BMU indices
- `embeddings_raw`: 6-D C-MMAE latent vectors
- `embeddings_pca`: PCA-transformed embeddings (SOM input)
- `label_map`: label ID to name mapping

## Input Data Format

Your DataFrame must contain the following columns, all in physical
(linear-scale) units:

| Column(s) | Description | Units |
|-----------|-------------|-------|
| `C0` - `C30` | Ion energy flux (31 log-spaced energy bins, ~5 eV to ~25 keV) | eV/(cm^2 s sr eV) |
| `n` | Ion number density | cm^-3 |
| `SCPot` | Spacecraft potential | V |
| `BX`, `BY`, `BZ` | Magnetic field components in GSE | nT |

The classifier internally applies `log10(eflux + 1)` to energy channels
and `log10(n)` to density, then standardizes all features using fixed
training-set statistics. No pre-transforms are needed.

The index should be a `DatetimeIndex` named `time`, though the classifier
itself does not use timestamps.

## Sample Data

A one-month sample dataset (September 2020) is bundled with the package.
Load it with `load_sample_data()` or find it at
`artemis_cmmae/sample_data/sample.csv`. See `examples/quickstart.py` for usage.

## Paper and Data

This package accompanies the following publication, submitted to
Journal of Geophysical Research: Machine Learning and Computation:

> Edmond, J., Ferdousi, B., Johnston, W. R., & Lewis, N. (2026).
> *Semi-Supervised Plasma Region Classification in Earth's Cislunar
> Magnetotail using ARTEMIS Observations*.

The full training data, HPO results, and reproducibility archive are
available on Zenodo: [DOI placeholder]

## License

MIT
