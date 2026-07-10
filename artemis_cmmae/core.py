"""
PlasmaClassifier - Plasma Region Classification using MMAE + SOM

This module provides a class-based interface for classifying plasma regions
in ARTEMIS spacecraft data using a pre-trained Multi-Modal Autoencoder (MMAE)
with supervised contrastive learning and Self-Organizing Map (SOM) clustering.

Usage:
    from artemis_cmmae import PlasmaClassifier

    clf = PlasmaClassifier()
    predictions, label_map = clf.predict(df)
"""

import json
import pickle
import importlib.resources
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .model import MMAEWithContrastive, InferenceDataset

# Get package directory
PACKAGE_DIR = Path(importlib.resources.files("artemis_cmmae"))


# ============================================================================
# Main Classifier Class
# ============================================================================

class PlasmaClassifier:
    """
    Plasma Region Classifier using MMAE + SOM

    This classifier uses a pre-trained Multi-Modal Autoencoder with supervised
    contrastive learning to extract latent features, followed by a Self-Organizing
    Map (SOM) for classification.

    Attributes:
        device: torch device (cuda or cpu)
        model: MMAE model
        preprocessors: dict with 'scaler' and 'pca' for latent space preprocessing
        som: trained SOM model
        node_labels: array of class labels for each SOM node
        label_map: dict mapping label IDs to human-readable names

    Example:
        clf = PlasmaClassifier()
        predictions, label_map = clf.predict(df)

        # Get BMU indices instead
        bmu_1d = clf.predict_bmu(df, mode='1d')
        bmu_2d = clf.predict_bmu(df, mode='2d')
    """

    # Feature columns expected in input data
    CHANNEL_COLS = [f"C{i}" for i in range(31)]
    TABULAR_COLS = ["n", "SCPot", "BX_GSE", "BY_GSE", "BZ_GSE"]

    # Label mapping
    LABEL_MAP = {
        -1: "Unknown",
        0: "Solar Wind",
        1: "Magnetosheath",
        2: "Lobe",
        3: "Plasma Sheet",
    }

    # Available alpha values (map size penalty)
    AVAILABLE_ALPHAS = [1, 5, 10]

    def __init__(self, alpha: int = 1, weights_dir: Optional[Path] = None, device: str = None):
        """
        Initialize the classifier by loading all pre-trained components.

        Args:
            alpha: Map size penalty (1, 5, or 10). Controls SOM size:
                   - alpha=1:  16x20 = 320 nodes (largest, best accuracy, default)
                   - alpha=5:  9x11 = 99 nodes (medium)
                   - alpha=10: 7x9 = 63 nodes (smallest, most compact)
            weights_dir: Path to weights directory (default: package weights/)
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if alpha not in self.AVAILABLE_ALPHAS:
            raise ValueError(f"alpha must be one of {self.AVAILABLE_ALPHAS}, got {alpha}")

        self.alpha = alpha

        if weights_dir is None:
            weights_dir = PACKAGE_DIR / "weights"
        weights_dir = Path(weights_dir)

        # SOM-specific weights directory
        self.som_weights_dir = weights_dir / f"som_alpha{alpha}"

        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        # Load all components
        self._load_mmae(weights_dir)
        self._load_preprocessors()
        self._load_som()
        self._load_node_labels()

        print(f"PlasmaClassifier initialized on {self.device}")
        print(f"  Alpha (map size penalty): {self.alpha}")
        print(f"  MMAE latent_dim: {self.latent_dim}")
        print(f"  SOM shape: {self.som_shape} ({self.som_shape[0] * self.som_shape[1]} nodes)")

    def _load_mmae(self, weights_dir: Path):
        """Load MMAE model and training normalization statistics."""
        mmae_dir = weights_dir / "mmae"
        config_path = mmae_dir / "config.json"
        model_path = mmae_dir / "model_best.pt"
        norm_path = mmae_dir / "norm_stats.json"

        with open(config_path) as f:
            config = json.load(f)

        self.latent_dim = config['latent_dim']
        self.projection_dim = config['projection_dim']

        # Load training normalization statistics
        if not norm_path.exists():
            raise FileNotFoundError(
                f"Normalization stats not found at {norm_path}. "
                "The weights directory may be incomplete."
            )
        with open(norm_path) as f:
            norm = json.load(f)
        self._ch_norm = {
            "mean": np.array(norm["ch_norm"]["mean"], dtype=np.float32),
            "std": np.array(norm["ch_norm"]["std"], dtype=np.float32),
        }
        self._tab_norm = {
            "mean": np.array(norm["tab_norm"]["mean"], dtype=np.float32),
            "std": np.array(norm["tab_norm"]["std"], dtype=np.float32),
        }

        self.model = MMAEWithContrastive(
            num_channels=31,
            tabular_dim=5,
            latent_dim=self.latent_dim,
            projection_dim=self.projection_dim
        )

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

    def _load_preprocessors(self):
        """Load preprocessing pipeline (scaler + PCA) for current alpha"""
        preprocessors_path = self.som_weights_dir / "preprocessors.pkl"

        with open(preprocessors_path, 'rb') as f:
            self.preprocessors = pickle.load(f)

        self.scaler = self.preprocessors['scaler']
        self.pca = self.preprocessors['pca']

    def _load_som(self):
        """Load SOM model for current alpha"""
        som_path = self.som_weights_dir / "model.joblib"
        params_path = self.som_weights_dir / "params.json"

        self.som = joblib.load(som_path)

        with open(params_path) as f:
            params = json.load(f)

        self.som_shape = (params['init_params']['x'], params['init_params']['y'])

    def _get_bmus(self, X: np.ndarray):
        """Get BMUs from SOM, handling both XPySom and SOMWrapper"""
        if hasattr(self.som, 'get_bmus'):
            return self.som.get_bmus(X)
        elif hasattr(self.som, 'winner'):
            # XPySom uses winner() method
            return self.som.winner(X)
        else:
            raise AttributeError("SOM object has neither get_bmus nor winner method")

    def _load_node_labels(self):
        """Load pre-computed node labels (majority vote assignments)."""
        node_labels_path = self.som_weights_dir / "node_labels.npy"

        if not node_labels_path.exists():
            raise FileNotFoundError(
                f"Node labels not found at {node_labels_path}. "
                "The weights directory may be incomplete."
            )
        self.node_labels = np.load(node_labels_path)

    @staticmethod
    def _log_transform(df: pd.DataFrame) -> pd.DataFrame:
        """Apply log10 transforms to raw linear-scale input."""
        df = df.copy()
        ch_cols = [f"C{i}" for i in range(31)]
        df[ch_cols] = np.log10(df[ch_cols].values + 1)
        df["n"] = np.log10(df["n"].values)
        return df

    def _extract_embeddings(self, df: pd.DataFrame, batch_size: int = 2048) -> np.ndarray:
        """Extract latent embeddings from MMAE using training normalization."""
        df = self._log_transform(df)
        dataset = InferenceDataset(df, self.CHANNEL_COLS, self.TABULAR_COLS,
                                   self._ch_norm, self._tab_norm)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        embeddings = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                x_ch = batch['x_ch'].to(self.device)
                x_tb = batch['x_tb'].to(self.device)
                _, _, z, _ = self.model(x_ch, x_tb, return_all=True)
                embeddings.append(z.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def predict(self, df: pd.DataFrame, batch_size: int = 2048) -> Tuple[np.ndarray, Dict]:
        """
        Predict plasma region labels for input data.

        Args:
            df: DataFrame with required columns (C0-C30, n, SCPot,
                BX_GSE, BY_GSE, BZ_GSE)
            batch_size: Batch size for inference

        Returns:
            predictions: Array of predicted label IDs
            label_map: Dictionary mapping label IDs to names
        """
        # Extract embeddings
        embeddings = self._extract_embeddings(df, batch_size)

        # Preprocess
        embeddings_scaled = self.scaler.transform(embeddings)
        embeddings_pca = self.pca.transform(embeddings_scaled)

        # Get BMUs
        bmus = self._get_bmus(embeddings_pca)

        # Map to node labels
        predictions = np.array([
            self.node_labels[i * self.som_shape[1] + j]
            for i, j in bmus
        ])

        return predictions, self.LABEL_MAP.copy()

    def predict_from_ds(self, ds, **kwargs):
        """Classify a THEMIS/ARTEMIS xarray/dict dataset (delegates to
        :func:`artemis_cmmae.pipeline.predict_from_ds` with this classifier).
        Accepts the same keyword args as that function EXCEPT ``classifier``
        (this instance is used) and ``alpha`` (this instance's model is used):
        ``fgm_ds``, ``probe``, ``return_features``, plus the build_features_from_ds
        options (``filter_out_wake``, ``wake_scpot_threshold``, ``wake_pad``,
        ``drop_wake``, ``verbose``, ``b_source``, ``time_res``, ``merge_tolerance``,
        ``drop_top_channel``)."""
        from .pipeline import predict_from_ds as _predict_from_ds
        return _predict_from_ds(ds, classifier=self, **kwargs)

    def predict_from_pyspedas(self, probe, trange, **kwargs):
        """Download + classify via pyspedas (delegates to
        :func:`artemis_cmmae.pipeline.predict_from_pyspedas` with this classifier).
        Accepts the same keyword args as that function EXCEPT ``classifier`` and
        ``alpha`` (this instance is used): ``time_res``, ``b_source``,
        ``filter_out_wake``, ``wake_scpot_threshold``, ``wake_pad``, ``drop_wake``,
        ``return_features``, ``verbose``, ``no_update``, ``files``, ``fgm_files``,
        ``state_files``, ``merge_tolerance``."""
        from .pipeline import predict_from_pyspedas as _predict_from_pyspedas
        return _predict_from_pyspedas(probe, trange, classifier=self, **kwargs)

    def predict_bmu(self, df: pd.DataFrame, mode: str = '1d',
                    batch_size: int = 2048) -> np.ndarray:
        """
        Predict Best Matching Unit (BMU) indices for input data.

        Args:
            df: DataFrame with required columns
            mode: '1d' for flattened index, '2d' for (i, j) tuples
            batch_size: Batch size for inference

        Returns:
            bmu_indices: Array of BMU indices (1D) or list of (i, j) tuples (2D)
        """
        # Extract embeddings
        embeddings = self._extract_embeddings(df, batch_size)

        # Preprocess
        embeddings_scaled = self.scaler.transform(embeddings)
        embeddings_pca = self.pca.transform(embeddings_scaled)

        # Get BMUs
        bmus = self._get_bmus(embeddings_pca)

        if mode == '2d':
            return bmus
        else:  # 1d
            return np.array([i * self.som_shape[1] + j for i, j in bmus])

    def predict_with_embeddings(self, df: pd.DataFrame, batch_size: int = 2048) -> Dict:
        """
        Full prediction with all intermediate results.

        Returns dict with: predictions, bmu_1d, bmu_2d, embeddings_raw,
        embeddings_pca, label_map
        """
        # Extract embeddings
        embeddings = self._extract_embeddings(df, batch_size)

        # Preprocess
        embeddings_scaled = self.scaler.transform(embeddings)
        embeddings_pca = self.pca.transform(embeddings_scaled)

        # Get BMUs
        bmus = self._get_bmus(embeddings_pca)
        bmu_1d = np.array([i * self.som_shape[1] + j for i, j in bmus])

        # Map to labels
        predictions = self.node_labels[bmu_1d]

        return {
            'predictions': predictions,
            'bmu_1d': bmu_1d,
            'bmu_2d': bmus,
            'embeddings_raw': embeddings,
            'embeddings_pca': embeddings_pca,
            'label_map': self.LABEL_MAP.copy()
        }


def load_sample_data() -> pd.DataFrame:
    """
    Load the bundled sample dataset (September 2020, ~9600 observations).

    All values are in physical (linear-scale) units. The classifier handles
    log transforms internally.

    Returns:
        DataFrame with DatetimeIndex and columns C0-C30 (eV/(cm^2 s sr eV)),
        n (cm^-3), SCPot (V), BX_GSE/BY_GSE/BZ_GSE (nT),
        X_GSE/Y_GSE/Z_GSE (R_E), prediction.
    """
    sample_path = PACKAGE_DIR / "sample_data" / "sample.csv"
    df = pd.read_csv(sample_path)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    return df
