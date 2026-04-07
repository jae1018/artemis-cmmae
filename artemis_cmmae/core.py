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
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Get package directory
PACKAGE_DIR = Path(__file__).parent


# ============================================================================
# Model Architecture
# ============================================================================

@dataclass
class MMAEConfig:
    """Configuration for Multi-Modal Autoencoder"""
    num_channels: int = 31
    tabular_dim: int = 5
    latent_dim: int = 6

    # CNN encoder
    cnn_channels: List[int] = None
    cnn_kernel_sizes: List[int] = None
    cnn_use_bn: bool = True
    cnn_dropout: float = 0.0
    cnn_proj_dim: int = 64

    # CNN decoder
    cnn_dec_channels: List[int] = None
    cnn_dec_kernel_sizes: List[int] = None
    cnn_dec_use_bn: bool = True
    cnn_dec_dropout: float = 0.0

    # Tabular MLP
    ff_hidden: List[int] = None
    ff_dropout: float = 0.0
    ff_proj_dim: int = 64
    dec_hidden_tabular: List[int] = None

    # Other
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 1
    device: str = "cpu"

    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64]
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [5, 3]
        if self.cnn_dec_channels is None:
            self.cnn_dec_channels = list(reversed(self.cnn_channels))
        if self.cnn_dec_kernel_sizes is None:
            self.cnn_dec_kernel_sizes = list(reversed(self.cnn_kernel_sizes))
        if self.ff_hidden is None:
            self.ff_hidden = [64, 64]
        if self.dec_hidden_tabular is None:
            self.dec_hidden_tabular = [64]


class CNN1DEncoder(nn.Module):
    def __init__(self, in_len: int, chans: List[int], ks: List[int],
                 use_bn=True, dropout=0.0, proj_dim=64):
        super().__init__()
        layers = []
        in_c = 1
        for out_c, k in zip(chans, ks):
            pad = (k - 1) // 2
            layers += [
                nn.Conv1d(in_c, out_c, kernel_size=k, padding=pad, bias=not use_bn),
                nn.BatchNorm1d(out_c) if use_bn else nn.Identity(),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ]
            in_c = out_c
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(in_c, proj_dim)

    def forward(self, x):
        h = self.net(x)
        h = self.pool(h).squeeze(-1)
        z = self.proj(h)
        return z


class CNN1DDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_len: int,
                 chans: List[int], ks: List[int],
                 use_bn: bool = True, dropout: float = 0.0):
        super().__init__()
        self.out_len = out_len
        self.proj = nn.Linear(latent_dim, chans[0] * out_len)
        blocks = []
        in_c = chans[0]
        for out_c, k in zip(chans[1:], ks):
            pad = (k - 1) // 2
            blocks += [
                nn.Conv1d(in_c, out_c, kernel_size=k, padding=pad, bias=not use_bn),
                nn.BatchNorm1d(out_c) if use_bn else nn.Identity(),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ]
            in_c = out_c
        self.conv = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.head = nn.Conv1d(in_c, 1, kernel_size=1)

    def forward(self, z):
        B = z.size(0)
        h = self.proj(z)
        h = h.view(B, -1, self.out_len)
        h = self.conv(h)
        y = self.head(h).squeeze(1)
        return y


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int,
                 dropout=0.0, last_act=None):
        super().__init__()
        dims = [in_dim] + hidden + [out_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if last_act is not None:
            layers.append(last_act)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MultiModalAE(nn.Module):
    def __init__(self, cfg: MMAEConfig):
        super().__init__()
        self.cfg = cfg

        # Encoders
        self.enc_ch = CNN1DEncoder(
            in_len=cfg.num_channels,
            chans=cfg.cnn_channels,
            ks=cfg.cnn_kernel_sizes,
            use_bn=cfg.cnn_use_bn,
            dropout=cfg.cnn_dropout,
            proj_dim=cfg.cnn_proj_dim,
        )
        self.enc_tab = MLP(
            in_dim=cfg.tabular_dim,
            hidden=cfg.ff_hidden,
            out_dim=cfg.ff_proj_dim,
            dropout=cfg.ff_dropout,
        )

        # Fusion -> latent
        self.to_latent = MLP(
            in_dim=cfg.cnn_proj_dim + cfg.ff_proj_dim,
            hidden=[max(64, cfg.latent_dim)],
            out_dim=cfg.latent_dim,
            dropout=0.0,
        )

        # Decoders
        self.dec_ch = CNN1DDecoder(
            latent_dim=cfg.latent_dim,
            out_len=cfg.num_channels,
            chans=cfg.cnn_dec_channels,
            ks=cfg.cnn_dec_kernel_sizes,
            use_bn=cfg.cnn_dec_use_bn,
            dropout=cfg.cnn_dec_dropout,
        )
        self.dec_tab = MLP(
            in_dim=cfg.latent_dim,
            hidden=cfg.dec_hidden_tabular,
            out_dim=cfg.tabular_dim,
            dropout=cfg.ff_dropout,
        )

    def encode(self, x_ch: torch.Tensor, x_tab: torch.Tensor):
        z_ch = self.enc_ch(x_ch.unsqueeze(1))
        z_tab = self.enc_tab(x_tab)
        z = torch.cat([z_ch, z_tab], dim=-1)
        z = self.to_latent(z)
        return z, (z_ch, z_tab)

    def decode(self, z: torch.Tensor):
        y_ch = self.dec_ch(z)
        y_tab = self.dec_tab(z)
        return y_ch, y_tab

    def forward(self, x_ch: torch.Tensor, x_tab: torch.Tensor):
        z, _ = self.encode(x_ch, x_tab)
        y_ch, y_tab = self.decode(z)
        return y_ch, y_tab, z


class MMAEWithContrastive(nn.Module):
    """Multi-Modal Autoencoder with Supervised Contrastive Learning"""

    def __init__(self, num_channels, tabular_dim, latent_dim, projection_dim):
        super().__init__()
        self.num_channels = num_channels
        self.tabular_dim = tabular_dim
        self.latent_dim = latent_dim
        self.projection_dim = projection_dim

        cfg = MMAEConfig(
            num_channels=num_channels,
            tabular_dim=tabular_dim,
            latent_dim=latent_dim,
            lr=1e-3,
            weight_decay=1e-4,
            epochs=1,
            device='cpu'
        )
        self.autoencoder = MultiModalAE(cfg)

        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, projection_dim)
        )

    def forward(self, x_ch, x_tb, return_all=False):
        z, _ = self.autoencoder.encode(x_ch, x_tb)
        recon_ch = self.autoencoder.dec_ch(z)
        recon_tb = self.autoencoder.dec_tab(z)
        z_proj = self.projection_head(z)
        z_proj_normalized = F.normalize(z_proj, dim=1)

        if return_all:
            return recon_ch, recon_tb, z, z_proj_normalized
        else:
            return recon_ch, recon_tb


# ============================================================================
# Dataset for inference
# ============================================================================

class InferenceDataset(Dataset):
    """Dataset for inference - applies normalization"""

    def __init__(self, df: pd.DataFrame, channel_cols: List[str],
                 tabular_cols: List[str], ch_norm: Dict, tab_norm: Dict):
        super().__init__()
        self.channel_cols = channel_cols
        self.tabular_cols = tabular_cols

        # Get raw data
        raw_channels = df[channel_cols].to_numpy(dtype=np.float32)
        raw_tabular = df[tabular_cols].to_numpy(dtype=np.float32)

        # Normalize
        self.x_ch = ((raw_channels - ch_norm["mean"]) /
                     (ch_norm["std"] + 1e-8)).astype(np.float32)
        self.x_tb = ((raw_tabular - tab_norm["mean"]) /
                     (tab_norm["std"] + 1e-8)).astype(np.float32)

    def __len__(self):
        return self.x_ch.shape[0]

    def __getitem__(self, idx):
        return {
            "x_ch": torch.from_numpy(self.x_ch[idx]),
            "x_tb": torch.from_numpy(self.x_tb[idx]),
        }


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
    TABULAR_COLS = ["n", "SCPot", "BX", "BY", "BZ"]

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
            df: DataFrame with required columns (C0-C30, n, SCPot, BX, BY, BZ)
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
        n (cm^-3), SCPot (V), BX/BY/BZ (nT), X/Y/Z (R_E), prediction.
    """
    sample_path = PACKAGE_DIR / "sample_data" / "sample.csv"
    df = pd.read_csv(sample_path)
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)
    return df
