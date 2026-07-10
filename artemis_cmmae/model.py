"""
Neural-network architecture for artemis_cmmae.

This module holds the Multi-Modal Autoencoder (MMAE) building blocks and the
inference Dataset used by :class:`artemis_cmmae.core.PlasmaClassifier`. It was
split out of ``core.py`` as a pure mechanical move; the class definitions are
unchanged.
"""

from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


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
