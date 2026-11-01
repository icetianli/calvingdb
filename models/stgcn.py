"""
STGCN model for calving front SDT displacement prediction (Yu et al., IJCAI 2018).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalGatedConv(nn.Module):
    """
    Gated linear unit (GLU) temporal convolution along the K time axis.

    out = tanh(a) ⊙ σ(b)
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kt:           int = 3,
    ) -> None:
        super().__init__()
        pad = kt // 2
        self.conv = nn.Conv3d(
            in_channels,
            out_channels * 2,
            kernel_size=(kt, 1, 1),
            padding=(pad, 0, 0),
            bias=True,
        )
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.conv(x).chunk(2, dim=1)
        return self.bn(torch.tanh(a) * torch.sigmoid(b))


class SpatialGraphConv(nn.Module):
    """
    Spatial graph convolution on the regular pixel lattice.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class STConvBlock(nn.Module):
    """
    BasicTS-style Spatio-Temporal Convolution Block (Yu et al. 2018, §3.2).
    """

    def __init__(
        self,
        in_channels:  int,
        spatial_ch:   int,
        out_channels: int,
        kt:           int = 3,
    ) -> None:
        super().__init__()
        self.tconv1 = TemporalGatedConv(in_channels, spatial_ch,   kt)
        self.sconv  = SpatialGraphConv( spatial_ch,  spatial_ch)
        self.tconv2 = TemporalGatedConv(spatial_ch,  out_channels, kt)

        self.residual = (
            nn.Conv3d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.ln = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the ST-Conv block, mapping (B, C_in, K, H, W) to (B, C_out, K, H, W).
        """
        B, _, K, H, W = x.shape

        out = self.tconv1(x)
        sc  = out.shape[1]

        out = out.permute(0, 2, 1, 3, 4).reshape(B * K, sc, H, W)
        out = self.sconv(out)
        out = out.reshape(B, K, sc, H, W).permute(0, 2, 1, 3, 4)

        out = self.tconv2(out)

        out = out + self.residual(x)

        out = self.ln(
            out.permute(0, 2, 3, 4, 1)
        ).permute(0, 4, 1, 2, 3)

        return out


class CalvingSTGCN(nn.Module):
    """
    Calving front SDT predictor built on a STGCN core (BasicTS-style).
    """

    def __init__(
        self,
        in_channels:   int,
        base_channels: int = 32,
        climate_dim:   int = 10,
        tau_dim:       int = 20,
        num_layers:    int = 2,
        kt:            int = 3,
    ) -> None:
        super().__init__()

        K       = tau_dim // 4
        C_total = in_channels // K
        D_c     = climate_dim // K
        F       = base_channels
        hid_ch  = base_channels * 2

        self.K       = K
        self.C_total = C_total

        frame_enc_in = C_total + 4 + D_c
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(frame_enc_in, F, 3, padding=1, bias=False),
            nn.BatchNorm2d(F),
            nn.ReLU(inplace=True),
            nn.Conv2d(F, F, 3, padding=1, bias=False),
            nn.BatchNorm2d(F),
            nn.ReLU(inplace=True),
        )

        blocks = []
        for i in range(num_layers):
            in_ch = F if i == 0 else hid_ch
            blocks.append(STConvBlock(in_ch, hid_ch, hid_ch, kt=kt))
        self.st_blocks = nn.ModuleList(blocks)

        self.head = nn.Sequential(
            nn.Conv2d(hid_ch, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Predict the (B, 1, H, W) normalised SDT displacement from a batch dict.
        """
        x       = batch["inputs"]
        climate = batch["climate"]
        tau     = batch["tau"]

        B, K, C, H, W = x.shape

        tau_spatial  = tau.reshape(B * K, 4, 1, 1).expand(-1, -1, H, W)
        clim_spatial = climate.reshape(B * K, -1, 1, 1).expand(-1, -1, H, W)
        x_aug        = torch.cat([x.reshape(B * K, C, H, W),
                                   tau_spatial, clim_spatial], dim=1)

        enc = self.frame_encoder(x_aug)
        enc = enc.reshape(B, K, -1, H, W)

        h = enc.permute(0, 2, 1, 3, 4).contiguous()

        for block in self.st_blocks:
            h = block(h)

        h = h[:, :, -1, :, :]

        return self.head(h)
