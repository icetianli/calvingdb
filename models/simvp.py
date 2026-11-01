"""
SimVP model for calving front SDT displacement prediction (Gao et al., 2022)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    """
    Multi-scale spatial feature extractor used as the per-frame encoder.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        assert out_channels % 4 == 0, "out_channels must be divisible by 4"
        b = out_channels // 4

        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, b, 1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, b, 1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, b, 1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )
        self.b4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, b, 1, bias=False),
            nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class GroupedInceptionBlock(nn.Module):
    """
    One translator block (SimVP Mid unit).
    """

    def __init__(self, channels: int, K: int, kernels: tuple[int, ...] = (3, 5, 7)) -> None:
        super().__init__()
        assert channels % K == 0, "channels must be divisible by K"

        self.branches = nn.ModuleList()
        n_branch = len(kernels)
        branch_ch = (channels // n_branch // K) * K
        branch_chs = [branch_ch] * (n_branch - 1) + [channels - branch_ch * (n_branch - 1)]

        for k, out_ch in zip(kernels, branch_chs):
            self.branches.append(nn.Sequential(
                nn.Conv2d(channels, out_ch, k,
                          padding=k // 2, groups=K, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))

        self.mixer = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([b(x) for b in self.branches], dim=1)
        feat = self.mixer(feat)
        return feat + self.residual(x)


class TemporalReduce(nn.Module):
    """
    Collapses the K time-step feature maps into a single representation.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reduce(x)


class CalvingSimVP(nn.Module):
    """
    Calving front SDT predictor built on a SimVP feed-forward core.
    """

    def __init__(
        self,
        in_channels:        int,
        base_channels:      int = 40,
        climate_dim:        int = 10,
        tau_dim:            int = 20,
        num_layers:         int = 4,
        translator_kernels: tuple[int, ...] = (3, 5, 7),
    ) -> None:
        super().__init__()

        K       = tau_dim // 4
        C_total = in_channels // K
        D_c     = climate_dim // K
        hid_S   = base_channels

        if hid_S % K != 0:
            raise ValueError(
                f"base_channels ({hid_S}) must be divisible by K ({K}) "
                f"for the grouped translator.  Try e.g. 40 for K=5."
            )

        self.K       = K
        self.C_total = C_total
        self.hid_S   = hid_S

        frame_enc_in = C_total + 4 + D_c
        self.encoder = InceptionBlock(frame_enc_in, hid_S)

        self.translator = nn.Sequential(*[
            GroupedInceptionBlock(K * hid_S, K, kernels=translator_kernels)
            for _ in range(num_layers)
        ])

        self.temporal_reduce = TemporalReduce(K * hid_S, hid_S)

        self.head = nn.Sequential(
            nn.Conv2d(hid_S, hid_S, 3, padding=1, bias=False),
            nn.BatchNorm2d(hid_S),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_S, 1, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
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

        enc = self.encoder(x_aug)
        enc = enc.reshape(B, K * self.hid_S, H, W)

        enc = self.translator(enc)

        feat = self.temporal_reduce(enc)

        return self.head(feat)
