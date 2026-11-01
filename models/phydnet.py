"""
PhyDNet model for calving front SDT displacement prediction (Le Guen & Thome, CVPR 2020)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MomentBank(nn.Module):
    """
    N learnable depthwise convolutions that approximate Taylor-expansion PDE terms on
    the hidden state.

    ĥ = Σ_k softmax(w)_k · M_k(h)
    """

    def __init__(
        self,
        hidden_channels: int,
        n_moments:       int = 4,
        kernel_size:     int = 7,
    ) -> None:
        super().__init__()
        self.ops = nn.ModuleList([
            nn.Conv2d(
                hidden_channels, hidden_channels, kernel_size,
                padding=kernel_size // 2, groups=hidden_channels, bias=False,
            )
            for _ in range(n_moments)
        ])
        self.log_w = nn.Parameter(torch.zeros(n_moments))

        for conv in self.ops:
            nn.init.zeros_(conv.weight)
            k = kernel_size
            centre = k // 2
            with torch.no_grad():
                conv.weight[:, 0, centre, centre] = 1.0

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.log_w, dim=0)
        return sum(w * op(h) for w, op in zip(weights, self.ops))


class PhyCell(nn.Module):
    """
    Physics-constrained ConvLSTM cell (PhyDNet PhyCell).
    """

    def __init__(
        self,
        in_channels:     int,
        hidden_channels: int,
        n_moments:       int = 4,
        kernel_size:     int = 7,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.moments = MomentBank(hidden_channels, n_moments, kernel_size)

        self.gates = nn.Conv2d(
            in_channels + 2 * hidden_channels,
            4 * hidden_channels,
            3, padding=1, bias=True,
        )

    def forward(
        self,
        x:     torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, c = state
        h_phys = self.moments(h)

        i, f, g, o = self.gates(
            torch.cat([x, h, h_phys], dim=1)
        ).chunk(4, dim=1)

        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)
        return h_new, c_new, h_phys

    def init_state(
        self, B: int, H: int, W: int, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(B, self.hidden_channels, H, W, device=device)
        return z, z.clone()


class AppCell(nn.Module):
    """
    Standard ConvLSTM cell for the appearance (non-physical) branch.
    """

    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            3, padding=1, bias=True,
        )

    def forward(
        self,
        x:     torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        i, f, g, o = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)
        return h_new, c_new

    def init_state(
        self, B: int, H: int, W: int, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(B, self.hidden_channels, H, W, device=device)
        return z, z.clone()


class CalvingPhyDNet(nn.Module):
    """
    Calving front SDT predictor built on a PhyDNet core.
    """

    def __init__(
        self,
        in_channels:   int,
        base_channels: int = 32,
        climate_dim:   int = 10,
        tau_dim:       int = 20,
        num_layers:    int = 1,
        n_moments:     int = 4,
        moment_kernel: int = 7,
    ) -> None:
        super().__init__()

        K       = tau_dim // 4
        C_total = in_channels // K
        D_c     = climate_dim // K
        enc_ch  = base_channels
        phy_ch  = base_channels
        app_ch  = base_channels

        self.K       = K
        self.C_total = C_total
        self.enc_ch  = enc_ch
        self.phy_ch  = phy_ch
        self.app_ch  = app_ch

        frame_enc_in = C_total + 4 + D_c
        self.encoder = nn.Sequential(
            nn.Conv2d(frame_enc_in, enc_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(enc_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_ch, enc_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(enc_ch),
            nn.ReLU(inplace=True),
        )

        self.phy_cell = PhyCell(
            in_channels     = enc_ch,
            hidden_channels = phy_ch,
            n_moments       = n_moments,
            kernel_size     = moment_kernel,
        )
        self.app_cell = AppCell(
            in_channels     = enc_ch,
            hidden_channels = app_ch,
        )

        hid_ch = phy_ch + app_ch
        self.head = nn.Sequential(
            nn.Conv2d(hid_ch, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if any(m is op for op in self._moment_ops()):
                    continue
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

    def _moment_ops(self):
        """Yield the moment Conv2d layers so _init_weights can skip them."""
        return list(self.phy_cell.moments.ops)

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
        enc = enc.reshape(B, K, self.enc_ch, H, W)

        device = x.device
        phy_state = self.phy_cell.init_state(B, H, W, device)
        app_state = self.app_cell.init_state(B, H, W, device)

        for t in range(K):
            x_t = enc[:, t]
            phy_h, phy_c, _ = self.phy_cell(x_t, phy_state)
            app_h, app_c    = self.app_cell(x_t, app_state)
            phy_state = (phy_h, phy_c)
            app_state = (app_h, app_c)

        feat = torch.cat([phy_h, app_h], dim=1)

        return self.head(feat)
