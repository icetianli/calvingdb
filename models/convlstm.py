"""
ConvLSTM model for calving front SDT displacement prediction
(Shi et al., NeurIPS 2015)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell (Shi et al. 2015).
    """

    def __init__(
        self,
        in_channels:     int,
        hidden_channels: int,
        kernel_size:     int = 3,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=True,
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
        self,
        B:      int,
        H:      int,
        W:      int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(B, self.hidden_channels, H, W, device=device)
        return z, z.clone()


class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM translator (OpenSTL convention).
    """

    def __init__(
        self,
        in_channels:     int,
        hidden_channels: int,
        num_layers:      int = 3,
        kernel_size:     int = 3,
    ) -> None:
        super().__init__()
        self.cells = nn.ModuleList([
            ConvLSTMCell(
                in_channels     if i == 0 else hidden_channels,
                hidden_channels,
                kernel_size,
            )
            for i in range(num_layers)
        ])

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Run the ConvLSTM over a (B, K, C, H, W) sequence and return the last layer's
        hidden state at the final time step.
        """
        B, K, _, H, W = x_seq.shape
        device = x_seq.device
        states = [cell.init_state(B, H, W, device) for cell in self.cells]

        for t in range(K):
            x_t = x_seq[:, t]
            for layer_idx, cell in enumerate(self.cells):
                h_new, c_new = cell(x_t, states[layer_idx])
                states[layer_idx] = (h_new, c_new)
                x_t = h_new

        return states[-1][0]


class CalvingUNet(nn.Module):
    """
    Calving front SDT predictor built on a ConvLSTM core.
    """

    def __init__(
        self,
        in_channels:   int,
        base_channels: int = 32,
        climate_dim:   int = 8,
        tau_dim:       int = 16,
        num_layers:    int = 3,
    ) -> None:
        super().__init__()

        K         = tau_dim // 4
        C_total   = in_channels // K
        D_c       = climate_dim // K
        hidden_ch = base_channels * 2

        self.K       = K
        self.C_total = C_total

        frame_enc_in = C_total + 4 + D_c
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(frame_enc_in, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.convlstm = ConvLSTM(
            in_channels     = base_channels,
            hidden_channels = hidden_ch,
            num_layers      = num_layers,
        )

        self.head = nn.Sequential(
            nn.Conv2d(hidden_ch, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 1),
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

        enc = self.frame_encoder(x_aug)
        enc = enc.reshape(B, K, -1, H, W)

        hidden = self.convlstm(enc)

        return self.head(hidden)
