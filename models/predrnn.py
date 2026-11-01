"""
PredRNN model for calving front SDT displacement prediction (Wang et al., TPAMI 2022)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpatioTemporalLSTMCell(nn.Module):
    """
    PredRNN ST-LSTM cell (Wang et al. 2017).
    """

    def __init__(
        self,
        in_channels:     int,
        hidden_channels: int,
        kernel_size:     int = 3,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        pad = kernel_size // 2

        self.gates_xh = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size, padding=pad, bias=True,
        )
        self.gates_m = nn.Conv2d(
            hidden_channels,
            3 * hidden_channels,
            kernel_size, padding=pad, bias=True,
        )
        self.conv_1x1 = nn.Conv2d(2 * hidden_channels, hidden_channels, 1, bias=False)

    def forward(
        self,
        x:     torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, c, m = state

        xh = torch.cat([x, h], dim=1)
        i, f, g, o = self.gates_xh(xh).chunk(4, dim=1)
        i_prime, f_prime, g_prime = self.gates_m(m).chunk(3, dim=1)

        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)

        m_new = torch.sigmoid(f_prime) * m + torch.sigmoid(i_prime) * torch.tanh(g_prime)

        h_new = torch.sigmoid(o) * torch.tanh(self.conv_1x1(
            torch.cat([c_new, m_new], dim=1)
        ))

        return h_new, c_new, m_new

    def init_state(
        self,
        B:      int,
        H:      int,
        W:      int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = torch.zeros(B, self.hidden_channels, H, W, device=device)
        return z, z.clone(), z.clone()


class PredRNN(nn.Module):
    """
    Multi-layer PredRNN translator.
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
            SpatioTemporalLSTMCell(
                in_channels     if i == 0 else hidden_channels,
                hidden_channels,
                kernel_size,
            )
            for i in range(num_layers)
        ])

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Run the PredRNN over a (B, K, C, H, W) sequence and return the last layer's
        hidden state at the final time step.
        """
        B, K, _, H, W = x_seq.shape
        device = x_seq.device

        states = [cell.init_state(B, H, W, device) for cell in self.cells]
        m = torch.zeros(B, self.cells[0].hidden_channels, H, W, device=device)

        for t in range(K):
            x_t = x_seq[:, t]

            for layer_idx, cell in enumerate(self.cells):
                h, c, _ = states[layer_idx]
                h_new, c_new, m = cell(x_t, (h, c, m))
                states[layer_idx] = (h_new, c_new, m)
                x_t = h_new

        return states[-1][0]


class CalvingPredRNN(nn.Module):
    """
    Calving front SDT predictor built on a PredRNN (ST-LSTM) core.
    """

    def __init__(
        self,
        in_channels:   int,
        base_channels: int = 32,
        climate_dim:   int = 10,
        tau_dim:       int = 20,
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

        self.predrnn = PredRNN(
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

        hidden = self.predrnn(enc)

        return self.head(hidden)
