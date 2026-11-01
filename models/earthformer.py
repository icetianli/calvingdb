"""
EarthFormer model for calving front SDT displacement prediction (Gao et al., NeurIPS 2022)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint


def _pad_to_multiple(
    x: torch.Tensor,
    window_size: tuple[int, int, int],
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """
    Pad (B, T, H, W, C) so T, H, W are each divisible by the corresponding window
    dimension.
    """
    _, T, H, W, _ = x.shape
    Tw, Hw, Ww = window_size

    pt = (Tw - T % Tw) % Tw
    ph = (Hw - H % Hw) % Hw
    pw = (Ww - W % Ww) % Ww

    if pt or ph or pw:
        x = F.pad(x, (0, 0,
                      0, pw,
                      0, ph,
                      0, pt))
    return x, (pt, ph, pw)


def _window_partition(
    x: torch.Tensor,
    window_size: tuple[int, int, int],
) -> torch.Tensor:
    """
    Partition (B, T, H, W, C) into non-overlapping cuboids of size (Tw, Hw, Ww).
    """
    B, T, H, W, C = x.shape
    Tw, Hw, Ww    = window_size

    x = x.view(B, T // Tw, Tw, H // Hw, Hw, W // Ww, Ww, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return x.view(-1, Tw * Hw * Ww, C)


def _window_reverse(
    windows: torch.Tensor,
    window_size: tuple[int, int, int],
    B: int, T: int, H: int, W: int,
) -> torch.Tensor:
    """Inverse of _window_partition, returning a (B, T, H, W, C) tensor."""
    Tw, Hw, Ww = window_size
    C          = windows.shape[-1]

    x = windows.view(B, T // Tw, H // Hw, W // Ww, Tw, Hw, Ww, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(B, T, H, W, C)


class _RelPosBias3D(nn.Module):
    """
    Learnable 3-D relative-position bias for cuboid attention.
    """

    def __init__(self, window_size: tuple[int, int, int], num_heads: int) -> None:
        super().__init__()
        Tw, Hw, Ww = window_size

        n_entries = (2 * Tw - 1) * (2 * Hw - 1) * (2 * Ww - 1)
        self.table = nn.Parameter(torch.zeros(n_entries, num_heads))
        nn.init.trunc_normal_(self.table, std=0.02)

        self.register_buffer("index", self._make_index(Tw, Hw, Ww))

    @staticmethod
    def _make_index(Tw: int, Hw: int, Ww: int) -> torch.Tensor:
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(Tw), torch.arange(Hw), torch.arange(Ww),
                indexing="ij",
            )
        )
        flat = coords.flatten(1)
        rel  = flat[:, :, None] - flat[:, None, :]
        rel  = rel.permute(1, 2, 0).contiguous()
        rel[..., 0] += Tw - 1
        rel[..., 1] += Hw - 1
        rel[..., 2] += Ww - 1
        rel[..., 0] *= (2 * Hw - 1) * (2 * Ww - 1)
        rel[..., 1] *= (2 * Ww - 1)
        return rel.sum(-1)

    def forward(self) -> torch.Tensor:
        """Returns (N, N, num_heads) additive bias."""
        N = self.index.shape[0]
        return self.table[self.index.view(-1)].view(N, N, -1)


class CuboidAttention(nn.Module):
    """
    3-D cuboid self-attention layer (EarthFormer §3.2).
    """

    def __init__(
        self,
        dim:         int,
        window_size: tuple[int, int, int],
        num_heads:   int,
        shift:       bool  = False,
        qkv_bias:    bool  = True,
        attn_drop:   float = 0.0,
        proj_drop:   float = 0.0,
    ) -> None:
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.num_heads   = num_heads
        self.shift       = shift
        self.scale       = (dim // num_heads) ** -0.5

        self.rel_pos   = _RelPosBias3D(window_size, num_heads)
        self.qkv       = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cuboid self-attention to an input of shape (B, T, H, W, C)."""
        B, T, H, W, C = x.shape
        Tw, Hw, Ww    = self.window_size

        x, (pt, ph, pw) = _pad_to_multiple(x, self.window_size)
        Tp, Hp, Wp = x.shape[1], x.shape[2], x.shape[3]

        if self.shift:
            st, sh, sw = Tw // 2, Hw // 2, Ww // 2
            x = torch.roll(x, shifts=(-st, -sh, -sw), dims=(1, 2, 3))

        wins           = _window_partition(x, self.window_size)
        n_wins, N, _   = wins.shape
        H_n            = self.num_heads
        d              = C // H_n

        qkv = (
            self.qkv(wins)
            .reshape(n_wins, N, 3, H_n, d)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        bias = self.rel_pos()
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)

        attn = self.attn_drop(attn.softmax(dim=-1))
        wins = (attn @ v).transpose(1, 2).reshape(n_wins, N, C)
        wins = self.proj_drop(self.proj(wins))

        x = _window_reverse(wins, self.window_size, B, Tp, Hp, Wp)

        if self.shift:
            x = torch.roll(x, shifts=(st, sh, sw), dims=(1, 2, 3))

        x = x[:, :T, :H, :W, :].contiguous()
        return x


class EarthFormerBlock(nn.Module):
    """
    Single EarthFormer block: pre-norm CuboidAttention → pre-norm FFN.
    """

    def __init__(
        self,
        dim:         int,
        window_size: tuple[int, int, int],
        num_heads:   int,
        shift:       bool  = False,
        mlp_ratio:   float = 4.0,
        drop:        float = 0.0,
        attn_drop:   float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = CuboidAttention(
            dim, window_size, num_heads,
            shift=shift, attn_drop=attn_drop, proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the block to an input of shape (B, T, H, W, C), preserving its shape.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CalvingEarthFormer(nn.Module):
    """
    Calving front SDT predictor built on EarthFormer's cuboid-attention core.
    """

    def __init__(
        self,
        in_channels:   int,
        base_channels: int                  = 64,
        climate_dim:   int                  = 10,
        tau_dim:       int                  = 20,
        num_layers:    int                  = 4,
        num_heads:     int                  = 4,
        window_size:   tuple[int, int, int] = (5, 8, 8),
        mlp_ratio:     float                = 4.0,
        drop:          float                = 0.0,
        attn_drop:     float                = 0.0,
    ) -> None:
        super().__init__()

        K       = tau_dim // 4
        C_total = in_channels // K
        D_c     = climate_dim // K
        E       = base_channels

        self.K = K
        self.E = E

        frame_enc_in = C_total + 4 + D_c
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(frame_enc_in, E, 3, padding=1, bias=False),
            nn.BatchNorm2d(E),
            nn.GELU(),
            nn.Conv2d(E, E, 3, padding=1, bias=False),
            nn.BatchNorm2d(E),
            nn.GELU(),
        )

        Tw, Hw, Ww  = window_size
        eff_window  = (min(Tw, K), Hw, Ww)

        self.blocks = nn.ModuleList([
            EarthFormerBlock(
                dim         = E,
                window_size = eff_window,
                num_heads   = num_heads,
                shift       = (i % 2 == 1),
                mlp_ratio   = mlp_ratio,
                drop        = drop,
                attn_drop   = attn_drop,
            )
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(E)

        self.head = nn.Sequential(
            nn.Conv2d(E, E, 3, padding=1, bias=False),
            nn.BatchNorm2d(E),
            nn.GELU(),
            nn.Conv2d(E, 1, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
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
        enc = enc.reshape(B, K, self.E, H, W)

        h = enc.permute(0, 1, 3, 4, 2).contiguous()

        for block in self.blocks:
            h = torch_checkpoint.checkpoint(
                block, h, use_reentrant=False,
            )
        h = self.norm(h)

        h = h[:, -1, :, :, :].permute(0, 3, 1, 2).contiguous()

        return self.head(h)
