from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int, max_positions: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.max_positions = max_positions

    def forward(self, noise_labels: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.arange(half_dim, device=noise_labels.device, dtype=torch.float32)
        freqs = freqs / max(half_dim - 1, 1)
        freqs = (1.0 / self.max_positions) ** freqs
        args = noise_labels.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([args.cos(), args.sin()], dim=1)

        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))

        return embedding


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_embed_dim, out_channels)

        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(time_embedding).unsqueeze(-1)
        h = self.conv2(self.act2(self.norm2(h)))
        return h + residual


class Downsample1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.op = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class TimeSeriesUNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4),
        time_embed_dim: int = 256,
    ) -> None:
        super().__init__()
        if len(channel_mults) < 2:
            raise ValueError("channel_mults must include at least two resolution levels")

        self.input_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
        self.noise_embed = SinusoidalEmbedding(base_channels)
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_channels: list[int] = []
        current_channels = base_channels

        for level_idx, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            self.down_blocks.append(ResidualBlock1D(current_channels, out_channels, time_embed_dim))
            current_channels = out_channels
            skip_channels.append(out_channels)

            if level_idx < len(channel_mults) - 1:
                self.downsamples.append(Downsample1D(current_channels))

        self.mid_block1 = ResidualBlock1D(current_channels, current_channels, time_embed_dim)
        self.mid_block2 = ResidualBlock1D(current_channels, current_channels, time_embed_dim)

        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for level_idx in range(len(channel_mults) - 2, -1, -1):
            out_channels = base_channels * channel_mults[level_idx]
            self.upsamples.append(Upsample1D(current_channels, out_channels))
            self.up_blocks.append(
                ResidualBlock1D(out_channels + skip_channels[level_idx], out_channels, time_embed_dim)
            )
            current_channels = out_channels

        self.output_norm = nn.GroupNorm(_group_count(current_channels), current_channels)
        self.output_act = nn.SiLU()
        self.output_conv = nn.Conv1d(current_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, noise_labels: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_embed(self.noise_embed(noise_labels))
        h = self.input_conv(x)

        skips = []
        for level_idx, block in enumerate(self.down_blocks):
            h = block(h, time_embedding)
            skips.append(h)
            if level_idx < len(self.downsamples):
                h = self.downsamples[level_idx](h)

        h = self.mid_block1(h, time_embedding)
        h = self.mid_block2(h, time_embedding)

        for upsample, block, skip in zip(self.upsamples, self.up_blocks, reversed(skips[:-1])):
            h = upsample(h)
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode="linear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h, time_embedding)

        return self.output_conv(self.output_act(self.output_norm(h)))


class EDMPrecond1D(nn.Module):
    def __init__(
        self,
        denoise_fn: nn.Module,
        sigma_min: float = 0.0,
        sigma_max: float = float("inf"),
        sigma_data: float = 1.0,
    ) -> None:
        super().__init__()
        self.denoise_fn = denoise_fn
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        x = x.float()
        sigma = sigma.float().reshape(-1, 1, 1)

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)
        c_in = 1.0 / torch.sqrt(self.sigma_data**2 + sigma**2)
        c_noise = sigma.log().flatten() / 4.0

        denoised = self.denoise_fn(c_in * x, c_noise)
        return c_skip * x + c_out * denoised.float()

    def round_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(sigma, device=sigma.device if torch.is_tensor(sigma) else None)
