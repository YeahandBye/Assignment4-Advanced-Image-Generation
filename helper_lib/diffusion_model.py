import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    """Encode diffusion time using sine and cosine."""

    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        frequencies = torch.exp(
            torch.arange(
                half, device=t.device, dtype=t.dtype
            ) * (-math.log(10000.0) / (half - 1))
        )

        angles = t[:, None] * frequencies[None, :] * 1000
        return torch.cat(
            [torch.sin(angles), torch.cos(angles)], dim=1
        )


class SmallUNet(nn.Module):
    """Small RGB noise predictor for CIFAR-10 (32x32)."""

    def __init__(self):
        super().__init__()

        self.time_embedding = SinusoidalEmbedding(32)
        self.time_projection = nn.Linear(32, 128)

        self.input_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.SiLU()
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU()
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.SiLU()
        )

        self.middle = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU()
        )

        self.up1 = nn.ConvTranspose2d(
            128, 64, 4, stride=2, padding=1
        )

        self.merge1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.SiLU()
        )

        self.up2 = nn.ConvTranspose2d(
            64, 32, 4, stride=2, padding=1
        )

        self.merge2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.SiLU()
        )

        self.output_conv = nn.Conv2d(
            32, 3, 3, padding=1
        )

    def forward(self, x, t):
        t = t.view(-1)
        time_features = self.time_embedding(t)
        time_features = self.time_projection(time_features)
        time_features = time_features[:, :, None, None]

        x0 = self.input_conv(x)
        d1 = self.down1(x0)
        d2 = self.down2(d1)

        middle = self.middle(d2 + time_features)

        u1 = self.up1(middle)
        u1 = self.merge1(torch.cat([u1, d1], dim=1))

        u2 = self.up2(u1)
        u2 = self.merge2(torch.cat([u2, x0], dim=1))

        return self.output_conv(u2)


def cosine_schedule(t):
    """Signal and noise rates for forward diffusion."""
    signal_rate = torch.cos(t * math.pi / 2)
    noise_rate = torch.sin(t * math.pi / 2)
    return signal_rate, noise_rate
