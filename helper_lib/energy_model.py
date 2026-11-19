import torch
import torch.nn as nn


class EnergyModel(nn.Module):
    """
    Energy-based model for CIFAR-10 RGB images.
    Input:  (batch_size, 3, 32, 32)
    Output: one energy score per image.
    """

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)
