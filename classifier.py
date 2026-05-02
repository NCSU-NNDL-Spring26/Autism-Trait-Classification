"""
classifier.py — MLP classifier for ASD trait classification.

Architecture: input_dim → 8 → 6 → 4 → 8 → 1 (sigmoid)
"""

import torch
import torch.nn as nn


class VanillaMLP(nn.Module):
    """
    MLP binary classifier.
    Architecture: input_dim → 8 → 6 → 4 → 8 → 1 (sigmoid)
    """
    def __init__(self, input_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
