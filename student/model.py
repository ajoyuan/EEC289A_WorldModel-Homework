"""Student world model.

Students may replace this residual MLP with a GRU or another dynamics model,
but the public interface must stay the same.
"""

from __future__ import annotations

import torch
from torch import nn


class StudentWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 3,
        use_gru: bool = False,
        delta_limit: float = 3.0,
    ):
        super().__init__()
        self.use_gru = bool(use_gru)
        self.delta_limit = float(delta_limit)

        in_dim = obs_dim + act_dim
        self.input_layer = nn.Linear(in_dim, hidden_dim)

        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)

        self.head = nn.Linear(hidden_dim, obs_dim)
        self.act = nn.SiLU()

    def initial_hidden(self, batch_size: int, device: torch.device):
        return None

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        if self.training:
            obs_norm = obs_norm + torch.randn_like(obs_norm) * 0.003

        h = self.act(self.input_layer(torch.cat([obs_norm, act_norm], dim=-1)))

        residual = h
        h = self.act(self.layer1(h))
        h = self.act(self.layer2(h) + 0.5 * residual)

        raw_delta = self.head(h)
        delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)
        return delta, hidden
