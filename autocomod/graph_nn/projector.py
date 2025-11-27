import torch
from torch import nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(in_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return F.normalize(x, p=2, dim=-1)
