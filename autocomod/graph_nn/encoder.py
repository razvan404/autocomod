import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv


class GnnEncoder(nn.Module):
    def __init__(self, in_channels, hidden=128, out_channels=64):
        super().__init__()

        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, out_channels)

        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x, edge_index):
        x1 = F.relu(self.conv1(x, edge_index))
        x1 = self.norm1(x1)

        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = self.norm2(x2)

        x3 = self.conv3(x2, edge_index)
        return x3
