"""RGCN-based model for trade flow regression."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class TradeFlowRGCN(nn.Module):
    """RGCN encoder + edge-level MLP decoder.
    
    In this implementation, we define 'relations' by binning the 
    distance between countries (near, medium, far).
    """

    def __init__(
        self,
        node_input_dim: int = 3,
        edge_input_dim: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_relations: int = 3,
        decoder_hidden_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.convs = nn.ModuleList()
        # First layer
        self.convs.append(RGCNConv(node_input_dim, hidden_dim, num_relations))
        
        # Subsequent layers
        for _ in range(num_layers - 1):
            self.convs.append(RGCNConv(hidden_dim, hidden_dim, num_relations))

        self.dropout = dropout

        # Edge decoder
        decoder_input = 2 * hidden_dim + edge_input_dim
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input, decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden_dim, decoder_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim // 2, 1),
        )

    def encode(self, x, edge_index, edge_type):
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index, edge_attr):
        # Infer edge types from distance (the first edge feature usually)
        # Assuming distw_harmonic is the first edge feature
        dist = edge_attr[:, 0]
        # Simple binning into relations
        # 0: Near, 1: Medium, 2: Far
        q1, q2 = 5000, 10000 # Rough km thresholds
        edge_type = torch.zeros(dist.size(0), dtype=torch.long, device=dist.device)
        edge_type[dist > q1] = 1
        edge_type[dist > q2] = 2
        
        h = self.encode(x, edge_index, edge_type)
        src, dst = edge_index
        decoder_in = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        return self.decoder(decoder_in).squeeze(-1)
