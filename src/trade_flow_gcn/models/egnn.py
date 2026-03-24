"""EGNN-based model for edge-level trade flow regression.

This model uses edge features to modulate the messages between nodes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class EGNNLayer(MessagePassing):
    """Simple Edge-conditioned Message Passing layer."""

    def __init__(self, in_channels: int, out_channels: int, edge_channels: int):
        super().__init__(aggr='mean')
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # Concatenate [src, dst, edge_feat]
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.msg_mlp(msg_input)

    def update(self, aggr_out, x):
        # Combine original node feature with aggregated messages
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.node_mlp(update_input)


class TradeFlowEGNN(nn.Module):
    """EGNN encoder + edge-level MLP decoder."""

    def __init__(
        self,
        node_input_dim: int = 3,
        edge_input_dim: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 3,
        decoder_hidden_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        # First layer
        self.layers.append(EGNNLayer(node_input_dim, hidden_dim, edge_input_dim))
        
        # Subsequent layers
        for _ in range(num_layers - 1):
            self.layers.append(EGNNLayer(hidden_dim, hidden_dim, edge_input_dim))

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

    def encode(self, x, edge_index, edge_attr):
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x, edge_index, edge_attr):
        h = self.encode(x, edge_index, edge_attr)
        src, dst = edge_index
        # Concatenate node embeddings and original edge features for decoder
        decoder_in = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        return self.decoder(decoder_in).squeeze(-1)
