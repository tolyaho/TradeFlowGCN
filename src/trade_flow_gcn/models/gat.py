"""GAT-based model for edge-level trade flow regression.

Architecture
------------
1. Encode node features through stacked GAT layers (Multi-head attention) → node embeddings.
2. For each edge (i → j), concatenate [h_i ‖ h_j ‖ edge_attr].
3. Pass through an MLP decoder → scalar predicted log-trade.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class EdgeDecoder(nn.Module):
    """MLP that maps concatenated (src, dst, edge_feat) → scalar."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        h_src: torch.Tensor,
        h_dst: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Predict trade flow for each edge."""
        x = torch.cat([h_src, h_dst, edge_attr], dim=-1)
        return self.mlp(x).squeeze(-1)


class TradeFlowGAT(nn.Module):
    """GAT encoder + edge-level MLP decoder for trade forecasting.

    Parameters
    ----------
    node_input_dim : int
        Dimension of input node features.
    edge_input_dim : int
        Dimension of edge features.
    hidden_dim : int
        Hidden dimension for GAT layers (per head).
    num_gnn_layers : int
        Number of GAT message-passing layers.
    heads : int
        Number of attention heads.
    decoder_hidden_dim : int
        Hidden dimension in the edge decoder MLP.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        node_input_dim: int = 3,
        edge_input_dim: int = 6,
        hidden_dim: int = 32,
        num_gnn_layers: int = 2,
        heads: int = 4,
        decoder_hidden_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer: node_input_dim → hidden_dim * heads
        self.convs.append(GATConv(node_input_dim, hidden_dim, heads=heads, dropout=dropout))
        self.norms.append(nn.LayerNorm(hidden_dim * heads))

        # Subsequent layers
        for _ in range(num_gnn_layers - 1):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden_dim * heads))

        self.dropout = dropout

        # Edge decoder (input is concatenated node embeddings + edge features)
        decoder_input = 2 * (hidden_dim * heads) + edge_input_dim
        self.decoder = EdgeDecoder(
            input_dim=decoder_input,
            hidden_dim=decoder_hidden_dim,
            dropout=dropout,
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Encode node features through GAT layers."""
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)  # GAT traditionally uses ELU
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Full forward pass."""
        h = self.encode(x, edge_index)
        src, dst = edge_index
        return self.decoder(h[src], h[dst], edge_attr)
