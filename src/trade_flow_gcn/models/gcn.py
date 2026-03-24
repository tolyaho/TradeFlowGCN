"""GCN-based model for edge-level trade flow regression.

Architecture
------------
1. Encode node features through stacked GCN layers → node embeddings.
2. For each edge (i → j), concatenate [h_i ‖ h_j ‖ edge_attr].
3. Pass through an MLP decoder → scalar predicted log-trade.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


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
        """Predict trade flow for each edge.

        Parameters
        ----------
        h_src : (E, H) node embeddings of source nodes
        h_dst : (E, H) node embeddings of destination nodes
        edge_attr : (E, F_edge) edge features

        Returns
        -------
        (E,) predicted log-trade values
        """
        x = torch.cat([h_src, h_dst, edge_attr], dim=-1)
        return self.mlp(x).squeeze(-1)


class TradeFlowGCN(nn.Module):
    """GCN encoder + edge-level MLP decoder for trade forecasting.

    Parameters
    ----------
    node_input_dim : int
        Dimension of input node features.
    edge_input_dim : int
        Dimension of edge features.
    hidden_dim : int
        Hidden dimension for GCN layers.
    num_gnn_layers : int
        Number of GCN message-passing layers.
    decoder_hidden_dim : int
        Hidden dimension in the edge decoder MLP.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        node_input_dim: int = 3,
        edge_input_dim: int = 6,
        hidden_dim: int = 64,
        num_gnn_layers: int = 3,
        decoder_hidden_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer: node_input_dim → hidden_dim
        self.convs.append(GCNConv(node_input_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))

        # Subsequent layers: hidden_dim → hidden_dim
        for _ in range(num_gnn_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = dropout

        # Edge decoder
        decoder_input = 2 * hidden_dim + edge_input_dim
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
        """Encode node features through GCN layers.

        Parameters
        ----------
        x : (N, F_node)
        edge_index : (2, E)

        Returns
        -------
        (N, hidden_dim) node embeddings
        """
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Full forward pass: encode nodes → decode edges.

        Parameters
        ----------
        x : (N, F_node) node features
        edge_index : (2, E) directed edges
        edge_attr : (E, F_edge) edge features

        Returns
        -------
        (E,) predicted log-trade values
        """
        h = self.encode(x, edge_index)
        src, dst = edge_index
        return self.decoder(h[src], h[dst], edge_attr)
