"""PyTorch Geometric dataset and Lightning DataModule for trade graphs.

Each year produces one ``torch_geometric.data.Data`` object with:
- ``x``: node features ``(N, F_node)``
- ``edge_index``: directed edges ``(2, E)``
- ``edge_attr``: edge features ``(E, F_edge)``
- ``y``: log-trade targets ``(E,)``
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch_geometric.data import Data

from trade_flow_gcn.data.preprocessing import (
    build_edge_features,
    build_node_features,
)

logger = logging.getLogger(__name__)


def build_graph_for_year(
    df_year: pd.DataFrame,
    country_list: list[str],
    node_feature_cols: list[str] | None = None,
    edge_feature_cols: list[str] | None = None,
) -> Data:
    """Construct a PyG ``Data`` object for a single year.

    Parameters
    ----------
    df_year : pd.DataFrame
        Preprocessed data filtered to a single year. Must contain
        ``iso3_o``, ``iso3_d``, ``log_trade``, and feature columns.
    country_list : list[str]
        Ordered list of ISO3 country codes (defines node indices).
    node_feature_cols : list[str], optional
        Node feature names (default: gdp, gdpcap, pop).
    edge_feature_cols : list[str], optional
        Edge feature column names.

    Returns
    -------
    Data
        PyG graph with ``x``, ``edge_index``, ``edge_attr``, ``y``.
    """
    country_to_idx = {c: i for i, c in enumerate(country_list)}
    year = int(df_year["year"].iloc[0])

    # ── Node features ─────────────────────────────────────────────────
    node_data = build_node_features(df_year, feature_cols=node_feature_cols)
    n_nodes = len(country_list)
    n_node_feat = len(node_feature_cols or ["gdp", "gdpcap", "pop"])
    x = torch.zeros(n_nodes, n_node_feat, dtype=torch.float32)
    for country in country_list:
        idx = country_to_idx[country]
        key = (country, year)
        if key in node_data:
            x[idx] = torch.from_numpy(node_data[key])

    # ── Edges ─────────────────────────────────────────────────────────
    src_indices = []
    dst_indices = []
    edge_feats = []
    targets = []

    for _, row in df_year.iterrows():
        o = row["iso3_o"]
        d = row["iso3_d"]
        if o not in country_to_idx or d not in country_to_idx:
            continue
        src_indices.append(country_to_idx[o])
        dst_indices.append(country_to_idx[d])
        edge_feats.append(build_edge_features(row, edge_feature_cols))
        targets.append(row["log_trade"])

    edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
    edge_attr = torch.from_numpy(np.stack(edge_feats)).float()
    y = torch.tensor(targets, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.year = year
    data.country_list = country_list

    return data


def build_graphs_from_dataframe(
    df: pd.DataFrame,
    country_list: list[str],
    config: dict[str, Any] | None = None,
) -> list[Data]:
    """Build one PyG graph per year from the preprocessed DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Full preprocessed DataFrame (multiple years).
    country_list : list[str]
        Ordered list of ISO3 country codes.
    config : dict, optional
        Data config for feature column names.

    Returns
    -------
    list[Data]
        List of PyG Data objects, one per year, sorted by year.
    """
    data_cfg = (config or {}).get("data", config or {})
    node_feat_cols = data_cfg.get("node_features", ["gdp", "gdpcap", "pop"])
    edge_feat_cols = data_cfg.get(
        "edge_features",
        ["distw", "contig", "comlang_off", "col_dep_ever", "comrelig"],
    )

    graphs = []
    for year, df_year in sorted(df.groupby("year")):
        logger.info("Building graph for year %d (%d edges) ...", year, len(df_year))
        g = build_graph_for_year(
            df_year,
            country_list=country_list,
            node_feature_cols=node_feat_cols,
            edge_feature_cols=edge_feat_cols,
        )
        graphs.append(g)

    return graphs


class TradeDataModule(pl.LightningDataModule):
    """Lightning DataModule that serves per-year trade graphs.

    Splits graphs temporally into train / val / test sets based on
    year ranges specified in the config.

    Since each year is a single graph (full-batch), the dataloaders
    simply iterate over a list of Data objects.
    """

    def __init__(
        self,
        graphs: list[Data],
        train_years: tuple[int, int] = (2000, 2014),
        val_years: tuple[int, int] = (2015, 2016),
        test_years: tuple[int, int] = (2017, 2019),
    ) -> None:
        super().__init__()
        self.graphs = graphs
        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years

        self.train_graphs: list[Data] = []
        self.val_graphs: list[Data] = []
        self.test_graphs: list[Data] = []

    def setup(self, stage: str | None = None) -> None:
        """Split graphs by year into train / val / test."""
        for g in self.graphs:
            y = g.year
            if self.train_years[0] <= y <= self.train_years[1]:
                self.train_graphs.append(g)
            elif self.val_years[0] <= y <= self.val_years[1]:
                self.val_graphs.append(g)
            elif self.test_years[0] <= y <= self.test_years[1]:
                self.test_graphs.append(g)

        logger.info(
            "Data split — train: %d graphs, val: %d graphs, test: %d graphs",
            len(self.train_graphs),
            len(self.val_graphs),
            len(self.test_graphs),
        )

    def train_dataloader(self):
        """Return training graphs as a simple list-based dataloader."""
        from torch_geometric.loader import DataLoader

        return DataLoader(self.train_graphs, batch_size=1, shuffle=True)

    def val_dataloader(self):
        """Return validation graphs."""
        from torch_geometric.loader import DataLoader

        return DataLoader(self.val_graphs, batch_size=1, shuffle=False)

    def test_dataloader(self):
        """Return test graphs."""
        from torch_geometric.loader import DataLoader

        return DataLoader(self.test_graphs, batch_size=1, shuffle=False)
