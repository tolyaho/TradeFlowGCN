"""Preprocessing pipeline for the CEPII Gravity dataset.

Transforms raw CSV into per-year graph-ready DataFrames with node features,
edge features, and log-trade targets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Default column mappings ───────────────────────────────────────────────


# CEPII columns we care about
_REQUIRED_COLS = [
    "year",
    "iso3_o",        # origin country ISO3
    "iso3_d",        # destination country ISO3
    "tradeflow_comtrade_o",  # export flow value (origin → destination)
    "gdp_o",
    "gdp_d",
    "gdpcap_o",
    "gdpcap_d",
    "pop_o",
    "pop_d",
    "distw",         # weighted distance
    "contig",        # contiguity
    "comlang_off",   # common official language
    "col_dep_ever",  # colonial relationship
    "comrelig",      # common religion index
]

# Optional columns (may or may not be in the dataset)
_OPTIONAL_COLS = [
    "fta_wto",       # WTO-notified FTA
]


def load_and_filter(
    csv_path: str | Path,
    countries: list[str],
    year_start: int = 2000,
    year_end: int = 2019,
) -> pd.DataFrame:
    """Load the raw CEPII CSV and filter to the study subset.

    Parameters
    ----------
    csv_path : str or Path
        Path to the raw Gravity CSV.
    countries : list[str]
        ISO3 codes of countries to include.
    year_start, year_end : int
        Inclusive year range.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with required columns.
    """
    logger.info("Loading raw data from %s ...", csv_path)

    # Determine which columns to load
    cols_to_use = list(_REQUIRED_COLS)
    # Read a small sample to check which optional cols exist
    sample = pd.read_csv(csv_path, nrows=5)
    for col in _OPTIONAL_COLS:
        if col in sample.columns:
            cols_to_use.append(col)

    df = pd.read_csv(csv_path, usecols=cols_to_use, low_memory=False)
    logger.info("Raw data shape: %s", df.shape)

    # Filter years
    df = df[(df["year"] >= year_start) & (df["year"] <= year_end)]

    # Filter countries (both origin and destination must be in the set)
    country_set = set(countries)
    df = df[df["iso3_o"].isin(country_set) & df["iso3_d"].isin(country_set)]

    # Remove self-loops (country exporting to itself)
    df = df[df["iso3_o"] != df["iso3_d"]]

    # Drop rows with missing trade flow values
    df = df.dropna(subset=["tradeflow_comtrade_o"])
    df = df[df["tradeflow_comtrade_o"] > 0]

    logger.info("Filtered data shape: %s", df.shape)
    return df.reset_index(drop=True)


def compute_log_trade_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add the log-trade target column: y = log(1 + trade_value).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``tradeflow_comtrade_o`` column.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``log_trade`` column added.
    """
    df = df.copy()
    df["log_trade"] = np.log1p(df["tradeflow_comtrade_o"])
    return df


def build_node_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> dict[tuple[str, int], np.ndarray]:
    """Build a mapping from (country, year) to node feature vector.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame.
    feature_cols : list[str], optional
        Node-level features to include. Defaults to GDP, GDPcap, pop.

    Returns
    -------
    dict
        Mapping ``(iso3, year) → np.ndarray`` of shape ``(n_features,)``.
    """
    if feature_cols is None:
        feature_cols = ["gdp", "gdpcap", "pop"]

    # Build from origin-side columns
    node_data: dict[tuple[str, int], np.ndarray] = {}
    for _, row in df.iterrows():
        year = int(row["year"])
        origin = row["iso3_o"]
        dest = row["iso3_d"]

        # Origin node features
        if (origin, year) not in node_data:
            feats = []
            for f in feature_cols:
                col_name = f"{f}_o"
                val = row.get(col_name, np.nan)
                feats.append(np.log1p(max(0, val)) if pd.notna(val) else 0.0)
            node_data[(origin, year)] = np.array(feats, dtype=np.float32)

        # Destination node features
        if (dest, year) not in node_data:
            feats = []
            for f in feature_cols:
                col_name = f"{f}_d"
                val = row.get(col_name, np.nan)
                feats.append(np.log1p(max(0, val)) if pd.notna(val) else 0.0)
            node_data[(dest, year)] = np.array(feats, dtype=np.float32)

    return node_data


def build_edge_features(
    row: pd.Series,
    edge_feature_cols: list[str] | None = None,
) -> np.ndarray:
    """Extract edge feature vector from a single DataFrame row.

    Parameters
    ----------
    row : pd.Series
        A single row of the preprocessed DataFrame.
    edge_feature_cols : list[str], optional
        Column names for edge features.

    Returns
    -------
    np.ndarray
        Edge feature vector of shape ``(n_edge_features,)``.
    """
    if edge_feature_cols is None:
        edge_feature_cols = [
            "distw", "contig", "comlang_off", "col_dep_ever", "comrelig",
        ]

    feats = []
    for col in edge_feature_cols:
        val = row.get(col, np.nan)
        if pd.isna(val):
            feats.append(0.0)
        elif col == "distw":
            # Log-transform distance
            feats.append(np.log1p(val))
        else:
            feats.append(float(val))
    return np.array(feats, dtype=np.float32)


def preprocess_pipeline(
    csv_path: str | Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Run the full preprocessing pipeline.

    Parameters
    ----------
    csv_path : str or Path
        Path to the raw CEPII CSV.
    config : dict
        Configuration dictionary (``data`` section).

    Returns
    -------
    pd.DataFrame
        Fully preprocessed DataFrame ready for graph construction.
    """
    data_cfg = config.get("data", config)

    df = load_and_filter(
        csv_path,
        countries=data_cfg["countries"],
        year_start=data_cfg.get("year_start", 2000),
        year_end=data_cfg.get("year_end", 2019),
    )
    df = compute_log_trade_target(df)

    # Fill missing edge features
    edge_feat_cols = data_cfg.get(
        "edge_features",
        ["distw", "contig", "comlang_off", "col_dep_ever", "comrelig"],
    )
    for col in edge_feat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    logger.info("Preprocessing complete. Final shape: %s", df.shape)
    return df
