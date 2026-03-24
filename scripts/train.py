"""Train a trade flow prediction model.

Usage:
    uv run python scripts/train.py --config configs/default.yaml
    uv run python scripts/train.py --config configs/default.yaml --model gcn
    uv run python scripts/train.py --config configs/default.yaml --model mlp_baseline
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from trade_flow_gcn.data.dataset import TradeDataModule, build_graphs_from_dataframe
from trade_flow_gcn.data.preprocessing import preprocess_pipeline
from trade_flow_gcn.models.gcn import TradeFlowGCN
from trade_flow_gcn.models.mlp_baseline import MLPBaseline
from trade_flow_gcn.training.lightning_module import TradeFlowModule
from trade_flow_gcn.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def build_model(config: dict, model_name: str | None = None):
    """Instantiate a model from config.

    Parameters
    ----------
    config : dict
        Full config dictionary.
    model_name : str, optional
        Override model name (gcn, mlp_baseline).

    Returns
    -------
    nn.Module
    """
    model_cfg = config.get("model", {})
    name = model_name or model_cfg.get("name", "gcn")

    if name == "gcn":
        gcn_cfg = model_cfg.get("gcn", {})
        return TradeFlowGCN(
            node_input_dim=gcn_cfg.get("node_input_dim", 3),
            edge_input_dim=gcn_cfg.get("edge_input_dim", 6),
            hidden_dim=gcn_cfg.get("hidden_dim", 64),
            num_gnn_layers=gcn_cfg.get("num_gnn_layers", 3),
            decoder_hidden_dim=gcn_cfg.get("decoder_hidden_dim", 32),
            dropout=gcn_cfg.get("dropout", 0.2),
        )
    elif name == "mlp_baseline":
        mlp_cfg = model_cfg.get("mlp", {})
        return MLPBaseline(
            input_dim=mlp_cfg.get("input_dim", 12),
            hidden_dims=mlp_cfg.get("hidden_dims", [64, 32]),
            dropout=mlp_cfg.get("dropout", 0.2),
        )
    else:
        raise ValueError(f"Unknown model: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TradeFlowGNN")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name override (gcn, mlp_baseline)",
    )
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────
    config = load_config(args.config)
    data_cfg = config["data"]
    train_cfg = config["training"]
    log_cfg = config.get("logging", {})

    # ── Seed ──────────────────────────────────────────────────────────
    pl.seed_everything(train_cfg.get("seed", 42), workers=True)

    # ── Data ──────────────────────────────────────────────────────────
    raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))
    
    # List all CSVs for debugging
    csv_candidates = list(raw_dir.glob("*.csv"))
    logger.info("Found %d CSV files in %s:", len(csv_candidates), raw_dir)
    for c in csv_candidates:
        logger.info("  - %s (%.2f MB)", c.name, c.stat().st_size / 1e6)

    if not csv_candidates:
        logger.error(
            "No CSV files found in %s. Run `python scripts/download_data.py` first.",
            raw_dir,
        )
        sys.exit(1)
    
    # Pick the largest file (The 1.25GB Gravity file)
    csv_path = max(csv_candidates, key=lambda p: p.stat().st_size)
    logger.info("Selected main data file: %s", csv_path.name)

    logger.info("Preprocessing data from %s ...", csv_path)
    df = preprocess_pipeline(csv_path, config)

    logger.info("Building per-year graphs ...")
    graphs = build_graphs_from_dataframe(df, data_cfg["countries"], config)

    datamodule = TradeDataModule(
        graphs=graphs,
        train_years=tuple(data_cfg["train_years"]),
        val_years=tuple(data_cfg["val_years"]),
        test_years=tuple(data_cfg["test_years"]),
        num_workers=data_cfg.get("num_workers", 0),
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = build_model(config, args.model)
    logger.info("Model: %s", model.__class__.__name__)

    lit_module = TradeFlowModule(
        model=model,
        learning_rate=train_cfg.get("learning_rate", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-4),
        scheduler_config=train_cfg.get("scheduler", {}),
    )

    # ── Callbacks ─────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best-{epoch:02d}-{val_loss:.4f}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    es_cfg = train_cfg.get("early_stopping", {})
    if es_cfg:
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg.get("monitor", "val_loss"),
                patience=es_cfg.get("patience", 20),
                mode=es_cfg.get("mode", "min"),
            )
        )

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=train_cfg.get("max_epochs", 200),
        callbacks=callbacks,
        default_root_dir=log_cfg.get("save_dir", "lightning_logs"),
        log_every_n_steps=log_cfg.get("log_every_n_steps", 1),
        deterministic=True,
    )

    # ── Train ─────────────────────────────────────────────────────────
    logger.info("Starting training ...")
    trainer.fit(lit_module, datamodule=datamodule)

    # ── Test ──────────────────────────────────────────────────────────
    logger.info("Running test evaluation ...")
    trainer.test(lit_module, datamodule=datamodule, ckpt_path="best")

    logger.info("✓ Training complete!")


if __name__ == "__main__":
    main()
