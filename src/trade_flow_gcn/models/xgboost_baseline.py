"""XGBoost-based baseline for trade flow regression."""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class XGBoostBaseline:
    """XGBoost model for bilateral trade prediction.
    
    This baseline ignores the graph structure but uses the same
    concatenated features as the MLP.
    """

    def __init__(self, **params) -> None:
        default_params = {
            "n_estimators": 1000,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "objective": "reg:squarederror",
            "early_stopping_rounds": 50
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model = xgb.XGBRegressor(**self.params)

    def _prepare_data(
        self,
        x_src: np.ndarray,
        x_dst: np.ndarray,
        edge_attr: np.ndarray,
    ) -> np.ndarray:
        """Concatenate features into a single tabular input."""
        return np.concatenate([x_src, x_dst, edge_attr], axis=1)

    def fit(
        self,
        x_src: np.ndarray,
        x_dst: np.ndarray,
        edge_attr: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list] = None
    ) -> None:
        """Fit the XGBoost model."""
        X = self._prepare_data(x_src, x_dst, edge_attr)
        
        if eval_set:
            X_val, y_val = eval_set
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X, y)

    def predict(
        self,
        x_src: np.ndarray,
        x_dst: np.ndarray,
        edge_attr: np.ndarray,
    ) -> np.ndarray:
        """Predict trade flows."""
        X = self._prepare_data(x_src, x_dst, edge_attr)
        return self.model.predict(X)

    def evaluate(
        self,
        x_src: np.ndarray,
        x_dst: np.ndarray,
        edge_attr: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        preds = self.predict(x_src, x_dst, edge_attr)
        return {
            "rmse": float(np.sqrt(mean_squared_error(y, preds))),
            "mae": float(mean_absolute_error(y, preds)),
            "r2": float(r2_score(y, preds)),
        }
