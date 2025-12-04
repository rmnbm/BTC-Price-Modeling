"""
Quick EDA + modeling pipeline for BTC daily features.

Steps:
1) Load btc_features_daily.csv, ensure sorted and clean.
2) Create a next-day close target.
3) Run correlation/summary checks.
4) Train simple models (persistence baseline, linear regression, gradient boosting, random forest).
5) Report MAE/RMSE/MAPE on a chronological train/val/test split to avoid look-ahead bias.

Requires: pandas, numpy, scikit-learn.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("btc_features_daily.csv")


@dataclass
class DatasetSplits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    # Predict next-day close; drop last row where target is missing.
    df = df.copy()
    df["target_close_next"] = df["close"].shift(-1)
    df = df.iloc[:-1]
    return df


def train_val_test_split_chrono(
    X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> DatasetSplits:
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return DatasetSplits(
        X_train=X.iloc[:train_end],
        y_train=y.iloc[:train_end],
        X_val=X.iloc[train_end:val_end],
        y_val=y.iloc[train_end:val_end],
        X_test=X.iloc[val_end:],
        y_test=y.iloc[val_end:],
    )


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    pct_err = np.abs((y_true - y_pred) / y_true)
    pct_err = pct_err.replace([np.inf, -np.inf], np.nan).dropna()
    return float(pct_err.mean() * 100)


def evaluate_model(
    name: str, model, splits: DatasetSplits
) -> Dict[str, float | str]:
    model.fit(splits.X_train, splits.y_train)
    preds = model.predict(splits.X_test)
    return {
        "model": name,
        "MAE": mean_absolute_error(splits.y_test, preds),
        "RMSE": rmse(splits.y_test, preds),
        "MAPE%": mape(splits.y_test, preds),
    }


def main() -> None:
    df = load_data()
    print(f"Loaded {len(df):,} rows with {len(df.columns)} columns from {DATA_PATH}")
    print("Missing ratio (top 10):")
    missing = df.isna().mean().sort_values(ascending=False).head(10)
    print(missing)

    df = create_target(df)
    feature_cols = [c for c in df.columns if c not in ["timestamp", "target_close_next"]]
    X = df[feature_cols]
    y = df["target_close_next"]

    splits = train_val_test_split_chrono(X, y)

    # Correlation check (top 10 absolute)
    corr = (
        df.drop(columns=["timestamp"])
        .corr()["target_close_next"]
        .drop("target_close_next")
        .abs()
        .sort_values(ascending=False)
        .head(10)
    )
    print("\nTop 10 correlations with target_close_next:")
    print(corr)

    # Baseline: persistence (predict today's close for tomorrow)
    persistence_pred = splits.X_test["close"]
    baseline = {
        "model": "persistence_close",
        "MAE": mean_absolute_error(splits.y_test, persistence_pred),
        "RMSE": rmse(splits.y_test, persistence_pred),
        "MAPE%": mape(splits.y_test, persistence_pred),
    }

    models = [
        make_pipeline(StandardScaler(), LinearRegression()),
        make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        GradientBoostingRegressor(random_state=0),
        RandomForestRegressor(
            n_estimators=200, random_state=0, n_jobs=-1, min_samples_leaf=2
        ),
    ]
    names = ["linear_reg", "ridge", "gboost", "rf"]

    results: List[Dict[str, float | str]] = [baseline]
    for name, model in zip(names, models):
        results.append(evaluate_model(name, model, splits))

    results = sorted(results, key=lambda r: r["RMSE"])
    print("\nMetrics on test set (chronological split):")
    for r in results:
        print(r)

    # Feature importance from RF for quick inspection
    rf_model = models[-1].fit(splits.X_train, splits.y_train)
    importances = (
        pd.Series(rf_model.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
        .head(10)
    )
    print("\nRF top 10 feature importances:")
    print(importances)

    print("\nTarget summary:")
    print(y.describe())


if __name__ == "__main__":
    main()
