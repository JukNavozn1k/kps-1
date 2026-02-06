from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


TargetName = Literal["home_goals", "away_goals", "total_goals", "goal_diff"]


@dataclass(frozen=True)
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    feature_names: list[str]
    target_name: TargetName
    y_mean: float | None
    y_std: float | None


def filter_dataset_features(
    dataset: Dataset,
    selected_features: list[str],
) -> Dataset:
    """Фильтрует датасет, оставляя только выбранные признаки."""
    if not selected_features:
        return dataset
    
    # Найти индексы выбранных признаков
    indices = []
    for feat in selected_features:
        if feat in dataset.feature_names:
            indices.append(dataset.feature_names.index(feat))
    
    if not indices:
        return dataset
    
    indices_arr = np.array(indices)
    
    return Dataset(
        X_train=dataset.X_train[:, indices_arr],
        y_train=dataset.y_train,
        X_val=dataset.X_val[:, indices_arr],
        y_val=dataset.y_val,
        feature_names=[dataset.feature_names[i] for i in indices],
        target_name=dataset.target_name,
        y_mean=dataset.y_mean,
        y_std=dataset.y_std,
    )


def _parse_result(result: str) -> tuple[float, float]:
    if not isinstance(result, str):
        return (np.nan, np.nan)
    parts = [p.strip() for p in result.split("-")]
    if len(parts) != 2:
        return (np.nan, np.nan)
    try:
        return (float(parts[0]), float(parts[1]))
    except ValueError:
        return (np.nan, np.nan)


def _safe_parse_odds(odds_str: str) -> dict[str, float]:
    if not isinstance(odds_str, str) or not odds_str.strip():
        return {}
    try:
        obj = ast.literal_eval(odds_str)
        if isinstance(obj, dict):
            out: dict[str, float] = {}
            for k, v in obj.items():
                if v is None:
                    continue
                try:
                    out[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
            return out
    except Exception:
        return {}
    return {}


def _standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (mean, std, (X - mean) / std)


def _standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std


def load_ftball_dataset(
    csv_path: str,
    *,
    target: TargetName = "total_goals",
    val_fraction: float = 0.2,
    seed: int = 42,
    standardize_X: bool = True,
    standardize_y: bool = False,
) -> Dataset:
    df = pd.read_csv(csv_path)

    home_away = df["result"].apply(_parse_result)
    df["home_goals"] = home_away.apply(lambda t: t[0])
    df["away_goals"] = home_away.apply(lambda t: t[1])
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["goal_diff"] = df["home_goals"] - df["away_goals"]

    df = df.replace([np.inf, -np.inf], np.nan)

    odds_dicts = df["odds"].apply(_safe_parse_odds)
    for key in ["1", "X", "2", "1X", "X2", "12"]:
        df[f"odds_{key}"] = odds_dicts.apply(lambda d: d.get(key, np.nan))

    if "start_date" in df.columns:
        dt = pd.to_datetime(df["start_date"], errors="coerce")
        df["start_year"] = dt.dt.year
        df["start_month"] = dt.dt.month
        df["start_day"] = dt.dt.day

    y = df[target].astype(float)

    feature_cols_numeric = [
        "odds_1",
        "odds_X",
        "odds_2",
    ]

    present_numeric = [c for c in feature_cols_numeric if c in df.columns]

    cat_cols = [c for c in ["prediction", "market"] if c in df.columns]

    X_num = df[present_numeric].copy()
    for c in X_num.columns:
        if X_num[c].dtype == bool:
            X_num[c] = X_num[c].astype(int)

    X_cat = pd.get_dummies(df[cat_cols].astype("string"), dummy_na=True) if cat_cols else pd.DataFrame(index=df.index)

    X_df = pd.concat([X_num, X_cat], axis=1)

    valid_mask = y.notna()
    X_df = X_df.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()

    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    med = X_df.median(numeric_only=True)
    X_df = X_df.fillna(med)
    X_df = X_df.fillna(0.0)

    X = X_df.to_numpy(dtype=np.float32)
    feature_names = list(X_df.columns)

    y_arr = y.to_numpy(dtype=np.float32).reshape(-1, 1)
    y_arr = np.nan_to_num(y_arr, nan=0.0, posinf=0.0, neginf=0.0)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)

    n_val = max(1, int(len(X) * val_fraction))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    X_train = X[train_idx]
    y_train = y_arr[train_idx]
    X_val = X[val_idx]
    y_val = y_arr[val_idx]

    if standardize_X:
        n_num = len(present_numeric)
        if n_num > 0:
            x_mean, x_std, X_train_num = _standardize_fit(X_train[:, :n_num])
            X_train = X_train.copy()
            X_train[:, :n_num] = X_train_num
            X_val = X_val.copy()
            X_val[:, :n_num] = _standardize_apply(X_val[:, :n_num], x_mean, x_std)

    y_mean: float | None = None
    y_std: float | None = None
    if standardize_y:
        y_mean = float(y_train.mean())
        y_std_ = float(y_train.std())
        y_std = 1.0 if y_std_ == 0 else y_std_
        y_train = (y_train - y_mean) / y_std
        y_val = (y_val - y_mean) / y_std

    return Dataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
        target_name=target,
        y_mean=y_mean,
        y_std=y_std,
    )
