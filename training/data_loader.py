# training/data_loader.py
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Utils
# -------------------------
def _as_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _glob_csvs(pattern_or_list) -> List[str]:
    """
    支持：
      - "data/train/*.csv"
      - ["a.csv","b.csv"]
      - 单个文件路径
    """
    if pattern_or_list is None:
        return []
    if isinstance(pattern_or_list, (list, tuple)):
        files = []
        for p in pattern_or_list:
            files += glob.glob(str(p))
        return sorted(list(set(files)))
    p = str(pattern_or_list)
    if any(ch in p for ch in ["*", "?", "[", "]"]):
        return sorted(glob.glob(p))
    return [p] if os.path.exists(p) else []


def _pick_first_numeric_column(df: pd.DataFrame) -> str:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("CSV 中没有数值列（numeric column）。请检查数据格式。")
    return numeric_cols[0]


def _read_one_csv(
    path: str,
    time_col: Optional[str],
    target_col: Optional[str],
    feature_cols: Optional[List[str]],
    dropna: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回：
      X: (T, C)  float32
      y: (T,)    float32  （目标序列）
    说明：
      - 如果 feature_cols is None：X 只用 target_col（单变量），C=1
      - 如果 feature_cols 给了：X 用 feature_cols（多变量），y 用 target_col
    """
    df = pd.read_csv(path)

    # 丢弃时间列（不作为数值特征）
    if time_col and time_col in df.columns:
        df = df.drop(columns=[time_col])

    if dropna:
        df = df.dropna()

    if target_col is None or target_col not in df.columns:
        # 默认选择第一列数值作为 target
        target_col = _pick_first_numeric_column(df)

    y = df[target_col].astype("float32").to_numpy()

    if feature_cols is None:
        # 单变量：X=target
        X = y.reshape(-1, 1)
    else:
        # 多变量：X=feature_cols, y=target_col
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"CSV 缺少 feature_cols 列：{missing}，文件：{path}")
        X = df[feature_cols].astype("float32").to_numpy()

    return X, y


@dataclass
class Scaler:
    mode: str = "zscore"  # "zscore" or "minmax"
    eps: float = 1e-8

    x_mean: Optional[np.ndarray] = None
    x_std: Optional[np.ndarray] = None
    x_min: Optional[np.ndarray] = None
    x_max: Optional[np.ndarray] = None

    y_mean: Optional[float] = None
    y_std: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.mode == "zscore":
            self.x_mean = X.mean(axis=0)
            self.x_std = X.std(axis=0) + self.eps
            self.y_mean = float(y.mean())
            self.y_std = float(y.std() + self.eps)
        elif self.mode == "minmax":
            self.x_min = X.min(axis=0)
            self.x_max = X.max(axis=0)
            self.y_min = float(y.min())
            self.y_max = float(y.max())
        else:
            raise ValueError(f"normalize 只能是 'zscore' 或 'minmax'，但你给的是 {self.mode}")

    def transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.mode == "zscore":
            Xn = (X - self.x_mean) / self.x_std
            yn = (y - self.y_mean) / self.y_std
            return Xn.astype("float32"), yn.astype("float32")
        else:
            Xn = (X - self.x_min) / (self.x_max - self.x_min + self.eps)
            yn = (y - self.y_min) / (self.y_max - self.y_min + self.eps)
            return Xn.astype("float32"), yn.astype("float32")


class TimeSeriesWindowDataset(Dataset):
    """
    把连续序列切成滑窗：
      输入:  X[t : t+seq_len, :]
      目标:  y[t+seq_len : t+seq_len+horizon]
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int, horizon: int):
        assert len(X) == len(y)
        self.X = torch.from_numpy(X)  # (T, C)
        self.y = torch.from_numpy(y)  # (T,)
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)

        self.max_start = len(self.X) - self.seq_len - self.horizon + 1
        if self.max_start <= 0:
            raise ValueError(
                f"数据太短：T={len(self.X)}，但 seq_len={self.seq_len}, horizon={self.horizon}。"
            )

    def __len__(self):
        return self.max_start

    def __getitem__(self, idx: int):
        x = self.X[idx : idx + self.seq_len]  # (seq_len, C)
        y = self.y[idx + self.seq_len : idx + self.seq_len + self.horizon]  # (horizon,)
        return x, y


def _concat_csvs(
    csvs: List[str],
    time_col: Optional[str],
    target_col: Optional[str],
    feature_cols: Optional[List[str]],
    dropna: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if not csvs:
        raise ValueError("CSV 列表为空。请检查 train_glob/val_glob/test_glob 是否写对、文件是否存在。")

    X_list, y_list = [], []
    for p in csvs:
        X, y = _read_one_csv(p, time_col, target_col, feature_cols, dropna=dropna)
        X_list.append(X)
        y_list.append(y)

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    return X_all, y_all


# -------------------------
# Public APIs
# -------------------------
def create_loaders(
    train_csvs: List[str],
    val_csvs: Optional[List[str]] = None,
    test_csvs: Optional[List[str]] = None,
    value_col: Optional[str] = None,
    seq_len: int = 100,
    horizon: int = 10,
    batch_size: int = 32,
    normalize: str = "zscore",
    time_col: Optional[str] = None,
    feature_cols: Optional[List[str]] = None,
    dropna: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Scaler]:
    """
    兼容/升级版：
      - 支持 train/val/test 三段数据（优先用 val_csvs）
      - 如果 val_csvs 为空：会从 train 末尾切出 10% 做 val（真实数据切分，不是模拟）
      - normalize 只用 train 统计量，应用到 val/test
    """
    val_csvs = val_csvs or []
    test_csvs = test_csvs or []

    # 读 train
    X_train, y_train = _concat_csvs(train_csvs, time_col, value_col, feature_cols, dropna)

    # 如果没有 val_csvs：从 train 末尾切分一段作为 val（真实切分）
    if not val_csvs:
        n = len(X_train)
        val_size = max(int(0.1 * n), seq_len + horizon + 1)
        val_size = min(val_size, n // 3)  # 防止太大
        split = n - val_size
        X_val, y_val = X_train[split:], y_train[split:]
        X_train, y_train = X_train[:split], y_train[:split]
    else:
        X_val, y_val = _concat_csvs(val_csvs, time_col, value_col, feature_cols, dropna)

    # test
    if test_csvs:
        X_test, y_test = _concat_csvs(test_csvs, time_col, value_col, feature_cols, dropna)
    else:
        # 没有 test 就复用 val（不推荐，但保证代码可跑）
        X_test, y_test = X_val.copy(), y_val.copy()

    # 归一化（用 train fit）
    scaler = Scaler(mode=normalize)
    scaler.fit(X_train, y_train)

    X_train, y_train = scaler.transform(X_train, y_train)
    X_val, y_val = scaler.transform(X_val, y_val)
    X_test, y_test = scaler.transform(X_test, y_test)

    # Dataset
    ds_train = TimeSeriesWindowDataset(X_train, y_train, seq_len=seq_len, horizon=horizon)
    ds_val = TimeSeriesWindowDataset(X_val, y_val, seq_len=seq_len, horizon=horizon)
    ds_test = TimeSeriesWindowDataset(X_test, y_test, seq_len=seq_len, horizon=horizon)

    # Loader
    train_loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True
    )
    val_loader = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    test_loader = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )

    return train_loader, val_loader, test_loader, scaler


def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    你 pretrain_tcn.py 里调用的是 build_dataloaders(cfg)，这里保持不变。
    """
    data_cfg = cfg.get("data", {})
    task_cfg = cfg.get("task", {})
    train_cfg = cfg.get("training", {})

    train_csvs = _glob_csvs(data_cfg.get("train_glob"))
    val_csvs = _glob_csvs(data_cfg.get("val_glob"))
    test_csvs = _glob_csvs(data_cfg.get("test_glob"))

    time_col = data_cfg.get("time_col", None)
    target_col = data_cfg.get("target_col", None)

    feature_cols = data_cfg.get("feature_cols", None)
    if feature_cols is not None:
        feature_cols = _as_list(feature_cols)

    normalize = data_cfg.get("normalize", "zscore")
    dropna = bool(data_cfg.get("dropna", True))

    seq_len = int(task_cfg.get("seq_len", 100))
    horizon = int(task_cfg.get("horizon", 10))

    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", True))

    train_loader, val_loader, test_loader, _ = create_loaders(
        train_csvs=train_csvs,
        val_csvs=val_csvs,
        test_csvs=test_csvs,
        value_col=target_col,
        seq_len=seq_len,
        horizon=horizon,
        batch_size=batch_size,
        normalize=normalize,
        time_col=time_col,
        feature_cols=feature_cols,
        dropna=dropna,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
