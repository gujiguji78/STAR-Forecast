"""
数据预处理模块 - STAR-Forecast
提供数据预处理功能
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DataProcessorConfig:
    """数据处理器配置"""
    name: str = "ETTh1"  # 添加 name 字段
    seq_len: int = 96
    pred_len: int = 24
    label_len: int = 48
    features: str = 'M'
    target: str = 'OT'
    scale: bool = True
    timeenc: int = 0
    freq: str = 'h'
    train_split: float = 0.7
    val_split: float = 0.2
    test_split: float = 0.1
    scaler_type: str = 'standard'  # 'standard' 或 'minmax'
    # 添加可能出现的其他字段
    dataset: Optional[str] = None  # 添加 dataset 字段以兼容旧代码


class DataProcessor:
    """数据处理器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化数据处理器

        参数:
            config: 配置字典
        """
        if config is None:
            config = {}

        self.config = DataProcessorConfig(**config)
        self.scaler = None
        self.feature_cols = None
        self.target_idx = None

        # 初始化标准化器
        if self.config.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()

    def process(self, data: pd.DataFrame) -> np.ndarray:
        """
        处理数据

        参数:
            data: 输入数据框

        返回:
            处理后的numpy数组
        """
        # 移除日期列（如果存在）
        if 'date' in data.columns:
            data = data.drop('date', axis=1)

        # 选择特征
        if self.config.features == 'M' or self.config.features == 'MS':
            # 多变量预测
            self.feature_cols = list(data.columns)
            processed_data = data.values
        elif self.config.features == 'S':
            # 单变量预测
            if self.config.target not in data.columns:
                raise ValueError(f"目标列 {self.config.target} 不存在于数据中")
            self.feature_cols = [self.config.target]
            processed_data = data[[self.config.target]].values
        else:
            raise ValueError(f"不支持的特征类型: {self.config.features}")

        # 设置目标索引
        if self.config.target in data.columns:
            self.target_idx = data.columns.get_loc(self.config.target)
        else:
            self.target_idx = -1  # 最后一列

        # 转换为浮点数
        processed_data = processed_data.astype(np.float32)

        return processed_data

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """拟合并转换数据"""
        processed_data = self.process(data)

        if self.config.scale and self.scaler is not None:
            processed_data = self.scaler.fit_transform(processed_data)

        return processed_data

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """转换数据（使用已拟合的标准化器）"""
        processed_data = self.process(data)

        if self.config.scale and self.scaler is not None:
            processed_data = self.scaler.transform(processed_data)

        return processed_data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """反标准化"""
        if self.config.scale and self.scaler is not None:
            return self.scaler.inverse_transform(data)
        return data

    def split_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        分割数据

        参数:
            data: 处理后的数据

        返回:
            (训练集, 验证集, 测试集)
        """
        n = len(data)
        train_end = int(n * self.config.train_split)
        val_end = train_end + int(n * self.config.val_split)

        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        return train_data, val_data, test_data

    def create_sequences(self, data: np.ndarray, seq_len: int, pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建序列数据

        参数:
            data: 输入数据
            seq_len: 序列长度
            pred_len: 预测长度

        返回:
            (输入序列, 输出序列)
        """
        X, Y = [], []
        n = len(data)

        for i in range(n - seq_len - pred_len + 1):
            X.append(data[i:i + seq_len])
            Y.append(data[i + seq_len:i + seq_len + pred_len])

        return np.array(X), np.array(Y)

    def get_feature_info(self) -> Dict[str, Any]:
        """获取特征信息"""
        return {
            'feature_cols': self.feature_cols,
            'target_idx': self.target_idx,
            'n_features': len(self.feature_cols) if self.feature_cols else 0,
            'target_column': self.config.target
        }


def normalize_data(data: np.ndarray, scaler_type: str = 'standard') -> Tuple[np.ndarray, Any]:
    """
    标准化数据

    参数:
        data: 输入数据
        scaler_type: 标准化器类型 ('standard' 或 'minmax')

    返回:
        (标准化后的数据, 标准化器)
    """
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    normalized_data = scaler.fit_transform(data)
    return normalized_data, scaler


def denormalize_data(data: np.ndarray, scaler: Any) -> np.ndarray:
    """
    反标准化数据

    参数:
        data: 标准化后的数据
        scaler: 标准化器

    返回:
        原始尺度的数据
    """
    return scaler.inverse_transform(data)