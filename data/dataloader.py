"""
数据加载模块 - STAR-Forecast
提供时间序列数据加载器
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict, Any
import pandas as pd
from sklearn.preprocessing import StandardScaler


class TimeSeriesDataset(Dataset):
    """时间序列数据集"""

    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int,
                 label_len: int = 0, stride: int = 1):
        """
        初始化时间序列数据集

        参数:
            data: 时间序列数据，形状为 (seq_length, feature_dim)
            seq_len: 输入序列长度
            pred_len: 预测序列长度
            label_len: 标签序列长度（用于decoder）
            stride: 滑动窗口步长
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.stride = stride
        self.total_len = seq_len + pred_len

        # 计算样本数量
        self.n_samples = (len(data) - self.total_len) // stride + 1

    def __len__(self) -> int:
        """返回数据集大小"""
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取单个样本"""
        start_idx = idx * self.stride
        end_idx = start_idx + self.total_len

        # 获取序列
        sequence = self.data[start_idx:end_idx]

        # 分割输入和输出
        seq_x = sequence[:self.seq_len]  # 输入序列
        seq_y = sequence[self.seq_len - self.label_len:self.total_len]  # 输出序列

        # 转换为张量
        seq_x = torch.FloatTensor(seq_x)
        seq_y = torch.FloatTensor(seq_y)

        return seq_x, seq_y


class TimeSeriesDataLoader:
    """时间序列数据加载器"""

    def __init__(self, data_path: str, seq_len: int = 96, pred_len: int = 24,
                 label_len: int = 48, batch_size: int = 32, scale: bool = True,
                 features: str = 'M', target: str = 'OT', timeenc: int = 0,
                 freq: str = 'h', train_split: float = 0.7, val_split: float = 0.2,
                 shuffle: bool = True, stride: int = 1):
        """
        初始化数据加载器

        参数:
            data_path: 数据文件路径
            seq_len: 输入序列长度
            pred_len: 预测序列长度
            label_len: 标签序列长度
            batch_size: 批次大小
            scale: 是否标准化
            features: 特征类型 ('M': 多变量, 'S': 单变量, 'MS': 多对单)
            target: 目标列名
            timeenc: 时间编码方式
            freq: 数据频率
            train_split: 训练集比例
            val_split: 验证集比例
            shuffle: 是否打乱数据
            stride: 滑动窗口步长
        """
        self.data_path = data_path
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.batch_size = batch_size
        self.scale = scale
        self.features = features
        self.target = target
        self.timeenc = timeenc
        self.freq = freq
        self.train_split = train_split
        self.val_split = val_split
        self.shuffle = shuffle
        self.stride = stride

        # 加载数据
        self.raw_data = self._load_data()

        # 处理数据
        self.processed_data = self._process_data()

        # 创建数据集
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # 创建数据加载器
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # 标准化器
        self.scaler = StandardScaler() if scale else None

        # 初始化
        self._prepare_datasets()

    def _load_data(self) -> pd.DataFrame:
        """加载数据"""
        if self.data_path.endswith('.csv'):
            df = pd.read_csv(self.data_path)
        elif self.data_path.endswith('.pkl'):
            df = pd.read_pickle(self.data_path)
        else:
            raise ValueError(f"不支持的数据格式: {self.data_path}")

        return df

    def _process_data(self) -> np.ndarray:
        """处理数据"""
        df = self.raw_data

        # 选择特征
        if self.features == 'M' or self.features == 'MS':
            # 多变量预测
            data_cols = [col for col in df.columns if col != 'date']
            data = df[data_cols].values
        elif self.features == 'S':
            # 单变量预测
            if self.target not in df.columns:
                raise ValueError(f"目标列 {self.target} 不存在于数据中")
            data = df[[self.target]].values
        else:
            raise ValueError(f"不支持的特征类型: {self.features}")

        return data.astype(np.float32)

    def _prepare_datasets(self):
        """准备数据集"""
        data = self.processed_data

        # 标准化
        if self.scale:
            train_size = int(len(data) * self.train_split)
            train_data = data[:train_size]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        # 划分数据集
        n = len(data)
        train_end = int(n * self.train_split)
        val_end = train_end + int(n * self.val_split)

        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        # 创建数据集
        self.train_dataset = TimeSeriesDataset(
            train_data, self.seq_len, self.pred_len, self.label_len, self.stride
        )
        self.val_dataset = TimeSeriesDataset(
            val_data, self.seq_len, self.pred_len, self.label_len, self.stride
        )
        self.test_dataset = TimeSeriesDataset(
            test_data, self.seq_len, self.pred_len, self.label_len, self.stride
        )

        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=0,  # Windows下设置为0避免问题
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

    def get_train_loader(self) -> DataLoader:
        """获取训练数据加载器"""
        return self.train_loader

    def get_val_loader(self) -> DataLoader:
        """获取验证数据加载器"""
        return self.val_loader

    def get_test_loader(self) -> DataLoader:
        """获取测试数据加载器"""
        return self.test_loader

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """反标准化"""
        if self.scaler and self.scale:
            return self.scaler.inverse_transform(data)
        return data

    def get_data_info(self) -> Dict[str, Any]:
        """获取数据信息"""
        return {
            'original_shape': self.raw_data.shape,
            'processed_shape': self.processed_data.shape,
            'n_features': self.processed_data.shape[1],
            'train_samples': len(self.train_dataset) if self.train_dataset else 0,
            'val_samples': len(self.val_dataset) if self.val_dataset else 0,
            'test_samples': len(self.test_dataset) if self.test_dataset else 0,
            'feature_names': list(self.raw_data.columns),
            'target_column': self.target
        }

    def print_info(self):
        """打印数据信息"""
        info = self.get_data_info()

        print("📊 数据信息:")
        print(f"   原始数据形状: {info['original_shape']}")
        print(f"   处理数据形状: {info['processed_shape']}")
        print(f"   特征数量: {info['n_features']}")
        print(f"   训练样本数: {info['train_samples']}")
        print(f"   验证样本数: {info['val_samples']}")
        print(f"   测试样本数: {info['test_samples']}")
        print(f"   特征列: {info['feature_names']}")
        print(f"   目标列: {info['target_column']}")


class BatchDataLoader:
    """批量数据加载器（用于已经分割好的数据）"""

    def __init__(self, train_data: np.ndarray, val_data: np.ndarray,
                 test_data: np.ndarray, batch_size: int = 32,
                 seq_len: int = 96, pred_len: int = 24, label_len: int = 48):
        """
        初始化批量数据加载器

        参数:
            train_data: 训练数据
            val_data: 验证数据
            test_data: 测试数据
            batch_size: 批次大小
            seq_len: 输入序列长度
            pred_len: 预测序列长度
            label_len: 标签序列长度
        """
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len

        # 创建数据集
        self.train_dataset = TimeSeriesDataset(
            train_data, seq_len, pred_len, label_len
        )
        self.val_dataset = TimeSeriesDataset(
            val_data, seq_len, pred_len, label_len
        )
        self.test_dataset = TimeSeriesDataset(
            test_data, seq_len, pred_len, label_len
        )

        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

    def get_train_loader(self) -> DataLoader:
        """获取训练数据加载器"""
        return self.train_loader

    def get_val_loader(self) -> DataLoader:
        """获取验证数据加载器"""
        return self.val_loader

    def get_test_loader(self) -> DataLoader:
        """获取测试数据加载器"""
        return self.test_loader