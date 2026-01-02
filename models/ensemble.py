"""
模型集成模块 - STAR-Forecast
提供模型集成功能
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional
from collections import OrderedDict


class EnsembleModel(nn.Module):
    """模型集成器"""

    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super(EnsembleModel, self).__init__()

        self.models = nn.ModuleList(models)
        self.n_models = len(models)

        # 设置权重
        if weights is None:
            self.weights = nn.Parameter(torch.ones(self.n_models) / self.n_models)
        else:
            if len(weights) != self.n_models:
                raise ValueError(f"权重数量({len(weights)})与模型数量({self.n_models})不匹配")
            self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))

        # 归一化权重
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x_enc: torch.Tensor, x_dec: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        predictions = []

        # 获取每个模型的预测
        for model in self.models:
            pred = model(x_enc, x_dec)
            predictions.append(pred)

        # 归一化权重
        normalized_weights = self.softmax(self.weights)

        # 加权平均
        ensemble_pred = torch.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            ensemble_pred += normalized_weights[i] * pred

        return ensemble_pred

    def get_model_predictions(self, x_enc: torch.Tensor, x_dec: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """获取每个模型的预测"""
        predictions = []
        for model in self.models:
            pred = model(x_enc, x_dec)
            predictions.append(pred)
        return predictions


class WeightedEnsemble:
    """加权集成（非神经网络版本）"""

    def __init__(self, models: List[Any], weights: Optional[List[float]] = None):
        self.models = models
        self.n_models = len(models)

        if weights is None:
            self.weights = [1.0 / self.n_models] * self.n_models
        else:
            if len(weights) != self.n_models:
                raise ValueError(f"权重数量({len(weights)})与模型数量({self.n_models})不匹配")
            self.weights = weights

        # 归一化权重
        weight_sum = sum(self.weights)
        self.weights = [w / weight_sum for w in self.weights]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """预测"""
        predictions = []

        for model in self.models:
            if hasattr(model, 'predict'):
                pred = model.predict(x)
            elif hasattr(model, '__call__'):
                pred = model(x)
            else:
                raise ValueError(f"模型不支持预测: {type(model)}")

            predictions.append(pred)

        # 加权平均
        ensemble_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            ensemble_pred += self.weights[i] * pred

        return ensemble_pred


class StackingEnsemble:
    """堆叠集成"""

    def __init__(self, base_models: List[Any], meta_model: Any):
        self.base_models = base_models
        self.meta_model = meta_model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """训练集成模型"""
        # 训练基模型
        for model in self.base_models:
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)

        # 生成基模型预测作为元特征
        meta_features = self._generate_meta_features(X_train)

        # 训练元模型
        if hasattr(self.meta_model, 'fit'):
            self.meta_model.fit(meta_features, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        # 生成基模型预测
        meta_features = self._generate_meta_features(X)

        # 元模型预测
        if hasattr(self.meta_model, 'predict'):
            return self.meta_model.predict(meta_features)
        elif hasattr(self.meta_model, '__call__'):
            return self.meta_model(meta_features)
        else:
            raise ValueError(f"元模型不支持预测: {type(self.meta_model)}")

    def _generate_meta_features(self, X: np.ndarray) -> np.ndarray:
        """生成元特征"""
        predictions = []

        for model in self.base_models:
            if hasattr(model, 'predict'):
                pred = model.predict(X)
            elif hasattr(model, '__call__'):
                pred = model(X)
            else:
                raise ValueError(f"模型不支持预测: {type(model)}")

            predictions.append(pred)

        # 堆叠预测结果
        return np.column_stack(predictions)