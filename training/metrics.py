"""
metrics.py - 真实的时间序列评估指标计算
包含所有论文所需的指标，完全真实实现
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from typing import Dict, List, Tuple, Union
import warnings

warnings.filterwarnings('ignore')


class TimeSeriesMetrics:
    """时间序列预测评估指标"""

    def __init__(self, scaler=None):
        """
        初始化评估器

        Args:
            scaler: 标准化器，用于反标准化
        """
        self.scaler = scaler
        self.metrics_history = []

    def calculate_all_metrics(self,
                              predictions: np.ndarray,
                              targets: np.ndarray,
                              unscaled: bool = False) -> Dict[str, float]:
        """
        计算所有评估指标

        Args:
            predictions: 预测值
            targets: 真实值
            unscaled: 如果为True，则假设输入已反标准化

        Returns:
            指标字典
        """
        # 转换为numpy数组
        preds = np.array(predictions).flatten()
        trues = np.array(targets).flatten()

        # 检查数据有效性
        if len(preds) != len(trues):
            raise ValueError(f"预测和真实值长度不一致: {len(preds)} vs {len(trues)}")

        # 反标准化（如果需要）
        if not unscaled and self.scaler is not None:
            preds = self._inverse_transform(preds)
            trues = self._inverse_transform(trues)

        # 计算所有指标
        metrics = {
            # 基础指标
            'mse': self._mse(preds, trues),
            'mae': self._mae(preds, trues),
            'rmse': self._rmse(preds, trues),

            # 百分比误差
            'mape': self._mape(preds, trues),
            'smape': self._smape(preds, trues),
            'mase': self._mase(preds, trues),

            # 相关性指标
            'r2': self._r2_score(preds, trues),
            'correlation': self._correlation(preds, trues),

            # 统计指标
            'error_mean': np.mean(preds - trues),
            'error_std': np.std(preds - trues),

            # 时间序列特定指标
            'direction_accuracy': self._direction_accuracy(preds, trues),
            'peak_accuracy': self._peak_accuracy(preds, trues),
            'volatility_ratio': self._volatility_ratio(preds, trues)
        }

        # 计算分布统计
        errors = preds - trues
        metrics.update(self._error_distribution(errors))

        # 记录历史
        self.metrics_history.append(metrics)

        return metrics

    def _mse(self, preds: np.ndarray, trues: np.ndarray) -> float:
        """均方误差"""
        return float(np.mean((preds - trues) ** 2))

    def _mae(self, preds: np.ndarray, trues: np.ndarray) -> float:
        """平均绝对误差"""
        return float(np.mean(np.abs(preds - trues)))

    def _rmse(self, preds: np.ndarray, trues: np.ndarray) -> float:
        """均方根误差"""
        return float(np.sqrt(self._mse(preds, trues)))

    def _mape(self, preds: np.ndarray, trues: np.ndarray) -> float:
        """平均绝对百分比误差"""
        epsilon = 1e-8
        return float(np.mean(np.abs((trues - preds) / (trues + epsilon))) * 100)

    def _smape(self, preds: np.ndarray, trues: np.ndarray) -> float:
        """对称平均绝对百分比误差"""
        epsilon = 1e-8
        return float(100 * np.mean(2 * np.abs(preds - trues) /
                                   (np.abs(preds) + np.abs(trues) + epsilon)))

    def _mase(self, preds: np.ndarray, trues: np.ndarray,
              seasonality: int = 1) -> float:
        """平均绝对标度误差"""
        naive_forecast = trues[:-seasonality]
        actual = trues[seasonality:]
        naive_error = np.mean(np.abs(actual - naive_forecast))

        if naive_error < 1e-8:
            return float('inf')

        return float(np.mean(np.abs(preds - trues)) / naive_error)

    def _r2_score(self, preds: np.ndarray, trues: np.ndarray) -> float:
        """R²分数"""
        ss_res = np.sum((trues - preds) ** 2)
        ss_tot = np.sum((trues - np.mean(trues)) ** 2)

        if ss_tot < 1e-8:
            return 0.0

        return float(1 - ss_res / ss_tot)

    def _correlation(self, preds: np.ndarray, trues: np.ndarray) -> float:
        """相关系数"""
        if len(preds) < 2:
            return 0.0

        corr_matrix = np.corrcoef(preds, trues)
        if corr_matrix.shape == (2, 2):
            return float(corr_matrix[0, 1])
        return 0.0

    def _direction_accuracy(self, preds: np.ndarray, trues: np.ndarray) -> float:
        """方向准确性"""
        if len(preds) < 2:
            return 0.5

        pred_dir = np.sign(np.diff(preds))
        true_dir = np.sign(np.diff(trues))

        # 对齐长度
        min_len = min(len(pred_dir), len(true_dir))
        pred_dir = pred_dir[:min_len]
        true_dir = true_dir[:min_len]

        # 计算准确率
        matches = np.sum(pred_dir == true_dir)
        return float(matches / min_len)

    def _peak_accuracy(self, preds: np.ndarray, trues: np.ndarray,
                       threshold: float = 0.8) -> float:
        """峰值预测准确率"""
        if len(trues) == 0:
            return 0.0

        # 识别真实峰值（前threshold%的值）
        peak_threshold = np.percentile(trues, threshold * 100)
        true_peaks = trues > peak_threshold

        # 识别预测峰值
        pred_peaks = preds > peak_threshold

        # 确保长度一致
        min_len = min(len(true_peaks), len(pred_peaks))
        true_peaks = true_peaks[:min_len]
        pred_peaks = pred_peaks[:min_len]

        # 计算峰值准确率
        if np.sum(true_peaks) > 0:
            return float(np.mean(pred_peaks[true_peaks]))
        return 0.0

    def _volatility_ratio(self, preds: np.ndarray, trues: np.ndarray) -> float:
        """波动率比率"""
        if len(preds) < 2 or len(trues) < 2:
            return 1.0

        pred_vol = np.std(np.diff(preds))
        true_vol = np.std(np.diff(trues))

        if true_vol < 1e-8:
            return 1.0

        return float(pred_vol / true_vol)

    def _error_distribution(self, errors: np.ndarray) -> Dict[str, float]:
        """误差分布统计"""
        if len(errors) == 0:
            return {
                'error_skew': 0.0,
                'error_kurtosis': 0.0,
                'normality_p': 1.0
            }

        try:
            skew = float(stats.skew(errors))
            kurtosis = float(stats.kurtosis(errors))

            # 正态性检验
            if len(errors) > 7:
                _, normality_p = stats.normaltest(errors)
                normality_p = float(normality_p)
            else:
                normality_p = 0.0
        except:
            skew, kurtosis, normality_p = 0.0, 0.0, 0.0

        return {
            'error_skew': skew,
            'error_kurtosis': kurtosis,
            'normality_p': normality_p
        }

    def _inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """反标准化"""
        if self.scaler is None:
            return data

        # 假设scaler是StandardScaler
        if hasattr(self.scaler, 'inverse_transform'):
            # 重塑为2D数组
            data_2d = data.reshape(-1, 1)
            if data_2d.shape[1] != self.scaler.n_features_in_:
                # 如果维度不匹配，填充其他特征
                full_data = np.zeros((len(data_2d), self.scaler.n_features_in_))
                full_data[:, -1] = data_2d.flatten()  # OT特征在最后一列
                inverted = self.scaler.inverse_transform(full_data)[:, -1]
            else:
                inverted = self.scaler.inverse_transform(data_2d).flatten()
            return inverted
        return data

    def calculate_horizon_metrics(self,
                                  predictions: np.ndarray,
                                  targets: np.ndarray,
                                  pred_len: int) -> Dict[str, List[float]]:
        """
        按预测时间步计算指标

        Args:
            predictions: [n_samples, pred_len]
            targets: [n_samples, pred_len]
            pred_len: 预测长度

        Returns:
            各时间步的指标
        """
        preds = np.array(predictions)
        trues = np.array(targets)

        # 确保形状正确
        if preds.ndim == 1:
            preds = preds.reshape(-1, pred_len)
        if trues.ndim == 1:
            trues = trues.reshape(-1, pred_len)

        horizon_metrics = {
            'mse': [],
            'mae': [],
            'rmse': [],
            'mape': []
        }

        for h in range(pred_len):
            pred_h = preds[:, h]
            true_h = trues[:, h]

            horizon_metrics['mse'].append(self._mse(pred_h, true_h))
            horizon_metrics['mae'].append(self._mae(pred_h, true_h))
            horizon_metrics['rmse'].append(self._rmse(pred_h, true_h))
            horizon_metrics['mape'].append(self._mape(pred_h, true_h))

        return horizon_metrics

    def statistical_test(self,
                         preds_a: np.ndarray,
                         preds_b: np.ndarray,
                         targets: np.ndarray,
                         test_type: str = 'wilcoxon',
                         alpha: float = 0.05) -> Dict[str, float]:
        """
        统计显著性检验

        Args:
            preds_a: 方法A的预测
            preds_b: 方法B的预测
            targets: 真实值
            test_type: 检验类型 ('wilcoxon' 或 't_test')
            alpha: 显著性水平

        Returns:
            检验结果
        """
        # 计算绝对误差
        errors_a = np.abs(preds_a.flatten() - targets.flatten())
        errors_b = np.abs(preds_b.flatten() - targets.flatten())

        if len(errors_a) != len(errors_b):
            raise ValueError("误差长度必须相同")

        try:
            if test_type == 'wilcoxon':
                # Wilcoxon符号秩检验
                stat, p_value = stats.wilcoxon(errors_a, errors_b)
                test_name = 'Wilcoxon'
            elif test_type == 't_test':
                # 配对t检验
                stat, p_value = stats.ttest_rel(errors_a, errors_b)
                test_name = 'Paired t-test'
            else:
                raise ValueError(f"不支持的检验类型: {test_type}")

            # 计算效应量 (Cohen's d)
            mean_diff = np.mean(errors_a - errors_b)
            std_diff = np.std(errors_a - errors_b)
            if std_diff > 1e-8:
                effect_size = mean_diff / std_diff
            else:
                effect_size = 0.0

            return {
                'test': test_name,
                'statistic': float(stat),
                'p_value': float(p_value),
                'significant': p_value < alpha,
                'effect_size': float(effect_size),
                'mean_diff': float(mean_diff),
                'std_diff': float(std_diff)
            }

        except Exception as e:
            return {
                'test': test_type,
                'error': str(e),
                'statistic': 0.0,
                'p_value': 1.0,
                'significant': False,
                'effect_size': 0.0
            }


def calculate_model_complexity(model: torch.nn.Module) -> Dict[str, float]:
    """
    计算模型复杂度指标

    Args:
        model: PyTorch模型

    Returns:
        复杂度指标
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 计算FLOPs估计（简化版本）
    # 这里使用简单的启发式估计，实际应该使用更精确的方法
    flops_estimate = total_params * 2  # 粗略估计

    return {
        'total_params': float(total_params),
        'trainable_params': float(trainable_params),
        'trainable_ratio': float(trainable_params / total_params) if total_params > 0 else 0.0,
        'flops_estimate': float(flops_estimate)
    }


def compute_baseline_metrics(data_path: str) -> Dict[str, float]:
    """
    计算基线方法的指标（使用简单方法作为基准）

    Args:
        data_path: 数据路径

    Returns:
        基线指标
    """
    import pandas as pd
    import numpy as np

    # 加载数据
    df = pd.read_csv(data_path)
    data = df.iloc[:, 1:].values  # 去掉日期列

    # 简单的基准方法：历史平均值
    target_col = data[:, -1]  # OT列

    # 使用前70%训练，后30%测试
    split_idx = int(len(target_col) * 0.7)
    train_data = target_col[:split_idx]
    test_data = target_col[split_idx:]

    # 历史平均值预测
    historical_mean = np.mean(train_data)
    baseline_pred = np.full_like(test_data, historical_mean)

    # 计算指标
    metrics_calculator = TimeSeriesMetrics()
    metrics = metrics_calculator.calculate_all_metrics(baseline_pred, test_data, unscaled=False)

    return {
        'method': 'Historical Mean',
        **metrics
    }


if __name__ == "__main__":
    # 测试代码
    print("测试评估指标计算...")

    # 生成测试数据
    np.random.seed(42)
    n_samples = 100
    preds = np.random.randn(n_samples) * 0.1 + 1.0
    trues = np.random.randn(n_samples) * 0.1 + 1.0 + np.random.randn(n_samples) * 0.05

    # 计算指标
    calculator = TimeSeriesMetrics()
    metrics = calculator.calculate_all_metrics(preds, trues, unscaled=True)

    print("测试结果:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")

    # 测试按时间步计算
    preds_2d = preds.reshape(-1, 24)
    trues_2d = trues.reshape(-1, 24)
    horizon_metrics = calculator.calculate_horizon_metrics(preds_2d, trues_2d, 24)

    print("\n时间步指标（前5步）:")
    for i in range(5):
        print(f"  步长 {i}: MSE={horizon_metrics['mse'][i]:.6f}, MAE={horizon_metrics['mae'][i]:.6f}")