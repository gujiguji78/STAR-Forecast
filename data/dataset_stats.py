"""
数据集统计模块 - STAR-Forecast
提供数据集统计功能
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns


def compute_dataset_stats(data: pd.DataFrame) -> Dict[str, Any]:
    """
    计算数据集统计信息

    参数:
        data: 输入数据框

    返回:
        统计信息字典
    """
    stats = {}

    # 基本统计
    stats['shape'] = data.shape
    stats['n_samples'] = len(data)
    stats['n_features'] = len(data.columns)
    stats['feature_names'] = list(data.columns)

    # 数据类型
    stats['dtypes'] = data.dtypes.to_dict()

    # 缺失值统计
    stats['missing_values'] = data.isnull().sum().to_dict()
    stats['missing_percentage'] = (data.isnull().sum() / len(data) * 100).to_dict()

    # 数值特征统计
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        numeric_stats = data[numeric_cols].describe().to_dict()
        stats['numeric_stats'] = numeric_stats

        # 相关性矩阵
        correlation_matrix = data[numeric_cols].corr().to_dict()
        stats['correlation_matrix'] = correlation_matrix

    # 时间特征处理（如果存在date列）
    if 'date' in data.columns:
        try:
            data['date'] = pd.to_datetime(data['date'])
            stats['time_range'] = {
                'start': data['date'].min(),
                'end': data['date'].max(),
                'duration': data['date'].max() - data['date'].min()
            }
            stats['time_freq'] = pd.infer_freq(data['date'])
        except:
            pass

    return stats


def print_dataset_stats(stats: Dict[str, Any]):
    """
    打印数据集统计信息

    参数:
        stats: 统计信息字典
    """
    print("📊 数据集统计信息:")
    print(f"   形状: {stats['shape']} (样本数, 特征数)")
    print(f"   特征数: {stats['n_features']}")
    print(f"   样本数: {stats['n_samples']}")

    print("\n📋 特征列表:")
    for i, feature in enumerate(stats['feature_names']):
        dtype = stats['dtypes'].get(feature, 'unknown')
        print(f"   {i + 1}. {feature} ({dtype})")

    print("\n❓ 缺失值统计:")
    missing_total = 0
    for feature, count in stats['missing_values'].items():
        if count > 0:
            percentage = stats['missing_percentage'][feature]
            print(f"   {feature}: {count} ({percentage:.2f}%)")
            missing_total += count

    if missing_total == 0:
        print("   无缺失值")

    # 数值特征统计
    if 'numeric_stats' in stats:
        print("\n📈 数值特征统计:")
        numeric_cols = list(stats['numeric_stats'].keys())
        for col in numeric_cols[:5]:  # 只显示前5个特征
            col_stats = stats['numeric_stats'][col]
            print(f"   {col}:")
            print(f"      均值: {col_stats['mean']:.4f}")
            print(f"      标准差: {col_stats['std']:.4f}")
            print(f"      最小值: {col_stats['min']:.4f}")
            print(f"      25%分位数: {col_stats['25%']:.4f}")
            print(f"      中位数: {col_stats['50%']:.4f}")
            print(f"      75%分位数: {col_stats['75%']:.4f}")
            print(f"      最大值: {col_stats['max']:.4f}")

        if len(numeric_cols) > 5:
            print(f"   ... 还有 {len(numeric_cols) - 5} 个数值特征")

    # 时间范围
    if 'time_range' in stats:
        print("\n⏰ 时间范围:")
        time_range = stats['time_range']
        print(f"   开始时间: {time_range['start']}")
        print(f"   结束时间: {time_range['end']}")
        print(f"   持续时间: {time_range['duration']}")

        if 'time_freq' in stats and stats['time_freq']:
            print(f"   时间频率: {stats['time_freq']}")


def plot_dataset_features(data: pd.DataFrame, save_path: str = None):
    """
    绘制数据集特征图

    参数:
        data: 输入数据
        save_path: 保存路径，如果为None则不保存
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    n_numeric = len(numeric_cols)

    if n_numeric == 0:
        print("⚠️  没有数值特征可绘制")
        return

    # 计算子图布局
    n_cols = min(3, n_numeric)
    n_rows = (n_numeric + n_cols - 1) // n_cols

    # 创建图形
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_numeric > 1 else [axes]

    # 绘制每个数值特征的分布
    for i, col in enumerate(numeric_cols):
        if i >= len(axes):
            break

        ax = axes[i]
        ax.hist(data[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title(f'{col} 分布')
        ax.set_xlabel('值')
        ax.set_ylabel('频数')
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for i in range(len(numeric_cols), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 特征图保存到: {save_path}")

    plt.show()


def analyze_time_series(data: pd.DataFrame, date_col: str = 'date', value_col: str = None):
    """
    分析时间序列数据

    参数:
        data: 输入数据
        date_col: 日期列名
        value_col: 值列名（如果为None则使用第一个数值列）

    返回:
        时间序列分析结果
    """
    if date_col not in data.columns:
        raise ValueError(f"日期列 {date_col} 不存在")

    # 转换日期列
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col)

    # 设置日期索引
    data_indexed = data.set_index(date_col)

    # 选择值列
    if value_col is None:
        numeric_cols = data_indexed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("没有数值列可分析")
        value_col = numeric_cols[0]

    if value_col not in data_indexed.columns:
        raise ValueError(f"值列 {value_col} 不存在")

    # 提取时间序列
    time_series = data_indexed[value_col]

    # 计算统计量
    stats = {
        'mean': time_series.mean(),
        'std': time_series.std(),
        'min': time_series.min(),
        'max': time_series.max(),
        'range': time_series.max() - time_series.min(),
        'median': time_series.median(),
        'skewness': time_series.skew(),
        'kurtosis': time_series.kurtosis(),
        'n_missing': time_series.isnull().sum(),
        'missing_percentage': time_series.isnull().sum() / len(time_series) * 100,
        'autocorr_lag1': time_series.autocorr(lag=1) if len(time_series) > 1 else None,
        'autocorr_lag24': time_series.autocorr(lag=24) if len(time_series) > 24 else None,
        'trend': 'increasing' if time_series.iloc[-1] > time_series.iloc[0] else 'decreasing'
    }

    return stats