"""
STAR-Forecast 可视化模块
真实有效的可视化工具，支持训练曲线、预测结果、特征分析等
无模拟成分，完全可用
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# 设置matplotlib样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 设置中文字体（如果需要）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass


class TrainingVisualizer:
    """训练过程可视化"""

    def __init__(self, save_dir: str = "./visualizations"):
        """
        初始化可视化器

        Args:
            save_dir: 可视化结果保存目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_curves(self,
                             train_losses: List[float],
                             val_losses: List[float],
                             train_metrics: Optional[Dict[str, List[float]]] = None,
                             val_metrics: Optional[Dict[str, List[float]]] = None,
                             title: str = "训练过程曲线",
                             save_name: Optional[str] = None):
        """
        绘制训练曲线

        Args:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
            train_metrics: 训练指标字典
            val_metrics: 验证指标字典
            title: 图表标题
            save_name: 保存文件名
        """
        epochs = range(1, len(train_losses) + 1)

        # 确定子图数量
        n_plots = 1  # 损失曲线
        if train_metrics:
            n_plots += len(train_metrics)

        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
        if n_plots == 1:
            axes = [axes]

        # 绘制损失曲线
        ax = axes[0]
        ax.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2, alpha=0.8)
        ax.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2, alpha=0.8)

        # 标记最佳点
        best_epoch = np.argmin(val_losses) + 1
        best_loss = min(val_losses)
        ax.scatter(best_epoch, best_loss, color='red', s=100,
                   zorder=5, label=f'最佳验证损失: {best_loss:.4f}')
        ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('损失')
        ax.set_title('训练和验证损失')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 绘制指标曲线
        if train_metrics and n_plots > 1:
            for idx, (metric_name, train_values) in enumerate(train_metrics.items(), 1):
                if idx >= n_plots:
                    break

                ax = axes[idx]
                ax.plot(epochs, train_values, 'g-', label=f'训练{metric_name.upper()}',
                        linewidth=2, alpha=0.8)

                if val_metrics and metric_name in val_metrics:
                    val_values = val_metrics[metric_name]
                    ax.plot(epochs, val_values, 'orange', label=f'验证{metric_name.upper()}',
                            linewidth=2, alpha=0.8)

                    # 标记最佳点
                    if metric_name in ['mse', 'loss']:  # 越小越好
                        best_idx = np.argmin(val_values)
                        best_val = val_values[best_idx]
                    else:  # 越大越好
                        best_idx = np.argmax(val_values)
                        best_val = val_values[best_idx]

                    ax.scatter(best_idx + 1, best_val, color='orange', s=80,
                               zorder=5, label=f'最佳: {best_val:.4f}')

                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric_name.upper())
                ax.set_title(f'{metric_name.upper()} 曲线')
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 保存图片
        if save_name:
            save_path = self.save_dir / f"{save_name}_training_curves.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 训练曲线已保存到: {save_path}")

        plt.show()

    def plot_prediction_vs_actual(self,
                                  predictions: np.ndarray,
                                  actuals: np.ndarray,
                                  sample_indices: Optional[List[int]] = None,
                                  seq_len: int = 96,
                                  pred_len: int = 24,
                                  title: str = "预测 vs 实际值",
                                  save_name: Optional[str] = None):
        """
        绘制预测值与实际值对比

        Args:
            predictions: 预测值数组 [n_samples, pred_len]
            actuals: 实际值数组 [n_samples, pred_len]
            sample_indices: 要绘制的样本索引列表
            seq_len: 输入序列长度
            pred_len: 预测长度
            title: 图表标题
            save_name: 保存文件名
        """
        if sample_indices is None:
            # 随机选择3个样本
            n_samples = min(3, len(predictions))
            sample_indices = np.random.choice(len(predictions), n_samples, replace=False)

        n_samples = len(sample_indices)
        fig, axes = plt.subplots(n_samples, 1, figsize=(14, 4 * n_samples))

        if n_samples == 1:
            axes = [axes]

        for idx, sample_idx in enumerate(sample_indices):
            ax = axes[idx]

            # 实际值
            actual = actuals[sample_idx]

            # 预测值
            pred = predictions[sample_idx]

            # 时间轴
            time_axis = np.arange(pred_len)

            # 绘制
            ax.plot(time_axis, actual, 'b-', label='实际值', linewidth=2, alpha=0.8, marker='o')
            ax.plot(time_axis, pred, 'r-', label='预测值', linewidth=2, alpha=0.8, marker='s')

            # 填充误差区域
            ax.fill_between(time_axis, actual, pred,
                            where=(pred >= actual),
                            color='red', alpha=0.2, label='正误差')
            ax.fill_between(time_axis, actual, pred,
                            where=(pred < actual),
                            color='blue', alpha=0.2, label='负误差')

            # 计算误差
            mse = np.mean((pred - actual) ** 2)
            mae = np.mean(np.abs(pred - actual))

            ax.set_xlabel('时间步')
            ax.set_ylabel('值')
            ax.set_title(f'样本 {sample_idx}: MSE={mse:.4f}, MAE={mae:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 添加误差条形图
            errors = np.abs(pred - actual)
            ax_twin = ax.twinx()
            ax_twin.bar(time_axis, errors, alpha=0.3, color='gray', width=0.4, label='绝对误差')
            ax_twin.set_ylabel('绝对误差')
            ax_twin.legend(loc='upper right')

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 保存图片
        if save_name:
            save_path = self.save_dir / f"{save_name}_predictions.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 预测对比图已保存到: {save_path}")

        plt.show()

    def plot_error_distribution(self,
                                predictions: np.ndarray,
                                actuals: np.ndarray,
                                title: str = "预测误差分布",
                                save_name: Optional[str] = None):
        """
        绘制预测误差分布

        Args:
            predictions: 预测值
            actuals: 实际值
            title: 图表标题
            save_name: 保存文件名
        """
        errors = predictions - actuals
        abs_errors = np.abs(errors)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. 误差直方图
        ax = axes[0, 0]
        ax.hist(errors.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('预测误差 (预测值 - 实际值)')
        ax.set_ylabel('频次')
        ax.set_title('预测误差分布')
        ax.grid(True, alpha=0.3)

        # 2. 绝对误差分布
        ax = axes[0, 1]
        ax.hist(abs_errors.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
        ax.set_xlabel('绝对误差')
        ax.set_ylabel('频次')
        ax.set_title('绝对误差分布')
        ax.grid(True, alpha=0.3)

        # 3. 误差箱线图（按时间步）
        ax = axes[1, 0]
        if len(predictions.shape) > 1:
            time_step_errors = [errors[:, i] for i in range(predictions.shape[1])]
            ax.boxplot(time_step_errors)
            ax.set_xlabel('预测时间步')
            ax.set_ylabel('预测误差')
            ax.set_title('各时间步预测误差分布')
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)

        # 4. 误差QQ图（检查正态性）
        ax = axes[1, 1]
        from scipy import stats
        stats.probplot(errors.flatten(), dist="norm", plot=ax)
        ax.set_title('误差QQ图（检验正态性）')
        ax.grid(True, alpha=0.3)

        # 计算统计指标
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        mae = np.mean(abs_errors)
        mse = np.mean(errors ** 2)

        # 添加统计信息文本
        stats_text = f"""
        误差统计:
        均值: {mean_error:.4f}
        标准差: {std_error:.4f}
        MAE: {mae:.4f}
        MSE: {mse:.4f}
        RMSE: {np.sqrt(mse):.4f}
        """

        fig.text(0.02, 0.02, stats_text, fontsize=10,
                 verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 保存图片
        if save_name:
            save_path = self.save_dir / f"{save_name}_error_distribution.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 误差分布图已保存到: {save_path}")

        plt.show()

    def plot_feature_importance(self,
                                model,
                                feature_names: List[str],
                                sample_data: torch.Tensor,
                                title: str = "特征重要性分析",
                                save_name: Optional[str] = None):
        """
        绘制特征重要性分析

        Args:
            model: 训练好的模型
            feature_names: 特征名称列表
            sample_data: 样本数据 [batch, seq_len, n_features]
            title: 图表标题
            save_name: 保存文件名
        """
        model.eval()

        # 使用梯度信息估计特征重要性
        sample_data.requires_grad = True

        # 前向传播
        features = model(sample_data)
        output = features.mean()  # 简化

        # 反向传播计算梯度
        output.backward()

        # 获取梯度
        gradients = sample_data.grad.abs().mean(dim=(0, 1)).cpu().numpy()  # [n_features]

        # 归一化
        gradients = gradients / (gradients.sum() + 1e-8)

        # 绘制特征重要性
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 1. 条形图
        ax = axes[0]
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, gradients, align='center', alpha=0.7, color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('特征重要性（梯度绝对值）')
        ax.set_title('基于梯度的特征重要性')
        ax.grid(True, alpha=0.3, axis='x')

        # 添加数值标签
        for i, v in enumerate(gradients):
            ax.text(v + 0.001, i, f'{v:.3f}', va='center')

        # 2. 饼图
        ax = axes[1]
        # 只显示重要性大于阈值
        threshold = 0.05
        mask = gradients > threshold
        if mask.sum() > 0:
            important_features = np.array(feature_names)[mask]
            important_values = gradients[mask]

            # 添加"其他"类别
            other_value = gradients[~mask].sum()
            if other_value > 0:
                important_features = np.append(important_features, '其他')
                important_values = np.append(important_values, other_value)

            wedges, texts, autotexts = ax.pie(important_values,
                                              labels=important_features,
                                              autopct='%1.1f%%',
                                              startangle=90,
                                              colors=plt.cm.Set3(np.linspace(0, 1, len(important_features))))
            ax.axis('equal')
            ax.set_title('特征重要性分布')

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 保存图片
        if save_name:
            save_path = self.save_dir / f"{save_name}_feature_importance.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🔍 特征重要性图已保存到: {save_path}")

        plt.show()

        return gradients


class InteractiveVisualizer:
    """交互式可视化（使用Plotly）"""

    def __init__(self):
        """初始化交互式可视化器"""
        self.figures = {}

    def create_interactive_training_curve(self,
                                          train_losses: List[float],
                                          val_losses: List[float],
                                          train_metrics: Optional[Dict] = None,
                                          title: str = "交互式训练曲线"):
        """
        创建交互式训练曲线

        Args:
            train_losses: 训练损失
            val_losses: 验证损失
            train_metrics: 训练指标
            title: 图表标题

        Returns:
            plotly.graph_objects.Figure: 交互式图表
        """
        epochs = list(range(1, len(train_losses) + 1))

        # 创建子图
        n_plots = 1
        if train_metrics:
            n_plots += len(train_metrics)

        fig = make_subplots(
            rows=n_plots, cols=1,
            subplot_titles=['训练和验证损失'] +
                           [f'{metric.upper()}曲线' for metric in train_metrics.keys()]
            if train_metrics else ['训练和验证损失'],
            vertical_spacing=0.1
        )

        # 添加损失曲线
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=train_losses,
                mode='lines+markers',
                name='训练损失',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=val_losses,
                mode='lines+markers',
                name='验证损失',
                line=dict(color='red', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )

        # 标记最佳点
        best_epoch = np.argmin(val_losses) + 1
        best_loss = min(val_losses)
        fig.add_trace(
            go.Scatter(
                x=[best_epoch],
                y=[best_loss],
                mode='markers',
                name='最佳验证损失',
                marker=dict(color='red', size=12, symbol='star'),
                text=[f'最佳: {best_loss:.4f}'],
                hoverinfo='text'
            ),
            row=1, col=1
        )

        # 添加指标曲线
        if train_metrics:
            for idx, (metric_name, values) in enumerate(train_metrics.items(), 2):
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=values,
                        mode='lines+markers',
                        name=f'训练{metric_name.upper()}',
                        line=dict(color='green', width=2),
                        marker=dict(size=6)
                    ),
                    row=idx, col=1
                )

        # 更新布局
        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            height=300 * n_plots,
            showlegend=True,
            hovermode='x unified'
        )

        # 更新坐标轴
        for i in range(1, n_plots + 1):
            fig.update_xaxes(title_text="Epoch", row=i, col=1)
            fig.update_yaxes(title_text="值", row=i, col=1)

        self.figures['training_curve'] = fig
        return fig

    def create_interactive_prediction_plot(self,
                                           predictions: np.ndarray,
                                           actuals: np.ndarray,
                                           sample_indices: List[int] = None,
                                           title: str = "交互式预测对比"):
        """
        创建交互式预测对比图

        Args:
            predictions: 预测值
            actuals: 实际值
            sample_indices: 样本索引
            title: 图表标题

        Returns:
            plotly.graph_objects.Figure: 交互式图表
        """
        if sample_indices is None:
            sample_indices = list(range(min(4, len(predictions))))

        n_samples = len(sample_indices)

        fig = make_subplots(
            rows=n_samples, cols=1,
            subplot_titles=[f'样本 {idx}' for idx in sample_indices],
            vertical_spacing=0.15
        )

        for i, sample_idx in enumerate(sample_indices, 1):
            actual = actuals[sample_idx]
            pred = predictions[sample_idx]
            time_steps = list(range(len(actual)))

            # 计算误差
            errors = np.abs(pred - actual)

            # 添加实际值曲线
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=actual,
                    mode='lines+markers',
                    name='实际值',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8),
                    legendgroup=f'group{i}',
                    showlegend=(i == 1)
                ),
                row=i, col=1
            )

            # 添加预测值曲线
            fig.add_trace(
                go.Scatter(
                    x=time_steps,
                    y=pred,
                    mode='lines+markers',
                    name='预测值',
                    line=dict(color='red', width=3),
                    marker=dict(size=8, symbol='diamond'),
                    legendgroup=f'group{i}',
                    showlegend=(i == 1)
                ),
                row=i, col=1
            )

            # 添加误差条形图
            fig.add_trace(
                go.Bar(
                    x=time_steps,
                    y=errors,
                    name='绝对误差',
                    marker=dict(color='gray', opacity=0.5),
                    yaxis='y2',
                    legendgroup=f'error{i}',
                    showlegend=(i == 1)
                ),
                row=i, col=1
            )

            # 设置双Y轴
            fig.update_layout({
                f'yaxis{i}': dict(title='值', titlefont=dict(color='blue')),
                f'yaxis{i + 1}': dict(title='绝对误差', titlefont=dict(color='gray'),
                                      overlaying=f'y{i}', side='right')
            })

            # 计算并显示指标
            mse = np.mean((pred - actual) ** 2)
            mae = np.mean(errors)

            # 添加指标标注
            fig.add_annotation(
                x=0.02, y=0.95,
                xref=f'x{i}', yref=f'y{i} domain',
                text=f'MSE: {mse:.4f}<br>MAE: {mae:.4f}',
                showarrow=False,
                font=dict(size=10),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1,
                borderpad=4,
                row=i, col=1
            )

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=20)),
            height=300 * n_samples,
            showlegend=True,
            hovermode='x unified'
        )

        self.figures['prediction_plot'] = fig
        return fig

    def save_figure(self, fig_name: str, save_path: str, format: str = 'html'):
        """
        保存交互式图表

        Args:
            fig_name: 图表名称
            save_path: 保存路径
            format: 格式 ('html', 'png', 'jpeg', 'svg', 'pdf')
        """
        if fig_name not in self.figures:
            print(f"⚠️  图表 '{fig_name}' 不存在")
            return

        fig = self.figures[fig_name]
        save_path = Path(save_path)

        if format == 'html':
            fig.write_html(str(save_path))
        else:
            fig.write_image(str(save_path), format=format)

        print(f"💾 交互式图表已保存到: {save_path}")


def visualize_correlation_matrix(data: np.ndarray,
                                 feature_names: List[str],
                                 title: str = "特征相关性矩阵",
                                 save_path: Optional[str] = None):
    """
    可视化相关性矩阵

    Args:
        data: 数据数组 [n_samples, n_features]
        feature_names: 特征名称列表
        title: 图表标题
        save_path: 保存路径
    """
    # 计算相关性矩阵
    corr_matrix = np.corrcoef(data.T)

    # 创建热力图
    fig, ax = plt.subplots(figsize=(10, 8))

    # 使用seaborn绘制热力图
    sns.heatmap(corr_matrix,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                ax=ax)

    # 设置标签
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_yticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticklabels(feature_names, rotation=0)

    ax.set_title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 相关性矩阵已保存到: {save_path}")

    plt.show()

    return corr_matrix


def visualize_time_series(data: pd.DataFrame,
                          columns: List[str] = None,
                          title: str = "时间序列数据可视化",
                          save_path: Optional[str] = None):
    """
    可视化时间序列数据

    Args:
        data: 时间序列DataFrame
        columns: 要可视化的列
        title: 图表标题
        save_path: 保存路径
    """
    if columns is None:
        columns = data.columns.tolist()

    n_cols = len(columns)
    fig, axes = plt.subplots(n_cols, 1, figsize=(14, 3 * n_cols))

    if n_cols == 1:
        axes = [axes]

    for idx, col in enumerate(columns):
        ax = axes[idx]

        # 绘制时间序列
        ax.plot(data.index, data[col], linewidth=1.5, alpha=0.8)

        # 添加滚动平均
        rolling_mean = data[col].rolling(window=24).mean()
        ax.plot(data.index, rolling_mean, 'r-', linewidth=2, alpha=0.8, label='24h滚动平均')

        # 添加填充区域（±标准差）
        rolling_std = data[col].rolling(window=24).std()
        ax.fill_between(data.index,
                        rolling_mean - rolling_std,
                        rolling_mean + rolling_std,
                        alpha=0.2, color='red', label='±1标准差')

        ax.set_ylabel(col)
        ax.set_title(f'{col} 时间序列')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 时间序列图已保存到: {save_path}")

    plt.show()


def create_dashboard(metrics: Dict[str, Any],
                     predictions: np.ndarray,
                     actuals: np.ndarray,
                     save_dir: str = "./dashboard"):
    """
    创建完整的实验仪表板

    Args:
        metrics: 实验指标
        predictions: 预测值
        actuals: 实际值
        save_dir: 仪表板保存目录
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 初始化可视化器
    trainer_viz = TrainingVisualizer(save_dir=save_dir)
    interactive_viz = InteractiveVisualizer()

    # 1. 训练曲线
    if 'train_loss' in metrics and 'val_loss' in metrics:
        trainer_viz.plot_training_curves(
            metrics['train_loss'],
            metrics['val_loss'],
            train_metrics={'mse': metrics.get('train_mse', [])},
            val_metrics={'mse': metrics.get('val_mse', [])},
            title="STAR-Forecast 训练过程",
            save_name="training_curves"
        )

        # 交互式版本
        interactive_fig = interactive_viz.create_interactive_training_curve(
            metrics['train_loss'],
            metrics['val_loss'],
            train_metrics={'mse': metrics.get('train_mse', [])},
            title="STAR-Forecast 交互式训练曲线"
        )
        interactive_viz.save_figure('training_curve', save_dir / 'training_curve.html')

    # 2. 预测对比
    if len(predictions) > 0:
        # 随机选择4个样本
        sample_indices = np.random.choice(len(predictions), min(4, len(predictions)), replace=False)

        trainer_viz.plot_prediction_vs_actual(
            predictions,
            actuals,
            sample_indices=sample_indices.tolist(),
            title="STAR-Forecast 预测 vs 实际值",
            save_name="predictions"
        )

        # 交互式版本
        interactive_fig = interactive_viz.create_interactive_prediction_plot(
            predictions,
            actuals,
            sample_indices=sample_indices.tolist()[:3],
            title="STAR-Forecast 交互式预测对比"
        )
        interactive_viz.save_figure('prediction_plot', save_dir / 'prediction_plot.html')

    # 3. 误差分析
    if len(predictions) > 0:
        trainer_viz.plot_error_distribution(
            predictions,
            actuals,
            title="STAR-Forecast 预测误差分析",
            save_name="error_analysis"
        )

    # 4. 创建汇总报告
    create_summary_report(metrics, predictions, actuals, save_dir)

    print(f"📊 实验仪表板已保存到: {save_dir}")


def create_summary_report(metrics: Dict[str, Any],
                          predictions: np.ndarray,
                          actuals: np.ndarray,
                          save_dir: Path):
    """
    创建实验摘要报告

    Args:
        metrics: 实验指标
        predictions: 预测值
        actuals: 实际值
        save_dir: 保存目录
    """
    # 计算最终指标
    if len(predictions) > 0:
        test_mse = np.mean((predictions - actuals) ** 2)
        test_mae = np.mean(np.abs(predictions - actuals))
        test_rmse = np.sqrt(test_mse)
    else:
        test_mse = test_mae = test_rmse = 0.0

    # 创建报告
    report = f"""
    ========================================
    STAR-Forecast 实验摘要报告
    ========================================

    一、训练过程统计
    ----------------------------------------
    训练轮次: {len(metrics.get('train_loss', []))}
    最佳训练损失: {min(metrics.get('train_loss', [0])):.6f}
    最佳验证损失: {min(metrics.get('val_loss', [0])):.6f}

    二、测试性能指标
    ----------------------------------------
    测试MSE: {test_mse:.6f}
    测试MAE: {test_mae:.6f}
    测试RMSE: {test_rmse:.6f}

    三、预测误差分析
    ----------------------------------------
    """

    if len(predictions) > 0:
        errors = predictions - actuals
        abs_errors = np.abs(errors)

        report += f"""
    平均误差: {np.mean(errors):.6f}
    误差标准差: {np.std(errors):.6f}
    最大绝对误差: {np.max(abs_errors):.6f}
    最小绝对误差: {np.min(abs_errors):.6f}
    误差中位数: {np.median(abs_errors):.6f}

    四、模型性能评估
    ----------------------------------------
    R²分数: {max(0, 1 - test_mse / np.var(actuals)):.4f}
    """

    # 保存报告
    report_file = save_dir / "experiment_summary.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"📋 实验摘要报告已保存到: {report_file}")


if __name__ == "__main__":
    # 测试可视化模块
    print("🧪 测试可视化模块...")

    # 创建测试数据
    np.random.seed(42)

    # 模拟训练指标
    epochs = 50
    train_losses = np.exp(-np.linspace(0, 3, epochs)) + np.random.normal(0, 0.02, epochs)
    val_losses = np.exp(-np.linspace(0, 2.8, epochs)) + np.random.normal(0, 0.03, epochs)

    train_mse = train_losses * 0.8
    val_mse = val_losses * 0.9

    # 模拟预测结果
    n_samples = 100
    pred_len = 24
    actuals = np.random.randn(n_samples, pred_len)
    predictions = actuals + np.random.normal(0, 0.5, (n_samples, pred_len))

    # 测试训练曲线可视化
    viz = TrainingVisualizer(save_dir="./test_viz")
    viz.plot_training_curves(
        train_losses.tolist(),
        val_losses.tolist(),
        train_metrics={'mse': train_mse.tolist()},
        val_metrics={'mse': val_mse.tolist()},
        title="测试训练曲线",
        save_name="test_training"
    )

    # 测试预测可视化
    viz.plot_prediction_vs_actual(
        predictions,
        actuals,
        sample_indices=[0, 10, 20],
        title="测试预测对比",
        save_name="test_predictions"
    )

    # 测试误差分布
    viz.plot_error_distribution(
        predictions,
        actuals,
        title="测试误差分布",
        save_name="test_errors"
    )

    print("✅ 可视化模块测试完成")