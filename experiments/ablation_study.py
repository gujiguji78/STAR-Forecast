"""
消融实验管理器 - 系统性地测试STAR-Forecast各组件的作用
支持多种消融变体、统计分析、结果可视化
"""
import json
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import logging
import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import itertools
from enum import Enum
import uuid
import pickle

# 导入项目模块
import sys

sys.path.append('..')
from training.trainer import STARForecastTrainer
from models.istr import ISTRNetwork
from client.api_client import AgentLightningClient
from agents.autogen_system import AutoGenMultiAgentSystem


class AblationVariant(Enum):
    """消融变体类型"""
    FULL_MODEL = "full"  # 完整模型
    NO_AUTOGEN = "no_autogen"  # 无AutoGen智能体
    NO_AGENT_LIGHTNING = "no_agent_lightning"  # 无Agent Lightning
    NO_ISTR = "no_istr"  # 无ISTR网络（基础TCN）
    NO_LAPLACIAN = "no_laplacian"  # 无拉普拉斯正则化
    NO_SPECTRAL_GATE = "no_spectral_gate"  # 无谱门控
    FROZEN_ISTR = "frozen_istr"  # ISTR完全冻结
    SINGLE_AGENT = "single_agent"  # 单智能体（非多智能体）
    NO_SEMANTIC_REWARD = "no_semantic_reward"  # 无语义奖励
    SIMPLE_BASELINE = "simple_baseline"  # 简单基线（线性模型）


@dataclass
class AblationConfig:
    """消融实验配置"""
    variant: AblationVariant
    description: str
    config_modifications: Dict[str, Any]
    training_epochs: int = 50  # 消融实验用较少epochs
    num_runs: int = 3  # 每个变体运行次数（减少随机性）
    random_seeds: List[int] = field(default_factory=lambda: [42, 43, 44])


@dataclass
class AblationResult:
    """消融实验结果"""
    variant: AblationVariant
    run_id: str
    seed: int
    config: Dict[str, Any]
    training_history: Dict[str, List[float]]
    test_metrics: Dict[str, float]
    training_time: float  # 秒
    resource_usage: Dict[str, float]  # GPU内存、显存等
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'variant': self.variant.value,
            'run_id': self.run_id,
            'seed': self.seed,
            'config': self.config,
            'test_metrics': self.test_metrics,
            'training_time': self.training_time,
            'resource_usage': self.resource_usage,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class AblationComparison:
    """消融实验对比结果"""
    experiment_id: str
    variants: List[AblationVariant]
    results: Dict[str, List[AblationResult]]  # variant -> list of results
    summary_stats: Dict[str, Dict[str, float]]
    statistical_tests: Dict[str, Dict[str, float]]
    created_at: datetime = field(default_factory=datetime.now)

    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame便于分析"""
        rows = []

        for variant, result_list in self.results.items():
            for result in result_list:
                row = {
                    'experiment_id': self.experiment_id,
                    'variant': variant,
                    'run_id': result.run_id,
                    'seed': result.seed,
                    'training_time': result.training_time,
                    **result.test_metrics
                }
                rows.append(row)

        return pd.DataFrame(rows)


class ResourceMonitor:
    """资源使用监控器"""

    def __init__(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self.has_gpu = True
            self.gpu_count = pynvml.nvmlDeviceGetCount()
        except:
            self.has_gpu = False
            self.gpu_count = 0

        self.start_time = None
        self.max_gpu_memory = 0
        self.max_cpu_memory = 0

    def start_monitoring(self):
        """开始监控"""
        self.start_time = datetime.now()

        if self.has_gpu:
            self._reset_gpu_stats()

        self._reset_cpu_stats()

    def stop_monitoring(self) -> Dict[str, float]:
        """停止监控并返回统计"""
        if not self.start_time:
            return {}

        # 计算运行时间
        duration = (datetime.now() - self.start_time).total_seconds()

        # 获取资源使用峰值
        resource_stats = {
            'training_time_seconds': duration,
            'has_gpu': self.has_gpu
        }

        if self.has_gpu:
            gpu_stats = self._get_gpu_stats()
            resource_stats.update(gpu_stats)

        cpu_stats = self._get_cpu_stats()
        resource_stats.update(cpu_stats)

        return resource_stats

    def _reset_gpu_stats(self):
        """重置GPU统计"""
        self.max_gpu_memory = 0

    def _reset_cpu_stats(self):
        """重置CPU统计"""
        self.max_cpu_memory = 0

    def _get_gpu_stats(self) -> Dict[str, float]:
        """获取GPU统计"""
        try:
            import pynvml

            gpu_stats = {}
            total_memory = 0
            max_used = 0

            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                total_memory += info.total / 1024 ** 3  # GB
                used = info.used / 1024 ** 3
                max_used = max(max_used, used)

            gpu_stats['gpu_memory_total_gb'] = total_memory
            gpu_stats['gpu_memory_max_used_gb'] = max_used
            gpu_stats['gpu_memory_utilization'] = max_used / total_memory if total_memory > 0 else 0

            return gpu_stats

        except Exception as e:
            logging.warning(f"无法获取GPU统计: {e}")
            return {}

    def _get_cpu_stats(self) -> Dict[str, float]:
        """获取CPU统计"""
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()

            cpu_stats = {
                'cpu_memory_rss_gb': memory_info.rss / 1024 ** 3,
                'cpu_memory_vms_gb': memory_info.vms / 1024 ** 3,
                'cpu_percent': process.cpu_percent(interval=1)
            }

            return cpu_stats

        except Exception as e:
            logging.warning(f"无法获取CPU统计: {e}")
            return {}


class AblationStudyManager:
    """
    消融实验管理器

    功能：
    1. 自动生成不同消融变体的配置
    2. 并行/顺序运行多个实验
    3. 收集和分析实验结果
    4. 生成统计检验和可视化
    5. 保存和比较实验结果
    """

    def __init__(self, base_config_path: str = "./config.yaml"):
        self.base_config = self._load_config(base_config_path)
        self.logger = logging.getLogger(__name__)

        # 实验结果存储
        self.results_dir = Path("./experiments/ablation_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 实验历史
        self.experiment_history: Dict[str, AblationComparison] = {}

        # 资源监控器
        self.resource_monitor = ResourceMonitor()

        self.logger.info("🔬 消融实验管理器初始化完成")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def get_ablation_variants(self) -> Dict[AblationVariant, AblationConfig]:
        """获取所有消融变体配置"""
        variants = {}

        # 1. 完整模型（基准）
        variants[AblationVariant.FULL_MODEL] = AblationConfig(
            variant=AblationVariant.FULL_MODEL,
            description="完整STAR-Forecast模型",
            config_modifications={}
        )

        # 2. 无AutoGen智能体
        variants[AblationVariant.NO_AUTOGEN] = AblationConfig(
            variant=AblationVariant.NO_AUTOGEN,
            description="无AutoGen多智能体系统",
            config_modifications={
                'autogen': {
                    'trigger': {'check_interval': 1000000},  # 极大值，基本不触发
                    'conversation': {'max_rounds': 0}
                }
            }
        )

        # 3. 无Agent Lightning
        variants[AblationVariant.NO_AGENT_LIGHTNING] = AblationConfig(
            variant=AblationVariant.NO_AGENT_LIGHTNING,
            description="无Agent Lightning强化学习",
            config_modifications={
                'agent_lightning': {
                    'client': {'fallback_enabled': False},
                    'rl': {'reward': {'weights': {'semantic': 0.0, 'constraint': 0.0}}
                           }
                }
            }
        )

        # 4. 无ISTR网络（使用简单TCN）
        variants[AblationVariant.NO_ISTR] = AblationConfig(
            variant=AblationVariant.NO_ISTR,
            description="无ISTR网络，使用标准TCN",
            config_modifications={
                'istr': {
                    'tcn': {'num_blocks': 1},
                    'spectral_gate': {'enabled': False},
                    'laplacian': {'enabled': False},
                    'trainable_ratio': 1.0  # 全部可训练
                }
            }
        )

        # 5. 无拉普拉斯正则化
        variants[AblationVariant.NO_LAPLACIAN] = AblationConfig(
            variant=AblationVariant.NO_LAPLACIAN,
            description="无拉普拉斯正则化",
            config_modifications={
                'istr': {'laplacian': {'enabled': False}}
            }
        )

        # 6. 无谱门控
        variants[AblationVariant.NO_SPECTRAL_GATE] = AblationConfig(
            variant=AblationVariant.NO_SPECTRAL_GATE,
            description="无谱门控机制",
            config_modifications={
                'istr': {'spectral_gate': {'enabled': False}}
            }
        )

        # 7. 冻结ISTR
        variants[AblationVariant.FROZEN_ISTR] = AblationConfig(
            variant=AblationVariant.FROZEN_ISTR,
            description="ISTR网络完全冻结",
            config_modifications={
                'istr': {'trainable_ratio': 0.0}
            }
        )

        # 8. 单智能体
        variants[AblationVariant.SINGLE_AGENT] = AblationConfig(
            variant=AblationVariant.SINGLE_AGENT,
            description="单智能体（非多智能体协同）",
            config_modifications={
                'autogen': {
                    'agents': {
                        'architect': None,  # 禁用架构师
                        'critic': None  # 禁用批评家
                    }
                }
            }
        )

        # 9. 无语义奖励
        variants[AblationVariant.NO_SEMANTIC_REWARD] = AblationConfig(
            variant=AblationVariant.NO_SEMANTIC_REWARD,
            description="强化学习中无语义奖励",
            config_modifications={
                'agent_lightning': {
                    'rl': {'reward': {'weights': {'semantic': 0.0}}}
                }
            }
        )

        # 10. 简单基线
        variants[AblationVariant.SIMPLE_BASELINE] = AblationConfig(
            variant=AblationVariant.SIMPLE_BASELINE,
            description="简单线性模型基线",
            config_modifications={
                'istr': {
                    'hidden_dim': 16,
                    'tcn': {'num_blocks': 0},
                    'spectral_gate': {'enabled': False},
                    'laplacian': {'enabled': False}
                },
                'predictor': {
                    'type': 'linear',
                    'hidden_dims': []
                }
            }
        )

        return variants

    def run_ablation_experiment(self,
                                variants: List[AblationVariant] = None,
                                data_path: str = "./data/ETTh1.csv",
                                experiment_name: str = None) -> AblationComparison:
        """
        运行消融实验

        Args:
            variants: 要测试的变体列表（None则测试所有）
            data_path: 数据路径
            experiment_name: 实验名称

        Returns:
            消融实验对比结果
        """
        # 获取变体配置
        all_variants = self.get_ablation_variants()

        if variants is None:
            variants_to_test = list(all_variants.keys())
        else:
            variants_to_test = variants

        # 创建实验ID
        experiment_id = experiment_name or f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.logger.info(f"🧪 开始消融实验: {experiment_id}")
        self.logger.info(f"   测试变体: {[v.value for v in variants_to_test]}")

        # 运行每个变体
        results = {}

        for variant in variants_to_test:
            if variant not in all_variants:
                self.logger.warning(f"未知变体: {variant}")
                continue

            config = all_variants[variant]
            variant_results = []

            self.logger.info(f"\n🔬 测试变体: {variant.value}")
            self.logger.info(f"   描述: {config.description}")

            for run_idx, seed in enumerate(config.random_seeds[:config.num_runs]):
                run_id = f"{variant.value}_run{run_idx + 1}"

                self.logger.info(f"   运行 {run_id} (种子: {seed})...")

                # 运行单个实验
                result = self._run_single_experiment(
                    config, seed, data_path, run_id
                )

                if result:
                    variant_results.append(result)
                    self.logger.info(f"      MSE: {result.test_metrics.get('mse', 0):.6f}, "
                                     f"MAE: {result.test_metrics.get('mae', 0):.6f}, "
                                     f"时间: {result.training_time:.1f}s")

            results[variant.value] = variant_results

        # 分析结果
        comparison = self._analyze_results(experiment_id, results)

        # 保存结果
        self._save_experiment(comparison)

        # 可视化
        self._visualize_results(comparison)

        self.logger.info(f"✅ 消融实验完成: {experiment_id}")

        return comparison

    def _run_single_experiment(self,
                               ablation_config: AblationConfig,
                               seed: int,
                               data_path: str,
                               run_id: str) -> Optional[AblationResult]:
        """运行单个消融实验"""
        try:
            # 创建配置副本并应用修改
            config = self._deep_copy_config(self.base_config)
            config = self._apply_config_modifications(config, ablation_config.config_modifications)

            # 设置随机种子
            config['experiment']['seed'] = seed

            # 减少训练轮次以加快消融实验
            if 'training' in config:
                config['training']['epochs'] = ablation_config.training_epochs

            # 开始资源监控
            self.resource_monitor.start_monitoring()
            start_time = datetime.now()

            # 创建并运行训练器
            trainer = STARForecastTrainer(config)
            trainer.build_models()
            trainer.build_optimizer()

            # 对于某些变体，需要特殊处理
            if ablation_config.variant == AblationVariant.NO_AUTOGEN:
                # 不初始化AutoGen系统
                pass
            else:
                trainer.initialize_agents()

            # 训练模型
            test_metrics = trainer.train(data_path)

            # 停止资源监控
            training_time = (datetime.now() - start_time).total_seconds()
            resource_usage = self.resource_monitor.stop_monitoring()

            # 创建结果
            result = AblationResult(
                variant=ablation_config.variant,
                run_id=run_id,
                seed=seed,
                config=config,
                training_history=getattr(trainer, 'training_history', {}),
                test_metrics=test_metrics,
                training_time=training_time,
                resource_usage=resource_usage
            )

            return result

        except Exception as e:
            self.logger.error(f"❌ 实验 {run_id} 失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _deep_copy_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """深拷贝配置"""
        import copy
        return copy.deepcopy(config)

    def _apply_config_modifications(self,
                                    config: Dict[str, Any],
                                    modifications: Dict[str, Any]) -> Dict[str, Any]:
        """应用配置修改"""
        if not modifications:
            return config

        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        return update_dict(config, modifications)

    def _analyze_results(self,
                         experiment_id: str,
                         results: Dict[str, List[AblationResult]]) -> AblationComparison:
        """分析实验结果"""
        summary_stats = {}
        statistical_tests = {}

        # 提取主要指标
        primary_metric = 'mse'  # 主要比较MSE

        for variant, result_list in results.items():
            if not result_list:
                continue

            # 计算统计量
            metrics = []
            for result in result_list:
                if primary_metric in result.test_metrics:
                    metrics.append(result.test_metrics[primary_metric])

            if metrics:
                summary_stats[variant] = {
                    'mean': np.mean(metrics),
                    'std': np.std(metrics),
                    'min': np.min(metrics),
                    'max': np.max(metrics),
                    'median': np.median(metrics),
                    'count': len(metrics)
                }

        # 统计检验（与完整模型比较）
        if 'full' in results and results['full']:
            full_model_metrics = []
            for result in results['full']:
                if primary_metric in result.test_metrics:
                    full_model_metrics.append(result.test_metrics[primary_metric])

            if full_model_metrics:
                for variant, result_list in results.items():
                    if variant == 'full' or not result_list:
                        continue

                    other_metrics = []
                    for result in result_list:
                        if primary_metric in result.test_metrics:
                            other_metrics.append(result.test_metrics[primary_metric])

                    if other_metrics:
                        # t检验
                        t_stat, p_value = stats.ttest_ind(
                            full_model_metrics,
                            other_metrics,
                            equal_var=False  # Welch's t-test
                        )

                        # Wilcoxon秩和检验
                        if len(full_model_metrics) == len(other_metrics):
                            w_stat, w_pvalue = stats.wilcoxon(
                                full_model_metrics,
                                other_metrics
                            )
                        else:
                            w_stat, w_pvalue = stats.ranksums(
                                full_model_metrics,
                                other_metrics
                            )

                        statistical_tests[variant] = {
                            't_test': {
                                'statistic': float(t_stat),
                                'p_value': float(p_value),
                                'significant': p_value < 0.05
                            },
                            'wilcoxon': {
                                'statistic': float(w_stat),
                                'p_value': float(w_pvalue),
                                'significant': w_pvalue < 0.05
                            },
                            'effect_size': self._calculate_effect_size(
                                full_model_metrics, other_metrics
                            )
                        }

        # 创建对比结果
        comparison = AblationComparison(
            experiment_id=experiment_id,
            variants=[AblationVariant(v) for v in results.keys()],
            results=results,
            summary_stats=summary_stats,
            statistical_tests=statistical_tests
        )

        return comparison

    def _calculate_effect_size(self, group1: List[float], group2: List[float]) -> float:
        """计算效应大小（Cohen's d）"""
        if len(group1) == 0 or len(group2) == 0:
            return 0.0

        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)

        # 合并标准差
        pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return abs(mean1 - mean2) / pooled_std

    def _save_experiment(self, comparison: AblationComparison):
        """保存实验结果"""
        # 保存为JSON
        json_path = self.results_dir / f"{comparison.experiment_id}.json"

        with open(json_path, 'w', encoding='utf-8') as f:
            # 转换为可序列化格式
            data = {
                'experiment_id': comparison.experiment_id,
                'created_at': comparison.created_at.isoformat(),
                'variants': [v.value for v in comparison.variants],
                'summary_stats': comparison.summary_stats,
                'statistical_tests': comparison.statistical_tests
            }

            # 保存详细结果
            detailed_results = {}
            for variant, result_list in comparison.results.items():
                detailed_results[variant] = [r.to_dict() for r in result_list]

            data['detailed_results'] = detailed_results

            json.dump(data, f, indent=2, ensure_ascii=False)

        # 保存为CSV（便于分析）
        df = comparison.to_dataframe()
        csv_path = self.results_dir / f"{comparison.experiment_id}.csv"
        df.to_csv(csv_path, index=False)

        # 保存为pickle（完整对象）
        pickle_path = self.results_dir / f"{comparison.experiment_id}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(comparison, f)

        self.logger.info(f"💾 实验结果保存到:")
        self.logger.info(f"   JSON: {json_path}")
        self.logger.info(f"   CSV: {csv_path}")
        self.logger.info(f"   Pickle: {pickle_path}")

        # 更新实验历史
        self.experiment_history[comparison.experiment_id] = comparison

    def _visualize_results(self, comparison: AblationComparison):
        """可视化实验结果"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # 设置样式
            plt.style.use('seaborn-v0_8-darkgrid')
            sns.set_palette("husl")

            # 创建可视化目录
            vis_dir = self.results_dir / "visualizations"
            vis_dir.mkdir(exist_ok=True)

            # 1. 性能对比图（箱线图）
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Ablation Study: {comparison.experiment_id}', fontsize=16)

            # 准备数据
            rows = []
            for variant, result_list in comparison.results.items():
                for result in result_list:
                    rows.append({
                        'Variant': variant,
                        'MSE': result.test_metrics.get('mse', 0),
                        'MAE': result.test_metrics.get('mae', 0),
                        'Training Time (s)': result.training_time
                    })

            df = pd.DataFrame(rows)

            # 1.1 MSE箱线图
            ax1 = axes[0, 0]
            sns.boxplot(data=df, x='Variant', y='MSE', ax=ax1)
            ax1.set_title('Test MSE Distribution')
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
            ax1.set_ylabel('MSE (Lower is Better)')

            # 1.2 MAE箱线图
            ax2 = axes[0, 1]
            sns.boxplot(data=df, x='Variant', y='MAE', ax=ax2)
            ax2.set_title('Test MAE Distribution')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
            ax2.set_ylabel('MAE (Lower is Better)')

            # 1.3 训练时间条形图
            ax3 = axes[1, 0]
            time_stats = df.groupby('Variant')['Training Time (s)'].mean()
            time_stats.plot(kind='bar', ax=ax3, color='skyblue')
            ax3.set_title('Average Training Time')
            ax3.set_xlabel('Variant')
            ax3.set_ylabel('Time (seconds)')
            ax3.tick_params(axis='x', rotation=45)

            # 1.4 性能-时间散点图
            ax4 = axes[1, 1]
            avg_metrics = df.groupby('Variant').agg({
                'MSE': 'mean',
                'Training Time (s)': 'mean'
            }).reset_index()

            sns.scatterplot(data=avg_metrics, x='Training Time (s)', y='MSE',
                            hue='Variant', s=100, ax=ax4)
            ax4.set_title('Performance vs Training Time Trade-off')
            ax4.set_xlabel('Training Time (seconds)')
            ax4.set_ylabel('Average MSE')

            # 添加标签
            for idx, row in avg_metrics.iterrows():
                ax4.annotate(row['Variant'],
                             (row['Training Time (s)'], row['MSE']),
                             textcoords="offset points",
                             xytext=(0, 10), ha='center')

            plt.tight_layout()
            plt.savefig(vis_dir / f"{comparison.experiment_id}_performance.png",
                        dpi=300, bbox_inches='tight')
            plt.close()

            # 2. 统计显著性热图
            if comparison.statistical_tests:
                variants = list(comparison.statistical_tests.keys())
                p_values = np.zeros((len(variants), 2))  # t-test和Wilcoxon

                for i, variant in enumerate(variants):
                    tests = comparison.statistical_tests[variant]
                    p_values[i, 0] = tests['t_test']['p_value']
                    p_values[i, 1] = tests['wilcoxon']['p_value']

                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(p_values, cmap='Reds', aspect='auto')

                ax.set_xticks([0, 1])
                ax.set_xticklabels(['t-test', 'Wilcoxon'])
                ax.set_yticks(range(len(variants)))
                ax.set_yticklabels(variants)

                # 添加数值
                for i in range(len(variants)):
                    for j in range(2):
                        text = ax.text(j, i, f'{p_values[i, j]:.3f}',
                                       ha="center", va="center",
                                       color="white" if p_values[i, j] > 0.5 else "black")

                ax.set_title('Statistical Significance (p-values)\nvs Full Model')
                plt.colorbar(im, ax=ax, label='p-value')
                plt.tight_layout()
                plt.savefig(vis_dir / f"{comparison.experiment_id}_significance.png",
                            dpi=300, bbox_inches='tight')
                plt.close()

            # 3. 效应大小条形图
            if comparison.statistical_tests:
                effect_sizes = []
                variant_names = []

                for variant, tests in comparison.statistical_tests.items():
                    effect_sizes.append(tests['effect_size'])
                    variant_names.append(variant)

                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(variant_names, effect_sizes, color='lightcoral')

                # 添加效应大小标签
                ax.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
                ax.axhline(y=0.5, color='gray', linestyle='-.', alpha=0.5, label='Medium effect')
                ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, label='Large effect')

                ax.set_xlabel('Variant')
                ax.set_ylabel("Cohen's d Effect Size")
                ax.set_title('Effect Size vs Full Model')
                ax.tick_params(axis='x', rotation=45)
                ax.legend()

                # 在柱子上添加数值
                for bar, effect in zip(bars, effect_sizes):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                            f'{effect:.2f}', ha='center', va='bottom')

                plt.tight_layout()
                plt.savefig(vis_dir / f"{comparison.experiment_id}_effect_size.png",
                            dpi=300, bbox_inches='tight')
                plt.close()

            self.logger.info(f"📊 可视化结果保存到: {vis_dir}")

        except Exception as e:
            self.logger.warning(f"可视化失败: {e}")

    def load_experiment(self, experiment_id: str) -> Optional[AblationComparison]:
        """加载实验"""
        pickle_path = self.results_dir / f"{experiment_id}.pkl"

        if pickle_path.exists():
            with open(pickle_path, 'rb') as f:
                comparison = pickle.load(f)
            return comparison

        return None

    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """比较多个实验"""
        comparisons = []

        for exp_id in experiment_ids:
            comparison = self.load_experiment(exp_id)
            if comparison:
                df = comparison.to_dataframe()
                df['experiment'] = exp_id
                comparisons.append(df)

        if comparisons:
            return pd.concat(comparisons, ignore_index=True)

        return pd.DataFrame()

    def generate_report(self, comparison: AblationComparison) -> str:
        """生成实验报告"""
        report = []

        report.append("=" * 80)
        report.append("消融实验报告")
        report.append("=" * 80)
        report.append(f"实验ID: {comparison.experiment_id}")
        report.append(f"创建时间: {comparison.created_at}")
        report.append(f"测试变体数: {len(comparison.variants)}")
        report.append("")

        # 性能总结
        report.append("📊 性能总结 (MSE)")
        report.append("-" * 40)

        for variant, stats in comparison.summary_stats.items():
            report.append(f"{variant:<20} Mean: {stats['mean']:.6f} ± {stats['std']:.6f} "
                          f"(Min: {stats['min']:.6f}, Max: {stats['max']:.6f})")

        report.append("")

        # 统计显著性
        if comparison.statistical_tests:
            report.append("📈 统计显著性检验 (vs 完整模型)")
            report.append("-" * 40)

            for variant, tests in comparison.statistical_tests.items():
                t_sig = "✓" if tests['t_test']['significant'] else "✗"
                w_sig = "✓" if tests['wilcoxon']['significant'] else "✗"

                report.append(f"{variant:<20} t-test: p={tests['t_test']['p_value']:.4f} {t_sig} "
                              f"| Wilcoxon: p={tests['wilcoxon']['p_value']:.4f} {w_sig} "
                              f"| Effect size: {tests['effect_size']:.3f}")

        report.append("")

        # 关键发现
        report.append("🔑 关键发现")
        report.append("-" * 40)

        # 找出性能最好的变体
        best_variant = None
        best_mse = float('inf')

        for variant, stats in comparison.summary_stats.items():
            if stats['mean'] < best_mse:
                best_mse = stats['mean']
                best_variant = variant

        if best_variant:
            report.append(f"1. 最佳性能变体: {best_variant} (MSE: {best_mse:.6f})")

        # 找出性能下降最多的变体（相比完整模型）
        if 'full' in comparison.summary_stats:
            full_mse = comparison.summary_stats['full']['mean']

            worst_relative = None
            worst_ratio = 0

            for variant, stats in comparison.summary_stats.items():
                if variant != 'full':
                    ratio = stats['mean'] / full_mse
                    if ratio > worst_ratio:
                        worst_ratio = ratio
                        worst_relative = variant

            if worst_relative:
                report.append(f"2. 性能下降最多: {worst_relative} ({worst_ratio:.1%} of full model)")

        # 找出训练时间差异
        time_data = []
        for variant, result_list in comparison.results.items():
            if result_list:
                avg_time = np.mean([r.training_time for r in result_list])
                time_data.append((variant, avg_time))

        if time_data:
            fastest = min(time_data, key=lambda x: x[1])
            slowest = max(time_data, key=lambda x: x[1])

            report.append(f"3. 最快训练: {fastest[0]} ({fastest[1]:.1f}s)")
            report.append(f"4. 最慢训练: {slowest[0]} ({slowest[1]:.1f}s)")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def export_for_latex(self, comparison: AblationComparison,
                         output_path: str = None) -> str:
        """导出为LaTeX表格格式"""
        if not output_path:
            output_path = self.results_dir / f"{comparison.experiment_id}_table.tex"

        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Ablation Study Results}")
        latex.append("\\label{tab:ablation_results}")
        latex.append("\\begin{tabular}{lcccc}")
        latex.append("\\toprule")
        latex.append("Variant & MSE (mean ± std) & MAE & Training Time (s) & Significant \\\\")
        latex.append("\\midrule")

        for variant, stats in comparison.summary_stats.items():
            # 获取MAE（如果存在）
            mae = comparison.results.get(variant, [{}])[0].test_metrics.get('mae', 0)

            # 检查统计显著性
            sig_marker = ""
            if variant in comparison.statistical_tests:
                if (comparison.statistical_tests[variant]['t_test']['significant'] or
                        comparison.statistical_tests[variant]['wilcoxon']['significant']):
                    sig_marker = "\\checkmark"
                else:
                    sig_marker = "\\times"

            # 获取平均训练时间
            avg_time = np.mean([r.training_time for r in comparison.results.get(variant, [])])

            latex.append(f"{variant:<20} & {stats['mean']:.4f} ± {stats['std']:.4f} & "
                         f"{mae:.4f} & {avg_time:.1f} & {sig_marker} \\\\")

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

        latex_str = "\n".join(latex)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_str)

        self.logger.info(f"📋 LaTeX表格保存到: {output_path}")

        return latex_str


# 使用示例
def main():
    """消融实验主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="运行消融实验")
    parser.add_argument("--config", type=str, default="./config.yaml",
                        help="基础配置文件路径")
    parser.add_argument("--data", type=str, default="./data/ETTh1.csv",
                        help="数据文件路径")
    parser.add_argument("--variants", type=str, nargs='+',
                        choices=[v.value for v in AblationVariant],
                        help="要测试的变体列表")
    parser.add_argument("--name", type=str, help="实验名称")
    parser.add_argument("--epochs", type=int, default=50,
                        help="每个变体的训练轮数")
    parser.add_argument("--runs", type=int, default=3,
                        help="每个变体的运行次数")

    args = parser.parse_args()

    # 创建消融实验管理器
    manager = AblationStudyManager(args.config)

    # 转换变体参数
    variants = None
    if args.variants:
        variants = [AblationVariant(v) for v in args.variants]

    # 运行实验
    comparison = manager.run_ablation_experiment(
        variants=variants,
        data_path=args.data,
        experiment_name=args.name
    )

    # 生成报告
    report = manager.generate_report(comparison)
    print(report)

    # 导出LaTeX表格
    manager.export_for_latex(comparison)

    return comparison


if __name__ == "__main__":
    comparison = main()