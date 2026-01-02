"""
STAR-Forecast Web界面
使用Streamlit构建交互式可视化界面
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import yaml
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys
import asyncio
from typing import Dict, List, Optional, Any
import logging

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from models.istr import ISTRNetwork
from models.predictor import create_predictor
from data.dataloader import ETTh1Dataset, create_dataloaders
from client.api_client import AgentLightningClient
from agents.autogen_system import AutoGenMultiAgentSystem

# 设置页面配置
st.set_page_config(
    page_title="STAR-Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class STARForecastUI:
    """STAR-Forecast Web界面"""

    def __init__(self):
        self.config = self.load_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化状态
        self.model = None
        self.predictor = None
        self.dataset = None
        self.agent_client = None
        self.autogen_system = None

        # 会话状态
        if 'predictions' not in st.session_state:
            st.session_state.predictions = []
        if 'agent_decisions' not in st.session_state:
            st.session_state.agent_decisions = []
        if 'training_history' not in st.session_state:
            st.session_state.training_history = []
        if 'selected_sample' not in st.session_state:
            st.session_state.selected_sample = 0

    def load_config(self) -> Dict[str, Any]:
        """加载配置"""
        config_path = Path("./config.yaml")
        if not config_path.exists():
            st.error("配置文件不存在: config.yaml")
            return {}

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def setup_sidebar(self):
        """设置侧边栏"""
        with st.sidebar:
            st.title("⚙️ 控制面板")

            # 模型加载部分
            st.header("📦 模型管理")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("加载模型", use_container_width=True):
                    self.load_models()

            with col2:
                if st.button("重新加载数据", use_container_width=True):
                    self.load_data()

            # 数据选择
            st.header("📊 数据选择")

            if self.dataset:
                sample_idx = st.slider(
                    "选择样本索引",
                    0,
                    len(self.dataset) - 1,
                    st.session_state.selected_sample,
                    key="sample_slider"
                )
                st.session_state.selected_sample = sample_idx

                # 显示样本信息
                st.info(f"样本 {sample_idx}/{len(self.dataset) - 1}")

            # 预测设置
            st.header("🔮 预测设置")

            self.num_predictions = st.slider(
                "预测步数",
                1,
                48,
                self.config['data']['pred_len'],
                key="pred_len_slider"
            )

            self.batch_size = st.slider(
                "批处理大小",
                1,
                64,
                16,
                key="batch_size_slider"
            )

            # 智能体设置
            st.header("🤖 智能体控制")

            self.agent_enabled = st.checkbox("启用智能体", value=True)

            if self.agent_enabled:
                self.agent_check_interval = st.slider(
                    "智能体检查间隔",
                    10,
                    1000,
                    self.config['autogen']['trigger']['check_interval'],
                    step=10
                )

            # 操作按钮
            st.header("🚀 操作")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("运行预测", type="primary", use_container_width=True):
                    self.run_prediction()

            with col2:
                if st.button("调用智能体", type="secondary", use_container_width=True):
                    self.call_agent()

            if st.button("重置所有", type="secondary", use_container_width=True):
                st.session_state.predictions = []
                st.session_state.agent_decisions = []
                st.rerun()

            # 信息显示
            st.header("📈 状态信息")

            if self.model:
                st.success("✅ 模型已加载")
            else:
                st.warning("⚠️ 模型未加载")

            if self.dataset:
                st.success("✅ 数据已加载")
            else:
                st.warning("⚠️ 数据未加载")

    def load_models(self):
        """加载模型"""
        with st.spinner("加载模型中..."):
            try:
                # 加载检查点
                checkpoint_dir = Path("./checkpoints")
                checkpoints = list(checkpoint_dir.glob("*.pth"))

                if not checkpoints:
                    st.error("未找到模型检查点")
                    return

                # 选择最新检查点
                latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                checkpoint = torch.load(latest_checkpoint, map_location=self.device)

                # 创建模型
                self.model = ISTRNetwork(self.config).to(self.device)
                self.predictor = create_predictor(self.config).to(self.device)

                # 加载权重
                self.model.load_state_dict(checkpoint['istr_state_dict'])
                self.predictor.load_state_dict(checkpoint['predictor_state_dict'])

                # 设置为评估模式
                self.model.eval()
                self.predictor.eval()

                st.success(f"✅ 模型加载成功: {latest_checkpoint.name}")

            except Exception as e:
                st.error(f"❌ 模型加载失败: {e}")

    def load_data(self):
        """加载数据"""
        with st.spinner("加载数据中..."):
            try:
                data_path = self.config['data']['data_path']

                if not Path(data_path).exists():
                    st.error(f"数据文件不存在: {data_path}")
                    return

                # 创建数据集
                self.dataset = ETTh1Dataset(
                    data_path,
                    seq_len=self.config['data']['seq_len'],
                    pred_len=self.config['data']['pred_len'],
                    split='test',
                    scale=True
                )

                st.success(f"✅ 数据加载成功: {len(self.dataset)} 个样本")

            except Exception as e:
                st.error(f"❌ 数据加载失败: {e}")

    def initialize_agents(self):
        """初始化智能体"""
        if not self.agent_client:
            try:
                self.agent_client = AgentLightningClient(
                    base_url=self.config['agent_lightning']['client']['base_url'],
                    client_id=f"web_ui_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    timeout=self.config['agent_lightning']['client']['timeout']
                )
                st.success("✅ Agent Lightning客户端初始化成功")
            except Exception as e:
                st.warning(f"⚠️ Agent Lightning客户端初始化失败: {e}")

        if not self.autogen_system:
            try:
                self.autogen_system = AutoGenMultiAgentSystem(self.config)
                st.success("✅ AutoGen系统初始化成功")
            except Exception as e:
                st.warning(f"⚠️ AutoGen系统初始化失败: {e}")

    def run_prediction(self):
        """运行预测"""
        if not self.model or not self.dataset:
            st.error("请先加载模型和数据")
            return

        with st.spinner("运行预测中..."):
            try:
                # 获取样本
                sample_idx = st.session_state.selected_sample
                x, y_true = self.dataset[sample_idx]

                # 转换为批量
                x = x.unsqueeze(0).to(self.device)
                y_true = y_true.unsqueeze(0).to(self.device)

                # 运行预测
                with torch.no_grad():
                    # ISTR特征提取
                    features = self.model(x)

                    # 预测
                    y_pred = self.predictor(features)

                    # 提取特征供分析
                    feature_analysis = self.model.extract_features_for_analysis(x)

                # 转换为numpy
                x_np = x.squeeze().cpu().numpy()
                y_true_np = y_true.squeeze().cpu().numpy()
                y_pred_np = y_pred.squeeze().cpu().numpy()

                # 反标准化（如果数据有标准化）
                if hasattr(self.dataset, 'scaler'):
                    # 构建完整序列（只取OT特征）
                    full_actual = np.concatenate([x_np[:, -1], y_true_np])
                    full_pred = np.concatenate([x_np[:, -1], y_pred_np])

                    # 反标准化
                    full_actual = self.dataset.inverse_transform(full_actual)
                    full_pred = self.dataset.inverse_transform(full_pred)

                    actual = full_actual
                    predicted = full_pred[-len(y_pred_np):]
                else:
                    actual = np.concatenate([x_np[:, -1], y_true_np])
                    predicted = y_pred_np

                # 计算指标
                metrics = self.calculate_metrics(actual[-len(predicted):], predicted)

                # 保存到会话状态
                prediction_result = {
                    'sample_idx': sample_idx,
                    'actual': actual,
                    'predicted': predicted,
                    'metrics': metrics,
                    'feature_analysis': feature_analysis,
                    'timestamp': datetime.now()
                }

                st.session_state.predictions.append(prediction_result)

                st.success(f"✅ 预测完成 - MSE: {metrics['mse']:.4f}")

            except Exception as e:
                st.error(f"❌ 预测失败: {e}")

    def call_agent(self):
        """调用智能体"""
        if not self.agent_enabled:
            st.warning("智能体功能已禁用")
            return

        if not self.agent_client or len(st.session_state.predictions) == 0:
            st.error("请先初始化智能体并运行预测")
            return

        with st.spinner("智能体分析中..."):
            try:
                # 获取最新的预测结果
                latest_pred = st.session_state.predictions[-1]

                # 准备上下文
                context = {
                    'features': latest_pred['feature_analysis'],
                    'metrics': latest_pred['metrics'],
                    'current_params': {
                        'spectral_threshold': 0.5,
                        'laplacian_weight': 0.01
                    },
                    'training_info': {
                        'sample_idx': latest_pred['sample_idx'],
                        'timestamp': latest_pred['timestamp'].isoformat()
                    }
                }

                # 调用智能体
                decision = self.agent_client.get_decision(context)

                # 保存决策
                decision_record = {
                    'context': context,
                    'decision': decision,
                    'timestamp': datetime.now()
                }

                st.session_state.agent_decisions.append(decision_record)

                # 显示决策结果
                st.success("✅ 智能体分析完成")

                # 显示决策详情
                with st.expander("查看智能体决策详情"):
                    st.json(decision)

            except Exception as e:
                st.error(f"❌ 智能体调用失败: {e}")

    def calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """计算评估指标"""
        # 确保长度一致
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]

        # 计算指标
        mse = np.mean((actual - predicted) ** 2)
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(mse)

        # MAPE（避免除以0）
        epsilon = 1e-8
        mape = np.mean(np.abs((actual - predicted) / (actual + epsilon))) * 100

        # R²分数
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + epsilon))

        # 方向准确性
        if len(actual) > 1:
            actual_dir = np.sign(actual[1:] - actual[:-1])
            pred_dir = np.sign(predicted[1:] - predicted[:-1])
            dir_acc = np.mean(actual_dir == pred_dir) * 100
        else:
            dir_acc = 0.0

        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2),
            'direction_accuracy': float(dir_acc)
        }

    def create_prediction_plot(self, actual: np.ndarray, predicted: np.ndarray,
                               title: str = "预测结果") -> go.Figure:
        """创建预测结果图"""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f"{title}", "预测误差"),
            vertical_spacing=0.15
        )

        # 时间轴
        time_actual = list(range(len(actual)))
        time_pred = list(range(len(actual) - len(predicted), len(actual)))

        # 第一张图：实际 vs 预测
        fig.add_trace(
            go.Scatter(
                x=time_actual,
                y=actual,
                mode='lines',
                name='实际值',
                line=dict(color='#1f77b4', width=2)
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=time_pred,
                y=predicted,
                mode='lines+markers',
                name='预测值',
                line=dict(color='#ff7f0e', width=2, dash='dash')
            ),
            row=1, col=1
        )

        # 添加预测区间阴影
        if len(predicted) > 0:
            # 简单置信区间（基于历史误差）
            error_std = np.std(actual[-len(predicted):] - predicted)
            upper_bound = predicted + 1.96 * error_std
            lower_bound = predicted - 1.96 * error_std

            fig.add_trace(
                go.Scatter(
                    x=time_pred + time_pred[::-1],
                    y=np.concatenate([upper_bound, lower_bound[::-1]]),
                    fill='toself',
                    fillcolor='rgba(255, 127, 14, 0.2)',
                    line=dict(color='rgba(255, 255, 255, 0)'),
                    name='95% 置信区间',
                    showlegend=True
                ),
                row=1, col=1
            )

        # 第二张图：误差
        error = actual[-len(predicted):] - predicted
        fig.add_trace(
            go.Scatter(
                x=time_pred,
                y=error,
                mode='lines',
                name='误差',
                line=dict(color='#2ca02c', width=2)
            ),
            row=2, col=1
        )

        # 添加零线
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

        # 更新布局
        fig.update_layout(
            height=600,
            showlegend=True,
            template="plotly_white",
            hovermode="x unified"
        )

        fig.update_xaxes(title_text="时间步", row=2, col=1)
        fig.update_yaxes(title_text="数值", row=1, col=1)
        fig.update_yaxes(title_text="误差", row=2, col=1)

        return fig

    def create_metrics_dashboard(self, metrics: Dict[str, float]) -> go.Figure:
        """创建指标仪表盘"""
        fig = go.Figure()

        # 准备数据
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())

        # 创建条形图
        fig.add_trace(go.Bar(
            x=metric_names,
            y=metric_values,
            text=[f'{v:.4f}' for v in metric_values],
            textposition='auto',
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        ))

        # 更新布局
        fig.update_layout(
            title="评估指标",
            template="plotly_white",
            height=400,
            showlegend=False
        )

        fig.update_yaxes(title_text="指标值")

        return fig

    def create_feature_analysis_plot(self, features: Dict[str, Any]) -> go.Figure:
        """创建特征分析图"""
        if not features or 'statistics' not in features:
            return go.Figure()

        stats = features['statistics']

        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("统计特征", "自相关", "频域特征", "自适应参数"),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # 1. 统计特征
        if 'mean' in stats and 'std' in stats:
            mean_vals = stats['mean']
            std_vals = stats['std']

            fig.add_trace(
                go.Bar(
                    x=[f'特征{i}' for i in range(len(mean_vals))],
                    y=mean_vals,
                    error_y=dict(type='data', array=std_vals, visible=True),
                    name='均值±标准差'
                ),
                row=1, col=1
            )

        # 2. 自相关
        if 'autocorrelation' in stats:
            autocorr = stats['autocorrelation']
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(autocorr))),
                    y=autocorr,
                    mode='lines+markers',
                    name='自相关'
                ),
                row=1, col=2
            )

        # 3. 频域特征
        if 'frequency' in features:
            freq_features = features['frequency']
            if 'dominant_frequency' in freq_features:
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=freq_features['dominant_frequency'],
                        title={'text': "主导频率"},
                        domain={'row': 0, 'column': 0},
                        gauge={'axis': {'range': [0, 50]}}
                    ),
                    row=2, col=1
                )

        # 4. 自适应参数
        if 'adaptive_parameters' in features:
            adaptive_params = features['adaptive_parameters']

            param_names = list(adaptive_params.keys())
            param_values = list(adaptive_params.values())

            fig.add_trace(
                go.Bar(
                    x=param_names,
                    y=param_values,
                    name='自适应参数'
                ),
                row=2, col=2
            )

        # 更新布局
        fig.update_layout(
            height=600,
            showlegend=False,
            template="plotly_white"
        )

        return fig

    def display_prediction_results(self):
        """显示预测结果"""
        if not st.session_state.predictions:
            return

        st.header("📊 预测结果")

        # 显示最近的预测
        latest_pred = st.session_state.predictions[-1]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("MSE", f"{latest_pred['metrics']['mse']:.4f}")
        with col2:
            st.metric("MAE", f"{latest_pred['metrics']['mae']:.4f}")
        with col3:
            st.metric("R²", f"{latest_pred['metrics']['r2']:.4f}")

        # 显示预测图
        st.subheader("预测可视化")

        fig = self.create_prediction_plot(
            latest_pred['actual'],
            latest_pred['predicted'],
            f"样本 {latest_pred['sample_idx']} 预测结果"
        )

        st.plotly_chart(fig, use_container_width=True)

        # 显示指标仪表盘
        st.subheader("评估指标")

        metrics_fig = self.create_metrics_dashboard(latest_pred['metrics'])
        st.plotly_chart(metrics_fig, use_container_width=True)

        # 显示特征分析
        if 'feature_analysis' in latest_pred:
            st.subheader("特征分析")

            feature_fig = self.create_feature_analysis_plot(latest_pred['feature_analysis'])
            st.plotly_chart(feature_fig, use_container_width=True)

        # 显示历史预测
        if len(st.session_state.predictions) > 1:
            st.subheader("历史预测记录")

            history_df = pd.DataFrame([
                {
                    '样本索引': p['sample_idx'],
                    'MSE': p['metrics']['mse'],
                    'MAE': p['metrics']['mae'],
                    'R²': p['metrics']['r2'],
                    '时间': p['timestamp'].strftime('%H:%M:%S')
                }
                for p in st.session_state.predictions[-10:]  # 显示最近10条
            ])

            st.dataframe(history_df, use_container_width=True)

    def display_agent_decisions(self):
        """显示智能体决策"""
        if not st.session_state.agent_decisions:
            return

        st.header("🤖 智能体决策")

        # 显示最近的决策
        latest_decision = st.session_state.agent_decisions[-1]
        decision = latest_decision['decision']

        # 决策概览
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("动作", decision.get('action', 'N/A'))
        with col2:
            st.metric("语义奖励", f"{decision.get('semantic_reward', 0):.3f}")
        with col3:
            if decision.get('fallback', False):
                st.warning("备用决策")
            else:
                st.success("智能体决策")

        # 显示参数调整
        if 'parameters' in decision and decision['parameters']:
            st.subheader("参数调整建议")

            params = decision['parameters']
            param_df = pd.DataFrame({
                '参数': list(params.keys()),
                '建议值': list(params.values())
            })

            st.dataframe(param_df, use_container_width=True)

        # 显示AutoGen对话摘要
        if 'autogen_conversation' in decision:
            conversation = decision['autogen_conversation']

            with st.expander("AutoGen对话详情"):
                st.write(f"对话ID: {conversation.get('conversation_id', 'N/A')}")
                st.write(f"共识程度: {conversation.get('consensus_level', 0):.2%}")

                if 'summary' in conversation:
                    st.write("对话摘要:")
                    st.write(conversation['summary'])

        # 显示历史决策
        if len(st.session_state.agent_decisions) > 1:
            st.subheader("历史决策记录")

            decision_history = []
            for i, d in enumerate(st.session_state.agent_decisions[-5:]):  # 显示最近5条
                decision_data = d['decision']
                decision_history.append({
                    '序号': i + 1,
                    '动作': decision_data.get('action', 'N/A'),
                    '参数数量': len(decision_data.get('parameters', {})),
                    '奖励': f"{decision_data.get('semantic_reward', 0):.3f}",
                    '时间': d['timestamp'].strftime('%H:%M:%S')
                })

            history_df = pd.DataFrame(decision_history)
            st.dataframe(history_df, use_container_width=True)

    def display_model_info(self):
        """显示模型信息"""
        if not self.model:
            return

        st.header("🧠 模型信息")

        # 计算模型参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("总参数", f"{total_params:,}")
        with col2:
            st.metric("可训练参数", f"{trainable_params:,}")

        # 显示架构信息
        with st.expander("查看架构详情"):
            # ISTR网络信息
            st.subheader("ISTR网络")

            if hasattr(self.model, 'config'):
                istr_config = self.model.config['istr']

                info_df = pd.DataFrame({
                    '参数': ['输入维度', '隐藏维度', 'TCN块数', '谱门控', '拉普拉斯正则化'],
                    '值': [
                        istr_config['input_dim'],
                        istr_config['hidden_dim'],
                        len(istr_config['tcn']['kernel_sizes']),
                        '启用' if istr_config['spectral_gate']['enabled'] else '禁用',
                        '启用' if istr_config['laplacian']['enabled'] else '禁用'
                    ]
                })

                st.dataframe(info_df, use_container_width=True)

        # 显示训练状态
        if hasattr(self.model, 'adaptation_count'):
            st.subheader("自适应状态")
            st.write(f"参数调整次数: {self.model.adaptation_count.item()}")

            if hasattr(self.model, 'adaptive_params'):
                adaptive_df = pd.DataFrame({
                    '参数': list(self.model.adaptive_params.keys()),
                    '当前值': [p.item() for p in self.model.adaptive_params.values()]
                })

                st.dataframe(adaptive_df, use_container_width=True)

    def display_data_info(self):
        """显示数据信息"""
        if not self.dataset:
            return

        st.header("📈 数据信息")

        # 获取数据统计
        try:
            data_path = self.config['data']['data_path']
            df = pd.read_csv(data_path)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("总样本数", f"{len(df):,}")
            with col2:
                st.metric("特征数量", f"{len(df.columns) - 1}")  # 减去日期列
            with col3:
                st.metric("数据范围", f"{df.iloc[0, 0]} 到 {df.iloc[-1, 0]}")

            # 显示数据预览
            with st.expander("查看数据预览"):
                st.dataframe(df.head(10), use_container_width=True)

            # 显示特征统计
            st.subheader("特征统计")

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            stats_df = df[numeric_cols].describe().T

            st.dataframe(stats_df, use_container_width=True)

        except Exception as e:
            st.error(f"数据信息加载失败: {e}")

    def display_training_monitor(self):
        """显示训练监控"""
        st.header("🏋️ 训练监控")

        # 训练状态指示器
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("预测次数", len(st.session_state.predictions))
        with col2:
            st.metric("智能体决策", len(st.session_state.agent_decisions))
        with col3:
            if st.session_state.predictions:
                latest_mse = st.session_state.predictions[-1]['metrics']['mse']
                st.metric("最新MSE", f"{latest_mse:.4f}")

        # 训练历史图表
        if len(st.session_state.predictions) > 1:
            st.subheader("训练历史")

            # 提取历史指标
            history = st.session_state.predictions
            epochs = list(range(len(history)))
            mse_values = [h['metrics']['mse'] for h in history]
            mae_values = [h['metrics']['mae'] for h in history]

            # 创建图表
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=epochs,
                y=mse_values,
                mode='lines+markers',
                name='MSE',
                line=dict(color='#1f77b4', width=2)
            ))

            fig.add_trace(go.Scatter(
                x=epochs,
                y=mae_values,
                mode='lines+markers',
                name='MAE',
                line=dict(color='#ff7f0e', width=2)
            ))

            fig.update_layout(
                title="训练指标变化",
                xaxis_title="预测次数",
                yaxis_title="指标值",
                template="plotly_white",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

    def run(self):
        """运行Web界面"""
        # 页面标题
        st.title("🚀 STAR-Forecast: 神经-符号-强化三重协同时序预测")
        st.markdown("---")

        # 初始化智能体
        self.initialize_agents()

        # 侧边栏
        self.setup_sidebar()

        # 主内容区域
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 预测结果",
            "🤖 智能体分析",
            "🧠 模型信息",
            "📈 数据监控"
        ])

        with tab1:
            self.display_prediction_results()

        with tab2:
            self.display_agent_decisions()

        with tab3:
            self.display_model_info()

        with tab4:
            self.display_data_info()
            st.markdown("---")
            self.display_training_monitor()

        # 页脚
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center'>
                <p><b>STAR-Forecast</b> © 2024 梁德隆 - 硕士论文实现</p>
                <p>神经-符号-强化三重协同自适应时序预测框架</p>
            </div>
            """,
            unsafe_allow_html=True
        )


def main():
    """主函数"""
    # 创建UI实例
    ui = STARForecastUI()

    # 运行UI
    ui.run()


if __name__ == "__main__":
    main()