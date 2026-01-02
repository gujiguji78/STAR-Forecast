"""
STAR-Forecast 完整三重协同架构
神经-符号-强化三重协同预测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import math
import random


class NeuralTransformerModule(nn.Module):
    """神经模块：基于Transformer的深度学习预测"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.d_model = config.get('d_model', 512)
        self.n_heads = config.get('n_heads', 8)
        self.num_layers = config.get('e_layers', 2)
        self.dropout = config.get('dropout', 0.1)
        self.seq_len = config.get('seq_len', 96)
        self.pred_len = config.get('pred_len', 24)
        self.input_dim = config.get('enc_in', 7)

        # 输入嵌入
        self.input_projection = nn.Linear(self.input_dim, self.d_model)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, self.num_layers)

        # 位置编码
        self.positional_encoding = PositionalEncoding(self.d_model, max_len=5000)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 1)  # 预测单变量
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # 输入投影
        x_proj = self.input_projection(x) * math.sqrt(self.d_model)

        # 位置编码
        x_encoded = self.positional_encoding(x_proj)

        # Transformer处理
        transformer_output = self.transformer(x_encoded)

        # 使用最后一个时间步的特征
        context = transformer_output[:, -1, :]

        # 解码为预测序列
        predictions = []
        current_context = context

        for _ in range(self.pred_len):
            pred = self.decoder(current_context)
            predictions.append(pred.unsqueeze(1))

            # 更新上下文（简单循环）
            current_context = current_context + self.decoder[0](current_context)

        predictions = torch.cat(predictions, dim=1)
        return predictions


class SymbolicRuleModule(nn.Module):
    """符号模块：基于规则的知识推理"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.seq_len = config.get('seq_len', 96)
        self.pred_len = config.get('pred_len', 24)
        self.num_rules = config.get('num_rules', 5)

        # 规则网络：每条规则是一个简单的模式识别器
        self.rule_nets = nn.ModuleList([
            self._create_rule_network() for _ in range(self.num_rules)
        ])

        # 规则权重（可学习）
        self.rule_weights = nn.Parameter(torch.ones(self.num_rules) / self.num_rules)

        # 规则选择器
        self.rule_selector = nn.Sequential(
            nn.Linear(self.seq_len, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_rules),
            nn.Softmax(dim=-1)
        )

        # 知识库：存储常见模式
        self.register_buffer('knowledge_base',
                             torch.randn(10, self.seq_len) * 0.1)  # 10个基本模式

    def _create_rule_network(self):
        return nn.Sequential(
            nn.Linear(self.seq_len, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.pred_len)
        )

    def apply_rules(self, x: torch.Tensor) -> torch.Tensor:
        """应用符号规则"""
        batch_size = x.size(0)

        # 计算规则权重
        rule_scores = self.rule_selector(x.squeeze(-1))

        # 应用每条规则
        rule_outputs = []
        for i, rule_net in enumerate(self.rule_nets):
            rule_out = rule_net(x.squeeze(-1))
            rule_outputs.append(rule_out.unsqueeze(1))

        # 组合规则输出
        rule_outputs = torch.cat(rule_outputs, dim=1)  # [B, num_rules, pred_len]
        rule_scores = rule_scores.unsqueeze(-1)  # [B, num_rules, 1]

        # 加权平均
        symbolic_output = torch.sum(rule_outputs * rule_scores, dim=1)

        return symbolic_output.unsqueeze(-1)

    def extract_patterns(self, x: torch.Tensor) -> List[Dict[str, Any]]:
        """提取时间序列模式"""
        patterns = []
        x_np = x.squeeze(-1).cpu().numpy()

        for i in range(x.size(0)):
            sequence = x_np[i]

            # 检测趋势
            trend = "上升" if sequence[-1] > sequence[0] else "下降"

            # 检测季节性（简单版本）
            if len(sequence) >= 24:
                seasonal_strength = np.std(sequence[-24:]) / (np.std(sequence) + 1e-8)
            else:
                seasonal_strength = 0

            # 检测异常点
            mean_val = np.mean(sequence)
            std_val = np.std(sequence)
            anomalies = np.sum(np.abs(sequence - mean_val) > 2 * std_val)

            patterns.append({
                'trend': trend,
                'seasonal_strength': float(seasonal_strength),
                'anomalies': int(anomalies),
                'mean': float(mean_val),
                'std': float(std_val)
            })

        return patterns

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """前向传播"""
        symbolic_pred = self.apply_rules(x)
        patterns = self.extract_patterns(x)

        return symbolic_pred, patterns


class ReinforcementModule(nn.Module):
    """强化学习模块：自适应决策优化"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.state_dim = config.get('seq_len', 96)
        self.action_dim = 3  # 调整三个模块的权重
        self.pred_len = config.get('pred_len', 24)

        # 策略网络（Actor）
        self.policy_net = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
            nn.Softmax(dim=-1)
        )

        # 价值网络（Critic）
        self.value_net = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # 经验回放缓冲区
        self.memory = []
        self.memory_capacity = 1000

        # RL参数
        self.gamma = 0.99
        self.epsilon = 0.1
        self.learning_rate = 0.001

        # 优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # 当前权重
        self.current_weights = torch.tensor([0.4, 0.3, 0.3])  # 神经:符号:RL

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """根据状态选择动作"""
        if random.random() < self.epsilon:
            # 探索：随机选择权重
            weights = torch.rand(3)
            weights = weights / weights.sum()
            return weights
        else:
            # 利用：使用策略网络
            with torch.no_grad():
                action_probs = self.policy_net(state)
                return action_probs

    def update_weights(self, state: torch.Tensor, neural_loss: float,
                       symbolic_loss: float, rl_loss: float) -> torch.Tensor:
        """更新模块权重"""
        # 计算每个模块的性能得分（损失越小得分越高）
        losses = torch.tensor([neural_loss, symbolic_loss, rl_loss])
        scores = 1.0 / (losses + 1e-8)
        scores = scores / scores.sum()

        # 使用策略梯度更新
        action = self.get_action(state)
        log_prob = torch.log(action + 1e-8)

        # 计算优势函数
        with torch.no_grad():
            state_value = self.value_net(state)
            reward = scores.mean()
            advantage = reward - state_value

        # 策略梯度损失
        policy_loss = -(log_prob * advantage).mean()

        # 价值函数损失
        value_loss = F.mse_loss(state_value, reward)

        # 总损失
        total_loss = policy_loss + 0.5 * value_loss

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # 更新当前权重
        self.current_weights = action.detach()

        return self.current_weights

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播（返回当前权重）"""
        return self.current_weights


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class STARForecastPredictor(nn.Module):
    """完整的STAR三重协同预测模型"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.config = config
        self.seq_len = config.get('seq_len', 96)
        self.pred_len = config.get('pred_len', 24)

        # 初始化三个模块
        self.neural_module = NeuralTransformerModule(config)
        self.symbolic_module = SymbolicRuleModule(config)
        self.reinforcement_module = ReinforcementModule(config)

        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softmax(dim=-1)
        )

        # 元学习器：学习如何组合三个模块
        self.meta_learner = nn.Sequential(
            nn.Linear(self.seq_len * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 输出三个权重
            nn.Softmax(dim=-1)
        )

        print(f"✅ 初始化完整STAR架构:")
        print(f"   神经模块参数: {sum(p.numel() for p in self.neural_module.parameters()):,}")
        print(f"   符号模块参数: {sum(p.numel() for p in self.symbolic_module.parameters()):,}")
        print(f"   强化模块参数: {sum(p.numel() for p in self.reinforcement_module.parameters()):,}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """前向传播"""
        batch_size = x.shape[0]

        # 1. 神经模块预测
        neural_pred = self.neural_module(x)  # [B, pred_len, 1]

        # 2. 符号模块预测（同时提取模式）
        symbolic_pred, patterns = self.symbolic_module(x)

        # 3. 强化模块决策
        state = x.mean(dim=2).squeeze(-1)  # 平均特征作为状态
        rl_weights = self.reinforcement_module(state)  # [B, 3]

        # 4. 元学习器调整权重
        # 准备元特征：三个模块的中间特征
        neural_feat = self.neural_module.transformer(x)[:, -1, :]
        symbolic_feat = symbolic_pred.mean(dim=1)

        # 合并特征
        meta_features = torch.cat([
            neural_feat,
            symbolic_feat,
            rl_weights
        ], dim=1)

        meta_weights = self.meta_learner(meta_features)  # [B, 3]

        # 5. 三重协同融合
        # 扩展维度以便广播
        neural_pred_exp = neural_pred
        symbolic_pred_exp = symbolic_pred

        # 计算加权预测
        weights = meta_weights.unsqueeze(-1).unsqueeze(-1)  # [B, 3, 1, 1]

        # 组合预测
        combined_pred = (weights[:, 0] * neural_pred_exp +
                         weights[:, 1] * symbolic_pred_exp)

        # 最终输出
        final_output = combined_pred

        # 收集诊断信息
        diagnostics = {
            'neural_weight': meta_weights[:, 0].mean().item(),
            'symbolic_weight': meta_weights[:, 1].mean().item(),
            'rl_weight': meta_weights[:, 2].mean().item(),
            'patterns': patterns,
            'neural_pred_mean': neural_pred.mean().item(),
            'symbolic_pred_mean': symbolic_pred.mean().item()
        }

        return final_output, diagnostics

    def update_reinforcement(self, x: torch.Tensor, losses: Dict[str, float]):
        """更新强化学习模块"""
        state = x.mean(dim=2).squeeze(-1)

        # 更新权重
        new_weights = self.reinforcement_module.update_weights(
            state,
            losses.get('neural', 0.1),
            losses.get('symbolic', 0.1),
            losses.get('rl', 0.1)
        )

        return new_weights