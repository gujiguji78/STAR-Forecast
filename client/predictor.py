"""
多尺度预测头网络
融合多种时间尺度的特征进行预测
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import math


class TemporalAttention(nn.Module):
    """时间注意力机制"""

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 查询、键、值投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # 缩放因子
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入序列 [batch_size, seq_len, hidden_dim]
            mask: 注意力掩码 [batch_size, seq_len, seq_len]

        Returns:
            注意力加权输出 [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # 投影到查询、键、值
        q = self.q_proj(x)  # [batch, seq_len, hidden_dim]
        k = self.k_proj(x)  # [batch, seq_len, hidden_dim]
        v = self.v_proj(x)  # [batch, seq_len, hidden_dim]

        # 重塑为多头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch, heads, seq_len, seq_len]

        # 应用掩码（如果有）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # 注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 注意力输出
        attn_output = torch.matmul(attn_weights, v)  # [batch, heads, seq_len, head_dim]

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )

        # 输出投影
        output = self.out_proj(attn_output)
        output = self.proj_dropout(output)

        return output


class MultiScaleConvolution(nn.Module):
    """多尺度卷积模块"""

    def __init__(self, hidden_dim: int, kernel_sizes: List[int] = None, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.kernel_sizes = kernel_sizes or [1, 3, 5, 7]

        # 多尺度卷积层
        self.conv_layers = nn.ModuleList()
        for kernel_size in self.kernel_sizes:
            conv = nn.Sequential(
                nn.Conv1d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // 2,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(
                    in_channels=hidden_dim // 2,
                    out_channels=hidden_dim // 4,
                    kernel_size=1
                )
            )
            self.conv_layers.append(conv)

        # 注意力融合
        self.attention = nn.Sequential(
            nn.Linear(len(self.kernel_sizes) * (hidden_dim // 4), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(self.kernel_sizes)),
            nn.Softmax(dim=-1)
        )

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim // 4, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入序列 [batch_size, seq_len, hidden_dim]

        Returns:
            多尺度融合特征 [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # 转置以适应卷积 [batch, hidden_dim, seq_len]
        x_t = x.transpose(1, 2)

        # 多尺度特征提取
        multi_scale_features = []
        for conv in self.conv_layers:
            features = conv(x_t)  # [batch, hidden_dim//4, seq_len]
            features = features.transpose(1, 2)  # [batch, seq_len, hidden_dim//4]
            multi_scale_features.append(features)

        # 拼接多尺度特征
        concatenated = torch.cat(multi_scale_features, dim=-1)  # [batch, seq_len, hidden_dim]

        # 注意力加权融合
        attention_weights = self.attention(concatenated)  # [batch, seq_len, num_scales]

        # 应用注意力权重
        weighted_features = []
        for i, features in enumerate(multi_scale_features):
            weight = attention_weights[..., i:i + 1]  # [batch, seq_len, 1]
            weighted = features * weight
            weighted_features.append(weighted)

        # 求和融合
        fused_features = sum(weighted_features)  # [batch, seq_len, hidden_dim//4]

        # 输出投影
        output = self.output_proj(fused_features)  # [batch, seq_len, hidden_dim]

        return output


class FrequencyDomainPredictor(nn.Module):
    """频域预测模块"""

    def __init__(self, hidden_dim: int, pred_len: int, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.pred_len = pred_len

        # 频域特征提取
        self.freq_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # 逆傅里叶变换投影
        self.ifft_proj = nn.Sequential(
            nn.Linear(hidden_dim // 2, pred_len * 2),  # 实部和虚部
            nn.Tanh()
        )

        # 时域精调
        self.temporal_refine = nn.Sequential(
            nn.Conv1d(pred_len, pred_len * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(pred_len * 2, pred_len, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        频域预测

        Args:
            x: 输入特征 [batch_size, seq_len, hidden_dim]

        Returns:
            预测序列 [batch_size, pred_len]
        """
        batch_size, seq_len, _ = x.shape

        # 傅里叶变换
        x_fft = torch.fft.rfft(x, dim=1)  # [batch, freq_bins, hidden_dim]

        # 提取幅度和相位
        magnitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)

        # 拼接频域特征
        freq_features = torch.cat([magnitude, phase], dim=-1)  # [batch, freq_bins, hidden_dim*2]
        freq_features = freq_features.mean(dim=1)  # [batch, hidden_dim*2]

        # 频域编码
        encoded = self.freq_encoder(freq_features)  # [batch, hidden_dim//2]

        # 生成频域预测
        freq_pred = self.ifft_proj(encoded)  # [batch, pred_len*2]

        # 分割实部和虚部
        real = freq_pred[:, :self.pred_len].unsqueeze(2)  # [batch, pred_len, 1]
        imag = freq_pred[:, self.pred_len:].unsqueeze(2)  # [batch, pred_len, 1]

        # 组合为复数
        complex_pred = torch.complex(real, imag)  # [batch, pred_len, 1]

        # 逆傅里叶变换（简化：直接取实部作为预测）
        pred = torch.real(complex_pred).squeeze(-1)  # [batch, pred_len]

        # 时域精调
        pred_t = pred.unsqueeze(1)  # [batch, 1, pred_len]
        refined = self.temporal_refine(pred_t)  # [batch, 1, pred_len]
        refined = refined.squeeze(1)  # [batch, pred_len]

        return refined


class AdaptiveFusion(nn.Module):
    """自适应融合模块"""

    def __init__(self, hidden_dim: int, num_modalities: int = 3, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities

        # 模态特征编码
        self.modality_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 4)
            ) for _ in range(num_modalities)
        ])

        # 自适应门控
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim // 4 * num_modalities, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_modalities),
            nn.Softmax(dim=-1)
        )

        # 融合投影
        self.fusion_proj = nn.Linear(hidden_dim // 4, hidden_dim)

    def forward(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        """
        自适应融合

        Args:
            modalities: 模态特征列表，每个 [batch_size, seq_len, hidden_dim]

        Returns:
            融合特征 [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = modalities[0].shape

        # 编码每个模态
        encoded_modalities = []
        for i, (encoder, modality) in enumerate(zip(self.modality_encoders, modalities)):
            encoded = encoder(modality)  # [batch, seq_len, hidden_dim//4]
            encoded_modalities.append(encoded)

        # 拼接所有模态
        concatenated = torch.cat(encoded_modalities, dim=-1)  # [batch, seq_len, hidden_dim//4*num_modalities]

        # 计算门控权重
        gate_weights = self.gate_network(concatenated)  # [batch, seq_len, num_modalities]

        # 加权融合
        fused = torch.zeros_like(encoded_modalities[0])
        for i, encoded in enumerate(encoded_modalities):
            weight = gate_weights[..., i:i + 1]  # [batch, seq_len, 1]
            fused += encoded * weight

        # 投影到原始维度
        output = self.fusion_proj(fused)  # [batch, seq_len, hidden_dim]

        return output, gate_weights


class ResidualPredictionBlock(nn.Module):
    """残差预测块"""

    def __init__(self, hidden_dim: int, pred_len: int, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.pred_len = pred_len

        # 时间卷积
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)
        )

        # 注意力机制
        self.attention = TemporalAttention(hidden_dim, num_heads=8, dropout=dropout)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        # 归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        残差预测

        Args:
            x: 输入特征 [batch_size, seq_len, hidden_dim]

        Returns:
            预测序列 [batch_size, pred_len]
        """
        residual = x

        # 时间卷积
        x_t = x.transpose(1, 2)  # [batch, hidden_dim, seq_len]
        conv_out = self.temporal_conv(x_t)  # [batch, hidden_dim, seq_len]
        conv_out = conv_out.transpose(1, 2)  # [batch, seq_len, hidden_dim]

        x = self.norm1(residual + conv_out)

        # 自注意力
        attn_out = self.attention(x)
        x = self.norm2(x + attn_out)

        # 前馈网络
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)

        # 全局池化
        pooled = x.mean(dim=1)  # [batch, hidden_dim]

        # 输出投影
        pred = self.output_proj(pooled)  # [batch, pred_len]

        return pred


class MultiHeadPredictor(nn.Module):
    """
    多尺度预测头 - 完整实现

    包含：
    1. 多尺度卷积特征提取
    2. 时间注意力机制
    3. 频域预测模块
    4. 自适应融合
    5. 残差预测块
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.config = config
        hidden_dim = config['istr']['hidden_dim']
        pred_len = config['data']['pred_len']
        predictor_config = config['predictor']

        # 多尺度卷积模块
        self.multi_scale_conv = MultiScaleConvolution(
            hidden_dim=hidden_dim,
            kernel_sizes=predictor_config['multi_scale']['kernel_sizes'],
            dropout=predictor_config['dropout']
        )

        # 时间注意力
        self.temporal_attention = TemporalAttention(
            hidden_dim=hidden_dim,
            num_heads=8,
            dropout=predictor_config['dropout']
        )

        # 频域预测模块
        self.frequency_predictor = FrequencyDomainPredictor(
            hidden_dim=hidden_dim,
            pred_len=pred_len,
            dropout=predictor_config['dropout']
        )

        # 自适应融合模块
        self.adaptive_fusion = AdaptiveFusion(
            hidden_dim=hidden_dim,
            num_modalities=3,  # 多尺度、注意力、原始特征
            dropout=predictor_config['dropout']
        )

        # 残差预测块
        self.residual_blocks = nn.ModuleList([
            ResidualPredictionBlock(
                hidden_dim=hidden_dim,
                pred_len=pred_len,
                dropout=predictor_config['dropout']
            ) for _ in range(predictor_config.get('num_blocks', 2))
        ])

        # 最终投影
        self.final_proj = nn.Sequential(
            nn.Linear(pred_len * len(self.residual_blocks), pred_len * 2),
            nn.ReLU(),
            nn.Dropout(predictor_config['dropout']),
            nn.Linear(pred_len * 2, pred_len)
        )

        # 初始化权重
        self._init_weights()

        # 打印架构信息
        self._print_architecture_info()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _print_architecture_info(self):
        """打印架构信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("=" * 60)
        print("多尺度预测头架构信息")
        print("=" * 60)
        print(f"输入维度: {self.config['istr']['hidden_dim']}")
        print(f"预测长度: {self.config['data']['pred_len']}")
        print(f"多尺度卷积核: {self.config['predictor']['multi_scale']['kernel_sizes']}")
        print(f"残差块数量: {len(self.residual_blocks)}")
        print(f"总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,} ({trainable_params / total_params:.1%})")
        print("=" * 60)

    def forward(self, x: torch.Tensor, return_details: bool = False) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入特征 [batch_size, seq_len, hidden_dim]
            return_details: 是否返回详细信息

        Returns:
            预测序列 [batch_size, pred_len]
        """
        batch_size, seq_len, hidden_dim = x.shape

        # 1. 多尺度特征提取
        multi_scale_features = self.multi_scale_conv(x)  # [batch, seq_len, hidden_dim]

        # 2. 时间注意力
        attention_features = self.temporal_attention(multi_scale_features)  # [batch, seq_len, hidden_dim]

        # 3. 频域预测
        freq_pred = self.frequency_predictor(x)  # [batch, pred_len]

        # 4. 自适应融合三种特征
        fused_features, gate_weights = self.adaptive_fusion([
            multi_scale_features,  # 多尺度特征
            attention_features,  # 注意力特征
            x  # 原始特征
        ])  # [batch, seq_len, hidden_dim]

        # 5. 残差预测块
        predictions = []
        for block in self.residual_blocks:
            pred = block(fused_features)  # [batch, pred_len]
            predictions.append(pred)

        # 6. 融合多个预测
        concatenated_preds = torch.cat(predictions, dim=-1)  # [batch, pred_len * num_blocks]
        final_pred = self.final_proj(concatenated_preds)  # [batch, pred_len]

        # 7. 添加频域预测（加权融合）
        if self.config['predictor']['multi_scale'].get('attention', True):
            # 使用门控权重融合频域预测
            freq_weight = gate_weights.mean(dim=[0, 1])[0].item()  # 取第一个模态的权重
            final_pred = final_pred * (1 - freq_weight) + freq_pred * freq_weight

        if return_details:
            details = {
                'multi_scale_features': multi_scale_features.detach(),
                'attention_features': attention_features.detach(),
                'freq_pred': freq_pred.detach(),
                'fused_features': fused_features.detach(),
                'gate_weights': gate_weights.detach(),
                'intermediate_predictions': [p.detach() for p in predictions],
                'final_pred': final_pred.detach()
            }
            return final_pred, details
        else:
            return final_pred

    def analyze_predictions(self, predictions: torch.Tensor) -> Dict[str, Any]:
        """分析预测结果"""
        with torch.no_grad():
            stats = {
                'mean': predictions.mean(dim=0).cpu().numpy(),
                'std': predictions.std(dim=0).cpu().numpy(),
                'min': predictions.min(dim=0)[0].cpu().numpy(),
                'max': predictions.max(dim=0)[0].cpu().numpy(),
                'trend': self._calculate_trend(predictions),
                'volatility': self._calculate_volatility(predictions)
            }

            return stats

    def _calculate_trend(self, predictions: torch.Tensor) -> float:
        """计算趋势强度"""
        if predictions.shape[1] < 2:
            return 0.0

        # 计算斜率
        x = torch.arange(predictions.shape[1], dtype=torch.float32, device=predictions.device)
        slopes = []

        for i in range(predictions.shape[0]):
            y = predictions[i]
            slope = torch.polyfit(x, y, 1)[0]
            slopes.append(slope.item())

        return float(np.mean(np.abs(slopes)))

    def _calculate_volatility(self, predictions: torch.Tensor) -> float:
        """计算波动性"""
        if predictions.shape[1] < 2:
            return 0.0

        # 计算相邻点差异的方差
        diffs = predictions[:, 1:] - predictions[:, :-1]
        volatility = diffs.std(dim=1).mean().item()

        return volatility


class EnsemblePredictor(nn.Module):
    """集成预测器 - 结合多个预测头"""

    def __init__(self, config: Dict[str, Any], num_heads: int = 3):
        super().__init__()

        self.config = config
        self.num_heads = num_heads

        # 创建多个预测头
        self.predictors = nn.ModuleList([
            MultiHeadPredictor(config) for _ in range(num_heads)
        ])

        # 集成权重学习
        self.ensemble_weights = nn.Parameter(torch.ones(num_heads) / num_heads)

        # 自适应加权
        self.weight_network = nn.Sequential(
            nn.Linear(config['istr']['hidden_dim'], config['istr']['hidden_dim'] // 2),
            nn.ReLU(),
            nn.Linear(config['istr']['hidden_dim'] // 2, num_heads),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        """
        集成预测

        Args:
            x: 输入特征 [batch_size, seq_len, hidden_dim]
            return_all: 是否返回所有预测头的输出

        Returns:
            集成预测 [batch_size, pred_len]
        """
        # 获取每个预测头的输出
        all_predictions = []
        all_details = []

        for predictor in self.predictors:
            pred, details = predictor(x, return_details=True)
            all_predictions.append(pred)
            all_details.append(details)

        # 计算自适应权重
        features = x.mean(dim=1)  # [batch, hidden_dim]
        adaptive_weights = self.weight_network(features)  # [batch, num_heads]

        # 加权集成
        ensemble_pred = torch.zeros_like(all_predictions[0])
        for i, pred in enumerate(all_predictions):
            weight = adaptive_weights[:, i:i + 1]  # [batch, 1]
            ensemble_pred += pred * weight

        if return_all:
            return ensemble_pred, all_predictions, all_details, adaptive_weights
        else:
            return ensemble_pred

    def get_head_performance(self, x: torch.Tensor, y: torch.Tensor) -> Dict[int, Dict[str, float]]:
        """获取每个预测头的性能"""
        performance = {}

        with torch.no_grad():
            for i, predictor in enumerate(self.predictors):
                pred = predictor(x)

                # 计算指标
                mse = F.mse_loss(pred, y).item()
                mae = F.l1_loss(pred, y).item()

                performance[i] = {
                    'mse': mse,
                    'mae': mae,
                    'weight': self.ensemble_weights[i].item()
                }

        return performance


def create_predictor(config: Dict[str, Any]) -> nn.Module:
    """创建预测器"""
    predictor_type = config['predictor']['type']

    if predictor_type == 'multi_scale':
        return MultiHeadPredictor(config)
    elif predictor_type == 'ensemble':
        return EnsemblePredictor(config, num_heads=config['predictor'].get('num_heads', 3))
    else:
        raise ValueError(f"不支持的预测器类型: {predictor_type}")


# 测试代码
if __name__ == "__main__":
    # 模拟配置
    config = {
        'istr': {'hidden_dim': 64},
        'data': {'pred_len': 24},
        'predictor': {
            'type': 'multi_scale',
            'dropout': 0.1,
            'multi_scale': {
                'kernel_sizes': [1, 3, 5, 7],
                'attention': True
            },
            'num_blocks': 2
        }
    }

    # 创建预测器
    predictor = create_predictor(config)

    # 测试输入
    batch_size = 32
    seq_len = 96
    hidden_dim = 64

    x = torch.randn(batch_size, seq_len, hidden_dim)

    # 前向传播
    pred, details = predictor(x, return_details=True)

    print(f"输入形状: {x.shape}")
    print(f"预测形状: {pred.shape}")
    print(f"多尺度特征形状: {details['multi_scale_features'].shape}")
    print(f"门控权重形状: {details['gate_weights'].shape}")

    # 分析预测
    stats = predictor.analyze_predictions(pred)
    print(f"预测统计: 均值={stats['mean'].mean():.4f}, 标准差={stats['std'].mean():.4f}")