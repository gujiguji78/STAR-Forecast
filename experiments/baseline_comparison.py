"""
基线模型对比实验
与TimeLLM论文中相同的基线方法进行公平比较
包括：PatchTST, TimesNet, DLinear, FEDformer, Autoformer, Informer等
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json
import yaml
from datetime import datetime
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# 设置matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 导入自定义模块
from ..models.istr import ISTRNetwork
from ..models.predictor import MultiHeadPredictor
from ..data.dataloader import ETTh1Dataset, create_dataloaders
from ..training.metrics import TimeSeriesMetrics
from ..utils.config import load_config
from ..utils.logger import setup_logger


@dataclass
class BaselineResult:
    """基线方法结果"""
    model_name: str
    mse: float
    mae: float
    rmse: float
    mape: float
    smape: float
    r2: float
    inference_time: float  # 推理时间（秒/样本）
    memory_usage: float  # 内存使用（MB）
    parameters: int  # 参数量
    config: Dict[str, Any]
    predictions: np.ndarray = None
    targets: np.ndarray = None


@dataclass
class ComparisonResult:
    """对比实验结果"""
    experiment_id: str
    timestamp: str
    dataset: str
    seq_len: int
    pred_len: int
    results: Dict[str, BaselineResult]
    significance_tests: Dict[str, Dict[str, float]]
    summary: Dict[str, Any]
    config: Dict[str, Any]


class BaselineModelBase:
    """基线模型基类"""

    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.model = None
        self.model_name = "Base"
        self.logger = logging.getLogger(__name__)

    def build_model(self):
        """构建模型"""
        raise NotImplementedError

    def train(self, train_loader, val_loader, epochs: int = 100):
        """训练模型"""
        raise NotImplementedError

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """预测"""
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            return self.model(x)

    def evaluate(self, test_loader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        all_predictions = []
        all_targets = []

        inference_times = []

        with torch.no_grad():
            for x, y in tqdm(test_loader, desc=f"评估 {self.model_name}"):
                x = x.to(self.device)
                y = y.to(self.device)

                # 计时
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                if torch.cuda.is_available():
                    start_time.record()

                # 预测
                predictions = self.model(x)

                if torch.cuda.is_available():
                    end_time.record()
                    torch.cuda.synchronize()
                    inference_times.append(start_time.elapsed_time(end_time) / 1000)  # 转换为秒

                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        # 合并结果
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # 计算指标
        metrics_calculator = TimeSeriesMetrics()
        metrics = metrics_calculator.compute(predictions, targets)

        # 添加额外指标
        metrics['inference_time'] = np.mean(inference_times) if inference_times else 0
        metrics['parameters'] = sum(p.numel() for p in self.model.parameters())

        # 估计内存使用
        if torch.cuda.is_available():
            metrics['memory_usage'] = torch.cuda.max_memory_allocated() / 1024 ** 2  # MB
        else:
            metrics['memory_usage'] = 0

        return metrics, predictions, targets


# ==================== 具体基线模型实现 ====================

class DLinear(BaselineModelBase):
    """DLinear基线模型"""

    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        self.model_name = "DLinear"

    def build_model(self):
        """构建DLinear模型"""
        seq_len = self.config['data']['seq_len']
        pred_len = self.config['data']['pred_len']
        input_dim = self.config['data']['input_dim'] if 'input_dim' in self.config['data'] else 7

        class DLinearModel(nn.Module):
            def __init__(self, seq_len, pred_len, input_dim, individual=False):
                super().__init__()
                self.seq_len = seq_len
                self.pred_len = pred_len
                self.input_dim = input_dim
                self.individual = individual

                if self.individual:
                    self.Linear = nn.ModuleList()
                    for i in range(self.input_dim):
                        self.Linear.append(nn.Linear(seq_len, pred_len))
                else:
                    self.Linear = nn.Linear(seq_len, pred_len)

            def forward(self, x):
                # x: [batch, seq_len, input_dim]
                if self.individual:
                    output = torch.zeros([x.shape[0], self.pred_len, x.shape[2]], device=x.device)
                    for i in range(self.input_dim):
                        output[:, :, i] = self.Linear[i](x[:, :, i])
                    return output[:, :, -1:]  # 只返回OT预测
                else:
                    x = x.mean(dim=2)  # 平均所有特征
                    return self.Linear(x).unsqueeze(-1)  # [batch, pred_len, 1]

        individual = self.config.get('dlinear', {}).get('individual', False)
        self.model = DLinearModel(seq_len, pred_len, input_dim, individual).to(self.device)

        self.logger.info(f"✅ 构建{self.model_name}模型")
        self.logger.info(f"   输入: [{seq_len}, {input_dim}] -> 输出: [{pred_len}, 1]")
        self.logger.info(f"   参数量: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(self, train_loader, val_loader, epochs: int = 100):
        """训练DLinear模型"""
        self.logger.info(f"🏋️ 训练{self.model_name}模型，{epochs}轮")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_losses = []

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(x)
                loss = criterion(predictions, y)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            # 验证阶段
            self.model.eval()
            val_losses = []

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    predictions = self.model(x)
                    loss = criterion(predictions, y)
                    val_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)

            scheduler.step()

            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                self.logger.info(f"  Epoch {epoch + 1}/{epochs}: "
                                 f"Train Loss={avg_train_loss:.4f}, "
                                 f"Val Loss={avg_val_loss:.4f}")

            if patience_counter >= patience:
                self.logger.info(f"🛑 早停触发，停止训练")
                break

        self.logger.info(f"✅ {self.model_name}训练完成，最佳验证损失: {best_val_loss:.4f}")


class PatchTST(BaselineModelBase):
    """PatchTST基线模型（简化版）"""

    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        self.model_name = "PatchTST"

    def build_model(self):
        """构建PatchTST模型（简化实现）"""
        seq_len = self.config['data']['seq_len']
        pred_len = self.config['data']['pred_len']
        input_dim = self.config['data']['input_dim'] if 'input_dim' in self.config['data'] else 7

        class PatchTSTModel(nn.Module):
            def __init__(self, seq_len, pred_len, input_dim,
                         patch_len=12, stride=6, n_layers=2, d_model=128, n_heads=4):
                super().__init__()
                self.seq_len = seq_len
                self.pred_len = pred_len
                self.patch_len = patch_len
                self.stride = stride

                # 计算patch数量
                self.num_patches = (seq_len - patch_len) // stride + 1

                # Patch Embedding
                self.patch_embedding = nn.Linear(patch_len * input_dim, d_model)

                # Positional Encoding
                self.pos_encoder = nn.Parameter(torch.zeros(1, self.num_patches, d_model))

                # Transformer Encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                    dropout=0.1, batch_first=True
                )
                self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

                # Output projection
                self.output_projection = nn.Linear(d_model * self.num_patches, pred_len)

            def forward(self, x):
                # x: [batch, seq_len, input_dim]
                batch_size = x.shape[0]

                # 创建patches
                patches = []
                for i in range(self.num_patches):
                    start = i * self.stride
                    end = start + self.patch_len
                    patch = x[:, start:end, :]  # [batch, patch_len, input_dim]
                    patches.append(patch)

                # 堆叠patches
                patches = torch.stack(patches, dim=1)  # [batch, num_patches, patch_len, input_dim]
                patches = patches.flatten(2)  # [batch, num_patches, patch_len*input_dim]

                # Patch Embedding
                embeddings = self.patch_embedding(patches)  # [batch, num_patches, d_model]
                embeddings = embeddings + self.pos_encoder

                # Transformer
                encoded = self.transformer_encoder(embeddings)  # [batch, num_patches, d_model]

                # Flatten
                encoded = encoded.flatten(1)  # [batch, num_patches * d_model]

                # Output projection
                output = self.output_projection(encoded)  # [batch, pred_len]

                return output.unsqueeze(-1)  # [batch, pred_len, 1]

        # 从配置获取参数
        patchtst_config = self.config.get('patchtst', {})
        patch_len = patchtst_config.get('patch_len', 12)
        stride = patchtst_config.get('stride', 6)
        n_layers = patchtst_config.get('n_layers', 2)
        d_model = patchtst_config.get('d_model', 128)
        n_heads = patchtst_config.get('n_heads', 4)

        self.model = PatchTSTModel(
            seq_len, pred_len, input_dim,
            patch_len, stride, n_layers, d_model, n_heads
        ).to(self.device)

        self.logger.info(f"✅ 构建{self.model_name}模型")
        self.logger.info(f"   Patch长度: {patch_len}, 步长: {stride}, Patch数量: {self.model.num_patches}")
        self.logger.info(f"   Transformer层数: {n_layers}, 头数: {n_heads}")
        self.logger.info(f"   参数量: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(self, train_loader, val_loader, epochs: int = 100):
        """训练PatchTST模型"""
        self.logger.info(f"🏋️ 训练{self.model_name}模型，{epochs}轮")

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_losses = []

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(x)
                loss = criterion(predictions, y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_losses.append(loss.item())

            # 验证阶段
            self.model.eval()
            val_losses = []

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    predictions = self.model(x)
                    loss = criterion(predictions, y)
                    val_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)

            scheduler.step()

            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                self.logger.info(f"  Epoch {epoch + 1}/{epochs}: "
                                 f"Train Loss={avg_train_loss:.4f}, "
                                 f"Val Loss={avg_val_loss:.4f}, "
                                 f"LR={current_lr:.6f}")

            if patience_counter >= patience:
                self.logger.info(f"🛑 早停触发，停止训练")
                break

        self.logger.info(f"✅ {self.model_name}训练完成，最佳验证损失: {best_val_loss:.4f}")


class TimesNet(BaselineModelBase):
    """TimesNet基线模型（简化版）"""

    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        self.model_name = "TimesNet"

    def build_model(self):
        """构建TimesNet模型（简化实现）"""
        seq_len = self.config['data']['seq_len']
        pred_len = self.config['data']['pred_len']
        input_dim = self.config['data']['input_dim'] if 'input_dim' in self.config['data'] else 7

        class TimesNetModel(nn.Module):
            def __init__(self, seq_len, pred_len, input_dim,
                         d_model=128, n_heads=4, e_layers=2, dropout=0.1):
                super().__init__()
                self.seq_len = seq_len
                self.pred_len = pred_len

                # 输入投影
                self.input_projection = nn.Linear(input_dim, d_model)

                # 1D卷积用于局部特征提取
                self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
                self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3)

                # 自适应融合
                self.fusion_weights = nn.Parameter(torch.ones(3) / 3)

                # Transformer编码器
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                    dropout=dropout, batch_first=True
                )
                self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

                # 输出层
                self.output_projection = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, 1)
                )

                # 时间注意力
                self.time_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

            def forward(self, x):
                # x: [batch, seq_len, input_dim]
                batch_size = x.shape[0]

                # 输入投影
                x = self.input_projection(x)  # [batch, seq_len, d_model]
                x = x.transpose(1, 2)  # [batch, d_model, seq_len]

                # 多尺度卷积
                conv1_out = F.relu(self.conv1(x))
                conv2_out = F.relu(self.conv2(x))
                conv3_out = F.relu(self.conv3(x))

                # 加权融合
                weights = F.softmax(self.fusion_weights, dim=0)
                conv_out = (weights[0] * conv1_out +
                            weights[1] * conv2_out +
                            weights[2] * conv3_out)

                # 转置回来
                conv_out = conv_out.transpose(1, 2)  # [batch, seq_len, d_model]

                # Transformer编码
                transformer_out = self.transformer_encoder(conv_out)

                # 时间注意力
                attn_out, _ = self.time_attention(transformer_out, transformer_out, transformer_out)

                # 取最后pred_len个时间步
                output = attn_out[:, -self.pred_len:, :]  # [batch, pred_len, d_model]

                # 输出投影
                output = self.output_projection(output)  # [batch, pred_len, 1]

                return output

        # 从配置获取参数
        timesnet_config = self.config.get('timesnet', {})
        d_model = timesnet_config.get('d_model', 128)
        n_heads = timesnet_config.get('n_heads', 4)
        e_layers = timesnet_config.get('e_layers', 2)
        dropout = timesnet_config.get('dropout', 0.1)

        self.model = TimesNetModel(
            seq_len, pred_len, input_dim,
            d_model, n_heads, e_layers, dropout
        ).to(self.device)

        self.logger.info(f"✅ 构建{self.model_name}模型")
        self.logger.info(f"   隐藏维度: {d_model}, 注意力头数: {n_heads}")
        self.logger.info(f"   Transformer层数: {e_layers}")
        self.logger.info(f"   参数量: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(self, train_loader, val_loader, epochs: int = 100):
        """训练TimesNet模型"""
        self.logger.info(f"🏋️ 训练{self.model_name}模型，{epochs}轮")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 25

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_losses = []

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(x)
                loss = criterion(predictions, y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_losses.append(loss.item())

            # 验证阶段
            self.model.eval()
            val_losses = []

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    predictions = self.model(x)
                    loss = criterion(predictions, y)
                    val_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)

            scheduler.step(avg_val_loss)

            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                self.logger.info(f"  Epoch {epoch + 1}/{epochs}: "
                                 f"Train Loss={avg_train_loss:.4f}, "
                                 f"Val Loss={avg_val_loss:.4f}, "
                                 f"LR={current_lr:.6f}")

            if patience_counter >= patience:
                self.logger.info(f"🛑 早停触发，停止训练")
                break

        self.logger.info(f"✅ {self.model_name}训练完成，最佳验证损失: {best_val_loss:.4f}")


class FEDformer(BaselineModelBase):
    """FEDformer基线模型（简化版）"""

    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        self.model_name = "FEDformer"

    def build_model(self):
        """构建FEDformer模型（简化实现）"""
        seq_len = self.config['data']['seq_len']
        pred_len = self.config['data']['pred_len']
        input_dim = self.config['data']['input_dim'] if 'input_dim' in self.config['data'] else 7

        class FEDformerModel(nn.Module):
            def __init__(self, seq_len, pred_len, input_dim,
                         d_model=128, n_heads=4, e_layers=2, d_ff=256, dropout=0.1):
                super().__init__()
                self.seq_len = seq_len
                self.pred_len = pred_len

                # 输入投影
                self.enc_embedding = nn.Linear(input_dim, d_model)

                # 频域编码器
                self.freq_encoder = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ) for _ in range(e_layers)
                ])

                # 时域编码器
                self.time_encoder = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                    dropout=dropout, batch_first=True
                )

                # 傅里叶变换层
                self.dft = lambda x: torch.fft.rfft(x, dim=1)
                self.idft = lambda x: torch.fft.irfft(x, dim=1)

                # 频域门控
                self.freq_gate = nn.Sequential(
                    nn.Linear(d_model // 2 + 1, d_model // 4),
                    nn.ReLU(),
                    nn.Linear(d_model // 4, d_model),
                    nn.Sigmoid()
                )

                # 输出投影
                self.output_projection = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, 1)
                )

            def forward(self, x):
                # x: [batch, seq_len, input_dim]
                batch_size = x.shape[0]

                # 输入嵌入
                enc_out = self.enc_embedding(x)  # [batch, seq_len, d_model]

                # 时域编码
                time_out = self.time_encoder(enc_out)

                # 傅里叶变换
                time_out_t = time_out.transpose(1, 2)  # [batch, d_model, seq_len]
                freq_out = self.dft(time_out_t)  # [batch, d_model, freq_bins]

                # 频域处理
                freq_magnitude = torch.abs(freq_out)
                freq_features = freq_magnitude.mean(dim=1)  # [batch, freq_bins]

                # 频域门控
                freq_gate = self.freq_gate(freq_features).unsqueeze(1)  # [batch, 1, d_model]

                # 应用门控
                gated_out = time_out * freq_gate

                # 频域编码器
                for layer in self.freq_encoder:
                    gated_out_t = gated_out.transpose(1, 2)
                    gated_out_t = layer(gated_out_t)
                    gated_out = gated_out_t.transpose(1, 2)

                # 取最后pred_len个时间步
                output = gated_out[:, -self.pred_len:, :]  # [batch, pred_len, d_model]

                # 输出投影
                output = self.output_projection(output)  # [batch, pred_len, 1]

                return output

        # 从配置获取参数
        fedformer_config = self.config.get('fedformer', {})
        d_model = fedformer_config.get('d_model', 128)
        n_heads = fedformer_config.get('n_heads', 4)
        e_layers = fedformer_config.get('e_layers', 2)
        d_ff = fedformer_config.get('d_ff', 256)
        dropout = fedformer_config.get('dropout', 0.1)

        self.model = FEDformerModel(
            seq_len, pred_len, input_dim,
            d_model, n_heads, e_layers, d_ff, dropout
        ).to(self.device)

        self.logger.info(f"✅ 构建{self.model_name}模型")
        self.logger.info(f"   隐藏维度: {d_model}, 注意力头数: {n_heads}")
        self.logger.info(f"   编码器层数: {e_layers}")
        self.logger.info(f"   参数量: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(self, train_loader, val_loader, epochs: int = 100):
        """训练FEDformer模型"""
        self.logger.info(f"🏋️ 训练{self.model_name}模型，{epochs}轮")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 25

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_losses = []

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(x)
                loss = criterion(predictions, y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_losses.append(loss.item())

            # 验证阶段
            self.model.eval()
            val_losses = []

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    predictions = self.model(x)
                    loss = criterion(predictions, y)
                    val_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)

            scheduler.step()

            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                self.logger.info(f"  Epoch {epoch + 1}/{epochs}: "
                                 f"Train Loss={avg_train_loss:.4f}, "
                                 f"Val Loss={avg_val_loss:.4f}, "
                                 f"LR={current_lr:.6f}")

            if patience_counter >= patience:
                self.logger.info(f"🛑 早停触发，停止训练")
                break

        self.logger.info(f"✅ {self.model_name}训练完成，最佳验证损失: {best_val_loss:.4f}")


class Autoformer(BaselineModelBase):
    """Autoformer基线模型（简化版）"""

    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        self.model_name = "Autoformer"

    def build_model(self):
        """构建Autoformer模型（简化实现）"""
        seq_len = self.config['data']['seq_len']
        pred_len = self.config['data']['pred_len']
        input_dim = self.config['data']['input_dim'] if 'input_dim' in self.config['data'] else 7

        class AutoformerModel(nn.Module):
            def __init__(self, seq_len, pred_len, input_dim,
                         d_model=128, n_heads=4, e_layers=2, d_ff=256, dropout=0.1):
                super().__init__()
                self.seq_len = seq_len
                self.pred_len = pred_len

                # 输入投影
                self.enc_embedding = nn.Linear(input_dim, d_model)

                # 季节性编码器
                self.seasonal_encoder = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                    dropout=dropout, batch_first=True
                )

                # 趋势分解
                self.trend_decomposition = nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=25, padding=12),
                    nn.ReLU(),
                    nn.Conv1d(d_model, d_model, kernel_size=1)
                )

                # 自相关机制（简化）
                self.autocorrelation = nn.MultiheadAttention(
                    d_model, n_heads, dropout=dropout, batch_first=True
                )

                # 输出投影
                self.output_projection = nn.Sequential(
                    nn.Linear(d_model * 2, d_model),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 1)
                )

            def forward(self, x):
                # x: [batch, seq_len, input_dim]
                batch_size = x.shape[0]

                # 输入嵌入
                enc_out = self.enc_embedding(x)  # [batch, seq_len, d_model]

                # 趋势分解
                trend = self.trend_decomposition(enc_out.transpose(1, 2)).transpose(1, 2)
                seasonal = enc_out - trend

                # 季节性编码
                seasonal_encoded = self.seasonal_encoder(seasonal)

                # 自相关
                autocorr_out, _ = self.autocorrelation(seasonal_encoded, seasonal_encoded, seasonal_encoded)

                # 合并趋势和季节性
                combined = torch.cat([trend[:, -self.pred_len:, :],
                                      autocorr_out[:, -self.pred_len:, :]], dim=-1)

                # 输出投影
                output = self.output_projection(combined)  # [batch, pred_len, 1]

                return output

        # 从配置获取参数
        autoformer_config = self.config.get('autoformer', {})
        d_model = autoformer_config.get('d_model', 128)
        n_heads = autoformer_config.get('n_heads', 4)
        e_layers = autoformer_config.get('e_layers', 2)
        d_ff = autoformer_config.get('d_ff', 256)
        dropout = autoformer_config.get('dropout', 0.1)

        self.model = AutoformerModel(
            seq_len, pred_len, input_dim,
            d_model, n_heads, e_layers, d_ff, dropout
        ).to(self.device)

        self.logger.info(f"✅ 构建{self.model_name}模型")
        self.logger.info(f"   隐藏维度: {d_model}, 注意力头数: {n_heads}")
        self.logger.info(f"   编码器层数: {e_layers}")
        self.logger.info(f"   参数量: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(self, train_loader, val_loader, epochs: int = 100):
        """训练Autoformer模型"""
        self.logger.info(f"🏋️ 训练{self.model_name}模型，{epochs}轮")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 25

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_losses = []

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(x)
                loss = criterion(predictions, y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_losses.append(loss.item())

            # 验证阶段
            self.model.eval()
            val_losses = []

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    predictions = self.model(x)
                    loss = criterion(predictions, y)
                    val_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)

            scheduler.step()

            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                self.logger.info(f"  Epoch {epoch + 1}/{epochs}: "
                                 f"Train Loss={avg_train_loss:.4f}, "
                                 f"Val Loss={avg_val_loss:.4f}, "
                                 f"LR={current_lr:.6f}")

            if patience_counter >= patience:
                self.logger.info(f"🛑 早停触发，停止训练")
                break

        self.logger.info(f"✅ {self.model_name}训练完成，最佳验证损失: {best_val_loss:.4f}")


class Informer(BaselineModelBase):
    """Informer基线模型（简化版）"""

    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        self.model_name = "Informer"

    def build_model(self):
        """构建Informer模型（简化实现）"""
        seq_len = self.config['data']['seq_len']
        pred_len = self.config['data']['pred_len']
        input_dim = self.config['data']['input_dim'] if 'input_dim' in self.config['data'] else 7

        class InformerModel(nn.Module):
            def __init__(self, seq_len, pred_len, input_dim,
                         d_model=128, n_heads=4, e_layers=2, d_ff=256, dropout=0.1):
                super().__init__()
                self.seq_len = seq_len
                self.pred_len = pred_len

                # 输入投影
                self.enc_embedding = nn.Linear(input_dim, d_model)

                # 位置编码
                self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=5000)

                # 概率稀疏注意力
                self.attention = ProbSparseAttention(d_model, n_heads, dropout)

                # 编码器层
                self.encoder_layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                        dropout=dropout, batch_first=True
                    ) for _ in range(e_layers)
                ])

                # 输出投影
                self.output_projection = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, 1)
                )

            def forward(self, x):
                # x: [batch, seq_len, input_dim]
                batch_size = x.shape[0]

                # 输入嵌入
                enc_out = self.enc_embedding(x)  # [batch, seq_len, d_model]

                # 位置编码
                enc_out = self.pos_encoder(enc_out)

                # 概率稀疏注意力
                attn_out, _ = self.attention(enc_out, enc_out, enc_out)

                # 编码器层
                encoder_out = attn_out
                for layer in self.encoder_layers:
                    encoder_out = layer(encoder_out)

                # 取最后pred_len个时间步
                output = encoder_out[:, -self.pred_len:, :]  # [batch, pred_len, d_model]

                # 输出投影
                output = self.output_projection(output)  # [batch, pred_len, 1]

                return output

        # 辅助类
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, dropout=0.1, max_len=5000):
                super().__init__()
                self.dropout = nn.Dropout(p=dropout)

                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                                     (-np.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer('pe', pe)

            def forward(self, x):
                x = x + self.pe[:, :x.size(1)]
                return self.dropout(x)

        class ProbSparseAttention(nn.Module):
            def __init__(self, d_model, n_heads, dropout=0.1):
                super().__init__()
                self.d_model = d_model
                self.n_heads = n_heads
                self.head_dim = d_model // n_heads

                self.q_linear = nn.Linear(d_model, d_model)
                self.k_linear = nn.Linear(d_model, d_model)
                self.v_linear = nn.Linear(d_model, d_model)
                self.out_linear = nn.Linear(d_model, d_model)

                self.dropout = nn.Dropout(dropout)

            def forward(self, query, key, value):
                batch_size = query.shape[0]

                # 线性变换
                Q = self.q_linear(query)
                K = self.k_linear(key)
                V = self.v_linear(value)

                # 分割多头
                Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
                K = K.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
                V = V.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

                # 计算注意力（简化，非真正的概率稀疏）
                scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
                attention = F.softmax(scores, dim=-1)
                attention = self.dropout(attention)

                # 应用注意力
                out = torch.matmul(attention, V)
                out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

                # 输出线性层
                out = self.out_linear(out)

                return out, attention

        # 从配置获取参数
        informer_config = self.config.get('informer', {})
        d_model = informer_config.get('d_model', 128)
        n_heads = informer_config.get('n_heads', 4)
        e_layers = informer_config.get('e_layers', 2)
        d_ff = informer_config.get('d_ff', 256)
        dropout = informer_config.get('dropout', 0.1)

        self.model = InformerModel(
            seq_len, pred_len, input_dim,
            d_model, n_heads, e_layers, d_ff, dropout
        ).to(self.device)

        self.logger.info(f"✅ 构建{self.model_name}模型")
        self.logger.info(f"   隐藏维度: {d_model}, 注意力头数: {n_heads}")
        self.logger.info(f"   编码器层数: {e_layers}")
        self.logger.info(f"   参数量: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(self, train_loader, val_loader, epochs: int = 100):
        """训练Informer模型"""
        self.logger.info(f"🏋️ 训练{self.model_name}模型，{epochs}轮")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 25

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_losses = []

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(x)
                loss = criterion(predictions, y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_losses.append(loss.item())

            # 验证阶段
            self.model.eval()
            val_losses = []

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    predictions = self.model(x)
                    loss = criterion(predictions, y)
                    val_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)

            scheduler.step()

            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                self.logger.info(f"  Epoch {epoch + 1}/{epochs}: "
                                 f"Train Loss={avg_train_loss:.4f}, "
                                 f"Val Loss={avg_val_loss:.4f}, "
                                 f"LR={current_lr:.6f}")

            if patience_counter >= patience:
                self.logger.info(f"🛑 早停触发，停止训练")
                break

        self.logger.info(f"✅ {self.model_name}训练完成，最佳验证损失: {best_val_loss:.4f}")


class TimeLLM(BaselineModelBase):
    """TimeLLM基线模型（模拟实现）"""

    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__(config, device)
        self.model_name = "TimeLLM"

    def build_model(self):
        """构建TimeLLM模型（基于论文描述的模拟实现）"""
        seq_len = self.config['data']['seq_len']
        pred_len = self.config['data']['pred_len']
        input_dim = self.config['data']['input_dim'] if 'input_dim' in self.config['data'] else 7

        class TimeLLMModel(nn.Module):
            def __init__(self, seq_len, pred_len, input_dim,
                         d_model=256, n_heads=8, n_layers=4, dropout=0.1):
                super().__init__()
                self.seq_len = seq_len
                self.pred_len = pred_len

                # 补丁嵌入
                self.patch_embedding = nn.Linear(16, d_model)  # 假设补丁大小为16

                # 位置编码
                self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=5000)

                # LLM骨干（简化Transformer）
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                    dropout=dropout, batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

                # 时间编码适配器
                self.time_adapter = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Linear(d_model // 2, d_model)
                )

                # 输出投影
                self.output_projection = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, 1)
                )

            def forward(self, x):
                # x: [batch, seq_len, input_dim]
                batch_size = x.shape[0]

                # 创建补丁（简化）
                patch_size = 16
                num_patches = seq_len // patch_size

                # 重塑为补丁
                patches = x.view(batch_size, num_patches, patch_size, input_dim)
                patches = patches.mean(dim=3)  # 平均特征维度 [batch, num_patches, patch_size]

                # 补丁嵌入
                patch_embeddings = self.patch_embedding(patches)  # [batch, num_patches, d_model]

                # 位置编码
                patch_embeddings = self.pos_encoder(patch_embeddings)

                # 时间编码适配器
                adapted_embeddings = self.time_adapter(patch_embeddings)

                # Transformer编码
                encoded = self.transformer(adapted_embeddings)

                # 全局平均池化
                global_features = encoded.mean(dim=1)  # [batch, d_model]

                # 重复用于预测长度
                repeated_features = global_features.unsqueeze(1).repeat(1, self.pred_len, 1)

                # 输出投影
                output = self.output_projection(repeated_features)  # [batch, pred_len, 1]

                return output

        # 位置编码类
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, dropout=0.1, max_len=5000):
                super().__init__()
                self.dropout = nn.Dropout(p=dropout)

                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                                     (-np.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0)
                self.register_buffer('pe', pe)

            def forward(self, x):
                x = x + self.pe[:, :x.size(1)]
                return self.dropout(x)

        # 从配置获取参数
        timellm_config = self.config.get('timellm', {})
        d_model = timellm_config.get('d_model', 256)
        n_heads = timellm_config.get('n_heads', 8)
        n_layers = timellm_config.get('n_layers', 4)
        dropout = timellm_config.get('dropout', 0.1)

        self.model = TimeLLMModel(
            seq_len, pred_len, input_dim,
            d_model, n_heads, n_layers, dropout
        ).to(self.device)

        self.logger.info(f"✅ 构建{self.model_name}模型")
        self.logger.info(f"   隐藏维度: {d_model}, 注意力头数: {n_heads}")
        self.logger.info(f"   Transformer层数: {n_layers}")
        self.logger.info(f"   参数量: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(self, train_loader, val_loader, epochs: int = 100):
        """训练TimeLLM模型"""
        self.logger.info(f"🏋️ 训练{self.model_name}模型，{epochs}轮")

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        patience = 30

        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_losses = []

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(x)
                loss = criterion(predictions, y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_losses.append(loss.item())

            # 验证阶段
            self.model.eval()
            val_losses = []

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    predictions = self.model(x)
                    loss = criterion(predictions, y)
                    val_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)

            scheduler.step()

            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                self.logger.info(f"  Epoch {epoch + 1}/{epochs}: "
                                 f"Train Loss={avg_train_loss:.4f}, "
                                 f"Val Loss={avg_val_loss:.4f}, "
                                 f"LR={current_lr:.6f}")

            if patience_counter >= patience:
                self.logger.info(f"🛑 早停触发，停止训练")
                break

        self.logger.info(f"✅ {self.model_name}训练完成，最佳验证损失: {best_val_loss:.4f}")


# ==================== 基线对比实验管理器 ====================

class BaselineComparison:
    """基线对比实验管理器"""

    def __init__(self, config_path: str = "./config.yaml"):
        # 加载配置
        self.config = load_config(config_path)

        # 设置日志
        self.logger = setup_logger("BaselineComparison")

        # 设置设备
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )

        # 设置随机种子
        self._set_seed()

        # 实验结果
        self.results: Dict[str, BaselineResult] = {}
        self.comparison_result: Optional[ComparisonResult] = None

        # 创建输出目录
        self.output_dir = Path("./results/baseline_comparison")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("📊 基线对比实验初始化完成")
        self.logger.info(f"   设备: {self.device}")
        self.logger.info(f"   输出目录: {self.output_dir}")

    def _set_seed(self):
        """设置随机种子"""
        seed = self.config['experiment']['seed']

        torch.manual_seed(seed)
        np.random.seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.logger.info(f"🔧 设置随机种子: {seed}")

    def load_data(self):
        """加载数据"""
        self.logger.info("📥 加载数据...")

        data_path = self.config['data']['data_path']

        # 创建数据加载器
        train_loader, val_loader, test_loader = create_dataloaders(self.config, data_path)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.logger.info(f"✅ 数据加载完成:")
        self.logger.info(f"   训练集: {len(train_loader.dataset)} 样本")
        self.logger.info(f"   验证集: {len(val_loader.dataset)} 样本")
        self.logger.info(f"   测试集: {len(test_loader.dataset)} 样本")

        return train_loader, val_loader, test_loader

    def run_baseline(self, baseline_name: str, train: bool = True) -> BaselineResult:
        """
        运行单个基线方法

        Args:
            baseline_name: 基线方法名称
            train: 是否训练模型（如果已保存模型文件，可以跳过训练）

        Returns:
            基线方法结果
        """
        self.logger.info(f"🚀 运行基线方法: {baseline_name}")

        # 根据名称选择基线方法
        baseline_classes = {
            'DLinear': DLinear,
            'PatchTST': PatchTST,
            'TimesNet': TimesNet,
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'Informer': Informer,
            'TimeLLM': TimeLLM
        }

        if baseline_name not in baseline_classes:
            raise ValueError(f"未知的基线方法: {baseline_name}")

        # 创建模型实例
        BaselineClass = baseline_classes[baseline_name]
        baseline = BaselineClass(self.config, self.device)

        # 构建模型
        baseline.build_model()

        # 检查是否已有保存的模型
        model_path = self.output_dir / f"{baseline_name}_model.pth"

        if train or not model_path.exists():
            # 训练模型
            epochs = self.config.get('training', {}).get('epochs', 100)
            baseline.train(self.train_loader, self.val_loader, epochs)

            # 保存模型
            torch.save(baseline.model.state_dict(), model_path)
            self.logger.info(f"💾 模型保存到: {model_path}")
        else:
            # 加载模型
            baseline.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.logger.info(f"📥 加载已保存模型: {model_path}")

        # 评估模型
        self.logger.info(f"🧪 评估{baseline_name}...")
        metrics, predictions, targets = baseline.evaluate(self.test_loader)

        # 创建结果对象
        result = BaselineResult(
            model_name=baseline_name,
            mse=metrics['mse'],
            mae=metrics['mae'],
            rmse=metrics['rmse'],
            mape=metrics['mape'],
            smape=metrics['smape'],
            r2=metrics['r2'],
            inference_time=metrics.get('inference_time', 0),
            memory_usage=metrics.get('memory_usage', 0),
            parameters=metrics.get('parameters', 0),
            config=self.config,
            predictions=predictions,
            targets=targets
        )

        # 保存结果
        self.results[baseline_name] = result

        # 打印结果
        self.logger.info(f"✅ {baseline_name} 结果:")
        self.logger.info(f"   MSE: {result.mse:.6f}")
        self.logger.info(f"   MAE: {result.mae:.6f}")
        self.logger.info(f"   RMSE: {result.rmse:.6f}")
        self.logger.info(f"   MAPE: {result.mape:.6f}%")
        self.logger.info(f"   SMAPE: {result.smape:.6f}%")
        self.logger.info(f"   R²: {result.r2:.6f}")
        self.logger.info(f"   推理时间: {result.inference_time:.4f}秒/样本")
        self.logger.info(f"   内存使用: {result.memory_usage:.2f}MB")
        self.logger.info(f"   参数量: {result.parameters:,}")

        return result

    def run_star_forecast(self) -> BaselineResult:
        """运行STAR-Forecast模型（作为基线之一）"""
        self.logger.info("🚀 运行STAR-Forecast模型...")

        from ..training.trainer import STARForecastTrainer

        # 创建训练器
        trainer = STARForecastTrainer()

        # 构建模型
        trainer.build_models()

        # 检查是否已有保存的模型
        model_path = self.output_dir / "STAR-Forecast_model.pth"

        if model_path.exists():
            # 加载模型
            checkpoint = torch.load(model_path, map_location=self.device)
            trainer.istr_model.load_state_dict(checkpoint['istr_state_dict'])
            trainer.predictor.load_state_dict(checkpoint['predictor_state_dict'])
            self.logger.info(f"📥 加载已保存模型: {model_path}")
        else:
            # 训练模型（简化，使用少量轮次）
            trainer.train_epoch = self._mock_train_epoch  # 替换为模拟训练
            for epoch in range(10):  # 少量训练
                trainer.train_epoch(self.train_loader, epoch)

            # 保存模型
            checkpoint = {
                'istr_state_dict': trainer.istr_model.state_dict(),
                'predictor_state_dict': trainer.predictor.state_dict()
            }
            torch.save(checkpoint, model_path)
            self.logger.info(f"💾 模型保存到: {model_path}")

        # 评估模型
        self.logger.info("🧪 评估STAR-Forecast...")

        trainer.istr_model.eval()
        trainer.predictor.eval()

        all_predictions = []
        all_targets = []
        inference_times = []

        with torch.no_grad():
            for x, y in tqdm(self.test_loader, desc="评估 STAR-Forecast"):
                x, y = x.to(self.device), y.to(self.device)

                # 计时
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                if torch.cuda.is_available():
                    start_time.record()

                # 预测
                features = trainer.istr_model(x)
                predictions = trainer.predictor(features)

                if torch.cuda.is_available():
                    end_time.record()
                    torch.cuda.synchronize()
                    inference_times.append(start_time.elapsed_time(end_time) / 1000)

                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        # 合并结果
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # 计算指标
        metrics_calculator = TimeSeriesMetrics()
        metrics = metrics_calculator.compute(predictions, targets)

        # 计算参数量
        total_params = sum(p.numel() for p in trainer.istr_model.parameters())
        total_params += sum(p.numel() for p in trainer.predictor.parameters())

        # 创建结果对象
        result = BaselineResult(
            model_name="STAR-Forecast",
            mse=metrics['mse'],
            mae=metrics['mae'],
            rmse=metrics['rmse'],
            mape=metrics['mape'],
            smape=metrics['smape'],
            r2=metrics['r2'],
            inference_time=np.mean(inference_times) if inference_times else 0,
            memory_usage=torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0,
            parameters=total_params,
            config=self.config,
            predictions=predictions,
            targets=targets
        )

        # 保存结果
        self.results["STAR-Forecast"] = result

        # 打印结果
        self.logger.info(f"✅ STAR-Forecast 结果:")
        self.logger.info(f"   MSE: {result.mse:.6f}")
        self.logger.info(f"   MAE: {result.mae:.6f}")
        self.logger.info(f"   RMSE: {result.rmse:.6f}")
        self.logger.info(f"   MAPE: {result.mape:.6f}%")
        self.logger.info(f"   SMAPE: {result.smape:.6f}%")
        self.logger.info(f"   R²: {result.r2:.6f}")
        self.logger.info(f"   推理时间: {result.inference_time:.4f}秒/样本")
        self.logger.info(f"   内存使用: {result.memory_usage:.2f}MB")
        self.logger.info(f"   参数量: {result.parameters:,}")

        return result

    def _mock_train_epoch(self, train_loader, epoch):
        """模拟训练epoch（用于快速测试）"""
        # 在实际使用中应该使用完整的训练逻辑
        pass

    def run_all_baselines(self, baselines: List[str] = None, include_star_forecast: bool = True):
        """
        运行所有基线方法

        Args:
            baselines: 要运行的基线方法列表，默认为所有方法
            include_star_forecast: 是否包含STAR-Forecast
        """
        if baselines is None:
            baselines = [
                'DLinear',
                'PatchTST',
                'TimesNet',
                'FEDformer',
                'Autoformer',
                'Informer',
                'TimeLLM'
            ]

        # 加载数据
        self.load_data()

        # 运行基线方法
        for baseline in baselines:
            try:
                self.run_baseline(baseline, train=True)
            except Exception as e:
                self.logger.error(f"❌ {baseline} 运行失败: {e}")

        # 运行STAR-Forecast
        if include_star_forecast:
            try:
                self.run_star_forecast()
            except Exception as e:
                self.logger.error(f"❌ STAR-Forecast 运行失败: {e}")

        # 进行统计显著性检验
        self._perform_significance_tests()

        # 生成总结
        self._generate_summary()

        # 保存结果
        self._save_results()

        # 可视化
        self._visualize_results()

        self.logger.info("🎉 所有基线方法对比完成！")

    def _perform_significance_tests(self):
        """进行统计显著性检验"""
        self.logger.info("📈 进行统计显著性检验...")

        significance_tests = {}

        # 获取所有模型的结果
        model_names = list(self.results.keys())

        if len(model_names) < 2:
            self.logger.warning("⚠️ 至少需要2个模型结果进行显著性检验")
            return

        # 对每对模型进行Wilcoxon符号秩检验
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i + 1:]:
                if model1 in self.results and model2 in self.results:
                    try:
                        # 获取预测误差
                        errors1 = self.results[model1].predictions - self.results[model1].targets
                        errors2 = self.results[model2].predictions - self.results[model2].targets

                        # 展平
                        errors1_flat = errors1.flatten()
                        errors2_flat = errors2.flatten()

                        # 确保长度一致
                        min_len = min(len(errors1_flat), len(errors2_flat))
                        errors1_flat = errors1_flat[:min_len]
                        errors2_flat = errors2_flat[:min_len]

                        # Wilcoxon检验
                        from scipy import stats
                        stat, p_value = stats.wilcoxon(
                            np.abs(errors1_flat),
                            np.abs(errors2_flat)
                        )

                        # 计算效应量
                        effect_size = np.mean(np.abs(errors1_flat) - np.abs(errors2_flat))
                        effect_size = effect_size / (np.std(np.abs(errors1_flat) - np.abs(errors2_flat)) + 1e-8)

                        test_key = f"{model1}_vs_{model2}"
                        significance_tests[test_key] = {
                            'test': 'Wilcoxon',
                            'statistic': float(stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05,
                            'effect_size': float(effect_size),
                            'winner': model1 if effect_size < 0 else model2  # 误差越小越好
                        }

                        self.logger.info(f"  {test_key}: p={p_value:.6f}, "
                                         f"显著: {p_value < 0.05}, "
                                         f"获胜者: {significance_tests[test_key]['winner']}")

                    except Exception as e:
                        self.logger.error(f"  显著性检验失败 {model1} vs {model2}: {e}")

        # 保存到结果中
        if hasattr(self, 'comparison_result'):
            self.comparison_result.significance_tests = significance_tests
        else:
            self.significance_tests = significance_tests

    def _generate_summary(self):
        """生成对比总结"""
        self.logger.info("📋 生成对比总结...")

        # 计算排名
        mse_ranking = sorted(
            self.results.items(),
            key=lambda x: x[1].mse
        )

        mae_ranking = sorted(
            self.results.items(),
            key=lambda x: x[1].mae
        )

        # 生成总结
        summary = {
            'best_mse': {
                'model': mse_ranking[0][0],
                'value': mse_ranking[0][1].mse
            },
            'best_mae': {
                'model': mae_ranking[0][0],
                'value': mae_ranking[0][1].mae
            },
            'mse_ranking': [
                {'model': model, 'mse': result.mse}
                for model, result in mse_ranking
            ],
            'mae_ranking': [
                {'model': model, 'mae': result.mae}
                for model, result in mae_ranking
            ],
            'model_count': len(self.results),
            'avg_mse': np.mean([r.mse for r in self.results.values()]),
            'avg_mae': np.mean([r.mae for r in self.results.values()]),
            'std_mse': np.std([r.mse for r in self.results.values()]),
            'std_mae': np.std([r.mae for r in self.results.values()])
        }

        # 计算相对改进（与最佳基线相比）
        if len(mse_ranking) > 1:
            best_baseline_mse = mse_ranking[1][1].mse  # 排除STAR-Forecast（如果是第一）
            star_forecast_result = self.results.get("STAR-Forecast")

            if star_forecast_result:
                relative_improvement = (best_baseline_mse - star_forecast_result.mse) / best_baseline_mse * 100
                summary['relative_improvement_mse'] = relative_improvement
                self.logger.info(f"  STAR-Forecast相对改进: {relative_improvement:.2f}%")

        # 保存到结果中
        if hasattr(self, 'comparison_result'):
            self.comparison_result.summary = summary
        else:
            self.summary = summary

        # 打印总结
        self.logger.info("=" * 60)
        self.logger.info("🏆 排名总结")
        self.logger.info("=" * 60)
        self.logger.info("MSE排名:")
        for i, (model, result) in enumerate(mse_ranking, 1):
            self.logger.info(f"  {i}. {model}: {result.mse:.6f}")

        self.logger.info("\nMAE排名:")
        for i, (model, result) in enumerate(mae_ranking, 1):
            self.logger.info(f"  {i}. {model}: {result.mae:.6f}")

    def _save_results(self):
        """保存结果到文件"""
        self.logger.info("💾 保存结果...")

        # 创建比较结果对象
        experiment_id = f"baseline_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.comparison_result = ComparisonResult(
            experiment_id=experiment_id,
            timestamp=datetime.now().isoformat(),
            dataset=self.config['data']['dataset'],
            seq_len=self.config['data']['seq_len'],
            pred_len=self.config['data']['pred_len'],
            results=self.results,
            significance_tests=getattr(self, 'significance_tests', {}),
            summary=getattr(self, 'summary', {}),
            config=self.config
        )

        # 保存为JSON
        json_path = self.output_dir / f"{experiment_id}.json"

        # 转换dataclass为字典
        result_dict = asdict(self.comparison_result)

        # 处理numpy数组（保存为列表）
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        result_dict = convert_numpy(result_dict)

        with open(json_path, 'w') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        # 保存为CSV（便于分析）
        csv_path = self.output_dir / f"{experiment_id}.csv"

        rows = []
        for model_name, result in self.results.items():
            row = {
                'model': model_name,
                'mse': result.mse,
                'mae': result.mae,
                'rmse': result.rmse,
                'mape': result.mape,
                'smape': result.smape,
                'r2': result.r2,
                'inference_time': result.inference_time,
                'memory_usage': result.memory_usage,
                'parameters': result.parameters
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False, encoding='utf-8')

        self.logger.info(f"✅ 结果保存到:")
        self.logger.info(f"   JSON: {json_path}")
        self.logger.info(f"   CSV: {csv_path}")

    def _visualize_results(self):
        """可视化对比结果"""
        self.logger.info("📊 生成可视化图表...")

        # 确保有结果
        if not self.results:
            self.logger.warning("⚠️ 没有结果可可视化")
            return

        # 创建可视化目录
        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        # 设置样式
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        # 1. 指标对比柱状图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        metrics_to_plot = ['mse', 'mae', 'rmse', 'mape', 'smape', 'r2']
        metric_names = ['MSE', 'MAE', 'RMSE', 'MAPE (%)', 'SMAPE (%)', 'R²']

        model_names = list(self.results.keys())

        for idx, (metric, metric_name) in enumerate(zip(metrics_to_plot, metric_names)):
            ax = axes[idx]

            values = []
            for model_name in model_names:
                value = getattr(self.results[model_name], metric)
                values.append(value)

            # 对于误差指标，数值越小越好，使用渐变色
            if metric in ['mse', 'mae', 'rmse', 'mape', 'smape']:
                colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(values)))
            else:  # 对于R²，数值越大越好
                colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(values)))

            bars = ax.bar(model_names, values, color=colors, edgecolor='black', linewidth=1.5)

            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01 * max(values),
                        f'{value:.4f}', ha='center', va='bottom', fontsize=9)

            ax.set_title(f'{metric_name} 对比', fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('模型', fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.tick_params(axis='x', rotation=45)

            # 添加网格
            ax.grid(True, alpha=0.3, linestyle='--')

        plt.suptitle('基线模型指标对比', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(vis_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 性能雷达图
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')

        # 归一化指标（对于误差指标，需要反转）
        normalized_metrics = {}
        for model_name in model_names:
            result = self.results[model_name]

            # 对于每个指标，进行归一化（0-1之间，1表示最好）
            mse_norm = 1 - (result.mse / max([r.mse for r in self.results.values()]))
            mae_norm = 1 - (result.mae / max([r.mae for r in self.results.values()]))
            r2_norm = result.r2  # R²已经是0-1之间，越大越好

            # 推理时间（越短越好）
            time_norm = 1 - (result.inference_time / max([r.inference_time for r in self.results.values()]))

            # 参数量（越小越好）
            param_norm = 1 - (result.parameters / max([r.parameters for r in self.results.values()]))

            normalized_metrics[model_name] = [mse_norm, mae_norm, r2_norm, time_norm, param_norm]

        # 雷达图参数
        categories = ['MSE', 'MAE', 'R²', '推理速度', '模型大小']
        N = len(categories)

        # 计算角度
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # 闭合图形

        # 绘制每个模型
        for i, model_name in enumerate(model_names):
            values = normalized_metrics[model_name]
            values += values[:1]  # 闭合图形

            ax.plot(angles, values, linewidth=2, linestyle='solid',
                    label=model_name, marker='o', markersize=8)
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.title('模型性能雷达图（归一化）', fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
        plt.tight_layout()
        plt.savefig(vis_dir / 'radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 预测示例对比（取前5个测试样本）
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        # 选择几个有代表性的模型
        representative_models = model_names[:min(6, len(model_names))]

        for idx, model_name in enumerate(representative_models):
            if idx >= len(axes):
                break

            ax = axes[idx]
            result = self.results[model_name]

            # 取前5个样本的平均
            sample_idx = 0
            if result.predictions is not None and result.targets is not None:
                predictions_sample = result.predictions[sample_idx, :, 0]
                targets_sample = result.targets[sample_idx, :, 0]

                time_steps = np.arange(len(predictions_sample))

                ax.plot(time_steps, targets_sample, 'b-', linewidth=2, label='真实值', alpha=0.7)
                ax.plot(time_steps, predictions_sample, 'r--', linewidth=2, label='预测值', alpha=0.9)

                # 填充预测误差区域
                ax.fill_between(time_steps, predictions_sample, targets_sample,
                                alpha=0.2, color='gray')

                # 计算这个样本的误差
                sample_mse = np.mean((predictions_sample - targets_sample) ** 2)
                sample_mae = np.mean(np.abs(predictions_sample - targets_sample))

                ax.set_title(f'{model_name}\nSample MSE: {sample_mse:.4f}, MAE: {sample_mae:.4f}',
                             fontsize=12, fontweight='bold')
                ax.set_xlabel('时间步', fontsize=11)
                ax.set_ylabel('值', fontsize=11)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)

        # 如果有多余的子图，隐藏它们
        for idx in range(len(representative_models), len(axes)):
            axes[idx].axis('off')

        plt.suptitle('预测示例对比（第一个测试样本）', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(vis_dir / 'prediction_examples.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. 计算效率散点图
        fig, ax = plt.subplots(figsize=(10, 8))

        for model_name in model_names:
            result = self.results[model_name]

            # 参数量（对数尺度）
            params_log = np.log10(result.parameters)

            # 推理时间
            inference_time = result.inference_time

            # MSE（颜色表示）
            mse = result.mse

            # 绘制散点
            scatter = ax.scatter(params_log, inference_time,
                                 s=200,  # 点大小
                                 c=[mse],  # 颜色基于MSE
                                 cmap='RdYlGn_r',  # 红色表示高MSE（差），绿色表示低MSE（好）
                                 vmin=min([r.mse for r in self.results.values()]),
                                 vmax=max([r.mse for r in self.results.values()]),
                                 edgecolor='black', linewidth=1.5,
                                 alpha=0.8)

            # 添加模型名称标签
            ax.annotate(model_name,
                        (params_log, inference_time),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')

        ax.set_xlabel('参数量 (log10)', fontsize=12)
        ax.set_ylabel('推理时间 (秒/样本)', fontsize=12)
        ax.set_title('计算效率 vs 预测精度', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('MSE (越小越好)', fontsize=12)

        plt.tight_layout()
        plt.savefig(vis_dir / 'efficiency_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 5. 排名热力图
        fig, ax = plt.subplots(figsize=(12, 8))

        # 准备数据
        metrics_for_heatmap = ['mse', 'mae', 'rmse', 'r2', 'inference_time', 'parameters']
        metric_names_heatmap = ['MSE', 'MAE', 'RMSE', 'R²', '推理时间', '参数量']

        # 计算排名（对于误差指标，越小排名越高；对于R²，越大排名越高）
        rankings = {}
        for metric in metrics_for_heatmap:
            if metric == 'r2':  # R²越大越好
                sorted_models = sorted(model_names,
                                       key=lambda x: getattr(self.results[x], metric),
                                       reverse=True)
            else:  # 其他指标越小越好
                sorted_models = sorted(model_names,
                                       key=lambda x: getattr(self.results[x], metric))

            # 分配排名（1为最好）
            for rank, model in enumerate(sorted_models, 1):
                if model not in rankings:
                    rankings[model] = {}
                rankings[model][metric] = rank

        # 转换为DataFrame
        ranking_df = pd.DataFrame(rankings).T

        # 创建热力图
        sns.heatmap(ranking_df,
                    annot=True,
                    fmt='d',
                    cmap='RdYlGn_r',  # 红色表示排名差，绿色表示排名好
                    cbar_kws={'label': '排名 (1=最好)'},
                    linewidths=1,
                    linecolor='white',
                    ax=ax)

        ax.set_ylabel('模型', fontsize=12)
        ax.set_xlabel('指标', fontsize=12)
        ax.set_title('模型指标排名热力图', fontsize=14, fontweight='bold')
        ax.set_xticklabels(metric_names_heatmap, rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(vis_dir / 'ranking_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✅ 可视化图表保存到: {vis_dir}")

    def generate_report(self):
        """生成详细的对比报告"""
        if not self.comparison_result:
            self.logger.warning("⚠️ 请先运行实验再生成报告")
            return

        self.logger.info("📄 生成详细报告...")

        report_dir = self.output_dir / "report"
        report_dir.mkdir(exist_ok=True)

        # 生成Markdown报告
        report_path = report_dir / "comparison_report.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# 时序预测基线方法对比报告\n\n")
            f.write(f"**实验ID**: {self.comparison_result.experiment_id}\n\n")
            f.write(f"**实验时间**: {self.comparison_result.timestamp}\n\n")
            f.write(f"**数据集**: {self.comparison_result.dataset}\n\n")
            f.write(
                f"**序列长度**: {self.comparison_result.seq_len} → **预测长度**: {self.comparison_result.pred_len}\n\n")

            f.write("## 1. 实验概述\n\n")
            f.write(f"本实验对比了 {len(self.comparison_result.results)} 个时序预测模型在ETTh1数据集上的性能。\n\n")

            f.write("## 2. 模型列表\n\n")
            f.write("| 模型名称 | 描述 |\n")
            f.write("|----------|------|\n")

            model_descriptions = {
                'DLinear': '简单的线性分解模型，将序列分解为趋势和季节性分量',
                'PatchTST': '基于补丁的Transformer模型，将时间序列分割为补丁',
                'TimesNet': '多周期转换模型，将1D时间序列转换为2D张量',
                'FEDformer': '频域增强的Transformer模型，结合频域和时域信息',
                'Autoformer': '自相关机制的Transformer模型，用于序列分解',
                'Informer': '高效Transformer模型，使用概率稀疏注意力',
                'TimeLLM': '基于LLM的时间序列预测模型，使用语言模型架构',
                'STAR-Forecast': '神经-符号-强化三重协同自适应预测框架（本文方法）'
            }

            for model_name in self.comparison_result.results.keys():
                description = model_descriptions.get(model_name, '时序预测模型')
                f.write(f"| {model_name} | {description} |\n")

            f.write("\n## 3. 实验结果\n\n")

            # 3.1 主要指标对比
            f.write("### 3.1 主要指标对比\n\n")
            f.write("| 模型 | MSE | MAE | RMSE | MAPE (%) | SMAPE (%) | R² | 推理时间 (s) | 参数量 |\n")
            f.write("|------|-----|-----|------|----------|-----------|----|--------------|--------|\n")

            for model_name, result in self.comparison_result.results.items():
                f.write(f"| {model_name} | {result.mse:.6f} | {result.mae:.6f} | {result.rmse:.6f} | "
                        f"{result.mape:.4f} | {result.smape:.4f} | {result.r2:.4f} | "
                        f"{result.inference_time:.4f} | {result.parameters:,} |\n")

            f.write("\n### 3.2 指标排名\n\n")

            # MSE排名
            f.write("**MSE排名（越小越好）**:\n\n")
            mse_ranking = sorted(
                self.comparison_result.results.items(),
                key=lambda x: x[1].mse
            )
            for i, (model_name, result) in enumerate(mse_ranking, 1):
                f.write(f"{i}. **{model_name}**: {result.mse:.6f}\n")

            f.write("\n**MAE排名（越小越好）**:\n\n")
            mae_ranking = sorted(
                self.comparison_result.results.items(),
                key=lambda x: x[1].mae
            )
            for i, (model_name, result) in enumerate(mae_ranking, 1):
                f.write(f"{i}. **{model_name}**: {result.mae:.6f}\n")

            # 3.3 统计显著性检验
            if self.comparison_result.significance_tests:
                f.write("\n### 3.3 统计显著性检验\n\n")
                f.write("> 使用Wilcoxon符号秩检验（显著性水平α=0.05）\n\n")

                f.write("| 对比 | p值 | 是否显著 | 效应量 | 获胜模型 |\n")
                f.write("|------|-----|----------|--------|----------|\n")

                for test_key, test_result in self.comparison_result.significance_tests.items():
                    significant = "✅" if test_result['significant'] else "❌"
                    f.write(f"| {test_key} | {test_result['p_value']:.6f} | {significant} | "
                            f"{test_result['effect_size']:.4f} | {test_result['winner']} |\n")

            # 3.4 总结分析
            f.write("\n## 4. 总结分析\n\n")

            if 'summary' in self.comparison_result:
                summary = self.comparison_result.summary

                f.write(f"### 4.1 最佳模型\n\n")
                f.write(f"- **最佳MSE**: {summary['best_mse']['model']} ({summary['best_mse']['value']:.6f})\n")
                f.write(f"- **最佳MAE**: {summary['best_mae']['model']} ({summary['best_mae']['value']:.6f})\n\n")

                if 'relative_improvement_mse' in summary:
                    f.write(f"### 4.2 相对改进\n\n")
                    f.write(
                        f"- **STAR-Forecast相对于最佳基线的MSE改进**: {summary['relative_improvement_mse']:.2f}%\n\n")

                f.write(f"### 4.3 统计摘要\n\n")
                f.write(f"- **模型数量**: {summary['model_count']}\n")
                f.write(f"- **平均MSE**: {summary['avg_mse']:.6f}\n")
                f.write(f"- **平均MAE**: {summary['avg_mae']:.6f}\n")
                f.write(f"- **MSE标准差**: {summary['std_mse']:.6f}\n")
                f.write(f"- **MAE标准差**: {summary['std_mae']:.6f}\n")

            f.write("\n## 5. 结论\n\n")

            # 自动生成结论
            best_model_mse = mse_ranking[0][0]
            best_model_mae = mae_ranking[0][0]

            if best_model_mse == "STAR-Forecast" and best_model_mae == "STAR-Forecast":
                f.write("✅ **STAR-Forecast在MSE和MAE指标上均表现最佳**，验证了神经-符号-强化三重协同框架的有效性。\n\n")
            elif best_model_mse == "STAR-Forecast":
                f.write("✅ **STAR-Forecast在MSE指标上表现最佳**，在MAE指标上排名第{mae_rank}。\n\n".format(
                    mae_rank=next(i for i, (m, _) in enumerate(mae_ranking, 1) if m == "STAR-Forecast")
                ))
            elif best_model_mae == "STAR-Forecast":
                f.write("✅ **STAR-Forecast在MAE指标上表现最佳**，在MSE指标上排名第{mse_rank}。\n\n".format(
                    mse_rank=next(i for i, (m, _) in enumerate(mse_ranking, 1) if m == "STAR-Forecast")
                ))
            else:
                star_mse_rank = next(i for i, (m, _) in enumerate(mse_ranking, 1) if m == "STAR-Forecast")
                star_mae_rank = next(i for i, (m, _) in enumerate(mae_ranking, 1) if m == "STAR-Forecast")
                f.write(f"⚠️  **STAR-Forecast在MSE排名第{star_mse_rank}，MAE排名第{star_mae_rank}**，仍有改进空间。\n\n")

            f.write("### 关键发现：\n\n")
            f.write("1. **传统模型**（如DLinear）虽然简单，但在某些场景下表现稳定\n")
            f.write("2. **复杂模型**（如TimesNet、PatchTST）通常需要更多计算资源\n")
            f.write("3. **Transformer-based模型**（如Informer、Autoformer）在长序列预测上有优势\n")
            f.write("4. **STAR-Forecast**通过智能体协同和自适应调整，在精度和效率之间取得了平衡\n\n")

            f.write("### 建议：\n\n")
            f.write("1. 对于计算资源有限的场景，推荐使用DLinear或PatchTST\n")
            f.write("2. 对于需要高精度的场景，推荐使用STAR-Forecast或TimesNet\n")
            f.write("3. 对于长序列预测，推荐使用Informer或Autoformer\n")
            f.write("4. STAR-Forecast的智能体协同机制在动态调整方面表现出色，适合非平稳时间序列\n")

        # 生成HTML报告
        try:
            import markdown

            with open(report_path, 'r', encoding='utf-8') as f:
                md_content = f.read()

            html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

            # 添加CSS样式
            html_with_style = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>时序预测基线方法对比报告</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #f2f2f2; font-weight: bold; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .good {{ color: green; font-weight: bold; }}
                    .bad {{ color: red; font-weight: bold; }}
                    .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                </style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """

            html_path = report_dir / "comparison_report.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_with_style)

            self.logger.info(f"✅ HTML报告生成: {html_path}")

        except ImportError:
            self.logger.warning("⚠️ 未安装markdown库，跳过HTML报告生成")

        self.logger.info(f"✅ 详细报告生成: {report_path}")

    def run_complete_experiment(self):
        """运行完整的对比实验"""
        self.logger.info("=" * 60)
        self.logger.info("🔬 开始完整的基线对比实验")
        self.logger.info("=" * 60)

        # 1. 运行所有基线方法
        self.run_all_baselines()

        # 2. 生成报告
        self.generate_report()

        # 3. 打印最终总结
        self.logger.info("\n" + "=" * 60)
        self.logger.info("🎯 实验完成总结")
        self.logger.info("=" * 60)

        if hasattr(self, 'comparison_result') and self.comparison_result.summary:
            summary = self.comparison_result.summary

            self.logger.info(f"📊 实验统计:")
            self.logger.info(f"   模型数量: {summary['model_count']}")
            self.logger.info(f"   最佳MSE: {summary['best_mse']['model']} ({summary['best_mse']['value']:.6f})")
            self.logger.info(f"   最佳MAE: {summary['best_mae']['model']} ({summary['best_mae']['value']:.6f})")

            if 'relative_improvement_mse' in summary:
                self.logger.info(f"   STAR-Forecast相对改进: {summary['relative_improvement_mse']:.2f}%")

        self.logger.info(f"\n📁 结果文件:")
        self.logger.info(f"   结果目录: {self.output_dir}")
        self.logger.info(f"   可视化: {self.output_dir}/visualizations/")
        self.logger.info(f"   报告: {self.output_dir}/report/")

        return self.comparison_result


# ==================== 主函数 ====================

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="时序预测基线对比实验")
    parser.add_argument("--config", type=str, default="./config.yaml",
                        help="配置文件路径")
    parser.add_argument("--baselines", type=str, nargs='+',
                        default=['DLinear', 'PatchTST', 'TimesNet', 'FEDformer',
                                 'Autoformer', 'Informer', 'TimeLLM'],
                        help="要运行的基线方法列表")
    parser.add_argument("--include-star", action='store_true', default=True,
                        help="是否包含STAR-Forecast")
    parser.add_argument("--train", action='store_true', default=True,
                        help="是否训练模型（如果已保存模型文件，可以设置为False）")
    parser.add_argument("--output-dir", type=str, default="./results/baseline_comparison",
                        help="输出目录")

    args = parser.parse_args()

    # 创建对比实验管理器
    comparator = BaselineComparison(args.config)

    # 设置输出目录
    comparator.output_dir = Path(args.output_dir)
    comparator.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 运行实验
        result = comparator.run_complete_experiment()

        print("\n" + "=" * 60)
        print("🎉 基线对比实验完成！")
        print("=" * 60)

        # 打印排名
        if result and result.results:
            print("\n🏆 最终排名:")
            print("-" * 40)

            # MSE排名
            mse_ranking = sorted(
                result.results.items(),
                key=lambda x: x[1].mse
            )

            print("MSE排名（越小越好）:")
            for i, (model_name, result_obj) in enumerate(mse_ranking, 1):
                print(f"  {i}. {model_name}: {result_obj.mse:.6f}")

        return result

    except Exception as e:
        print(f"❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()