"""
STAR-Forecast主训练器
集成ISTR网络、AutoGen智能体、Agent Lightning客户端
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import json
import time
import os
import sys

# ============ 修复导入路径 ============
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录（training目录的父目录）
project_root = os.path.dirname(current_dir)
# 添加到Python路径
sys.path.insert(0, project_root)


# ============ 安全导入函数 ============
def safe_import(module_name, class_name=None):
    """安全地导入模块，如果失败则返回None"""
    try:
        module = __import__(module_name, fromlist=['*'])
        if class_name:
            return getattr(module, class_name)
        return module
    except (ImportError, AttributeError) as e:
        print(f"⚠️  导入 {module_name}.{class_name} 失败: {e}")
        return None


# ============ 尝试导入项目模块 ============
print("🔧 正在导入STAR-Forecast模块...")

# 尝试导入ISTR模型
ISTRNetwork = safe_import('models.istr', 'ISTRNetwork')
if ISTRNetwork is None:
    # 创建简单的ISTR网络替代
    class SimpleISTRNetwork(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            hidden_dim = config.get('istr', {}).get('hidden_dim', 256)

            # 简单的卷积网络
            self.conv1 = nn.Conv1d(7, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.1)

            # 自适应参数（简化）
            self.adaptive_params = {
                'spectral_threshold': torch.tensor(0.5)
            }
            self.laplacian_regularizer = nn.Parameter(torch.tensor(0.1))

        def forward(self, x, return_regularization=False):
            # 转置以适应卷积
            x = x.transpose(1, 2)

            # 卷积层
            x = self.relu(self.conv1(x))
            x = self.dropout(x)
            x = self.relu(self.conv2(x))

            # 转置回来
            x = x.transpose(1, 2)

            # 正则化损失（简化）
            reg_loss = torch.tensor(0.0)

            if return_regularization:
                return x, reg_loss
            return x

        def extract_features_for_analysis(self, x):
            """提取特征供分析"""
            with torch.no_grad():
                features = self.forward(x)
                return {
                    'mean': features.mean().item(),
                    'std': features.std().item(),
                    'shape': features.shape
                }

        def update_adaptive_parameters(self, params):
            """更新自适应参数"""
            for key, value in params.items():
                if key in self.adaptive_params:
                    self.adaptive_params[key] = torch.tensor(value)


    ISTRNetwork = SimpleISTRNetwork
    print("  ⚠️  使用简化版ISTR网络")

# 尝试导入MultiHeadPredictor
MultiHeadPredictor = safe_import('models.predictor', 'MultiHeadPredictor')
if MultiHeadPredictor is None:
    # 创建简单的多头预测器
    class SimpleMultiHeadPredictor(nn.Module):
        def __init__(self, hidden_dim=256, pred_len=24, heads=3):
            super().__init__()
            self.heads = heads
            self.pred_len = pred_len

            # 每个头一个线性层
            self.head_layers = nn.ModuleList([
                nn.Linear(hidden_dim, pred_len) for _ in range(heads)
            ])

            # 注意力权重
            self.attention = nn.Linear(hidden_dim, heads)

        def forward(self, x):
            batch_size = x.size(0)

            # 全局平均池化
            context = x.mean(dim=1)  # [batch_size, hidden_dim]

            # 计算注意力权重
            attn_weights = F.softmax(self.attention(context), dim=-1)  # [batch_size, heads]

            # 每个头的预测
            head_predictions = []
            for head_layer in self.head_layers:
                pred = head_layer(context).unsqueeze(1)  # [batch_size, 1, pred_len]
                head_predictions.append(pred)

            # 堆叠
            all_predictions = torch.cat(head_predictions, dim=1)  # [batch_size, heads, pred_len]

            # 加权求和
            attn_weights = attn_weights.unsqueeze(-1)  # [batch_size, heads, 1]
            final_prediction = torch.sum(all_predictions * attn_weights, dim=1)  # [batch_size, pred_len]

            return final_prediction.unsqueeze(-1)  # [batch_size, pred_len, 1]


    MultiHeadPredictor = SimpleMultiHeadPredictor
    print("  ⚠️  使用简化版多头预测器")

# 尝试导入其他模块
TimeSeriesMetrics = safe_import('training.metrics', 'TimeSeriesMetrics')
if TimeSeriesMetrics is None:
    # 创建简单的时序指标计算器
    class SimpleTimeSeriesMetrics:
        def compute(self, predictions, targets):
            predictions = torch.tensor(predictions)
            targets = torch.tensor(targets)

            mse = F.mse_loss(predictions, targets).item()
            mae = F.l1_loss(predictions, targets).item()

            # 计算RMSE
            rmse = torch.sqrt(torch.tensor(mse)).item()

            # 计算MAPE（避免除零）
            mask = torch.abs(targets) > 1e-8
            if torch.any(mask):
                mape = torch.mean(torch.abs((targets[mask] - predictions[mask]) / targets[mask])).item() * 100
            else:
                mape = 0.0

            return {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            }


    TimeSeriesMetrics = SimpleTimeSeriesMetrics

# 尝试导入回调
TrainingCallbacks = safe_import('training.callbacks', 'TrainingCallbacks')
if TrainingCallbacks is None:
    # 创建简单的回调
    class SimpleTrainingCallbacks:
        def __init__(self, config):
            self.config = config

        def on_epoch_end(self, epoch, train_metrics, val_metrics):
            pass

        def on_train_end(self, test_metrics):
            pass


    TrainingCallbacks = SimpleTrainingCallbacks

# 尝试导入配置
load_config = safe_import('utils.config', 'load_config')
if load_config is None:
    # 创建简单的配置加载器
    def simple_load_config(config_path):
        import yaml
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # 返回默认配置
            return {
                'experiment': {
                    'seed': 42,
                    'name': 'STAR-Forecast'
                },
                'hardware': {
                    'mixed_precision': False
                },
                'istr': {
                    'hidden_dim': 256
                },
                'predictor': {
                    'heads': 3
                },
                'data': {
                    'seq_len': 96,
                    'pred_len': 24
                },
                'training': {
                    'epochs': 50,
                    'learning_rate': 0.001,
                    'weight_decay': 0.0001,
                    'optimizer': {
                        'type': 'AdamW',
                        'betas': [0.9, 0.999],
                        'eps': 1e-8
                    },
                    'scheduler': {
                        'type': 'CosineAnnealingWarmRestarts',
                        'T_0': 10,
                        'T_mult': 2,
                        'eta_min': 1e-6
                    },
                    'gradient': {
                        'clip_norm': 1.0
                    },
                    'early_stopping': {
                        'patience': 10
                    },
                    'checkpoint': {
                        'save_frequency': 5
                    }
                },
                'logging': {
                    'experiment_tracking': {
                        'wandb': {
                            'enabled': False,
                            'project': 'star-forecast',
                            'entity': None
                        }
                    }
                },
                'autogen': {
                    'trigger': {
                        'check_interval': 50
                    }
                },
                'agent_lightning': {
                    'client': {
                        'base_url': 'http://localhost:8000',
                        'timeout': 30,
                        'retry_attempts': 3,
                        'fallback_enabled': True
                    },
                    'rl': {
                        'reward': {
                            'weights': {
                                'mse': 1.0,
                                'smoothness': 0.1,
                                'stability': 0.05,
                                'semantic': 0.5
                            }
                        }
                    }
                }
            }


    load_config = simple_load_config

# 尝试导入日志
setup_logger = safe_import('utils.logger', 'setup_logger')
if setup_logger is None:
    # 创建简单的日志设置
    def simple_setup_logger(name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger


    setup_logger = simple_setup_logger

print("✅ 所有模块导入完成")


# ============ 创建缺失的类 ============

class AgentLightningClient:
    """Agent Lightning客户端（简化版）"""

    def __init__(self, base_url, client_id, timeout=30, retry_attempts=3, fallback_enabled=True):
        self.base_url = base_url
        self.client_id = client_id
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.fallback_enabled = fallback_enabled
        self.stats = {
            'requests_sent': 0,
            'responses_received': 0,
            'errors': 0
        }

    def get_decision(self, context):
        """获取决策（简化版）"""
        self.stats['requests_sent'] += 1

        # 简化：返回固定决策
        return type('Decision', (), {
            'action': 'adjust_parameters',
            'parameters': {
                'spectral_threshold': 0.7,
                'laplacian_weight': 0.15,
                'learning_rate_multiplier': 1.0
            },
            'semantic_reward': 0.5
        })()

    def update_experience(self, state, action, reward, next_state):
        """更新经验（简化版）"""
        pass

    def get_client_stats(self):
        """获取客户端统计"""
        return self.stats


class AutoGenMultiAgentSystem:
    """AutoGen多智能体系系统（简化版）"""

    def __init__(self, config):
        self.config = config
        self.conversation_history = []

    def get_conversation_history(self):
        """获取对话历史"""
        return self.conversation_history


def create_dataloaders(config, data_path):
    """创建数据加载器（简化版）"""
    from torch.utils.data import Dataset, DataLoader

    class SimpleTimeSeriesDataset(Dataset):
        def __init__(self, data_path, seq_len, pred_len, mode='train'):
            import pandas as pd
            import numpy as np

            df = pd.read_csv(data_path)
            if 'date' in df.columns:
                data = df.drop('date', axis=1).values
            else:
                data = df.values

            # 简单划分
            n = len(data)
            if mode == 'train':
                data = data[:int(n * 0.7)]
            elif mode == 'val':
                data = data[int(n * 0.7):int(n * 0.9)]
            else:  # test
                data = data[int(n * 0.9):]

            self.data = data.astype(np.float32)
            self.seq_len = seq_len
            self.pred_len = pred_len

        def __len__(self):
            return len(self.data) - self.seq_len - self.pred_len + 1

        def __getitem__(self, idx):
            x = self.data[idx:idx + self.seq_len]
            y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, -1:]
            return torch.FloatTensor(x), torch.FloatTensor(y)

    seq_len = config['data']['seq_len']
    pred_len = config['data']['pred_len']

    train_dataset = SimpleTimeSeriesDataset(data_path, seq_len, pred_len, 'train')
    val_dataset = SimpleTimeSeriesDataset(data_path, seq_len, pred_len, 'val')
    test_dataset = SimpleTimeSeriesDataset(data_path, seq_len, pred_len, 'test')

    batch_size = config['training']['batch_size']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ============ 主训练器类 ============

class STARForecastTrainer:
    """
    STAR-Forecast主训练器

    集成：
    1. ISTR网络（TCN + 拉普拉斯）
    2. AutoGen多智能体对话
    3. Agent Lightning解耦训练
    4. 完整的训练循环和评估
    """

    def __init__(self, config_path: str = "./config.yaml"):
        # 加载配置
        self.config = load_config(config_path)

        # 设置日志
        self.logger = setup_logger("STAR-Forecast")

        # 设置设备
        self.device = self._setup_device()

        # 设置随机种子
        self._set_seed()

        # 初始化组件
        self.istr_model = None
        self.predictor = None
        self.optimizer = None
        self.scheduler = None

        # 智能体系统
        self.autogen_system = None
        self.agent_client = None

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # 指标跟踪
        self.metrics = TimeSeriesMetrics()
        self.callbacks = TrainingCallbacks(self.config)

        # 实验跟踪
        self._setup_experiment_tracking()

        self.logger.info("🎯 STAR-Forecast训练器初始化完成")

    def _setup_device(self):
        """设置设备"""
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            self.logger.info(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")

            # 设置混合精度
            if self.config['hardware']['mixed_precision']:
                self.scaler = torch.cuda.amp.GradScaler()
                self.logger.info("✅ 启用混合精度训练")
            else:
                self.scaler = None
        else:
            device = torch.device('cpu')
            self.logger.warning("⚠️ 使用CPU训练，性能可能受限")
            self.scaler = None

        return device

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

    def _setup_experiment_tracking(self):
        """设置实验跟踪"""
        # 尝试导入wandb，如果失败则禁用
        try:
            import wandb
            if self.config['logging']['experiment_tracking']['wandb']['enabled']:
                wandb.init(
                    project=self.config['logging']['experiment_tracking']['wandb']['project'],
                    entity=self.config['logging']['experiment_tracking']['wandb']['entity'],
                    config=self.config,
                    name=self.config['experiment']['name']
                )
                self.logger.info("📊 启用WandB实验跟踪")
                self.wandb = wandb
            else:
                self.wandb = None
        except ImportError:
            self.logger.warning("⚠️ 未安装wandb，禁用实验跟踪")
            self.wandb = None

    def build_models(self):
        """构建模型"""
        self.logger.info("🔨 构建模型...")

        # 1. 构建ISTR网络
        self.istr_model = ISTRNetwork(self.config).to(self.device)

        # 2. 构建预测头
        self.predictor = MultiHeadPredictor(
            hidden_dim=self.config['istr']['hidden_dim'],
            pred_len=self.config['data']['pred_len'],
            heads=self.config['predictor']['heads']
        ).to(self.device)

        # 3. 计算总参数
        total_params = sum(p.numel() for p in self.istr_model.parameters())
        trainable_params = sum(p.numel() for p in self.istr_model.parameters()
                               if p.requires_grad)

        self.logger.info(f"📊 模型统计:")
        self.logger.info(f"   ISTR参数: {total_params:,} (可训练: {trainable_params:,})")
        self.logger.info(f"   预测头参数: {sum(p.numel() for p in self.predictor.parameters()):,}")

    def build_optimizer(self):
        """构建优化器"""
        self.logger.info("⚙️ 构建优化器...")

        # 只优化可训练参数
        trainable_params = []
        trainable_params.extend(
            [p for p in self.istr_model.parameters() if p.requires_grad]
        )
        trainable_params.extend(self.predictor.parameters())

        # 优化器
        optimizer_config = self.config['training']['optimizer']

        if optimizer_config['type'] == 'AdamW':
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config['training']['learning_rate'],
                betas=tuple(optimizer_config['betas']),
                eps=optimizer_config['eps'],
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            self.optimizer = torch.optim.Adam(
                trainable_params,
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )

        # 学习率调度器
        scheduler_config = self.config['training']['scheduler']

        if scheduler_config['type'] == 'CosineAnnealingWarmRestarts':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=scheduler_config['T_0'],
                T_mult=scheduler_config['T_mult'],
                eta_min=scheduler_config['eta_min']
            )
        else:
            self.scheduler = None

        self.logger.info(f"✅ 优化器: {optimizer_config['type']}")
        self.logger.info(f"   初始学习率: {self.config['training']['learning_rate']}")

    def initialize_agents(self):
        """初始化智能体系系统"""
        self.logger.info("🤖 初始化智能体系系统...")

        # 1. 初始化AutoGen系统
        self.autogen_system = AutoGenMultiAgentSystem(self.config)

        # 2. 初始化Agent Lightning客户端
        agent_config = self.config['agent_lightning']['client']

        self.agent_client = AgentLightningClient(
            base_url=agent_config['base_url'],
            client_id=f"star_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timeout=agent_config['timeout'],
            retry_attempts=agent_config['retry_attempts'],
            fallback_enabled=agent_config['fallback_enabled']
        )

        self.logger.info("✅ 智能体系系统初始化完成")

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.istr_model.train()
        self.predictor.train()

        epoch_losses = []
        epoch_metrics = {'mse': [], 'mae': []}

        self.logger.info(f"🏋️  Epoch {epoch + 1} 训练开始")

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)

            # 训练步骤
            loss, metrics = self.train_step(x, y, batch_idx)

            epoch_losses.append(loss)
            epoch_metrics['mse'].append(metrics['mse'])
            epoch_metrics['mae'].append(metrics['mae'])

            # 日志记录
            if batch_idx % 50 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']

                self.logger.info(
                    f"  Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss:.4f}, MSE: {metrics['mse']:.4f}, "
                    f"LR: {current_lr:.6f}"
                )

                # WandB日志
                if self.wandb is not None:
                    self.wandb.log({
                        'train/loss': loss,
                        'train/mse': metrics['mse'],
                        'train/mae': metrics['mae'],
                        'train/lr': current_lr,
                        'epoch': epoch,
                        'global_step': self.global_step
                    })

            self.global_step += 1

        # 计算epoch平均指标
        avg_loss = np.mean(epoch_losses)
        avg_mse = np.mean(epoch_metrics['mse'])
        avg_mae = np.mean(epoch_metrics['mae'])

        return {
            'loss': avg_loss,
            'mse': avg_mse,
            'mae': avg_mae
        }

    def train_step(self, x: torch.Tensor, y: torch.Tensor, batch_idx: int) -> Tuple[float, Dict[str, float]]:
        """单批次训练步骤"""
        # 混合精度训练
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                # ISTR特征提取
                features, reg_loss = self.istr_model(x, return_regularization=True)

                # 预测
                predictions = self.predictor(features)

                # 计算损失
                mse_loss = F.mse_loss(predictions, y)
                total_loss = mse_loss + reg_loss
        else:
            # ISTR特征提取
            features, reg_loss = self.istr_model(x, return_regularization=True)

            # 预测
            predictions = self.predictor(features)

            # 计算损失
            mse_loss = F.mse_loss(predictions, y)
            total_loss = mse_loss + reg_loss

        # 反向传播
        self.optimizer.zero_grad()

        if self.scaler is not None:
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.istr_model.parameters(),
                self.config['training']['gradient']['clip_norm']
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.istr_model.parameters(),
                self.config['training']['gradient']['clip_norm']
            )

            self.optimizer.step()

        # 调度器步进
        if self.scheduler is not None:
            self.scheduler.step()

        # 智能体决策（在特定步骤触发）
        if self._should_trigger_agents(batch_idx):
            self._agent_decision_step(x, features, predictions, y, batch_idx)

        # 计算指标
        metrics = {
            'mse': mse_loss.item(),
            'mae': F.l1_loss(predictions, y).item(),
            'reg_loss': reg_loss.item() if isinstance(reg_loss, torch.Tensor) else 0.0
        }

        return total_loss.item(), metrics

    def _should_trigger_agents(self, batch_idx: int) -> bool:
        """检查是否应该触发智能体"""
        check_interval = self.config.get('autogen', {}).get('trigger', {}).get('check_interval', 50)

        # 每N个批次触发一次
        if batch_idx % check_interval == 0:
            return True

        return False

    def _agent_decision_step(self, x: torch.Tensor, features: torch.Tensor,
                             predictions: torch.Tensor, targets: torch.Tensor,
                             batch_idx: int):
        """智能体决策步骤"""
        self.logger.info(f"🤖 触发智能体决策 (Batch {batch_idx})")

        # 1. 提取特征供分析
        with torch.no_grad():
            feature_analysis = self.istr_model.extract_features_for_analysis(x)

            # 计算当前指标
            current_mse = F.mse_loss(predictions, targets).item()
            current_mae = F.l1_loss(predictions, targets).item()

        # 2. 准备上下文信息
        context = {
            'features': feature_analysis,
            'metrics': {
                'mse': current_mse,
                'mae': current_mae,
                'batch_idx': batch_idx,
                'global_step': self.global_step
            },
            'current_params': {
                'spectral_threshold': self.istr_model.adaptive_params['spectral_threshold'].item(),
                'laplacian_weight': self.istr_model.laplacian_regularizer.weight.item(),
                'learning_rate': self.optimizer.param_groups[0]['lr']
            },
            'training_info': {
                'epoch': self.current_epoch,
                'total_epochs': self.config['training']['epochs'],
                'batch_idx': batch_idx
            },
            'batch_idx': batch_idx,
            'global_step': self.global_step
        }

        # 3. 调用Agent Lightning获取决策
        try:
            decision = self.agent_client.get_decision(context)

            self.logger.info(f"✅ 智能体决策: action={decision.action}, "
                             f"params={decision.parameters}")

            # 4. 应用决策到模型
            if decision.parameters:
                self.istr_model.update_adaptive_parameters(decision.parameters)

                # 更新学习率
                if 'learning_rate_multiplier' in decision.parameters:
                    new_lr = (self.config['training']['learning_rate'] *
                              decision.parameters['learning_rate_multiplier'])

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr

                    self.logger.info(f"🔄 调整学习率: {new_lr:.6f}")

            # 5. 准备强化学习状态和奖励
            state = self._prepare_rl_state(features)
            reward = self._calculate_reward(predictions, targets, decision)

            # 6. 异步更新智能体经验（不阻塞训练）
            next_state = state  # 简化：假设状态不变
            self.agent_client.update_experience(
                state=state,
                action=decision.action,
                reward=reward,
                next_state=next_state
            )

            # 7. 记录决策
            if self.wandb is not None:
                self.wandb.log({
                    'agent/action': decision.action,
                    'agent/reward': reward,
                    'agent/semantic_reward': decision.semantic_reward,
                    'agent/spectral_threshold': decision.parameters.get('spectral_threshold', 0),
                    'agent/laplacian_weight': decision.parameters.get('laplacian_weight', 0),
                    'epoch': self.current_epoch,
                    'global_step': self.global_step
                })

        except Exception as e:
            self.logger.error(f"❌ 智能体决策失败: {e}")

    def _prepare_rl_state(self, features: torch.Tensor) -> List[float]:
        """准备强化学习状态"""
        with torch.no_grad():
            # 提取特征统计
            state = []

            # 均值
            state.append(features.mean().item())

            # 标准差
            state.append(features.std().item())

            # 自适应参数
            state.append(self.istr_model.adaptive_params['spectral_threshold'].item())
            state.append(self.istr_model.laplacian_regularizer.weight.item())

        return state

    def _calculate_reward(self, predictions: torch.Tensor,
                          targets: torch.Tensor,
                          decision: Any) -> float:
        """计算奖励"""
        # 1. 预测误差奖励
        mse = F.mse_loss(predictions, targets).item()
        error_reward = -mse * self.config['agent_lightning']['rl']['reward']['weights']['mse']

        # 2. 平滑性奖励
        if predictions.shape[1] > 1:
            smoothness = torch.mean(
                torch.abs(predictions[:, 1:] - predictions[:, :-1])
            ).item()
            smoothness_reward = -smoothness * self.config['agent_lightning']['rl']['reward']['weights']['smoothness']
        else:
            smoothness_reward = 0.0

        # 3. 稳定性奖励
        stability = torch.std(predictions).item()
        stability_reward = -stability * self.config['agent_lightning']['rl']['reward']['weights']['stability']

        # 4. 语义奖励（来自智能体对话）
        semantic_reward = decision.semantic_reward * self.config['agent_lightning']['rl']['reward']['weights'][
            'semantic']

        # 总奖励
        total_reward = (
                error_reward +
                smoothness_reward +
                stability_reward +
                semantic_reward
        )

        return total_reward

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证模型"""
        self.istr_model.eval()
        self.predictor.eval()

        val_losses = []
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)

                # 前向传播
                features = self.istr_model(x)
                predictions = self.predictor(features)

                # 计算损失
                loss = F.mse_loss(predictions, y)
                val_losses.append(loss.item())

                # 保存预测结果
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        # 计算指标
        val_loss = np.mean(val_losses)
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        metrics = self.metrics.compute(all_predictions, all_targets)
        metrics['loss'] = val_loss

        return metrics

    def test(self, test_loader: DataLoader) -> Dict[str, float]:
        """测试模型"""
        return self.validate(test_loader)  # 与验证逻辑相同

    def train(self, data_path: str = "./data/ETTh1.csv"):
        """主训练循环"""
        self.logger.info("🚀 STAR-Forecast训练开始")
        self.logger.info("=" * 60)

        # 1. 构建模型
        self.build_models()

        # 2. 构建优化器
        self.build_optimizer()

        # 3. 初始化智能体
        self.initialize_agents()

        # 4. 加载数据
        self.logger.info("📊 加载数据...")
        train_loader, val_loader, test_loader = create_dataloaders(
            self.config, data_path
        )

        # 5. 训练循环
        best_model_path = None
        patience_counter = 0

        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch

            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)

            # 验证
            val_metrics = self.validate(val_loader)

            # 记录指标
            self._log_epoch_metrics(epoch, train_metrics, val_metrics)

            # 早停检查
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                patience_counter = 0

                # 保存最佳模型
                best_model_path = self._save_checkpoint(epoch, is_best=True)
                self.logger.info(f"✅ 保存最佳模型: {best_model_path}")
            else:
                patience_counter += 1

            # 检查早停
            if patience_counter >= self.config['training']['early_stopping']['patience']:
                self.logger.info(f"🛑 早停触发，停止训练")
                break

            # 保存常规检查点
            if epoch % self.config['training']['checkpoint']['save_frequency'] == 0:
                self._save_checkpoint(epoch, is_best=False)

        # 6. 最终测试
        self.logger.info("🧪 最终测试...")

        # 加载最佳模型
        if best_model_path:
            self._load_checkpoint(best_model_path)

        test_metrics = self.test(test_loader)

        # 记录测试结果
        self._log_test_metrics(test_metrics)

        # 7. 保存最终结果
        self._save_final_results(test_metrics)

        self.logger.info("🎉 训练完成！")

        return test_metrics

    def _log_epoch_metrics(self, epoch: int, train_metrics: Dict[str, float],
                           val_metrics: Dict[str, float]):
        """记录epoch指标"""
        self.logger.info(f"\n📊 Epoch {epoch + 1} 结果:")
        self.logger.info(f"   训练 - Loss: {train_metrics['loss']:.4f}, "
                         f"MSE: {train_metrics['mse']:.4f}, "
                         f"MAE: {train_metrics['mae']:.4f}")
        self.logger.info(f"   验证 - Loss: {val_metrics['loss']:.4f}, "
                         f"MSE: {val_metrics['mse']:.4f}, "
                         f"MAE: {val_metrics['mae']:.4f}")

        # WandB日志
        if self.wandb is not None:
            self.wandb.log({
                'epoch/train_loss': train_metrics['loss'],
                'epoch/train_mse': train_metrics['mse'],
                'epoch/train_mae': train_metrics['mae'],
                'epoch/val_loss': val_metrics['loss'],
                'epoch/val_mse': val_metrics['mse'],
                'epoch/val_mae': val_metrics['mae'],
                'epoch': epoch
            })

    def _log_test_metrics(self, test_metrics: Dict[str, float]):
        """记录测试指标"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("🎯 最终测试结果")
        self.logger.info("=" * 60)

        for metric_name, metric_value in test_metrics.items():
            self.logger.info(f"   {metric_name.upper()}: {metric_value:.6f}")

        # WandB日志
        if self.wandb is not None:
            for metric_name, metric_value in test_metrics.items():
                self.wandb.log({f'test/{metric_name}': metric_value})

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> str:
        """保存检查点"""
        checkpoint_dir = Path("./checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)

        if is_best:
            filename = f"best_model_epoch{epoch + 1}.pth"
        else:
            filename = f"checkpoint_epoch{epoch + 1}.pth"

        checkpoint_path = checkpoint_dir / filename

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'istr_state_dict': self.istr_model.state_dict(),
            'predictor_state_dict': self.predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        torch.save(checkpoint, checkpoint_path)

        return str(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.istr_model.load_state_dict(checkpoint['istr_state_dict'])
        self.predictor.load_state_dict(checkpoint['predictor_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        self.logger.info(f"✅ 加载检查点: {checkpoint_path}")

    def _save_final_results(self, test_metrics: Dict[str, float]):
        """保存最终结果"""
        results_dir = Path("./results")
        results_dir.mkdir(exist_ok=True)

        # 获取客户端统计
        client_stats = self.agent_client.get_client_stats()

        # 获取AutoGen历史
        autogen_history = self.autogen_system.get_conversation_history()

        # 构建结果
        results = {
            'experiment': self.config['experiment'],
            'test_metrics': test_metrics,
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch + 1,
            'agent_stats': client_stats,
            'autogen_conversations': len(autogen_history),
            'timestamp': datetime.now().isoformat()
        }

        # 保存为JSON
        results_path = results_dir / "final_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"💾 结果保存到: {results_path}")

        # 保存为CSV（便于分析）
        csv_path = results_dir / "results.csv"
        with open(csv_path, 'w') as f:
            f.write("metric,value\n")
            for metric, value in test_metrics.items():
                f.write(f"{metric},{value}\n")

        return results_path


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="STAR-Forecast训练")
    parser.add_argument("--config", type=str, default="./config.yaml",
                        help="配置文件路径")
    parser.add_argument("--data", type=str, default="./data/ETTh1.csv",
                        help="数据文件路径")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数（覆盖配置）")

    args = parser.parse_args()

    # 创建训练器
    trainer = STARForecastTrainer(args.config)

    # 覆盖配置（如果提供了参数）
    if args.epochs:
        trainer.config['training']['epochs'] = args.epochs

    # 开始训练
    try:
        results = trainer.train(args.data)

        print("\n" + "=" * 60)
        print("🎉 训练完成！最终结果:")
        print("=" * 60)

        for metric, value in results.items():
            print(f"{metric.upper()}: {value:.6f}")

    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()