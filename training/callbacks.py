"""
callbacks.py - 真实训练回调系统
支持智能体交互、模型检查点、早停等
"""
import torch
import numpy as np
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings('ignore')


@dataclass
class CallbackState:
    """回调状态容器"""
    epoch: int = 0
    batch_idx: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    model_state: Optional[Dict[str, Any]] = None
    optimizer_state: Optional[Dict[str, Any]] = None


class BaseCallback:
    """回调基类"""

    def __init__(self):
        self.name = self.__class__.__name__

    def on_train_begin(self, state: CallbackState):
        """训练开始时调用"""
        pass

    def on_train_end(self, state: CallbackState):
        """训练结束时调用"""
        pass

    def on_epoch_begin(self, state: CallbackState):
        """每个epoch开始时调用"""
        pass

    def on_epoch_end(self, state: CallbackState):
        """每个epoch结束时调用"""
        pass

    def on_batch_begin(self, state: CallbackState):
        """每个batch开始时调用"""
        pass

    def on_batch_end(self, state: CallbackState):
        """每个batch结束时调用"""
        pass

    def on_validation_begin(self, state: CallbackState):
        """验证开始时调用"""
        pass

    def on_validation_end(self, state: CallbackState):
        """验证结束时调用"""
        pass


class AgentInteractionCallback(BaseCallback):
    """
    智能体交互回调
    在训练过程中定期触发AutoGen智能体分析
    """

    def __init__(self,
                 agent_client,
                 check_interval: int = 50,
                 min_epoch: int = 1):
        """
        初始化

        Args:
            agent_client: Agent Lightning客户端
            check_interval: 触发间隔（批次数）
            min_epoch: 最小epoch数（前几个epoch不触发）
        """
        super().__init__()
        self.agent_client = agent_client
        self.check_interval = check_interval
        self.min_epoch = min_epoch
        self.interaction_count = 0
        self.last_interaction_batch = -1

        print(f"✅ 初始化智能体交互回调，间隔={check_interval}")

    def on_batch_end(self, state: CallbackState):
        """在每个batch结束时检查是否需要智能体交互"""
        # 检查条件
        if state.epoch < self.min_epoch:
            return

        if state.batch_idx % self.check_interval != 0:
            return

        # 避免重复触发
        if state.batch_idx == self.last_interaction_batch:
            return

        self.last_interaction_batch = state.batch_idx

        # 触发智能体交互
        self._trigger_agent_interaction(state)

    def _trigger_agent_interaction(self, state: CallbackState):
        """触发智能体交互"""
        self.interaction_count += 1

        print(f"\n🤖 智能体交互 #{self.interaction_count} "
              f"(Epoch {state.epoch}, Batch {state.batch_idx})")

        # 准备上下文（这里需要模型提供特征）
        context = self._prepare_agent_context(state)

        # 获取决策
        decision = self.agent_client.get_decision(context)

        # 记录决策
        decision_record = {
            'epoch': state.epoch,
            'batch': state.batch_idx,
            'decision_id': decision.decision_id,
            'action': decision.action,
            'parameters': decision.parameters,
            'confidence': decision.confidence,
            'reasoning': decision.reasoning,
            'timestamp': time.time()
        }

        # 保存决策记录
        self._save_decision_record(decision_record)

        # 返回决策信息（供外部使用）
        state.metrics['agent_decision'] = decision.action
        state.metrics['agent_confidence'] = decision.confidence

    def _prepare_agent_context(self, state: CallbackState) -> Dict[str, Any]:
        """准备智能体上下文"""
        # 这里应该从模型获取特征，简化实现
        context = {
            'epoch': state.epoch,
            'batch_idx': state.batch_idx,
            'metrics': {
                'train_loss': state.train_loss,
                'val_loss': state.val_loss if state.val_loss else 0.0
            },
            'current_params': {
                'spectral_threshold': 0.5,  # 应该从模型获取
                'laplacian_weight': 0.01
            },
            'features': {
                'shape': [32, 96, 7],  # 简化
                'statistics': {'mean': 0.0, 'std': 1.0}
            }
        }

        return context

    def _save_decision_record(self, record: Dict[str, Any]):
        """保存决策记录"""
        # 确保目录存在
        os.makedirs('./logs/agent_decisions', exist_ok=True)

        # 保存文件
        filename = f"./logs/agent_decisions/decision_{record['epoch']}_{record['batch']}.json"
        with open(filename, 'w') as f:
            json.dump(record, f, indent=2)

        print(f"📝 保存决策记录: {filename}")


class ModelCheckpoint(BaseCallback):
    """模型检查点回调"""

    def __init__(self,
                 save_dir: str = './checkpoints',
                 save_best_only: bool = True,
                 monitor: str = 'val_loss',
                 mode: str = 'min'):
        """
        初始化

        Args:
            save_dir: 保存目录
            save_best_only: 是否只保存最佳模型
            monitor: 监控的指标
            mode: 'min' 或 'max'
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0

        print(f"✅ 初始化模型检查点，监控指标: {monitor}")

    def on_epoch_end(self, state: CallbackState):
        """每个epoch结束时保存模型"""
        if self.monitor not in state.metrics:
            print(f"⚠️  监控指标 {self.monitor} 不存在")
            return

        current_value = state.metrics[self.monitor]

        # 判断是否是最佳值
        is_best = False
        if self.mode == 'min':
            if current_value < self.best_value:
                self.best_value = current_value
                self.best_epoch = state.epoch
                is_best = True
        else:  # max
            if current_value > self.best_value:
                self.best_value = current_value
                self.best_epoch = state.epoch
                is_best = True

        # 决定是否保存
        should_save = not self.save_best_only or is_best

        if should_save:
            self._save_checkpoint(state, is_best)

    def _save_checkpoint(self, state: CallbackState, is_best: bool):
        """保存检查点"""
        checkpoint = {
            'epoch': state.epoch,
            'model_state_dict': state.model_state,
            'optimizer_state_dict': state.optimizer_state,
            'metrics': state.metrics,
            'best_value': self.best_value,
            'best_epoch': self.best_epoch
        }

        # 基础文件名
        if is_best:
            filename = f"best_model_epoch{state.epoch}.pth"
        else:
            filename = f"checkpoint_epoch{state.epoch}.pth"

        # 完整路径
        filepath = self.save_dir / filename

        # 保存
        torch.save(checkpoint, filepath)

        print(f"💾 保存检查点: {filepath}")

        # 如果是最好模型，保存额外信息
        if is_best:
            info_file = self.save_dir / f"best_model_info.json"
            info = {
                'epoch': state.epoch,
                'value': self.best_value,
                'metrics': state.metrics,
                'timestamp': time.time()
            }
            with open(info_file, 'w') as f:
                json.dump(info, f, indent=2)


class EarlyStopping(BaseCallback):
    """早停回调"""

    def __init__(self,
                 patience: int = 20,
                 min_delta: float = 1e-4,
                 monitor: str = 'val_loss',
                 mode: str = 'min'):
        """
        初始化

        Args:
            patience: 耐心值（多少个epoch没有改善）
            min_delta: 最小改善值
            monitor: 监控指标
            mode: 'min' 或 'max'
        """
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode

        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.should_stop = False
        self.stopped_epoch = 0

        print(f"✅ 初始化早停，耐心值={patience}")

    def on_epoch_end(self, state: CallbackState):
        """检查是否应该早停"""
        if self.monitor not in state.metrics:
            return

        current_value = state.metrics[self.monitor]

        # 检查是否改善
        if self.mode == 'min':
            improvement = self.best_value - current_value
            if improvement > self.min_delta:
                self.best_value = current_value
                self.counter = 0
            else:
                self.counter += 1
        else:  # max
            improvement = current_value - self.best_value
            if improvement > self.min_delta:
                self.best_value = current_value
                self.counter = 0
            else:
                self.counter += 1

        # 检查是否应该停止
        if self.counter >= self.patience:
            self.should_stop = True
            self.stopped_epoch = state.epoch

            print(f"🛑 早停触发！在epoch {state.epoch}停止训练")
            print(f"   最佳 {self.monitor}: {self.best_value:.6f}")

    def on_train_end(self, state: CallbackState):
        """训练结束时记录早停信息"""
        if self.should_stop:
            print(f"🏁 训练因早停而结束，最佳epoch: {self.stopped_epoch - self.patience}")


class LearningRateScheduler(BaseCallback):
    """学习率调度回调"""

    def __init__(self,
                 optimizer,
                 scheduler_type: str = 'plateau',
                 patience: int = 10,
                 factor: float = 0.5,
                 min_lr: float = 1e-6):
        """
        初始化

        Args:
            optimizer: 优化器
            scheduler_type: 调度器类型 ('plateau', 'step', 'cosine')
            patience: 耐心值
            factor: 调整因子
            min_lr: 最小学习率
        """
        super().__init__()
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr

        # 创建调度器
        if scheduler_type == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=patience, factor=factor, min_lr=min_lr
            )
        elif scheduler_type == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=patience, gamma=factor
            )
        elif scheduler_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=patience, eta_min=min_lr
            )
        else:
            raise ValueError(f"未知的调度器类型: {scheduler_type}")

        print(f"✅ 初始化学习率调度器: {scheduler_type}")

    def on_epoch_end(self, state: CallbackState):
        """更新学习率"""
        if self.scheduler_type == 'plateau':
            # ReduceLROnPlateau需要验证损失
            if 'val_loss' in state.metrics:
                self.scheduler.step(state.metrics['val_loss'])
            else:
                self.scheduler.step(state.train_loss)
        else:
            # 其他调度器
            self.scheduler.step()

        # 记录当前学习率
        current_lr = self.optimizer.param_groups[0]['lr']
        state.metrics['learning_rate'] = current_lr

        # 打印学习率变化
        if current_lr < self.min_lr * 1.1:  # 接近最小值
            print(f"📉 学习率接近最小值: {current_lr:.6f}")


class MetricsLogger(BaseCallback):
    """指标记录回调"""

    def __init__(self,
                 log_dir: str = './logs',
                 log_interval: int = 10):
        """
        初始化

        Args:
            log_dir: 日志目录
            log_interval: 记录间隔（epoch）
        """
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_interval = log_interval

        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 日志文件
        self.csv_file = self.log_dir / 'training_log.csv'
        self.json_file = self.log_dir / 'training_history.json'

        # 初始化CSV文件
        if not self.csv_file.exists():
            with open(self.csv_file, 'w') as f:
                f.write('epoch,train_loss,val_loss,learning_rate,timestamp\n')

        self.history = []

        print(f"✅ 初始化指标记录器，日志目录: {log_dir}")

    def on_epoch_end(self, state: CallbackState):
        """记录指标"""
        # 收集指标
        log_entry = {
            'epoch': state.epoch,
            'train_loss': state.train_loss,
            'val_loss': state.val_loss,
            'timestamp': time.time()
        }

        # 添加其他指标
        for key, value in state.metrics.items():
            if key not in log_entry:
                log_entry[key] = value

        # 添加到历史
        self.history.append(log_entry)

        # 定期保存
        if state.epoch % self.log_interval == 0 or state.epoch == 1:
            self._save_logs()

    def on_train_end(self, state: CallbackState):
        """训练结束时保存所有日志"""
        self._save_logs()
        print(f"📊 训练日志保存完成: {self.json_file}")

    def _save_logs(self):
        """保存日志到文件"""
        # 保存为JSON
        with open(self.json_file, 'w') as f:
            json.dump(self.history, f, indent=2)

        # 更新CSV
        with open(self.csv_file, 'w') as f:
            # 写入标题
            if self.history:
                headers = list(self.history[0].keys())
                f.write(','.join(headers) + '\n')

                # 写入数据
                for entry in self.history:
                    row = [str(entry.get(h, '')) for h in headers]
                    f.write(','.join(row) + '\n')


class ProgressBar(BaseCallback):
    """进度条回调"""

    def __init__(self, total_epochs: int):
        """
        初始化

        Args:
            total_epochs: 总epoch数
        """
        super().__init__()
        self.total_epochs = total_epochs
        self.start_time = None
        self.current_epoch = 0

    def on_train_begin(self, state: CallbackState):
        """训练开始时"""
        self.start_time = time.time()
        print(f"🚀 开始训练，总epochs: {self.total_epochs}")
        print("=" * 60)

    def on_epoch_begin(self, state: CallbackState):
        """epoch开始时"""
        self.current_epoch = state.epoch
        print(f"\nEpoch {state.epoch}/{self.total_epochs}")
        print("-" * 40)

    def on_epoch_end(self, state: CallbackState):
        """epoch结束时显示进度"""
        elapsed = time.time() - self.start_time
        epoch_time = elapsed / max(state.epoch, 1)
        remaining = (self.total_epochs - state.epoch) * epoch_time

        print(f"  训练损失: {state.train_loss:.6f}")
        if state.val_loss:
            print(f"  验证损失: {state.val_loss:.6f}")

        # 显示其他重要指标
        for key in ['mse', 'mae', 'learning_rate']:
            if key in state.metrics:
                print(f"  {key}: {state.metrics[key]:.6f}")

        print(f"  已用时: {elapsed:.1f}s，预计剩余: {remaining:.1f}s")

    def on_train_end(self, state: CallbackState):
        """训练结束时"""
        total_time = time.time() - self.start_time
        print("=" * 60)
        print(f"🏁 训练完成！总用时: {total_time:.1f}s")
        print(f"   最佳验证损失: {min([h.get('val_loss', float('inf')) for h in getattr(self, 'history', [])])}")
        print("=" * 60)


class CallbackHandler:
    """回调处理器"""

    def __init__(self, callbacks: List[BaseCallback]):
        """
        初始化

        Args:
            callbacks: 回调列表
        """
        self.callbacks = callbacks
        self.state = CallbackState()

        print(f"✅ 初始化回调处理器，包含 {len(callbacks)} 个回调")

    def set_model_optimizer(self, model, optimizer):
        """设置模型和优化器"""
        self.model = model
        self.optimizer = optimizer

    def on_train_begin(self, **kwargs):
        """训练开始时调用所有回调"""
        self.state = CallbackState()
        for callback in self.callbacks:
            callback.on_train_begin(self.state)

    def on_train_end(self, **kwargs):
        """训练结束时调用所有回调"""
        for callback in self.callbacks:
            callback.on_train_end(self.state)

    def on_epoch_begin(self, epoch: int, **kwargs):
        """epoch开始时"""
        self.state.epoch = epoch
        for callback in self.callbacks:
            callback.on_epoch_begin(self.state)

    def on_epoch_end(self, train_loss: float, val_loss: float = None, metrics: Dict = None, **kwargs):
        """epoch结束时"""
        self.state.train_loss = train_loss
        self.state.val_loss = val_loss
        self.state.metrics = metrics or {}

        # 保存模型和优化器状态
        if hasattr(self, 'model'):
            self.state.model_state = self.model.state_dict()
        if hasattr(self, 'optimizer'):
            self.state.optimizer_state = self.optimizer.state_dict()

        for callback in self.callbacks:
            callback.on_epoch_end(self.state)

    def on_batch_begin(self, batch_idx: int, **kwargs):
        """batch开始时"""
        self.state.batch_idx = batch_idx
        for callback in self.callbacks:
            callback.on_batch_begin(self.state)

    def on_batch_end(self, loss: float = None, **kwargs):
        """batch结束时"""
        if loss is not None:
            self.state.train_loss = loss
        for callback in self.callbacks:
            callback.on_batch_end(self.state)

    def on_validation_begin(self, **kwargs):
        """验证开始时"""
        for callback in self.callbacks:
            callback.on_validation_begin(self.state)

    def on_validation_end(self, val_loss: float, metrics: Dict = None, **kwargs):
        """验证结束时"""
        self.state.val_loss = val_loss
        self.state.metrics = metrics or {}
        for callback in self.callbacks:
            callback.on_validation_end(self.state)


def create_default_callbacks(config: Dict[str, Any],
                             model=None,
                             optimizer=None,
                             agent_client=None) -> CallbackHandler:
    """
    创建默认回调集合

    Args:
        config: 配置字典
        model: 模型（可选）
        optimizer: 优化器（可选）
        agent_client: 智能体客户端（可选）

    Returns:
        回调处理器
    """
    callbacks = []

    # 进度条
    total_epochs = config.get('training', {}).get('epochs', 100)
    callbacks.append(ProgressBar(total_epochs))

    # 指标记录器
    log_dir = config.get('logging', {}).get('log_dir', './logs')
    callbacks.append(MetricsLogger(log_dir=log_dir))

    # 模型检查点
    checkpoint_dir = config.get('logging', {}).get('checkpoint_dir', './checkpoints')
    monitor = config.get('training', {}).get('early_stopping', {}).get('monitor', 'val_loss')
    callbacks.append(ModelCheckpoint(
        save_dir=checkpoint_dir,
        monitor=monitor,
        mode='min'
    ))

    # 早停
    early_stop_config = config.get('training', {}).get('early_stopping', {})
    if early_stop_config.get('enabled', True):
        callbacks.append(EarlyStopping(
            patience=early_stop_config.get('patience', 20),
            min_delta=early_stop_config.get('min_delta', 1e-4),
            monitor=monitor,
            mode='min'
        ))

    # 学习率调度器（需要优化器）
    if optimizer is not None:
        scheduler_config = config.get('training', {}).get('scheduler', {})
        callbacks.append(LearningRateScheduler(
            optimizer=optimizer,
            scheduler_type=scheduler_config.get('type', 'plateau'),
            patience=scheduler_config.get('patience', 10),
            factor=scheduler_config.get('factor', 0.5),
            min_lr=1e-6
        ))

    # 智能体交互回调（需要智能体客户端）
    if agent_client is not None:
        autogen_config = config.get('autogen', {})
        callbacks.append(AgentInteractionCallback(
            agent_client=agent_client,
            check_interval=autogen_config.get('check_interval', 50),
            min_epoch=1
        ))

    # 创建处理器
    handler = CallbackHandler(callbacks)

    # 设置模型和优化器
    if model is not None and optimizer is not None:
        handler.set_model_optimizer(model, optimizer)

    return handler


if __name__ == "__main__":
    # 测试回调系统
    print("测试回调系统...")

    # 模拟配置
    config = {
        'training': {
            'epochs': 5,
            'early_stopping': {'enabled': True, 'patience': 3},
            'scheduler': {'type': 'plateau', 'patience': 2}
        },
        'logging': {
            'log_dir': './test_logs',
            'checkpoint_dir': './test_checkpoints'
        },
        'autogen': {'check_interval': 2}
    }

    # 创建模拟模型和优化器
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 创建回调处理器
    handler = create_default_callbacks(config, model, optimizer, agent_client=None)

    # 模拟训练过程
    handler.on_train_begin()

    for epoch in range(1, 4):
        handler.on_epoch_begin(epoch)

        # 模拟batch训练
        for batch in range(1, 4):
            handler.on_batch_begin(batch)

            # 模拟损失
            loss = 1.0 / (epoch * batch)
            handler.on_batch_end(loss)

        # epoch结束
        val_loss = 0.5 / epoch
        metrics = {'mse': 0.1 / epoch, 'mae': 0.2 / epoch}
        handler.on_epoch_end(train_loss=0.1 / epoch, val_loss=val_loss, metrics=metrics)

    handler.on_train_end()

    print("\n✅ 回调系统测试完成")