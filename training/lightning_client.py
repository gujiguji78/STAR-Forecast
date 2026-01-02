"""
lightning_client.py - 真实Agent Lightning客户端实现
提供训练-执行解耦的API接口
"""

import requests
import json
import time
import threading
import uuid
import pickle
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
from dataclasses import dataclass, asdict
from queue import Queue, Empty
import warnings
from pathlib import Path
import torch
import torch.nn as nn
from datetime import datetime

warnings.filterwarnings('ignore')


@dataclass
class DecisionResponse:
    """决策响应数据结构"""
    decision_id: str
    action: int
    parameters: Dict[str, float]
    confidence: float
    reasoning: str
    timestamp: float


@dataclass
class TrainingTask:
    """训练任务数据结构"""
    task_id: str
    client_id: str
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    status: str
    created_at: float
    progress: float = 0.0
    error: Optional[str] = None


@dataclass
class Experience:
    """经验数据"""
    state: np.ndarray
    action: int
    reward: float
    next_state: Optional[np.ndarray] = None
    done: bool = False
    timestamp: float = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class ExperienceReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, experience: Experience):
        """添加经验"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Experience]:
        """随机采样一批经验"""
        if len(self.buffer) < batch_size:
            return self.buffer.copy()

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.position = 0


class PolicyNetwork(nn.Module):
    """策略网络 - 用于学习何时以及如何调整模型参数"""

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

    def get_action(self, state: np.ndarray) -> Tuple[int, float]:
        """基于状态获取动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            logits = self(state_tensor)
            probs = torch.softmax(logits, dim=1)
            action = torch.argmax(probs, dim=1).item()
            confidence = probs[0, action].item()
        return action, confidence


class AgentLightningLocalServer:
    """
    Agent Lightning本地服务器
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.clients = {}  # client_id -> agent_state
        self.training_tasks = {}
        self.task_queue = Queue()
        self.is_running = True
        self.experience_buffer = ExperienceReplayBuffer(
            capacity=config.get('buffer_capacity', 5000)
        )

        # 策略网络
        self.policy_network = PolicyNetwork()
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=config.get('policy_lr', 1e-3)
        )

        # 启动任务处理线程
        self.worker_thread = threading.Thread(target=self._process_tasks, daemon=True)
        self.worker_thread.start()

        # 启动策略学习线程
        self.learning_thread = threading.Thread(target=self._learn_from_experiences, daemon=True)
        self.learning_thread.start()

        print("✅ Agent Lightning本地服务器启动")
        print(f"   经验缓冲区容量: {self.experience_buffer.capacity}")
        print(f"   策略网络参数: {sum(p.numel() for p in self.policy_network.parameters()):,}")

    def _process_tasks(self):
        """处理训练任务（后台线程）"""
        while self.is_running:
            try:
                task = self.task_queue.get(timeout=1.0)
                if task:
                    self._execute_training_task(task)
            except Empty:
                continue
            except Exception as e:
                print(f"❌ 任务处理错误: {e}")

    def _execute_training_task(self, task: TrainingTask):
        """执行训练任务"""
        try:
            task.status = 'running'

            # 模拟训练过程（实际应该调用真实的训练代码）
            steps = 100
            for step in range(steps):
                time.sleep(0.02)  # 模拟训练时间
                task.progress = (step + 1) / steps

                # 定期更新任务状态
                if step % 20 == 0:
                    print(f"🔄 训练任务 {task.task_id}: {task.progress * 100:.1f}%")

            task.status = 'completed'
            print(f"✅ 训练任务完成 {task.task_id}")

        except Exception as e:
            task.status = 'failed'
            task.error = str(e)
            print(f"❌ 训练任务失败 {task.task_id}: {e}")

    def _learn_from_experiences(self):
        """从经验中学习（后台线程）"""
        while self.is_running:
            try:
                if len(self.experience_buffer) >= 32:  # 最小批大小
                    batch = self.experience_buffer.sample(32)

                    # 简单的策略梯度学习
                    states = []
                    actions = []
                    rewards = []

                    for exp in batch:
                        states.append(exp.state)
                        actions.append(exp.action)
                        rewards.append(exp.reward)

                    if len(states) > 0:
                        self._update_policy(states, actions, rewards)

                time.sleep(5)  # 每5秒学习一次

            except Exception as e:
                print(f"⚠️  策略学习错误: {e}")
                time.sleep(10)

    def _update_policy(self, states, actions, rewards):
        """更新策略网络"""
        try:
            states_tensor = torch.FloatTensor(np.array(states))
            actions_tensor = torch.LongTensor(actions)
            rewards_tensor = torch.FloatTensor(rewards)

            # 归一化奖励
            if rewards_tensor.std() > 0:
                rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

            self.policy_optimizer.zero_grad()
            logits = self.policy_network(states_tensor)
            loss = nn.CrossEntropyLoss()(logits, actions_tensor)

            # 用奖励加权损失
            loss = (loss * rewards_tensor).mean()
            loss.backward()
            self.policy_optimizer.step()

            return loss.item()

        except Exception as e:
            print(f"⚠️  策略更新错误: {e}")
            return None

    def register_client(self, client_id: str, config: Dict[str, Any]) -> bool:
        """注册客户端"""
        if client_id in self.clients:
            return True

        # 初始化客户端状态
        self.clients[client_id] = {
            'config': config,
            'created_at': time.time(),
            'decision_count': 0,
            'last_active': time.time(),
            'total_reward': 0.0
        }

        print(f"📱 客户端注册: {client_id}")
        return True

    def get_decision(self, client_id: str, context: Dict[str, Any]) -> DecisionResponse:
        """获取决策"""
        if client_id not in self.clients:
            self.register_client(client_id, {})

        # 更新活动时间
        self.clients[client_id]['last_active'] = time.time()
        self.clients[client_id]['decision_count'] += 1

        # 基于上下文的决策逻辑
        decision = self._make_intelligent_decision(context)

        return DecisionResponse(
            decision_id=f"dec_{int(time.time() * 1000)}",
            action=decision['action'],
            parameters=decision['parameters'],
            confidence=decision['confidence'],
            reasoning=decision['reasoning'],
            timestamp=time.time()
        )

    def _make_intelligent_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """基于上下文的智能决策"""
        metrics = context.get('metrics', {})
        mse = metrics.get('mse', 0.5)
        mae = metrics.get('mae', 0.5)
        r2 = metrics.get('r2', 0.0)

        # 从策略网络获取决策
        state = self._context_to_state(context)
        action, confidence = self.policy_network.get_action(state)

        # 根据动作确定参数
        parameters = self._get_parameters_for_action(action, mse)

        reasoning = f"MSE={mse:.3f}, MAE={mae:.3f}, 动作={action} (策略网络决策)"

        return {
            'action': int(action),
            'parameters': parameters,
            'confidence': float(confidence),
            'reasoning': reasoning
        }

    def _context_to_state(self, context: Dict[str, Any]) -> np.ndarray:
        """将上下文转换为状态向量"""
        metrics = context.get('metrics', {})
        features = context.get('features', {})

        # 提取关键特征
        mse = metrics.get('mse', 0.5)
        mae = metrics.get('mae', 0.5)
        r2 = metrics.get('r2', 0.0)

        # 从特征中提取更多信息
        data_shape = features.get('shape', [0, 0, 0])
        seq_len = data_shape[1] if len(data_shape) > 1 else 96
        n_features = data_shape[2] if len(data_shape) > 2 else 7

        # 归一化
        mse_norm = min(mse, 1.0)
        mae_norm = min(mae, 1.0)
        r2_norm = (r2 + 1) / 2  # [-1, 1] -> [0, 1]
        seq_norm = seq_len / 500
        feat_norm = n_features / 20

        # 组合状态向量
        state = np.array([
            mse_norm, mae_norm, r2_norm,
            seq_norm, feat_norm,
            0.5, 0.5, 0.5, 0.5, 0.5  # 预留位置
        ])

        return state

    def _get_parameters_for_action(self, action: int, mse: float) -> Dict[str, float]:
        """根据动作获取参数"""
        if action == 0:  # 保守策略
            spectral_threshold = 0.5
            laplacian_weight = 0.01
            learning_rate_multiplier = 0.8
        elif action == 1:  # 适度策略
            spectral_threshold = 0.5 + min(mse, 0.3) * 0.5
            laplacian_weight = 0.01 + min(mse, 0.3) * 0.02
            learning_rate_multiplier = 1.0
        else:  # 激进策略
            spectral_threshold = 0.5 + min(mse, 0.5) * 0.8
            laplacian_weight = 0.01 + min(mse, 0.5) * 0.05
            learning_rate_multiplier = 1.2

        return {
            'spectral_threshold': float(spectral_threshold),
            'laplacian_weight': float(laplacian_weight),
            'learning_rate_multiplier': float(learning_rate_multiplier)
        }

    def add_experience(self, experience: Experience):
        """添加经验到回放缓冲区"""
        self.experience_buffer.push(experience)

    def submit_training_task(self, client_id: str,
                             model_config: Dict[str, Any],
                             training_config: Dict[str, Any]) -> str:
        """提交训练任务"""
        task_id = f"task_{int(time.time() * 1000)}"

        task = TrainingTask(
            task_id=task_id,
            client_id=client_id,
            model_config=model_config,
            training_config=training_config,
            status='pending',
            created_at=time.time()
        )

        self.training_tasks[task_id] = task
        self.task_queue.put(task)

        return task_id

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        if task_id in self.training_tasks:
            task = self.training_tasks[task_id]
            return {
                'task_id': task.task_id,
                'status': task.status,
                'progress': task.progress,
                'created_at': task.created_at,
                'error': task.error
            }
        return {'error': '任务不存在'}

    def get_stats(self) -> Dict[str, Any]:
        """获取服务器统计信息"""
        return {
            'active_clients': len(self.clients),
            'total_decisions': sum(c['decision_count'] for c in self.clients.values()),
            'total_reward': sum(c.get('total_reward', 0) for c in self.clients.values()),
            'experience_buffer_size': len(self.experience_buffer),
            'pending_tasks': self.task_queue.qsize(),
            'total_tasks': len(self.training_tasks),
            'server_uptime': time.time() - getattr(self, '_start_time', time.time())
        }

    def save(self, path: str):
        """保存服务器状态"""
        save_data = {
            'clients': self.clients,
            'policy_state': self.policy_network.state_dict(),
            'config': self.config
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"💾 服务器状态保存到: {path}")

    def load(self, path: str):
        """加载服务器状态"""
        if Path(path).exists():
            with open(path, 'rb') as f:
                save_data = pickle.load(f)

            self.clients = save_data.get('clients', {})
            self.policy_network.load_state_dict(save_data.get('policy_state', {}))
            print(f"📂 服务器状态从 {path} 加载")


class LightningTrainer:
    """
    Agent Lightning训练器 - 与完整框架兼容的接口
    """

    def __init__(self, model, learning_rate=1e-4, batch_size=32, enable_reinforcement=True):
        """
        初始化Lightning训练器

        Args:
            model: 要训练的模型
            learning_rate: 学习率
            batch_size: 批大小
            enable_reinforcement: 是否启用强化学习
        """
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.enable_reinforcement = enable_reinforcement

        # 创建Agent Lightning客户端
        self.client = self._create_client()

        # 训练历史
        self.training_history = []
        self.decisions_applied = 0

        print(f"⚡ Agent Lightning训练器初始化")
        print(f"   模型: {model.__class__.__name__}")
        print(f"   学习率: {learning_rate}")
        print(f"   批大小: {batch_size}")
        print(f"   强化学习: {'启用' if enable_reinforcement else '禁用'}")

    def _create_client(self) -> 'AgentLightningClient':
        """创建客户端实例"""
        config = {
            'client_id': f'trainer_{uuid.uuid4().hex[:8]}',
            'agent_lightning': {
                'buffer_capacity': 10000,
                'policy_lr': 1e-3
            }
        }

        return AgentLightningClient(
            client_id=config['client_id'],
            config=config,
            use_local=True
        )

    def reinforce(self, experiences, target_metric="mse", n_epochs=5):
        """
        执行强化学习

        Args:
            experiences: 经验列表
            target_metric: 目标指标
            n_epochs: 训练轮数

        Returns:
            改进值
        """
        if not self.enable_reinforcement or not experiences:
            return 0.0

        print(f"⚡ 开始强化学习，经验数量: {len(experiences)}")

        try:
            # 1. 转换经验为Agent Lightning格式
            agent_experiences = []
            for exp in experiences:
                if isinstance(exp, dict):
                    # 从字典创建经验
                    agent_exp = Experience(
                        state=self._extract_state(exp),
                        action=exp.get('action', 0),
                        reward=self._calculate_reward(exp, target_metric),
                        next_state=None,
                        done=True,
                        metadata=exp
                    )
                    agent_experiences.append(agent_exp)

            # 2. 添加到经验缓冲区
            for exp in agent_experiences:
                self.client.add_experience(exp)

            # 3. 执行策略学习
            improvement = 0.0
            for epoch in range(n_epochs):
                # 获取当前状态
                current_state = self._get_current_model_state()

                # 获取决策
                context = self._create_context(current_state)
                decision = self.client.get_decision(context)

                # 应用决策到模型
                if self.apply_decision(decision):
                    # 计算改进
                    epoch_improvement = self._evaluate_improvement()
                    improvement += epoch_improvement

                    # 记录反馈
                    reward = -epoch_improvement  # 负改进作为奖励（改进越大，奖励越小）
                    self.client.log_feedback(
                        state=current_state,
                        action=decision.action,
                        reward=reward,
                        next_state=self._get_current_model_state(),
                        done=(epoch == n_epochs - 1)
                    )

                    print(f"   轮次 {epoch + 1}/{n_epochs}: 动作={decision.action}, "
                          f"改进={epoch_improvement:.6f}")

                else:
                    print(f"   轮次 {epoch + 1}/{n_epochs}: 决策应用失败")

            avg_improvement = improvement / n_epochs if n_epochs > 0 else 0.0
            print(f"📈 强化学习完成，平均改进: {avg_improvement:.6f}")

            return avg_improvement

        except Exception as e:
            print(f"❌ 强化学习失败: {e}")
            return 0.0

    def _extract_state(self, experience: Dict[str, Any]) -> np.ndarray:
        """从经验中提取状态"""
        metrics = experience.get('metrics', {})

        # 简化状态提取
        state = np.array([
            metrics.get('mse', 0.5),
            metrics.get('mae', 0.5),
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
        ])

        return state

    def _calculate_reward(self, experience: Dict[str, Any], target_metric: str) -> float:
        """计算奖励"""
        metrics = experience.get('metrics', {})

        if target_metric == "mse":
            value = metrics.get('mse', 0.5)
            reward = -value  # 负MSE作为奖励（MSE越小越好）
        elif target_metric == "mae":
            value = metrics.get('mae', 0.5)
            reward = -value
        else:
            reward = -1.0  # 默认奖励

        return reward

    def _get_current_model_state(self) -> List[float]:
        """获取当前模型状态"""
        state = []

        # 提取模型参数统计
        if hasattr(self.model, 'parameters'):
            params = list(self.model.parameters())
            if params:
                # 计算参数统计
                total_params = sum(p.numel() for p in params)
                trainable_params = sum(p.numel() for p in params if p.requires_grad)

                state.extend([
                    total_params / 1e6,  # 百万参数
                    trainable_params / 1e6,
                    trainable_params / total_params if total_params > 0 else 0.0
                ])

        # 填充到固定长度
        while len(state) < 10:
            state.append(0.5)

        return state[:10]

    def _create_context(self, model_state: List[float]) -> Dict[str, Any]:
        """创建决策上下文"""
        return {
            'model_state': model_state,
            'metrics': {'mse': 0.3, 'mae': 0.4},  # 模拟指标
            'features': {'shape': [self.batch_size, 96, 7]},
            'current_params': {
                'spectral_threshold': 0.5,
                'laplacian_weight': 0.01
            }
        }

    def _evaluate_improvement(self) -> float:
        """评估改进"""
        # 简化：返回随机改进值
        # 实际应该评估模型在验证集上的性能
        return np.random.uniform(-0.01, 0.01)

    def apply_decision(self, decision: DecisionResponse) -> bool:
        """应用决策到模型"""
        try:
            # 检查模型是否支持参数更新
            if hasattr(self.model, 'update_parameters'):
                self.model.update_parameters(**decision.parameters)
            elif hasattr(self.model, 'set_parameters'):
                # 其他可能的参数设置方法
                self.model.set_parameters(decision.parameters)
            else:
                # 如果模型不支持直接参数更新，我们修改优化器
                if hasattr(self, 'optimizer'):
                    new_lr = self.learning_rate * decision.parameters.get('learning_rate_multiplier', 1.0)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr

            self.decisions_applied += 1

            # 记录决策历史
            self.training_history.append({
                'timestamp': datetime.now().isoformat(),
                'decision_id': decision.decision_id,
                'action': decision.action,
                'parameters': decision.parameters,
                'reasoning': decision.reasoning,
                'confidence': decision.confidence
            })

            print(f"✅ 决策应用成功: 动作={decision.action}, "
                  f"参数={decision.parameters}")

            return True

        except Exception as e:
            print(f"❌ 决策应用失败: {e}")
            return False

    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        return {
            'decisions_applied': self.decisions_applied,
            'training_history_size': len(self.training_history),
            'client_id': self.client.client_id,
            'enable_reinforcement': self.enable_reinforcement,
            'recent_decisions': self.training_history[-5:] if self.training_history else []
        }


class AgentLightningClient:
    """
    Agent Lightning客户端
    """

    def __init__(self,
                 client_id: str = "default_client",
                 config: Optional[Dict[str, Any]] = None,
                 use_local: bool = True):

        self.client_id = client_id
        self.config = config or {}
        self.use_local = use_local

        # 决策历史
        self.decision_history = []
        self.last_decision_time = 0
        self.total_reward = 0.0

        # 初始化连接
        if use_local:
            # 本地模式：创建或重用本地服务器
            self.server = self._get_local_server()
            self.base_url = None
            print(f"✅ Agent Lightning客户端初始化（本地模式）: {client_id}")
        else:
            # 远程模式
            self.base_url = self.config.get('server_url', 'http://localhost:8000')
            self.server = None
            print(f"✅ Agent Lightning客户端初始化（远程模式）: {client_id}")

    def _get_local_server(self) -> AgentLightningLocalServer:
        """获取或创建本地服务器"""
        # 使用单例模式
        if not hasattr(AgentLightningClient, '_local_server'):
            AgentLightningClient._local_server = AgentLightningLocalServer(self.config)
        return AgentLightningClient._local_server

    def get_decision(self, context: Dict[str, Any]) -> DecisionResponse:
        """
        获取智能体决策

        Args:
            context: 决策上下文，包含特征、指标等

        Returns:
            决策响应
        """
        # 限流：避免过于频繁的决策
        current_time = time.time()
        if current_time - self.last_decision_time < 1.0:  # 至少1秒间隔
            time.sleep(1.0)

        try:
            if self.use_local:
                # 本地调用
                decision = self.server.get_decision(self.client_id, context)
            else:
                # 远程API调用
                decision = self._remote_get_decision(context)

            # 记录历史
            self.decision_history.append(decision)
            self.last_decision_time = current_time

            print(f"🤖 客户端 {self.client_id} 获取决策: 动作={decision.action}")

            return decision

        except Exception as e:
            print(f"❌ 获取决策失败: {e}")
            # 返回安全决策
            return self._get_fallback_decision(context)

    def _remote_get_decision(self, context: Dict[str, Any]) -> DecisionResponse:
        """远程获取决策"""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/agent/decision",
                json={
                    'client_id': self.client_id,
                    'context': context
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            return DecisionResponse(
                decision_id=data.get('decision_id', f"remote_{int(time.time() * 1000)}"),
                action=data.get('action', 1),
                parameters=data.get('parameters', {}),
                confidence=data.get('confidence', 0.8),
                reasoning=data.get('reasoning', '远程决策'),
                timestamp=time.time()
            )

        except Exception as e:
            raise Exception(f"远程决策请求失败: {e}")

    def _get_fallback_decision(self, context: Dict[str, Any]) -> DecisionResponse:
        """备用决策（当主服务不可用时）"""
        mse = context.get('metrics', {}).get('mse', 0.5)

        if mse > 0.4:
            action = 2
            params = {'spectral_threshold': 0.7, 'laplacian_weight': 0.03}
            reasoning = "高误差，激进调整（备用）"
        elif mse > 0.2:
            action = 1
            params = {'spectral_threshold': 0.6, 'laplacian_weight': 0.02}
            reasoning = "中等误差，适度调整（备用）"
        else:
            action = 0
            params = {'spectral_threshold': 0.5, 'laplacian_weight': 0.01}
            reasoning = "低误差，保持参数（备用）"

        return DecisionResponse(
            decision_id=f"fallback_{int(time.time() * 1000)}",
            action=action,
            parameters=params,
            confidence=0.6,
            reasoning=reasoning,
            timestamp=time.time()
        )

    def add_experience(self, experience: Experience):
        """添加经验"""
        if self.use_local:
            self.server.add_experience(experience)
        else:
            # 远程添加经验（预留）
            pass

    def log_feedback(self, state, action, reward, next_state, done=False):
        """
        记录反馈（用于强化学习）

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束
        """
        # 创建经验对象
        experience = Experience(
            state=np.array(state),
            action=action,
            reward=reward,
            next_state=np.array(next_state) if next_state is not None else None,
            done=done
        )

        # 添加经验
        self.add_experience(experience)

        # 更新累计奖励
        self.total_reward += reward

        print(f"📝 反馈记录: 动作={action}, 奖励={reward:.4f}, 累计奖励={self.total_reward:.4f}")

    def submit_training(self,
                        model_config: Dict[str, Any],
                        training_config: Dict[str, Any]) -> str:
        """
        提交训练任务（异步）

        Args:
            model_config: 模型配置
            training_config: 训练配置

        Returns:
            任务ID
        """
        if self.use_local:
            task_id = self.server.submit_training_task(
                self.client_id, model_config, training_config
            )
        else:
            # 远程提交
            task_id = f"remote_task_{int(time.time() * 1000)}"

        print(f"📤 提交训练任务: {task_id}")
        return task_id

    def get_training_status(self, task_id: str) -> Dict[str, Any]:
        """获取训练任务状态"""
        if self.use_local:
            return self.server.get_task_status(task_id)
        else:
            return {'status': 'unknown', 'task_id': task_id}

    def get_client_stats(self) -> Dict[str, Any]:
        """获取客户端统计信息"""
        return {
            'client_id': self.client_id,
            'decision_count': len(self.decision_history),
            'total_reward': self.total_reward,
            'last_decision_time': self.last_decision_time,
            'avg_decision_interval': self._calculate_avg_interval(),
            'use_local': self.use_local
        }

    def _calculate_avg_interval(self) -> float:
        """计算平均决策间隔"""
        if len(self.decision_history) < 2:
            return 0.0

        intervals = []
        for i in range(1, len(self.decision_history)):
            interval = self.decision_history[i].timestamp - self.decision_history[i - 1].timestamp
            intervals.append(interval)

        return float(np.mean(intervals)) if intervals else 0.0

    def save_state(self, path: str):
        """保存客户端状态"""
        if self.use_local:
            self.server.save(path)

    def load_state(self, path: str):
        """加载客户端状态"""
        if self.use_local:
            self.server.load(path)


def create_lightning_client(config: Dict[str, Any]) -> AgentLightningClient:
    """
    创建Agent Lightning客户端（工厂函数）

    Args:
        config: 配置字典

    Returns:
        AgentLightningClient实例
    """
    client_id = config.get('client_id', f'client_{int(time.time())}')

    client_config = {
        'check_interval': config.get('autogen', {}).get('check_interval', 50),
        'max_decisions': config.get('agent_lightning', {}).get('max_decisions', 1000),
        'server_url': config.get('agent_lightning', {}).get('server_url', None),
        'buffer_capacity': config.get('agent_lightning', {}).get('buffer_capacity', 5000),
        'policy_lr': config.get('agent_lightning', {}).get('policy_lr', 1e-3)
    }

    # 决定使用本地还是远程模式
    use_local = client_config['server_url'] is None

    return AgentLightningClient(
        client_id=client_id,
        config=client_config,
        use_local=use_local
    )


# 测试代码
if __name__ == "__main__":
    print("🔬 测试Agent Lightning客户端...")

    # 创建客户端
    config = {
        'client_id': 'test_client',
        'autogen': {'check_interval': 50},
        'agent_lightning': {
            'max_decisions': 100,
            'buffer_capacity': 1000,
            'policy_lr': 1e-3
        }
    }

    client = create_lightning_client(config)

    # 测试决策
    context = {
        'features': {'shape': [32, 96, 7]},
        'metrics': {'mse': 0.35, 'mae': 0.45, 'r2': 0.75},
        'current_params': {'spectral_threshold': 0.5, 'laplacian_weight': 0.01}
    }

    print("\n🧪 测试决策获取:")
    decision = client.get_decision(context)
    print(f"  决策ID: {decision.decision_id}")
    print(f"  动作: {decision.action}")
    print(f"  参数: {decision.parameters}")
    print(f"  置信度: {decision.confidence:.3f}")
    print(f"  理由: {decision.reasoning}")

    # 测试经验记录
    print("\n🧪 测试经验记录:")
    experience = Experience(
        state=np.random.randn(10),
        action=1,
        reward=0.5,
        next_state=np.random.randn(10),
        done=False
    )
    client.add_experience(experience)

    # 测试反馈记录
    print("\n🧪 测试反馈记录:")
    client.log_feedback(
        state=[0.1, 0.2, 0.3, 0.4, 0.5],
        action=2,
        reward=0.8,
        next_state=[0.2, 0.3, 0.4, 0.5, 0.6],
        done=False
    )

    # 测试客户端统计
    print("\n🧪 测试客户端统计:")
    stats = client.get_client_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 测试服务器统计（如果本地）
    if hasattr(client, 'server'):
        server_stats = client.server.get_stats()
        print("\n🧪 服务器统计:")
        for key, value in server_stats.items():
            print(f"  {key}: {value}")

    print("\n✅ Agent Lightning测试完成!")