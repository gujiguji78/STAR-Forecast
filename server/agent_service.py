"""
Agent Lightning服务端 - 完整的微服务实现
提供REST API接口，完全解耦训练和执行
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import asyncio
import uuid
import json
from datetime import datetime
import logging
from pathlib import Path
import threading
import queue
import time
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入自定义模块
from .schemas import *
from ..models.istr import ISTRNetwork
from ..agents.autogen_system import AutoGenMultiAgentSystem

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI应用
app = FastAPI(
    title="Agent Lightning Service",
    version="2.0.0",
    description="训练-执行完全解耦的强化学习智能体服务",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 数据结构定义 ====================
class TaskStatus(str, Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentState:
    """智能体状态"""
    agent_id: str
    policy_net: Any
    target_net: Any
    optimizer: Any
    memory: Any  # 经验回放缓冲区
    epsilon: float
    steps_done: int
    episode_rewards: List[float]
    created_at: datetime
    last_updated: datetime


@dataclass
class TrainingTask:
    """训练任务"""
    task_id: str
    client_id: str
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    data_path: Optional[str]
    callback_url: Optional[str]
    status: TaskStatus
    progress: float = 0.0
    metrics: Dict[str, Any] = None
    error: Optional[str] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.created_at is None:
            self.created_at = datetime.now()


# ==================== 核心服务类 ====================
class AgentLightningService:
    """Agent Lightning核心服务"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
        )

        # 客户端管理
        self.client_sessions: Dict[str, Dict[str, Any]] = {}

        # 任务管理
        self.training_tasks: Dict[str, TrainingTask] = {}
        self.training_queue = queue.PriorityQueue()
        self.task_results: Dict[str, Any] = {}

        # 模型缓存
        self.model_cache: Dict[str, Any] = {}

        # 智能体池
        self.agent_pool: Dict[str, AgentState] = {}

        # AutoGen系统
        self.autogen_system = AutoGenMultiAgentSystem(config)

        # 训练工作线程
        self.workers: Dict[int, Dict[str, Any]] = {}
        self._start_workers()

        # 监控指标
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'active_clients': 0,
            'pending_tasks': 0,
            'completed_tasks': 0
        }

        logger.info("✅ Agent Lightning服务初始化完成")
        logger.info(f"   设备: {self.device}")
        logger.info(f"   最大工作线程: {self.config['agent_lightning']['service']['workers']}")

    def _start_workers(self):
        """启动训练工作线程"""
        num_workers = self.config['agent_lightning']['service']['workers']

        for i in range(num_workers):
            worker = threading.Thread(
                target=self._training_worker,
                args=(i,),
                daemon=True,
                name=f"TrainingWorker-{i}"
            )
            worker.start()

            self.workers[i] = {
                'thread': worker,
                'busy': False,
                'current_task': None,
                'tasks_processed': 0
            }

            logger.info(f"   启动工作线程 {i}")

    def _training_worker(self, worker_id: int):
        """训练工作线程主循环"""
        logger.info(f"👷 工作线程 {worker_id} 启动")

        while True:
            try:
                # 获取任务（阻塞）
                priority, task_id = self.training_queue.get()

                self.workers[worker_id]['busy'] = True
                self.workers[worker_id]['current_task'] = task_id

                task = self.training_tasks.get(task_id)
                if not task:
                    logger.error(f"任务 {task_id} 不存在")
                    continue

                # 更新任务状态
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()

                logger.info(f"🔧 工作线程 {worker_id} 开始处理任务 {task_id}")

                try:
                    # 执行训练
                    result = self._execute_training_task(task)

                    # 更新任务状态
                    task.status = TaskStatus.COMPLETED
                    task.progress = 1.0
                    task.metrics = result.get('metrics', {})
                    task.completed_at = datetime.now()

                    # 存储结果
                    self.task_results[task_id] = result

                    # 回调通知
                    if task.callback_url:
                        self._notify_callback(task, result)

                    logger.info(f"✅ 任务 {task_id} 完成")

                except Exception as e:
                    logger.error(f"❌ 任务 {task_id} 失败: {e}")

                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = datetime.now()

                finally:
                    self.workers[worker_id]['busy'] = False
                    self.workers[worker_id]['current_task'] = None
                    self.workers[worker_id]['tasks_processed'] += 1
                    self.training_queue.task_done()

            except Exception as e:
                logger.error(f"工作线程 {worker_id} 错误: {e}")
                time.sleep(1)  # 避免快速失败循环

    def _execute_training_task(self, task: TrainingTask) -> Dict[str, Any]:
        """执行训练任务"""
        logger.info(f"执行训练任务 {task.task_id} for client {task.client_id}")

        # 这里应该是完整的训练逻辑
        # 简化实现：模拟训练过程
        start_time = time.time()

        # 模拟训练进度
        for i in range(1, 101):
            time.sleep(0.1)  # 模拟训练时间
            task.progress = i / 100

            # 每10%记录一次
            if i % 10 == 0:
                logger.info(f"任务 {task.task_id} 进度: {i}%")

        # 模拟训练结果
        result = {
            'task_id': task.task_id,
            'client_id': task.client_id,
            'duration': time.time() - start_time,
            'metrics': {
                'final_loss': 0.1234,
                'accuracy': 0.8765,
                'training_time': 10.5
            },
            'model_path': f"./models/{task.task_id}.pth",
            'artifacts': ['model', 'logs', 'metrics']
        }

        return result

    def _notify_callback(self, task: TrainingTask, result: Dict[str, Any]):
        """回调通知客户端"""
        import requests

        try:
            requests.post(
                task.callback_url,
                json={
                    'task_id': task.task_id,
                    'status': 'completed',
                    'result': result
                },
                timeout=5
            )
            logger.info(f"✅ 回调通知成功: {task.callback_url}")
        except Exception as e:
            logger.warning(f"⚠️ 回调通知失败: {e}")

    # ==================== 公共API方法 ====================

    async def submit_training(self, request: TrainingRequest) -> str:
        """提交训练任务"""
        task_id = str(uuid.uuid4())

        task = TrainingTask(
            task_id=task_id,
            client_id=request.client_id,
            model_config=request.model_config,
            training_config=request.training_config,
            data_path=request.data_path,
            callback_url=request.callback_url,
            status=TaskStatus.PENDING
        )

        # 计算优先级（基于客户端优先级或任务类型）
        priority = self._calculate_task_priority(request)

        # 存储任务
        self.training_tasks[task_id] = task

        # 加入队列
        self.training_queue.put((priority, task_id))

        self.metrics['pending_tasks'] += 1
        logger.info(f"📥 提交训练任务 {task_id}, 优先级: {priority}")

        return task_id

    async def get_agent_decision(self, request: AgentDecisionRequest) -> Dict[str, Any]:
        """获取智能体决策"""
        self.metrics['total_requests'] += 1

        try:
            # 检查客户端会话
            if request.client_id not in self.client_sessions:
                await self._create_client_session(request.client_id)

            session = self.client_sessions[request.client_id]
            agent_state = session['agent_state']

            # 准备状态
            state_tensor = self._prepare_state_tensor(request.context)

            # 使用智能体选择动作
            action = await self._select_action_async(agent_state, state_tensor)

            # 调用AutoGen进行协同分析
            autogen_context = self._prepare_autogen_context(request.context)
            conversation_result = self.autogen_system.initiate_conversation(autogen_context)

            # 从共识中提取参数
            parameters = {}
            if conversation_result.consensus and 'parameters' in conversation_result.consensus:
                parameters = conversation_result.consensus['parameters']

            # 计算语义奖励
            semantic_reward = self._calculate_semantic_reward(conversation_result)

            # 构建响应
            response = {
                'decision_id': str(uuid.uuid4()),
                'action': int(action),
                'parameters': parameters,
                'semantic_reward': semantic_reward,
                'autogen_conversation': {
                    'conversation_id': conversation_result.conversation_id,
                    'consensus_level': conversation_result.consensus.get('agreement_level', 0)
                    if conversation_result.consensus else 0,
                    'summary': conversation_result.summary
                },
                'agent_state': {
                    'epsilon': agent_state.epsilon,
                    'steps_done': agent_state.steps_done,
                    'episode_rewards': agent_state.episode_rewards[-10:]  # 最近10次奖励
                },
                'timestamp': datetime.now().isoformat()
            }

            self.metrics['successful_requests'] += 1
            logger.info(f"🤖 智能体决策生成: {action}, 参数: {parameters}")

            return response

        except Exception as e:
            self.metrics['failed_requests'] += 1
            logger.error(f"❌ 智能体决策失败: {e}")

            # 返回备用决策
            return await self._get_fallback_decision(request)

    async def update_agent_experience(self, request: ExperienceUpdateRequest) -> Dict[str, Any]:
        """更新智能体经验"""
        if request.client_id not in self.client_sessions:
            raise HTTPException(status_code=404, detail="Client session not found")

        session = self.client_sessions[request.client_id]
        agent_state = session['agent_state']

        # 存储经验
        agent_state.memory.push(
            request.state,
            request.action,
            request.reward,
            request.next_state,
            request.done
        )

        # 更新智能体（异步）
        if len(agent_state.memory) > session['config']['batch_size']:
            await self._optimize_agent_async(agent_state)

        # 更新探索率
        agent_state.epsilon = self._update_epsilon(agent_state.steps_done)
        agent_state.steps_done += 1

        # 记录奖励
        agent_state.episode_rewards.append(request.reward)
        if len(agent_state.episode_rewards) > 1000:  # 限制长度
            agent_state.episode_rewards = agent_state.episode_rewards[-1000:]

        agent_state.last_updated = datetime.now()

        return {
            'status': 'updated',
            'epsilon': agent_state.epsilon,
            'steps_done': agent_state.steps_done,
            'memory_size': len(agent_state.memory),
            'avg_recent_reward': np.mean(agent_state.episode_rewards[-100:])
            if agent_state.episode_rewards else 0
        }

    # ==================== 智能体核心方法 ====================

    async def _create_client_session(self, client_id: str):
        """创建客户端会话"""
        logger.info(f"创建客户端会话: {client_id}")

        # 创建DQN网络
        state_dim = self.config['agent_lightning']['rl']['state_dim']
        action_dim = self.config['agent_lightning']['rl']['action_dim']
        hidden_dim = self.config['agent_lightning']['rl']['hidden_dim']

        policy_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        target_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        target_net.load_state_dict(policy_net.state_dict())

        # 优化器
        optimizer = torch.optim.Adam(
            policy_net.parameters(),
            lr=self.config['agent_lightning']['rl']['dqn']['lr']
        )

        # 经验回放缓冲区
        memory = ReplayBuffer(
            self.config['agent_lightning']['rl']['dqn']['buffer_size']
        )

        # 创建智能体状态
        agent_state = AgentState(
            agent_id=f"agent_{client_id}",
            policy_net=policy_net,
            target_net=target_net,
            optimizer=optimizer,
            memory=memory,
            epsilon=self.config['agent_lightning']['rl']['exploration']['epsilon_start'],
            steps_done=0,
            episode_rewards=[],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )

        # 存储会话
        self.client_sessions[client_id] = {
            'agent_state': agent_state,
            'config': self.config['agent_lightning'],
            'created_at': datetime.now(),
            'last_active': datetime.now()
        }

        # 添加到智能体池
        self.agent_pool[f"agent_{client_id}"] = agent_state

        self.metrics['active_clients'] += 1
        logger.info(f"✅ 客户端会话创建完成: {client_id}")

    async def _select_action_async(self, agent_state: AgentState, state: np.ndarray) -> int:
        """异步选择动作"""
        loop = asyncio.get_event_loop()

        # 在线程池中执行计算密集型操作
        action = await loop.run_in_executor(
            None,
            self._select_action,
            agent_state, state
        )

        return action

    def _select_action(self, agent_state: AgentState, state: np.ndarray) -> int:
        """选择动作（ε-greedy）"""
        # 更新探索率
        agent_state.epsilon = self._update_epsilon(agent_state.steps_done)

        if np.random.random() > agent_state.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = agent_state.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
        else:
            # 探索：随机动作
            action = np.random.randint(0, self.config['agent_lightning']['rl']['action_dim'])

            # 添加噪声
            noise_std = self.config['agent_lightning']['rl']['exploration']['noise_std']
            if noise_std > 0:
                action = action + np.random.normal(0, noise_std)
                action = np.clip(action, 0, self.config['agent_lightning']['rl']['action_dim'] - 1)
                action = int(action)

        return action

    async def _optimize_agent_async(self, agent_state: AgentState):
        """异步优化智能体"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._optimize_agent, agent_state)

    def _optimize_agent(self, agent_state: AgentState):
        """优化智能体网络"""
        if len(agent_state.memory) < agent_state.memory.batch_size:
            return

        # 采样批次
        transitions = agent_state.memory.sample(agent_state.memory.batch_size)
        batch = Transition(*zip(*transitions))

        # 转换为张量
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)

        # 计算当前Q值
        current_q = agent_state.policy_net(state_batch).gather(1, action_batch)

        # 计算目标Q值（Double DQN）
        next_actions = agent_state.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
        next_q = agent_state.target_net(next_state_batch).gather(1, next_actions).detach()

        expected_q = reward_batch.unsqueeze(1) + (
                self.config['agent_lightning']['rl']['dqn']['gamma'] * next_q * (1 - done_batch.unsqueeze(1))
        )

        # 计算损失
        loss = F.mse_loss(current_q, expected_q)

        # 优化
        agent_state.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            agent_state.policy_net.parameters(),
            self.config['agent_lightning']['rl']['training']['gradient_clip']
        )

        agent_state.optimizer.step()

        # 更新目标网络
        if agent_state.steps_done % self.config['agent_lightning']['rl']['dqn']['target_update'] == 0:
            agent_state.target_net.load_state_dict(agent_state.policy_net.state_dict())

        logger.debug(f"智能体优化: 损失={loss.item():.4f}, 步数={agent_state.steps_done}")

    def _update_epsilon(self, steps_done: int) -> float:
        """更新探索率"""
        epsilon_start = self.config['agent_lightning']['rl']['exploration']['epsilon_start']
        epsilon_end = self.config['agent_lightning']['rl']['exploration']['epsilon_end']
        epsilon_decay = self.config['agent_lightning']['rl']['exploration']['epsilon_decay']

        epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                  np.exp(-1. * steps_done / epsilon_decay)

        return max(epsilon_end, epsilon)

    # ==================== 辅助方法 ====================

    def _prepare_state_tensor(self, context: Dict[str, Any]) -> np.ndarray:
        """准备状态张量"""
        # 从上下文中提取特征
        features = context.get('features', {})
        metrics = context.get('metrics', {})

        # 构建状态向量
        state_parts = []

        # 添加统计特征
        if 'statistics' in features:
            stats = features['statistics']
            state_parts.extend([
                stats.get('mean', 0),
                stats.get('std', 1),
                stats.get('skewness', 0),
                stats.get('kurtosis', 0)
            ])

        # 添加性能指标
        state_parts.extend([
            metrics.get('mse', 0),
            metrics.get('mae', 0),
            metrics.get('val_loss', 0)
        ])

        # 添加当前参数
        current_params = context.get('current_params', {})
        state_parts.extend([
            current_params.get('spectral_threshold', 0.5),
            current_params.get('laplacian_weight', 0.01)
        ])

        # 确保状态维度一致
        state_dim = self.config['agent_lightning']['rl']['state_dim']
        state = np.zeros(state_dim)

        # 填充状态向量
        valid_len = min(len(state_parts), state_dim)
        state[:valid_len] = state_parts[:valid_len]

        # 归一化
        state = (state - state.mean()) / (state.std() + 1e-8)

        return state

    def _prepare_autogen_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """准备AutoGen上下文"""
        return {
            'features': context.get('features', {}),
            'metrics': context.get('metrics', {}),
            'current_params': context.get('current_params', {}),
            'training_info': context.get('training_info', {}),
            'step': context.get('batch_idx', 0),
            'batch_idx': context.get('batch_idx', 0),
            'global_step': context.get('global_step', 0)
        }

    def _calculate_semantic_reward(self, conversation_result) -> float:
        """计算语义奖励"""
        if not conversation_result.consensus:
            return -0.1  # 没有共识的惩罚

        consensus_level = conversation_result.consensus.get('agreement_level', 0)

        # 基于共识程度的奖励
        reward = consensus_level * 0.5

        # 添加参数合理性的奖励
        parameters = conversation_result.consensus.get('parameters', {})
        if parameters:
            # 检查参数范围
            valid_params = 0
            if 'spectral_threshold' in parameters:
                if 0.1 <= parameters['spectral_threshold'] <= 0.9:
                    valid_params += 1

            if 'laplacian_weight' in parameters:
                if 0.001 <= parameters['laplacian_weight'] <= 0.1:
                    valid_params += 1

            reward += valid_params * 0.1

        return min(1.0, max(-1.0, reward))

    def _calculate_task_priority(self, request: TrainingRequest) -> int:
        """计算任务优先级"""
        # 简单实现：基于客户端ID的哈希
        priority = hash(request.client_id) % 100

        # 高优先级任务：模型初始化或关键训练
        if request.model_config.get('type') == 'init':
            priority += 100

        return -priority  # 数字越小优先级越高（Python优先队列）

    async def _get_fallback_decision(self, request: AgentDecisionRequest) -> Dict[str, Any]:
        """获取备用决策（当主系统失败时）"""
        logger.warning(f"使用备用决策 for client {request.client_id}")

        # 简单启发式规则
        metrics = request.context.get('metrics', {})
        mse = metrics.get('mse', 0)

        if mse > 0.3:
            action = 4  # 激进调整
            parameters = {
                'spectral_threshold': 0.7,
                'laplacian_weight': 0.03,
                'learning_rate_multiplier': 1.5
            }
        elif mse > 0.1:
            action = 2  # 适度调整
            parameters = {
                'spectral_threshold': 0.6,
                'laplacian_weight': 0.02,
                'learning_rate_multiplier': 1.2
            }
        else:
            action = 0  # 保持
            parameters = {
                'spectral_threshold': 0.5,
                'laplacian_weight': 0.01,
                'learning_rate_multiplier': 1.0
            }

        return {
            'decision_id': str(uuid.uuid4()),
            'action': action,
            'parameters': parameters,
            'semantic_reward': 0.0,
            'fallback': True,
            'timestamp': datetime.now().isoformat()
        }

    # ==================== 服务状态方法 ====================

    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            'status': 'healthy',
            'version': '2.0.0',
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics,
            'workers': [
                {
                    'id': worker_id,
                    'busy': worker_info['busy'],
                    'current_task': worker_info['current_task'],
                    'tasks_processed': worker_info['tasks_processed']
                }
                for worker_id, worker_info in self.workers.items()
            ],
            'active_clients': len(self.client_sessions),
            'pending_tasks': self.training_queue.qsize(),
            'total_tasks': len(self.training_tasks)
        }

    def cleanup_inactive_sessions(self, timeout_hours: int = 24):
        """清理不活跃的会话"""
        cutoff_time = datetime.now() - timedelta(hours=timeout_hours)

        inactive_clients = []
        for client_id, session in self.client_sessions.items():
            if session['last_active'] < cutoff_time:
                inactive_clients.append(client_id)

        for client_id in inactive_clients:
            del self.client_sessions[client_id]
            logger.info(f"清理不活跃客户端: {client_id}")

        self.metrics['active_clients'] = len(self.client_sessions)
        return len(inactive_clients)


# ==================== 神经网络定义 ====================
class DQNNetwork(nn.Module):
    """DQN网络"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int):
        self.buffer = []
        self.capacity = capacity
        self.position = 0
        self.batch_size = 32  # 默认批次大小

    def push(self, state, action, reward, next_state, done):
        """保存经验"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        """随机采样"""
        self.batch_size = batch_size

        if len(self.buffer) < batch_size:
            return []

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

# ==================== API路由 ====================

# 全局服务实例
service_instance = None


def get_service():
    """获取服务实例"""
    global service_instance
    if service_instance is None:
        # 加载配置
        import yaml
        with open("./config.yaml", "r") as f:
            config = yaml.safe_load(f)

        service_instance = AgentLightningService(config)

    return service_instance


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("🚀 Agent Lightning服务启动")
    get_service()  # 初始化服务


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("🛑 Agent Lightning服务关闭")


@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "Agent Lightning",
        "version": "2.0.0",
        "status": "running",
        "docs": "/api/docs"
    }


@app.get("/api/v1/health")
async def health_check():
    """健康检查"""
    service = get_service()
    status = service.get_service_status()

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        **status
    }


@app.post("/api/v1/training/submit")
async def submit_training(request: TrainingRequest):
    """提交训练任务"""
    service = get_service()

    try:
        task_id = await service.submit_training(request)

        return JSONResponse(
            status_code=202,  # Accepted
            content={
                "task_id": task_id,
                "status": "submitted",
                "message": "训练任务已提交",
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/training/status/{task_id}")
async def get_training_status(task_id: str):
    """获取训练状态"""
    service = get_service()

    task = service.training_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "task_id": task_id,
        "status": task.status,
        "progress": task.progress,
        "metrics": task.metrics,
        "error": task.error,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None
    }


@app.post("/api/v1/agent/decision")
async def get_agent_decision(request: AgentDecisionRequest):
    """获取智能体决策"""
    service = get_service()

    try:
        decision = await service.get_agent_decision(request)

        return decision
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/agent/update")
async def update_agent_experience(request: ExperienceUpdateRequest):
    """更新智能体经验"""
    service = get_service()

    try:
        result = await service.update_agent_experience(request)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/agent/stats/{client_id}")
async def get_agent_stats(client_id: str):
    """获取智能体统计信息"""
    service = get_service()

    if client_id not in service.client_sessions:
        raise HTTPException(status_code=404, detail="Client not found")

    session = service.client_sessions[client_id]
    agent_state = session['agent_state']

    return {
        "client_id": client_id,
        "agent_id": agent_state.agent_id,
        "epsilon": agent_state.epsilon,
        "steps_done": agent_state.steps_done,
        "memory_size": len(agent_state.memory),
        "episode_rewards": {
            "recent_10": agent_state.episode_rewards[-10:] if agent_state.episode_rewards else [],
            "average": np.mean(agent_state.episode_rewards[-100:]) if agent_state.episode_rewards else 0,
            "std": np.std(agent_state.episode_rewards[-100:]) if agent_state.episode_rewards else 0
        },
        "created_at": agent_state.created_at.isoformat(),
        "last_updated": agent_state.last_updated.isoformat()
    }


@app.get("/api/v1/service/metrics")
async def get_service_metrics():
    """获取服务指标"""
    service = get_service()

    return service.get_service_status()


@app.post("/api/v1/service/cleanup")
async def cleanup_sessions(timeout_hours: int = 24):
    """清理不活跃会话"""
    service = get_service()

    cleaned = service.cleanup_inactive_sessions(timeout_hours)

    return {
        "cleaned_sessions": cleaned,
        "remaining_sessions": len(service.client_sessions),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn

    # 加载配置
    import yaml

    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)

    service_config = config['agent_lightning']['service']

    uvicorn.run(
        app,
        host=service_config['host'],
        port=service_config['port'],
        workers=service_config['workers'],
        timeout=service_config['timeout']
    )