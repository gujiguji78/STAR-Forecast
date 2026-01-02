"""
Agent Lightning客户端 - 几乎不改动原有代码
通过API与服务端交互，实现训练-执行解耦
"""
import requests
import json
import time
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import aiohttp
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import numpy as np


@dataclass
class AgentDecision:
    """智能体决策"""
    decision_id: str
    action: int
    parameters: Dict[str, float]
    semantic_reward: float
    timestamp: str
    fallback: bool = False


@dataclass
class TrainingTaskStatus:
    """训练任务状态"""
    task_id: str
    status: str
    progress: float
    metrics: Dict[str, Any]
    error: Optional[str] = None


class AgentLightningClient:
    """
    Agent Lightning客户端

    特点：
    1. 通过REST API与服务端交互
    2. 几乎不改动原有训练代码
    3. 自动重试和降级处理
    4. 异步和同步两种模式
    """

    def __init__(self,
                 base_url: str = "http://localhost:8000",
                 client_id: str = "default_client",
                 timeout: int = 10,
                 retry_attempts: int = 3,
                 fallback_enabled: bool = True):

        self.base_url = base_url.rstrip('/')
        self.client_id = client_id
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.fallback_enabled = fallback_enabled

        self.logger = logging.getLogger(__name__)
        self.session = None  # aiohttp会话

        # 缓存
        self.decision_cache = {}
        self.task_cache = {}

        # 统计信息
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'fallback_decisions': 0,
            'avg_response_time': 0.0
        }

        # 测试连接
        self._test_connection()

        self.logger.info(f"✅ Agent Lightning客户端初始化: {client_id}")
        self.logger.info(f"   服务端: {base_url}")
        self.logger.info(f"   超时: {timeout}s, 重试次数: {retry_attempts}")

    def _test_connection(self):
        """测试服务连接"""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/health",
                timeout=self.timeout
            )

            if response.status_code == 200:
                self.logger.info(f"🔗 连接到Agent Lightning服务")
                return True
            else:
                self.logger.warning(f"⚠️ 服务响应异常: {response.status_code}")
                return False

        except Exception as e:
            self.logger.error(f"❌ 无法连接到Agent Lightning服务: {e}")

            if not self.fallback_enabled:
                raise ConnectionError(f"Agent Lightning服务不可用: {e}")

            return False

    async def async_init(self):
        """异步初始化"""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )

    async def close(self):
        """关闭客户端"""
        if self.session:
            await self.session.close()
            self.session = None

    # ==================== 核心API方法 ====================

    def get_decision(self, context: Dict[str, Any]) -> AgentDecision:
        """
        获取智能体决策 - 同步版本

        这是前端唯一需要调用的方法
        几乎不需要修改原有代码，只需要在需要决策时调用此方法

        Args:
            context: 上下文信息，包含特征、指标等

        Returns:
            智能体决策
        """
        self.stats['total_requests'] += 1
        start_time = time.time()

        try:
            # 准备请求
            request_data = {
                "client_id": self.client_id,
                "context": context,
                "require_feedback": True
            }

            # 发送请求
            response = self._make_request(
                "POST",
                f"{self.base_url}/api/v1/agent/decision",
                json=request_data
            )

            # 解析响应
            decision_data = response.json()

            decision = AgentDecision(
                decision_id=decision_data['decision_id'],
                action=decision_data['action'],
                parameters=decision_data.get('parameters', {}),
                semantic_reward=decision_data.get('semantic_reward', 0.0),
                timestamp=decision_data['timestamp'],
                fallback=decision_data.get('fallback', False)
            )

            # 更新缓存
            self.decision_cache[decision.decision_id] = {
                'decision': decision,
                'context': context,
                'timestamp': datetime.now()
            }

            # 限制缓存大小
            if len(self.decision_cache) > 100:
                oldest_key = next(iter(self.decision_cache))
                del self.decision_cache[oldest_key]

            # 更新统计
            self.stats['successful_requests'] += 1
            response_time = time.time() - start_time
            self._update_response_time(response_time)

            if decision.fallback:
                self.stats['fallback_decisions'] += 1
                self.logger.warning(f"⚠️ 使用备用决策: {decision.decision_id}")
            else:
                self.logger.info(f"🤖 获取决策: action={decision.action}, "
                                 f"params={decision.parameters}")

            return decision

        except Exception as e:
            self.stats['failed_requests'] += 1
            self.logger.error(f"❌ 决策请求失败: {e}")

            # 返回备用决策
            return self._get_fallback_decision(context)

    async def async_get_decision(self, context: Dict[str, Any]) -> AgentDecision:
        """异步获取智能体决策"""
        await self.async_init()

        self.stats['total_requests'] += 1
        start_time = time.time()

        try:
            # 准备请求
            request_data = {
                "client_id": self.client_id,
                "context": context,
                "require_feedback": True
            }

            # 发送异步请求
            async with self.session.post(
                    f"{self.base_url}/api/v1/agent/decision",
                    json=request_data
            ) as response:

                if response.status == 200:
                    decision_data = await response.json()

                    decision = AgentDecision(
                        decision_id=decision_data['decision_id'],
                        action=decision_data['action'],
                        parameters=decision_data.get('parameters', {}),
                        semantic_reward=decision_data.get('semantic_reward', 0.0),
                        timestamp=decision_data['timestamp'],
                        fallback=decision_data.get('fallback', False)
                    )

                    # 更新统计
                    self.stats['successful_requests'] += 1
                    response_time = time.time() - start_time
                    self._update_response_time(response_time)

                    if decision.fallback:
                        self.stats['fallback_decisions'] += 1

                    return decision
                else:
                    raise Exception(f"HTTP {response.status}: {await response.text()}")

        except Exception as e:
            self.stats['failed_requests'] += 1
            self.logger.error(f"❌ 异步决策请求失败: {e}")

            # 返回备用决策
            return self._get_fallback_decision(context)

    def update_experience(self,
                          state: List[float],
                          action: int,
                          reward: float,
                          next_state: List[float],
                          done: bool = False) -> bool:
        """
        更新智能体经验

        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束

        Returns:
            是否成功
        """
        try:
            request_data = {
                "client_id": self.client_id,
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done
            }

            # 发送更新请求（不等待响应）
            response = self._make_request(
                "POST",
                f"{self.base_url}/api/v1/agent/update",
                json=request_data,
                timeout=2  # 短超时，不阻塞主流程
            )

            if response.status_code == 200:
                self.logger.debug(f"✅ 经验更新成功: reward={reward}")
                return True
            else:
                self.logger.warning(f"⚠️ 经验更新失败: {response.status_code}")
                return False

        except Exception as e:
            self.logger.debug(f"经验更新异常（可忽略）: {e}")
            return False

    async def async_update_experience(self,
                                      state: List[float],
                                      action: int,
                                      reward: float,
                                      next_state: List[float],
                                      done: bool = False) -> bool:
        """异步更新智能体经验"""
        await self.async_init()

        try:
            request_data = {
                "client_id": self.client_id,
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done
            }

            async with self.session.post(
                    f"{self.base_url}/api/v1/agent/update",
                    json=request_data,
                    timeout=2
            ) as response:

                if response.status == 200:
                    return True
                else:
                    return False

        except Exception:
            return False

    def submit_training_task(self,
                             model_config: Dict[str, Any],
                             training_config: Dict[str, Any],
                             data_path: Optional[str] = None,
                             callback_url: Optional[str] = None) -> str:
        """
        提交训练任务

        Args:
            model_config: 模型配置
            training_config: 训练配置
            data_path: 数据路径
            callback_url: 回调URL

        Returns:
            任务ID
        """
        try:
            request_data = {
                "client_id": self.client_id,
                "model_config": model_config,
                "training_config": training_config,
                "data_path": data_path,
                "callback_url": callback_url
            }

            response = self._make_request(
                "POST",
                f"{self.base_url}/api/v1/training/submit",
                json=request_data
            )

            if response.status_code == 202:  # Accepted
                task_data = response.json()
                task_id = task_data['task_id']

                self.task_cache[task_id] = {
                    'status': 'submitted',
                    'submitted_at': datetime.now()
                }

                self.logger.info(f"📥 提交训练任务: {task_id}")
                return task_id
            else:
                raise Exception(f"提交失败: {response.status_code}")

        except Exception as e:
            self.logger.error(f"❌ 训练任务提交失败: {e}")
            raise

    def get_training_status(self, task_id: str) -> TrainingTaskStatus:
        """获取训练任务状态"""
        try:
            response = self._make_request(
                "GET",
                f"{self.base_url}/api/v1/training/status/{task_id}"
            )

            if response.status_code == 200:
                status_data = response.json()

                status = TrainingTaskStatus(
                    task_id=task_id,
                    status=status_data['status'],
                    progress=status_data['progress'],
                    metrics=status_data['metrics'],
                    error=status_data.get('error')
                )

                # 更新缓存
                self.task_cache[task_id] = {
                    'status': status.status,
                    'progress': status.progress,
                    'updated_at': datetime.now()
                }

                return status
            else:
                raise Exception(f"状态查询失败: {response.status_code}")

        except Exception as e:
            self.logger.error(f"❌ 训练状态查询失败: {e}")

            # 返回缓存状态
            cached = self.task_cache.get(task_id, {})
            return TrainingTaskStatus(
                task_id=task_id,
                status=cached.get('status', 'unknown'),
                progress=cached.get('progress', 0.0),
                metrics={},
                error=str(e)
            )

    def get_agent_stats(self) -> Dict[str, Any]:
        """获取智能体统计信息"""
        try:
            response = self._make_request(
                "GET",
                f"{self.base_url}/api/v1/agent/stats/{self.client_id}"
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {}

        except Exception as e:
            self.logger.warning(f"⚠️ 统计信息获取失败: {e}")
            return {}

    # ==================== 辅助方法 ====================

    def _make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """发送HTTP请求（带重试）"""
        for attempt in range(self.retry_attempts):
            try:
                response = requests.request(
                    method, url,
                    timeout=self.timeout,
                    **kwargs
                )

                # 检查响应状态
                if response.status_code < 500:  # 非服务器错误
                    return response

                # 服务器错误，重试
                self.logger.warning(f"请求失败 (尝试 {attempt + 1}/{self.retry_attempts}): "
                                    f"HTTP {response.status_code}")

            except requests.exceptions.Timeout:
                self.logger.warning(f"请求超时 (尝试 {attempt + 1}/{self.retry_attempts})")

            except requests.exceptions.ConnectionError:
                self.logger.warning(f"连接错误 (尝试 {attempt + 1}/{self.retry_attempts})")

            except Exception as e:
                self.logger.error(f"请求异常: {e}")
                break

            # 指数退避
            if attempt < self.retry_attempts - 1:
                time.sleep(2 ** attempt)  # 1, 2, 4秒...

        # 所有重试都失败
        raise ConnectionError(f"请求失败: {method} {url}")

    def _get_fallback_decision(self, context: Dict[str, Any]) -> AgentDecision:
        """获取备用决策（当服务不可用时）"""
        self.stats['fallback_decisions'] += 1

        # 简单启发式规则
        metrics = context.get('metrics', {})
        mse = metrics.get('mse', 0)

        if mse > 0.3:
            action = 4
            parameters = {
                'spectral_threshold': 0.7,
                'laplacian_weight': 0.03,
                'learning_rate_multiplier': 1.5
            }
        elif mse > 0.1:
            action = 2
            parameters = {
                'spectral_threshold': 0.6,
                'laplacian_weight': 0.02,
                'learning_rate_multiplier': 1.2
            }
        else:
            action = 0
            parameters = {
                'spectral_threshold': 0.5,
                'laplacian_weight': 0.01,
                'learning_rate_multiplier': 1.0
            }

        decision = AgentDecision(
            decision_id=f"fallback_{int(time.time())}",
            action=action,
            parameters=parameters,
            semantic_reward=0.0,
            timestamp=datetime.now().isoformat(),
            fallback=True
        )

        self.logger.warning(f"⚠️ 使用备用决策: action={action}, params={parameters}")
        return decision

    def _update_response_time(self, response_time: float):
        """更新平均响应时间"""
        total_requests = self.stats['successful_requests'] + self.stats['failed_requests']

        if total_requests == 1:
            self.stats['avg_response_time'] = response_time
        else:
            # 指数移动平均
            alpha = 0.1
            self.stats['avg_response_time'] = (
                    alpha * response_time +
                    (1 - alpha) * self.stats['avg_response_time']
            )

    def get_client_stats(self) -> Dict[str, Any]:
        """获取客户端统计信息"""
        return {
            **self.stats,
            'client_id': self.client_id,
            'base_url': self.base_url,
            'decision_cache_size': len(self.decision_cache),
            'task_cache_size': len(self.task_cache),
            'timestamp': datetime.now().isoformat()
        }

    def clear_cache(self):
        """清空缓存"""
        self.decision_cache.clear()
        self.task_cache.clear()
        self.logger.info("🗑️ 客户端缓存已清空")


# ==================== 训练集成示例 ====================

def train_with_agent_lightning_example():
    """
    使用Agent Lightning的训练示例
    展示如何几乎不改动原有代码集成智能体
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader

    # 1. 初始化客户端（唯一新增的代码）
    agent_client = AgentLightningClient(
        base_url="http://localhost:8000",
        client_id="training_example",
        timeout=10,
        retry_attempts=3,
        fallback_enabled=True
    )

    # 2. 原有训练代码基本不变
    model = nn.Linear(10, 1)  # 示例模型
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 模拟数据加载器
    dataloader = DataLoader([(torch.randn(10), torch.randn(1)) for _ in range(100)],
                            batch_size=32)

    # 3. 训练循环（只添加了智能体调用）
    for epoch in range(10):
        for batch_idx, (x, y) in enumerate(dataloader):
            # 前向传播
            predictions = model(x)
            loss = criterion(predictions, y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 4. 在特定步骤调用智能体（唯一修改点）
            if batch_idx % 50 == 0:  # 每50个批次调用一次
                # 准备上下文信息
                context = {
                    'features': {
                        'shape': list(x.shape),
                        'mean': x.mean().item(),
                        'std': x.std().item()
                    },
                    'metrics': {
                        'mse': loss.item(),
                        'mae': torch.abs(predictions - y).mean().item()
                    },
                    'current_params': {
                        'learning_rate': optimizer.param_groups[0]['lr'],
                        'batch_size': x.shape[0]
                    },
                    'batch_idx': batch_idx,
                    'epoch': epoch
                }

                # 调用智能体获取决策（这是唯一的新代码）
                decision = agent_client.get_decision(context)

                # 5. 应用决策（可选）
                if decision.parameters:
                    # 例如，调整学习率
                    if 'learning_rate_multiplier' in decision.parameters:
                        new_lr = 0.001 * decision.parameters['learning_rate_multiplier']
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = new_lr

                        print(f"🔄 调整学习率: {new_lr:.6f}")

                # 6. 更新智能体经验（可选，异步不阻塞）
                # 准备状态和奖励
                state = [x.mean().item(), x.std().item(), loss.item()]
                reward = -loss.item()  # 负损失作为奖励

                # 异步更新（不等待）
                agent_client.update_experience(
                    state=state,
                    action=decision.action,
                    reward=reward,
                    next_state=state  # 简化：假设状态不变
                )

            # 打印训练信息
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    # 7. 获取统计信息
    stats = agent_client.get_client_stats()
    print(f"\n📊 客户端统计:")
    print(f"   总请求: {stats['total_requests']}")
    print(f"   成功请求: {stats['successful_requests']}")
    print(f"   备用决策: {stats['fallback_decisions']}")
    print(f"   平均响应时间: {stats['avg_response_time']:.3f}s")

    return agent_client


# ==================== 上下文管理器版本 ====================

class AgentLightningContext:
    """
    Agent Lightning上下文管理器
    自动管理客户端生命周期
    """

    def __init__(self, **kwargs):
        self.client = AgentLightningClient(**kwargs)

    def __enter__(self):
        return self.client

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 可以在这里添加清理逻辑
        pass

    async def __aenter__(self):
        await self.client.async_init()
        return self.client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.close()


# 使用示例
if __name__ == "__main__":
    # 同步使用
    with AgentLightningContext(
            base_url="http://localhost:8000",
            client_id="test_client"
    ) as client:
        # 获取决策
        context = {
            'features': {'mean': 0.0, 'std': 1.0},
            'metrics': {'mse': 0.25, 'mae': 0.4}
        }

        decision = client.get_decision(context)
        print(f"决策: {decision}")


    # 异步使用
    async def async_example():
        async with AgentLightningContext(
                base_url="http://localhost:8000",
                client_id="async_client"
        ) as client:
            context = {'features': {}, 'metrics': {'mse': 0.1}}
            decision = await client.async_get_decision(context)
            print(f"异步决策: {decision}")


    asyncio.run(async_example())