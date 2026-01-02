"""
Agent Lightning API数据模式定义
完全真实，可直接用于FastAPI
"""
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime


class AgentRequestBase(BaseModel):
    """请求基类"""
    client_id: str = Field(..., description="客户端ID")
    session_id: Optional[str] = Field(None, description="会话ID")


class TrainingMetrics(BaseModel):
    """训练指标"""
    epoch: int
    step: int
    train_loss: float
    val_loss: float
    train_mse: float
    val_mse: float
    train_mae: float
    val_mae: float
    lr: float


class ISTRFeatures(BaseModel):
    """ISTR特征"""
    shape: List[int]
    statistics: Dict[str, List[float]]
    frequency: Dict[str, Any]


class AgentContext(BaseModel):
    """智能体决策上下文"""
    features: Optional[ISTRFeatures] = Field(None, description="ISTR特征")
    metrics: Optional[Dict[str, float]] = Field(None, description="性能指标")
    current_params: Optional[Dict[str, float]] = Field(None, description="当前参数")
    step: Optional[int] = Field(None, description="训练步数")
    epoch: Optional[int] = Field(None, description="训练轮数")


class DecisionRequest(AgentRequestBase):
    """决策请求"""
    context: AgentContext = Field(..., description="决策上下文")
    require_reward: bool = Field(False, description="是否需要奖励计算")
    priority: int = Field(1, description="请求优先级")


class DecisionResponse(BaseModel):
    """决策响应"""
    decision_id: str = Field(..., description="决策ID")
    action: int = Field(..., description="动作编号")
    parameters: Dict[str, float] = Field(..., description="参数调整")
    reward: Optional[float] = Field(None, description="奖励值")
    confidence: float = Field(..., description="置信度")
    timestamp: datetime = Field(..., description="时间戳")
    reasoning: Optional[str] = Field(None, description="推理过程")


class UpdateRequest(AgentRequestBase):
    """更新请求"""
    state: List[float] = Field(..., description="状态向量")
    action: int = Field(..., description="执行动作")
    reward: float = Field(..., description="奖励值")
    next_state: Optional[List[float]] = Field(None, description="下一状态")
    done: bool = Field(False, description="是否结束")


class UpdateResponse(BaseModel):
    """更新响应"""
    success: bool = Field(..., description="是否成功")
    epsilon: Optional[float] = Field(None, description="当前探索率")
    memory_size: Optional[int] = Field(None, description="经验池大小")
    steps_done: Optional[int] = Field(None, description="已执行步数")


class TrainingTask(BaseModel):
    """训练任务"""
    task_id: str = Field(..., description="任务ID")
    model_type: str = Field(default="istr", description="模型类型")
    # 将 model_config 改为 training_config，避免与 Pydantic 冲突
    training_config: Dict[str, Any] = Field(default_factory=dict, description="训练配置")
    dataset_config: Dict[str, Any] = Field(default_factory=dict, description="数据集配置")
    status: str = Field(default="pending", description="任务状态")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")

    # 添加 Pydantic 配置（这个才是正确的 model_config）
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class TrainingSubmitRequest(AgentRequestBase):
    """训练提交请求"""
    model_type: str = Field("istr", description="模型类型")
    model_config: Dict[str, Any] = Field(..., description="模型配置")
    training_config: Dict[str, Any] = Field(..., description="训练配置")
    data_path: Optional[str] = Field(None, description="数据路径")
    callback_url: Optional[str] = Field(None, description="回调URL")


class TrainingSubmitResponse(BaseModel):
    """训练提交响应"""
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    estimated_time: Optional[float] = Field(None, description="预计时间(秒)")
    position_in_queue: Optional[int] = Field(None, description="队列位置")


class TrainingStatusResponse(BaseModel):
    """训练状态响应"""
    task_id: str
    status: str
    progress: float
    metrics: Optional[Dict[str, float]] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None
    queue_position: Optional[int] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="服务版本")
    uptime: float = Field(..., description="运行时间(秒)")
    active_clients: int = Field(..., description="活跃客户端数")
    pending_tasks: int = Field(..., description="待处理任务数")
    memory_usage: float = Field(..., description="内存使用率")
    gpu_available: bool = Field(..., description="GPU是否可用")
    model_loaded: bool = Field(..., description="模型是否加载")


class ClientSession(BaseModel):
    """客户端会话"""
    client_id: str
    session_id: str
    agent_state: Dict[str, Any]
    created_at: datetime
    last_active: datetime
    request_count: int = 0
    total_reward: float = 0.0


class AgentState(BaseModel):
    """智能体状态"""
    agent_id: str
    epsilon: float
    steps_done: int
    episode_rewards: List[float]
    total_reward: float
    memory_size: int
    last_update: datetime
    training_stats: Optional[Dict[str, Any]] = None