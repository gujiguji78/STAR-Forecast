"""
AutoGen智能体对话API服务
提供多智能体对话的REST API接口
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uuid
import json
from datetime import datetime
import logging
import asyncio

from ..agents.autogen_system import AutoGenMultiAgentSystem, ConversationResult

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI应用
app = FastAPI(
    title="AutoGen对话服务",
    version="1.0.0",
    description="提供多智能体对话API服务"
)

# 全局AutoGen系统实例
autogen_system = None


class ConversationRequest(BaseModel):
    """对话请求"""
    context: Dict[str, Any]
    client_id: str
    require_summary: bool = True
    max_rounds: Optional[int] = None


class ConversationResponse(BaseModel):
    """对话响应"""
    conversation_id: str
    summary: Optional[str]
    consensus: Optional[Dict[str, Any]]
    metrics: Dict[str, Any]
    messages_count: int
    timestamp: str


class ConversationHistoryRequest(BaseModel):
    """对话历史请求"""
    client_id: str
    limit: int = 10


def init_autogen_system(config_path: str = "./config.yaml"):
    """初始化AutoGen系统"""
    global autogen_system

    if autogen_system is not None:
        return autogen_system

    # 加载配置
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    autogen_system = AutoGenMultiAgentSystem(config)
    logger.info("✅ AutoGen系统初始化完成")

    return autogen_system


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("🚀 AutoGen对话服务启动")
    init_autogen_system()


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("🛑 AutoGen对话服务关闭")


@app.post("/api/v1/conversation/start", response_model=ConversationResponse)
async def start_conversation(request: ConversationRequest):
    """启动对话"""
    if autogen_system is None:
        raise HTTPException(status_code=500, detail="AutoGen系统未初始化")

    try:
        # 如果指定了最大轮数，临时修改配置
        original_max_rounds = autogen_system.config['autogen']['conversation']['max_rounds']
        if request.max_rounds is not None:
            autogen_system.config['autogen']['conversation']['max_rounds'] = request.max_rounds

        # 启动对话
        result = autogen_system.initiate_conversation(request.context)

        # 恢复原始配置
        if request.max_rounds is not None:
            autogen_system.config['autogen']['conversation']['max_rounds'] = original_max_rounds

        # 构建响应
        response = ConversationResponse(
            conversation_id=result.conversation_id,
            summary=result.summary if request.require_summary else None,
            consensus=result.consensus,
            metrics=result.metrics,
            messages_count=len(result.messages),
            timestamp=datetime.now().isoformat()
        )

        logger.info(f"💬 对话完成: {result.conversation_id}, "
                    f"消息数: {len(result.messages)}, "
                    f"共识: {result.consensus is not None}")

        return response

    except Exception as e:
        logger.error(f"❌ 对话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/conversation/history")
async def get_conversation_history(request: ConversationHistoryRequest):
    """获取对话历史"""
    if autogen_system is None:
        raise HTTPException(status_code=500, detail="AutoGen系统未初始化")

    try:
        history = autogen_system.get_conversation_history(request.limit)

        return {
            "client_id": request.client_id,
            "history": history,
            "count": len(history),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ 获取历史失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/conversation/{conversation_id}")
async def get_conversation_details(conversation_id: str):
    """获取对话详情"""
    if autogen_system is None:
        raise HTTPException(status_code=500, detail="AutoGen系统未初始化")

    if conversation_id not in autogen_system.conversation_history:
        raise HTTPException(status_code=404, detail="对话不存在")

    try:
        result = autogen_system.conversation_history[conversation_id]

        # 转换消息为可序列化格式
        messages = []
        for msg in result.messages:
            messages.append({
                "id": msg.id,
                "timestamp": msg.timestamp.isoformat(),
                "sender": msg.sender,
                "role": msg.role,
                "content": msg.content[:500] + "..." if len(msg.content) > 500 else msg.content,
                "metadata": msg.metadata
            })

        return {
            "conversation_id": conversation_id,
            "messages": messages,
            "consensus": result.consensus,
            "decisions": result.decisions,
            "summary": result.summary,
            "metrics": result.metrics
        }

    except Exception as e:
        logger.error(f"❌ 获取对话详情失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/conversation/clear")
async def clear_conversation_history(client_id: str):
    """清空对话历史"""
    if autogen_system is None:
        raise HTTPException(status_code=500, detail="AutoGen系统未初始化")

    try:
        autogen_system.clear_history()

        return {
            "status": "success",
            "message": "对话历史已清空",
            "client_id": client_id,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ 清空历史失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "AutoGen对话服务",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "system_initialized": autogen_system is not None
    }


@app.get("/api/v1/agents")
async def list_agents():
    """列出所有智能体"""
    if autogen_system is None:
        raise HTTPException(status_code=500, detail="AutoGen系统未初始化")

    try:
        agents = []
        for agent_key, agent_config in autogen_system.agent_configs.items():
            agents.append({
                "id": agent_key,
                "name": agent_config.name,
                "role": agent_config.role.value,
                "model": agent_config.model,
                "description": agent_config.description,
                "capabilities": agent_config.capabilities
            })

        return {
            "agents": agents,
            "count": len(agents)
        }

    except Exception as e:
        logger.error(f"❌ 获取智能体列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        workers=2
    )