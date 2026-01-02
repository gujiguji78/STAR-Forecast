"""
模型训练和推理API服务
提供模型训练、预测、管理的REST API接口
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uuid
import json
from datetime import datetime
import logging
import asyncio
from pathlib import Path
import shutil

import torch
import torch.nn as nn
import numpy as np

from ..models.istr import ISTRNetwork
from ..models.predictor import MultiHeadPredictor
from ..training.trainer import STARForecastTrainer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI应用
app = FastAPI(
    title="模型服务",
    version="1.0.0",
    description="提供模型训练和推理API服务"
)

# 全局模型缓存
model_cache = {}
trainer_cache = {}


class ModelInfo(BaseModel):
    """模型信息"""
    model_id: str
    model_type: str
    status: str
    created_at: str
    last_used: str
    metrics: Dict[str, Any]


class TrainRequest(BaseModel):
    """训练请求"""
    client_id: str
    model_type: str = "istr"
    config: Optional[Dict[str, Any]] = None
    data_path: str = "./data/ETTh1.csv"
    epochs: int = 100
    batch_size: int = 32


class PredictRequest(BaseModel):
    """预测请求"""
    model_id: str
    input_data: List[List[float]]  # [seq_len, features]
    return_features: bool = False


class ModelConfigUpdate(BaseModel):
    """模型配置更新"""
    model_id: str
    updates: Dict[str, Any]


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("🚀 模型服务启动")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭事件"""
    logger.info("🛑 模型服务关闭")

    # 清理模型缓存
    for model_id in list(model_cache.keys()):
        del model_cache[model_id]

    for trainer_id in list(trainer_cache.keys()):
        del trainer_cache[trainer_id]


@app.post("/api/v1/models/train", status_code=202)
async def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    """训练模型"""
    model_id = str(uuid.uuid4())

    # 存储训练任务
    trainer_cache[model_id] = {
        "status": "pending",
        "client_id": request.client_id,
        "created_at": datetime.now().isoformat(),
        "progress": 0.0,
        "metrics": {}
    }

    # 在后台执行训练
    background_tasks.add_task(
        execute_training,
        model_id,
        request.client_id,
        request.model_type,
        request.config or {},
        request.data_path,
        request.epochs,
        request.batch_size
    )

    return {
        "model_id": model_id,
        "status": "training_started",
        "message": "训练任务已提交",
        "timestamp": datetime.now().isoformat()
    }


async def execute_training(model_id: str, client_id: str, model_type: str,
                           config: Dict[str, Any], data_path: str,
                           epochs: int, batch_size: int):
    """执行训练任务"""
    try:
        trainer_cache[model_id]["status"] = "training"
        trainer_cache[model_id]["started_at"] = datetime.now().isoformat()

        logger.info(f"🔧 开始训练模型 {model_id}")

        # 创建训练器
        trainer = STARForecastTrainer()

        # 更新配置
        if config:
            trainer.config.update(config)

        trainer.config['training']['epochs'] = epochs
        trainer.config['data']['batch_size'] = batch_size

        # 构建模型
        trainer.build_models()
        trainer.build_optimizer()

        # 训练
        results = trainer.train(data_path)

        # 保存模型
        model_path = Path(f"./models/{model_id}")
        model_path.mkdir(parents=True, exist_ok=True)

        # 保存模型状态
        torch.save({
            'istr_state_dict': trainer.istr_model.state_dict(),
            'predictor_state_dict': trainer.predictor.state_dict(),
            'config': trainer.config
        }, model_path / "model.pth")

        # 保存结果
        with open(model_path / "results.json", 'w') as f:
            json.dump(results, f, indent=2)

        # 更新缓存
        model_cache[model_id] = {
            "model_type": model_type,
            "model_path": str(model_path),
            "config": trainer.config,
            "results": results,
            "created_at": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat()
        }

        trainer_cache[model_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "progress": 1.0,
            "metrics": results,
            "model_path": str(model_path)
        })

        logger.info(f"✅ 模型训练完成: {model_id}")

    except Exception as e:
        logger.error(f"❌ 模型训练失败: {e}")

        trainer_cache[model_id].update({
            "status": "failed",
            "completed_at": datetime.now().isoformat(),
            "error": str(e)
        })


@app.get("/api/v1/models/training/status/{model_id}")
async def get_training_status(model_id: str):
    """获取训练状态"""
    if model_id not in trainer_cache:
        raise HTTPException(status_code=404, detail="训练任务不存在")

    return trainer_cache[model_id]


@app.post("/api/v1/models/predict")
async def predict(request: PredictRequest):
    """模型预测"""
    if request.model_id not in model_cache:
        raise HTTPException(status_code=404, detail="模型不存在")

    try:
        model_info = model_cache[request.model_id]

        # 加载模型
        model_path = Path(model_info["model_path"])
        checkpoint = torch.load(model_path / "model.pth", map_location='cpu')

        # 创建模型实例
        config = model_info["config"]

        istr_model = ISTRNetwork(config)
        predictor = MultiHeadPredictor(
            hidden_dim=config['istr']['hidden_dim'],
            pred_len=config['data']['pred_len'],
            heads=config['predictor']['heads']
        )

        # 加载权重
        istr_model.load_state_dict(checkpoint['istr_state_dict'])
        predictor.load_state_dict(checkpoint['predictor_state_dict'])

        # 设置为评估模式
        istr_model.eval()
        predictor.eval()

        # 准备输入数据
        input_tensor = torch.FloatTensor(request.input_data).unsqueeze(0)  # [1, seq_len, features]

        with torch.no_grad():
            # 提取特征
            features = istr_model(input_tensor)

            # 预测
            predictions = predictor(features)

            # 转换为列表
            pred_list = predictions.squeeze(0).cpu().numpy().tolist()

            result = {
                "predictions": pred_list,
                "timestamp": datetime.now().isoformat()
            }

            # 如果需要返回特征
            if request.return_features:
                result["features"] = features.squeeze(0).cpu().numpy().tolist()

            # 更新最后使用时间
            model_cache[request.model_id]["last_used"] = datetime.now().isoformat()

            return result

    except Exception as e:
        logger.error(f"❌ 预测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models")
async def list_models():
    """列出所有模型"""
    models = []

    for model_id, model_info in model_cache.items():
        models.append({
            "model_id": model_id,
            "model_type": model_info["model_type"],
            "created_at": model_info["created_at"],
            "last_used": model_info["last_used"],
            "metrics": model_info.get("results", {}).get("test_metrics", {})
        })

    return {
        "models": models,
        "count": len(models)
    }


@app.get("/api/v1/models/{model_id}")
async def get_model_info(model_id: str):
    """获取模型信息"""
    if model_id not in model_cache:
        raise HTTPException(status_code=404, detail="模型不存在")

    model_info = model_cache[model_id].copy()

    # 添加训练历史（如果有）
    if model_id in trainer_cache:
        model_info["training_history"] = trainer_cache[model_id]

    return model_info


@app.delete("/api/v1/models/{model_id}")
async def delete_model(model_id: str):
    """删除模型"""
    if model_id not in model_cache:
        raise HTTPException(status_code=404, detail="模型不存在")

    try:
        # 删除模型文件
        model_path = Path(model_cache[model_id]["model_path"])
        if model_path.exists():
            shutil.rmtree(model_path)

        # 从缓存中删除
        del model_cache[model_id]

        if model_id in trainer_cache:
            del trainer_cache[model_id]

        return {
            "status": "success",
            "message": f"模型 {model_id} 已删除",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ 删除模型失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/models/{model_id}/config")
async def update_model_config(model_id: str, update: ModelConfigUpdate):
    """更新模型配置"""
    if model_id not in model_cache:
        raise HTTPException(status_code=404, detail="模型不存在")

    try:
        model_cache[model_id]["config"].update(update.updates)

        return {
            "status": "success",
            "message": "配置已更新",
            "model_id": model_id,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ 更新配置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/models/upload")
async def upload_model(file: UploadFile = File(...)):
    """上传模型"""
    try:
        # 生成模型ID
        model_id = str(uuid.uuid4())
        model_dir = Path(f"./models/uploaded_{model_id}")
        model_dir.mkdir(parents=True, exist_ok=True)

        # 保存上传的文件
        file_path = model_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 这里可以添加模型验证逻辑
        # 暂时假设上传的是有效的模型文件

        model_cache[model_id] = {
            "model_type": "uploaded",
            "model_path": str(model_dir),
            "filename": file.filename,
            "created_at": datetime.now().isoformat(),
            "last_used": datetime.now().isoformat()
        }

        return {
            "status": "success",
            "message": "模型上传成功",
            "model_id": model_id,
            "filename": file.filename,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"❌ 模型上传失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "模型服务",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "models_cached": len(model_cache),
        "training_tasks": len(trainer_cache)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        workers=2
    )