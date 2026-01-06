# api/server.py
from __future__ import annotations

import os
import re
import json
import time
import yaml
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# 你的真实模型：Laplace + TCN (+ optional LLM)
from models.ensemble_model import TimeSeriesLLM


# ----------------------------
# Config helpers
# ----------------------------
def _load_cfg(config_path: str) -> Dict[str, Any]:
    if not config_path or not isinstance(config_path, str):
        raise ValueError("config_path 不能为空，且必须是字符串路径。")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml 内容必须是一个 dict（YAML 顶层是键值对）。")
    return cfg


def _infer_input_dim_from_cfg(cfg: Dict[str, Any]) -> int:
    """
    从 cfg 推断 input_dim（特征维度）
    - 若 data.feature_cols 是 list：input_dim = len(feature_cols)
    - 否则：默认单变量 input_dim = 1
    """
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    feature_cols = data_cfg.get("feature_cols", None)

    if isinstance(feature_cols, list) and len(feature_cols) > 0:
        return len(feature_cols)

    return 1


def _ensure_input_dim(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    确保 cfg 顶层存在 input_dim，没有则自动补上。
    """
    if "input_dim" not in cfg or cfg["input_dim"] is None:
        inferred = _infer_input_dim_from_cfg(cfg)
        cfg["input_dim"] = inferred
        print(f"[API Auto] 推断 input_dim = {inferred} 并写入 cfg['input_dim']")
    return cfg


def _get_device(cfg: Dict[str, Any]) -> torch.device:
    prefer_cuda = bool(cfg.get("device", {}).get("prefer_cuda", True))
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_checkpoint_path(cfg: Dict[str, Any]) -> str:
    """
    兼容你项目里可能的多种写法：
    - cfg['checkpoint_path']
    - cfg['training']['checkpoint_path']
    - 默认: checkpoints/pretrained_model.pth
    """
    if isinstance(cfg.get("checkpoint_path"), str) and cfg["checkpoint_path"]:
        return cfg["checkpoint_path"]

    tr = cfg.get("training", {})
    if isinstance(tr, dict) and isinstance(tr.get("checkpoint_path"), str) and tr["checkpoint_path"]:
        return tr["checkpoint_path"]

    return "checkpoints/pretrained_model.pth"


# ----------------------------
# Request/Response schemas (OpenAI-compatible)
# ----------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "time-series-llm"
    messages: List[ChatMessage]
    temperature: float = 0.2
    max_tokens: int = 256
    top_p: float = 0.9
    # 允许多余字段（OpenAI兼容时客户端可能会带其它字段）
    class Config:
        extra = "allow"


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


class HealthResponse(BaseModel):
    loaded: bool
    device: str
    llm_enabled: bool
    model_name: str
    time: str

    # 解决你看到的 pydantic protected namespace 警告（真实可用的做法）
    model_config = {"protected_namespaces": ()}


# ----------------------------
# Parsing helpers
# ----------------------------
_LIST_RE = re.compile(r"\[([^\]]+)\]")
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _extract_series_from_text(text: str) -> List[float]:
    """
    从用户输入里提取序列：
    - 优先解析 [...] 里的内容
    - 否则尝试提取所有数字
    """
    if not text:
        return []

    m = _LIST_RE.search(text)
    if m:
        inside = m.group(1)
        nums = _NUM_RE.findall(inside)
        return [float(x) for x in nums] if nums else []

    nums = _NUM_RE.findall(text)
    return [float(x) for x in nums] if nums else []


def _extract_horizon(cfg: Dict[str, Any], text: str) -> int:
    """
    horizon 优先来自 cfg.task.horizon
    如果用户说“预测未来5步/未来 5 步”，尝试覆盖
    """
    default_h = int(cfg.get("task", {}).get("horizon", 10))

    m = re.search(r"(未来|向后|预测)\s*(\d+)\s*步", text)
    if m:
        try:
            return max(1, int(m.group(2)))
        except Exception:
            return default_h

    return default_h


def _to_model_input(series: List[float], input_dim: int) -> torch.Tensor:
    """
    把 list[float] -> Tensor shape [B, T, C]
    单变量：C=1
    多变量：这里默认用户输入是单变量；多变量你应该走 CSV 特征列推理（API 的 text 输入通常是单变量）
    """
    x = torch.tensor(series, dtype=torch.float32).view(1, -1, 1)
    if input_dim != 1:
        # 如果你确实要支持多变量文本输入，需要用户传 [[...],[...]] 结构，这里先严格限制
        raise ValueError(f"当前 input_dim={input_dim}（多变量），但文本输入只支持单变量序列。")
    return x


# ----------------------------
# Model Manager
# ----------------------------
class ModelManager:
    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        self.cfg = cfg
        self.device = device
        self.model: Optional[TimeSeriesLLM] = None
        self.loaded = False
        self.checkpoint_path = _get_checkpoint_path(cfg)

    def load(self) -> None:
        self.cfg = _ensure_input_dim(self.cfg)

        model = TimeSeriesLLM(self.cfg).to(self.device)
        model.eval()

        ckpt = self.checkpoint_path
        if ckpt and os.path.exists(ckpt):
            state = torch.load(ckpt, map_location=self.device)

            # 兼容不同保存格式
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"], strict=False)
            elif isinstance(state, dict):
                # 直接保存 state_dict 的情况
                model.load_state_dict(state, strict=False)
            else:
                raise RuntimeError(f"checkpoint 格式不支持: {type(state)}")

            print(f"[API] Loaded checkpoint: {ckpt}")
        else:
            print(f"[API] WARNING: checkpoint 不存在，继续启动但不加载权重: {ckpt}")

        self.model = model
        self.loaded = True

    @torch.no_grad()
    def predict(self, series: List[float], horizon: int) -> Tuple[List[float], str]:
        if not self.loaded or self.model is None:
            raise RuntimeError("模型尚未加载，无法预测。")

        input_dim = int(self.cfg.get("input_dim", 1))
        x = _to_model_input(series, input_dim).to(self.device)

        # 兼容你的 TimeSeriesLLM 可能提供的不同接口：
        # 1) model.predict(x, horizon=...)
        # 2) model(x, horizon=...)
        # 3) model(x) 直接输出 horizon（不推荐但也兼容）
        out = None
        if hasattr(self.model, "predict"):
            try:
                out = self.model.predict(x, horizon=horizon)
            except TypeError:
                out = self.model.predict(x)
        else:
            try:
                out = self.model(x, horizon=horizon)
            except TypeError:
                out = self.model(x)

        # out 可能是 Tensor / dict / tuple，这里尽量鲁棒处理
        if isinstance(out, dict):
            # 常见键名
            for k in ["preds", "y_pred", "forecast", "output"]:
                if k in out:
                    out = out[k]
                    break

        if isinstance(out, (tuple, list)) and len(out) > 0:
            out = out[0]

        if not torch.is_tensor(out):
            raise RuntimeError(f"模型输出不是 Tensor，无法解析: {type(out)}")

        # 期望 shape: [B, horizon] 或 [B, horizon, 1]
        if out.dim() == 3 and out.size(-1) == 1:
            out = out.squeeze(-1)
        if out.dim() == 2:
            preds = out[0].detach().float().cpu().tolist()
        elif out.dim() == 1:
            preds = out.detach().float().cpu().tolist()
        else:
            raise RuntimeError(f"不支持的输出维度: out.shape={tuple(out.shape)}")

        # 截断/补齐到 horizon（避免模型输出长度不一致）
        preds = preds[:horizon] if len(preds) >= horizon else preds + [preds[-1]] * (horizon - len(preds))

        # LLM 输出（如果 enabled）
        llm_enabled = bool(self.cfg.get("llm", {}).get("enabled", False))
        if llm_enabled and hasattr(self.model, "llm_analyze"):
            try:
                analysis = self.model.llm_analyze(series=series, preds=preds, horizon=horizon)
                if isinstance(analysis, str) and analysis.strip():
                    return preds, analysis
            except Exception as e:
                # LLM 失败不影响预测
                return preds, f"[LLM 调用失败] {e}"

        return preds, ""


# ----------------------------
# FastAPI App
# ----------------------------
def create_app(config_path: str) -> FastAPI:
    cfg = _load_cfg(config_path)
    device = _get_device(cfg)

    manager = ModelManager(cfg=cfg, device=device)

    app = FastAPI(title="TimeSeriesLaplaceTCNLLM (OpenAI Compatible)", version="1.0.0")

    @app.on_event("startup")
    def startup():
        manager.load()

    @app.get("/health", response_model=HealthResponse)
    def health():
        llm_enabled = bool(manager.cfg.get("llm", {}).get("enabled", False))
        return HealthResponse(
            loaded=manager.loaded,
            device=str(manager.device),
            llm_enabled=llm_enabled,
            model_name=str(manager.cfg.get("model_name", "TimeSeriesLaplaceTCNLLM")),
            time=datetime.now().isoformat(),
        )

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages 不能为空")

        # 取最后一条 user message
        last = req.messages[-1]
        user_text = last.content or ""

        series = _extract_series_from_text(user_text)
        if len(series) < 5:
            raise HTTPException(status_code=400, detail="未能从输入中提取到足够的数值序列（至少 5 个点）。")

        horizon = _extract_horizon(manager.cfg, user_text)

        preds, analysis = manager.predict(series=series, horizon=horizon)

        # 组装 assistant 内容：数值预测 + 可选 LLM 分析
        content = f"预测结果（未来{horizon}步）：\n{json.dumps(preds, ensure_ascii=False)}"
        if analysis.strip():
            content += "\n\n解释：\n" + analysis.strip()

        # OpenAI compatible response
        created = int(time.time())
        resp = ChatCompletionResponse(
            id=f"chatcmpl-{created}",
            created=created,
            model=req.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=len(user_text.split()),
                completion_tokens=len(content.split()),
                total_tokens=len(user_text.split()) + len(content.split()),
            ),
        )
        return JSONResponse(resp.model_dump())

    return app


def run_api(config_path: str, host: str = "127.0.0.1", port: int = 8000):
    """
    给 run.py 调用的入口：run_api(config_path=...)
    """
    import uvicorn

    app = create_app(config_path=config_path)
    print(f"API启动：请访问 http://{host}:{port}/docs （不要用0.0.0.0）")
    uvicorn.run(app, host=host, port=port, log_level="info")
