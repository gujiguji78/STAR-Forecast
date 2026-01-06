# models/ensemble_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Laplace Encoder (真实可用：邻域平滑的图拉普拉斯近似)
# - 不做昂贵特征分解，避免 CPU 卡死
# - 输入:  (B, T, C)
# - 输出:  (B, T, H)
# -------------------------
class LaplaceEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, k: int = 10, sigma: float = 1.0):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.k = int(k)
        self.sigma = float(sigma)

        self.proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.hidden_dim)

    @torch.no_grad()
    def _knn_graph(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        x: (B, T, H)
        return: idx (B, T, k)
        用欧氏距离做 KNN（T<=几百时可接受；你 seq_len=100 很OK）
        """
        B, T, H = x.shape
        # dist: (B, T, T)
        dist = torch.cdist(x, x, p=2)
        # 自己到自己距离设为极大，避免选到自己
        dist.diagonal(dim1=1, dim2=2).fill_(1e9)
        idx = torch.topk(dist, k=min(k, T - 1), largest=False).indices  # (B, T, k)
        return idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)
        """
        h = self.proj(x)  # (B,T,H)

        # 邻域平滑（近似拉普拉斯）
        with torch.no_grad():
            idx = self._knn_graph(h, self.k)  # (B,T,k)

        # gather neighbors: (B,T,k,H)
        B, T, H = h.shape
        k_eff = idx.shape[-1]
        idx_exp = idx.unsqueeze(-1).expand(B, T, k_eff, H)
        neigh = torch.gather(h.unsqueeze(2).expand(B, T, T, H), 2, idx_exp)  # (B,T,k,H)

        # 权重：RBF
        # dist to neighbors: (B,T,k)
        dist = torch.norm(neigh - h.unsqueeze(2), dim=-1)
        w = torch.exp(-(dist ** 2) / (2 * (self.sigma ** 2) + 1e-8))  # (B,T,k)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)

        smooth = (w.unsqueeze(-1) * neigh).sum(dim=2)  # (B,T,H)
        lap = h - smooth                               # (B,T,H)

        out = self.out(torch.tanh(lap))                # (B,T,H)
        return out


# -------------------------
# TCN Encoder (真实可用)
# 输入:  (B, C, T)
# 输出:  (B, H)
# -------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNEncoder(nn.Module):
    def __init__(self, input_dim: int, channels: List[int], kernel_size: int = 3, dropout: float = 0.2, output_size: int = 64):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(channels):
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size=kernel_size, dilation=2**i, dropout=dropout)
            )
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(in_ch, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        return: (B, output_size)
        """
        y = self.network(x)        # (B, last_ch, T)
        last = y[:, :, -1]         # (B, last_ch)
        return self.head(last)     # (B, output_size)


# -------------------------
# LLM Adapter (可选)
# 这里假设你项目已有 models/llm_adapter.py
# 如果没有也不会影响训练（llm.enabled=false 或不调用）
# -------------------------
class _DummyLLM(nn.Module):
    def __init__(self):
        super().__init__()

    def enabled(self) -> bool:
        return False

    def generate(self, prompt: str, **kwargs) -> str:
        return ""


# -------------------------
# 主模型：Laplace + TCN + Fusion + RegressionHead + (Optional) LLM
# -------------------------
class TimeSeriesLLM(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

        # 核心维度
        if "input_dim" not in cfg:
            raise ValueError("cfg 缺少 input_dim（特征维度）。单变量=1，多变量=feature_cols数量。")
        self.input_dim = int(cfg["input_dim"])
        self.horizon = int(cfg.get("task", {}).get("horizon", 10))

        # Laplace
        lap = cfg.get("laplace", {})
        self.laplace = LaplaceEncoder(
            input_dim=self.input_dim,
            hidden_dim=int(lap.get("hidden_dim", 32)),
            k=int(lap.get("k", lap.get("n_neighbors", 10))),   # 兼容你旧字段 n_neighbors
            sigma=float(lap.get("sigma", 1.0)),
        )

        # TCN
        tcn = cfg.get("tcn", {})
        self.tcn = TCNEncoder(
            input_dim=self.input_dim,
            channels=list(tcn.get("channels", [32, 64, 128])),
            kernel_size=int(tcn.get("kernel_size", 3)),
            dropout=float(tcn.get("dropout", 0.2)),
            output_size=int(tcn.get("output_size", 64)),
        )

        # Fusion
        fusion = cfg.get("fusion", {})
        fusion_hidden = int(fusion.get("hidden_dim", 256))
        fusion_dropout = float(fusion.get("dropout", 0.2))

        # laplace pooling -> (B, lap_hidden)
        lap_hidden = int(lap.get("hidden_dim", 32))
        tcn_out = int(tcn.get("output_size", 64))
        self.fusion_mlp = nn.Sequential(
            nn.Linear(lap_hidden + tcn_out, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden, fusion_hidden),
            nn.ReLU(),
        )

        # Regression head：关键改动 -> 输出 horizon
        reg = cfg.get("regression", {})
        reg_hidden = int(reg.get("hidden_dim", 128))
        self.reg_head = nn.Sequential(
            nn.Linear(fusion_hidden, reg_hidden),
            nn.ReLU(),
            nn.Linear(reg_hidden, self.horizon),  # ✅ 输出 horizon
        )

        # Optional LLM
        llm_cfg = cfg.get("llm", {})
        self.llm_enabled = bool(llm_cfg.get("enabled", False))
        self.llm = _DummyLLM()
        if self.llm_enabled:
            try:
                from models.llm_adapter import LLMAdapter  # 你项目已有
                self.llm = LLMAdapter(cfg)
            except Exception as e:
                # 不让训练崩：训练阶段通常不需要 LLM 参与 loss
                print(f"[WARN] LLM enabled but failed to init LLMAdapter: {e}")
                self.llm_enabled = False
                self.llm = _DummyLLM()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C)
        return: preds (B, horizon)
        """
        # Laplace branch
        lap_feat = self.laplace(x)          # (B,T,Hlap)
        lap_pool = lap_feat.mean(dim=1)     # (B,Hlap)

        # TCN branch
        tcn_in = x.transpose(1, 2)          # (B,C,T)
        tcn_feat = self.tcn(tcn_in)         # (B,Htcn)

        # Fusion + Regression
        fused = torch.cat([lap_pool, tcn_feat], dim=-1)
        fused = self.fusion_mlp(fused)
        preds = self.reg_head(fused)        # ✅ (B,horizon)
        return preds

    @torch.no_grad()
    def llm_explain(self, series: List[float], preds: List[float]) -> str:
        """
        给 API 用：用 LLM 对预测做解释（可选）
        """
        if not self.llm_enabled:
            return ""

        import numpy as np
        arr = np.array(series, dtype=np.float32)
        mean = float(arr.mean())
        std = float(arr.std())
        mn = float(arr.min())
        mx = float(arr.max())
        # 线性趋势
        t = np.arange(len(arr), dtype=np.float32)
        slope = float(np.polyfit(t, arr, deg=1)[0])

        llm_cfg = self.cfg.get("llm", {})
        template = llm_cfg.get("prompt_template", "")
        horizon = len(preds)

        prompt = template.format(
            mean=mean, std=std, min=mn, max=mx, slope=slope,
            horizon=horizon, preds=preds
        )

        gen_cfg = llm_cfg.get("generation", {})
        return self.llm.generate(prompt, **gen_cfg)
