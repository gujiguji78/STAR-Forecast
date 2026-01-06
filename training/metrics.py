# training/metrics.py
from __future__ import annotations

import torch


@torch.no_grad()
def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    """
    计算回归任务常见指标：MSE / MAE / RMSE
    - 输入必须是 torch.Tensor
    - 自动拉平到 (N,) 再算
    """
    if not isinstance(y_true, torch.Tensor) or not isinstance(y_pred, torch.Tensor):
        raise TypeError("compute_metrics 仅接受 torch.Tensor 输入")

    # 保证同设备同dtype
    y_true = y_true.detach()
    y_pred = y_pred.detach()
    if y_true.device != y_pred.device:
        y_pred = y_pred.to(y_true.device)

    # 拉平成一维，避免 (B,T,1) 等形状造成广播问题
    y_true = y_true.reshape(-1).float()
    y_pred = y_pred.reshape(-1).float()

    mse = torch.mean((y_pred - y_true) ** 2)
    mae = torch.mean(torch.abs(y_pred - y_true))
    rmse = torch.sqrt(mse + 1e-12)

    return {
        "mse": float(mse.item()),
        "mae": float(mae.item()),
        "rmse": float(rmse.item()),
    }
