# training/pretrain_tcn.py
from __future__ import annotations

import os
import math
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from models.ensemble_model import TimeSeriesLLM
from training.data_loader import build_dataloaders  # 你项目里应已有：从真实CSV构建 train/val/test
from training.metrics import compute_metrics  # 你项目里应已有：MSE/MAE/RMSE等


def load_config(path: str = "config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_input_dim_from_loader(train_loader) -> int:
    # 从真实 batch 推断 input_dim
    batch = next(iter(train_loader))
    # 兼容 batch = (x,y) 或 dict
    if isinstance(batch, (list, tuple)):
        x = batch[0]
    elif isinstance(batch, dict):
        x = batch["x"]
    else:
        raise ValueError(f"无法解析 DataLoader batch 类型: {type(batch)}")

    if not isinstance(x, torch.Tensor):
        raise ValueError("DataLoader 输出的 x 不是 torch.Tensor，请检查 data_loader.py")

    if x.dim() != 3:
        raise ValueError(f"期望 x 形状为 (B,T,C)，但得到 {tuple(x.shape)}")

    return int(x.shape[-1])


def pretrain_tcn(config_path: str = "config.yaml"):
    cfg = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("开始预训练：Laplace + TCN（真实数据）")
    print("=" * 60)
    print(f"使用设备: {device}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # 1) 构建 DataLoaders（真实CSV）
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # 2) 自动推断 input_dim（如果 cfg 里没有）
    if cfg.get("input_dim", None) is None:
        inferred = infer_input_dim_from_loader(train_loader)
        cfg["input_dim"] = inferred
        print(f"[Auto] 从真实数据推断 input_dim = {inferred}，并写入 cfg['input_dim']")

    # 3) 构建模型
    model = TimeSeriesLLM(cfg).to(device)

    # 4) loss / optim
    lr = float(cfg.get("train", {}).get("lr", 1e-3))
    weight_decay = float(cfg.get("train", {}).get("weight_decay", 1e-4))
    epochs = int(cfg.get("train", {}).get("epochs", 20))

    criterion = nn.MSELoss()
    optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 5) 训练循环
    best_val_mse = math.inf
    ckpt_dir = cfg.get("train", {}).get("checkpoint_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "pretrained_model.pth")

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}")
        train_losses = []

        for batch in pbar:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch["x"], batch["y"]

            x = x.to(device)
            y = y.to(device)

            pred = model(x)  # (B, horizon)
            loss = criterion(pred, y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()

            train_losses.append(loss.item())
            pbar.set_postfix(train_mse=float(loss.item()))

        # 6) val
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0], batch[1]
                else:
                    x, y = batch["x"], batch["y"]
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                val_preds.append(pred.detach().cpu())
                val_trues.append(y.detach().cpu())

        val_preds = torch.cat(val_preds, dim=0)
        val_trues = torch.cat(val_trues, dim=0)

        metrics = compute_metrics(val_preds, val_trues)  # 返回 dict: mse/mae/rmse
        train_mse = float(sum(train_losses) / max(1, len(train_losses)))

        print(
            f"\n[Epoch {ep}] TrainMSE={train_mse:.6f} | "
            f"ValMSE={metrics['mse']:.6f} ValMAE={metrics['mae']:.6f} ValRMSE={metrics['rmse']:.6f}"
        )

        if metrics["mse"] < best_val_mse:
            best_val_mse = metrics["mse"]
            torch.save({"model_state_dict": model.state_dict(), "config": cfg}, best_path)
            print(f"✓ 保存最佳模型: {best_path} (best ValMSE={best_val_mse:.6f})")

    print("训练结束。")
    print(f"Best ValMSE={best_val_mse:.6f} | Best model saved to: {best_path}")
