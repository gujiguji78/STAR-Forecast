# training/evaluate_test.py
import os
import glob
import yaml
import torch

from models.ensemble_model import TimeSeriesLLM
from training.data_loader import build_dataloaders
from training.metrics import compute_metrics


def _default_config_path(config_path: str | None) -> str:
    # 兼容：不传就默认项目根目录的 config.yaml
    if config_path and os.path.exists(config_path):
        return config_path

    candidates = [
        "config.yaml",
        os.path.join("configs", "config.yaml"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "找不到 config.yaml。请把 config.yaml 放在项目根目录，"
        "或者运行时指定：python run.py --mode test --config <path>"
    )


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


def _infer_input_dim_from_cfg(cfg: dict) -> int:
    """
    真实推断 input_dim：
    - 如果 data.feature_cols 是列表 => input_dim=len(feature_cols)
    - 否则 => 单变量 => 1
    """
    data_cfg = cfg.get("data", {}) or {}
    feature_cols = data_cfg.get("feature_cols", None)

    if isinstance(feature_cols, list) and len(feature_cols) > 0:
        return len(feature_cols)

    # 单变量：只用 target_col 或默认第一列数值
    return 1


def _resolve_device(cfg: dict) -> torch.device:
    prefer_cuda = bool((cfg.get("device", {}) or {}).get("prefer_cuda", True))
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_test(config_path: str | None = None):
    config_path = _default_config_path(config_path)
    cfg = load_config(config_path)

    print("============================================================")
    print("开始 TEST 评估（真实数据）")
    print("============================================================")

    device = _resolve_device(cfg)
    print(f"使用设备: {device}")

    # ✅ 关键：test 也要推断 input_dim
    if "input_dim" not in cfg or cfg["input_dim"] is None:
        inferred = _infer_input_dim_from_cfg(cfg)
        cfg["input_dim"] = int(inferred)
        print(f"[Auto] 从 cfg 推断 input_dim = {cfg['input_dim']} 并写入 cfg['input_dim']")

    # dataloaders（真实 CSV）
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # model
    model = TimeSeriesLLM(cfg).to(device)

    # load ckpt
    ckpt_path = cfg.get("training", {}).get("ckpt_path", None)
    if not ckpt_path:
        # 兼容你之前保存的默认路径
        ckpt_path = os.path.join("checkpoints", "pretrained_model.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"找不到模型权重：{ckpt_path}\n"
            f"请先训练生成 checkpoints/pretrained_model.pth，或在 config.yaml 里设置 training.ckpt_path"
        )

    state = torch.load(ckpt_path, map_location="cpu")
    # 兼容两种保存格式：直接 state_dict 或 {'model_state_dict': ...}
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)

    model.eval()

    preds_all = []
    trues_all = []

    with torch.no_grad():
        for batch in test_loader:
            # 兼容你的 dataloader 可能返回 dict 或 tuple
            if isinstance(batch, dict):
                x = batch["x"]
                y = batch["y"]
            else:
                x, y = batch

            x = x.to(device)
            y = y.to(device)

            out = model(x)  # 期望 out shape: [B, horizon] 或 [B, 1]（取决于你模型）
            preds_all.append(out.detach().cpu())
            trues_all.append(y.detach().cpu())

    y_pred = torch.cat(preds_all, dim=0)
    y_true = torch.cat(trues_all, dim=0)

    metrics = compute_metrics(y_pred, y_true)

    print("============================================================")
    print("TEST METRICS (Real Data)")
    print(f"MSE : {metrics['mse']:.6f}")
    print(f"MAE : {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print("============================================================")

    return metrics
