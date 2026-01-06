import os
import yaml
import torch
from models.ensemble_model import TimeSeriesLLM

class ModelManager:
    def __init__(self):
        self.model = None
        self.cfg = None

    def load(self, config_path="configs/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        ckpt_path = self.cfg.get("api_model_ckpt", "checkpoints/pretrained_model.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"API模型ckpt不存在: {ckpt_path}（请先训练）")

        # API阶段允许启用LLM（但必须本地模型路径有效）
        model = TimeSeriesLLM(self.cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.eval()
        self.model = model
        return True

    def get(self):
        if self.model is None:
            raise RuntimeError("模型未加载")
        return self.model

model_manager = ModelManager()
