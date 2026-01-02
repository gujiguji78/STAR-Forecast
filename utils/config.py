"""
配置管理模块 - STAR-Forecast
提供配置文件的加载和管理功能
"""

import os
import yaml
import json
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict
import logging


@dataclass
class DataConfig:
    """数据配置"""
    name: str = "ETTh1"
    seq_len: int = 96
    pred_len: int = 24
    label_len: int = 48
    features: str = "M"
    target: str = "OT"
    scale: bool = True
    timeenc: int = 0
    freq: str = "h"
    train_split: float = 0.7
    val_split: float = 0.2
    test_split: float = 0.1
    data_path: Optional[str] = None


@dataclass
class ModelConfig:
    """模型配置"""
    # Transformer 配置
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 2048
    dropout: float = 0.05
    activation: str = "gelu"

    # STAR 架构配置
    use_neural: bool = True
    use_symbolic: bool = True
    use_rl: bool = True
    symbolic_rules: int = 5
    rl_epochs: int = 10

    # 模型尺寸
    enc_in: int = 7
    c_out: int = 1


@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    patience: int = 10
    gradient_clip: float = 1.0
    weight_decay: float = 0.0001

    # 学习率调度
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 0.00001

    # 检查点
    checkpoint_dir: str = "checkpoints"
    save_frequency: int = 5
    checkpoint_path: str = "checkpoints/best_model.pth"

    # 早停
    early_stopping: bool = True
    min_delta: float = 0.001

    # 优化器
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/starcast.log"
    console_output: bool = True


@dataclass
class ServerConfig:
    """服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    debug: bool = False


@dataclass
class Config:
    """全局配置类"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    device: str = "auto"
    seed: int = 42

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """从字典创建配置"""
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        server_config = ServerConfig(**config_dict.get('server', {}))

        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            logging=logging_config,
            server=server_config,
            device=config_dict.get('device', 'auto'),
            seed=config_dict.get('seed', 42)
        )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """从YAML文件加载配置"""
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"配置文件不存在: {yaml_path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, json_path: str) -> 'Config':
        """从JSON文件加载配置"""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"配置文件不存在: {json_path}")

        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'logging': asdict(self.logging),
            'server': asdict(self.server),
            'device': self.device,
            'seed': self.seed
        }

    def to_yaml(self, yaml_path: str):
        """保存为YAML文件"""
        config_dict = self.to_dict()

        # 确保目录存在
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

    def to_json(self, json_path: str):
        """保存为JSON文件"""
        config_dict = self.to_dict()

        # 确保目录存在
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)


def load_config(config_path: Optional[str] = None, args: Optional[Any] = None) -> Config:
    """
    加载配置

    参数:
        config_path: 配置文件路径，如果为None则使用默认配置
        args: 命令行参数，用于覆盖配置文件中的设置

    返回:
        Config: 配置对象
    """
    # 尝试加载配置文件
    config = None
    if config_path and os.path.exists(config_path):
        try:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config = Config.from_yaml(config_path)
            elif config_path.endswith('.json'):
                config = Config.from_json(config_path)
            else:
                # 尝试YAML格式
                try:
                    config = Config.from_yaml(config_path)
                except:
                    # 尝试JSON格式
                    config = Config.from_json(config_path)
        except Exception as e:
            logging.warning(f"配置文件加载失败: {e}, 使用默认配置")

    # 如果没有配置文件，使用默认配置
    if config is None:
        config = Config()

    # 使用命令行参数覆盖配置
    if args:
        if hasattr(args, 'data') and args.data:
            config.data.name = args.data

        if hasattr(args, 'seq_len') and args.seq_len:
            config.data.seq_len = args.seq_len

        if hasattr(args, 'pred_len') and args.pred_len:
            config.data.pred_len = args.pred_len

        if hasattr(args, 'label_len') and args.label_len:
            config.data.label_len = args.label_len

        if hasattr(args, 'epochs') and args.epochs:
            config.training.epochs = args.epochs

        if hasattr(args, 'batch_size') and args.batch_size:
            config.training.batch_size = args.batch_size

        if hasattr(args, 'lr') and args.lr:
            config.training.learning_rate = args.lr

        if hasattr(args, 'device') and args.device:
            config.device = args.device

        if hasattr(args, 'seed') and args.seed:
            config.seed = args.seed

    return config