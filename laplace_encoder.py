import torch
import torch.nn as nn

class LaplaceEncoder(nn.Module):
    """
    这里用“可训练的局部窗口MLP”做Laplace/谱特征的替代编码器（真实工程中常用）：
    - 不引入合成数据
    - 不做昂贵的全图特征分解（否则大数据集会非常慢）
    - 训练稳定、可复现
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32, dropout: float = 0.1, window: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.window = window

        self.encoder = nn.Sequential(
            nn.Linear(input_dim * window, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        b, t, c = x.shape
        w = min(self.window, t)
        recent = x[:, -w:, :].reshape(b, w * c)
        if w < self.window:
            pad = torch.zeros(b, (self.window - w) * c, device=x.device, dtype=x.dtype)
            recent = torch.cat([pad, recent], dim=1)

        z = self.encoder(recent)           # (B, H)
        z = z.unsqueeze(1).expand(-1, t, -1)  # (B, T, H)
        return z
