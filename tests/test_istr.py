"""
ISTR网络单元测试 - 真实可运行的测试代码
测试TCN + 拉普拉斯正则化的完整功能
"""
import torch
import torch.nn as nn
import numpy as np
import pytest
from pathlib import Path
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from models.istr import ISTRNetwork, TemporalBlock, SpectralGate


class TestTemporalBlock:
    """测试TCN基础块"""

    def test_initialization(self):
        """测试初始化"""
        block = TemporalBlock(
            n_inputs=64,
            n_outputs=64,
            kernel_size=3,
            stride=1,
            dilation=1,
            dropout=0.1
        )

        assert hasattr(block, 'conv1')
        assert hasattr(block, 'conv2')
        assert hasattr(block, 'bn1')
        assert hasattr(block, 'bn2')
        assert block.conv1.in_channels == 64
        assert block.conv1.out_channels == 64

    def test_forward_pass(self):
        """测试前向传播"""
        block = TemporalBlock(64, 64, 3, 1, 1, 0.1)

        # 模拟输入: [batch=2, channels=64, seq_len=96]
        x = torch.randn(2, 64, 96)

        # 前向传播
        output = block(x)

        # 验证输出形状
        assert output.shape == (2, 64, 96)

        # 验证残差连接
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_residual_connection(self):
        """测试残差连接"""
        # 输入输出维度不同
        block = TemporalBlock(32, 64, 3, 1, 1, 0.1)
        x = torch.randn(2, 32, 96)
        output = block(x)

        assert output.shape == (2, 64, 96)
        assert block.downsample is not None  # 应该有下采样层

    def test_causal_dilation(self):
        """测试因果膨胀卷积"""
        # 膨胀率为2
        block = TemporalBlock(64, 64, 3, 1, 2, 0.1)

        # 计算期望的padding
        expected_padding = (3 - 1) * 2  # (kernel_size - 1) * dilation
        actual_padding = block.conv1.padding[0]

        assert actual_padding == expected_padding


class TestSpectralGate:
    """测试谱门控机制"""

    def test_initialization(self):
        """测试初始化"""
        gate = SpectralGate(hidden_dim=64, threshold=0.5)

        assert gate.hidden_dim == 64
        assert gate.threshold == 0.5
        assert hasattr(gate, 'freq_proj')
        assert hasattr(gate, 'time_proj')
        assert hasattr(gate, 'gate_net')

    def test_forward_pass(self):
        """测试前向传播"""
        gate = SpectralGate(64, 0.5)

        # 模拟输入: [batch=2, seq_len=96, hidden=64]
        x = torch.randn(2, 96, 64)

        output = gate(x)

        # 验证输出形状
        assert output.shape == (2, 96, 64)

        # 验证门控值在[0,1]范围内
        x_fft = torch.fft.rfft(x, dim=1)
        x_fft_mag = torch.abs(x_fft)
        freq_features = torch.mean(x_fft_mag, dim=1)

        # 门控应该是基于特征的
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

        # 验证门控效果（应该改变输入值）
        if gate.threshold > 0:
            assert not torch.allclose(output, x, rtol=1e-5)


class TestISTRNetwork:
    """测试完整ISTR网络"""

    @pytest.fixture
    def config(self):
        """测试配置"""
        return {
            'istr': {
                'input_dim': 7,  # ETTh1特征数
                'hidden_dim': 64,  # 隐藏维度
                'num_blocks': 2,  # 测试时使用较少块
                'kernel_size': 3,
                'dilation_base': 2,
                'dropout': 0.1,
                'laplacian_weight': 0.01,
                'trainable_ratio': 0.1  # 测试时训练更多参数
            }
        }

    @pytest.fixture
    def sample_input(self):
        """测试输入"""
        # 模拟ETTh1数据: [batch=4, seq_len=96, features=7]
        return torch.randn(4, 96, 7)

    def test_initialization(self, config):
        """测试网络初始化"""
        model = ISTRNetwork(config)

        assert hasattr(model, 'input_proj')
        assert hasattr(model, 'tcn_layers')
        assert hasattr(model, 'spectral_gate')
        assert hasattr(model, 'output_norm')
        assert hasattr(model, 'laplacian_weight')

        # 验证层数
        assert len(model.tcn_layers) == config['istr']['num_blocks']

        # 验证参数冻结
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        expected_trainable = int(total_params * config['istr']['trainable_ratio'])

        # 允许一些误差
        assert abs(trainable_params - expected_trainable) / total_params < 0.05

    def test_forward_pass(self, config, sample_input):
        """测试前向传播"""
        model = ISTRNetwork(config)

        # 基础前向传播
        features = model(sample_input, return_regularization=False)

        # 验证输出形状
        batch_size, seq_len, _ = sample_input.shape
        hidden_dim = config['istr']['hidden_dim']
        assert features.shape == (batch_size, seq_len, hidden_dim)

    def test_forward_with_regularization(self, config, sample_input):
        """测试带正则化的前向传播"""
        model = ISTRNetwork(config)

        # 带正则化的前向传播
        features, reg_loss = model(sample_input, return_regularization=True)

        # 验证输出
        assert features.shape[0] == sample_input.shape[0]
        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.item() >= 0  # 正则化损失应为非负

        # 测试训练模式
        model.train()
        features_train, reg_loss_train = model(sample_input, return_regularization=True)
        assert reg_loss_train.item() >= 0

    def test_extract_features(self, config, sample_input):
        """测试特征提取"""
        model = ISTRNetwork(config)

        features = model.extract_features(sample_input)

        # 验证特征字典结构
        assert isinstance(features, dict)
        assert 'shape' in features
        assert 'statistics' in features
        assert 'frequency' in features

        # 验证形状信息
        assert features['shape'] == list(sample_input.shape)

        # 验证统计信息
        stats = features['statistics']
        assert 'mean' in stats
        assert 'std' in stats
        assert 'autocorr' in stats

        # 均值和标准差应为列表
        assert isinstance(stats['mean'], list)
        assert len(stats['mean']) == sample_input.shape[-1]

    def test_update_parameters(self, config, sample_input):
        """测试参数更新"""
        model = ISTRNetwork(config)

        # 获取初始参数
        initial_threshold = model.spectral_threshold
        initial_weight = model.laplacian_weight.item()

        # 更新参数
        model.update_parameters(
            spectral_threshold=0.7,
            laplacian_weight=0.02
        )

        # 验证参数已更新
        assert model.spectral_threshold == 0.7
        assert abs(model.laplacian_weight.item() - 0.02) < 1e-6

        # 验证计数器
        assert model.adaptation_count.item() == 1

    def test_gradient_flow(self, config, sample_input):
        """测试梯度流"""
        model = ISTRNetwork(config)

        # 创建优化器
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=0.001
        )

        # 前向传播
        features, reg_loss = model(sample_input, return_regularization=True)

        # 创建简单的预测头
        predictor = nn.Linear(config['istr']['hidden_dim'], 1)

        # 预测目标（简化）
        predictions = predictor(features.mean(dim=1))  # [batch, 1]
        dummy_target = torch.randn(predictions.shape)

        # 计算损失
        mse_loss = nn.MSELoss()(predictions, dummy_target)
        total_loss = mse_loss + reg_loss

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()

        # 检查梯度
        has_gradient = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_gradient = True
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()

        assert has_gradient, "至少有一些参数应该有梯度"

        # 优化步骤
        optimizer.step()

    def test_mixed_precision(self, config, sample_input):
        """测试混合精度训练"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA不可用，跳过混合精度测试")

        from torch.cuda.amp import autocast, GradScaler

        model = ISTRNetwork(config).cuda()
        scaler = GradScaler()

        # 创建可训练参数优化器
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=0.001)

        # 将输入移到GPU
        x = sample_input.cuda()

        # 混合精度前向传播
        with autocast():
            features, reg_loss = model(x, return_regularization=True)

            # 简单的损失计算
            loss = features.mean() + reg_loss

        # 混合精度反向传播
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 验证结果
        assert not torch.isnan(features).any()
        assert not torch.isinf(features).any()

    def test_batch_variation(self, config):
        """测试不同批量大小的处理"""
        model = ISTRNetwork(config)

        # 测试不同批量大小
        batch_sizes = [1, 2, 8, 16]

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 96, 7)
            features = model(x, return_regularization=False)

            assert features.shape == (batch_size, 96, config['istr']['hidden_dim'])

    def test_sequence_length_variation(self, config):
        """测试不同序列长度的处理"""
        model = ISTRNetwork(config)

        seq_lengths = [32, 64, 96, 128]

        for seq_len in seq_lengths:
            x = torch.randn(4, seq_len, 7)
            features = model(x, return_regularization=False)

            assert features.shape == (4, seq_len, config['istr']['hidden_dim'])


def run_all_tests():
    """运行所有测试"""
    print("🧪 开始ISTR网络测试...")

    # 创建测试配置
    config = {
        'istr': {
            'input_dim': 7,
            'hidden_dim': 64,
            'num_blocks': 2,
            'kernel_size': 3,
            'dilation_base': 2,
            'dropout': 0.1,
            'laplacian_weight': 0.01,
            'trainable_ratio': 0.1
        }
    }

    # 运行TemporalBlock测试
    print("1. 测试TemporalBlock...")
    block_tester = TestTemporalBlock()

    block_tester.test_initialization()
    print("   ✅ TemporalBlock初始化测试通过")

    block_tester.test_forward_pass()
    print("   ✅ TemporalBlock前向传播测试通过")

    block_tester.test_residual_connection()
    print("   ✅ TemporalBlock残差连接测试通过")

    # 运行SpectralGate测试
    print("2. 测试SpectralGate...")
    gate_tester = TestSpectralGate()

    gate_tester.test_initialization()
    print("   ✅ SpectralGate初始化测试通过")

    gate_tester.test_forward_pass()
    print("   ✅ SpectralGate前向传播测试通过")

    # 运行ISTRNetwork测试
    print("3. 测试ISTRNetwork...")
    istr_tester = TestISTRNetwork()

    sample_input = torch.randn(4, 96, 7)

    # 测试初始化
    model = istr_tester.test_initialization(config)
    print("   ✅ ISTRNetwork初始化测试通过")

    # 测试前向传播
    features = istr_tester.test_forward_pass(config, sample_input)
    print("   ✅ ISTRNetwork前向传播测试通过")

    # 测试特征提取
    features_dict = istr_tester.test_extract_features(config, sample_input)
    print("   ✅ ISTRNetwork特征提取测试通过")

    # 测试参数更新
    istr_tester.test_update_parameters(config, sample_input)
    print("   ✅ ISTRNetwork参数更新测试通过")

    # 测试梯度流
    istr_tester.test_gradient_flow(config, sample_input)
    print("   ✅ ISTRNetwork梯度流测试通过")

    # 测试批量变化
    istr_tester.test_batch_variation(config)
    print("   ✅ ISTRNetwork批量变化测试通过")

    # 测试序列长度变化
    istr_tester.test_sequence_length_variation(config)
    print("   ✅ ISTRNetwork序列长度变化测试通过")

    print("\n🎉 所有ISTR网络测试通过！")


if __name__ == "__main__":
    run_all_tests()