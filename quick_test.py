# quick_test.py - 快速测试完整框架

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🧪 STAR-Forecast 快速测试")
print("=" * 50)

# 测试所有组件
print("1. 测试组件导入...")

try:
    from models.istr import ISTRPredictor

    print("✅ ISTRPredictor: 可用")

    # 创建模型
    model = ISTRPredictor(
        input_dim=7,
        hidden_dim=64,
        pred_len=24,
        num_blocks=3,
        trainable_ratio=0.01,
        laplacian_weight=0.01
    )
    print(f"✅ 模型创建成功")
    print(f"   总参数: {model.total_params:,}")
    print(f"   可训练参数: {model.trainable_params:,}")

    # 测试预测
    import torch
    import numpy as np

    test_input = torch.randn(1, 96, 7)
    with torch.no_grad():
        output = model(test_input)
        print(f"✅ 前向传播测试通过")
        print(f"   输入形状: {test_input.shape}")
        print(f"   输出形状: {output.shape}")

except Exception as e:
    print(f"❌ ISTR测试失败: {e}")
    import traceback

    traceback.print_exc()

print("\n2. 测试AutoGen系统...")
try:
    from agents.autogen_system import AutoGenDebateSystem
    from dataclasses import dataclass


    @dataclass
    class DebateConfig:
        agent_count: int = 3
        debate_rounds: int = 2
        temperature: float = 0.7
        use_memory: bool = True


    debate_system = AutoGenDebateSystem(config=DebateConfig())
    print("✅ AutoGen系统初始化成功")

    # 测试辩论
    context = {
        "data_description": "测试数据",
        "prediction": [1.0, 1.1, 1.2]
    }
    result = debate_system.start_debate(
        topic="测试",
        context=context,
        question="如何改进？"
    )
    print(f"✅ 辩论测试通过")
    print(f"   共识: {result.consensus[:50]}...")

except Exception as e:
    print(f"⚠️  AutoGen测试失败: {e}")

print("\n3. 测试记忆银行...")
try:
    from agents.memory_bank import MemoryBank

    memory = MemoryBank()
    memory.store_experience({
        "test": "测试记忆",
        "timestamp": "2024-01-01"
    })
    print(f"✅ 记忆银行测试通过")
    print(f"   记忆数量: {len(memory)}")

except Exception as e:
    print(f"❌ 记忆银行测试失败: {e}")

print("\n4. 测试Agent Lightning...")
try:
    from training.lightning_client import LightningTrainer

    if 'model' in locals():
        trainer = LightningTrainer(model=model, learning_rate=1e-4)
        print("✅ Agent Lightning初始化成功")
    else:
        print("⚠️  跳过Agent Lightning测试（需要模型）")

except Exception as e:
    print(f"⚠️  Agent Lightning测试失败: {e}")

print("\n5. 测试数据加载...")
try:
    import pandas as pd

    data_path = "data/raw/ETTh1.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"✅ 数据加载成功")
        print(f"   形状: {df.shape}")
        print(f"   列名: {df.columns.tolist()[:5]}...")
    else:
        print(f"⚠️  数据文件不存在: {data_path}")

except Exception as e:
    print(f"⚠️  数据加载失败: {e}")

print("\n" + "=" * 50)
print("🎉 快速测试完成!")
print("=" * 50)
print("\n📋 下一步:")
print("1. 运行完整训练和预测: python train_and_predict.py")
print("2. 仅训练模型: python train_and_predict.py --mode train")
print("3. 仅使用训练好的模型预测: python train_and_predict.py --mode predict")
print("\n📁 配置文件: 可以创建config.yaml来自定义参数")