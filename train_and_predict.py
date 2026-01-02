# train_and_predict.py - STAR-Forecast 完整端到端训练和预测

import sys
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🚀 STAR-Forecast 完整端到端训练和预测")
print("=" * 80)
print("🌟 整合: ISTR(神经) + AutoGen(符号) + Agent Lightning(强化) + 记忆银行")
print("=" * 80)


class STARForecastTrainer:
    """STAR-Forecast 完整训练和预测器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self.get_default_config()
        self.device = self.setup_device()
        self.initialize_components()
        self.setup_directories()

    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            # 数据配置
            'dataset': 'ETTh1',
            'data_path': 'data/raw/ETTh1.csv',
            'seq_len': 96,
            'pred_len': 24,
            'batch_size': 32,
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,

            # ISTR模型配置
            'input_dim': 7,
            'hidden_dim': 64,
            'num_blocks': 3,
            'trainable_ratio': 0.01,
            'laplacian_weight': 0.01,

            # 训练配置
            'epochs': 50,
            'learning_rate': 1e-3,
            'patience': 10,

            # AutoGen配置
            'use_autogen': True,
            'agent_count': 3,
            'debate_rounds': 2,

            # Agent Lightning配置
            'use_lightning': True,
            'reinforcement_epochs': 5,

            # 系统配置
            'save_checkpoints': True,
            'checkpoint_freq': 5,
            'log_interval': 10,
            'seed': 42
        }

    def setup_device(self):
        """设置计算设备"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("ℹ️  使用CPU")
        return device

    def setup_directories(self):
        """创建必要的目录"""
        directories = [
            "results",
            "results/checkpoints",
            "results/predictions",
            "results/logs",
            "results/memory"
        ]

        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        print("📁 目录结构已准备")

    def initialize_components(self):
        """初始化所有组件"""
        print("\n🔧 初始化框架组件...")

        # 1. 初始化ISTR模型
        print("1️⃣ 初始化ISTR神经网络...")
        try:
            from models.istr import ISTRPredictor

            self.istr_model = ISTRPredictor(
                input_dim=self.config['input_dim'],
                hidden_dim=self.config['hidden_dim'],
                pred_len=self.config['pred_len'],
                num_blocks=self.config['num_blocks'],
                trainable_ratio=self.config['trainable_ratio'],
                laplacian_weight=self.config['laplacian_weight']
            ).to(self.device)

            print(f"   ✅ ISTR模型初始化成功")
            print(f"      总参数: {self.istr_model.total_params:,}")
            print(f"      可训练参数: {self.istr_model.trainable_params:,}")

        except Exception as e:
            print(f"   ❌ ISTR模型初始化失败: {e}")
            raise

        # 2. 初始化AutoGen系统
        if self.config['use_autogen']:
            print("2️⃣ 初始化AutoGen多智能体系统...")
            try:
                from agents.autogen_system import AutoGenDebateSystem
                from dataclasses import dataclass

                @dataclass
                class DebateConfig:
                    agent_count: int = self.config['agent_count']
                    debate_rounds: int = self.config['debate_rounds']
                    temperature: float = 0.7
                    use_memory: bool = True

                self.autogen_system = AutoGenDebateSystem(config=DebateConfig())
                print(f"   ✅ AutoGen系统初始化成功")

            except Exception as e:
                print(f"   ⚠️  AutoGen系统初始化失败: {e}")
                self.config['use_autogen'] = False

        # 3. 初始化记忆银行
        print("3️⃣ 初始化记忆银行...")
        try:
            from agents.memory_bank import MemoryBank

            self.memory_bank = MemoryBank(
                persistence_path="results/memory/memory_bank.json",
                max_memory_items=1000
            )
            print(f"   ✅ 记忆银行初始化成功")

        except Exception as e:
            print(f"   ❌ 记忆银行初始化失败: {e}")
            raise

        # 4. 初始化Agent Lightning
        if self.config['use_lightning']:
            print("4️⃣ 初始化Agent Lightning强化学习...")
            try:
                from training.lightning_client import LightningTrainer

                self.lightning_trainer = LightningTrainer(
                    model=self.istr_model,
                    learning_rate=self.config['learning_rate'] * 0.1,  # 强化学习使用更小的学习率
                    batch_size=self.config['batch_size'],
                    enable_reinforcement=True
                )
                print(f"   ✅ Agent Lightning初始化成功")

            except Exception as e:
                print(f"   ⚠️  Agent Lightning初始化失败: {e}")
                self.config['use_lightning'] = False

        print("\n✅ 所有组件初始化完成!")

    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print("\n📊 加载和预处理数据...")

        # 加载CSV数据
        data_path = Path(self.config['data_path'])
        if not data_path.exists():
            print(f"❌ 数据文件不存在: {data_path}")
            # 尝试从ETTh1.csv加载
            data_path = Path("data/raw/ETTh1.csv")
            if not data_path.exists():
                raise FileNotFoundError(f"数据文件不存在: {data_path}")

        df = pd.read_csv(data_path)
        print(f"✅ 数据加载成功: {df.shape}")

        # 提取特征
        feature_columns = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
        if all(col in df.columns for col in feature_columns):
            features = df[feature_columns].values.astype(np.float32)
        else:
            # 使用前7列作为特征
            features = df.iloc[:, 1:8].values.astype(np.float32)

        print(f"   特征矩阵形状: {features.shape}")

        # 数据标准化
        self.data_mean = features.mean(axis=0)
        self.data_std = features.std(axis=0) + 1e-8
        features_normalized = (features - self.data_mean) / self.data_std

        # 创建序列
        seq_len = self.config['seq_len']
        pred_len = self.config['pred_len']

        X, y = self.create_sequences(features_normalized, seq_len, pred_len)
        print(f"   创建序列完成: X={X.shape}, y={y.shape}")

        # 划分数据集
        n_samples = len(X)
        train_size = int(n_samples * self.config['train_ratio'])
        val_size = int(n_samples * self.config['val_ratio'])

        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        print(f"   数据集划分:")
        print(f"     训练集: {len(X_train)} 样本")
        print(f"     验证集: {len(X_val)} 样本")
        print(f"     测试集: {len(X_test)} 样本")

        # 转换为PyTorch张量
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(-1).to(self.device)  # [batch, pred_len, 1]
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(-1).to(self.device)
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        y_test_t = torch.FloatTensor(y_test).unsqueeze(-1).to(self.device)

        return {
            'train': (X_train_t, y_train_t),
            'val': (X_val_t, y_val_t),
            'test': (X_test_t, y_test_t),
            'mean': self.data_mean,
            'std': self.data_std,
            'original': features
        }

    def create_sequences(self, data: np.ndarray, seq_len: int, pred_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """创建时间序列样本"""
        n_samples = len(data) - seq_len - pred_len
        X, y = [], []

        for i in range(n_samples):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len:i + seq_len + pred_len, -1])  # 只预测OT列

        return np.array(X), np.array(y)

    def train_istr_model(self, train_data, val_data):
        """训练ISTR模型"""
        print("\n🎯 开始训练ISTR模型...")

        X_train, y_train = train_data
        X_val, y_val = val_data

        # 创建数据加载器
        from torch.utils.data import TensorDataset, DataLoader

        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )

        # 定义损失函数和优化器
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.istr_model.parameters()),
            lr=self.config['learning_rate']
        )

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0

        train_losses = []
        val_losses = []

        for epoch in range(self.config['epochs']):
            # 训练阶段
            self.istr_model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()

                # 前向传播
                predictions = self.istr_model(batch_X)

                # 计算损失
                mse_loss = criterion(predictions, batch_y)
                reg_loss = self.istr_model.compute_regularization_loss(predictions)
                loss = mse_loss + reg_loss

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.istr_model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # 验证阶段
            self.istr_model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    predictions = self.istr_model(batch_X)
                    mse_loss = criterion(predictions, batch_y)
                    reg_loss = self.istr_model.compute_regularization_loss(predictions)
                    loss = mse_loss + reg_loss
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # 学习率调度
            scheduler.step(avg_val_loss)

            # 打印进度
            if (epoch + 1) % self.config['log_interval'] == 0:
                print(f"   Epoch {epoch + 1}/{self.config['epochs']}: "
                      f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0

                if self.config['save_checkpoints']:
                    self.save_checkpoint(epoch, avg_val_loss, "best")
                    print(f"   💾 保存最佳模型 (Val Loss: {avg_val_loss:.6f})")
            else:
                patience_counter += 1

            # 定期保存检查点
            if self.config['save_checkpoints'] and (epoch + 1) % self.config['checkpoint_freq'] == 0:
                self.save_checkpoint(epoch, avg_val_loss, f"epoch_{epoch + 1}")

            # 早停
            if patience_counter >= self.config['patience']:
                print(f"   ⏹️  早停触发于第 {epoch + 1} 轮")
                break

        # 保存训练历史
        self.save_training_history(train_losses, val_losses)

        print(f"\n✅ 训练完成! 最佳验证损失: {best_val_loss:.6f}")

        return best_val_loss

    def save_checkpoint(self, epoch: int, val_loss: float, name: str):
        """保存检查点"""
        checkpoint_dir = Path("results/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"istr_{name}.pt"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.istr_model.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'data_stats': {
                'mean': self.data_mean.tolist(),
                'std': self.data_std.tolist()
            }
        }, checkpoint_path)

    def save_training_history(self, train_losses: List[float], val_losses: List[float]):
        """保存训练历史"""
        history_dir = Path("results/logs")
        history_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_path = history_dir / f"training_history_{timestamp}.json"

        history = {
            'timestamp': timestamp,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'config': self.config
        }

        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

    def run_autogen_debate(self, historical_data: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
        """运行AutoGen辩论"""
        if not self.config['use_autogen']:
            return None

        print("\n🤖 启动AutoGen多智能体辩论...")

        try:
            # 准备辩论上下文
            context = {
                "data_description": f"{self.config['dataset']} 时间序列数据",
                "historical_stats": {
                    "mean": float(historical_data.mean()),
                    "std": float(historical_data.std()),
                    "min": float(historical_data.min()),
                    "max": float(historical_data.max())
                },
                "prediction_stats": {
                    "mean": float(predictions.mean()),
                    "std": float(predictions.std()),
                    "min": float(predictions.min()),
                    "max": float(predictions.max())
                },
                "model_info": {
                    "type": "ISTR",
                    "trainable_params": self.istr_model.trainable_params,
                    "trainable_ratio": self.config['trainable_ratio']
                }
            }

            # 启动辩论
            debate_result = self.autogen_system.start_debate(
                topic="时间序列预测分析与优化",
                context=context,
                question="当前的预测结果是否合理？有哪些改进建议？"
            )

            print(f"   ✅ 辩论完成")
            print(f"     共识: {debate_result.consensus[:100]}...")
            print(f"     建议数量: {len(debate_result.recommendations)}")

            # 存储到记忆银行
            if hasattr(self, 'memory_bank'):
                self.memory_bank.store_experience({
                    "type": "debate",
                    "timestamp": datetime.now().isoformat(),
                    "topic": "预测优化",
                    "consensus": debate_result.consensus,
                    "recommendations": debate_result.recommendations,
                    "context": context
                })

            return {
                'consensus': debate_result.consensus,
                'recommendations': debate_result.recommendations,
                'insights': debate_result.get_consensus_insights() if hasattr(debate_result,
                                                                              'get_consensus_insights') else {}
            }

        except Exception as e:
            print(f"   ⚠️  辩论失败: {e}")
            return None

    def apply_debate_insights(self, predictions: np.ndarray, debate_result: Dict[str, Any]) -> np.ndarray:
        """应用辩论见解优化预测"""
        if not debate_result or 'insights' not in debate_result:
            return predictions

        insights = debate_result['insights']
        optimized = predictions.copy()

        print("\n🔧 应用辩论见解优化预测...")

        # 应用趋势调整
        if "adjust_trend" in insights:
            adjustment = insights["adjust_trend"]
            optimized = optimized * (1 + adjustment)
            print(f"   应用趋势调整: {adjustment * 100:.1f}%")

        # 应用平滑
        if "smooth_variance" in insights and insights["smooth_variance"]:
            try:
                from scipy.ndimage import gaussian_filter1d
                optimized = gaussian_filter1d(optimized, sigma=1)
                print(f"   应用方差平滑")
            except:
                pass

        return optimized

    def run_reinforcement_learning(self, experiences: List[Dict[str, Any]]) -> float:
        """运行强化学习"""
        if not self.config['use_lightning'] or not experiences:
            return 0.0

        print("\n⚡ 启动Agent Lightning强化学习...")

        try:
            if len(experiences) < 10:
                print(f"   ⚠️  经验数据不足 ({len(experiences)}个)，跳过强化学习")
                return 0.0

            improvement = self.lightning_trainer.reinforce(
                experiences=experiences,
                target_metric="mse",
                n_epochs=self.config['reinforcement_epochs']
            )

            print(f"   ✅ 强化学习完成，改进: {improvement:.6f}")

            # 存储到记忆银行
            if hasattr(self, 'memory_bank'):
                self.memory_bank.store_experience({
                    "type": "reinforcement",
                    "timestamp": datetime.now().isoformat(),
                    "improvement": improvement,
                    "experience_count": len(experiences),
                    "epochs": self.config['reinforcement_epochs']
                })

            return improvement

        except Exception as e:
            print(f"   ⚠️  强化学习失败: {e}")
            return 0.0

    def evaluate_predictions(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """评估预测结果"""
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)

        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse)
        }

    def run_prediction_pipeline(self, test_data, n_samples: int = 5):
        """运行完整预测管道"""
        print("\n" + "=" * 80)
        print("🔮 运行完整预测管道")
        print("=" * 80)

        X_test, y_test = test_data

        # 限制测试样本数量
        n_samples = min(n_samples, len(X_test))
        print(f"   处理 {n_samples} 个测试样本")

        all_results = []
        valuable_experiences = []

        for i in range(n_samples):
            print(f"\n📊 样本 {i + 1}/{n_samples}")
            print("-" * 40)

            # 1. ISTR基础预测
            print("1️⃣ ISTR基础预测...")
            sample_input = X_test[i:i + 1].cpu().numpy()  # [1, seq_len, features]
            sample_target = y_test[i:i + 1].cpu().numpy().squeeze()  # [pred_len]

            # 反标准化输入用于辩论上下文
            historical_data_denorm = sample_input.squeeze() * self.data_std + self.data_mean

            # 模型预测
            self.istr_model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(sample_input).to(self.device)
                prediction_tensor = self.istr_model(input_tensor)
                base_prediction = prediction_tensor.cpu().numpy().squeeze()  # [pred_len]

            # 反标准化预测和目标
            base_pred_denorm = base_prediction * self.data_std[-1] + self.data_mean[-1]
            target_denorm = sample_target * self.data_std[-1] + self.data_mean[-1]

            # 评估基础预测
            base_metrics = self.evaluate_predictions(base_pred_denorm, target_denorm)
            print(f"   基础预测MSE: {base_metrics['mse']:.6f}")

            # 2. AutoGen辩论优化
            print("2️⃣ AutoGen辩论优化...")
            debate_result = self.run_autogen_debate(
                historical_data_denorm[:, -1],  # 只使用OT列的历史数据
                base_pred_denorm
            )

            # 3. 应用辩论见解
            print("3️⃣ 应用辩论见解...")
            optimized_prediction = self.apply_debate_insights(base_pred_denorm, debate_result)

            # 评估优化后预测
            optimized_metrics = self.evaluate_predictions(optimized_prediction, target_denorm)
            improvement = base_metrics['mse'] - optimized_metrics['mse']

            print(f"   优化后预测MSE: {optimized_metrics['mse']:.6f}")
            print(f"   改进: {improvement:.6f} ({improvement / base_metrics['mse'] * 100:.1f}%)")

            # 4. 收集经验用于强化学习
            if improvement > 0:  # 只有改进的经验才收集
                experience = {
                    "state": sample_input.squeeze(),  # [seq_len, features]
                    "action": base_prediction,  # 基础预测
                    "reward": -optimized_metrics['mse'],  # 负MSE作为奖励
                    "next_state": sample_input.squeeze(),  # 简化，使用相同状态
                    "improvement": improvement
                }
                valuable_experiences.append(experience)

            # 5. 存储结果
            sample_result = {
                "sample_id": i,
                "base_prediction": base_pred_denorm.tolist(),
                "optimized_prediction": optimized_prediction.tolist(),
                "true_values": target_denorm.tolist(),
                "base_metrics": base_metrics,
                "optimized_metrics": optimized_metrics,
                "improvement": float(improvement),
                "debate_consensus": debate_result['consensus'] if debate_result else None,
                "recommendations": debate_result['recommendations'] if debate_result else []
            }

            all_results.append(sample_result)

            # 存储到记忆银行
            if hasattr(self, 'memory_bank'):
                self.memory_bank.store_experience({
                    "type": "prediction",
                    "timestamp": datetime.now().isoformat(),
                    "sample_id": i,
                    "base_mse": base_metrics['mse'],
                    "optimized_mse": optimized_metrics['mse'],
                    "improvement": improvement,
                    "debate_used": debate_result is not None
                })

        # 6. Agent Lightning强化学习
        print(f"\n" + "=" * 80)
        print("⚡ 阶段4: Agent Lightning强化学习")
        print("=" * 80)

        rl_improvement = self.run_reinforcement_learning(valuable_experiences)

        # 7. 汇总结果
        print(f"\n" + "=" * 80)
        print("📊 最终结果汇总")
        print("=" * 80)

        if all_results:
            base_mses = [r["base_metrics"]["mse"] for r in all_results]
            optimized_mses = [r["optimized_metrics"]["mse"] for r in all_results]

            avg_base_mse = np.mean(base_mses)
            avg_optimized_mse = np.mean(optimized_mses)
            avg_improvement = avg_base_mse - avg_optimized_mse

            print(f"📈 性能统计:")
            print(f"   平均基础MSE: {avg_base_mse:.6f}")
            print(f"   平均优化MSE: {avg_optimized_mse:.6f}")
            print(f"   平均改进: {avg_improvement:.6f} ({avg_improvement / avg_base_mse * 100:.1f}%)")
            print(f"   强化学习改进: {rl_improvement:.6f}")

            print(f"\n🔧 组件使用情况:")
            print(f"   ISTR模型: ✅ 已使用")
            print(f"   AutoGen系统: {'✅' if self.config['use_autogen'] else '❌'}")
            print(f"   Agent Lightning: {'✅' if self.config['use_lightning'] else '❌'}")
            print(f"   记忆银行: ✅ 已使用 ({len(self.memory_bank) if hasattr(self, 'memory_bank') else 0} 条记忆)")

            # 保存预测结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = Path("results/predictions")
            results_file = results_dir / f"prediction_results_{timestamp}.json"

            final_results = {
                "timestamp": timestamp,
                "dataset": self.config['dataset'],
                "samples_processed": n_samples,
                "average_base_mse": float(avg_base_mse),
                "average_optimized_mse": float(avg_optimized_mse),
                "average_improvement": float(avg_improvement),
                "rl_improvement": float(rl_improvement),
                "config": self.config,
                "sample_results": all_results
            }

            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)

            print(f"\n💾 详细结果保存到: {results_file}")

            # 保存记忆银行
            if hasattr(self, 'memory_bank'):
                memory_file = results_dir / f"memory_bank_{timestamp}.json"
                self.memory_bank.save(memory_file)
                print(f"💾 记忆银行保存到: {memory_file}")

            return final_results

        return None

    def run(self, train_model: bool = True, test_samples: int = 5):
        """运行完整训练和预测流程"""
        try:
            # 1. 加载数据
            print("\n" + "=" * 80)
            print("📊 步骤1: 数据加载和预处理")
            print("=" * 80)
            data_dict = self.load_and_preprocess_data()

            # 2. 训练模型
            if train_model:
                print("\n" + "=" * 80)
                print("🎯 步骤2: 训练ISTR模型")
                print("=" * 80)
                self.train_istr_model(data_dict['train'], data_dict['val'])

            # 3. 加载最佳模型
            print("\n" + "=" * 80)
            print("📂 步骤3: 加载最佳模型")
            print("=" * 80)
            self.load_best_checkpoint()

            # 4. 运行预测管道
            print("\n" + "=" * 80)
            print("🔮 步骤4: 完整预测管道")
            print("=" * 80)
            results = self.run_prediction_pipeline(data_dict['test'], test_samples)

            print("\n" + "=" * 80)
            print("🎉 STAR-Forecast 完整流程执行成功!")
            print("=" * 80)

            return results

        except Exception as e:
            print(f"\n❌ 执行失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_best_checkpoint(self):
        """加载最佳检查点"""
        checkpoint_dir = Path("results/checkpoints")
        if not checkpoint_dir.exists():
            print("   ℹ️  没有检查点，使用初始模型")
            return

        checkpoint_files = list(checkpoint_dir.glob("istr_best*.pt"))
        if not checkpoint_files:
            checkpoint_files = list(checkpoint_dir.glob("*.pt"))

        if checkpoint_files:
            # 按修改时间排序，取最新的
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            checkpoint_path = checkpoint_files[0]

            print(f"   📂 加载检查点: {checkpoint_path.name}")

            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.istr_model.load_state_dict(checkpoint['model_state_dict'])

                # 更新数据统计
                if 'data_stats' in checkpoint:
                    self.data_mean = np.array(checkpoint['data_stats']['mean'])
                    self.data_std = np.array(checkpoint['data_stats']['std'])

                print(f"   ✅ 检查点加载成功 (Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f})")

            except Exception as e:
                print(f"   ⚠️  检查点加载失败: {e}")
        else:
            print("   ℹ️  没有找到检查点")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='STAR-Forecast 完整训练和预测')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['train', 'predict', 'full'],
                        help='运行模式: train=仅训练, predict=仅预测, full=完整流程')
    parser.add_argument('--dataset', type=str, default='ETTh1',
                        help='数据集名称')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--samples', type=int, default=5,
                        help='测试样本数量')
    parser.add_argument('--no-autogen', action='store_true',
                        help='禁用AutoGen系统')
    parser.add_argument('--no-lightning', action='store_true',
                        help='禁用Agent Lightning')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径')

    args = parser.parse_args()

    print(f"🚀 STAR-Forecast 启动")
    print(f"   模式: {args.mode}")
    print(f"   数据集: {args.dataset}")
    print(f"   样本数: {args.samples}")
    print(f"   AutoGen: {'禁用' if args.no_autogen else '启用'}")
    print(f"   Agent Lightning: {'禁用' if args.no_lightning else '启用'}")

    # 创建配置
    config = {
        'dataset': args.dataset,
        'data_path': f"data/raw/{args.dataset}.csv",
        'epochs': args.epochs,
        'use_autogen': not args.no_autogen,
        'use_lightning': not args.no_lightning
    }

    # 创建训练器
    trainer = STARForecastTrainer(config)

    # 根据模式运行
    if args.mode == 'train':
        # 仅训练
        data_dict = trainer.load_and_preprocess_data()
        trainer.train_istr_model(data_dict['train'], data_dict['val'])

    elif args.mode == 'predict':
        # 仅预测（需要已有训练好的模型）
        data_dict = trainer.load_and_preprocess_data()
        trainer.load_best_checkpoint()
        trainer.run_prediction_pipeline(data_dict['test'], args.samples)

    else:  # full
        # 完整流程
        trainer.run(train_model=True, test_samples=args.samples)

    print("\n✅ 程序执行完成!")


if __name__ == "__main__":
    main()