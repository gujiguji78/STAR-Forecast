# main.py - 完整的多智能体强化学习闭环
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# 导入AutoGen多智能体系统
from agents.autogen_system import AutoGenDebateSystem, DebateConfig
from agents.memory_bank import MemoryBank
from agents.debate_system import DebateOrchestrator

# 导入Agent Lightning强化学习
from training.lightning_client import LightningTrainer
from models.istr import ISTRPredictor
from models.ensemble import EnsemblePredictor

# 导入数据处理
from data.dataloader import TimeSeriesDataLoader
from data.processor import DataProcessor


class STARForecastSystem:
    """STAR-Forecast: 智能体强化预测系统"""

    def __init__(self, config_path: str = None):
        # 1. 初始化配置
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._default_config()

        # 2. 初始化记忆银行
        self.memory_bank = MemoryBank(
            persistence_path="results/memory_store.json",
            max_memory_items=1000
        )

        # 3. 初始化多智能体辩论系统
        debate_config = DebateConfig(
            agent_count=3,  # 3个专家智能体
            debate_rounds=2,
            temperature=0.7,
            use_memory=True
        )
        self.debate_system = AutoGenDebateSystem(
            config=debate_config,
            memory_bank=self.memory_bank
        )

        # 4. 初始化深度学习预测器
        self.predictor = ISTRPredictor.load_from_checkpoint(
            "results/train_ETTh1_96to24_20260101_230210/checkpoint.ckpt"
        )

        # 5. 初始化Agent Lightning强化学习器
        self.lightning_trainer = LightningTrainer(
            model=self.predictor,
            memory_bank=self.memory_bank,
            learning_rate=1e-4,
            batch_size=32
        )

        # 6. 初始化数据处理器
        self.data_processor = DataProcessor()

    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            "prediction_horizon": 24,
            "lookback_window": 96,
            "debate_enabled": True,
            "reinforcement_enabled": True,
            "ensemble_method": "weighted_average"
        }

    def forecast_with_debate(self, historical_data: np.ndarray) -> Dict:
        """
        多智能体辩论预测流程
        """
        print("🚀 开始多智能体辩论预测...")

        # 步骤1: 基础深度学习预测
        base_prediction = self.predictor.predict(historical_data)

        if not self.config["debate_enabled"]:
            return {"prediction": base_prediction, "debate_log": None}

        # 步骤2: 启动多智能体辩论
        debate_context = {
            "historical_data": historical_data.tolist(),
            "base_prediction": base_prediction.tolist(),
            "confidence_scores": self.predictor.get_confidence_scores(historical_data)
        }

        debate_result = self.debate_system.start_debate(
            topic="时间序列预测优化",
            context=debate_context,
            question="如何改进当前预测结果？"
        )

        # 步骤3: 解析辩论结果并修正预测
        refined_prediction = self._apply_debate_insights(
            base_prediction,
            debate_result
        )

        # 步骤4: 存储到记忆银行
        self.memory_bank.store_experience({
            "timestamp": pd.Timestamp.now(),
            "historical_data": historical_data,
            "base_prediction": base_prediction,
            "debate_result": debate_result,
            "refined_prediction": refined_prediction
        })

        return {
            "base_prediction": base_prediction,
            "refined_prediction": refined_prediction,
            "debate_log": debate_result.debate_log,
            "confidence": self._calculate_confidence(refined_prediction)
        }

    def _apply_debate_insights(self, base_pred: np.ndarray, debate_result) -> np.ndarray:
        """应用智能体辩论的见解"""
        insights = debate_result.get_consensus_insights()

        # 根据辩论结果调整预测
        if "adjust_trend" in insights:
            trend_adjustment = insights["adjust_trend"]
            base_pred = base_pred * (1 + trend_adjustment)

        if "smooth_variance" in insights and insights["smooth_variance"]:
            # 应用平滑
            from scipy.ndimage import gaussian_filter1d
            base_pred = gaussian_filter1d(base_pred, sigma=1)

        return base_pred

    def reinforcement_training_loop(self, validation_data: Dict):
        """
        Agent Lightning强化训练闭环
        """
        print("⚡ 启动Agent Lightning强化训练...")

        # 步骤1: 在验证集上评估当前表现
        current_performance = self._evaluate_on_validation(validation_data)

        # 步骤2: 从记忆银行获取高质量经验
        valuable_experiences = self.memory_bank.retrieve_relevant_experiences(
            query="high_confidence_predictions",
            top_k=50
        )

        # 步骤3: 执行强化学习
        if valuable_experiences and self.config["reinforcement_enabled"]:
            improvement = self.lightning_trainer.reinforce(
                experiences=valuable_experiences,
                target_metric="mse",  # 目标是最小化均方误差
                n_epochs=10
            )

            print(f"📈 强化学习提升: {improvement:.4f}")

            # 步骤4: 评估提升效果
            new_performance = self._evaluate_on_validation(validation_data)

            return {
                "old_performance": current_performance,
                "new_performance": new_performance,
                "improvement": new_performance["mse"] - current_performance["mse"],
                "training_samples": len(valuable_experiences)
            }

        return {"status": "no_training_performed"}

    def _evaluate_on_validation(self, data: Dict) -> Dict:
        """在验证集上评估模型"""
        predictions = []
        truths = []

        for batch in data["loader"]:
            pred = self.predictor.predict(batch["x"])
            predictions.append(pred)
            truths.append(batch["y"])

        predictions = np.concatenate(predictions)
        truths = np.concatenate(truths)

        # 计算指标
        mse = np.mean((predictions - truths) ** 2)
        mae = np.mean(np.abs(predictions - truths))

        return {"mse": mse, "mae": mae}

    def _calculate_confidence(self, prediction: np.ndarray) -> float:
        """计算预测置信度"""
        # 使用预测的方差作为置信度指标
        variance = np.var(prediction)
        confidence = 1.0 / (1.0 + variance)
        return float(confidence)

    def run_full_pipeline(self, train_data, val_data, test_data):
        """
        运行完整预测管道
        """
        results = []

        print("=" * 50)
        print("🔮 STAR-Forecast 智能预测系统启动")
        print("=" * 50)

        # 阶段1: 多智能体辩论预测
        print("\n📊 阶段1: 多智能体辩论预测")
        for i, test_sample in enumerate(test_data[:10]):  # 测试前10个样本
            result = self.forecast_with_debate(test_sample)
            results.append(result)
            print(f"样本 {i + 1}: 置信度 {result['confidence']:.3f}")

        # 阶段2: Agent Lightning强化学习
        print("\n⚡ 阶段2: Agent Lightning强化学习")
        training_result = self.reinforcement_training_loop(val_data)
        print(f"训练结果: {training_result}")

        # 阶段3: 集成预测
        print("\n🤝 阶段3: 智能体集成预测")
        ensemble_result = self._ensemble_predictions(results)

        # 阶段4: 生成报告
        print("\n📈 阶段4: 生成预测报告")
        report = self._generate_report(results, ensemble_result, training_result)

        return report

    def _ensemble_predictions(self, predictions_list: List[Dict]) -> Dict:
        """集成多个智能体的预测"""
        ensemble_predictor = EnsemblePredictor(
            methods=[self.config["ensemble_method"]],
            weights=[0.4, 0.3, 0.3]  # 可调整权重
        )

        all_predictions = [r["refined_prediction"] for r in predictions_list]
        ensemble_pred = ensemble_predictor.ensemble(all_predictions)

        return {
            "ensemble_prediction": ensemble_pred,
            "variance": np.var(all_predictions, axis=0),
            "agreement_score": self._calculate_agreement(all_predictions)
        }

    def _calculate_agreement(self, predictions: List[np.ndarray]) -> float:
        """计算智能体间的一致性"""
        if len(predictions) < 2:
            return 1.0

        # 计算两两之间的相关系数
        corrs = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                corr = np.corrcoef(predictions[i].flatten(), predictions[j].flatten())[0, 1]
                corrs.append(corr)

        return float(np.mean(corrs))

    def _generate_report(self, results, ensemble_result, training_result) -> Dict:
        """生成完整报告"""
        confidences = [r["confidence"] for r in results]

        return {
            "timestamp": pd.Timestamp.now().isoformat(),
            "total_samples": len(results),
            "average_confidence": np.mean(confidences),
            "ensemble_prediction": ensemble_result["ensemble_prediction"].tolist(),
            "agent_agreement": ensemble_result["agreement_score"],
            "reinforcement_improvement": training_result.get("improvement", 0),
            "memory_bank_size": len(self.memory_bank),
            "config": self.config
        }


# 客户端使用示例
def main():
    """主函数示例"""
    import argparse

    parser = argparse.ArgumentParser(description='STAR-Forecast 智能预测系统')
    parser.add_argument('--config', type=str, default='configs/default_config.json',
                        help='配置文件路径')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['debate', 'reinforce', 'full'],
                        help='运行模式')

    args = parser.parse_args()

    # 1. 初始化系统
    system = STARForecastSystem(config_path=args.config)

    # 2. 加载数据
    print("📂 加载数据...")
    data_loader = TimeSeriesDataLoader(
        data_path="data/raw",
        lookback=system.config["lookback_window"],
        horizon=system.config["prediction_horizon"]
    )

    train_data, val_data, test_data = data_loader.load_split_data(
        split_ratio=[0.7, 0.2, 0.1]
    )

    # 3. 根据模式运行
    if args.mode == 'debate':
        # 仅运行多智能体辩论
        test_sample = test_data[0]
        result = system.forecast_with_debate(test_sample)
        print("辩论预测结果:", result)

    elif args.mode == 'reinforce':
        # 仅运行强化学习
        result = system.reinforcement_training_loop(val_data)
        print("强化学习结果:", result)

    else:  # full
        # 运行完整管道
        report = system.run_full_pipeline(train_data, val_data, test_data)

        # 保存结果
        output_dir = Path("results") / f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "forecast_report.json", 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✅ 完成！结果保存至: {output_dir}")
        print(f"📊 平均置信度: {report['average_confidence']:.3f}")
        print(f"🤝 智能体一致性: {report['agent_agreement']:.3f}")

        # 可视化结果（可选）
        system._visualize_results(report, output_dir)


# 快速测试脚本
if __name__ == "__main__":
    # 简单测试多智能体系统
    print("🧪 测试多智能体辩论系统...")

    # 创建测试数据
    test_historical = np.random.randn(96, 7)  # 96时间步，7个特征

    # 初始化简化版系统
    from agents.autogen_system import AutoGenDebateSystem
    from agents.memory_bank import MemoryBank

    memory = MemoryBank()
    debate_system = AutoGenDebateSystem(
        agent_count=2,
        debate_rounds=1,
        memory_bank=memory
    )

    # 运行一次辩论
    context = {
        "data_description": "测试时间序列数据",
        "current_prediction": [1.2, 1.3, 1.4]
    }

    result = debate_system.start_debate(
        topic="测试预测辩论",
        context=context,
        question="这个预测合理吗？"
    )

    print(f"辩论完成！共识: {result.consensus}")
    print(f"建议: {result.recommendations}")

    # 如果需要运行完整系统
    # main()