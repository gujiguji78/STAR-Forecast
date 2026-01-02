"""
智能体系统测试 - 真实可运行的测试代码
测试AutoGen三智能体协同
"""
import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from agents.autogen_system import (
    AutoGenController,
    DeepSeekReasonerAnalyst,
    QwenMaxArchitect,
    DeepSeekChatOptimizer,
    AnalystResult,
    ArchitectDecision,
    OptimizerDecision
)


class TestAnalystResult:
    """测试分析师结果数据结构"""

    def test_initialization(self):
        """测试初始化"""
        result = AnalystResult(
            pattern="non_stationary",
            frequencies=[0.1, 0.2, 0.3],
            hurst=0.65,
            anomaly=0.15,
            recommendations=["调整谱门控", "增加正则化"],
            reasoning=["数据呈现非平稳特征", "建议增加正则化约束"],
            confidence=0.8
        )

        assert result.pattern == "non_stationary"
        assert len(result.frequencies) == 3
        assert 0.6 <= result.hurst <= 0.7
        assert result.confidence == 0.8
        assert len(result.recommendations) == 2


class TestArchitectDecision:
    """测试架构师决策数据结构"""

    def test_initialization(self):
        """测试初始化"""
        decision = ArchitectDecision(
            spectral_threshold=0.6,
            laplacian_weight=0.015,
            learning_rate_multiplier=1.2,
            rationale="基于非平稳数据特性调整",
            confidence=0.75
        )

        assert 0.1 <= decision.spectral_threshold <= 0.9
        assert 0.001 <= decision.laplacian_weight <= 0.1
        assert 0.1 <= decision.learning_rate_multiplier <= 5.0
        assert len(decision.rationale) > 0


class TestOptimizerDecision:
    """测试优化师决策数据结构"""

    def test_initialization(self):
        """测试初始化"""
        decision = OptimizerDecision(
            apply_changes=True,
            parameters={
                'spectral_threshold': 0.6,
                'laplacian_weight': 0.015,
                'learning_rate_multiplier': 1.2
            },
            steps=["更新谱门控阈值", "调整拉普拉斯权重"],
            risk="medium",
            expected_improvement={"mse": 0.05, "mae": 0.03}
        )

        assert decision.apply_changes is True
        assert 'spectral_threshold' in decision.parameters
        assert len(decision.steps) > 0
        assert decision.risk in ['low', 'medium', 'high']


class TestMockAnalyst:
    """模拟分析师测试"""

    @pytest.fixture
    def mock_api_key(self):
        """模拟API密钥"""
        return "test-api-key-123456"

    @pytest.fixture
    def analyst(self, mock_api_key):
        """创建模拟分析师"""
        with patch('agents.autogen_system.requests') as mock_requests:
            # 模拟成功的API响应
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'choices': [{
                    'message': {
                        'content': json.dumps({
                            'pattern': 'non_stationary',
                            'frequencies': [0.1, 0.2],
                            'hurst': 0.65,
                            'anomaly': 0.12,
                            'recommendations': ['调整谱门控阈值'],
                            'reasoning': ['数据呈现趋势特征'],
                            'confidence': 0.78
                        })
                    }
                }]
            }
            mock_requests.get.return_value = mock_response
            mock_requests.post.return_value = mock_response

            return DeepSeekReasonerAnalyst(mock_api_key)

    def test_analyze(self, analyst):
        """测试分析功能"""
        context = {
            'features': {
                'shape': [32, 96, 64],
                'statistics': {
                    'mean': [0.1, 0.2, 0.3],
                    'std': [0.05, 0.06, 0.07]
                },
                'frequency': {
                    'dominant': 3
                }
            },
            'metrics': {
                'mse': 0.25,
                'mae': 0.35
            },
            'current_params': {
                'spectral_threshold': 0.5,
                'laplacian_weight': 0.01
            }
        }

        result = analyst.analyze(context)

        assert isinstance(result, AnalystResult)
        assert result.pattern in ['stationary', 'non_stationary', 'regime_shift']
        assert 0 <= result.hurst <= 1
        assert 0 <= result.anomaly <= 1
        assert result.confidence >= 0


class TestMockArchitect:
    """模拟架构师测试"""

    @pytest.fixture
    def mock_api_key(self):
        """模拟API密钥"""
        return "test-qwen-api-key"

    @pytest.fixture
    def architect(self, mock_api_key):
        """创建模拟架构师"""
        # 模拟dashscope模块
        mock_dashscope = Mock()

        # 模拟成功响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.output = {'text': json.dumps({
            'spectral_threshold': 0.65,
            'laplacian_weight': 0.018,
            'learning_rate_multiplier': 1.3,
            'rationale': '基于非平稳特性调整',
            'confidence': 0.72
        })}

        mock_dashscope.Generation.call.return_value = mock_response

        with patch.dict('sys.modules', {'dashscope': mock_dashscope}):
            return QwenMaxArchitect(mock_api_key)

    @pytest.fixture
    def analysis_result(self):
        """创建模拟分析结果"""
        return AnalystResult(
            pattern="non_stationary",
            frequencies=[0.1, 0.2],
            hurst=0.65,
            anomaly=0.12,
            recommendations=["增加谱门控阈值"],
            reasoning=["数据有趋势"],
            confidence=0.75
        )

    def test_design(self, architect, analysis_result):
        """测试设计功能"""
        decision = architect.design(analysis_result)

        assert isinstance(decision, ArchitectDecision)
        assert 0.1 <= decision.spectral_threshold <= 0.9
        assert 0.001 <= decision.laplacian_weight <= 0.1
        assert 0.1 <= decision.learning_rate_multiplier <= 5.0


class TestMockOptimizer:
    """模拟优化师测试"""

    @pytest.fixture
    def mock_api_key(self):
        """模拟API密钥"""
        return "test-deepseek-chat-key"

    @pytest.fixture
    def optimizer(self, mock_api_key):
        """创建模拟优化师"""
        with patch('agents.autogen_system.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                'choices': [{
                    'message': {
                        'content': json.dumps({
                            'apply_changes': True,
                            'parameters': {
                                'spectral_threshold': 0.65,
                                'laplacian_weight': 0.018
                            },
                            'steps': ['验证参数范围', '应用调整'],
                            'risk': 'medium',
                            'expected_improvement': {'mse': 0.04, 'mae': 0.025}
                        })
                    }
                }]
            }
            mock_requests.post.return_value = mock_response

            return DeepSeekChatOptimizer(mock_api_key)

    @pytest.fixture
    def analysis_result(self):
        """模拟分析结果"""
        return AnalystResult(
            pattern="non_stationary",
            frequencies=[0.1, 0.2],
            hurst=0.65,
            anomaly=0.12,
            recommendations=["调整参数"],
            reasoning=["数据特性需要"],
            confidence=0.75
        )

    @pytest.fixture
    def architecture_decision(self):
        """模拟架构决策"""
        return ArchitectDecision(
            spectral_threshold=0.65,
            laplacian_weight=0.018,
            learning_rate_multiplier=1.3,
            rationale="适应数据特性",
            confidence=0.72
        )

    def test_optimize(self, optimizer, analysis_result, architecture_decision):
        """测试优化功能"""
        decision = optimizer.optimize(analysis_result, architecture_decision)

        assert isinstance(decision, OptimizerDecision)
        assert isinstance(decision.apply_changes, bool)
        assert 'parameters' in decision.__dict__
        assert 'risk' in decision.__dict__


class TestAutoGenController:
    """测试AutoGen控制器"""

    @pytest.fixture
    def mock_config(self):
        """模拟配置"""
        return {
            'autogen': {
                'deepseek_api_key': 'test-deepseek-key',
                'qwen_api_key': 'test-qwen-key',
                'max_rounds': 3,
                'check_interval': 50,
                'timeout': 30
            }
        }

    @pytest.fixture
    def controller(self, mock_config):
        """创建模拟控制器"""
        # 模拟所有API调用
        with patch('agents.autogen_system.requests') as mock_requests:
            # 模拟所有API响应
            mock_response = Mock()
            mock_response.status_code = 200

            # 分析师响应
            analyst_response = json.dumps({
                'pattern': 'non_stationary',
                'frequencies': [0.1, 0.2],
                'hurst': 0.65,
                'anomaly': 0.12,
                'recommendations': ['调整谱门控'],
                'reasoning': ['数据有趋势'],
                'confidence': 0.75
            })

            # 架构师响应
            architect_response = json.dumps({
                'spectral_threshold': 0.65,
                'laplacian_weight': 0.018,
                'learning_rate_multiplier': 1.3,
                'rationale': '适应非平稳特性',
                'confidence': 0.72
            })

            # 优化师响应
            optimizer_response = json.dumps({
                'apply_changes': True,
                'parameters': {
                    'spectral_threshold': 0.65,
                    'laplacian_weight': 0.018
                },
                'steps': ['应用调整'],
                'risk': 'medium',
                'expected_improvement': {'mse': 0.04, 'mae': 0.025}
            })

            # 设置响应序列
            mock_response.json.side_effect = [
                {'choices': [{'message': {'content': analyst_response}}]},
                {'choices': [{'message': {'content': architect_response}}]},
                {'choices': [{'message': {'content': optimizer_response}}]}
            ]

            mock_requests.get.return_value = mock_response
            mock_requests.post.return_value = mock_response

            return AutoGenController(mock_config)

    def test_initialization(self, controller):
        """测试初始化"""
        assert controller.interaction_count == 0
        assert len(controller.conversation_history) == 0
        assert hasattr(controller, 'analyst')
        assert hasattr(controller, 'architect')
        assert hasattr(controller, 'optimizer')

    def test_collaborative_analysis(self, controller):
        """测试协同分析"""
        context = {
            'features': {
                'shape': [32, 96, 64],
                'statistics': {'mean': [0.1], 'std': [0.05]},
                'frequency': {'dominant': 3}
            },
            'metrics': {'mse': 0.25, 'mae': 0.35},
            'current_params': {
                'spectral_threshold': 0.5,
                'laplacian_weight': 0.01
            }
        }

        result = controller.collaborative_analysis(context)

        # 验证结果结构
        assert 'final_decision' in result
        assert 'analysis_summary' in result
        assert 'conversation_id' in result

        # 验证对话历史
        assert len(controller.conversation_history) == 1
        conversation = controller.conversation_history[0]

        assert conversation['step'] == 1
        assert 'analysis' in conversation
        assert 'architecture' in conversation
        assert 'optimization' in conversation

    def test_apply_decision(self, controller):
        """测试应用决策"""
        # 模拟模型
        mock_model = Mock()
        mock_model.update_parameters = Mock()

        parameters = {
            'spectral_threshold': 0.65,
            'laplacian_weight': 0.018
        }

        # 应用决策
        success = controller.apply_decision(mock_model, parameters)

        assert success is True
        mock_model.update_parameters.assert_called_with(
            spectral_threshold=0.65,
            laplacian_weight=0.018
        )

    def test_get_stats(self, controller):
        """测试获取统计信息"""
        # 先进行一次分析
        context = {
            'features': {'shape': [1, 1, 1]},
            'metrics': {'mse': 0.1},
            'current_params': {'spectral_threshold': 0.5}
        }

        controller.collaborative_analysis(context)

        # 获取统计
        stats = controller.get_stats()

        assert stats['interaction_count'] == 1
        assert stats['conversation_count'] == 1
        assert 'total_duration' in stats
        assert 'avg_duration' in stats

    def test_error_handling(self, mock_config):
        """测试错误处理"""
        # 模拟API失败
        with patch('agents.autogen_system.requests') as mock_requests:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_requests.post.return_value = mock_response
            mock_requests.get.return_value = mock_response

            controller = AutoGenController(mock_config)

            context = {
                'features': {'shape': [1, 1, 1]},
                'metrics': {'mse': 0.1},
                'current_params': {}
            }

            result = controller.collaborative_analysis(context)

            # 应该返回默认结果
            assert result['final_decision']['apply_changes'] is False
            assert 'error' not in result  # 或者应该有错误信息


class TestIntegration:
    """集成测试"""

    def test_workflow(self, tmp_path):
        """测试完整工作流程"""
        # 创建临时目录
        test_dir = tmp_path / "test_agents"
        test_dir.mkdir()

        # 创建测试配置
        config = {
            'autogen': {
                'deepseek_api_key': 'test-key-1',
                'qwen_api_key': 'test-key-2',
                'max_rounds': 2,
                'check_interval': 10,
                'timeout': 10
            }
        }

        # 创建模拟上下文
        context = {
            'features': {
                'shape': [32, 96, 64],
                'statistics': {
                    'mean': [0.0, 0.1, 0.2],
                    'std': [1.0, 1.1, 1.2]
                },
                'frequency': {
                    'dominant': 5
                }
            },
            'metrics': {
                'mse': 0.234,
                'mae': 0.345
            },
            'current_params': {
                'spectral_threshold': 0.5,
                'laplacian_weight': 0.01
            },
            'step': 100
        }

        # 使用模拟的控制器
        with patch('agents.autogen_system.requests') as mock_requests:
            # 设置模拟响应
            mock_response = Mock()
            mock_response.status_code = 200

            responses = [
                json.dumps({
                    'pattern': 'non_stationary',
                    'frequencies': [0.1, 0.15, 0.2],
                    'hurst': 0.68,
                    'anomaly': 0.18,
                    'recommendations': ['增加谱门控阈值', '调整正则化权重'],
                    'reasoning': ['数据显示明显趋势', '需要更好的正则化'],
                    'confidence': 0.77
                }),
                json.dumps({
                    'spectral_threshold': 0.67,
                    'laplacian_weight': 0.019,
                    'learning_rate_multiplier': 1.25,
                    'rationale': '针对非平稳数据优化',
                    'confidence': 0.74
                }),
                json.dumps({
                    'apply_changes': True,
                    'parameters': {
                        'spectral_threshold': 0.67,
                        'laplacian_weight': 0.019
                    },
                    'steps': ['验证参数合理性', '分步实施'],
                    'risk': 'low',
                    'expected_improvement': {'mse': 0.035, 'mae': 0.028}
                })
            ]

            mock_response.json.side_effect = [
                {'choices': [{'message': {'content': resp}}]} for resp in responses
            ]

            mock_requests.get.return_value = mock_response
            mock_requests.post.return_value = mock_response

            # 创建控制器
            controller = AutoGenController(config)

            # 执行协同分析
            result = controller.collaborative_analysis(context)

            # 验证结果
            assert result['conversation_id'] == 1
            assert 'final_decision' in result
            assert 'analysis_summary' in result

            decision = result['final_decision']
            assert 'apply_changes' in decision
            assert 'parameters' in decision

            # 验证对话历史
            assert len(controller.conversation_history) == 1

            # 测试应用决策
            mock_model = Mock()
            mock_model.update_parameters = Mock()

            success = controller.apply_decision(
                mock_model,
                decision['parameters']
            )

            assert success is True
            mock_model.update_parameters.assert_called_once()

            # 测试统计信息
            stats = controller.get_stats()
            assert stats['interaction_count'] == 1
            assert stats['conversation_count'] == 1


def run_all_tests():
    """运行所有测试"""
    print("🤖 开始智能体系统测试...")

    # 临时目录用于测试
    temp_dir = tempfile.mkdtemp()

    try:
        print("1. 测试数据结构...")

        # 测试AnalystResult
        analyst_result = AnalystResult(
            pattern="stationary",
            frequencies=[0.1, 0.2],
            hurst=0.5,
            anomaly=0.1,
            recommendations=["测试建议"],
            reasoning=["测试推理"],
            confidence=0.8
        )
        assert analyst_result.pattern == "stationary"
        print("   ✅ AnalystResult测试通过")

        # 测试ArchitectDecision
        architect_decision = ArchitectDecision(
            spectral_threshold=0.6,
            laplacian_weight=0.015,
            learning_rate_multiplier=1.2,
            rationale="测试理由",
            confidence=0.7
        )
        assert 0.1 <= architect_decision.spectral_threshold <= 0.9
        print("   ✅ ArchitectDecision测试通过")

        # 测试OptimizerDecision
        optimizer_decision = OptimizerDecision(
            apply_changes=True,
            parameters={'test': 0.5},
            steps=["步骤1"],
            risk="low",
            expected_improvement={"mse": 0.05}
        )
        assert optimizer_decision.apply_changes is True
        print("   ✅ OptimizerDecision测试通过")

        print("\n2. 测试AutoGenController...")

        # 创建模拟配置
        config = {
            'autogen': {
                'deepseek_api_key': 'test-key',
                'qwen_api_key': 'test-qwen-key',
                'max_rounds': 2,
                'check_interval': 10,
                'timeout': 10
            }
        }

        # 使用模拟测试控制器
        import requests_mock

        with requests_mock.Mocker() as m:
            # 模拟API响应
            m.get(requests_mock.ANY, status_code=200)

            analyst_response = {
                'choices': [{
                    'message': {
                        'content': json.dumps({
                            'pattern': 'non_stationary',
                            'frequencies': [0.1, 0.2],
                            'hurst': 0.65,
                            'anomaly': 0.12,
                            'recommendations': ['调整参数'],
                            'reasoning': ['数据特性'],
                            'confidence': 0.75
                        })
                    }
                }]
            }

            architect_response = {
                'choices': [{
                    'message': {
                        'content': json.dumps({
                            'spectral_threshold': 0.65,
                            'laplacian_weight': 0.018,
                            'learning_rate_multiplier': 1.3,
                            'rationale': '调整理由',
                            'confidence': 0.72
                        })
                    }
                }]
            }

            optimizer_response = {
                'choices': [{
                    'message': {
                        'content': json.dumps({
                            'apply_changes': True,
                            'parameters': {
                                'spectral_threshold': 0.65,
                                'laplacian_weight': 0.018
                            },
                            'steps': ['应用调整'],
                            'risk': 'medium',
                            'expected_improvement': {'mse': 0.04}
                        })
                    }
                }]
            }

            # 设置不同的端点响应
            m.post("https://api.deepseek.com/v1/chat/completions",
                   [{'json': analyst_response}, {'json': optimizer_response}])

            # 对于Qwen，我们需要模拟requests.post
            def qwen_callback(request, context):
                return json.dumps(architect_response['choices'][0]['message']['content'])

            m.post("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                   text=qwen_callback)

            # 创建控制器
            controller = AutoGenController(config)

            # 测试协同分析
            context = {
                'features': {
                    'shape': [32, 96, 64],
                    'statistics': {'mean': [0.1], 'std': [0.05]},
                    'frequency': {'dominant': 3}
                },
                'metrics': {'mse': 0.25, 'mae': 0.35},
                'current_params': {'spectral_threshold': 0.5}
            }

            result = controller.collaborative_analysis(context)

            assert 'final_decision' in result
            assert result['conversation_id'] == 1
            print("   ✅ AutoGenController协同分析测试通过")

            # 测试应用决策
            mock_model = Mock()
            mock_model.update_parameters = Mock()

            success = controller.apply_decision(
                mock_model,
                {'spectral_threshold': 0.65}
            )

            assert success is True
            print("   ✅ AutoGenController应用决策测试通过")

            # 测试统计信息
            stats = controller.get_stats()
            assert stats['interaction_count'] == 1
            print("   ✅ AutoGenController统计信息测试通过")

        print("\n🎉 所有智能体系统测试通过！")

    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_all_tests()