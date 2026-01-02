"""
AutoGen多智能体系统 - 兼容旧版本
使用旧版OpenAI API (0.28.1) 和旧版AutoGen
"""

import os
import sys
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
import warnings
from datetime import datetime

# 尝试导入旧版OpenAI
try:
    import openai  # 版本 0.28.1

    OPENAI_VERSION = getattr(openai, '__version__', 'unknown')
    print(f"✅ 使用旧版OpenAI: {OPENAI_VERSION}")
    OPENAI_AVAILABLE = True
except ImportError:
    print("❌ OpenAI未安装，请运行: pip install openai==0.28.1")
    OPENAI_AVAILABLE = False

# 尝试导入AutoGen
try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

    AUTO_GEN_VERSION = getattr(autogen, '__version__', 'unknown')
    print(f"✅ 使用AutoGen版本: {AUTO_GEN_VERSION}")
    AUTO_GEN_AVAILABLE = True
except ImportError as e:
    print(f"❌ AutoGen导入失败: {e}")
    AUTO_GEN_AVAILABLE = False

from .memory_bank import MemoryBank


@dataclass
class DebateConfig:
    """辩论配置 - 旧版兼容"""
    agent_count: int = 3
    debate_rounds: int = 2
    temperature: float = 0.7
    use_memory: bool = True
    use_real_llm: bool = False  # 是否使用真实LLM
    api_config: Optional[Dict] = None


class DebateResult:
    """辩论结果"""

    def __init__(self, consensus: str = "", recommendations: List[str] = None):
        self.consensus = consensus
        self.recommendations = recommendations or []
        self.debate_log = []
        self.raw_messages = []

    def get_consensus_insights(self) -> Dict[str, Any]:
        """从共识中提取见解"""
        insights = {}
        if "调整趋势" in self.consensus or "adjust trend" in self.consensus.lower():
            insights["adjust_trend"] = 0.1  # 默认10%调整
        if "平滑" in self.consensus or "smooth" in self.consensus.lower():
            insights["smooth_variance"] = True
        if "季节性" in self.consensus or "seasonal" in self.consensus.lower():
            insights["seasonal_adjust"] = True
        return insights


class AutoGenDebateSystem:
    """AutoGen多智能体辩论系统 - 旧版兼容"""

    def __init__(self, config: DebateConfig, memory_bank: Optional[MemoryBank] = None):
        self.config = config
        self.memory_bank = memory_bank or MemoryBank(config={})

        # 从环境变量获取API密钥
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.qwen_api_key = os.getenv("QWEN_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        print(f"🤖 初始化AutoGen辩论系统 (使用真实LLM: {config.use_real_llm})")

        if config.use_real_llm and OPENAI_AVAILABLE and AUTO_GEN_AVAILABLE:
            self.agents = self._initialize_real_agents()
            self.llm_mode = "real"
        else:
            self.agents = self._initialize_mock_agents()
            self.llm_mode = "mock"

        self.conversation_history = []

    def _initialize_real_agents(self) -> Dict[str, Any]:
        """初始化真实的智能体（使用旧版AutoGen API）"""
        print("🔧 使用旧版AutoGen API初始化真实智能体...")

        agents = {}

        try:
            # 配置DeepSeek API (旧版OpenAI格式)
            deepseek_config = {
                "model": "deepseek-chat",  # 或 "deepseek-reasoner"
                "api_key": self.deepseek_api_key,
                "api_base": "https://api.deepseek.com/v1",
                "api_type": "open_ai"
            }

            # 配置Qwen API (旧版格式)
            qwen_config = {
                "model": "qwen-max",
                "api_key": self.qwen_api_key,
                "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "api_type": "open_ai"
            }

            # 1. 统计专家 (使用DeepSeek)
            statistician = AssistantAgent(
                name="Statistician",
                llm_config={
                    "config_list": [deepseek_config],
                    "temperature": self.config.temperature,
                    "request_timeout": 120,
                },
                system_message="""你是统计学家专家，专注于时间序列数据分析。
                分析数据趋势、季节性、周期性，评估预测的统计有效性。
                提供基于统计学的改进建议。"""
            )
            agents["statistician"] = statistician

            # 2. 领域专家 (使用Qwen)
            domain_expert = AssistantAgent(
                name="DomainExpert",
                llm_config={
                    "config_list": [qwen_config],
                    "temperature": self.config.temperature,
                    "request_timeout": 120,
                },
                system_message="""你是时间序列预测领域专家。
                基于实际经验判断预测的合理性，识别异常模式。
                提供基于领域知识的改进建议。"""
            )
            agents["domain_expert"] = domain_expert

            # 3. 模型专家 (使用DeepSeek)
            model_expert = AssistantAgent(
                name="ModelExpert",
                llm_config={
                    "config_list": [deepseek_config],
                    "temperature": self.config.temperature,
                    "request_timeout": 120,
                },
                system_message="""你是深度学习模型专家。
                分析模型架构对预测的影响，建议参数调整。
                评估不同预测技术的适用性。"""
            )
            agents["model_expert"] = model_expert

            # 4. 协调者
            coordinator = UserProxyAgent(
                name="Coordinator",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=5,
                code_execution_config=False,
                system_message="你是辩论协调者，引导讨论并总结共识。"
            )
            agents["coordinator"] = coordinator

            print(f"✅ 真实智能体初始化完成: {list(agents.keys())}")

            # 测试API连接
            if not self.test_api_connection():
                print("⚠️  API连接测试失败，切换到模拟模式")
                return self._initialize_mock_agents()

        except Exception as e:
            print(f"❌ 真实智能体初始化失败: {e}")
            print("🔄 切换到模拟智能体")
            return self._initialize_mock_agents()

        return agents

    def _initialize_mock_agents(self) -> Dict[str, Any]:
        """初始化模拟智能体"""
        print("🔄 使用模拟智能体")

        class MockAgent:
            def __init__(self, name, role):
                self.name = name
                self.role = role
                self.llm_config = {}

        agents = {
            "statistician": MockAgent("Statistician", "统计专家"),
            "domain_expert": MockAgent("DomainExpert", "领域专家"),
            "model_expert": MockAgent("ModelExpert", "模型专家"),
            "coordinator": MockAgent("Coordinator", "协调者")
        }

        return agents

    def test_api_connection(self) -> bool:
        """测试API连接"""
        if self.llm_mode == "mock":
            return True

        try:
            # 测试DeepSeek连接
            if self.deepseek_api_key:
                print("🔍 测试DeepSeek API连接...")
                import openai

                # 使用旧版OpenAI API
                openai.api_key = self.deepseek_api_key
                openai.api_base = "https://api.deepseek.com/v1"

                try:
                    response = openai.ChatCompletion.create(
                        model="deepseek-chat",
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=5,
                        timeout=10
                    )
                    if response and hasattr(response, 'choices'):
                        print("✅ DeepSeek API连接正常")
                        return True
                except Exception as e:
                    print(f"❌ DeepSeek API连接失败: {e}")

            return False

        except Exception as e:
            print(f"❌ API连接测试异常: {e}")
            return False

    def start_debate(self, topic: str, context: Dict[str, Any], question: str) -> DebateResult:
        """启动多智能体辩论"""
        print(f"\n🤖 启动{self.llm_mode.upper()}模式辩论: {topic}")

        result = DebateResult()

        if self.llm_mode == "mock" or not AUTO_GEN_AVAILABLE:
            # 模拟辩论
            return self._simulate_debate(topic, context, question)

        try:
            # 准备辩论上下文
            debate_context = self._prepare_debate_context(context)

            # 创建群聊
            agent_list = [
                self.agents["statistician"],
                self.agents["domain_expert"],
                self.agents["model_expert"],
                self.agents["coordinator"]
            ]

            group_chat = GroupChat(
                agents=agent_list,
                messages=[],
                max_round=self.config.debate_rounds * 2,
                allow_repeat_speaker=False
            )

            manager = GroupChatManager(
                groupchat=group_chat,
                llm_config=self.agents["model_expert"].llm_config
            )

            # 启动讨论
            initial_message = f"""辩论主题：{topic}

上下文信息：
{debate_context}

讨论问题：
{question}

请各位专家基于专业领域发表意见，最后形成共识。"""

            # 使用旧版AutoGen的聊天方式
            chat_result = self.agents["coordinator"].initiate_chat(
                manager,
                message=initial_message,
                max_turns=self.config.debate_rounds * 2
            )

            # 处理结果
            if hasattr(chat_result, 'chat_history'):
                result.raw_messages = chat_result.chat_history

                # 提取共识
                consensus, recommendations = self._extract_consensus(chat_result.chat_history)
                result.consensus = consensus
                result.recommendations = recommendations

                # 保存日志
                for msg in chat_result.chat_history:
                    if isinstance(msg, dict) and 'content' in msg:
                        speaker = msg.get('name', 'Unknown')
                        content = msg['content']
                        result.debate_log.append(f"{speaker}: {content[:100]}...")

            print(f"✅ 真实辩论完成！共识: {result.consensus[:50]}...")

        except Exception as e:
            print(f"❌ 真实辩论失败: {e}")
            import traceback
            traceback.print_exc()
            print("🔄 切换到模拟辩论")
            return self._simulate_debate(topic, context, question)

        # 存储到记忆银行
        if self.memory_bank:
            self.memory_bank.store_experience({
                "type": "real_debate",
                "timestamp": datetime.now().isoformat(),
                "topic": topic,
                "llm_mode": self.llm_mode,
                "consensus": result.consensus,
                "recommendations": result.recommendations
            })

        return result

    def _prepare_debate_context(self, context: Dict[str, Any]) -> str:
        """准备辩论上下文"""
        lines = []

        if "data_description" in context:
            lines.append(f"数据描述: {context['data_description']}")

        if "historical_stats" in context:
            lines.append("历史数据统计:")
            for k, v in context["historical_stats"].items():
                lines.append(f"  - {k}: {v}")

        if "base_prediction_stats" in context:
            lines.append("预测结果统计:")
            for k, v in context["base_prediction_stats"].items():
                lines.append(f"  - {k}: {v}")

        if "model_info" in context:
            lines.append("模型信息:")
            for k, v in context["model_info"].items():
                lines.append(f"  - {k}: {v}")

        return "\n".join(lines)

    def _extract_consensus(self, chat_history: List) -> tuple:
        """从聊天历史提取共识"""
        if not chat_history:
            return "未达成共识", []

        # 查找包含总结的消息
        last_messages = chat_history[-5:] if len(chat_history) >= 5 else chat_history
        consensus_parts = []
        recommendations = []

        for msg in last_messages:
            if isinstance(msg, dict):
                content = msg.get('content', '')
                if content:
                    lower_content = content.lower()

                    # 识别总结性内容
                    summary_keywords = ['总结', '共识', '结论', '建议', '建议如下', 'recommend', 'conclusion']
                    if any(kw in lower_content for kw in summary_keywords):
                        consensus_parts.append(content)

                    # 提取建议列表
                    lines = content.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith(('1.', '2.', '3.', '-', '•', '建议', 'recommend')):
                            recommendations.append(line)

        # 生成共识
        if consensus_parts:
            consensus = " ".join(consensus_parts[-2:])  # 取最后两个部分
        else:
            # 使用最后的消息
            last_contents = [m.get('content', '') for m in last_messages if isinstance(m, dict)]
            consensus = " ".join(last_contents[-2:])

        # 限制长度
        if len(consensus) > 300:
            consensus = consensus[:297] + "..."

        # 如果没有建议，生成默认的
        if not recommendations:
            recommendations = [
                "建议对预测趋势进行小幅调整",
                "增加数据平滑处理",
                "优化模型正则化参数"
            ]

        return consensus, recommendations[:5]

    def _simulate_debate(self, topic: str, context: Dict[str, Any], question: str) -> DebateResult:
        """模拟辩论"""
        result = DebateResult()

        # 基于上下文生成智能模拟
        pred_stats = context.get("base_prediction_stats", {})

        if pred_stats:
            mean_val = pred_stats.get("mean", 1.0)
            std_val = pred_stats.get("std", 0.1)

            if std_val > 0.5:
                result.consensus = "预测波动较大，建议增强模型稳定性。"
                result.recommendations = [
                    f"应用滑动平均平滑（窗口大小建议: {int(10 / std_val)}）",
                    f"增加L2正则化权重: {std_val * 0.2:.3f}",
                    "考虑使用更长历史数据训练"
                ]
            else:
                result.consensus = "预测相对稳定，可优化模型表达能力。"
                result.recommendations = [
                    "增加神经网络层数",
                    "尝试不同激活函数",
                    "调整学习率调度"
                ]
        else:
            result.consensus = "专家建议从多个角度优化预测模型。"
            result.recommendations = [
                "调整趋势预测",
                "优化超参数",
                "增强特征工程"
            ]

        result.debate_log = [
            "Statistician: 分析了数据的统计特性",
            "DomainExpert: 提供了领域经验建议",
            "ModelExpert: 建议了模型优化方案"
        ]

        print(f"✅ 模拟辩论完成")

        # 存储到记忆银行
        if self.memory_bank:
            self.memory_bank.store_experience({
                "type": "mock_debate",
                "timestamp": datetime.now().isoformat(),
                "topic": topic,
                "consensus": result.consensus,
                "recommendations": result.recommendations
            })

        return result