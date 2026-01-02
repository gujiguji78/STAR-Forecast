"""
辩论式智能体系统 - 支持多智能体辩论、投票和共识形成
比普通AutoGen更加强调批判性思维和深度辩论
"""
import asyncio
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime
import uuid
import logging
from collections import defaultdict

from .autogen_system import AutoGenMultiAgentSystem, ConversationResult


class DebatePhase(Enum):
    """辩论阶段"""
    OPENING = "opening"  # 开篇陈述
    REBUTTAL = "rebuttal"  # 反驳阶段
    CROSS_EXAMINATION = "cross"  # 交叉质询
    CLOSING = "closing"  # 结案陈词
    VOTING = "voting"  # 投票阶段


class DebateRole(Enum):
    """辩论角色"""
    PROPOSITION = "proposition"  # 正方
    OPPOSITION = "opposition"  # 反方
    MODERATOR = "moderator"  # 主持人
    JUDGE = "judge"  # 评委


@dataclass
class DebateArgument:
    """辩论论点"""
    argument_id: str
    speaker: str
    role: DebateRole
    phase: DebatePhase
    content: str
    claims: List[str] = field(default_factory=list)  # 主张列表
    evidence: Dict[str, Any] = field(default_factory=dict)  # 证据
    fallacies: List[str] = field(default_factory=list)  # 逻辑谬误标记
    strength: float = 0.0  # 论点强度
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DebateRound:
    """辩论轮次"""
    round_id: str
    phase: DebatePhase
    arguments: List[DebateArgument] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    summary: Optional[str] = None


@dataclass
class DebateResult:
    """辩论结果"""
    debate_id: str
    topic: str
    rounds: List[DebateRound]
    final_vote: Dict[str, int] = field(default_factory=dict)  # 支持方 -> 票数
    consensus: Optional[Dict[str, Any]] = None
    winner: Optional[str] = None
    reasoning: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


class LogicalFallacyDetector:
    """逻辑谬误检测器"""

    FALLACIES = {
        'ad_hominem': ['人身攻击', '攻击人格', '贬低对方'],
        'straw_man': ['稻草人', '曲解论点', '歪曲立场'],
        'false_cause': ['虚假因果', '相关当因果', '因果颠倒'],
        'slippery_slope': ['滑坡谬误', '极端推论', '连锁反应'],
        'appeal_to_emotion': ['诉诸情感', '情绪化论证', '煽情'],
        'black_or_white': ['非黑即白', '二元对立', '排除中间'],
        'bandwagon': ['从众谬误', '大家都这样', '流行即正确'],
        'appeal_to_authority': ['诉诸权威', '专家说', '权威论断'],
        'hasty_generalization': ['草率概括', '以偏概全', '样本不足'],
        'red_herring': ['转移话题', '偷换概念', '离题万里']
    }

    @classmethod
    def detect(cls, text: str) -> List[str]:
        """检测文本中的逻辑谬误"""
        detected = []
        text_lower = text.lower()

        for fallacy, keywords in cls.FALLACIES.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected.append(fallacy)
                    break

        return detected


class ArgumentStrengthAnalyzer:
    """论点强度分析器"""

    @classmethod
    def analyze(cls, argument: str, evidence: Dict[str, Any] = None) -> float:
        """分析论点强度"""
        strength = 0.0

        # 1. 长度分析（适中的长度更好）
        length_score = min(len(argument) / 1000, 1.0)  # 不超过1000字
        strength += length_score * 0.2

        # 2. 证据支持
        if evidence:
            evidence_score = cls._evaluate_evidence(evidence)
            strength += evidence_score * 0.3

        # 3. 逻辑结构
        logic_score = cls._evaluate_logic(argument)
        strength += logic_score * 0.3

        # 4. 清晰度
        clarity_score = cls._evaluate_clarity(argument)
        strength += clarity_score * 0.2

        # 5. 减去谬误惩罚
        fallacies = LogicalFallacyDetector.detect(argument)
        if fallacies:
            strength -= len(fallacies) * 0.1

        return max(0.0, min(1.0, strength))

    @classmethod
    def _evaluate_evidence(cls, evidence: Dict[str, Any]) -> float:
        """评估证据质量"""
        score = 0.0

        if 'data' in evidence and isinstance(evidence['data'], (list, dict)):
            score += 0.3

        if 'sources' in evidence and evidence['sources']:
            score += 0.3

        if 'statistics' in evidence:
            score += 0.2

        if 'examples' in evidence:
            score += 0.2

        return min(1.0, score)

    @classmethod
    def _evaluate_logic(cls, argument: str) -> float:
        """评估逻辑结构"""
        # 检查逻辑连接词
        connectors = ['因此', '所以', '因为', '由于', '导致', '结果',
                      '由此可见', '综上所述', '总而言之', '首先', '其次']

        connector_count = sum(1 for c in connectors if c in argument)
        logic_score = min(connector_count / 5, 1.0) * 0.7

        # 检查结构
        if '主张' in argument and '理由' in argument:
            logic_score += 0.3

        return min(1.0, logic_score)

    @classmethod
    def _evaluate_clarity(cls, argument: str) -> float:
        """评估清晰度"""
        # 简单启发式：检查句子长度和标点
        sentences = argument.replace('。', '.').replace('！', '!').replace('？', '?').split('.')
        avg_sentence_len = sum(len(s.strip()) for s in sentences) / max(len(sentences), 1)

        # 理想句子长度：20-50字
        if 20 <= avg_sentence_len <= 50:
            clarity = 1.0
        elif avg_sentence_len < 10:
            clarity = 0.3
        elif avg_sentence_len > 100:
            clarity = 0.3
        else:
            clarity = 0.7

        # 检查专业术语过多
        jargon = ['泛化误差', '梯度消失', '过拟合', '正则化', '注意力机制']
        jargon_count = sum(1 for j in jargon if j in argument)
        if jargon_count > 5:
            clarity *= 0.5

        return clarity


class DebateSystem:
    """
    辩论式智能体系统

    特点：
    1. 正式辩论结构（开篇、反驳、质询、结案）
    2. 逻辑谬误检测
    3. 论点强度分析
    4. 评委投票机制
    5. 共识形成过程
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 辩论参数
        self.debate_params = config.get('debate', {
            'max_rounds': 4,
            'time_limit_per_speaker': 120,  # 秒
            'min_arguments_per_side': 2,
            'require_evidence': True,
            'voting_threshold': 0.6  # 60%投票通过
        })

        # 智能体池
        self.agents = {}
        self.initialize_agents()

        # 辩论历史
        self.debate_history: Dict[str, DebateResult] = {}

        # 评委系统
        self.judges = self._create_judges()

        self.logger.info("✅ 辩论式智能体系统初始化完成")

    def initialize_agents(self):
        """初始化辩论智能体"""
        # 创建不同类型的辩论者
        self.agents = {
            'pro_analyst': self._create_agent(
                name="ProAnalyst",
                role=DebateRole.PROPOSITION,
                style="严谨的分析师，善于用数据和逻辑支持论点",
                bias="倾向于优化和改进现有方案"
            ),
            'con_critic': self._create_agent(
                name="ConCritic",
                role=DebateRole.OPPOSITION,
                style="严格的批评家，专注于发现问题和风险",
                bias="倾向于保守和谨慎的方案"
            ),
            'balanced_architect': self._create_agent(
                name="BalancedArchitect",
                role=DebateRole.PROPOSITION,
                style="平衡的建筑师，寻求折中和创新",
                bias="倾向于融合不同观点的方案"
            ),
            'radical_innovator': self._create_agent(
                name="RadicalInnovator",
                role=DebateRole.PROPOSITION,
                style="激进的创新者，追求突破性改变",
                bias="倾向于颠覆性方案"
            )
        }

        # 创建主持人
        self.moderator = self._create_moderator()

    def _create_agent(self, name: str, role: DebateRole, style: str, bias: str) -> Dict[str, Any]:
        """创建辩论智能体"""
        # 根据角色设置系统提示词
        if role == DebateRole.PROPOSITION:
            system_prompt = f"""你是一位辩论的正方代表，{style}。

辩论风格：{style}
固有偏见：{bias}

你的任务是：
1. 提出强有力的支持论点
2. 提供数据和证据支持
3. 预测并反驳可能的反对意见
4. 在反驳阶段有效回应反方
5. 在质询阶段清晰回答问题

辩论技巧：
- 使用三段论：主张、理由、结论
- 提供具体证据和例子
- 避免逻辑谬误
- 保持理性和专业性
- 攻击论点而非个人

输出格式：
主张：[你的核心主张]
理由：[支持主张的理由]
证据：[相关证据和数据]
预测反驳：[预测的反方论点]
回应策略：[如何回应]
"""
        else:  # OPPOSITION
            system_prompt = f"""你是一位辩论的反方代表，{style}。

辩论风格：{style}
固有偏见：{bias}

你的任务是：
1. 指出正方论点的缺陷和风险
2. 提出替代方案和更好选择
3. 质疑证据的有效性和相关性
4. 在反驳阶段有效削弱正方论点
5. 在质询阶段提出尖锐问题

辩论技巧：
- 指出逻辑谬误和假设问题
- 强调被忽略的风险和成本
- 提供对比和替代方案
- 保持建设性批评态度
- 聚焦议题本身

输出格式：
反对点：[主要反对意见]
理由：[反对的理由]
风险：[识别出的风险]
替代方案：[更好的选择]
质疑问题：[质询阶段的问题]
"""

        return {
            'name': name,
            'role': role,
            'style': style,
            'bias': bias,
            'system_prompt': system_prompt,
            'arguments': [],
            'performance': {
                'arguments_made': 0,
                'argument_strength_avg': 0.0,
                'fallacies_detected': 0,
                'rebuttals_successful': 0
            }
        }

    def _create_moderator(self) -> Dict[str, Any]:
        """创建主持人"""
        return {
            'name': 'Moderator',
            'role': DebateRole.MODERATOR,
            'system_prompt': """你是辩论的主持人，负责：
1. 控制辩论流程和时间
2. 确保遵守辩论规则
3. 维持秩序和尊重氛围
4. 提出澄清性问题
5. 总结各方观点

主持原则：
- 保持中立和公正
- 确保每个人都有发言机会
- 及时制止人身攻击
- 聚焦核心议题
- 促进有建设性的讨论
""",
            'rules': self._get_debate_rules()
        }

    def _create_judges(self) -> List[Dict[str, Any]]:
        """创建评委"""
        return [
            {
                'name': 'TechnicalJudge',
                'specialty': '技术可行性',
                'criteria': ['创新性', '可行性', '效率', '可扩展性'],
                'weight': 0.4
            },
            {
                'name': 'RiskJudge',
                'specialty': '风险评估',
                'criteria': ['安全性', '稳定性', '风险控制', '容错性'],
                'weight': 0.3
            },
            {
                'name': 'PracticalJudge',
                'specialty': '实践应用',
                'criteria': ['易用性', '成本效益', '部署难度', '维护性'],
                'weight': 0.3
            }
        ]

    def _get_debate_rules(self) -> List[str]:
        """获取辩论规则"""
        return [
            "每位辩手有2分钟发言时间",
            "发言顺序：正方→反方→正方→反方",
            "质询阶段：每位辩手可以提问1分钟",
            "禁止人身攻击和情绪化言论",
            "必须提供证据支持论点",
            "必须回应对方的直接质疑",
            "结论必须基于已提出的论据"
        ]

    async def conduct_debate(self, topic: str, context: Dict[str, Any]) -> DebateResult:
        """
        执行辩论

        Args:
            topic: 辩论议题
            context: 上下文信息

        Returns:
            辩论结果
        """
        debate_id = str(uuid.uuid4())
        self.logger.info(f"⚖️ 开始辩论: {topic}")

        # 准备辩论
        prepared_topic = self._prepare_debate_topic(topic, context)

        # 辩论流程
        rounds = []

        # 阶段1：开篇陈述
        opening_round = await self._conduct_round(
            debate_id, DebatePhase.OPENING, prepared_topic, context
        )
        rounds.append(opening_round)

        # 阶段2：反驳阶段
        rebuttal_round = await self._conduct_round(
            debate_id, DebatePhase.REBUTTAL, prepared_topic, context,
            previous_round=opening_round
        )
        rounds.append(rebuttal_round)

        # 阶段3：交叉质询
        cross_round = await self._conduct_round(
            debate_id, DebatePhase.CROSS_EXAMINATION, prepared_topic, context,
            previous_round=rebuttal_round
        )
        rounds.append(cross_round)

        # 阶段4：结案陈词
        closing_round = await self._conduct_round(
            debate_id, DebatePhase.CLOSING, prepared_topic, context,
            previous_round=cross_round
        )
        rounds.append(closing_round)

        # 阶段5：投票和判决
        voting_result = await self._conduct_voting(rounds, context)

        # 构建辩论结果
        debate_result = DebateResult(
            debate_id=debate_id,
            topic=topic,
            rounds=rounds,
            final_vote=voting_result['votes'],
            winner=voting_result['winner'],
            consensus=voting_result['consensus'],
            reasoning=voting_result['reasoning'],
            metrics=self._calculate_debate_metrics(rounds)
        )

        # 保存到历史
        self.debate_history[debate_id] = debate_result

        self.logger.info(f"✅ 辩论完成: {debate_id}, 胜方: {debate_result.winner}")

        return debate_result

    def _prepare_debate_topic(self, topic: str, context: Dict[str, Any]) -> str:
        """准备辩论议题"""
        features = context.get('features', {})
        metrics = context.get('metrics', {})
        current_params = context.get('current_params', {})

        prepared = f"""
辩论议题：{topic}

背景信息：
- 模型性能：MSE={metrics.get('mse', 0):.4f}, MAE={metrics.get('mae', 0):.4f}
- 当前参数：{json.dumps(current_params, indent=2, ensure_ascii=False)}
- 数据特征：{json.dumps(features.get('statistics', {}), indent=2, ensure_ascii=False)}

辩论焦点：
1. 当前调整方案的技术可行性
2. 可能的风险和收益权衡
3. 替代方案的比较优势
4. 实施的优先顺序和策略

请基于以上背景进行辩论。
"""
        return prepared

    async def _conduct_round(self, debate_id: str, phase: DebatePhase,
                             topic: str, context: Dict[str, Any],
                             previous_round: DebateRound = None) -> DebateRound:
        """执行一个辩论轮次"""
        round_id = f"{debate_id}_{phase.value}"
        self.logger.info(f"  🗣️  {phase.value}阶段开始")

        round_args = []

        # 根据阶段确定发言顺序
        if phase == DebatePhase.OPENING:
            speakers = ['ProAnalyst', 'ConCritic', 'BalancedArchitect', 'RadicalInnovator']
        elif phase == DebatePhase.REBUTTAL:
            speakers = ['ConCritic', 'ProAnalyst', 'RadicalInnovator', 'BalancedArchitect']
        elif phase == DebatePhase.CROSS_EXAMINATION:
            speakers = self._create_cross_examination_pairs()
        elif phase == DebatePhase.CLOSING:
            speakers = ['ProAnalyst', 'ConCritic']  # 主要辩手结案

        for speaker_name in speakers:
            # 获取智能体
            agent = self._get_agent_by_name(speaker_name)
            if not agent:
                continue

            # 生成论点
            argument = await self._generate_argument(
                agent, phase, topic, context, previous_round
            )

            # 分析论点
            argument.strength = ArgumentStrengthAnalyzer.analyze(
                argument.content, argument.evidence
            )
            argument.fallacies = LogicalFallacyDetector.detect(argument.content)

            # 更新智能体表现
            self._update_agent_performance(agent['name'], argument)

            round_args.append(argument)

            self.logger.debug(f"    {speaker_name}: 强度={argument.strength:.2f}, "
                              f"谬误={len(argument.fallacies)}")

        # 生成轮次总结
        summary = self._generate_round_summary(phase, round_args)

        round_result = DebateRound(
            round_id=round_id,
            phase=phase,
            arguments=round_args,
            summary=summary
        )

        return round_result

    def _create_cross_examination_pairs(self) -> List[str]:
        """创建交叉质询对"""
        # 正方问反方，反方问正方
        pairs = []
        pro_agents = [name for name, agent in self.agents.items()
                      if agent['role'] == DebateRole.PROPOSITION]
        con_agents = [name for name, agent in self.agents.items()
                      if agent['role'] == DebateRole.OPPOSITION]

        # 创建配对
        for pro, con in zip(pro_agents, con_agents):
            pairs.extend([pro, con])

        return pairs

    def _get_agent_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """通过名称获取智能体"""
        for key, agent in self.agents.items():
            if agent['name'] == name:
                return agent
        return None

    async def _generate_argument(self, agent: Dict[str, Any], phase: DebatePhase,
                                 topic: str, context: Dict[str, Any],
                                 previous_round: DebateRound = None) -> DebateArgument:
        """生成论点"""
        # 根据阶段准备提示词
        prompt = self._build_argument_prompt(agent, phase, topic, context, previous_round)

        # 模拟API调用（实际应调用LLM API）
        # 这里使用模拟响应
        content = await self._simulate_llm_call(prompt, agent)

        # 解析论点内容
        claims, evidence = self._parse_argument_content(content)

        argument = DebateArgument(
            argument_id=str(uuid.uuid4()),
            speaker=agent['name'],
            role=agent['role'],
            phase=phase,
            content=content,
            claims=claims,
            evidence=evidence
        )

        return argument

    def _build_argument_prompt(self, agent: Dict[str, Any], phase: DebatePhase,
                               topic: str, context: Dict[str, Any],
                               previous_round: DebateRound = None) -> str:
        """构建论点提示词"""
        base_prompt = agent['system_prompt']

        phase_instructions = {
            DebatePhase.OPENING: "请进行开篇陈述，提出你的核心观点。",
            DebatePhase.REBUTTAL: "请反驳对方的观点，指出其缺陷。",
            DebatePhase.CROSS_EXAMINATION: "请向对方提问，或回答对方的问题。",
            DebatePhase.CLOSING: "请进行结案陈词，总结你的立场。"
        }

        prompt = f"""{base_prompt}

当前阶段：{phase_instructions.get(phase, '')}
辩论议题：{topic}

上下文信息：
{json.dumps(context, indent=2, ensure_ascii=False)}
"""

        if previous_round and phase != DebatePhase.OPENING:
            # 添加前一轮的论点摘要
            previous_summary = self._summarize_previous_round(previous_round, agent['role'])
            prompt += f"\n前一轮讨论摘要：\n{previous_summary}"

        return prompt

    async def _simulate_llm_call(self, prompt: str, agent: Dict[str, Any]) -> str:
        """模拟LLM调用（实际项目应替换为真实API调用）"""
        # 模拟思考时间
        await asyncio.sleep(0.5)

        # 根据智能体类型生成不同风格的响应
        style = agent.get('style', '')
        bias = agent.get('bias', '')

        if '激进' in style:
            responses = [
                "我们必须采取大胆的行动！当前的渐进式调整已经无法满足需求。",
                "突破性创新是唯一的出路，我们不能被传统思维束缚。",
                "高风险意味着高回报，我们应该勇敢尝试新的架构。"
            ]
        elif '批评' in style:
            responses = [
                "这个方案存在明显的风险，我们需要更谨慎的评估。",
                "对方忽略了关键的成本问题，实施起来困难重重。",
                "有更好的替代方案，为什么选择这个高风险选项？"
            ]
        elif '平衡' in style:
            responses = [
                "我们需要在创新和稳定之间找到平衡点。",
                "综合双方观点，我认为折中方案是最佳选择。",
                "既要考虑技术可行性，也要评估实际风险。"
            ]
        else:
            responses = [
                "基于数据分析，我认为这个方向是正确的。",
                "实验结果支持我们的观点，应该继续推进。",
                "从技术角度，这个方案具有明显优势。"
            ]

        # 添加一些技术内容
        tech_terms = [
            "谱门控阈值需要调整以优化频域特征提取。",
            "拉普拉斯正则化可以提升模型的平滑性。",
            "学习率调度应该根据损失曲线动态调整。",
            "注意力机制需要重新设计以捕捉长期依赖。"
        ]

        response = random.choice(responses) + " " + random.choice(tech_terms)

        # 根据偏见调整语气
        if '倾向' in bias:
            response += " " + bias

        return response

    def _parse_argument_content(self, content: str) -> Tuple[List[str], Dict[str, Any]]:
        """解析论点内容，提取主张和证据"""
        claims = []
        evidence = {}

        # 简单解析：查找主张和证据关键词
        lines = content.split('\n')

        for line in lines:
            if '主张' in line or '观点' in line or '认为' in line:
                claim = line.replace('主张：', '').replace('观点：', '').replace('认为', '').strip()
                if claim and len(claim) > 5:
                    claims.append(claim)

            if '证据' in line or '数据' in line or '实验' in line:
                # 提取证据信息
                evidence_key = line.split('：')[0] if '：' in line else 'evidence'
                evidence_value = line.split('：')[1] if '：' in line else line
                evidence[evidence_key] = evidence_value.strip()

        # 如果没有明确的主张，使用整个内容作为主张
        if not claims and len(content) > 10:
            claims.append(content[:100] + '...')

        return claims, evidence

    def _summarize_previous_round(self, previous_round: DebateRound,
                                  current_role: DebateRole) -> str:
        """总结前一轮辩论"""
        if not previous_round or not previous_round.arguments:
            return "无前一轮讨论。"

        # 提取对立方的论点
        opposing_args = []
        for arg in previous_round.arguments:
            if arg.role != current_role:
                opposing_args.append(arg)

        if not opposing_args:
            return "前一轮没有对立观点。"

        # 生成摘要
        summary = f"前一轮({previous_round.phase.value})中，对方提出了{len(opposing_args)}个论点：\n"

        for i, arg in enumerate(opposing_args[:3], 1):  # 最多3个论点
            summary += f"{i}. {arg.content[:100]}...\n"

        return summary

    def _update_agent_performance(self, agent_name: str, argument: DebateArgument):
        """更新智能体表现"""
        agent = self._get_agent_by_name(agent_name)
        if not agent:
            return

        perf = agent['performance']
        perf['arguments_made'] += 1

        # 更新平均强度
        current_avg = perf['argument_strength_avg']
        total_args = perf['arguments_made']
        perf['argument_strength_avg'] = (
                                                current_avg * (total_args - 1) + argument.strength
                                        ) / total_args

        # 更新谬误计数
        perf['fallacies_detected'] += len(argument.fallacies)

    def _generate_round_summary(self, phase: DebatePhase,
                                arguments: List[DebateArgument]) -> str:
        """生成轮次摘要"""
        if not arguments:
            return "本轮无论点。"

        # 按角色分组
        pro_args = [arg for arg in arguments if arg.role == DebateRole.PROPOSITION]
        con_args = [arg for arg in arguments if arg.role == DebateRole.OPPOSITION]

        summary = f"{phase.value}阶段总结：\n"
        summary += f"正方论点：{len(pro_args)}个，平均强度："
        summary += f"{sum(a.strength for a in pro_args) / len(pro_args):.2f}\n"
        summary += f"反方论点：{len(con_args)}个，平均强度："
        summary += f"{sum(a.strength for a in con_args) / len(con_args):.2f}\n"

        # 关键论点
        if arguments:
            strongest = max(arguments, key=lambda x: x.strength)
            summary += f"最强论点：{strongest.speaker}（强度：{strongest.strength:.2f}）\n"

        return summary

    async def _conduct_voting(self, rounds: List[DebateRound],
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """执行投票"""
        self.logger.info("  🗳️  开始投票...")

        # 收集所有论点
        all_arguments = []
        for round_obj in rounds:
            all_arguments.extend(round_obj.arguments)

        # 评委投票
        votes = {'proposition': 0, 'opposition': 0, 'abstain': 0}
        judge_reasons = []

        for judge in self.judges:
            vote, reason = await self._judge_vote(judge, all_arguments, context)
            votes[vote] += 1
            judge_reasons.append(f"{judge['name']}（{judge['specialty']}）：{reason}")

        # 确定胜方
        total_votes = sum(votes.values())
        pro_ratio = votes['proposition'] / total_votes if total_votes > 0 else 0

        if pro_ratio > self.debate_params['voting_threshold']:
            winner = 'proposition'
        elif votes['opposition'] > votes['proposition']:
            winner = 'opposition'
        else:
            winner = 'abstain'

        # 生成共识
        consensus = await self._generate_consensus(all_arguments, winner, context)

        return {
            'votes': votes,
            'winner': winner,
            'consensus': consensus,
            'judge_reasons': judge_reasons,
            'reasoning': '\n'.join(judge_reasons)
        }

    async def _judge_vote(self, judge: Dict[str, Any],
                          arguments: List[DebateArgument],
                          context: Dict[str, Any]) -> Tuple[str, str]:
        """评委投票"""
        # 根据评委专业领域评估论点
        pro_args = [arg for arg in arguments if arg.role == DebateRole.PROPOSITION]
        con_args = [arg for arg in arguments if arg.role == DebateRole.OPPOSITION]

        # 计算每个立场的得分
        pro_score = self._evaluate_by_criteria(pro_args, judge['criteria'])
        con_score = self._evaluate_by_criteria(con_args, judge['criteria'])

        # 应用评委权重
        pro_score *= judge['weight']
        con_score *= judge['weight']

        # 决定投票
        if pro_score > con_score * 1.1:  # 10%优势
            vote = 'proposition'
            reason = f"正方在{judge['specialty']}方面更具优势（{pro_score:.2f} vs {con_score:.2f}）"
        elif con_score > pro_score * 1.1:
            vote = 'opposition'
            reason = f"反方在{judge['specialty']}方面更具优势（{con_score:.2f} vs {pro_score:.2f}）"
        else:
            vote = 'abstain'
            reason = f"双方在{judge['specialty']}方面势均力敌（{pro_score:.2f} vs {con_score:.2f}）"

        return vote, reason

    def _evaluate_by_criteria(self, arguments: List[DebateArgument],
                              criteria: List[str]) -> float:
        """根据标准评估论点"""
        if not arguments:
            return 0.0

        total_score = 0.0

        for criterion in criteria:
            criterion_score = 0.0

            for arg in arguments:
                # 根据标准评估每个论点
                if criterion in ['创新性', '可行性']:
                    # 创新性和可行性评估
                    criterion_score += arg.strength * 0.5
                elif criterion in ['安全性', '稳定性']:
                    # 安全性和稳定性评估
                    if len(arg.fallacies) == 0:  # 无逻辑谬误
                        criterion_score += arg.strength * 0.6
                    else:
                        criterion_score += arg.strength * 0.3
                elif criterion in ['成本效益', '效率']:
                    # 成本和效率评估
                    if '证据' in arg.evidence:
                        criterion_score += arg.strength * 0.7
                    else:
                        criterion_score += arg.strength * 0.4

            total_score += criterion_score / len(criteria)

        return total_score / len(arguments) if arguments else 0.0

    async def _generate_consensus(self, arguments: List[DebateArgument],
                                  winner: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """生成共识决策"""
        # 提取最佳论点
        pro_args = [arg for arg in arguments if arg.role == DebateRole.PROPOSITION]
        con_args = [arg for arg in arguments if arg.role == DebateRole.OPPOSITION]

        # 选择最强论点
        best_pro = max(pro_args, key=lambda x: x.strength) if pro_args else None
        best_con = max(con_args, key=lambda x: x.strength) if con_args else None

        # 生成共识参数
        consensus = {
            'debate_winner': winner,
            'proposition_strength': sum(arg.strength for arg in pro_args) / len(pro_args) if pro_args else 0,
            'opposition_strength': sum(arg.strength for arg in con_args) / len(con_args) if con_args else 0,
            'total_arguments': len(arguments),
            'strongest_pro_argument': best_pro.content[:200] + '...' if best_pro else None,
            'strongest_con_argument': best_con.content[:200] + '...' if best_con else None,
            'parameters': self._extract_parameters_from_debate(arguments, winner),
            'recommendations': self._extract_recommendations(arguments, winner)
        }

        return consensus

    def _extract_parameters_from_debate(self, arguments: List[DebateArgument],
                                        winner: str) -> Dict[str, float]:
        """从辩论中提取参数"""
        parameters = {}

        # 分析论点中的参数建议
        param_patterns = {
            'spectral_threshold': ['谱.*?阈.*?值', '频.*?门.*?槛'],
            'laplacian_weight': ['拉普拉斯.*?权.*?重', '平滑.*?系数'],
            'learning_rate': ['学习.*?率', 'lr', 'learning rate']
        }

        for arg in arguments:
            content = arg.content

            for param_key, patterns in param_patterns.items():
                for pattern in patterns:
                    import re
                    match = re.search(pattern + r'.*?([0-9.]+)', content, re.IGNORECASE)
                    if match:
                        try:
                            value = float(match.group(1))

                            # 根据论点强度和立场调整权重
                            if arg.role.value == winner:
                                weight = arg.strength
                            else:
                                weight = arg.strength * 0.5  # 对立立场权重减半

                            if param_key not in parameters:
                                parameters[param_key] = {'values': [], 'weights': []}

                            parameters[param_key]['values'].append(value)
                            parameters[param_key]['weights'].append(weight)
                        except:
                            pass

        # 计算加权平均值
        final_params = {}
        for param_key, data in parameters.items():
            if data['values']:
                weighted_sum = sum(v * w for v, w in zip(data['values'], data['weights']))
                total_weight = sum(data['weights'])
                final_params[param_key] = weighted_sum / total_weight

        # 默认值
        defaults = {
            'spectral_threshold': 0.5,
            'laplacian_weight': 0.01,
            'learning_rate_multiplier': 1.0
        }

        for key, default in defaults.items():
            if key not in final_params:
                final_params[key] = default

        return final_params

    def _extract_recommendations(self, arguments: List[DebateArgument],
                                 winner: str) -> List[str]:
        """从辩论中提取建议"""
        recommendations = []

        # 收集所有主张
        all_claims = []
        for arg in arguments:
            all_claims.extend(arg.claims)

        # 去重和排序
        unique_claims = list(set(all_claims))

        # 根据胜方偏好排序
        if winner == 'proposition':
            # 优先考虑激进和创新建议
            for claim in unique_claims:
                if any(keyword in claim for keyword in ['创新', '突破', '优化', '改进']):
                    recommendations.append(claim)
        elif winner == 'opposition':
            # 优先考虑保守和稳健建议
            for claim in unique_claims:
                if any(keyword in claim for keyword in ['谨慎', '稳定', '风险', '验证']):
                    recommendations.append(claim)
        else:
            # 平衡考虑
            recommendations = unique_claims[:5]  # 取前5个

        return recommendations[:5]  # 最多5个建议

    def _calculate_debate_metrics(self, rounds: List[DebateRound]) -> Dict[str, Any]:
        """计算辩论指标"""
        total_arguments = sum(len(round_obj.arguments) for round_obj in rounds)

        if total_arguments == 0:
            return {}

        # 收集所有论点
        all_arguments = []
        for round_obj in rounds:
            all_arguments.extend(round_obj.arguments)

        # 计算指标
        metrics = {
            'total_rounds': len(rounds),
            'total_arguments': total_arguments,
            'avg_argument_strength': sum(arg.strength for arg in all_arguments) / total_arguments,
            'total_fallacies': sum(len(arg.fallacies) for arg in all_arguments),
            'pro_argument_count': sum(1 for arg in all_arguments if arg.role == DebateRole.PROPOSITION),
            'con_argument_count': sum(1 for arg in all_arguments if arg.role == DebateRole.OPPOSITION),
            'strongest_argument': max(all_arguments, key=lambda x: x.strength).strength if all_arguments else 0,
            'weakest_argument': min(all_arguments, key=lambda x: x.strength).strength if all_arguments else 0
        }

        return metrics

    def get_debate_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取辩论历史"""
        history = []

        for debate_id, result in list(self.debate_history.items())[-limit:]:
            history.append({
                'id': debate_id,
                'topic': result.topic,
                'winner': result.winner,
                'total_rounds': len(result.rounds),
                'total_arguments': result.metrics.get('total_arguments', 0),
                'consensus_reached': result.consensus is not None
            })

        return history

    def get_agent_performance(self) -> Dict[str, Dict[str, Any]]:
        """获取智能体表现统计"""
        performance = {}

        for agent_key, agent in self.agents.items():
            perf = agent['performance']
            performance[agent['name']] = {
                'role': agent['role'].value,
                'arguments_made': perf['arguments_made'],
                'avg_argument_strength': perf['argument_strength_avg'],
                'fallacies_per_argument': (
                    perf['fallacies_detected'] / perf['arguments_made']
                    if perf['arguments_made'] > 0 else 0
                ),
                'style': agent['style'],
                'bias': agent['bias']
            }

        return performance


# 使用示例
async def main():
    """辩论系统使用示例"""
    import yaml

    # 加载配置
    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 创建辩论系统
    debate_system = DebateSystem(config)

    # 准备辩论议题
    topic = "是否应该提高ISTR网络的谱门控阈值以提升性能？"
    context = {
        'features': {
            'statistics': {'mean': 0.1, 'std': 0.5},
            'frequency': {'dominant_frequency': 12}
        },
        'metrics': {'mse': 0.25, 'mae': 0.35},
        'current_params': {
            'spectral_threshold': 0.5,
            'laplacian_weight': 0.01
        }
    }

    # 执行辩论
    result = await debate_system.conduct_debate(topic, context)

    print(f"辩论结果：胜方 - {result.winner}")
    print(f"共识参数：{json.dumps(result.consensus['parameters'], indent=2)}")

    # 查看智能体表现
    performance = debate_system.get_agent_performance()
    for agent_name, perf in performance.items():
        print(f"{agent_name}: {perf['avg_argument_strength']:.2f} 强度")


if __name__ == "__main__":
    asyncio.run(main())