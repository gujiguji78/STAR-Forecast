"""
实验管理器 - 统一的实验跟踪、管理和分析系统
支持实验配置、版本控制、结果比较和知识库构建
"""
import json
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import logging
import hashlib
import pickle
import shutil
from enum import Enum
import uuid
import sqlite3
from contextlib import contextmanager
import itertools
import warnings

warnings.filterwarnings('ignore')

# 导入项目模块
import sys

sys.path.append('..')
from training.trainer import STARForecastTrainer
from experiments.ablation_study import AblationStudyManager, AblationComparison
from experiments.baseline_comparison import BaselineComparison, BaselineModel


class ExperimentStatus(Enum):
    """实验状态"""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentType(Enum):
    """实验类型"""
    FULL_TRAINING = "full_training"  # 完整训练
    ABLATION_STUDY = "ablation_study"  # 消融实验
    BASELINE_COMPARISON = "baseline_comparison"  # 基线比较
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"  # 超参数调优
    TRANSFER_LEARNING = "transfer_learning"  # 迁移学习
    ROBUSTNESS_TEST = "robustness_test"  # 鲁棒性测试


@dataclass
class ExperimentMetadata:
    """实验元数据"""
    experiment_id: str
    name: str
    description: str
    experiment_type: ExperimentType
    status: ExperimentStatus
    config: Dict[str, Any]
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    git_commit: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    parent_experiment: Optional[str] = None  # 父实验ID
    dependencies: List[str] = field(default_factory=list)  # 依赖的实验

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'experiment_id': self.experiment_id,
            'name': self.name,
            'description': self.description,
            'experiment_type': self.experiment_type.value,
            'status': self.status.value,
            'config': self.config,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_seconds': self.duration_seconds,
            'git_commit': self.git_commit,
            'tags': self.tags,
            'parent_experiment': self.parent_experiment,
            'dependencies': self.dependencies
        }


@dataclass
class ExperimentResult:
    """实验结果"""
    experiment_id: str
    metrics: Dict[str, Any]
    artifacts: Dict[str, str]  # 路径 -> 描述
    logs: Dict[str, Any]
    models: Dict[str, str]  # 模型名称 -> 模型路径
    visualizations: List[str]  # 可视化文件路径
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'experiment_id': self.experiment_id,
            'metrics': self.metrics,
            'artifacts': self.artifacts,
            'logs': self.logs,
            'models': self.models,
            'visualizations': self.visualizations,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class ExperimentComparison:
    """实验对比"""
    comparison_id: str
    experiment_ids: List[str]
    comparison_metrics: Dict[str, Dict[str, Any]]  # metric -> experiment_id -> value
    ranking: Dict[str, List[str]]  # metric -> 排序后的experiment_id列表
    insights: List[str]
    created_at: datetime = field(default_factory=datetime.now)

    def to_dataframe(self) -> pd.DataFrame:
        """转换为DataFrame"""
        rows = []

        for metric, experiment_values in self.comparison_metrics.items():
            for exp_id, value in experiment_values.items():
                rows.append({
                    'experiment_id': exp_id,
                    'metric': metric,
                    'value': value
                })

        return pd.DataFrame(rows)


class KnowledgeBaseEntry:
    """知识库条目"""

    def __init__(self,
                 knowledge_id: str,
                 title: str,
                 content: str,
                 experiment_ids: List[str],
                 evidence: Dict[str, Any],
                 confidence: float = 0.5,
                 tags: List[str] = None):
        self.knowledge_id = knowledge_id
        self.title = title
        self.content = content
        self.experiment_ids = experiment_ids
        self.evidence = evidence
        self.confidence = confidence
        self.tags = tags or []
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.citation_count = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'knowledge_id': self.knowledge_id,
            'title': self.title,
            'content': self.content,
            'experiment_ids': self.experiment_ids,
            'evidence': self.evidence,
            'confidence': self.confidence,
            'tags': self.tags,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'citation_count': self.citation_count
        }


class ExperimentDatabase:
    """实验数据库（SQLite）"""

    def __init__(self, db_path: str = "./experiments/experiments.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()

    @contextmanager
    def _get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_database(self):
        """初始化数据库"""
        with self._get_connection() as conn:
            # 实验元数据表
            conn.execute("""
                         CREATE TABLE IF NOT EXISTS experiments
                         (
                             experiment_id
                             TEXT
                             PRIMARY
                             KEY,
                             name
                             TEXT
                             NOT
                             NULL,
                             description
                             TEXT,
                             experiment_type
                             TEXT
                             NOT
                             NULL,
                             status
                             TEXT
                             NOT
                             NULL,
                             config
                             TEXT
                             NOT
                             NULL,
                             created_by
                             TEXT
                             DEFAULT
                             'system',
                             created_at
                             DATETIME
                             NOT
                             NULL,
                             started_at
                             DATETIME,
                             completed_at
                             DATETIME,
                             duration_seconds
                             REAL,
                             git_commit
                             TEXT,
                             tags
                             TEXT,
                             parent_experiment
                             TEXT,
                             dependencies
                             TEXT
                         )
                         """)

            # 实验结果表
            conn.execute("""
                         CREATE TABLE IF NOT EXISTS experiment_results
                         (
                             result_id
                             TEXT
                             PRIMARY
                             KEY,
                             experiment_id
                             TEXT
                             NOT
                             NULL,
                             metrics
                             TEXT
                             NOT
                             NULL,
                             artifacts
                             TEXT
                             NOT
                             NULL,
                             logs
                             TEXT
                             NOT
                             NULL,
                             models
                             TEXT
                             NOT
                             NULL,
                             visualizations
                             TEXT
                             NOT
                             NULL,
                             created_at
                             DATETIME
                             NOT
                             NULL,
                             FOREIGN
                             KEY
                         (
                             experiment_id
                         ) REFERENCES experiments
                         (
                             experiment_id
                         )
                             )
                         """)

            # 知识库表
            conn.execute("""
                         CREATE TABLE IF NOT EXISTS knowledge_base
                         (
                             knowledge_id
                             TEXT
                             PRIMARY
                             KEY,
                             title
                             TEXT
                             NOT
                             NULL,
                             content
                             TEXT
                             NOT
                             NULL,
                             experiment_ids
                             TEXT
                             NOT
                             NULL,
                             evidence
                             TEXT
                             NOT
                             NULL,
                             confidence
                             REAL
                             DEFAULT
                             0.5,
                             tags
                             TEXT,
                             created_at
                             DATETIME
                             NOT
                             NULL,
                             updated_at
                             DATETIME
                             NOT
                             NULL,
                             citation_count
                             INTEGER
                             DEFAULT
                             0
                         )
                         """)

            # 创建索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_type ON experiments (experiment_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_status ON experiments (status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_created ON experiments (created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_tags ON knowledge_base (tags)")

            conn.commit()

    def save_experiment(self, metadata: ExperimentMetadata):
        """保存实验元数据"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO experiments 
                (experiment_id, name, description, experiment_type, status, 
                 config, created_by, created_at, started_at, completed_at,
                 duration_seconds, git_commit, tags, parent_experiment, dependencies)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metadata.experiment_id,
                metadata.name,
                metadata.description,
                metadata.experiment_type.value,
                metadata.status.value,
                json.dumps(metadata.config, ensure_ascii=False),
                metadata.created_by,
                metadata.created_at.isoformat(),
                metadata.started_at.isoformat() if metadata.started_at else None,
                metadata.completed_at.isoformat() if metadata.completed_at else None,
                metadata.duration_seconds,
                metadata.git_commit,
                json.dumps(metadata.tags, ensure_ascii=False),
                metadata.parent_experiment,
                json.dumps(metadata.dependencies, ensure_ascii=False)
            ))

            conn.commit()

    def save_experiment_result(self, result: ExperimentResult):
        """保存实验结果"""
        result_id = hashlib.md5(
            f"{result.experiment_id}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        with self._get_connection() as conn:
            conn.execute("""
                         INSERT INTO experiment_results
                         (result_id, experiment_id, metrics, artifacts, logs, models, visualizations, created_at)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                         """, (
                             result_id,
                             result.experiment_id,
                             json.dumps(result.metrics, ensure_ascii=False),
                             json.dumps(result.artifacts, ensure_ascii=False),
                             json.dumps(result.logs, ensure_ascii=False),
                             json.dumps(result.models, ensure_ascii=False),
                             json.dumps(result.visualizations, ensure_ascii=False),
                             result.created_at.isoformat()
                         ))

            conn.commit()

    def save_knowledge(self, knowledge: KnowledgeBaseEntry):
        """保存知识库条目"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO knowledge_base 
                (knowledge_id, title, content, experiment_ids, evidence, 
                 confidence, tags, created_at, updated_at, citation_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                knowledge.knowledge_id,
                knowledge.title,
                knowledge.content,
                json.dumps(knowledge.experiment_ids, ensure_ascii=False),
                json.dumps(knowledge.evidence, ensure_ascii=False),
                knowledge.confidence,
                json.dumps(knowledge.tags, ensure_ascii=False),
                knowledge.created_at.isoformat(),
                knowledge.updated_at.isoformat(),
                knowledge.citation_count
            ))

            conn.commit()

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """获取实验"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            # 解析数据
            metadata = ExperimentMetadata(
                experiment_id=row['experiment_id'],
                name=row['name'],
                description=row['description'],
                experiment_type=ExperimentType(row['experiment_type']),
                status=ExperimentStatus(row['status']),
                config=json.loads(row['config']),
                created_by=row['created_by'],
                created_at=datetime.fromisoformat(row['created_at']),
                started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
                completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
                duration_seconds=row['duration_seconds'],
                git_commit=row['git_commit'],
                tags=json.loads(row['tags']) if row['tags'] else [],
                parent_experiment=row['parent_experiment'],
                dependencies=json.loads(row['dependencies']) if row['dependencies'] else []
            )

            return metadata

    def get_experiment_result(self, experiment_id: str) -> Optional[ExperimentResult]:
        """获取实验结果"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM experiment_results WHERE experiment_id = ? ORDER BY created_at DESC LIMIT 1",
                (experiment_id,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            # 解析数据
            result = ExperimentResult(
                experiment_id=row['experiment_id'],
                metrics=json.loads(row['metrics']),
                artifacts=json.loads(row['artifacts']),
                logs=json.loads(row['logs']),
                models=json.loads(row['models']),
                visualizations=json.loads(row['visualizations']),
                created_at=datetime.fromisoformat(row['created_at'])
            )

            return result

    def search_experiments(self,
                           experiment_type: Optional[ExperimentType] = None,
                           status: Optional[ExperimentStatus] = None,
                           tags: Optional[List[str]] = None,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           limit: int = 100) -> List[ExperimentMetadata]:
        """搜索实验"""
        conditions = []
        params = []

        if experiment_type:
            conditions.append("experiment_type = ?")
            params.append(experiment_type.value)

        if status:
            conditions.append("status = ?")
            params.append(status.value)

        if start_date:
            conditions.append("created_at >= ?")
            params.append(start_date.isoformat())

        if end_date:
            conditions.append("created_at <= ?")
            params.append(end_date.isoformat())

        if tags:
            # 简单的标签搜索（实际应使用全文搜索）
            tag_conditions = []
            for tag in tags:
                tag_conditions.append("tags LIKE ?")
                params.append(f'%{tag}%')

            if tag_conditions:
                conditions.append(f"({' OR '.join(tag_conditions)})")

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = f"""
            SELECT * FROM experiments 
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

            experiments = []
            for row in rows:
                metadata = ExperimentMetadata(
                    experiment_id=row['experiment_id'],
                    name=row['name'],
                    description=row['description'],
                    experiment_type=ExperimentType(row['experiment_type']),
                    status=ExperimentStatus(row['status']),
                    config=json.loads(row['config']),
                    created_by=row['created_by'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
                    completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
                    duration_seconds=row['duration_seconds'],
                    git_commit=row['git_commit'],
                    tags=json.loads(row['tags']) if row['tags'] else [],
                    parent_experiment=row['parent_experiment'],
                    dependencies=json.loads(row['dependencies']) if row['dependencies'] else []
                )
                experiments.append(metadata)

            return experiments

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        with self._get_connection() as conn:
            stats = {}

            # 实验统计
            cursor = conn.execute("SELECT COUNT(*) as total FROM experiments")
            stats['total_experiments'] = cursor.fetchone()['total']

            cursor = conn.execute("SELECT experiment_type, COUNT(*) as count FROM experiments GROUP BY experiment_type")
            stats['by_type'] = {row['experiment_type']: row['count'] for row in cursor.fetchall()}

            cursor = conn.execute("SELECT status, COUNT(*) as count FROM experiments GROUP BY status")
            stats['by_status'] = {row['status']: row['count'] for row in cursor.fetchall()}

            # 知识库统计
            cursor = conn.execute("SELECT COUNT(*) as total FROM knowledge_base")
            stats['total_knowledge'] = cursor.fetchone()['total']

            return stats


class ExperimentManager:
    """
    实验管理器

    功能：
    1. 统一的实验生命周期管理
    2. 实验版本控制和复现
    3. 实验结果跟踪和分析
    4. 知识库构建和检索
    5. 实验对比和洞察生成
    """

    def __init__(self,
                 base_config_path: str = "./config.yaml",
                 experiments_root: str = "./experiments"):

        self.base_config_path = Path(base_config_path)
        self.experiments_root = Path(experiments_root)

        # 创建目录结构
        self.experiments_root.mkdir(parents=True, exist_ok=True)
        (self.experiments_root / "configs").mkdir(exist_ok=True)
        (self.experiments_root / "results").mkdir(exist_ok=True)
        (self.experiments_root / "models").mkdir(exist_ok=True)
        (self.experiments_root / "logs").mkdir(exist_ok=True)
        (self.experiments_root / "visualizations").mkdir(exist_ok=True)
        (self.experiments_root / "reports").mkdir(exist_ok=True)

        # 初始化组件
        self.db = ExperimentDatabase(self.experiments_root / "experiments.db")
        self.logger = logging.getLogger(__name__)

        # 实验注册表
        self.active_experiments: Dict[str, Any] = {}

        # 加载基础配置
        self.base_config = self._load_config(base_config_path)

        self.logger.info("🧪 实验管理器初始化完成")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def _get_git_commit(self) -> Optional[str]:
        """获取当前Git提交"""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            if result.returncode == 0:
                return result.stdout.strip()[:8]
        except:
            pass
        return None

    def create_experiment(self,
                          name: str,
                          description: str,
                          experiment_type: ExperimentType,
                          config: Dict[str, Any] = None,
                          tags: List[str] = None,
                          parent_experiment: str = None,
                          dependencies: List[str] = None) -> ExperimentMetadata:
        """创建新实验"""
        # 生成实验ID
        experiment_id = f"{experiment_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # 合并配置
        if config is None:
            config = self.base_config.copy()
        else:
            # 深拷贝基础配置并更新
            import copy
            merged_config = copy.deepcopy(self.base_config)

            def update_dict(d, u):
                for k, v in u.items():
                    if isinstance(v, dict):
                        d[k] = update_dict(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d

            config = update_dict(merged_config, config)

        # 创建实验元数据
        metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            name=name,
            description=description,
            experiment_type=experiment_type,
            status=ExperimentStatus.CREATED,
            config=config,
            tags=tags or [],
            parent_experiment=parent_experiment,
            dependencies=dependencies or [],
            git_commit=self._get_git_commit()
        )

        # 保存到数据库
        self.db.save_experiment(metadata)

        # 保存配置文件
        config_path = self.experiments_root / "configs" / f"{experiment_id}.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        self.logger.info(f"📝 创建实验: {experiment_id}")
        self.logger.info(f"   名称: {name}")
        self.logger.info(f"   类型: {experiment_type.value}")

        return metadata

    def run_full_training(self,
                          experiment_id: str,
                          data_path: str = "./data/ETTh1.csv",
                          save_model: bool = True) -> ExperimentResult:
        """运行完整训练实验"""
        # 获取实验元数据
        metadata = self.db.get_experiment(experiment_id)
        if not metadata:
            raise ValueError(f"实验不存在: {experiment_id}")

        # 更新状态为运行中
        metadata.status = ExperimentStatus.RUNNING
        metadata.started_at = datetime.now()
        self.db.save_experiment(metadata)

        self.logger.info(f"🚀 开始运行实验: {experiment_id}")

        try:
            # 创建训练器
            trainer = STARForecastTrainer(metadata.config)
            trainer.build_models()
            trainer.build_optimizer()
            trainer.initialize_agents()

            # 训练模型
            test_metrics = trainer.train(data_path)

            # 更新实验状态
            metadata.status = ExperimentStatus.COMPLETED
            metadata.completed_at = datetime.now()
            metadata.duration_seconds = (metadata.completed_at - metadata.started_at).total_seconds()
            self.db.save_experiment(metadata)

            # 保存模型
            models = {}
            if save_model:
                model_path = self.experiments_root / "models" / f"{experiment_id}_model.pth"
                trainer._save_checkpoint(trainer.current_epoch, is_best=True)
                models['best_model'] = str(model_path)

            # 收集日志和可视化
            logs = {
                'training_history': getattr(trainer, 'training_history', {}),
                'agent_decisions': getattr(trainer.agent_client, 'stats', {}) if hasattr(trainer,
                                                                                         'agent_client') else {},
                'autogen_conversations': len(getattr(trainer.autogen_system, 'conversation_history', {}))
                if hasattr(trainer, 'autogen_system') else 0
            }

            # 生成可视化
            visualizations = self._generate_training_visualizations(trainer, experiment_id)

            # 保存结果
            result = ExperimentResult(
                experiment_id=experiment_id,
                metrics=test_metrics,
                artifacts={
                    'config': str(self.experiments_root / "configs" / f"{experiment_id}.yaml"),
                    'logs': str(self.experiments_root / "logs" / f"{experiment_id}.log")
                },
                logs=logs,
                models=models,
                visualizations=visualizations
            )

            self.db.save_experiment_result(result)

            # 生成报告
            self._generate_experiment_report(metadata, result)

            # 提取知识
            self._extract_knowledge_from_experiment(metadata, result)

            self.logger.info(f"✅ 实验完成: {experiment_id}")
            self.logger.info(f"   结果: {test_metrics}")

            return result

        except Exception as e:
            # 更新为失败状态
            metadata.status = ExperimentStatus.FAILED
            metadata.completed_at = datetime.now()
            self.db.save_experiment(metadata)

            self.logger.error(f"❌ 实验失败: {experiment_id}")
            self.logger.error(f"   错误: {e}")

            import traceback
            traceback.print_exc()

            raise

    def run_ablation_study(self,
                           experiment_id: str,
                           variants: List[str] = None,
                           data_path: str = "./data/ETTh1.csv") -> ExperimentResult:
        """运行消融实验"""
        # 获取实验元数据
        metadata = self.db.get_experiment(experiment_id)
        if not metadata:
            raise ValueError(f"实验不存在: {experiment_id}")

        # 更新状态
        metadata.status = ExperimentStatus.RUNNING
        metadata.started_at = datetime.now()
        self.db.save_experiment(metadata)

        self.logger.info(f"🔬 开始消融实验: {experiment_id}")

        try:
            # 创建消融实验管理器
            ablation_manager = AblationStudyManager(self.base_config_path)

            # 转换变体参数
            from experiments.ablation_study import AblationVariant
            ablation_variants = None
            if variants:
                ablation_variants = [AblationVariant(v) for v in variants]

            # 运行消融实验
            comparison = ablation_manager.run_ablation_experiment(
                variants=ablation_variants,
                data_path=data_path,
                experiment_name=experiment_id
            )

            # 更新实验状态
            metadata.status = ExperimentStatus.COMPLETED
            metadata.completed_at = datetime.now()
            metadata.duration_seconds = (metadata.completed_at - metadata.started_at).total_seconds()
            self.db.save_experiment(metadata)

            # 准备结果
            metrics = {
                'ablation_summary': comparison.summary_stats,
                'statistical_tests': comparison.statistical_tests
            }

            # 收集可视化
            vis_dir = Path("./experiments/ablation_results/visualizations")
            visualizations = []
            if vis_dir.exists():
                for vis_file in vis_dir.glob(f"{experiment_id}*.png"):
                    visualizations.append(str(vis_file))

            # 保存结果
            result = ExperimentResult(
                experiment_id=experiment_id,
                metrics=metrics,
                artifacts={
                    'comparison_json': str(Path("./experiments/ablation_results") / f"{experiment_id}.json"),
                    'comparison_csv': str(Path("./experiments/ablation_results") / f"{experiment_id}.csv"),
                    'comparison_pickle': str(Path("./experiments/ablation_results") / f"{experiment_id}.pkl")
                },
                logs={'ablation_comparison': comparison.to_dict()},
                models={},
                visualizations=visualizations
            )

            self.db.save_experiment_result(result)

            # 提取知识
            self._extract_knowledge_from_ablation(comparison, experiment_id)

            self.logger.info(f"✅ 消融实验完成: {experiment_id}")

            return result

        except Exception as e:
            metadata.status = ExperimentStatus.FAILED
            metadata.completed_at = datetime.now()
            self.db.save_experiment(metadata)

            self.logger.error(f"❌ 消融实验失败: {experiment_id}")
            self.logger.error(f"   错误: {e}")

            raise

    def run_baseline_comparison(self,
                                experiment_id: str,
                                baselines: List[str] = None,
                                data_path: str = "./data/ETTh1.csv") -> ExperimentResult:
        """运行基线比较实验"""
        # 获取实验元数据
        metadata = self.db.get_experiment(experiment_id)
        if not metadata:
            raise ValueError(f"实验不存在: {experiment_id}")

        # 更新状态
        metadata.status = ExperimentStatus.RUNNING
        metadata.started_at = datetime.now()
        self.db.save_experiment(metadata)

        self.logger.info(f"📊 开始基线比较: {experiment_id}")

        try:
            # 创建基线比较管理器
            from experiments.baseline_comparison import BaselineComparisonManager
            baseline_manager = BaselineComparisonManager(self.base_config_path)

            # 运行基线比较
            comparison = baseline_manager.run_baseline_comparison(
                baselines=baselines,
                data_path=data_path,
                experiment_name=experiment_id
            )

            # 更新实验状态
            metadata.status = ExperimentStatus.COMPLETED
            metadata.completed_at = datetime.now()
            metadata.duration_seconds = (metadata.completed_at - metadata.started_at).total_seconds()
            self.db.save_experiment(metadata)

            # 准备结果
            metrics = {
                'baseline_performance': comparison.performance_metrics,
                'statistical_comparison': comparison.statistical_comparison
            }

            # 收集可视化
            vis_dir = Path("./experiments/baseline_results/visualizations")
            visualizations = []
            if vis_dir.exists():
                for vis_file in vis_dir.glob(f"{experiment_id}*.png"):
                    visualizations.append(str(vis_file))

            # 保存结果
            result = ExperimentResult(
                experiment_id=experiment_id,
                metrics=metrics,
                artifacts={
                    'comparison_json': str(Path("./experiments/baseline_results") / f"{experiment_id}.json"),
                    'comparison_csv': str(Path("./experiments/baseline_results") / f"{experiment_id}.csv")
                },
                logs={'baseline_comparison': comparison.to_dict()},
                models={},
                visualizations=visualizations
            )

            self.db.save_experiment_result(result)

            self.logger.info(f"✅ 基线比较完成: {experiment_id}")

            return result

        except Exception as e:
            metadata.status = ExperimentStatus.FAILED
            metadata.completed_at = datetime.now()
            self.db.save_experiment(metadata)

            self.logger.error(f"❌ 基线比较失败: {experiment_id}")
            self.logger.error(f"   错误: {e}")

            raise

    def _generate_training_visualizations(self, trainer, experiment_id: str) -> List[str]:
        """生成训练可视化"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            vis_dir = self.experiments_root / "visualizations" / experiment_id
            vis_dir.mkdir(parents=True, exist_ok=True)

            visualizations = []

            # 1. 训练损失曲线
            if hasattr(trainer, 'training_history') and trainer.training_history:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                # 训练损失
                if 'train_loss' in trainer.training_history:
                    ax = axes[0, 0]
                    ax.plot(trainer.training_history['train_loss'], label='Train Loss')
                    if 'val_loss' in trainer.training_history:
                        ax.plot(trainer.training_history['val_loss'], label='Val Loss')
                    ax.set_title('Training and Validation Loss')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Loss')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                # 训练奖励（如果有）
                if 'train_reward' in trainer.training_history:
                    ax = axes[0, 1]
                    ax.plot(trainer.training_history['train_reward'])
                    ax.set_title('Training Reward')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Reward')
                    ax.grid(True, alpha=0.3)

                # 验证指标
                if 'val_mse' in trainer.training_history:
                    ax = axes[1, 0]
                    ax.plot(trainer.training_history['val_mse'], label='MSE')
                    if 'val_mae' in trainer.training_history:
                        ax.plot(trainer.training_history['val_mae'], label='MAE')
                    ax.set_title('Validation Metrics')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Metric Value')
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                # 学习率
                if hasattr(trainer, 'scheduler') and trainer.scheduler:
                    ax = axes[1, 1]
                    lr_history = []
                    for epoch in range(trainer.current_epoch):
                        lr_history.append(trainer.optimizer.param_groups[0]['lr'])
                        trainer.scheduler.step()

                    ax.plot(lr_history)
                    ax.set_title('Learning Rate Schedule')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Learning Rate')
                    ax.grid(True, alpha=0.3)

                plt.tight_layout()
                vis_path = vis_dir / "training_history.png"
                plt.savefig(vis_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualizations.append(str(vis_path))

            return visualizations

        except Exception as e:
            self.logger.warning(f"可视化生成失败: {e}")
            return []

    def _generate_experiment_report(self,
                                    metadata: ExperimentMetadata,
                                    result: ExperimentResult):
        """生成实验报告"""
        try:
            report_dir = self.experiments_root / "reports" / metadata.experiment_id
            report_dir.mkdir(parents=True, exist_ok=True)

            # 生成Markdown报告
            report_path = report_dir / "report.md"

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"# 实验报告: {metadata.name}\n\n")
                f.write(f"**实验ID**: {metadata.experiment_id}\n\n")
                f.write(f"**创建时间**: {metadata.created_at}\n\n")
                f.write(f"**完成时间**: {metadata.completed_at}\n\n")
                f.write(f"**持续时间**: {metadata.duration_seconds:.1f} 秒\n\n")
                f.write(f"**实验类型**: {metadata.experiment_type.value}\n\n")
                f.write(f"**描述**: {metadata.description}\n\n")

                # 标签
                if metadata.tags:
                    f.write(f"**标签**: {', '.join(metadata.tags)}\n\n")

                # 实验结果
                f.write("## 实验结果\n\n")

                for metric_name, metric_value in result.metrics.items():
                    if isinstance(metric_value, dict):
                        f.write(f"### {metric_name}\n\n")
                        for k, v in metric_value.items():
                            if isinstance(v, (int, float)):
                                f.write(f"- {k}: {v:.6f}\n")
                            else:
                                f.write(f"- {k}: {v}\n")
                        f.write("\n")
                    elif isinstance(metric_value, (int, float)):
                        f.write(f"- **{metric_name}**: {metric_value:.6f}\n")
                    else:
                        f.write(f"- **{metric_name}**: {metric_value}\n")

                # 配置摘要
                f.write("\n## 配置摘要\n\n")

                # 提取关键配置
                key_configs = [
                    ('data', ['seq_len', 'pred_len']),
                    ('training', ['epochs', 'learning_rate', 'batch_size']),
                    ('istr', ['hidden_dim', 'trainable_ratio']),
                    ('autogen', ['check_interval', 'max_rounds']),
                    ('agent_lightning', ['lr', 'gamma'])
                ]

                for section, keys in key_configs:
                    if section in metadata.config:
                        f.write(f"### {section}\n\n")
                        for key in keys:
                            if key in metadata.config[section]:
                                value = metadata.config[section][key]
                                f.write(f"- {key}: {value}\n")
                        f.write("\n")

                # 可视化链接
                if result.visualizations:
                    f.write("## 可视化\n\n")
                    for vis_path in result.visualizations:
                        vis_name = Path(vis_path).name
                        f.write(f"![{vis_name}]({vis_path})\n\n")

                # 总结
                f.write("## 总结\n\n")
                f.write("实验已成功完成。\n\n")

            # 生成HTML报告
            self._generate_html_report(report_path)

            self.logger.info(f"📄 实验报告保存到: {report_path}")

        except Exception as e:
            self.logger.warning(f"报告生成失败: {e}")

    def _generate_html_report(self, markdown_path: Path):
        """生成HTML报告"""
        try:
            import markdown

            with open(markdown_path, 'r', encoding='utf-8') as f:
                md_content = f.read()

            html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

            # 添加HTML模板
            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>实验报告</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                    h1, h2, h3 {{ color: #333; }}
                    code {{ background: #f4f4f4; padding: 2px 6px; }}
                    pre {{ background: #f4f4f4; padding: 10px; overflow: auto; }}
                    img {{ max-width: 100%; height: auto; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """

            html_path = markdown_path.with_suffix('.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_template)

        except Exception as e:
            self.logger.warning(f"HTML报告生成失败: {e}")

    def _extract_knowledge_from_experiment(self,
                                           metadata: ExperimentMetadata,
                                           result: ExperimentResult):
        """从实验中提取知识"""
        try:
            # 提取关键发现
            knowledge_id = f"knowledge_{metadata.experiment_id}_{uuid.uuid4().hex[:8]}"

            # 分析结果
            insights = []

            # 检查是否存在过拟合
            if ('train_loss' in result.logs.get('training_history', {}) and
                    'val_loss' in result.logs.get('training_history', {})):
                train_loss = result.logs['training_history']['train_loss'][-1] if result.logs['training_history'][
                    'train_loss'] else 0
                val_loss = result.logs['training_history']['val_loss'][-1] if result.logs['training_history'][
                    'val_loss'] else 0

                if val_loss > train_loss * 1.2:  # 验证损失比训练损失高20%
                    insights.append("发现过拟合迹象：验证损失显著高于训练损失")
                elif train_loss > val_loss * 1.2:
                    insights.append("发现欠拟合迹象：训练损失显著高于验证损失")

            # 检查训练稳定性
            if 'train_loss' in result.logs.get('training_history', {}):
                losses = result.logs['training_history']['train_loss']
                if len(losses) > 10:
                    final_loss = losses[-1]
                    initial_loss = losses[0]
                    improvement = (initial_loss - final_loss) / initial_loss

                    if improvement > 0.5:
                        insights.append(f"训练效果显著：损失降低了{improvement:.1%}")

            # 检查智能体交互
            if 'agent_decisions' in result.logs:
                decisions = result.logs['agent_decisions']
                if isinstance(decisions, dict) and 'total_requests' in decisions:
                    if decisions['total_requests'] > 0:
                        success_rate = decisions.get('successful_requests', 0) / decisions['total_requests']
                        if success_rate < 0.5:
                            insights.append("智能体系统交互成功率较低")

            # 创建知识条目
            if insights:
                knowledge = KnowledgeBaseEntry(
                    knowledge_id=knowledge_id,
                    title=f"实验发现: {metadata.name}",
                    content="\n".join(insights),
                    experiment_ids=[metadata.experiment_id],
                    evidence={
                        'metrics': result.metrics,
                        'config_summary': {
                            'learning_rate': metadata.config.get('training', {}).get('learning_rate'),
                            'batch_size': metadata.config.get('data', {}).get('batch_size'),
                            'epochs': metadata.config.get('training', {}).get('epochs')
                        }
                    },
                    confidence=0.7,
                    tags=metadata.tags + ['experiment_finding']
                )

                self.db.save_knowledge(knowledge)
                self.logger.info(f"🧠 提取知识: {knowledge_id}")

        except Exception as e:
            self.logger.warning(f"知识提取失败: {e}")

    def _extract_knowledge_from_ablation(self,
                                         comparison: AblationComparison,
                                         experiment_id: str):
        """从消融实验中提取知识"""
        try:
            # 分析最佳和最差变体
            best_variant = None
            best_mse = float('inf')
            worst_variant = None
            worst_mse = 0

            for variant, stats in comparison.summary_stats.items():
                mse = stats['mean']
                if mse < best_mse:
                    best_mse = mse
                    best_variant = variant
                if mse > worst_mse:
                    worst_mse = mse
                    worst_variant = variant

            # 创建知识条目
            if best_variant and worst_variant:
                knowledge_id = f"knowledge_ablation_{experiment_id}_{uuid.uuid4().hex[:8]}"

                content = f"""
消融实验发现：
1. 最佳性能变体：{best_variant} (MSE: {best_mse:.6f})
2. 最差性能变体：{worst_variant} (MSE: {worst_mse:.6f})
3. 性能差异：{worst_mse / best_mse:.1%}

关键洞察：
- {self._get_ablation_insight(best_variant, worst_variant)}
"""

                knowledge = KnowledgeBaseEntry(
                    knowledge_id=knowledge_id,
                    title=f"消融实验发现: {experiment_id}",
                    content=content,
                    experiment_ids=[experiment_id],
                    evidence={
                        'best_variant': best_variant,
                        'best_mse': best_mse,
                        'worst_variant': worst_variant,
                        'worst_mse': worst_mse,
                        'statistical_tests': comparison.statistical_tests
                    },
                    confidence=0.8,
                    tags=['ablation_study', 'performance_analysis']
                )

                self.db.save_knowledge(knowledge)
                self.logger.info(f"🧠 提取消融知识: {knowledge_id}")

        except Exception as e:
            self.logger.warning(f"消融知识提取失败: {e}")

    def _get_ablation_insight(self, best_variant: str, worst_variant: str) -> str:
        """获取消融实验洞察"""
        insights = {
            ('full', 'no_autogen'): "AutoGen多智能体系统对性能提升有显著贡献",
            ('full', 'no_agent_lightning'): "Agent Lightning强化学习机制有效提升了模型适应性",
            ('full', 'no_istr'): "ISTR网络（TCN+拉普拉斯）是性能提升的关键",
            ('full', 'no_laplacian'): "拉普拉斯正则化有效防止了过拟合",
            ('full', 'no_spectral_gate'): "谱门控机制增强了特征提取能力",
            ('full', 'frozen_istr'): "ISTR网络的自适应参数调整至关重要",
            ('full', 'single_agent'): "多智能体协同比单智能体更有效",
            ('full', 'no_semantic_reward'): "语义奖励机制提高了强化学习效果",
            ('full', 'simple_baseline'): "完整STAR-Forecast框架显著优于简单基线"
        }

        return insights.get((best_variant, worst_variant),
                            f"{best_variant}比{worst_variant}表现更好")

    def compare_experiments(self, experiment_ids: List[str]) -> ExperimentComparison:
        """比较多个实验"""
        comparison_id = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        comparison_metrics = {}
        ranking = {}

        for exp_id in experiment_ids:
            result = self.db.get_experiment_result(exp_id)
            if result and result.metrics:
                # 提取数值指标
                for metric_name, metric_value in result.metrics.items():
                    if isinstance(metric_value, (int, float)):
                        if metric_name not in comparison_metrics:
                            comparison_metrics[metric_name] = {}
                        comparison_metrics[metric_name][exp_id] = metric_value

        # 生成排名
        for metric_name, experiment_values in comparison_metrics.items():
            # 排序（越小越好对于MSE等损失指标）
            if any('loss' in metric_name.lower() or 'mse' in metric_name.lower() or 'mae' in metric_name.lower()
                   for keyword in ['loss', 'mse', 'mae', 'error']):
                sorted_experiments = sorted(experiment_values.items(), key=lambda x: x[1])
            else:  # 越大越好对于准确率等指标
                sorted_experiments = sorted(experiment_values.items(), key=lambda x: x[1], reverse=True)

            ranking[metric_name] = [exp_id for exp_id, _ in sorted_experiments]

        # 生成洞察
        insights = self._generate_comparison_insights(comparison_metrics, ranking)

        comparison = ExperimentComparison(
            comparison_id=comparison_id,
            experiment_ids=experiment_ids,
            comparison_metrics=comparison_metrics,
            ranking=ranking,
            insights=insights
        )

        return comparison

    def _generate_comparison_insights(self,
                                      comparison_metrics: Dict[str, Dict[str, float]],
                                      ranking: Dict[str, List[str]]) -> List[str]:
        """生成比较洞察"""
        insights = []

        if not comparison_metrics:
            return insights

        # 找出最佳实验
        primary_metric = next(iter(comparison_metrics))
        if primary_metric in ranking and ranking[primary_metric]:
            best_experiment = ranking[primary_metric][0]
            insights.append(f"最佳实验: {best_experiment} (在{primary_metric}上表现最好)")

        # 分析性能差异
        for metric_name, experiment_values in comparison_metrics.items():
            if len(experiment_values) >= 2:
                values = list(experiment_values.values())
                min_val, max_val = min(values), max(values)

                if min_val > 0:
                    ratio = max_val / min_val
                    if ratio > 1.5:
                        insights.append(f"在{metric_name}上，最佳和最差实验性能差异显著 ({ratio:.1f}倍)")

        return insights

    def search_knowledge(self,
                         query: str,
                         tags: List[str] = None,
                         min_confidence: float = 0.0,
                         limit: int = 10) -> List[KnowledgeBaseEntry]:
        """搜索知识库"""
        # 这里实现简单的关键词搜索
        # 实际项目应使用向量搜索或全文搜索

        with self.db._get_connection() as conn:
            conditions = ["(title LIKE ? OR content LIKE ?)"]
            params = [f'%{query}%', f'%{query}%']

            if tags:
                tag_conditions = []
                for tag in tags:
                    tag_conditions.append("tags LIKE ?")
                    params.append(f'%{tag}%')

                if tag_conditions:
                    conditions.append(f"({' OR '.join(tag_conditions)})")

            if min_confidence > 0:
                conditions.append("confidence >= ?")
                params.append(min_confidence)

            where_clause = " AND ".join(conditions)
            sql = f"""
                SELECT * FROM knowledge_base 
                WHERE {where_clause}
                ORDER BY citation_count DESC, confidence DESC
                LIMIT ?
            """
            params.append(limit)

            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

            knowledge_entries = []
            for row in rows:
                entry = KnowledgeBaseEntry(
                    knowledge_id=row['knowledge_id'],
                    title=row['title'],
                    content=row['content'],
                    experiment_ids=json.loads(row['experiment_ids']),
                    evidence=json.loads(row['evidence']),
                    confidence=row['confidence'],
                    tags=json.loads(row['tags']) if row['tags'] else []
                )
                entry.created_at = datetime.fromisoformat(row['created_at'])
                entry.updated_at = datetime.fromisoformat(row['updated_at'])
                entry.citation_count = row['citation_count']

                knowledge_entries.append(entry)

            return knowledge_entries

    def get_experiment_statistics(self) -> Dict[str, Any]:
        """获取实验统计"""
        return self.db.get_statistics()

    def export_experiment(self, experiment_id: str, export_path: str):
        """导出实验"""
        metadata = self.db.get_experiment(experiment_id)
        if not metadata:
            raise ValueError(f"实验不存在: {experiment_id}")

        result = self.db.get_experiment_result(experiment_id)

        # 创建导出目录
        export_dir = Path(export_path) / experiment_id
        export_dir.mkdir(parents=True, exist_ok=True)

        # 保存元数据
        with open(export_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)

        # 保存结果
        if result:
            with open(export_dir / "result.json", 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        # 复制配置文件
        config_src = self.experiments_root / "configs" / f"{experiment_id}.yaml"
        if config_src.exists():
            shutil.copy(config_src, export_dir / "config.yaml")

        # 复制模型文件
        if result and result.models:
            for model_name, model_path in result.models.items():
                if Path(model_path).exists():
                    shutil.copy(model_path, export_dir / f"{model_name}.pth")

        self.logger.info(f"📤 实验导出到: {export_dir}")

    def import_experiment(self, import_path: str) -> str:
        """导入实验"""
        import_dir = Path(import_path)

        # 读取元数据
        metadata_path = import_dir / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"元数据文件不存在: {metadata_path}")

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)

        # 创建新实验ID
        new_experiment_id = f"{metadata_dict['experiment_type']}_imported_{uuid.uuid4().hex[:8]}"

        # 更新元数据
        metadata_dict['experiment_id'] = new_experiment_id
        metadata_dict['created_at'] = datetime.now().isoformat()

        # 保存到数据库
        metadata = ExperimentMetadata(**metadata_dict)
        self.db.save_experiment(metadata)

        # 保存配置文件
        config_src = import_dir / "config.yaml"
        if config_src.exists():
            config_dst = self.experiments_root / "configs" / f"{new_experiment_id}.yaml"
            shutil.copy(config_src, config_dst)

        # 读取结果
        result_path = import_dir / "result.json"
        if result_path.exists():
            with open(result_path, 'r', encoding='utf-8') as f:
                result_dict = json.load(f)

            result_dict['experiment_id'] = new_experiment_id
            result_dict['created_at'] = datetime.now().isoformat()

            result = ExperimentResult(**result_dict)
            self.db.save_experiment_result(result)

        self.logger.info(f"📥 实验导入完成: {new_experiment_id}")

        return new_experiment_id


# Web界面类
class ExperimentDashboard:
    """实验仪表板（Web界面）"""

    def __init__(self, experiment_manager: ExperimentManager, port: int = 8080):
        self.manager = experiment_manager
        self.port = port

    def run(self):
        """运行Web仪表板"""
        try:
            from flask import Flask, render_template, jsonify, request
            import plotly
            import plotly.graph_objs as go
            import json

            app = Flask(__name__)

            @app.route('/')
            def index():
                """首页"""
                stats = self.manager.get_experiment_statistics()
                recent_experiments = self.manager.db.search_experiments(limit=10)

                return render_template('index.html',
                                       stats=stats,
                                       experiments=recent_experiments)

            @app.route('/api/experiments')
            def get_experiments():
                """获取实验列表"""
                experiments = self.manager.db.search_experiments(limit=100)
                return jsonify([exp.to_dict() for exp in experiments])

            @app.route('/api/experiment/<experiment_id>')
            def get_experiment(experiment_id):
                """获取实验详情"""
                metadata = self.manager.db.get_experiment(experiment_id)
                result = self.manager.db.get_experiment_result(experiment_id)

                if not metadata:
                    return jsonify({'error': 'Experiment not found'}), 404

                response = {
                    'metadata': metadata.to_dict(),
                    'result': result.to_dict() if result else None
                }

                return jsonify(response)

            @app.route('/api/knowledge')
            def get_knowledge():
                """获取知识库"""
                query = request.args.get('query', '')
                tags = request.args.getlist('tags')

                knowledge = self.manager.search_knowledge(query, tags)
                return jsonify([k.to_dict() for k in knowledge])

            @app.route('/api/statistics')
            def get_statistics():
                """获取统计信息"""
                stats = self.manager.get_experiment_statistics()
                return jsonify(stats)

            self.logger.info(f"🌐 实验仪表板启动: http://localhost:{self.port}")
            app.run(host='0.0.0.0', port=self.port, debug=False)

        except ImportError:
            self.logger.error("需要安装Flask和plotly来运行仪表板")
            self.logger.error("安装命令: pip install flask plotly")


# 使用示例
def main():
    """实验管理器使用示例"""
    import argparse

    parser = argparse.ArgumentParser(description="实验管理器")
    parser.add_argument("--action", type=str, required=True,
                        choices=['create', 'run', 'compare', 'search', 'export', 'import', 'dashboard'],
                        help="要执行的操作")
    parser.add_argument("--type", type=str,
                        choices=['full_training', 'ablation_study', 'baseline_comparison'],
                        help="实验类型")
    parser.add_argument("--name", type=str, help="实验名称")
    parser.add_argument("--description", type=str, help="实验描述")
    parser.add_argument("--experiment_id", type=str, help="实验ID")
    parser.add_argument("--data", type=str, default="./data/ETTh1.csv",
                        help="数据文件路径")
    parser.add_argument("--tags", type=str, nargs='+', help="实验标签")

    args = parser.parse_args()

    # 创建实验管理器
    manager = ExperimentManager()

    if args.action == 'create':
        if not args.type or not args.name:
            print("需要指定实验类型和名称")
            return

        experiment_type = ExperimentType(args.type)

        metadata = manager.create_experiment(
            name=args.name,
            description=args.description or f"{args.type} experiment",
            experiment_type=experiment_type,
            tags=args.tags
        )

        print(f"✅ 实验创建成功: {metadata.experiment_id}")
        print(f"   名称: {metadata.name}")
        print(f"   类型: {metadata.experiment_type.value}")

    elif args.action == 'run':
        if not args.experiment_id:
            print("需要指定实验ID")
            return

        metadata = manager.db.get_experiment(args.experiment_id)
        if not metadata:
            print(f"实验不存在: {args.experiment_id}")
            return

        if metadata.experiment_type == ExperimentType.FULL_TRAINING:
            result = manager.run_full_training(args.experiment_id, args.data)
        elif metadata.experiment_type == ExperimentType.ABLATION_STUDY:
            result = manager.run_ablation_study(args.experiment_id, data_path=args.data)
        elif metadata.experiment_type == ExperimentType.BASELINE_COMPARISON:
            result = manager.run_baseline_comparison(args.experiment_id, data_path=args.data)
        else:
            print(f"不支持的实验类型: {metadata.experiment_type}")
            return

        print(f"✅ 实验完成: {args.experiment_id}")
        print(f"   结果: {result.metrics}")

    elif args.action == 'compare':
        if not args.experiment_id:
            print("需要指定实验ID（多个用逗号分隔）")
            return

        experiment_ids = args.experiment_id.split(',')
        comparison = manager.compare_experiments(experiment_ids)

        print(f"📊 实验比较结果:")
        for insight in comparison.insights:
            print(f"   {insight}")

    elif args.action == 'search':
        knowledge = manager.search_knowledge(args.description or '', args.tags)

        print(f"🧠 找到 {len(knowledge)} 条知识:")
        for k in knowledge:
            print(f"   [{k.confidence:.1%}] {k.title}")
            print(f"      {k.content[:100]}...")
            print()

    elif args.action == 'dashboard':
        dashboard = ExperimentDashboard(manager)
        dashboard.run()

    else:
        print(f"未知操作: {args.action}")


if __name__ == "__main__":
    main()