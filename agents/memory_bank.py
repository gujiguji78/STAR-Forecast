"""
智能体记忆库 - 存储和检索对话历史、经验和知识
支持长期记忆、短期记忆和工作记忆
"""
import json
import pickle
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import heapq
from collections import defaultdict, deque
import logging
from pathlib import Path
import sqlite3
from contextlib import contextmanager
import faiss  # 向量数据库


class MemoryType(Enum):
    """记忆类型"""
    EPISODIC = "episodic"  # 情节记忆（具体事件）
    SEMANTIC = "semantic"  # 语义记忆（知识事实）
    PROCEDURAL = "procedural"  # 程序记忆（技能方法）
    WORKING = "working"  # 工作记忆（当前任务）
    ASSOCIATIVE = "associative"  # 关联记忆（关系网络）


class MemoryPriority(Enum):
    """记忆优先级"""
    CRITICAL = 5  # 关键记忆（必须记住）
    HIGH = 4  # 高优先级
    MEDIUM = 3  # 中优先级
    LOW = 2  # 低优先级
    TRIVIAL = 1  # 琐碎记忆（可遗忘）


@dataclass
class MemoryNode:
    """记忆节点"""
    memory_id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    embedding: Optional[np.ndarray] = None  # 向量表示
    timestamp: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 1
    priority: MemoryPriority = MemoryPriority.MEDIUM
    decay_rate: float = 0.1  # 遗忘速率（每天）
    associations: List[str] = field(default_factory=list)  # 关联记忆ID
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def strength(self) -> float:
        """计算记忆强度"""
        # 基础强度 = 优先级 + 访问频率 - 时间衰减
        time_elapsed = (datetime.now() - self.timestamp).total_seconds() / 86400  # 天

        # 艾宾浩斯遗忘曲线调整
        base_strength = self.priority.value
        frequency_boost = np.log1p(self.access_count) * 0.5
        time_decay = np.exp(-self.decay_rate * time_elapsed)

        strength = (base_strength + frequency_boost) * time_decay

        return max(0.0, min(10.0, strength))

    @property
    def relevance(self) -> float:
        """计算近期相关性"""
        time_elapsed = (datetime.now() - self.last_accessed).total_seconds() / 3600  # 小时
        return np.exp(-0.1 * time_elapsed) * self.strength


@dataclass
class MemoryQuery:
    """记忆查询"""
    query_text: Optional[str] = None
    query_embedding: Optional[np.ndarray] = None
    memory_type: Optional[MemoryType] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    priority_filter: Optional[MemoryPriority] = None
    min_strength: float = 0.0
    max_results: int = 10
    similarity_threshold: float = 0.7


@dataclass
class MemoryRetrieval:
    """记忆检索结果"""
    memories: List[MemoryNode]
    scores: List[float]
    query: MemoryQuery
    retrieval_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'memories': [
                {
                    'id': mem.memory_id,
                    'type': mem.memory_type.value,
                    'content': mem.content,
                    'strength': mem.strength,
                    'relevance': mem.relevance,
                    'timestamp': mem.timestamp.isoformat()
                }
                for mem in self.memories
            ],
            'scores': self.scores,
            'query': asdict(self.query) if self.query else None,
            'retrieval_time': self.retrieval_time.isoformat()
        }


class VectorIndex:
    """向量索引管理器"""

    def __init__(self, dimension: int = 384):  # 使用标准嵌入维度
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.id_to_index = {}  # memory_id -> index位置
        self.index_to_id = {}  # index位置 -> memory_id
        self.next_index = 0

    def add_memory(self, memory_id: str, embedding: np.ndarray):
        """添加记忆向量"""
        if memory_id in self.id_to_index:
            # 更新现有向量
            idx = self.id_to_index[memory_id]
            self.index.remove_ids(np.array([idx]))
            self.index.add(embedding.reshape(1, -1))
        else:
            # 添加新向量
            self.index.add(embedding.reshape(1, -1))
            self.id_to_index[memory_id] = self.next_index
            self.index_to_id[self.next_index] = memory_id
            self.next_index += 1

    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """搜索相似记忆"""
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx in self.index_to_id and idx != -1:
                memory_id = self.index_to_id[idx]
                # 转换距离为相似度（0-1）
                similarity = 1.0 / (1.0 + dist)
                results.append((memory_id, similarity))

        return results

    def remove_memory(self, memory_id: str):
        """移除记忆"""
        if memory_id in self.id_to_index:
            idx = self.id_to_index[memory_id]
            self.index.remove_ids(np.array([idx]))
            del self.id_to_index[memory_id]
            del self.index_to_id[idx]

    def save(self, path: str):
        """保存索引"""
        faiss.write_index(self.index, path)

        # 保存映射关系
        mapping = {
            'id_to_index': self.id_to_index,
            'index_to_id': self.index_to_id,
            'next_index': self.next_index
        }

        with open(f"{path}.mapping", 'wb') as f:
            pickle.dump(mapping, f)

    def load(self, path: str):
        """加载索引"""
        self.index = faiss.read_index(path)

        # 加载映射关系
        with open(f"{path}.mapping", 'rb') as f:
            mapping = pickle.load(f)

        self.id_to_index = mapping['id_to_index']
        self.index_to_id = mapping['index_to_id']
        self.next_index = mapping['next_index']


class EmbeddingModel:
    """嵌入模型（简化实现）"""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

        # 预定义的类别向量（实际应使用BERT等模型）
        self.category_vectors = {
            'model_architecture': np.random.randn(dimension),
            'hyperparameter': np.random.randn(dimension),
            'performance_metric': np.random.randn(dimension),
            'data_pattern': np.random.randn(dimension),
            'training_strategy': np.random.randn(dimension),
            'error_analysis': np.random.randn(dimension)
        }

    def encode(self, text: str) -> np.ndarray:
        """编码文本为向量（简化实现）"""
        # 实际项目应使用真实嵌入模型
        # 这里使用基于关键词的简单向量

        vector = np.zeros(self.dimension)

        # 关键词匹配
        keywords = {
            '谱门控': 'model_architecture',
            '拉普拉斯': 'model_architecture',
            'TCN': 'model_architecture',
            'MSE': 'performance_metric',
            'MAE': 'performance_metric',
            '学习率': 'hyperparameter',
            '正则化': 'training_strategy',
            '过拟合': 'error_analysis',
            '平稳性': 'data_pattern'
        }

        # 合并相关类别向量
        matched_categories = set()
        for keyword, category in keywords.items():
            if keyword in text:
                matched_categories.add(category)

        if matched_categories:
            for category in matched_categories:
                vector += self.category_vectors[category]
            vector /= len(matched_categories)
        else:
            # 随机向量作为后备
            vector = np.random.randn(self.dimension)
            vector = vector / np.linalg.norm(vector)

        # 添加文本长度特征
        length_feature = min(len(text) / 1000, 1.0)
        vector[:10] += length_feature * 0.1

        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector

    def encode_dict(self, data: Dict[str, Any]) -> np.ndarray:
        """编码字典数据为向量"""
        # 将字典转换为文本
        text = json.dumps(data, ensure_ascii=False)
        return self.encode(text)


class SQLiteMemoryStore:
    """SQLite记忆存储"""

    def __init__(self, db_path: str = "./memory.db"):
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
            # 创建记忆表
            conn.execute("""
                         CREATE TABLE IF NOT EXISTS memories
                         (
                             memory_id
                             TEXT
                             PRIMARY
                             KEY,
                             memory_type
                             TEXT
                             NOT
                             NULL,
                             content
                             TEXT
                             NOT
                             NULL,
                             embedding
                             BLOB,
                             timestamp
                             DATETIME
                             NOT
                             NULL,
                             last_accessed
                             DATETIME
                             NOT
                             NULL,
                             access_count
                             INTEGER
                             DEFAULT
                             1,
                             priority
                             INTEGER
                             DEFAULT
                             3,
                             decay_rate
                             REAL
                             DEFAULT
                             0.1,
                             associations
                             TEXT,
                             metadata
                             TEXT,
                             created_at
                             DATETIME
                             DEFAULT
                             CURRENT_TIMESTAMP
                         )
                         """)

            # 创建索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_type ON memories (memory_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories (timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_priority ON memories (priority)")

            conn.commit()

    def save_memory(self, memory: MemoryNode):
        """保存记忆"""
        with self._get_connection() as conn:
            # 转换数据
            content_json = json.dumps(memory.content, ensure_ascii=False)
            associations_json = json.dumps(memory.associations, ensure_ascii=False)
            metadata_json = json.dumps(memory.metadata, ensure_ascii=False)

            embedding_blob = None
            if memory.embedding is not None:
                embedding_blob = memory.embedding.tobytes()

            # 插入或更新
            conn.execute("""
                INSERT OR REPLACE INTO memories 
                (memory_id, memory_type, content, embedding, timestamp, 
                 last_accessed, access_count, priority, decay_rate, 
                 associations, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.memory_id,
                memory.memory_type.value,
                content_json,
                embedding_blob,
                memory.timestamp.isoformat(),
                memory.last_accessed.isoformat(),
                memory.access_count,
                memory.priority.value,
                memory.decay_rate,
                associations_json,
                metadata_json
            ))

            conn.commit()

    def load_memory(self, memory_id: str) -> Optional[MemoryNode]:
        """加载记忆"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM memories WHERE memory_id = ?",
                (memory_id,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            # 解析数据
            embedding = None
            if row['embedding']:
                embedding = np.frombuffer(row['embedding'], dtype=np.float32)

            associations = json.loads(row['associations']) if row['associations'] else []
            metadata = json.loads(row['metadata']) if row['metadata'] else {}

            memory = MemoryNode(
                memory_id=row['memory_id'],
                memory_type=MemoryType(row['memory_type']),
                content=json.loads(row['content']),
                embedding=embedding,
                timestamp=datetime.fromisoformat(row['timestamp']),
                last_accessed=datetime.fromisoformat(row['last_accessed']),
                access_count=row['access_count'],
                priority=MemoryPriority(row['priority']),
                decay_rate=row['decay_rate'],
                associations=associations,
                metadata=metadata
            )

            return memory

    def delete_memory(self, memory_id: str):
        """删除记忆"""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
            conn.commit()

    def search_memories(self, query: MemoryQuery) -> List[MemoryNode]:
        """搜索记忆（基于元数据）"""
        with self._get_connection() as conn:
            # 构建查询条件
            conditions = []
            params = []

            if query.memory_type:
                conditions.append("memory_type = ?")
                params.append(query.memory_type.value)

            if query.time_range:
                start_time, end_time = query.time_range
                conditions.append("timestamp BETWEEN ? AND ?")
                params.extend([start_time.isoformat(), end_time.isoformat()])

            if query.priority_filter:
                conditions.append("priority >= ?")
                params.append(query.priority_filter.value)

            # 执行查询
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            sql = f"""
                SELECT * FROM memories 
                WHERE {where_clause}
                ORDER BY last_accessed DESC
                LIMIT ?
            """
            params.append(query.max_results)

            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

            # 转换为MemoryNode对象
            memories = []
            for row in rows:
                # 计算强度（简化）
                access_count = row['access_count']
                priority = MemoryPriority(row['priority'])

                # 过滤低强度记忆
                strength = priority.value + np.log1p(access_count)
                if strength < query.min_strength:
                    continue

                # 解析数据
                embedding = None
                if row['embedding']:
                    embedding = np.frombuffer(row['embedding'], dtype=np.float32)

                associations = json.loads(row['associations']) if row['associations'] else []
                metadata = json.loads(row['metadata']) if row['metadata'] else {}

                memory = MemoryNode(
                    memory_id=row['memory_id'],
                    memory_type=MemoryType(row['memory_type']),
                    content=json.loads(row['content']),
                    embedding=embedding,
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    last_accessed=datetime.fromisoformat(row['last_accessed']),
                    access_count=row['access_count'],
                    priority=priority,
                    decay_rate=row['decay_rate'],
                    associations=associations,
                    metadata=metadata
                )

                memories.append(memory)

            return memories

    def get_statistics(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        with self._get_connection() as conn:
            stats = {}

            # 总数
            cursor = conn.execute("SELECT COUNT(*) as total FROM memories")
            stats['total_memories'] = cursor.fetchone()['total']

            # 按类型统计
            cursor = conn.execute("""
                                  SELECT memory_type, COUNT(*) as count
                                  FROM memories
                                  GROUP BY memory_type
                                  """)
            stats['by_type'] = {row['memory_type']: row['count'] for row in cursor.fetchall()}

            # 按优先级统计
            cursor = conn.execute("""
                                  SELECT priority, COUNT(*) as count
                                  FROM memories
                                  GROUP BY priority
                                  """)
            stats['by_priority'] = {row['priority']: row['count'] for row in cursor.fetchall()}

            # 时间范围
            cursor = conn.execute("""
                                  SELECT MIN(timestamp) as oldest, MAX(timestamp) as newest
                                  FROM memories
                                  """)
            row = cursor.fetchone()
            stats['time_range'] = {
                'oldest': row['oldest'],
                'newest': row['newest']
            }

            return stats


class MemoryBank:
    """
    智能体记忆库

    特点：
    1. 多类型记忆存储（情节、语义、程序等）
    2. 向量相似度检索
    3. SQLite持久化存储
    4. 记忆强度和遗忘机制
    5. 关联记忆网络
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # 记忆存储
        self.memory_store = SQLiteMemoryStore(
            config.get('memory', {}).get('db_path', './memory.db')
        )

        # 向量索引
        self.embedding_model = EmbeddingModel(
            dimension=config.get('memory', {}).get('embedding_dim', 384)
        )
        self.vector_index = VectorIndex(self.embedding_model.dimension)

        # 工作记忆（短期）
        self.working_memory = deque(maxlen=config.get('memory', {}).get('working_memory_size', 10))

        # 加载现有记忆
        self._load_existing_memories()

        self.logger.info("✅ 智能体记忆库初始化完成")

    def _load_existing_memories(self):
        """加载现有记忆到向量索引"""
        # 加载所有记忆
        query = MemoryQuery(max_results=1000)
        memories = self.memory_store.search_memories(query)

        # 添加到向量索引
        for memory in memories:
            if memory.embedding is not None:
                self.vector_index.add_memory(memory.memory_id, memory.embedding)

        self.logger.info(f"📚 加载 {len(memories)} 条现有记忆")

    def store(self, content: Dict[str, Any],
              memory_type: MemoryType = MemoryType.EPISODIC,
              priority: MemoryPriority = MemoryPriority.MEDIUM,
              associations: List[str] = None,
              metadata: Dict[str, Any] = None) -> str:
        """
        存储记忆

        Args:
            content: 记忆内容
            memory_type: 记忆类型
            priority: 记忆优先级
            associations: 关联记忆ID列表
            metadata: 元数据

        Returns:
            记忆ID
        """
        # 生成唯一ID
        memory_id = hashlib.md5(
            f"{json.dumps(content)}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        # 创建嵌入向量
        embedding = self.embedding_model.encode_dict(content)

        # 创建记忆节点
        memory = MemoryNode(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            embedding=embedding,
            priority=priority,
            associations=associations or [],
            metadata=metadata or {}
        )

        # 保存到存储
        self.memory_store.save_memory(memory)

        # 添加到向量索引
        self.vector_index.add_memory(memory_id, embedding)

        # 添加到工作记忆
        self.working_memory.append(memory)

        self.logger.debug(f"💾 存储记忆: {memory_id}, 类型: {memory_type.value}")

        return memory_id

    def retrieve(self, query: MemoryQuery) -> MemoryRetrieval:
        """
        检索记忆

        Args:
            query: 记忆查询

        Returns:
            检索结果
        """
        memories = []
        scores = []

        # 1. 向量相似度检索
        if query.query_embedding is not None:
            vector_results = self.vector_index.search(query.query_embedding, query.max_results)

            for memory_id, similarity in vector_results:
                if similarity >= query.similarity_threshold:
                    memory = self.memory_store.load_memory(memory_id)
                    if memory:
                        # 更新访问信息
                        memory.last_accessed = datetime.now()
                        memory.access_count += 1
                        self.memory_store.save_memory(memory)

                        memories.append(memory)
                        scores.append(similarity)

        # 2. 元数据检索（如果向量检索结果不足）
        if len(memories) < query.max_results:
            metadata_results = self.memory_store.search_memories(query)

            for memory in metadata_results:
                if memory.memory_id not in [m.memory_id for m in memories]:
                    # 计算相似度（基于时间衰减和强度）
                    time_decay = np.exp(-0.1 * (datetime.now() - memory.timestamp).total_seconds() / 3600)
                    relevance = memory.strength * time_decay

                    memories.append(memory)
                    scores.append(relevance)

        # 3. 限制结果数量
        if len(memories) > query.max_results:
            # 按分数排序
            sorted_pairs = sorted(zip(scores, memories), reverse=True)
            scores, memories = zip(*sorted_pairs[:query.max_results])
            scores, memories = list(scores), list(memories)

        return MemoryRetrieval(
            memories=memories,
            scores=scores,
            query=query
        )

    def retrieve_by_text(self, text: str, **kwargs) -> MemoryRetrieval:
        """通过文本检索记忆"""
        # 编码查询文本
        query_embedding = self.embedding_model.encode(text)

        # 创建查询
        query = MemoryQuery(
            query_text=text,
            query_embedding=query_embedding,
            **kwargs
        )

        return self.retrieve(query)

    def retrieve_by_context(self, context: Dict[str, Any], **kwargs) -> MemoryRetrieval:
        """通过上下文检索记忆"""
        # 编码上下文
        query_embedding = self.embedding_model.encode_dict(context)

        # 创建查询
        query = MemoryQuery(
            query_embedding=query_embedding,
            **kwargs
        )

        return self.retrieve(query)

    def retrieve_similar_experiences(self, current_situation: Dict[str, Any],
                                     max_results: int = 5) -> List[Dict[str, Any]]:
        """检索相似经验"""
        # 查找类似的历史情境
        retrieval = self.retrieve_by_context(
            current_situation,
            memory_type=MemoryType.EPISODIC,
            max_results=max_results
        )

        # 提取经验教训
        experiences = []
        for memory, score in zip(retrieval.memories, retrieval.scores):
            content = memory.content
            if 'outcome' in content and 'lessons' in content:
                experience = {
                    'situation': content.get('context', {}),
                    'action_taken': content.get('action', ''),
                    'outcome': content['outcome'],
                    'lessons': content['lessons'],
                    'similarity': score,
                    'timestamp': memory.timestamp
                }
                experiences.append(experience)

        return experiences

    def create_association(self, memory_id1: str, memory_id2: str,
                           relationship: str = "related"):
        """创建记忆关联"""
        memory1 = self.memory_store.load_memory(memory_id1)
        memory2 = self.memory_store.load_memory(memory_id2)

        if not memory1 or not memory2:
            return False

        # 添加到关联列表
        if memory_id2 not in memory1.associations:
            memory1.associations.append(memory_id2)

        if memory_id1 not in memory2.associations:
            memory2.associations.append(memory_id1)

        # 更新元数据
        memory1.metadata.setdefault('associations', {})[memory_id2] = {
            'relationship': relationship,
            'created_at': datetime.now().isoformat()
        }

        memory2.metadata.setdefault('associations', {})[memory_id1] = {
            'relationship': relationship,
            'created_at': datetime.now().isoformat()
        }

        # 保存更新
        self.memory_store.save_memory(memory1)
        self.memory_store.save_memory(memory2)

        self.logger.debug(f"🔗 创建关联: {memory_id1} <-> {memory_id2}")

        return True

    def get_association_network(self, memory_id: str, depth: int = 2) -> Dict[str, Any]:
        """获取关联网络"""
        memory = self.memory_store.load_memory(memory_id)
        if not memory:
            return {}

        network = {
            'center': {
                'id': memory.memory_id,
                'type': memory.memory_type.value,
                'content_preview': str(memory.content)[:100]
            },
            'associations': []
        }

        visited = set([memory_id])
        queue = [(memory_id, 0)]  # (memory_id, depth)

        while queue:
            current_id, current_depth = queue.pop(0)

            if current_depth >= depth:
                continue

            current_memory = self.memory_store.load_memory(current_id)
            if not current_memory:
                continue

            for assoc_id in current_memory.associations:
                if assoc_id not in visited:
                    visited.add(assoc_id)

                    assoc_memory = self.memory_store.load_memory(assoc_id)
                    if assoc_memory:
                        network['associations'].append({
                            'id': assoc_id,
                            'type': assoc_memory.memory_type.value,
                            'content_preview': str(assoc_memory.content)[:100],
                            'depth': current_depth + 1,
                            'relationship': assoc_memory.metadata.get('associations', {})
                            .get(current_id, {})
                            .get('relationship', 'unknown')
                        })

                        if current_depth + 1 < depth:
                            queue.append((assoc_id, current_depth + 1))

        return network

    def consolidate_memories(self):
        """记忆巩固 - 加强重要记忆，弱化不重要记忆"""
        # 检索所有记忆
        query = MemoryQuery(max_results=1000)
        memories = self.memory_store.search_memories(query)

        consolidated_count = 0
        forgotten_count = 0

        for memory in memories:
            current_strength = memory.strength

            # 根据记忆强度决定处理方式
            if current_strength < 1.0:  # 很弱的记忆
                # 检查是否应该遗忘
                if memory.priority == MemoryPriority.TRIVIAL:
                    self.forget(memory.memory_id)
                    forgotten_count += 1
                else:
                    # 加强记忆（模拟睡眠中的巩固）
                    if memory.access_count < 5:
                        memory.decay_rate *= 0.9  # 减慢遗忘
                        self.memory_store.save_memory(memory)
                        consolidated_count += 1

        self.logger.info(f"🔄 记忆巩固完成: 加强 {consolidated_count} 条, 遗忘 {forgotten_count} 条")

    def forget(self, memory_id: str):
        """遗忘记忆"""
        # 从向量索引移除
        self.vector_index.remove_memory(memory_id)

        # 从存储移除
        self.memory_store.delete_memory(memory_id)

        # 从工作记忆移除
        self.working_memory = deque(
            [m for m in self.working_memory if m.memory_id != memory_id],
            maxlen=self.working_memory.maxlen
        )

        self.logger.debug(f"🧹 遗忘记忆: {memory_id}")

    def cleanup_weak_memories(self, strength_threshold: float = 0.5):
        """清理弱记忆"""
        query = MemoryQuery(max_results=1000)
        memories = self.memory_store.search_memories(query)

        forgotten = []
        for memory in memories:
            if memory.strength < strength_threshold and memory.priority != MemoryPriority.CRITICAL:
                self.forget(memory.memory_id)
                forgotten.append(memory.memory_id)

        self.logger.info(f"🧹 清理弱记忆: {len(forgotten)} 条")
        return forgotten

    def get_working_memory(self) -> List[MemoryNode]:
        """获取工作记忆"""
        return list(self.working_memory)

    def add_to_working_memory(self, content: Dict[str, Any],
                              memory_type: MemoryType = MemoryType.WORKING):
        """添加到工作记忆"""
        memory_id = self.store(content, memory_type, MemoryPriority.HIGH)
        memory = self.memory_store.load_memory(memory_id)

        if memory:
            self.working_memory.append(memory)

        return memory_id

    def clear_working_memory(self):
        """清空工作记忆"""
        self.working_memory.clear()
        self.logger.debug("🧹 清空工作记忆")

    def get_statistics(self) -> Dict[str, Any]:
        """获取记忆库统计信息"""
        store_stats = self.memory_store.get_statistics()

        stats = {
            **store_stats,
            'vector_index_size': self.vector_index.next_index,
            'working_memory_size': len(self.working_memory),
            'working_memory_capacity': self.working_memory.maxlen
        }

        return stats

    def export_memories(self, export_path: str,
                        memory_type: Optional[MemoryType] = None):
        """导出记忆"""
        query = MemoryQuery(max_results=10000)
        if memory_type:
            query.memory_type = memory_type

        memories = self.memory_store.search_memories(query)

        export_data = []
        for memory in memories:
            export_data.append({
                'id': memory.memory_id,
                'type': memory.memory_type.value,
                'content': memory.content,
                'strength': memory.strength,
                'timestamp': memory.timestamp.isoformat(),
                'last_accessed': memory.last_accessed.isoformat(),
                'access_count': memory.access_count,
                'priority': memory.priority.value,
                'associations': memory.associations,
                'metadata': memory.metadata
            })

        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"💾 导出 {len(export_data)} 条记忆到 {export_path}")

    def import_memories(self, import_path: str):
        """导入记忆"""
        with open(import_path, 'r', encoding='utf-8') as f:
            import_data = json.load(f)

        imported_count = 0
        for item in import_data:
            # 创建记忆节点
            memory = MemoryNode(
                memory_id=item['id'],
                memory_type=MemoryType(item['type']),
                content=item['content'],
                timestamp=datetime.fromisoformat(item['timestamp']),
                last_accessed=datetime.fromisoformat(item['last_accessed']),
                access_count=item['access_count'],
                priority=MemoryPriority(item['priority']),
                associations=item.get('associations', []),
                metadata=item.get('metadata', {})
            )

            # 生成嵌入向量
            memory.embedding = self.embedding_model.encode_dict(memory.content)

            # 保存记忆
            self.memory_store.save_memory(memory)
            self.vector_index.add_memory(memory.memory_id, memory.embedding)

            imported_count += 1

        self.logger.info(f"📥 导入 {imported_count} 条记忆从 {import_path}")


# 使用示例
def main():
    """记忆库使用示例"""
    import yaml

    # 加载配置
    with open("./config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 创建记忆库
    memory_bank = MemoryBank(config)

    # 存储一些记忆
    memory1_id = memory_bank.store(
        content={
            'context': '训练过程中发现过拟合',
            'action': '增加了拉普拉斯正则化权重',
            'outcome': '验证损失下降了15%',
            'lessons': ['正则化对防止过拟合有效', '需要平衡正则化强度']
        },
        memory_type=MemoryType.EPISODIC,
        priority=MemoryPriority.HIGH
    )

    memory2_id = memory_bank.store(
        content={
            'concept': '谱门控机制',
            'description': '通过频域分析动态调整特征重要性',
            'applications': ['时序预测', '信号处理', '异常检测'],
            'parameters': {'threshold': 0.5, 'bands': 8}
        },
        memory_type=MemoryType.SEMANTIC,
        priority=MemoryPriority.CRITICAL
    )

    # 创建关联
    memory_bank.create_association(memory1_id, memory2_id, 'application_of_concept')

    # 检索记忆
    retrieval = memory_bank.retrieve_by_text("过拟合 正则化")
    print(f"检索到 {len(retrieval.memories)} 条相关记忆")

    for memory, score in zip(retrieval.memories, retrieval.scores):
        print(f"  记忆ID: {memory.memory_id}, 相似度: {score:.3f}")
        print(f"  内容: {memory.content.get('context', 'N/A')}")

    # 获取关联网络
    network = memory_bank.get_association_network(memory1_id)
    print(f"\n关联网络: {len(network['associations'])} 个关联")

    # 获取统计信息
    stats = memory_bank.get_statistics()
    print(f"\n记忆库统计:")
    print(f"  总记忆数: {stats['total_memories']}")
    print(f"  工作记忆: {stats['working_memory_size']}/{stats['working_memory_capacity']}")


if __name__ == "__main__":
    main()