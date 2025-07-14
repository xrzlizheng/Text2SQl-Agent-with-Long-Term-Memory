"""
智能查询助手：无Denodo依赖版本
============================

这个版本移除了对Denodo数据虚拟化平台的依赖，使用直接的数据库连接和火山引擎AI API进行SQL生成。
支持多种数据库类型：PostgreSQL、MySQL、SQLite等。

主要改动：
- 移除DenodoAPIClient
- 使用DatabaseClient替代
- 使用SQLGenerator替代Denodo AI SDK
- 保持原有的记忆系统功能

作者：xrzlizheng
版本：1.0
"""

import os
import json
import time
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# 导入新的模块
from database_client import DatabaseClient, DatabaseClientFactory
from sql_generator import SQLGenerator, QueryProcessor

# 加载环境变量
load_dotenv()

# 配置常量
AI_API_KEY = os.getenv("AI_API_KEY", "")  # 火山引擎AI API密钥
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://api.volcengine.com/v1")  # 火山引擎AI服务基础URL
AI_MODEL = os.getenv("AI_MODEL", "doubao-seed-1-6-250615")  # 火山引擎AI模型
THINKING = os.getenv("THINKING", "true").lower() == "true"  # 思维链模式
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")  # 嵌入模型

# 数据库配置
DB_TYPE = os.getenv("DB_TYPE", "postgresql")  # 数据库类型
DB_CONFIG = {
    "postgresql": {
        "connection_string": os.getenv("PG_CONN_STRING", "")
    },
    "mysql": {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "database": os.getenv("MYSQL_DATABASE", ""),
        "username": os.getenv("MYSQL_USERNAME", ""),
        "password": os.getenv("MYSQL_PASSWORD", "")
    },
    "sqlite": {
        "database_path": os.getenv("SQLITE_PATH", "database.db")
    }
}


class Memory:
    """
    记忆对象的基础类
    
    用于存储用户的偏好、术语、实体和指标等信息。
    每个记忆都包含内容、创建时间、来源和元数据。
    """
    def __init__(self, id: Optional[int] = None, content: str = "", created_at: float = 0, 
                 source: str = "", metadata: Dict = None, embedding: List[float] = None):
        self.id = id  # 记忆的唯一标识符
        self.content = content  # 记忆的内容文本
        self.created_at = created_at  # 创建时间戳
        self.source = source  # 记忆来源（如用户消息、AI响应等）
        self.metadata = metadata or {}  # 额外的元数据信息
        self.embedding = embedding  # 文本的向量嵌入表示


class JSONMemoryStore:
    """
    JSON文件记忆存储实现
    
    使用JSON文件存储记忆，适用于开发和测试环境。
    """
    
    def __init__(self, file_path: str = "memory_db.json"):
        self.file_path = file_path
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """确保JSON文件存在"""
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "memories": [],
                    "conversation_summaries": [],
                    "recent_messages": []
                }, f, ensure_ascii=False, indent=2)
    
    def load_memories(self) -> List[Memory]:
        """加载所有记忆"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            memories = []
            for mem_data in data.get("memories", []):
                memory = Memory(
                    id=mem_data.get("id"),
                    content=mem_data.get("content", ""),
                    created_at=mem_data.get("created_at", 0),
                    source=mem_data.get("source", ""),
                    metadata=mem_data.get("metadata", {}),
                    embedding=mem_data.get("embedding", [])
                )
                memories.append(memory)
            
            return memories
        except Exception as e:
            print(f"从JSON加载记忆时出错: {e}")
            return []
    
    def save_memory(self, memory: Memory) -> int:
        """保存记忆"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if memory.id is None:
                # 生成新ID
                existing_ids = [mem.get("id", 0) for mem in data.get("memories", [])]
                memory.id = max(existing_ids, default=0) + 1
            
            # 确保嵌入向量维度正确（Qwen嵌入模型通常是1024维）
            expected_dim = 1024
            if memory.embedding and len(memory.embedding) != expected_dim:
                print(f"警告：预期{expected_dim}维，实际为{len(memory.embedding)}。将截断或填充。")
                if len(memory.embedding) > expected_dim:
                    memory.embedding = memory.embedding[:expected_dim]
                else:
                    memory.embedding.extend([0.0] * (expected_dim - len(memory.embedding)))
            
            # 更新或添加记忆
            mem_data = {
                "id": memory.id,
                "content": memory.content,
                "created_at": memory.created_at,
                "source": memory.source,
                "metadata": memory.metadata,
                "embedding": memory.embedding
            }
            
            # 查找现有记忆
            memories = data.get("memories", [])
            existing_index = next((i for i, mem in enumerate(memories) if mem.get("id") == memory.id), None)
            
            if existing_index is not None:
                memories[existing_index] = mem_data
            else:
                memories.append(mem_data)
            
            data["memories"] = memories
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return memory.id
        except Exception as e:
            print(f"保存记忆时出错: {e}")
            return -1
    
    def delete_memory(self, memory_id: int) -> bool:
        """删除记忆"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            memories = data.get("memories", [])
            memories = [mem for mem in memories if mem.get("id") != memory_id]
            data["memories"] = memories
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"删除记忆时出错: {e}")
            return False
    
    def get_conversation_summary(self) -> str:
        """获取对话摘要"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            summaries = data.get("conversation_summaries", [])
            if summaries:
                return summaries[-1]["summary"]
            return ""
        except Exception as e:
            print(f"获取对话摘要时出错: {e}")
            return ""
    
    def save_conversation_summary(self, summary: str) -> bool:
        """保存对话摘要"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            summaries = data.get("conversation_summaries", [])
            summaries.append({
                "summary": summary,
                "timestamp": time.time()
            })
            data["conversation_summaries"] = summaries
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"保存对话摘要时出错: {e}")
            return False
    
    def get_recent_messages(self, limit: int = 10) -> List[str]:
        """获取最近消息"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            messages = data.get("recent_messages", [])
            return [msg["message"] for msg in messages[-limit:]]
        except Exception as e:
            print(f"获取最近消息时出错: {e}")
            return []
    
    def save_message(self, message: str) -> bool:
        """保存消息"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            messages = data.get("recent_messages", [])
            messages.append({
                "message": message,
                "timestamp": time.time()
            })
            
            # 只保留最近100条消息
            if len(messages) > 100:
                messages = messages[-100:]
            
            data["recent_messages"] = messages
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"保存消息时出错: {e}")
            return False
    
    def find_similar_memories(self, embedding: List[float], top_k: int = 5) -> List[Tuple[Memory, float]]:
        """查找相似记忆"""
        try:
            memories = self.load_memories()
            similarities = []
            
            for memory in memories:
                if memory.embedding:
                    # 计算余弦相似度
                    similarity = self._cosine_similarity(embedding, memory.embedding)
                    similarities.append((memory, similarity))
            
            # 按相似度排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
        except Exception as e:
            print(f"查找相似记忆时出错: {e}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        try:
            vec1 = np.array(vec1)
            vec2 = np.array(vec2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except Exception:
            return 0.0


class Mem0Agent:
    """
    Mem0记忆代理（无Denodo版本）
    
    实现长期记忆功能，但不依赖Denodo平台
    """
    
    def __init__(self, use_postgres: bool = False, postgres_conn_string: str = None):
        # 初始化记忆存储
        if use_postgres and postgres_conn_string:
            try:
                from memory_agent import PostgresMemoryStore
                self.store = PostgresMemoryStore(postgres_conn_string)
                print("使用PostgreSQL进行记忆存储")
            except ImportError:
                print("PostgreSQL支持不可用。使用JSON存储。")
                self.store = JSONMemoryStore()
        else:
            self.store = JSONMemoryStore()
        
        # 初始化嵌入模型
        try:
            print(f"正在加载嵌入模型: {EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            print("嵌入模型加载成功")
        except Exception as e:
            print(f"嵌入模型加载失败: {e}")
            print("将使用备用嵌入方法")
            self.embedding_model = None
        
        # 初始化其他属性
        self.memories = []
        self.conversation_summary = ""
        self.recent_messages = []
        self.max_recent_messages = 20
        self.current_user_id = None
    
    def load_user_context(self, user_id: str):
        """
        加载特定用户的上下文
        
        Args:
            user_id: 用户ID
        """
        self.current_user_id = user_id
        self.memories = self.store.load_memories()
        self.conversation_summary = self.store.get_conversation_summary()
        self.recent_messages = self.store.get_recent_messages()
    
    def _create_embedding(self, text: str) -> List[float]:
        """
        使用SentenceTransformer创建文本嵌入
        
        Args:
            text: 输入文本
            
        Returns:
            文本的嵌入向量
        """
        try:
            # 优先使用本地SentenceTransformer模型
            if self.embedding_model is not None:
                embedding = self.embedding_model.encode(text)
                return embedding.tolist()
            
            # 备用方案：使用火山引擎AI API
            print("使用备用火山引擎AI API进行嵌入")
            headers = {
                "Authorization": f"Bearer {AI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "text-embedding-ada-002",  # 备用嵌入模型
                "input": text
            }
            
            response = requests.post(
                f"{AI_BASE_URL}/embeddings",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                return response.json()["data"][0]["embedding"]
            else:
                print(f"嵌入响应错误: {response.json()}")
                # 返回默认维度的空嵌入向量
                return [0.0] * 1024  # Qwen嵌入模型通常是1024维
                
        except Exception as e:
            print(f"创建嵌入时出错: {e}")
            # 返回默认维度的空嵌入向量
            return [0.0] * 1024
    
    def _find_similar_memories(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Memory, float]]:
        """
        查找与查询嵌入向量最相似的记忆
        
        Args:
            query_embedding: 查询嵌入向量
            top_k: 返回的相似记忆数量
            
        Returns:
            相似记忆列表，每个元素为(记忆对象, 相似度)
        """
        if hasattr(self, 'current_user_id') and self.current_user_id:
            return self.store.find_similar_memories(query_embedding, self.current_user_id, top_k)
        return []
    
    async def extract_memories(self, message_pair: List[str]) -> List[str]:
        """
        从消息对中提取重要的记忆
        
        Args:
            message_pair: 消息对 (Human, AI)
            
        Returns:
            提取的记忆列表
        """
        
        prompt = f"""你是一个智能记忆提取系统。你的任务是从对话中识别出关于数据库查询的重要信息，这些信息应该被记住以供将来参考。

上下文：
当前对话摘要：{self.conversation_summary}
最近消息：{self.recent_messages}

当前消息对：
Human: {message_pair[0]}
AI: {message_pair[1]}

提取0-3个简洁、重要的记忆，这些记忆在未来的数据库查询中会很有用。 

特别注意以下类别，并相应地标记：
1. [PREFERENCE] 用户查询偏好或过滤条件 (例如，"用户偏好查看利率低于5%的贷款")
2. [TERM] 用户自定义术语或缩写 (例如，"WC表示西海岸州：CA、OR和WA")
3. [ENTITY] 用户感兴趣的具体表、列或关系
4. [METRIC] 用户自定义指标或计算 (例如，"高风险贷款定义为信用评分低于650且贷款金额超过$200,000")

将每个记忆格式化为一个简洁的句子，并带有适当的标签前缀。如果没有重要信息值得保留，则返回空列表。

提取的记忆："""

        try:
            headers = {
                "Authorization": f"Bearer {AI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": AI_MODEL,
                "messages": [
                    {"role": "system", "content": "你是一个专业的记忆提取系统。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1
            }
            
            # 如果启用思维链模式，添加相关参数
            if THINKING:
                data["thinking"] = True
            
            response = requests.post(
                f"{AI_BASE_URL}/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                
                # 解析响应以提取记忆
                memories = []
                for line in content.strip().split('\n'):
                    line = line.strip()
                    if line and not line.startswith("提取的记忆："):
                        # 移除编号（如果存在）
                        if line[0].isdigit() and line[1:3] in ['. ', '- ', ') ']:
                            line = line[3:].strip()
                        if line and any(tag in line for tag in ['[PREFERENCE]', '[TERM]', '[METRIC]', '[ENTITY]']):
                            memories.append(line)
                
                return memories
            else:
                print(f"提取记忆时出错: {response.text}")
                return []
                
        except Exception as e:
            print(f"提取记忆时出错: {e}")
            return []
            
    async def update_memories(self, candidate_facts: List[str]):
        """
        使用新事实更新记忆数据库
        
        Args:
            candidate_facts: 候选事实列表
        """
        # 确保我们有一个当前用户
        if not hasattr(self, 'current_user_id') or not self.current_user_id:
            print("错误：未设置当前用户。请先调用load_user_context。")
            return
            
        now = time.time()
        
        for fact in candidate_facts:
            # 为事实创建嵌入
            embedding = self._create_embedding(fact)
            
            # 查找相似的现有记忆
            similar_memories = self._find_similar_memories(embedding, top_k=3)
            
            # 确定要执行的操作 (ADD, UPDATE, DELETE, NOOP)
            operation = self._determine_operation(fact, similar_memories)
            
            if operation == "ADD":
                memory = Memory(content=fact, created_at=now, embedding=embedding)
                memory_id = self.store.save_memory(memory)
                if memory_id > 0:  # 如果保存成功
                    memory.id = memory_id
                    self.memories.append(memory)
                    print(f"添加记忆：{fact}")
            
            elif operation == "UPDATE" and similar_memories:
                memory_to_update = similar_memories[0][0]
                # 仅在新的事实包含更多信息时才更新
                if len(fact) > len(memory_to_update.content):
                    memory_to_update.content = fact
                    memory_to_update.embedding = embedding
                    self.store.save_memory(memory_to_update)
                    print(f"更新记忆：{fact}")
            
            elif operation == "DELETE" and similar_memories:
                memory_to_delete = similar_memories[0][0]
                self.store.delete_memory(memory_to_delete.id)
                self.memories.remove(memory_to_delete)
                print(f"删除记忆：{memory_to_delete.content}")
    
    def _determine_operation(self, fact: str, similar_memories: List[Tuple[Memory, float]]) -> str:
        """
        使用LLM确定要执行的记忆操作
        
        Args:
            fact: 新的事实
            similar_memories: 相似记忆列表
            
        Returns:
            记忆操作 (ADD, UPDATE, DELETE, NOOP)
        """
        
        if not similar_memories:
            return "ADD"
        
        if similar_memories[0][1] > 0.95:  # 非常高的相似度
            return "NOOP"
        
        if similar_memories[0][1] > 0.85:  # 高相似度
            # 检查新事实是否与现有记忆矛盾
            prompt = f"""你是一个智能记忆管理器。请确定这两个信息是否相互矛盾或一个增强另一个：

现有记忆：{similar_memories[0][0].content}
新信息：{fact}

请回答以下之一：
- DELETE 如果新信息与现有记忆矛盾并取代它
- UPDATE 如果新信息增强或添加到现有记忆
- NOOP 如果信息是冗余的或现有记忆更全面

决策："""
            
            try:
                headers = {
                    "Authorization": f"Bearer {AI_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": AI_MODEL,
                    "messages": [
                        {"role": "system", "content": "你是一个记忆管理专家。"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 10
                }
                
                # 如果启用思维链模式，添加相关参数
                if THINKING:
                    data["thinking"] = True
                
                response = requests.post(
                    f"{AI_BASE_URL}/chat/completions",
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    decision = response.json()["choices"][0]["message"]["content"].strip().upper()
                    if decision in ["DELETE", "UPDATE", "NOOP"]:
                        return decision
                    return "NOOP"
                else:
                    return "NOOP"
            except Exception as e:
                print(f"确定操作时出错: {e}")
                return "NOOP"
        
        return "ADD"
    
    async def retrieve_relevant_memories(self, query: str, top_k: int = 5) -> List[str]:
        """
        检索与查询相关的记忆
        
        Args:
            query: 查询文本
            top_k: 返回的记忆数量
            
        Returns:
            相关记忆列表
        """
        if not hasattr(self, 'current_user_id') or not self.current_user_id:
            print("错误：未设置当前用户。请先调用load_user_context。")
            return []
            
        query_embedding = self._create_embedding(query)
        similar_memories = self._find_similar_memories(query_embedding, top_k)
        return [memory.content for memory, _ in similar_memories]
    
    async def update_conversation_summary(self):
        """
        根据最近的消息更新对话摘要
        
        """
        if not hasattr(self, 'current_user_id') or not self.current_user_id:
            print("错误：未设置当前用户。请先调用load_user_context。")
            return
            
        if not self.recent_messages:
            return
        
        prompt = f"""总结以下对话中关于数据库查询的关键点。 
重点关注用户查询模式、他们感兴趣的表以及任何重复的主题。

对话：
{self.recent_messages}

之前的摘要：
{self.conversation_summary}

更新后的摘要："""
        
        try:
            headers = {
                "Authorization": f"Bearer {AI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": AI_MODEL,
                "messages": [
                    {"role": "system", "content": "你是一个对话总结专家。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3
            }
            
            # 如果启用思维链模式，添加相关参数
            if THINKING:
                data["thinking"] = True
            
            response = requests.post(
                f"{AI_BASE_URL}/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                self.conversation_summary = response.json()["choices"][0]["message"]["content"].strip()
                self.store.save_conversation_summary(self.conversation_summary)
        except Exception as e:
            print(f"更新对话摘要时出错: {e}")
    
    def add_message_to_history(self, role: str, content: str):
        """
        将消息添加到最近消息历史
        
        Args:
            role: 消息角色 (Human, AI)
            content: 消息内容
        """
        if not hasattr(self, 'current_user_id') or not self.current_user_id:
            print("错误：未设置当前用户。请先调用load_user_context。")
            return
            
        message = f"{role}: {content}"
        self.recent_messages.append(message)
        self.store.save_message(message)
        
        # 只保留最近的消息
        if len(self.recent_messages) > self.max_recent_messages:
            self.recent_messages = self.recent_messages[-self.max_recent_messages:]
    
    def set_db_schema(self, schema: Dict):
        """
        存储数据库模式信息
        
        Args:
            schema: 数据库模式字典
        """
        self.db_schema = schema


class SmartQueryAssistant:
    """
    智能查询助手（无Denodo版本）
    
    整合记忆系统和数据库查询功能
    """
    
    def __init__(self, 
                 db_type: str = "postgresql",
                 db_config: Dict = None,
                 use_postgres_memory: bool = False, 
                 postgres_conn_string: str = None):
        
        self.db_type = db_type
        self.db_config = db_config or DB_CONFIG.get(db_type, {})
        
        # 初始化数据库客户端
        try:
            self.db_client = DatabaseClientFactory.create_client(db_type, **self.db_config)
            print(f"数据库客户端初始化成功: {db_type}")
        except Exception as e:
            print(f"数据库客户端初始化失败: {e}")
            raise
        
        # 初始化SQL生成器
        if not AI_API_KEY:
            raise ValueError("请设置AI_API_KEY环境变量")
        
        self.sql_generator = SQLGenerator(
            ai_api_key=AI_API_KEY,
            ai_base_url=AI_BASE_URL,
            model=AI_MODEL,
            thinking=THINKING,
            embedding_model=EMBEDDING_MODEL
        )
        
        # 初始化查询处理器
        self.query_processor = QueryProcessor(
            self.db_client, 
            self.sql_generator, 
            db_type
        )
        
        # 初始化记忆代理
        self.agent = Mem0Agent(
            use_postgres=use_postgres_memory,
            postgres_conn_string=postgres_conn_string
        )
        
        # 用户相关
        self.current_user = None
        self.db_loaded = False
    
    def set_user(self, username: str, password: str = None):
        """
        设置当前用户
        
        Args:
            username: 用户名
            password: 密码（可选，用于某些数据库认证）
        """
        self.current_user = {
            "username": username,
            "password": password
        }
        # 从记忆存储加载用户上下文
        self.agent.load_user_context(username)
        # 重置数据库加载标志以用于新用户
        self.db_loaded = False
    
    async def initialize_db(self):
        """
        启动时加载数据库元数据
        
        """
        if not self.current_user:
            return False
            
        if not self.db_loaded:
            print(f"加载数据库模式 '{self.db_type}' 的用户 {self.current_user['username']}...")
            
            # 测试数据库连接
            connection_test = await self.db_client.test_connection()
            if not connection_test:
                print("数据库连接测试失败")
                return False
            
            # 获取数据库元数据
            metadata = await self.db_client.get_metadata()
            
            if "error" in metadata:
                print(f"加载数据库模式时出错：{metadata['error']}")
                return False
            
            self.agent.set_db_schema(metadata)
            self.db_loaded = True
            
            # 创建关于数据库的初始记忆
            db_description = self._generate_db_description(metadata)
            if db_description:
                initial_memory = Memory(
                    content=f"[ENTITY] {db_description}",
                    created_at=time.time(),
                    source="system_initialization"
                )
                self.agent.memories.append(initial_memory)
                self.agent.store.save_memory(initial_memory)
            
            print(f"数据库 '{self.db_type}' 加载成功")
            return True
        
        return True
    
    def _generate_db_description(self, metadata: Dict) -> str:
        """
        从元数据生成数据库的简洁描述
        
        Args:
            metadata: 数据库元数据字典
            
        Returns:
            数据库描述字符串
        """
        try:
            tables = list(metadata.get("tables", {}).keys())
            
            if tables:
                return f"数据库包含 {len(tables)} 个表，包括：{', '.join(tables[:10])}{' 等更多' if len(tables) > 10 else ''}。"
            return ""
        except Exception as e:
            print(f"生成数据库描述时出错: {e}")
            return ""
    
    async def process_message(self, user_message: str) -> str:
        """
        处理用户消息并生成带有增强上下文处理的响应
        
        Args:
            user_message: 用户输入消息
            
        Returns:
            生成的AI响应
        """
        
        # 确保我们有一个当前用户
        if not self.current_user:
            return "请先登录。"
        
        # 如果数据库未加载，则初始化
        if not self.db_loaded:
            db_initialized = await self.initialize_db()
            if not db_initialized:
                return "我无法连接到数据库。请稍后再试。"
        
        # 将用户消息添加到历史记录
        self.agent.add_message_to_history("Human", user_message)
        
        # 检索相关记忆
        relevant_memories = await self.agent.retrieve_relevant_memories(user_message)
        
        # 处理和分类记忆
        preferences = []
        terminology = []
        metrics = []
        general_context = []
        
        for memory in relevant_memories:
            if '[PREFERENCE]' in memory:
                preferences.append(memory.replace('[PREFERENCE]', '').strip())
            elif '[TERM]' in memory:
                terminology.append(memory.replace('[TERM]', '').strip())
            elif '[METRIC]' in memory:
                metrics.append(memory.replace('[METRIC]', '').strip())
            elif '[ENTITY]' in memory:
                general_context.append(memory)
        
        # 应用术语替换
        enhanced_question = user_message
        for term in terminology:
            if ":" in term:
                # 提取缩写及其含义
                try:
                    abbr, meaning = term.split("means", 1)
                    abbr = abbr.strip()
                    meaning = meaning.strip()
                    if abbr in enhanced_question:
                        enhanced_question = enhanced_question.replace(abbr, f"{abbr} ({meaning})")
                except ValueError:
                    pass  # 跳过不符合预期格式的项
        
        # 检查是否是后续问题
        follow_up_indicators = ["those", "them", "these", "that", "it", "the ones", "their", "them"]
        is_follow_up = any(indicator in " " + user_message.lower() + " " for indicator in 
                         [f" {ind} " for ind in follow_up_indicators])
        
        # 检查查询是否关于新数据或列出项目
        new_data_keywords = ["new", "recent", "latest", "today", "this week", "show me", "list", "get", "find"]
        is_new_data_query = any(keyword in user_message.lower() for keyword in new_data_keywords)
        
        # 准备上下文部分，不应用偏好
        context_parts = []
        
        # 为后续问题添加一般上下文
        if is_follow_up and general_context:
            context_parts.append("基于我们之前的对话关于：" + "; ".join(general_context))
        
        # 始终添加自定义指标
        if metrics:
            context_parts.append("使用这些定义：" + "; ".join(metrics))
        
        # 检查是否应该询问是否应用偏好
        should_ask_about_preferences = is_new_data_query and preferences and not is_follow_up
        
        if should_ask_about_preferences:
            # 我们需要先询问用户，然后再应用偏好
            preference_question = "您想让我"
            for i, pref in enumerate(preferences):
                if i > 0:
                    preference_question += " 和 "
                # 从偏好中提取动作
                # 例如："用户偏好查看利率低于5%的贷款"
                # 提取："filter for loans with interest rates below 5%"
                if "prefers" in pref:
                    action = pref.split("prefers", 1)[1].strip()
                    preference_question += f"{action}"
                else:
                    preference_question += f"应用这个偏好：{pref}"
            preference_question += " 作为您通常的偏好吗？"
            
            # 返回问题，而不是实际结果
            self.agent.add_message_to_history("AI", preference_question)
            return preference_question
        
        # 如果用户似乎相关或如果是同一领域的查询，则应用用户偏好
        preference_keywords = ["show", "list", "get", "find", "display", "what", "which", "how many"]
        should_apply_preferences = any(kw in user_message.lower() for kw in preference_keywords)
        
        # 处理用户是否响应偏好问题
        yes_indicators = ["yes", "yeah", "yep", "sure", "ok", "apply", "please do", "i do"]
        no_indicators = ["no", "nope", "don't", "do not", "negative"]
        
        # 获取最后一个AI消息
        last_ai_message = next((msg for msg in reversed(self.agent.recent_messages) 
                              if msg.startswith("AI: ") and "would you like me to" in msg.lower()), None)
        
        # 检查这是否是偏好问题的响应
        if last_ai_message and any(indicator in user_message.lower() for indicator in yes_indicators + no_indicators):
            if any(indicator in user_message.lower() for indicator in yes_indicators):
                # 用户想要应用偏好
                should_apply_preferences = True
            else:
                # 用户不想应用偏好
                should_apply_preferences = False
        
        if should_apply_preferences and preferences:
            context_parts.append("考虑到您的偏好：" + "; ".join(preferences))
        
        # 组合上下文部分
        if context_parts:
            context = " ".join(context_parts)
            enhanced_question = f"{context}. {enhanced_question}"
            print(f"增强后的问题：{enhanced_question}")
        
        # 使用查询处理器处理问题
        result = await self.query_processor.process_question(
            enhanced_question, 
            preferences if should_apply_preferences else None
        )
        
        if not result["success"]:
            ai_response = f"抱歉，我遇到了一个错误：{result['error']}"
        else:
            # 从响应中提取关键信息
            answer = result.get("answer", "")
            sql_query = result.get("sql_query", "")
            tables_used = result.get("tables_used", [])
            
            # 格式化全面的响应
            ai_response = answer
            
            # 如果应用了偏好，请注意
            if should_apply_preferences and preferences:
                ai_response += "\n\n(注意：我已根据您之前表达的偏好应用此查询。)"
            
            # 添加归属信息（如果可用）
            tables_attribution = ""
            if tables_used:
                tables_attribution = f"\n\n此信息来自：{', '.join(tables_used)}"
                if sql_query:
                    tables_attribution += f"\n\n查询使用如下：\n```sql\n{sql_query}\n```"
                    
            ai_response += tables_attribution
        
        # 将AI响应添加到历史记录
        self.agent.add_message_to_history("AI", ai_response)
        
        # 从对话中提取记忆
        memories = await self.agent.extract_memories([user_message, ai_response])
        
        # 更新记忆
        await self.agent.update_memories(memories)
        
        # 定期更新对话摘要
        if len(self.agent.recent_messages) % 5 == 0:
            await self.agent.update_conversation_summary()
        
        return ai_response


# 使用示例
async def main():
    """
    主函数运行聊天机器人
    
    """
    # 配置数据库连接
    db_type = "sqlite"  # 或 "postgresql", "mysql"
    db_config = {
        "database_path": "test.db"  # SQLite示例
    }
    
    print("启动智能查询助手（无Denodo版本）...")
    
    try:
        # 创建助手实例
        assistant = SmartQueryAssistant(
            db_type=db_type,
            db_config=db_config
        )
        
        # 设置用户
        assistant.set_user("test_user")
        
        # 初始化数据库
        await assistant.initialize_db()
        
        if assistant.db_loaded:
            print(f"数据库 '{db_type}' 加载成功。您现在可以询问数据。")
        else:
            print("未能加载数据库模式。聊天机器人将继续以有限功能运行。")
        
        print("聊天机器人已准备就绪。输入 'exit' 退出。")
        
        history = []
        while True:
            user_input = input("\n您：")
            
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("聊天机器人：再见！")
                break
            
            # 将用户消息添加到历史记录
            history.append({"role": "user", "content": user_input})
            
            response = await assistant.process_message(user_input)
            
            # 将助手响应添加到历史记录
            history.append({"role": "assistant", "content": response})
            print(f"\n聊天机器人：{response}")
            
    except Exception as e:
        print(f"启动失败：{e}")


if __name__ == "__main__":
    asyncio.run(main()) 