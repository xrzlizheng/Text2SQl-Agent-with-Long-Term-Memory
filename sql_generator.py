"""
SQL生成器模块
===========

使用火山引擎AI API生成SQL查询。
支持多种数据库方言和查询优化。

作者：xrzlizheng
版本：1.0
"""

import os
import json
import requests
from typing import Dict, List, Any, Optional
from database_client import DatabaseClient
from sentence_transformers import SentenceTransformer


class SQLGenerator:
    """
    SQL查询生成器
    
    使用火山引擎AI API将自然语言转换为SQL查询
    """
    
    def __init__(self, ai_api_key: str, ai_base_url: str = "https://api.volcengine.com/v1", 
                 model: str = "doubao-seed-1-6-250615", thinking: bool = True,
                 embedding_model: str = "Qwen/Qwen3-Embedding-8B"):
        self.api_key = ai_api_key
        self.ai_base_url = ai_base_url
        self.model = model
        self.thinking = thinking
        self.chat_url = f"{ai_base_url}/chat/completions"
        self.embedding_url = f"{ai_base_url}/embeddings"
        
        # 初始化嵌入模型
        try:
            print(f"正在加载嵌入模型: {embedding_model}")
            self.embedding_model = SentenceTransformer(embedding_model)
            print("嵌入模型加载成功")
        except Exception as e:
            print(f"嵌入模型加载失败: {e}")
            print("将使用备用嵌入方法")
            self.embedding_model = None
    
    def _create_embedding(self, text: str) -> List[float]:
        """
        创建文本嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        try:
            # 优先使用本地SentenceTransformer模型
            if self.embedding_model is not None:
                embedding = self.embedding_model.encode(text)
                return embedding.tolist()
            
            # 备用方案：使用火山引擎AI API
            print("使用备用火山引擎AI API进行嵌入")
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "text-embedding-ada-002",
                "input": text
            }
            
            response = requests.post(
                self.embedding_url,
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                return response.json()["data"][0]["embedding"]
            else:
                print(f"创建嵌入失败: {response.text}")
                return [0.0] * 1024  # Qwen嵌入模型通常是1024维
                
        except Exception as e:
            print(f"创建嵌入时出错: {e}")
            return [0.0] * 1024
    
    def generate_sql(self, 
                    question: str, 
                    database_metadata: Dict[str, Any],
                    db_type: str = "postgresql",
                    user_preferences: List[str] = None) -> Dict[str, Any]:
        """
        生成SQL查询
        
        Args:
            question: 自然语言问题
            database_metadata: 数据库元数据
            db_type: 数据库类型
            user_preferences: 用户偏好列表
            
        Returns:
            包含SQL查询和解释的字典
        """
        
        # 构建提示词
        prompt = self._build_prompt(question, database_metadata, db_type, user_preferences)
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 构建火山引擎API请求数据
            data = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的SQL查询生成器。根据用户的问题和数据库结构，生成准确、高效的SQL查询。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,  # 低温度确保一致性
                "max_tokens": 1000
            }
            
            # 如果启用思维链模式，添加相关参数
            if self.thinking:
                data["thinking"] = True
            
            response = requests.post(self.chat_url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                # 解析响应
                return self._parse_response(content, question)
            else:
                return {
                    "success": False,
                    "error": f"API调用失败: {response.text}",
                    "sql_query": "",
                    "explanation": ""
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"生成SQL时出错: {e}",
                "sql_query": "",
                "explanation": ""
            }
    
    def _build_prompt(self, 
                     question: str, 
                     metadata: Dict[str, Any], 
                     db_type: str,
                     user_preferences: List[str] = None) -> str:
        """
        构建提示词
        
        Args:
            question: 用户问题
            metadata: 数据库元数据
            db_type: 数据库类型
            user_preferences: 用户偏好
            
        Returns:
            格式化的提示词
        """
        
        # 数据库方言特定说明
        dialect_info = {
            "postgresql": "使用PostgreSQL语法，支持LIMIT、OFFSET、ILIKE等",
            "mysql": "使用MySQL语法，支持LIMIT、LIKE等",
            "sqlite": "使用SQLite语法，支持LIMIT、LIKE等",
            "oracle": "使用Oracle语法，支持ROWNUM、LIKE等",
            "sqlserver": "使用SQL Server语法，支持TOP、LIKE等"
        }
        
        dialect = dialect_info.get(db_type, "使用标准SQL语法")
        
        # 构建表结构信息
        tables_info = []
        for table_name, table_info in metadata.get("tables", {}).items():
            columns = []
            for col in table_info.get("columns", []):
                col_desc = f"{col['name']} ({col['type']})"
                if col.get('primary_key'):
                    col_desc += " [主键]"
                columns.append(col_desc)
            
            tables_info.append(f"表: {table_name}\n列: {', '.join(columns)}")
        
        # 构建用户偏好信息
        preferences_text = ""
        if user_preferences:
            preferences_text = f"\n用户偏好:\n" + "\n".join([f"- {pref}" for pref in user_preferences])
        
        prompt = f"""
请根据以下信息生成SQL查询：

数据库类型: {db_type}
{dialect}

数据库结构:
{chr(10).join(tables_info)}

用户问题: {question}
{preferences_text}

请生成一个SQL查询来回答用户的问题。要求：
1. 使用正确的数据库语法
2. 查询应该高效且准确
3. 如果涉及用户偏好，请在查询中体现
4. 提供查询的解释

请按以下格式返回：
SQL查询:
```sql
[你的SQL查询]
```

解释:
[查询的解释，包括使用的表、条件等]

相关表:
[列出查询中使用的主要表]
"""
        
        return prompt
    
    def _parse_response(self, content: str, original_question: str) -> Dict[str, Any]:
        """
        解析AI响应
        
        Args:
            content: AI响应内容
            original_question: 原始问题
            
        Returns:
            解析后的结果
        """
        try:
            # 提取SQL查询
            sql_start = content.find("```sql")
            sql_end = content.find("```", sql_start + 6)
            
            if sql_start != -1 and sql_end != -1:
                sql_query = content[sql_start + 6:sql_end].strip()
            else:
                # 尝试其他格式
                lines = content.split('\n')
                sql_query = ""
                for line in lines:
                    if line.strip().upper().startswith(('SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE')):
                        sql_query = line.strip()
                        break
            
            # 提取解释
            explanation_start = content.find("解释:")
            if explanation_start != -1:
                explanation = content[explanation_start:].split('\n', 1)[1].strip()
            else:
                explanation = "查询已生成，请检查结果。"
            
            # 提取相关表
            tables_start = content.find("相关表:")
            tables_used = []
            if tables_start != -1:
                tables_text = content[tables_start:].split('\n', 1)[1].strip()
                # 简单的表名提取
                import re
                table_matches = re.findall(r'\b\w+表\b|\b\w+_table\b|\b\w+_tbl\b', tables_text, re.IGNORECASE)
                tables_used = list(set(table_matches))
            
            return {
                "success": True,
                "sql_query": sql_query,
                "explanation": explanation,
                "tables_used": tables_used,
                "original_question": original_question
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"解析响应时出错: {e}",
                "sql_query": "",
                "explanation": content,
                "tables_used": [],
                "original_question": original_question
            }
    
    def optimize_query(self, sql_query: str, db_type: str) -> str:
        """
        优化SQL查询
        
        Args:
            sql_query: 原始SQL查询
            db_type: 数据库类型
            
        Returns:
            优化后的SQL查询
        """
        # 这里可以添加查询优化逻辑
        # 例如：添加LIMIT、优化JOIN顺序等
        
        optimized = sql_query
        
        # 为SELECT查询添加LIMIT（如果没有的话）
        if (sql_query.strip().upper().startswith('SELECT') and 
            'LIMIT' not in sql_query.upper() and 
            db_type in ['postgresql', 'mysql', 'sqlite']):
            optimized += " LIMIT 1000"
        
        return optimized


class QueryProcessor:
    """
    查询处理器
    
    整合SQL生成和数据库执行
    """
    
    def __init__(self, 
                 database_client: DatabaseClient,
                 sql_generator: SQLGenerator,
                 db_type: str = "postgresql"):
        self.db_client = database_client
        self.sql_generator = sql_generator
        self.db_type = db_type
    
    async def process_question(self, 
                             question: str, 
                             user_preferences: List[str] = None) -> Dict[str, Any]:
        """
        处理用户问题
        
        Args:
            question: 用户问题
            user_preferences: 用户偏好
            
        Returns:
            处理结果
        """
        try:
            # 获取数据库元数据
            metadata = await self.db_client.get_metadata()
            if "error" in metadata:
                return {
                    "success": False,
                    "error": f"获取数据库元数据失败: {metadata['error']}",
                    "answer": "无法连接到数据库，请检查连接设置。"
                }
            
            # 生成SQL查询
            sql_result = self.sql_generator.generate_sql(
                question, metadata, self.db_type, user_preferences
            )
            
            if not sql_result["success"]:
                return {
                    "success": False,
                    "error": sql_result["error"],
                    "answer": "无法生成SQL查询，请检查问题描述。"
                }
            
            # 优化查询
            optimized_sql = self.sql_generator.optimize_query(
                sql_result["sql_query"], self.db_type
            )
            
            # 执行查询
            query_result = await self.db_client.execute_query(optimized_sql)
            
            if not query_result["success"]:
                return {
                    "success": False,
                    "error": query_result["error"],
                    "answer": "查询执行失败，请检查SQL语法。",
                    "sql_query": optimized_sql
                }
            
            # 格式化结果
            answer = self._format_answer(query_result, sql_result["explanation"])
            
            return {
                "success": True,
                "answer": answer,
                "sql_query": optimized_sql,
                "tables_used": sql_result["tables_used"],
                "row_count": query_result.get("row_count", 0),
                "explanation": sql_result["explanation"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"处理问题时出错: {e}",
                "answer": "处理您的问题时出现错误，请稍后重试。"
            }
    
    def _format_answer(self, query_result: Dict[str, Any], explanation: str) -> str:
        """
        格式化查询结果
        
        Args:
            query_result: 查询结果
            explanation: 查询解释
            
        Returns:
            格式化的答案
        """
        if "data" not in query_result:
            return f"查询执行成功。{query_result.get('message', '')}"
        
        data = query_result["data"]
        row_count = len(data)
        
        if row_count == 0:
            return "查询完成，但没有找到匹配的数据。"
        
        # 构建答案
        answer_parts = [f"找到 {row_count} 条记录：\n"]
        
        # 显示前10条记录
        display_count = min(10, row_count)
        for i, row in enumerate(data[:display_count]):
            answer_parts.append(f"{i+1}. {dict(row)}")
        
        if row_count > display_count:
            answer_parts.append(f"\n... 还有 {row_count - display_count} 条记录")
        
        answer_parts.append(f"\n\n查询说明：{explanation}")
        
        return "\n".join(answer_parts)


# 使用示例
if __name__ == "__main__":
    # 配置
    ai_api_key = os.getenv("AI_API_KEY", "")
    ai_base_url = os.getenv("AI_BASE_URL", "https://api.volcengine.com/v1")
    ai_model = os.getenv("AI_MODEL", "doubao-seed-1-6-250615")
    thinking = os.getenv("THINKING", "true").lower() == "true"
    
    if not ai_api_key:
        print("请设置AI_API_KEY环境变量")
    else:
        # 创建SQL生成器
        sql_gen = SQLGenerator(
            ai_api_key=ai_api_key,
            ai_base_url=ai_base_url,
            model=ai_model,
            thinking=thinking
        )
        
        # 示例元数据
        sample_metadata = {
            "tables": {
                "customers": {
                    "columns": [
                        {"name": "id", "type": "integer", "primary_key": True},
                        {"name": "name", "type": "varchar"},
                        {"name": "email", "type": "varchar"}
                    ]
                },
                "orders": {
                    "columns": [
                        {"name": "id", "type": "integer", "primary_key": True},
                        {"name": "customer_id", "type": "integer"},
                        {"name": "amount", "type": "decimal"}
                    ]
                }
            }
        }
        
        # 测试SQL生成
        result = sql_gen.generate_sql(
            "显示所有客户及其订单总金额",
            sample_metadata,
            "postgresql"
        )
        
        print("SQL生成结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))