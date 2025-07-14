"""
数据库客户端模块
==============

数据库连接模块，支持多种数据库类型：
- PostgreSQL
- MySQL
- SQLite
- Oracle
- SQL Server

作者：xrzlizheng
版本：1.0
"""

import os
import json
import asyncio
import sqlite3
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod

# 数据库驱动导入
try:
    import psycopg2
    import psycopg2.extras
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import pymysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import cx_Oracle
    ORACLE_AVAILABLE = True
except ImportError:
    ORACLE_AVAILABLE = False

try:
    import pyodbc
    SQLSERVER_AVAILABLE = True
except ImportError:
    SQLSERVER_AVAILABLE = False


class DatabaseClient(ABC):
    """
    数据库客户端抽象基类
    """
    
    @abstractmethod
    async def get_metadata(self) -> Dict[str, Any]:
        """获取数据库元数据"""
        pass
    
    @abstractmethod
    async def execute_query(self, query: str) -> Dict[str, Any]:
        """执行SQL查询"""
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """测试数据库连接"""
        pass


class PostgreSQLClient(DatabaseClient):
    """
    PostgreSQL数据库客户端
    """
    
    def __init__(self, connection_string: str):
        self.conn_string = connection_string
        self.connection = None
    
    async def test_connection(self) -> bool:
        """测试PostgreSQL连接"""
        try:
            conn = psycopg2.connect(self.conn_string)
            conn.close()
            return True
        except Exception as e:
            print(f"PostgreSQL连接测试失败: {e}")
            return False
    
    async def get_metadata(self) -> Dict[str, Any]:
        """获取PostgreSQL数据库元数据"""
        try:
            conn = psycopg2.connect(self.conn_string)
            cursor = conn.cursor()
            
            # 获取所有表
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            tables = [row[0] for row in cursor.fetchall()]
            
            metadata = {"tables": {}}
            
            for table in tables:
                # 获取表结构
                cursor.execute(f"""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns 
                    WHERE table_name = '{table}'
                    ORDER BY ordinal_position
                """)
                columns = []
                for row in cursor.fetchall():
                    columns.append({
                        "name": row[0],
                        "type": row[1],
                        "nullable": row[2] == "YES",
                        "default": row[3]
                    })
                
                # 获取主键
                cursor.execute(f"""
                    SELECT column_name
                    FROM information_schema.key_column_usage
                    WHERE table_name = '{table}' AND constraint_name LIKE '%_pkey'
                """)
                primary_keys = [row[0] for row in cursor.fetchall()]
                
                # 获取外键
                cursor.execute(f"""
                    SELECT 
                        kcu.column_name,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM information_schema.table_constraints AS tc
                    JOIN information_schema.key_column_usage AS kcu
                        ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage AS ccu
                        ON ccu.constraint_name = tc.constraint_name
                    WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = '{table}'
                """)
                foreign_keys = []
                for row in cursor.fetchall():
                    foreign_keys.append({
                        "column": row[0],
                        "foreign_table": row[1],
                        "foreign_column": row[2]
                    })
                
                metadata["tables"][table] = {
                    "columns": columns,
                    "primary_keys": primary_keys,
                    "foreign_keys": foreign_keys
                }
            
            cursor.close()
            conn.close()
            
            return metadata
            
        except Exception as e:
            print(f"获取PostgreSQL元数据失败: {e}")
            return {"error": str(e)}
    
    async def execute_query(self, query: str) -> Dict[str, Any]:
        """执行PostgreSQL查询"""
        try:
            conn = psycopg2.connect(self.conn_string)
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            
            cursor.execute(query)
            
            if query.strip().upper().startswith('SELECT'):
                # SELECT查询
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                data = [dict(row) for row in rows]
                
                result = {
                    "success": True,
                    "data": data,
                    "columns": columns,
                    "row_count": len(data)
                }
            else:
                # INSERT/UPDATE/DELETE查询
                conn.commit()
                result = {
                    "success": True,
                    "affected_rows": cursor.rowcount,
                    "message": "查询执行成功"
                }
            
            cursor.close()
            conn.close()
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class MySQLClient(DatabaseClient):
    """
    MySQL数据库客户端
    """
    
    def __init__(self, host: str, port: int, database: str, username: str, password: str):
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
    
    async def test_connection(self) -> bool:
        """测试MySQL连接"""
        try:
            conn = pymysql.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password
            )
            conn.close()
            return True
        except Exception as e:
            print(f"MySQL连接测试失败: {e}")
            return False
    
    async def get_metadata(self) -> Dict[str, Any]:
        """获取MySQL数据库元数据"""
        try:
            conn = pymysql.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password
            )
            cursor = conn.cursor()
            
            # 获取所有表
            cursor.execute("SHOW TABLES")
            tables = [row[0] for row in cursor.fetchall()]
            
            metadata = {"tables": {}}
            
            for table in tables:
                # 获取表结构
                cursor.execute(f"DESCRIBE {table}")
                columns = []
                for row in cursor.fetchall():
                    columns.append({
                        "name": row[0],
                        "type": row[1],
                        "nullable": row[2] == "YES",
                        "key": row[3],
                        "default": row[4]
                    })
                
                # 获取主键
                primary_keys = [col["name"] for col in columns if col["key"] == "PRI"]
                
                # 获取外键
                cursor.execute(f"""
                    SELECT 
                        COLUMN_NAME,
                        REFERENCED_TABLE_NAME,
                        REFERENCED_COLUMN_NAME
                    FROM information_schema.KEY_COLUMN_USAGE
                    WHERE TABLE_SCHEMA = '{self.database}' 
                    AND TABLE_NAME = '{table}'
                    AND REFERENCED_TABLE_NAME IS NOT NULL
                """)
                foreign_keys = []
                for row in cursor.fetchall():
                    foreign_keys.append({
                        "column": row[0],
                        "foreign_table": row[1],
                        "foreign_column": row[2]
                    })
                
                metadata["tables"][table] = {
                    "columns": columns,
                    "primary_keys": primary_keys,
                    "foreign_keys": foreign_keys
                }
            
            cursor.close()
            conn.close()
            
            return metadata
            
        except Exception as e:
            print(f"获取MySQL元数据失败: {e}")
            return {"error": str(e)}
    
    async def execute_query(self, query: str) -> Dict[str, Any]:
        """执行MySQL查询"""
        try:
            conn = pymysql.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password
            )
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            
            cursor.execute(query)
            
            if query.strip().upper().startswith('SELECT'):
                # SELECT查询
                data = cursor.fetchall()
                columns = list(data[0].keys()) if data else []
                
                result = {
                    "success": True,
                    "data": data,
                    "columns": columns,
                    "row_count": len(data)
                }
            else:
                # INSERT/UPDATE/DELETE查询
                conn.commit()
                result = {
                    "success": True,
                    "affected_rows": cursor.rowcount,
                    "message": "查询执行成功"
                }
            
            cursor.close()
            conn.close()
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class SQLiteClient(DatabaseClient):
    """
    SQLite数据库客户端
    """
    
    def __init__(self, database_path: str):
        self.database_path = database_path
    
    async def test_connection(self) -> bool:
        """测试SQLite连接"""
        try:
            conn = sqlite3.connect(self.database_path)
            conn.close()
            return True
        except Exception as e:
            print(f"SQLite连接测试失败: {e}")
            return False
    
    async def get_metadata(self) -> Dict[str, Any]:
        """获取SQLite数据库元数据"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # 获取所有表
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            metadata = {"tables": {}}
            
            for table in tables:
                # 获取表结构
                cursor.execute(f"PRAGMA table_info({table})")
                columns = []
                for row in cursor.fetchall():
                    columns.append({
                        "name": row[1],
                        "type": row[2],
                        "nullable": not row[3],
                        "primary_key": bool(row[5])
                    })
                
                # 获取主键
                primary_keys = [col["name"] for col in columns if col["primary_key"]]
                
                # 获取外键
                cursor.execute(f"PRAGMA foreign_key_list({table})")
                foreign_keys = []
                for row in cursor.fetchall():
                    foreign_keys.append({
                        "column": row[3],
                        "foreign_table": row[2],
                        "foreign_column": row[4]
                    })
                
                metadata["tables"][table] = {
                    "columns": columns,
                    "primary_keys": primary_keys,
                    "foreign_keys": foreign_keys
                }
            
            cursor.close()
            conn.close()
            
            return metadata
            
        except Exception as e:
            print(f"获取SQLite元数据失败: {e}")
            return {"error": str(e)}
    
    async def execute_query(self, query: str) -> Dict[str, Any]:
        """执行SQLite查询"""
        try:
            conn = sqlite3.connect(self.database_path)
            conn.row_factory = sqlite3.Row  # 使结果可以按列名访问
            cursor = conn.cursor()
            
            cursor.execute(query)
            
            if query.strip().upper().startswith('SELECT'):
                # SELECT查询
                rows = cursor.fetchall()
                data = [dict(row) for row in rows]
                columns = list(data[0].keys()) if data else []
                
                result = {
                    "success": True,
                    "data": data,
                    "columns": columns,
                    "row_count": len(data)
                }
            else:
                # INSERT/UPDATE/DELETE查询
                conn.commit()
                result = {
                    "success": True,
                    "affected_rows": cursor.rowcount,
                    "message": "查询执行成功"
                }
            
            cursor.close()
            conn.close()
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class DatabaseClientFactory:
    """
    数据库客户端工厂类
    """
    
    @staticmethod
    def create_client(db_type: str, **kwargs) -> DatabaseClient:
        """
        创建数据库客户端
        
        Args:
            db_type: 数据库类型 (postgresql, mysql, sqlite, oracle, sqlserver)
            **kwargs: 数据库连接参数
            
        Returns:
            数据库客户端实例
        """
        db_type = db_type.lower()
        
        if db_type == "postgresql":
            if not POSTGRES_AVAILABLE:
                raise ImportError("PostgreSQL驱动未安装，请运行: pip install psycopg2-binary")
            return PostgreSQLClient(kwargs["connection_string"])
        
        elif db_type == "mysql":
            if not MYSQL_AVAILABLE:
                raise ImportError("MySQL驱动未安装，请运行: pip install pymysql")
            return MySQLClient(
                host=kwargs["host"],
                port=kwargs.get("port", 3306),
                database=kwargs["database"],
                username=kwargs["username"],
                password=kwargs["password"]
            )
        
        elif db_type == "sqlite":
            return SQLiteClient(kwargs["database_path"])
        
        elif db_type == "oracle":
            if not ORACLE_AVAILABLE:
                raise ImportError("Oracle驱动未安装，请运行: pip install cx_Oracle")
            # Oracle客户端实现（简化版）
            raise NotImplementedError("Oracle客户端暂未实现")
        
        elif db_type == "sqlserver":
            if not SQLSERVER_AVAILABLE:
                raise ImportError("SQL Server驱动未安装，请运行: pip install pyodbc")
            # SQL Server客户端实现（简化版）
            raise NotImplementedError("SQL Server客户端暂未实现")
        
        else:
            raise ValueError(f"不支持的数据库类型: {db_type}")


# 使用示例
if __name__ == "__main__":
    # PostgreSQL示例
    # client = DatabaseClientFactory.create_client(
    #     "postgresql",
    #     connection_string="postgresql://user:password@localhost:5432/dbname"
    # )
    
    # MySQL示例
    # client = DatabaseClientFactory.create_client(
    #     "mysql",
    #     host="localhost",
    #     port=3306,
    #     database="testdb",
    #     username="root",
    #     password="password"
    # )
    
    # SQLite示例
    # client = DatabaseClientFactory.create_client(
    #     "sqlite",
    #     database_path="test.db"
    # )
    
    print("数据库客户端模块加载完成")