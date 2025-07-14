"""
智能查询助手前端界面（无Denodo版本）
==================================

这个模块提供了基于Gradio的用户界面，用于与具有长期记忆功能的Text2SQL代理进行交互，支持多种数据库类型。

主要功能包括：
- 数据库连接配置
- 自然语言查询界面
- 记忆管理和查看
- 实时对话历史

作者：xrzlizheng
版本：1.0
"""

import os
import json
import asyncio
import gradio as gr
from dotenv import load_dotenv
from memory_agent import SmartQueryAssistant

# 加载环境变量
load_dotenv()

# 全局变量
assistant = None  # 智能助手实例
chat_history = []  # 聊天历史记录
memories_history = []  # 记忆历史记录
database_loaded = False  # 数据库加载状态
current_user = None  # 当前用户

# 配置常量
DB_TYPE = os.getenv("DB_TYPE", "sqlite")  # 数据库类型
AI_API_KEY = os.getenv("AI_API_KEY", "")  # 火山引擎AI API密钥
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://api.volcengine.com/v1")  # 火山引擎AI服务基础URL
AI_MODEL = os.getenv("AI_MODEL", "doubao-seed-1-6-250615")  # 火山引擎AI模型
THINKING = os.getenv("THINKING", "true").lower() == "true"  # 思维链模式
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

# PostgreSQL记忆存储配置
PG_CONN_STRING = os.getenv("PG_CONN_STRING", "")
USE_POSTGRES = bool(PG_CONN_STRING)

def test_database_connection(db_type, **kwargs):
    """
    测试数据库连接
    
    Args:
        db_type: 数据库类型
        **kwargs: 数据库连接参数
        
    Returns:
        连接测试结果
    """
    try:
        from database_client import DatabaseClientFactory
        
        # 创建数据库客户端
        client = DatabaseClientFactory.create_client(db_type, **kwargs)
        
        # 测试连接
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(client.test_connection())
            loop.close()
            
            if result:
                return "✅ 数据库连接成功！"
            else:
                return "❌ 数据库连接失败，请检查配置。"
        except Exception as e:
            loop.close()
            return f"❌ 连接测试失败: {e}"
            
    except ImportError as e:
        return f"❌ 缺少数据库驱动: {e}"
    except Exception as e:
        return f"❌ 连接测试失败: {e}"

def login(username, password=None):
    """
    登录并初始化智能助手
    
    Args:
        username: 用户名
        password: 密码（可选）
        
    Returns:
        登录状态消息
    """
    global assistant, current_user
    
    if not username:
        return "请输入用户名。"
    
    try:
        # 创建智能助手实例
        assistant = SmartQueryAssistant(
            db_type=DB_TYPE,
            db_config=DB_CONFIG[DB_TYPE],
            use_postgres_memory=USE_POSTGRES,
            postgres_conn_string=PG_CONN_STRING
        )
        
        # 设置用户
        assistant.set_user(username, password)
        current_user = username
        
        return f"✅ 登录成功。欢迎，{username}！"
    except Exception as e:
        return f"❌ 登录失败: {e}"

async def initialize_assistant():
    """
    初始化智能助手
    
    Returns:
        初始化状态消息
    """
    global assistant, database_loaded
    
    if not assistant or not current_user:
        return "请先登录。"
    
    # 初始化数据库
    success = await assistant.initialize_db()
    database_loaded = success
    
    if success:
        return f"✅ 数据库 '{DB_TYPE}' 已成功为用户 {current_user} 加载。您现在可以询问数据。"
    else:
        return f"❌ 加载数据库 '{DB_TYPE}' 失败。请检查您的连接设置。"

def load_database():
    """
    加载数据库并返回状态消息
    
    Returns:
        加载状态消息
    """
    if not current_user:
        return "请先登录。"
    
    result = asyncio.run(initialize_assistant())
    return result

async def process_message_async(message):
    """
    异步处理消息
    
    Args:
        message: 用户消息
        
    Returns:
        处理后的响应
    """
    global assistant, database_loaded, memories_history
    
    if not current_user:
        return "请先登录。"
        
    if not database_loaded:
        return "请先使用'加载数据库'按钮加载数据库。"
    
    # 处理消息并获取响应
    response = await assistant.process_message(message)
    
    # 获取最近提取的记忆
    if hasattr(assistant.agent, 'memories') and assistant.agent.memories:
        recent_memories = sorted(assistant.agent.memories, key=lambda m: m.created_at, reverse=True)[:5]
        recent_memory_texts = [f"- {memory.content}" for memory in recent_memories]
        memories_history.append("\n".join(recent_memory_texts))
    else:
        memories_history.append("未提取记忆")
    
    return response

def process_message(message, history):
    """
    处理用户消息并更新聊天历史
    
    Args:
        message: 用户消息
        history: 聊天历史
        
    Returns:
        更新后的聊天历史
    """
    global assistant, database_loaded, current_user
    
    if not current_user:
        return [{"role": "assistant", "content": "请先登录。"}]
        
    if not database_loaded:
        return [{"role": "assistant", "content": "请先使用'加载数据库'按钮加载数据库。"}]
    
    # 在新事件循环中运行异步函数
    response = asyncio.run(process_message_async(message))
    
    # 将响应格式化为消息字典列表
    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    
    return history

def get_categorized_memories():
    """
    从数据库获取分类的记忆
    
    Returns:
        分类记忆的格式化字符串
    """
    global assistant
    if not assistant or not hasattr(assistant, 'agent'):
        return "无可用记忆"
    
    # 获取所有记忆
    memories = assistant.agent.memories
    if not memories:
        return "无可用记忆"
    
    # 分类记忆
    categories = {
        'PREFERENCE': [],  # 偏好
        'TERM': [],        # 术语
        'METRIC': [],      # 指标
        'ENTITY': []       # 实体
    }
    
    # 按创建时间排序记忆，最新的在前
    sorted_memories = sorted(memories, key=lambda m: m.created_at, reverse=True)
    
    for memory in sorted_memories:
        content = memory.content
        if '[PREFERENCE]' in content:
            categories['PREFERENCE'].append(content.replace('[PREFERENCE]', '').strip())
        elif '[TERM]' in content:
            categories['TERM'].append(content.replace('[TERM]', '').strip())
        elif '[METRIC]' in content:
            categories['METRIC'].append(content.replace('[METRIC]', '').strip())
        elif '[ENTITY]' in content:
            categories['ENTITY'].append(content.replace('[ENTITY]', '').strip())
    
    # 格式化输出，每个类别显示最新的5个条目
    output = []
    for category, items in categories.items():
        if items:
            output.append(f"\n{category}:")
            for item in items[:5]:  # 每个类别只显示最新的5个记忆
                output.append(f"- {item}")
    
    return '\n'.join(output) if output else "无分类记忆可用"

def get_current_memories():
    """
    从智能助手获取当前记忆
    
    Returns:
        当前记忆的格式化字符串
    """
    global memories_history
    
    # 从数据库获取持久记忆
    persistent_memories = get_categorized_memories()
    
    # 获取当前会话记忆
    current_session = memories_history[-1] if memories_history else "当前会话中无最近记忆"
    
    # 组合两者并添加标题
    return f"""持久记忆:\n{persistent_memories}\n\n当前会话记忆:\n{current_session}"""

def logout():
    """
    用户登出功能
    
    Returns:
        登出状态消息
    """
    global assistant, current_user, database_loaded, chat_history, memories_history
    
    # 重置所有全局变量
    assistant = None
    current_user = None
    database_loaded = False
    chat_history = []
    memories_history = []
    
    return "✅ 已成功登出。"

def build_interface():
    """
    构建Gradio用户界面
    
    Returns:
        配置好的Gradio界面
    """
    # 创建界面标题和描述
    title = "智能查询助手：具有长期记忆的Text2SQL代理（无Denodo版本）"
    description = """
    这个智能助手能够记住您的偏好和术语，提供个性化的数据库查询体验。
    支持多种数据库类型：PostgreSQL、MySQL、SQLite等。
    
    **使用步骤：**
    1. 配置数据库连接参数
    2. 使用用户名登录
    3. 加载数据库
    4. 开始用自然语言询问数据
    5. 查看系统记住的关于您的偏好
    
    **功能特点：**
    - 跨会话记住用户偏好
    - 学习自定义术语和定义
    - 基于向量的相似性搜索
    - 用户隔离的记忆存储
    - 支持多种数据库类型
    """
    
    # 创建界面组件
    with gr.Blocks(title=title, theme=gr.themes.Soft()) as interface:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)
        
        with gr.Row():
            with gr.Column(scale=1):
                # 数据库配置部分
                gr.Markdown("## 🔧 数据库配置")
                db_type_dropdown = gr.Dropdown(
                    choices=["sqlite", "postgresql", "mysql"],
                    value=DB_TYPE,
                    label="数据库类型",
                    interactive=True
                )
                
                # SQLite配置
                with gr.Group(visible=DB_TYPE=="sqlite") as sqlite_config:
                    sqlite_path = gr.Textbox(
                        label="数据库文件路径",
                        value=DB_CONFIG["sqlite"]["database_path"],
                        placeholder="database.db"
                    )
                
                # PostgreSQL配置
                with gr.Group(visible=DB_TYPE=="postgresql") as postgres_config:
                    pg_conn_string = gr.Textbox(
                        label="连接字符串",
                        value=DB_CONFIG["postgresql"]["connection_string"],
                        placeholder="postgresql://user:pass@localhost:5432/dbname"
                    )
                
                # MySQL配置
                with gr.Group(visible=DB_TYPE=="mysql") as mysql_config:
                    mysql_host = gr.Textbox(
                        label="主机地址",
                        value=DB_CONFIG["mysql"]["host"],
                        placeholder="localhost"
                    )
                    mysql_port = gr.Number(
                        label="端口",
                        value=DB_CONFIG["mysql"]["port"],
                        placeholder="3306"
                    )
                    mysql_database = gr.Textbox(
                        label="数据库名",
                        value=DB_CONFIG["mysql"]["database"],
                        placeholder="database_name"
                    )
                    mysql_username = gr.Textbox(
                        label="用户名",
                        value=DB_CONFIG["mysql"]["username"],
                        placeholder="username"
                    )
                    mysql_password = gr.Textbox(
                        label="密码",
                        value=DB_CONFIG["mysql"]["password"],
                        type="password",
                        placeholder="password"
                    )
                
                test_connection_button = gr.Button("测试连接", variant="secondary")
                connection_status = gr.Textbox(label="连接状态", interactive=False)
                
                # 登录部分
                gr.Markdown("## 🔐 登录")
                username_input = gr.Textbox(
                    label="用户名",
                    placeholder="输入您的用户名",
                    type="text"
                )
                login_button = gr.Button("登录", variant="primary")
                login_status = gr.Textbox(label="登录状态", interactive=False)
                
                # 数据库加载部分
                gr.Markdown("## 🗄️ 数据库")
                load_db_button = gr.Button("加载数据库", variant="secondary")
                db_status = gr.Textbox(label="数据库状态", interactive=False)
                
                # 记忆查看部分
                gr.Markdown("## 🧠 记忆管理")
                view_memories_button = gr.Button("查看记忆", variant="secondary")
                memories_display = gr.Textbox(
                    label="您的记忆",
                    interactive=False,
                    lines=10,
                    max_lines=15
                )
                
                # 登出部分
                gr.Markdown("## 🚪 登出")
                logout_button = gr.Button("登出", variant="stop")
                logout_status = gr.Textbox(label="登出状态", interactive=False)
            
            with gr.Column(scale=2):
                # 聊天界面
                gr.Markdown("## 💬 智能查询")
                chat_interface = gr.Chatbot(
                    label="对话历史",
                    height=500,
                    show_label=True
                )
                message_input = gr.Textbox(
                    label="输入您的问题",
                    placeholder="例如：显示利率最高的贷款",
                    lines=2
                )
                send_button = gr.Button("发送", variant="primary")
                clear_button = gr.Button("清除对话", variant="secondary")
        
        # 设置事件处理
        def update_config_visibility(db_type):
            """更新配置组件的可见性"""
            return (
                gr.Group(visible=db_type=="sqlite"),
                gr.Group(visible=db_type=="postgresql"),
                gr.Group(visible=db_type=="mysql")
            )
        
        db_type_dropdown.change(
            fn=update_config_visibility,
            inputs=[db_type_dropdown],
            outputs=[sqlite_config, postgres_config, mysql_config]
        )
        
        def test_db_connection(db_type, sqlite_path, pg_conn_string, 
                             mysql_host, mysql_port, mysql_database, 
                             mysql_username, mysql_password):
            """测试数据库连接"""
            if db_type == "sqlite":
                return test_database_connection(db_type, database_path=sqlite_path)
            elif db_type == "postgresql":
                return test_database_connection(db_type, connection_string=pg_conn_string)
            elif db_type == "mysql":
                return test_database_connection(db_type, 
                                              host=mysql_host,
                                              port=int(mysql_port),
                                              database=mysql_database,
                                              username=mysql_username,
                                              password=mysql_password)
            return "请选择数据库类型"
        
        test_connection_button.click(
            fn=test_db_connection,
            inputs=[db_type_dropdown, sqlite_path, pg_conn_string,
                   mysql_host, mysql_port, mysql_database, mysql_username, mysql_password],
            outputs=[connection_status]
        )
        
        login_button.click(
            fn=login,
            inputs=[username_input],
            outputs=[login_status]
        )
        
        load_db_button.click(
            fn=load_database,
            inputs=[],
            outputs=[db_status]
        )
        
        view_memories_button.click(
            fn=get_current_memories,
            inputs=[],
            outputs=[memories_display]
        )
        
        logout_button.click(
            fn=logout,
            inputs=[],
            outputs=[logout_status]
        )
        
        # 聊天功能
        send_button.click(
            fn=process_message,
            inputs=[message_input, chat_interface],
            outputs=[chat_interface]
        ).then(
            fn=lambda: "",
            inputs=[],
            outputs=[message_input]
        )
        
        # 回车键发送消息
        message_input.submit(
            fn=process_message,
            inputs=[message_input, chat_interface],
            outputs=[chat_interface]
        ).then(
            fn=lambda: "",
            inputs=[],
            outputs=[message_input]
        )
        
        # 清除对话
        clear_button.click(
            fn=lambda: [],
            inputs=[],
            outputs=[chat_interface]
        )
    
    return interface

# 启动应用程序
if __name__ == "__main__":
    print("启动智能查询助手前端界面（无Denodo版本）...")
    print(f"数据库类型: {DB_TYPE}")
    print(f"使用PostgreSQL记忆存储: {USE_POSTGRES}")
    
    interface = build_interface()
    interface.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,       # 默认端口
        share=False,            # 不创建公共链接
        debug=True              # 启用调试模式
    ) 