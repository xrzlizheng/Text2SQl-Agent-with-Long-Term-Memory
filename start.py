#!/usr/bin/env python3
"""
智能查询助手启动脚本
==================

这个脚本提供了简单的项目启动方式，包括：
- 环境检查
- 依赖验证
- 服务启动

使用方法：
    python start.py [--frontend|--backend]

作者：xrzlizheng
版本：1.0
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def check_python_version():
    """检查Python版本是否满足要求"""
    if sys.version_info < (3, 8):
        print("❌ 错误：需要Python 3.8或更高版本")
        print(f"当前版本：{sys.version}")
        return False
    print(f"✅ Python版本检查通过：{sys.version}")
    return True

def check_dependencies():
    """检查项目依赖是否已安装"""
    required_packages = [
        'gradio',
        'numpy', 
        # 火山引擎AI使用标准HTTP请求，无需额外SDK
        'requests',
        'dotenv'
    ]
    
    missing_packages = []
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少依赖包：{', '.join(missing_packages)}")
        print("请运行：pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包检查通过")
    return True

def check_environment():
    """检查环境变量配置"""
    env_file = Path('.env')
    if not env_file.exists():
        print("⚠️  警告：未找到.env文件")
        print("请创建.env文件并配置以下环境变量：")
        print("  AI_API_KEY=your_volcengine_api_key")
        print("  AI_BASE_URL=https://api.volcengine.com/v1")
        print("  AI_MODEL=doubao-seed-1-6-250615")
        print("  THINKING=true")
        print("  EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B")
        print("  PG_CONN_STRING=postgresql://user:password@localhost:5432/memorydb")
        return False
    print("✅ 环境配置文件检查通过")
    return True

def start_frontend():
    """启动前端界面"""
    print("🚀 启动智能查询助手前端界面...")
    try:
        subprocess.run([sys.executable, "memory_frontend.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动前端失败：{e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 前端服务已停止")
        return True

def start_backend():
    """启动后端服务"""
    print("🚀 启动智能查询助手后端服务...")
    try:
        subprocess.run([sys.executable, "memory_agent.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动后端失败：{e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 后端服务已停止")
        return True

def run_security_check():
    """运行安全检查"""
    print("🔒 运行安全检查...")
    try:
        subprocess.run([sys.executable, "security_check.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 安全检查失败：{e}")
        return False
    except FileNotFoundError:
        print("❌ 未找到security_check.py文件")
        return False
    return True

def show_help():
    """显示帮助信息"""
    print("""
智能查询助手启动脚本

使用方法：
    python start.py [选项]

选项：
    --frontend    启动前端界面（默认）
    --backend     启动后端服务
    --security    运行安全检查
    --help        显示此帮助信息

示例：
    python start.py              # 启动前端界面
    python start.py --frontend   # 启动前端界面
    python start.py --backend    # 启动后端服务
    python start.py --security   # 运行安全检查

注意事项：
    1. 确保已安装所有依赖：pip install -r requirements.txt
    2. 配置.env文件中的环境变量
    3. 如果使用PostgreSQL，确保数据库服务正在运行
    4. 定期运行安全检查确保配置安全
    """)

def main():
    """主函数"""
    print("=" * 50)
    print("智能查询助手启动脚本")
    print("=" * 50)
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help':
            show_help()
            return
        elif sys.argv[1] == '--backend':
            mode = 'backend'
        elif sys.argv[1] == '--frontend':
            mode = 'frontend'
        elif sys.argv[1] == '--security':
            mode = 'security'
        else:
            print(f"❌ 未知选项：{sys.argv[1]}")
            show_help()
            return
    else:
        mode = 'frontend'  # 默认启动前端
    
    # 执行检查
    print("\n🔍 执行系统检查...")
    if not check_python_version():
        return
    
    if not check_dependencies():
        return
    
    check_environment()  # 环境检查失败不阻止启动
    
    # 启动服务
    print(f"\n🎯 启动模式：{mode}")
    if mode == 'frontend':
        success = start_frontend()
    elif mode == 'backend':
        success = start_backend()
    elif mode == 'security':
        success = run_security_check()
    else:
        print(f"❌ 未知模式：{mode}")
        return
    
    if success:
        print("✅ 服务启动成功")
    else:
        print("❌ 服务启动失败")
        sys.exit(1)

if __name__ == "__main__":
    main() 