#!/usr/bin/env python3
"""
火山引擎AI配置测试脚本
====================

用于测试火山引擎AI API配置是否正确。

作者：xrzlizheng
版本：1.0
"""

import os
import requests
from dotenv import load_dotenv

def test_volcengine_config():
    """测试火山引擎AI配置"""
    
    # 加载环境变量
    load_dotenv()
    
    # 获取配置
    ai_api_key = os.getenv("AI_API_KEY", "")
    ai_base_url = os.getenv("AI_BASE_URL", "https://api.volcengine.com/v1")
    ai_model = os.getenv("AI_MODEL", "doubao-seed-1-6-250615")
    thinking = os.getenv("THINKING", "true").lower() == "true"
    
    print("🔧 火山引擎AI配置测试")
    print("=" * 50)
    
    # 检查API密钥
    if not ai_api_key:
        print("❌ 错误：未设置AI_API_KEY环境变量")
        print("请在.env文件中设置您的火山引擎AI API密钥")
        return False
    
    print(f"✅ AI_API_KEY: {'*' * (len(ai_api_key) - 8) + ai_api_key[-8:]}")
    print(f"✅ AI_BASE_URL: {ai_base_url}")
    print(f"✅ AI_MODEL: {ai_model}")
    print(f"✅ THINKING: {thinking}")
    
    # 测试API连接
    print("\n🔍 测试API连接...")
    
    try:
        headers = {
            "Authorization": f"Bearer {ai_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": ai_model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个测试助手。请回复'配置测试成功'。"
                },
                {
                    "role": "user",
                    "content": "请确认配置是否正确。"
                }
            ],
            "temperature": 0.1,
            "max_tokens": 50
        }
        
        # 如果启用思维链模式，添加相关参数
        if thinking:
            data["thinking"] = True
        
        response = requests.post(
            f"{ai_base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print(f"✅ API连接成功！")
            print(f"📝 响应内容: {content}")
            return True
        else:
            print(f"❌ API连接失败: {response.status_code}")
            print(f"📝 错误信息: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 网络连接错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 未知错误: {e}")
        return False

def test_embedding_config():
    """测试嵌入模型配置"""
    
    print("\n🔍 测试嵌入模型...")
    
    try:
        embedding_model = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
        
        # 测试本地SentenceTransformer模型
        print(f"正在加载本地嵌入模型: {embedding_model}")
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(embedding_model)
        test_text = "这是一个测试文本"
        embedding = model.encode(test_text)
        
        print(f"✅ 本地嵌入模型测试成功！")
        print(f"📊 嵌入向量维度: {len(embedding)}")
        print(f"📝 测试文本: {test_text}")
        
        return True
            
    except Exception as e:
        print(f"❌ 本地嵌入模型测试失败: {e}")
        print("尝试使用备用火山引擎AI API...")
        
        try:
            ai_api_key = os.getenv("AI_API_KEY", "")
            ai_base_url = os.getenv("AI_BASE_URL", "https://api.volcengine.com/v1")
            
            if not ai_api_key:
                print("❌ 备用API也失败：未设置AI_API_KEY")
                return False
            
            headers = {
                "Authorization": f"Bearer {ai_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "text-embedding-ada-002",
                "input": "测试文本"
            }
            
            response = requests.post(
                f"{ai_base_url}/embeddings",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = result["data"][0]["embedding"]
                print(f"✅ 备用API嵌入模型测试成功！")
                print(f"📊 嵌入向量维度: {len(embedding)}")
                return True
            else:
                print(f"❌ 备用API嵌入模型测试失败: {response.status_code}")
                print(f"📝 错误信息: {response.text}")
                return False
                
        except Exception as e2:
            print(f"❌ 备用API嵌入模型测试错误: {e2}")
            return False

def main():
    """主函数"""
    print("🚀 开始火山引擎AI配置测试\n")
    
    # 测试基本配置
    config_ok = test_volcengine_config()
    
    if config_ok:
        # 测试嵌入模型
        embedding_ok = test_embedding_config()
        
        if embedding_ok:
            print("\n🎉 所有测试通过！配置正确。")
            print("您现在可以运行智能查询助手了。")
        else:
            print("\n⚠️  嵌入模型测试失败，但基本配置正确。")
    else:
        print("\n❌ 配置测试失败，请检查您的设置。")
        print("\n📋 配置检查清单：")
        print("1. 确保已创建.env文件")
        print("2. 确保AI_API_KEY已正确设置")
        print("3. 确保网络连接正常")
        print("4. 确保火山引擎AI服务可用")

if __name__ == "__main__":
    main() 