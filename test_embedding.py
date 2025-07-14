#!/usr/bin/env python3
"""
Qwen嵌入模型测试脚本
==================

专门用于测试Qwen/Qwen3-Embedding-8B嵌入模型的功能。

作者：xrzlizheng
版本：1.0
"""

import os
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np

def test_qwen_embedding():
    """测试Qwen嵌入模型"""
    
    # 加载环境变量
    load_dotenv()
    
    embedding_model = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
    
    print("🔧 Qwen嵌入模型测试")
    print("=" * 50)
    print(f"📋 模型名称: {embedding_model}")
    
    try:
        # 加载模型
        print("\n🔄 正在加载模型...")
        start_time = time.time()
        model = SentenceTransformer(embedding_model)
        load_time = time.time() - start_time
        print(f"✅ 模型加载成功！耗时: {load_time:.2f}秒")
        
        # 测试文本
        test_texts = [
            "这是一个测试文本",
            "用户偏好查看利率低于5%的贷款",
            "高风险贷款定义为信用评分低于650且贷款金额超过$200,000",
            "WC表示西海岸州：CA、OR和WA",
            "显示所有VIP客户的信息"
        ]
        
        print(f"\n🔍 测试{len(test_texts)}个文本的嵌入...")
        
        # 生成嵌入
        start_time = time.time()
        embeddings = model.encode(test_texts)
        encode_time = time.time() - start_time
        
        print(f"✅ 嵌入生成成功！耗时: {encode_time:.2f}秒")
        print(f"📊 嵌入向量维度: {embeddings.shape[1]}")
        print(f"📊 嵌入向量数量: {embeddings.shape[0]}")
        
        # 测试相似度计算
        print("\n🔍 测试相似度计算...")
        
        # 计算第一个文本与其他文本的相似度
        query_embedding = embeddings[0]
        similarities = []
        
        for i, text in enumerate(test_texts[1:], 1):
            similarity = np.dot(query_embedding, embeddings[i]) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embeddings[i])
            )
            similarities.append((i, text, similarity))
        
        print("📊 相似度结果（相对于第一个文本）:")
        for i, text, sim in similarities:
            print(f"  文本{i}: {text[:30]}... - 相似度: {sim:.4f}")
        
        # 测试单文本嵌入
        print("\n🔍 测试单文本嵌入...")
        single_text = "这是一个单独的测试文本"
        single_embedding = model.encode(single_text)
        print(f"✅ 单文本嵌入成功！维度: {len(single_embedding)}")
        
        # 测试批处理
        print("\n🔍 测试批处理嵌入...")
        batch_texts = ["文本1", "文本2", "文本3", "文本4", "文本5"]
        batch_embeddings = model.encode(batch_texts)
        print(f"✅ 批处理嵌入成功！形状: {batch_embeddings.shape}")
        
        print("\n🎉 所有测试通过！Qwen嵌入模型工作正常。")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print("\n📋 可能的解决方案：")
        print("1. 确保已安装sentence-transformers: pip install sentence-transformers")
        print("2. 确保网络连接正常，可以下载模型")
        print("3. 检查模型名称是否正确")
        print("4. 确保有足够的内存和磁盘空间")
        return False

def test_embedding_performance():
    """测试嵌入性能"""
    
    print("\n🚀 性能测试")
    print("=" * 30)
    
    try:
        embedding_model = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
        model = SentenceTransformer(embedding_model)
        
        # 准备测试数据
        test_texts = [f"这是第{i}个测试文本，用于性能测试。" for i in range(100)]
        
        print(f"📊 测试{len(test_texts)}个文本的嵌入性能...")
        
        # 测试单文本性能
        start_time = time.time()
        for text in test_texts[:10]:
            model.encode(text)
        single_time = time.time() - start_time
        print(f"⏱️  单文本处理（10个）: {single_time:.2f}秒")
        
        # 测试批处理性能
        start_time = time.time()
        model.encode(test_texts)
        batch_time = time.time() - start_time
        print(f"⏱️  批处理（{len(test_texts)}个）: {batch_time:.2f}秒")
        
        # 计算性能指标
        single_avg = single_time / 10
        batch_avg = batch_time / len(test_texts)
        speedup = single_avg / batch_avg if batch_avg > 0 else 0
        
        print(f"📈 批处理加速比: {speedup:.2f}x")
        print(f"📈 平均处理速度: {len(test_texts)/batch_time:.1f} 文本/秒")
        
        return True
        
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始Qwen嵌入模型测试\n")
    
    # 基本功能测试
    basic_ok = test_qwen_embedding()
    
    if basic_ok:
        # 性能测试
        performance_ok = test_embedding_performance()
        
        if performance_ok:
            print("\n🎉 所有测试通过！Qwen嵌入模型配置正确且性能良好。")
            print("您现在可以在智能查询助手中使用本地嵌入模型了。")
        else:
            print("\n⚠️  基本功能正常，但性能测试失败。")
    else:
        print("\n❌ 基本功能测试失败，请检查配置。")
        print("\n📋 配置检查清单：")
        print("1. 确保已安装sentence-transformers")
        print("2. 确保网络连接正常")
        print("3. 确保有足够的内存和磁盘空间")
        print("4. 检查EMBEDDING_MODEL环境变量设置")

if __name__ == "__main__":
    main() 