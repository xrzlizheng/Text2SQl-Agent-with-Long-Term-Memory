# 快速启动指南：火山引擎AI版本

## 🚀 5分钟快速开始

### 步骤1：获取火山引擎AI API密钥

1. 访问 [火山引擎控制台](https://console.volcengine.com/)
2. 注册/登录您的账户
3. 创建新项目或选择现有项目
4. 在AI服务中启用doubao-seed-1-6-250615模型
5. 获取API密钥

### 步骤2：配置环境变量

1. 复制配置示例文件：
   ```bash
   cp env_example.txt .env
   ```

2. 编辑.env文件，填入您的配置：
   ```bash
   # 火山引擎AI配置
   AI_API_KEY=your_actual_api_key_here
   AI_BASE_URL=https://api.volcengine.com/v1
   AI_MODEL=doubao-seed-1-6-250615
   THINKING=true
   EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
   
   # 数据库配置（可选）
   DB_TYPE=sqlite
   SQLITE_PATH=database.db
   ```

**🔒 安全提示**：
- `.env`文件包含敏感信息，已被添加到`.gitignore`中
- 请确保不要将`.env`文件提交到Git仓库
- 生产环境中请使用强密码和安全的API密钥

### 步骤3：安装依赖

```bash
pip install -r requirements.txt
```

**注意**：首次运行时会自动下载Qwen嵌入模型（约3GB），请确保网络连接正常。

### 步骤4：测试配置

```bash
# 测试火山引擎AI配置
python test_volcengine_config.py

# 测试Qwen嵌入模型
python test_embedding.py
```

如果看到"🎉 所有测试通过！配置正确。"，说明配置成功。

### 步骤5：启动应用

```bash
python memory_frontend.py
```

访问 http://localhost:7860 开始使用！

## 🔧 配置选项说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `AI_API_KEY` | 火山引擎AI API密钥（必需） | - |
| `AI_BASE_URL` | 火山引擎AI服务地址 | https://api.volcengine.com/v1 |
| `AI_MODEL` | 使用的AI模型 | doubao-seed-1-6-250615 |
| `THINKING` | 是否启用思维链模式 | true |
| `EMBEDDING_MODEL` | 嵌入模型 | Qwen/Qwen3-Embedding-8B |

## 🎯 使用示例

### 基本查询
```
用户：显示所有客户信息
AI：以下是所有客户的信息...
```

### 带记忆的查询
```
用户：我只对VIP客户感兴趣
AI：我理解您想专注于VIP客户。我会记住这个偏好。

用户：显示最近的订单
AI：以下是VIP客户的最近订单...
（注意：我已应用您之前表达的偏好）
```

## 🛠️ 故障排除

### 常见问题

1. **API密钥错误**
   - 检查AI_API_KEY是否正确设置
   - 确保API密钥有效且未过期

2. **网络连接问题**
   - 检查网络连接
   - 确认可以访问火山引擎API

3. **模型不可用**
   - 确认您的账户已启用doubao-seed-1-6-250615模型
   - 检查账户余额和配额

### 获取帮助

- 运行 `python test_volcengine_config.py` 进行诊断
- 查看控制台错误信息
- 检查.env文件配置

## 📚 更多信息

- [火山引擎AI文档](https://www.volcengine.com/docs/82379)
- [项目完整文档](README.md)
- [配置示例](env_example.txt) 