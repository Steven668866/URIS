# 🤖 Qwen2-VL 家庭机器人个性化数据集生成器

为 RTX 4060 (8GB VRAM) 生成优化的合成训练数据，用于微调多模态大语言模型。

## ✨ 特性

- ✅ **严格的 Token 限制**: 每个样本 < 800 tokens，防止 OOM 错误
- ✅ **个性化场景**: 基于用户画像生成定制化对话
- ✅ **异步并行生成**: 高效的批量数据生成
- ✅ **自动重试机制**: 确保所有样本符合 token 限制
- ✅ **ShareGPT 格式**: 兼容 LLaMA-Factory 训练框架
- ✅ **详细统计信息**: Token 分布、生成成功率等

## 📋 环境要求

- Python 3.10+
- OpenAI 兼容 API (DeepSeek / GPT-4 / 其他)

## 🚀 快速开始

### 1. 安装依赖

```bash
cd fine-tuning
pip install -r requirements.txt
```

### 2. 配置 API

复制环境变量模板并填写你的 API 信息：

```bash
cp env.example .env
```

编辑 `.env` 文件：

```env
OPENAI_API_KEY=sk-your-api-key-here
BASE_URL=https://api.deepseek.com/v1
MODEL=deepseek-chat
NUM_SAMPLES=100
```

**推荐配置:**
- **DeepSeek**: `BASE_URL=https://api.deepseek.com/v1`, `MODEL=deepseek-chat` (性价比高)
- **OpenAI**: `BASE_URL=https://api.openai.com/v1`, `MODEL=gpt-4`

### 3. 运行生成脚本

```bash
python generate_data.py
```

### 4. 输出文件

生成完成后，会产生两个文件：

- **`dataset_personalization.json`** - 训练数据集 (ShareGPT 格式)
- **`dataset_personalization_metadata.json`** - 元数据和统计信息

## 📊 数据格式

### ShareGPT 格式示例

```json
{
  "conversations": [
    {
      "from": "system",
      "value": "Current User Profile: Dietary: Vegetarian, Cuisine: Italian, Lifestyle: Busy professional"
    },
    {
      "from": "user",
      "value": "<image> What can I make for a quick dinner with these ingredients?"
    },
    {
      "from": "assistant",
      "value": "Based on your vegetarian preference and busy lifestyle, I'd suggest a quick 15-minute pasta primavera..."
    }
  ],
  "images": ["images/placeholder.jpg"]
}
```

## 🎯 数据集特点

### 用户画像维度

- **饮食偏好**: 素食、纯素、生酮、无限制等
- **过敏信息**: 坚果、海鲜、乳制品等
- **喜好菜系**: 意大利、中国、日本、墨西哥等
- **健康目标**: 减重、增肌、心脏健康等
- **生活方式**: 职场人士、学生、健身爱好者等

### 场景类型

- 厨房场景 (食材识别、烹饪建议)
- 客厅场景 (智能家居交互)
- 餐厅场景 (营养建议)
- 其他家居场景

## ⚙️ 配置选项

在 `.env` 文件中可以调整：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `NUM_SAMPLES` | 生成样本数量 | 100 |
| `MODEL` | 使用的模型 | deepseek-chat |
| `BASE_URL` | API 端点 | - |
| `OPENAI_API_KEY` | API 密钥 | - |

代码中的常量 (可在脚本中修改)：

| 常量 | 说明 | 默认值 |
|------|------|--------|
| `MAX_TOKENS_PER_SAMPLE` | 每个样本的 token 上限 | 800 |
| `MAX_RETRIES` | 重试次数 | 5 |
| `Semaphore` | 并发请求数 | 10 |

## 📈 生成统计

脚本运行后会显示：

```
✅ Successfully generated: 98/100 samples

📈 Token Statistics:
   Average: 654.3 tokens
   Min: 432 tokens
   Max: 798 tokens
   Limit: 800 tokens

💾 Dataset saved to: dataset_personalization.json
📦 File size: 245.67 KB
```

## 🔧 故障排除

### 问题 1: API 速率限制

**症状**: 大量请求失败

**解决方案**: 降低 `Semaphore` 值（代码第 201 行）

```python
semaphore = asyncio.Semaphore(5)  # 从 10 改为 5
```

### 问题 2: Token 超限频繁

**症状**: 生成成功率低

**解决方案**: 
1. 在生成提示中要求更短的响应
2. 降低 `temperature` 参数

### 问题 3: 内容质量不佳

**症状**: 生成的对话不够个性化

**解决方案**: 
1. 调整 `generation_prompt` 中的指令
2. 增加 `temperature` 提高多样性
3. 使用更强大的模型（如 GPT-4）

## 🎓 使用生成的数据集

### 使用 LLaMA-Factory 进行微调

1. 将数据集放入 LLaMA-Factory 数据目录
2. 配置训练脚本：

```yaml
dataset: dataset_personalization
model_name: Qwen2-VL-7B-Instruct
quantization: 4bit  # 8GB VRAM 必需
max_seq_length: 800
batch_size: 1
gradient_accumulation_steps: 8
```

3. 启动训练：

```bash
llamafactory-cli train config.yaml
```

## 📝 许可

本项目用于 URIS 家庭机器人核心模块的研究和开发。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系

如有问题，请在项目中创建 Issue。






