# 📚 详细使用指南

## 目录

1. [快速开始](#快速开始)
2. [API 配置](#api-配置)
3. [自定义生成](#自定义生成)
4. [数据质量优化](#数据质量优化)
5. [常见问题](#常见问题)
6. [进阶技巧](#进阶技巧)

---

## 快速开始

### Windows 用户

```batch
# 1. 配置环境变量
copy env.example .env
notepad .env  # 填写 API 密钥

# 2. 运行生成脚本
run.bat
```

### Linux/Mac 用户

```bash
# 1. 配置环境变量
cp env.example .env
nano .env  # 填写 API 密钥

# 2. 添加执行权限并运行
chmod +x run.sh
./run.sh
```

### 手动运行

```bash
# 安装依赖
pip install -r requirements.txt

# 生成数据集
python generate_data.py

# 验证数据集
python validate_dataset.py
```

---

## API 配置

### DeepSeek API (推荐)

**优点**: 高性价比，中文支持好，速度快

**配置**:
```env
OPENAI_API_KEY=sk-xxxxx  # 从 https://platform.deepseek.com 获取
BASE_URL=https://api.deepseek.com/v1
MODEL=deepseek-chat
NUM_SAMPLES=100
```

**费用估算**:
- 100 个样本约需 0.2-0.5 元人民币
- 1000 个样本约需 2-5 元人民币

### OpenAI GPT-4

**优点**: 质量最高，英文理解最好

**配置**:
```env
OPENAI_API_KEY=sk-xxxxx
BASE_URL=https://api.openai.com/v1
MODEL=gpt-4-turbo
NUM_SAMPLES=100
```

**费用估算**:
- 100 个样本约需 $3-5 USD
- 1000 个样本约需 $30-50 USD

### 其他兼容 API

任何支持 OpenAI API 格式的服务都可以使用：

- Azure OpenAI
- 国内大模型服务（智谱 GLM、百度文心等）
- 本地部署的 LLM（通过 vLLM/Text Generation WebUI）

---

## 自定义生成

### 调整样本数量

在 `.env` 文件中：

```env
NUM_SAMPLES=500  # 生成 500 个样本
```

或使用命令行参数：

```bash
NUM_SAMPLES=500 python generate_data.py
```

### 修改 Token 限制

如果你的显卡显存更大/更小，可以调整限制：

**编辑 `generate_data.py`，第 21 行**:

```python
# 对于 12GB 显存
MAX_TOKENS_PER_SAMPLE = 1200

# 对于 6GB 显存
MAX_TOKENS_PER_SAMPLE = 600

# 对于 4GB 显存
MAX_TOKENS_PER_SAMPLE = 400
```

### 调整并发数

控制同时发送的 API 请求数（第 201 行）：

```python
# 高速网络/高额度账户
semaphore = asyncio.Semaphore(20)

# 慢速网络/低额度账户
semaphore = asyncio.Semaphore(5)
```

---

## 数据质量优化

### 提高个性化程度

**编辑 `generation_prompt` (第 130-166 行)**:

在 "Assistant Response" 部分添加更严格的要求：

```python
3. **Assistant Response**: 
   - MUST explicitly mention at least TWO aspects of the user profile
   - MUST provide specific dietary/lifestyle reasoning
   - MUST suggest concrete actions
   - Use warm, conversational tone
   - 80-120 words
```

### 增加场景多样性

**编辑 LOCATIONS 列表 (第 63-70 行)**:

```python
LOCATIONS = [
    "Kitchen with modern appliances",
    "Living room with smart home devices",
    "Home office workspace",
    "Bedroom with morning sunlight",
    "Dining room table",
    "Kitchen pantry area",
    "Balcony garden",
    "Home gym corner",
    # 添加更多场景
    "Bathroom medicine cabinet",
    "Kids playroom",
    "Garage workshop",
    "Outdoor patio grill area"
]
```

### 添加更多用户画像维度

**在 UserProfile 模型中添加字段 (第 26-33 行)**:

```python
class UserProfile(BaseModel):
    dietary_preference: str
    allergies: Optional[str]
    favorite_cuisine: str
    health_goal: Optional[str]
    lifestyle: str
    # 新增字段
    age_group: str = Field(description="Young adult, Middle-aged, Senior")
    cooking_skill: str = Field(description="Beginner, Intermediate, Expert")
    time_availability: str = Field(description="5 min, 15 min, 30+ min")
```

---

## 常见问题

### Q1: 生成速度很慢怎么办？

**A**: 有几个方法：

1. **提高并发数** (如果 API 额度允许):
   ```python
   semaphore = asyncio.Semaphore(20)  # 增加到 20
   ```

2. **使用更快的模型**:
   ```env
   MODEL=gpt-3.5-turbo  # 比 GPT-4 快 3-5 倍
   ```

3. **批量生成**:
   ```bash
   # 分多次生成，每次 50 个
   NUM_SAMPLES=50 python generate_data.py
   ```

### Q2: Token 超限样本太多？

**A**: 调整生成提示的字数要求：

```python
# 在 generation_prompt 中
1. **Visual Description**: 100-150 words  # 从 150-200 减少
3. **Assistant Response**: 60-100 words   # 从 80-120 减少
```

### Q3: 内容质量不够个性化？

**A**: 

1. **使用更强的模型** (GPT-4 > GPT-3.5 > DeepSeek)
2. **提高 temperature**:
   ```python
   temperature=1.0,  # 从 0.9 提高到 1.0
   ```
3. **在提示中强调个性化**

### Q4: API 请求频繁失败？

**A**: 

1. **降低并发数**:
   ```python
   semaphore = asyncio.Semaphore(3)
   ```

2. **增加重试延迟** (第 215 行):
   ```python
   await asyncio.sleep(2)  # 从 1 秒增加到 2 秒
   ```

3. **检查 API 额度和余额**

### Q5: 如何验证数据集质量？

**A**: 使用验证脚本：

```bash
python validate_dataset.py dataset_personalization.json
```

查看输出的：
- Token 分布
- 个性化多样性
- 结构完整性

---

## 进阶技巧

### 1. 多轮对话生成

修改数据结构以支持多轮对话：

```python
"conversations": [
    {"from": "system", "value": "..."},
    {"from": "user", "value": "<image> First question"},
    {"from": "assistant", "value": "First response"},
    {"from": "user", "value": "Follow-up question"},
    {"from": "assistant", "value": "Follow-up response"}
]
```

### 2. 混合真实和合成数据

```python
# 1. 生成合成数据
python generate_data.py  # 产生 dataset_personalization.json

# 2. 手动标注一些真实场景数据
# 3. 合并数据集

import json

synthetic = json.load(open("dataset_personalization.json"))
real = json.load(open("dataset_real.json"))
combined = synthetic + real

json.dump(combined, open("dataset_final.json", "w"), indent=2)
```

### 3. 特定场景定制

如果只想生成厨房场景：

```python
LOCATIONS = [
    "Kitchen counter with fresh ingredients",
    "Kitchen stove with cooking pots",
    "Kitchen table with meal prep",
    "Open refrigerator with food items",
    "Kitchen pantry shelves"
]
```

### 4. A/B 测试不同提示

创建多个提示版本，生成不同数据集：

```bash
# 版本 A: 简短回复
python generate_data.py  # 输出 dataset_v1.json

# 修改提示，版本 B: 详细回复
# 编辑 generation_prompt
python generate_data.py  # 输出 dataset_v2.json

# 微调后对比效果
```

### 5. 使用真实图像描述

如果你有真实图像：

```python
from PIL import Image
import clip  # 或其他视觉模型

# 1. 用 CLIP 等模型生成图像描述
image = Image.open("kitchen_scene.jpg")
description = generate_image_description(image)

# 2. 将描述注入到 generation_prompt
```

---

## 性能基准

**测试环境**: Intel i7, 32GB RAM, 100Mbps 网络

| 配置 | 100 样本耗时 | 成功率 |
|------|-------------|--------|
| DeepSeek, 并发 10 | 3-5 分钟 | 98% |
| GPT-3.5, 并发 10 | 5-8 分钟 | 95% |
| GPT-4, 并发 5 | 10-15 分钟 | 99% |

---

## 联系与支持

遇到问题？

1. 查看错误日志
2. 检查 API 余额和额度
3. 阅读本指南的常见问题部分
4. 在项目 Issues 中提问

祝您微调顺利！🚀






