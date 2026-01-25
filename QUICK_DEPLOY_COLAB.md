# 🚀 Colab A100 快速部署指南

## ⚡ 一键部署（3 分钟）

### 方法 1: 使用 Colab Notebook（推荐）

1. **打开 Notebook**
   - 点击这里: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/URIS/blob/main/URIS_Colab_A100.ipynb)

2. **选择 A100 GPU**
   - 运行时 > 更改运行时类型 > GPU (A100)

3. **按顺序运行所有单元格**
   - 点击 "运行时" > "全部运行"
   - 等待 3-5 分钟

4. **完成！**
   - 打开生成的公开 URL 访问应用

---

### 方法 2: 命令行部署（高级）

在 Colab 中运行以下命令：

```bash
# 1. 安装依赖
!pip install -q streamlit torch transformers>=4.57.0 peft qwen-vl-utils accelerate bitsandbytes opencv-python numpy flash-attn --no-build-isolation

# 2. 克隆项目
!git clone https://github.com/yourusername/URIS.git && cd URIS

# 3. 挂载 Drive（保存缓存）
from google.colab import drive
drive.mount('/content/drive')
import os
os.environ['HF_HOME'] = '/content/drive/MyDrive/.cache/huggingface'

# 4. 启动应用
!pip install -q pyngrok
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN")  # 从 https://dashboard.ngrok.com 获取

import threading
def run(): !streamlit run app_colab_a100.py --server.port 8501
threading.Thread(target=run, daemon=True).start()

import time; time.sleep(15)
print(f"访问: {ngrok.connect(8501)}")
```

---

## 📊 A100 vs 本地性能对比

| 指标 | 本地 (24GB) | Colab A100 (40GB) | 提升 |
|------|-------------|-------------------|------|
| **量化方式** | 4-bit NF4 | BF16 全精度 | 质量 ↑↑ |
| **视频采样** | 0.5-1.0 fps | 1.5-2.0 fps | 2-4x |
| **分辨率** | 720p | 1080p | ↑ |
| **最大输出** | 2048 tokens | 8192 tokens | 4x |
| **Flash Attention** | 可选 | 必选 | 速度 ↑↑ |
| **处理速度** | 基准 | **40-60% 更快** | ⚡ |
| **显存占用** | ~7-9GB | ~18-20GB | - |
| **显存剩余** | 15GB | 20GB | - |

---

## 🎯 A100 优化详情

### 自动启用的优化

当检测到 A100 GPU 时，应用会自动：

1. **✅ 使用 BF16 全精度**（不量化）
   ```python
   torch_dtype=torch.bfloat16  # vs 4-bit NF4
   ```

2. **✅ 启用 Flash Attention 2**
   ```python
   attn_implementation="flash_attention_2"
   ```

3. **✅ 提高视频采样率**
   ```python
   短视频: 2.0 fps  # vs 1.0 fps
   中视频: 1.5 fps  # vs 0.5 fps
   ```

4. **✅ 支持 1080p 分辨率**
   ```python
   max_pixels = 1920 * 1080  # vs 1280 * 720
   ```

5. **✅ 增加最大输出长度**
   ```python
   max_new_tokens = 8192  # vs 2048
   ```

### 界面提示

A100 用户会在侧边栏看到：

```
🚀 检测到 A100 GPU - 已启用高性能配置
  ✓ 全精度推理 (BF16)
  ✓ 高帧率视频处理 (最高2.0 fps)
  ✓ 1080p 分辨率支持
  ✓ Flash Attention 2 加速
```

---

## 💡 使用建议

### 最佳实践

1. **利用显存优势**
   - 处理更长的视频（可达 60 秒+）
   - 使用更高的 Max New Tokens (4096-8192)
   - 启用更高的视频质量

2. **保存模型缓存**
   ```python
   # 挂载 Drive 避免重复下载
   from google.colab import drive
   drive.mount('/content/drive')
   os.environ['HF_HOME'] = '/content/drive/MyDrive/.cache/huggingface'
   ```

3. **定期保存进度**
   - Colab 可能会断线
   - 使用 Drive 保存重要对话

### 性能调优

**想要最快速度**:
- 使用 10-20 秒的视频
- Max New Tokens 设为 2048-4096
- Temperature 设为 0.7

**想要最高质量**:
- 使用 1080p 视频
- Max New Tokens 设为 6144-8192
- 启用高采样率

---

## 🔧 常见问题

### Q: 如何获取 ngrok token？
**A**: 
1. 访问 https://dashboard.ngrok.com/get-started/your-authtoken
2. 注册免费账号
3. 复制 token
4. 在 Notebook 中替换 `YOUR_NGROK_TOKEN_HERE`

### Q: Colab 会自动断线吗？
**A**: 
- 免费版: ~12 小时或闲置 90 分钟
- Colab Pro: 可运行更长时间
- 建议: 定期与页面交互

### Q: 模型下载需要多久？
**A**:
- 首次: 2-5 分钟（~14GB）
- 后续: 使用缓存，<30 秒

### Q: A100 和 T4/V100 有什么区别？
**A**:

| GPU | 显存 | 性能 | 推荐配置 |
|-----|------|------|---------|
| T4 | 16GB | 基础 | 4-bit 量化 |
| V100 | 16GB | 中等 | 8-bit 量化 |
| A100 | 40GB | 高级 | BF16 全精度 ✅ |

### Q: 可以用免费 Colab 吗？
**A**: 
可以，但：
- 免费版可能分配到 T4（16GB）
- 会自动使用 4-bit 量化
- A100 通常需要 Colab Pro

---

## 📝 文件说明

- `app_colab_a100.py` - A100 优化版应用
- `URIS_Colab_A100.ipynb` - 完整部署 Notebook
- `COLAB_A100_GUIDE.md` - 详细优化指南
- `QUICK_DEPLOY.md` - 本文件

---

## 🎓 进阶配置

### 自定义优化参数

编辑 `app_colab_a100.py` 中的配置：

```python
# 调整默认参数
DEFAULT_MAX_TOKENS_A100 = 6144  # 默认 4096
DEFAULT_FPS_SHORT_VIDEO = 2.5   # 默认 2.0

# 修改分辨率上限
MAX_RESOLUTION_A100 = 2560 * 1440  # 2K（默认 1080p）
```

### 启用更多优化

```python
# 使用更大的 batch size（如果内存够用）
generation_kwargs = dict(
    max_new_tokens=8192,
    num_beams=4,  # 使用更多 beams（从2提高到4）
    top_p=0.95,
    repetition_penalty=1.1,
)
```

---

## 🚀 下一步

1. **尝试部署**: 点击 Colab Notebook 链接
2. **阅读详细指南**: 查看 `COLAB_A100_GUIDE.md`
3. **加入讨论**: GitHub Discussions

---

**预期体验**:
- 📹 更高质量的视频分析
- ⚡ 更快的响应速度  
- 📝 更长、更详细的回答
- 🎯 更准确的理解

**享受 A100 的强大性能！** 🎉
