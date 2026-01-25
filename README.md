# 🎬 URIS - Video Reasoning Assistant

> **U**niversal **R**easoning and **I**nteractive **S**ystem for Video Analysis

基于 Qwen2.5-VL-7B 的智能视频问答助手，支持多轮对话、摄像头实时录制和深度推理。

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## ✨ 主要特性

### 🎥 多种视频输入方式
- **📤 上传视频**: 支持 MP4 格式视频上传
- **📹 摄像头录制**: 实时从摄像头录制视频（5-30秒）
- **💬 纯文字对话**: 无需视频，直接文字交互

### 🚀 性能优化
- **智能采样**: 根据视频长度自动调整帧采样率
- **KV缓存**: 生成速度提升 20-30%
- **内存优化**: 显存占用降低 22%
- **推理加速**: 整体响应速度提升 30-60%

### 🧠 智能分析
- **深度推理**: Chain-of-Thought 思考过程可视化
- **多轮对话**: 支持上下文理解和连续提问
- **用户偏好**: 记忆用户偏好，个性化响应

### 🎤 语音输入（可选）
- 实时语音转文字
- 支持连续语音识别

---

## 📸 功能演示

### 摄像头录制
```
1. 点击 "📹 测试摄像头" 确认设备正常
2. 选择录制时长（推荐 10-15 秒）
3. 点击 "🎬 开始录制"
4. 录制完成后即可提问
```

### 视频分析
```
提问: "视频中的人在做什么？"
回答: [详细的动作分析和场景描述]

提问: "他的动作标准吗？"
回答: [基于上下文的深入分析]
```

---

## 🚀 快速开始

### 1. 环境要求

- **Python**: 3.8 或更高
- **GPU**: 推荐使用 CUDA GPU 或 Apple Silicon（MPS）
- **内存**: 至少 16GB RAM（24GB 推荐）
- **摄像头**: 用于实时录制功能（可选）

### 2. 安装依赖

```bash
# 克隆项目
git clone https://github.com/yourusername/URIS.git
cd URIS

# 安装依赖
pip install -r requirements.txt

# 可选: 安装 Flash Attention 2 加速（仅 CUDA）
pip install flash-attn --no-build-isolation
```

### 3. 准备模型

模型会在首次运行时自动下载到 HuggingFace 缓存目录：

```bash
~/.cache/huggingface/hub/
```

或者手动下载：

```bash
python download_model.py
```

### 4. 启动应用

```bash
streamlit run app.py
```

应用会在浏览器中自动打开（默认 http://localhost:8501）

---

## 📖 使用指南

### 基础使用

1. **上传视频模式**
   - 点击 "Choose video file" 上传 MP4 视频
   - 在聊天框输入问题
   - 获得详细分析和回答

2. **摄像头模式**（⭐ 新功能）
   - 在侧边栏找到 "摄像头实时录制"
   - 点击 "开始录制" 捕获实时画面
   - 自动保存为视频并可立即分析

3. **纯文字模式**
   - 点击 "Skip Video, Start Text Chat"
   - 像使用普通聊天机器人一样对话

### 高级功能

#### 调整生成参数
- **Max New Tokens**: 控制回答长度（512-4096）
- **Temperature**: 控制回答随机性（0.1-2.0）

#### 用户偏好记忆
```
在侧边栏 "用户偏好记忆" 中添加偏好：
- "请用简洁的语言回答"
- "回答时添加幽默感"
- "使用专业术语"
```

系统会记住这些偏好，并在后续对话中应用。

---

## 🎯 性能对比

| 场景 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 短视频(5-10s) | ~8秒 | ~5秒 | ⚡ **37%** |
| 中视频(15-30s) | ~15秒 | ~8秒 | ⚡ **47%** |
| 长视频(60s+) | ~35秒 | ~15秒 | ⚡ **57%** |
| 生成速度 | 15 tok/s | 20 tok/s | ⚡ **33%** |
| 内存占用 | 18GB | 14GB | 💾 **-22%** |

---

## 📚 文档

- **[📹 摄像头功能指南](CAMERA_GUIDE.md)** - 摄像头录制详细使用说明
- **[🚀 优化指南](OPTIMIZATION_GUIDE.md)** - 性能优化技术细节
- **[🌐 部署指南](DEPLOYMENT_GUIDE.md)** - 生产环境部署
- **[🎤 语音输入](VOICE_INPUT_README.md)** - 语音功能配置

---

## 🛠️ 技术栈

- **模型**: Qwen2.5-VL-7B-Instruct
- **框架**: 
  - Streamlit (Web UI)
  - PyTorch (深度学习)
  - Transformers (模型加载)
  - OpenCV (视频处理)
- **优化**:
  - 4-bit 量化 (NF4)
  - LoRA 微调
  - Flash Attention 2（可选）
  - KV 缓存

---

## 📊 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        URIS System                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📹 Input Layer                                             │
│  ├─ Video Upload (MP4)                                      │
│  ├─ Camera Recording (OpenCV)                               │
│  ├─ Text Input                                              │
│  └─ Voice Input (Optional)                                  │
│                                                             │
│  🧠 Processing Layer                                        │
│  ├─ Video Preprocessing (Smart Sampling)                    │
│  ├─ Frame Extraction (Dynamic FPS)                          │
│  ├─ Qwen2.5-VL Model (4-bit Quantized)                     │
│  └─ LoRA Adapter (Fine-tuned)                               │
│                                                             │
│  💬 Output Layer                                            │
│  ├─ Streaming Response                                      │
│  ├─ Chain-of-Thought Display                                │
│  └─ Multi-turn Dialogue                                     │
│                                                             │
│  ⚡ Optimization Layer                                       │
│  ├─ KV Cache                                                │
│  ├─ Flash Attention 2                                       │
│  ├─ Memory Management                                       │
│  └─ GPU Acceleration                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎨 应用场景

### 教育培训
- 动作标准性分析
- 操作流程讲解
- 实验步骤指导

### 健康医疗
- 康复训练指导
- 运动姿势纠正
- 日常活动评估

### 安全监控
- 异常行为识别
- 场景理解分析
- 事件描述记录

### 娱乐创作
- 视频内容理解
- 剧本辅助创作
- 场景描述生成

---

## 🔧 配置说明

### 模型配置

编辑 `app.py` 中的配置：

```python
# LoRA adapter 路径
ADAPTER_PATH = "./Qwen2.5-VL-URIS-Final-LoRA"

# 本地模型路径（可选）
LOCAL_MODEL_PATH = None  # 或设置为本地路径

# 上下文窗口大小
MAX_CONTEXT_MESSAGES = 20  # 保留最近20条消息
```

### 视频处理配置

```python
# 视频采样策略（自动优化）
短视频(<10s): 1.0 fps
中视频(10-30s): 0.5 fps  
长视频(>30s): 0.33 fps

# 分辨率上限
max_pixels = 1280 * 720  # 720p
```

---

## 🐛 常见问题

<details>
<summary><b>Q: GPU 内存不足怎么办？</b></summary>

**解决方案**:
1. 降低 Max New Tokens (512-1024)
2. 使用更短的视频（10-15秒）
3. 清除对话历史
4. 重启应用
</details>

<details>
<summary><b>Q: 摄像头无法打开？</b></summary>

**解决方案**:
1. 检查摄像头是否被其他应用占用
2. 在系统隐私设置中授权访问
3. 尝试点击"测试摄像头"诊断问题
</details>

<details>
<summary><b>Q: 处理速度慢？</b></summary>

**解决方案**:
1. 使用 GPU 而非 CPU
2. 选择较短的视频
3. 确保已安装所有优化
4. 查看 [优化指南](OPTIMIZATION_GUIDE.md)
</details>

<details>
<summary><b>Q: 如何获得更准确的回答？</b></summary>

**建议**:
1. 提供清晰的视频（光线充足、画面稳定）
2. 提出具体明确的问题
3. 使用多轮对话逐步深入
4. 添加用户偏好指导回答风格
</details>

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 贡献指南
1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📜 更新日志

### v2.0.0 (2026-01-25)
- ✨ 新增摄像头实时录制功能
- 🚀 多项性能优化（提升 30-60%）
- 🎯 智能视频采样策略
- 💾 视频预处理缓存
- 🔧 UI 渲染优化

### v1.0.0 (2024-12-01)
- 🎉 初始版本发布
- 📤 视频上传和分析
- 💬 多轮对话支持
- 🧠 Chain-of-Thought 推理

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) - 强大的多模态模型
- [Streamlit](https://streamlit.io/) - 优秀的 Web 框架
- [OpenCV](https://opencv.org/) - 计算机视觉库
- ActivityNet 数据集 - 微调训练数据

---

## 📞 联系方式

- **问题反馈**: [GitHub Issues](https://github.com/yourusername/URIS/issues)
- **讨论交流**: [GitHub Discussions](https://github.com/yourusername/URIS/discussions)

---

<div align="center">

**如果这个项目对你有帮助，请给一个 ⭐ Star！**

Made with ❤️ by URIS Team

</div>
