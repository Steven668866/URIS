# 🚀 URIS Colab A100 部署和优化指南

## 📋 目录
- [A100 GPU 优化配置](#a100-gpu-优化配置)
- [Colab 快速部署](#colab-快速部署)
- [性能优化建议](#性能优化建议)
- [Colab 专用代码](#colab-专用代码)
- [常见问题](#常见问题)

---

## 🎯 A100 GPU 优化配置

### 硬件优势
- **显存**: 40GB（远超本地 24GB）
- **计算能力**: 19.5 TFLOPS (FP32)
- **张量核心**: 支持 TF32, BF16, FP16
- **带宽**: 1555 GB/s
- **Flash Attention 2**: 完全支持

### 推荐配置

#### 1. 使用更高精度（不用 4-bit 量化）
```python
# A100 显存充足，可以使用 float16/bfloat16 全精度
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name_or_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,  # 或 torch.float16
    trust_remote_code=True,
    attn_implementation="flash_attention_2",  # A100 完全支持
)
```

#### 2. 提高视频采样率
```python
# A100 可以处理更多帧
video_content = {
    "type": "video",
    "video": video_path,
    "fps": 2.0,  # 从 0.5-1.0 提高到 2.0（每秒2帧）
    "max_pixels": 1920 * 1080,  # 从 720p 提高到 1080p
}
```

#### 3. 增加生成长度
```python
# A100 可以生成更长的文本
max_new_tokens = 4096  # 从 2048 提高到 4096
```

#### 4. 启用所有优化
```python
generation_kwargs = dict(
    use_cache=True,
    num_beams=2,  # A100 可以用 beam search（从1提高到2）
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
)
```

---

## 🚀 Colab 快速部署

### 方法一：完整 Notebook（推荐）

创建新 Colab Notebook，复制以下代码：

```python
# ====================================
# URIS Colab A100 部署脚本
# ====================================

# 1️⃣ 检查 GPU
!nvidia-smi

# 2️⃣ 安装依赖
!pip install -q streamlit torch transformers>=4.57.0 peft qwen-vl-utils accelerate bitsandbytes opencv-python numpy pillow

# 3️⃣ 安装 Flash Attention 2（A100 加速）
!pip install -q flash-attn --no-build-isolation

# 4️⃣ 克隆项目
!git clone https://github.com/yourusername/URIS.git
%cd URIS

# 5️⃣ 下载优化配置文件
!wget https://raw.githubusercontent.com/yourusername/URIS/main/app_colab_a100.py -O app.py

# 6️⃣ 启动应用（使用 ngrok 或 cloudflared）
# 选项 A: 使用 pyngrok
!pip install -q pyngrok
from pyngrok import ngrok
import threading

# 设置 ngrok token（从 https://dashboard.ngrok.com/get-started/your-authtoken 获取）
ngrok.set_auth_token("YOUR_NGROK_TOKEN")

# 启动 streamlit
def run_streamlit():
    !streamlit run app.py --server.port 8501

thread = threading.Thread(target=run_streamlit)
thread.start()

# 创建公开 URL
public_url = ngrok.connect(8501)
print(f"🌐 访问地址: {public_url}")

# 选项 B: 使用 Cloudflare Tunnel（无需 token）
!npm install -g localtunnel
!streamlit run app.py --server.port 8501 & npx localtunnel --port 8501
```

### 方法二：使用预制 Colab 配置文件

我会创建一个专门的 `app_colab_a100.py`，包含所有 A100 优化。

---

## ⚡ 性能优化建议

### 1. 量化策略

#### 选项 A：不量化（推荐 A100）
```python
# 40GB 显存足够，使用全精度
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name_or_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
```

**优点**: 
- 最高精度
- 最快推理速度
- 最佳质量

**显存占用**: ~18-20GB（A100 完全足够）

#### 选项 B：8-bit 量化（平衡）
```python
# 如果需要更多显存空间
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name_or_path,
    device_map="auto",
    load_in_8bit=True,
    trust_remote_code=True,
)
```

**优点**:
- 显存占用 ~10-12GB
- 质量损失小
- 速度略慢

#### 选项 C：4-bit 量化（不推荐 A100）
```python
# 仅在显存真的不够时使用
# A100 通常不需要
```

### 2. 视频处理优化

```python
# A100 优化配置
def get_optimal_video_config_a100(duration):
    """A100 GPU 专用视频配置"""
    if duration < 10:
        return {
            "fps": 2.0,  # 高帧率
            "max_pixels": 1920 * 1080,  # 1080p
        }
    elif duration < 30:
        return {
            "fps": 1.5,
            "max_pixels": 1920 * 1080,
        }
    elif duration < 60:
        return {
            "fps": 1.0,
            "max_pixels": 1280 * 720,
        }
    else:
        return {
            "fps": 0.5,
            "max_pixels": 1280 * 720,
        }
```

### 3. 批处理优化

```python
# A100 可以处理更大的 batch
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
    max_length=2048,  # 增加上下文长度
)
```

### 4. 生成参数优化

```python
generation_kwargs = dict(
    max_new_tokens=4096,  # A100: 4096, 本地: 2048
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    do_sample=True,
    num_beams=2,  # A100 可以用 beam search
    repetition_penalty=1.1,
    use_cache=True,
)
```

---

## 🔧 Colab 专用配置

### 1. 环境检测

```python
# 检测是否在 Colab 环境
import os
IN_COLAB = 'COLAB_GPU' in os.environ

if IN_COLAB:
    print("✅ 运行在 Google Colab")
    # Colab 专用配置
else:
    print("ℹ️ 运行在本地环境")
    # 本地配置
```

### 2. GPU 验证

```python
import torch

# 检查 GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name}")
    print(f"   显存: {gpu_memory:.1f} GB")
    
    # 验证是否是 A100
    if "A100" in gpu_name:
        print("🚀 检测到 A100 GPU，启用高性能配置")
        USE_A100_CONFIG = True
    else:
        print(f"⚠️ 当前 GPU 不是 A100，是 {gpu_name}")
        USE_A100_CONFIG = False
else:
    print("❌ 未检测到 GPU")
```

### 3. 文件上传/下载

```python
from google.colab import files
import shutil

# 上传视频
uploaded = files.upload()
for filename in uploaded.keys():
    print(f"✅ 上传文件: {filename} ({len(uploaded[filename])/1024:.1f} KB)")

# 下载结果
def download_result(filepath):
    files.download(filepath)
```

### 4. Google Drive 集成

```python
from google.colab import drive

# 挂载 Google Drive
drive.mount('/content/drive')

# 使用 Drive 中的视频
VIDEO_PATH = "/content/drive/MyDrive/URIS/videos/test.mp4"

# 保存模型缓存到 Drive（避免重复下载）
MODEL_CACHE = "/content/drive/MyDrive/URIS/model_cache"
os.environ['HF_HOME'] = MODEL_CACHE
```

---

## 📊 性能对比

| 配置 | 本地 (24GB) | Colab A100 (40GB) | 提升 |
|------|-------------|-------------------|------|
| 量化方式 | 4-bit NF4 | BF16 全精度 | 质量 ↑↑ |
| 视频采样 | 0.5-1.0 fps | 1.5-2.0 fps | 帧数 2-4x |
| 视频分辨率 | 720p | 1080p | 质量 ↑ |
| 最大输出 | 2048 tokens | 4096 tokens | 长度 2x |
| Beam Search | 1 (贪婪) | 2-4 | 质量 ↑ |
| 批处理 | 1 | 1-2 | 吞吐 ↑ |
| Flash Attention | 可选 | 必选 | 速度 ↑↑ |

**估计性能提升**: 
- 处理速度: **40-60% 更快**
- 输出质量: **显著提升**
- 可处理视频长度: **2-3x 更长**

---

## 💻 完整 A100 优化代码片段

### 修改 `load_model()` 函数

```python
@st.cache_resource
def load_model():
    """A100 优化版模型加载"""
    
    # 检测 GPU
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU")
    
    gpu_name = torch.cuda.get_device_name(0)
    is_a100 = "A100" in gpu_name
    
    if is_a100:
        print("🚀 检测到 A100 GPU - 启用高性能配置")
    else:
        print(f"⚠️ 当前 GPU: {gpu_name}")
    
    model_name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    if is_a100:
        # A100 配置：全精度 + Flash Attention 2
        try:
            import flash_attn
            print("✅ Flash Attention 2 已启用")
            
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,  # 全精度
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
        except ImportError:
            print("⚠️ Flash Attention 2 未安装，使用标准配置")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
    else:
        # 非 A100：使用 4-bit 量化
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
    
    # 启用 KV 缓存
    model.config.use_cache = True
    
    # 加载 LoRA adapter（如果存在）
    if os.path.exists(ADAPTER_PATH):
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        print(f"✅ LoRA adapter 已加载")
    
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    
    return model, processor
```

### 修改视频配置

```python
def get_video_config(video_path):
    """根据 GPU 类型和视频长度返回最优配置"""
    
    # 获取视频时长
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 10
    cap.release()
    
    # 检测 GPU 类型
    gpu_name = torch.cuda.get_device_name(0)
    is_a100 = "A100" in gpu_name
    
    if is_a100:
        # A100 配置：更高质量
        if duration < 10:
            return {"fps": 2.0, "max_pixels": 1920 * 1080}
        elif duration < 30:
            return {"fps": 1.5, "max_pixels": 1920 * 1080}
        elif duration < 60:
            return {"fps": 1.0, "max_pixels": 1280 * 720}
        else:
            return {"fps": 0.5, "max_pixels": 1280 * 720}
    else:
        # 其他 GPU：保守配置
        if duration < 10:
            return {"fps": 1.0, "max_pixels": 1280 * 720}
        elif duration < 30:
            return {"fps": 0.5, "max_pixels": 1280 * 720}
        else:
            return {"fps": 0.33, "max_pixels": 1280 * 720}
```

---

## 🐛 常见问题

### Q1: Colab 断线怎么办？
**A**: 
- 使用 Colab Pro 获得更长运行时间
- 定期保存检查点
- 使用 Google Drive 挂载保存进度

### Q2: 如何访问 Streamlit 应用？
**A**: 使用以下任一方法：
1. **ngrok**: 需要注册获取 token
2. **localtunnel**: 无需注册，但 URL 随机
3. **Cloudflare Tunnel**: 稳定可靠

### Q3: Flash Attention 2 安装失败？
**A**:
```bash
# 使用预编译版本
pip install flash-attn --no-build-isolation

# 或者跳过（性能稍降）
# 模型会自动回退到标准 attention
```

### Q4: 显存不够？
**A**: 
A100 有 40GB，理论上完全够用。如果遇到问题：
1. 检查是否有其他进程占用 GPU
2. 降低 `max_new_tokens`
3. 使用 8-bit 量化而非全精度

### Q5: 如何保存模型避免重复下载？
**A**:
```python
# 挂载 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 设置缓存目录
os.environ['HF_HOME'] = '/content/drive/MyDrive/.cache/huggingface'
```

---

## 📝 完整 Colab Notebook 模板

保存以下内容为 `URIS_Colab_A100.ipynb`:

```python
# ===================================
# URIS Colab A100 部署 Notebook
# ===================================

# %% [markdown]
# # 🎬 URIS Video Reasoning Assistant
# ## Google Colab A100 优化版
# 
# 本 Notebook 针对 A100 GPU 进行了全面优化，提供最佳性能。

# %% 1. 环境准备
print("📦 安装依赖...")
!pip install -q streamlit torch transformers>=4.57.0 peft qwen-vl-utils accelerate bitsandbytes opencv-python numpy pillow flash-attn --no-build-isolation

# %% 2. 检查 GPU
import torch
print("\n🔍 检查 GPU...")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name}")
    print(f"   显存: {gpu_memory:.1f} GB")
else:
    print("❌ 未检测到 GPU")
    raise RuntimeError("需要 GPU 运行")

# %% 3. 挂载 Google Drive（可选）
from google.colab import drive
drive.mount('/content/drive')

# 设置模型缓存到 Drive
import os
os.environ['HF_HOME'] = '/content/drive/MyDrive/.cache/huggingface'

# %% 4. 克隆项目
!git clone https://github.com/yourusername/URIS.git
%cd URIS

# %% 5. 下载 A100 优化配置
!wget https://raw.githubusercontent.com/yourusername/URIS/main/app_colab_a100.py -O app.py

# %% 6. 启动应用
!pip install -q pyngrok

from pyngrok import ngrok
import threading

# 设置 ngrok token（从 https://dashboard.ngrok.com 获取）
ngrok.set_auth_token("YOUR_NGROK_TOKEN_HERE")

def run_app():
    !streamlit run app.py --server.port 8501 --server.headless true

thread = threading.Thread(target=run_app, daemon=True)
thread.start()

# 等待 streamlit 启动
import time
time.sleep(10)

# 创建公开 URL
public_url = ngrok.connect(8501)
print(f"\n🌐 访问地址: {public_url}")
print(f"📝 提示: 在新标签页中打开上面的 URL")
```

---

## 🚀 下一步

1. **创建优化版本**：我会创建 `app_colab_a100.py`
2. **准备 Colab Notebook**：完整的部署脚本
3. **性能测试**：对比优化效果

是否需要我创建完整的 A100 优化版本文件？

---

**预期性能提升**：
- 处理速度: ⚡ **40-60% 更快**
- 视频质量: 📹 **2x 帧率 + 1080p**
- 输出长度: 📝 **2x (4096 tokens)**
- 分析质量: 🎯 **显著提升（全精度）**
