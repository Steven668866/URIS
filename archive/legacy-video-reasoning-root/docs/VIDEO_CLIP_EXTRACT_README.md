# 视频片段提取工具使用说明

## 📋 概述

当上传长视频（如 3 分钟）到应用时，可能会因为视频太长而导致处理缓慢或卡死。解决方案是**只提取视频的精华片段（15-30 秒）**，然后上传这个短片段。

## 🚀 快速使用

### 方法一：使用批处理脚本（Windows，最简单）

```bash
# 从视频开头提取 20 秒
extract_clip.bat video.mp4 0 20

# 从第 10 秒开始，提取 30 秒
extract_clip.bat video.mp4 10 30
```

### 方法二：使用 Python 脚本（跨平台）

```bash
# 交互式使用
python extract_video_clip.py

# 命令行参数
python extract_video_clip.py video.mp4 0 20
python extract_video_clip.py video.mp4 10 30
```

### 方法三：直接使用 ffmpeg 命令

```bash
# 从开头提取 20 秒
ffmpeg -i input.mp4 -ss 0 -t 20 -c:v libx264 -c:a aac -preset fast output_clip.mp4

# 从第 10 秒开始，提取 30 秒
ffmpeg -i input.mp4 -ss 10 -t 30 -c:v libx264 -c:a aac -preset fast output_clip.mp4
```

## 📝 参数说明

- `-ss <时间>`: 开始时间（秒），例如 `0` 表示从开头，`10` 表示从第 10 秒
- `-t <时长>`: 片段时长（秒），例如 `20` 表示提取 20 秒
- `-c:v libx264`: 视频编码器（H.264）
- `-c:a aac`: 音频编码器（AAC）
- `-preset fast`: 编码速度预设（fast = 快速编码）

## 💡 使用建议

1. **选择精华片段**：
   - 对于吉他弹唱视频，前 20 秒通常就足够了
   - 选择最能体现视频内容的关键片段

2. **推荐时长**：
   - **15-30 秒**：最佳平衡点，既能包含足够信息，又不会太慢
   - **10-15 秒**：如果只需要快速测试
   - **30-60 秒**：如果内容比较复杂

3. **开始时间选择**：
   - 通常从开头（0 秒）开始就很好
   - 如果开头是黑屏或无关内容，可以跳过

## 🔧 安装 ffmpeg

如果提示找不到 ffmpeg，需要先安装：

### Windows
```bash
# 使用 chocolatey
choco install ffmpeg

# 或从官网下载
# https://ffmpeg.org/download.html
```

### Linux
```bash
sudo apt install ffmpeg
```

### Mac
```bash
brew install ffmpeg
```

### Colab
```python
!apt install ffmpeg
```

## 📊 示例

### 示例 1：提取吉他弹唱视频的前 20 秒

```bash
# Windows
extract_clip.bat guitar_video.mp4 0 20

# Linux/Mac/Colab
python extract_video_clip.py guitar_video.mp4 0 20
```

输出：`guitar_video_clip_0s_20s.mp4`

### 示例 2：提取视频中间 30 秒（从第 60 秒开始）

```bash
extract_clip.bat long_video.mp4 60 30
```

输出：`long_video_clip_60s_30s.mp4`

## ⚡ 性能对比

| 视频长度 | 处理时间 | 显存占用 | 推荐 |
|---------|---------|---------|------|
| 3 分钟（完整） | 30-60 秒 | 高 | ❌ 不推荐 |
| 30 秒片段 | 5-10 秒 | 低 | ✅ 推荐 |
| 20 秒片段 | 3-8 秒 | 低 | ✅ 最佳 |
| 15 秒片段 | 2-5 秒 | 低 | ✅ 快速测试 |

## 🎯 工作流程

1. **提取片段**：使用工具从长视频中提取 15-30 秒片段
2. **上传测试**：将片段上传到应用
3. **快速响应**：享受快速的处理速度！

---

**提示**：即使视频只有 20 秒，AI 也能很好地理解内容。对于吉他弹唱视频，前 20 秒通常就包含了足够的信息（旋律、弹奏动作等）。


