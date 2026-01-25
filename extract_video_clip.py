#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频片段提取工具
使用 ffmpeg 从长视频中截取指定时长的片段
"""

import os
import sys
import subprocess
from pathlib import Path

# 设置输出编码为 UTF-8（Windows 兼容）
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass


def check_ffmpeg():
    """检查 ffmpeg 是否已安装"""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_video_duration(video_path):
    """获取视频时长（秒）"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
        return None
    except:
        return None


def extract_clip(input_video, output_video, start_time=0, duration=20):
    """
    提取视频片段
    
    Args:
        input_video: 输入视频路径
        output_video: 输出视频路径
        start_time: 开始时间（秒），默认 0（从开头）
        duration: 片段时长（秒），默认 20 秒
    """
    if not os.path.exists(input_video):
        print(f"[错误] 输入视频不存在: {input_video}")
        return False
    
    # 构建 ffmpeg 命令
    # -ss: 开始时间
    # -t: 持续时间
    # -c copy: 使用流复制（快速，不重新编码）
    # 如果使用 -c copy，可能不够精确，所以使用重新编码方式
    cmd = [
        'ffmpeg',
        '-i', input_video,
        '-ss', str(start_time),  # 开始时间
        '-t', str(duration),     # 持续时间
        '-c:v', 'libx264',        # 视频编码器
        '-c:a', 'aac',            # 音频编码器
        '-preset', 'fast',        # 编码速度预设
        '-y',                     # 覆盖输出文件
        output_video
    ]
    
    print(f"[处理] 正在提取视频片段...")
    print(f"  输入: {input_video}")
    print(f"  输出: {output_video}")
    print(f"  开始时间: {start_time} 秒")
    print(f"  时长: {duration} 秒")
    print()
    
    try:
        # 运行 ffmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"[成功] 视频片段已保存到: {output_video}")
            # 显示输出文件大小
            if os.path.exists(output_video):
                size_mb = os.path.getsize(output_video) / (1024 * 1024)
                print(f"[信息] 输出文件大小: {size_mb:.2f} MB")
            return True
        else:
            print(f"[错误] ffmpeg 执行失败:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"[错误] 执行失败: {str(e)}")
        return False


def main():
    """主函数"""
    print("=" * 80)
    print("视频片段提取工具")
    print("=" * 80)
    print()
    
    # 检查 ffmpeg
    if not check_ffmpeg():
        print("[错误] 未找到 ffmpeg！")
        print()
        print("请先安装 ffmpeg:")
        print("  Windows: 下载 https://ffmpeg.org/download.html 或使用 chocolatey: choco install ffmpeg")
        print("  Linux: sudo apt install ffmpeg")
        print("  Mac: brew install ffmpeg")
        print("  Colab: !apt install ffmpeg")
        return
    
    print("[检查] ffmpeg 已安装")
    print()
    
    # 获取输入视频路径
    if len(sys.argv) > 1:
        input_video = sys.argv[1]
    else:
        input_video = input("请输入视频文件路径: ").strip().strip('"').strip("'")
    
    if not input_video:
        print("[错误] 未提供视频路径")
        return
    
    # 检查文件是否存在
    if not os.path.exists(input_video):
        print(f"[错误] 文件不存在: {input_video}")
        return
    
    # 获取视频时长
    duration_total = get_video_duration(input_video)
    if duration_total:
        print(f"[信息] 视频总时长: {duration_total:.2f} 秒 ({duration_total/60:.2f} 分钟)")
        print()
    
    # 获取开始时间
    if len(sys.argv) > 2:
        start_time = float(sys.argv[2])
    else:
        start_input = input("开始时间（秒，默认 0，从开头开始）: ").strip()
        start_time = float(start_input) if start_input else 0.0
    
    # 获取片段时长
    if len(sys.argv) > 3:
        clip_duration = float(sys.argv[3])
    else:
        duration_input = input("片段时长（秒，默认 20）: ").strip()
        clip_duration = float(duration_input) if duration_input else 20.0
    
    # 验证参数
    if duration_total and start_time + clip_duration > duration_total:
        print(f"[警告] 开始时间 + 时长 ({start_time + clip_duration:.2f} 秒) 超过视频总时长 ({duration_total:.2f} 秒)")
        clip_duration = duration_total - start_time
        print(f"[调整] 自动调整为: {clip_duration:.2f} 秒")
        print()
    
    # 生成输出文件名
    input_path = Path(input_video)
    output_video = input_path.parent / f"{input_path.stem}_clip_{int(start_time)}s_{int(clip_duration)}s{input_path.suffix}"
    
    # 提取片段
    print()
    success = extract_clip(str(input_path), output_video, start_time, clip_duration))
    
    if success:
        print()
        print("=" * 80)
        print("提取完成！")
        print("=" * 80)
        print(f"\n输出文件: {output_video}")
        print("\n提示: 可以将提取的片段上传到应用进行测试")


if __name__ == "__main__":
    main()


