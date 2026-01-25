#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
查找测试视频脚本
从数据集目录中随机选择视频文件作为测试样本
"""

import os
import shutil
import random
import glob
from pathlib import Path
from typing import List, Tuple

# 设置输出编码为 UTF-8（Windows 兼容）
import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass


def get_file_size_mb(file_path: str) -> float:
    """获取文件大小（MB）"""
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)


def find_all_mp4_files(search_paths: List[str]) -> List[str]:
    """
    在所有指定路径中递归搜索 .mp4 文件
    
    Args:
        search_paths: 要搜索的路径列表
        
    Returns:
        找到的所有 .mp4 文件路径列表
    """
    all_videos = []
    
    for search_path in search_paths:
        # 检查路径是否存在
        if not os.path.exists(search_path):
            print(f"[跳过] 路径不存在: {search_path}")
            continue
        
        print(f"[搜索] 正在搜索: {search_path}")
        
        # 使用 glob 递归搜索所有 .mp4 文件
        pattern = os.path.join(search_path, "**", "*.mp4")
        videos = glob.glob(pattern, recursive=True)
        
        if videos:
            print(f"  -> 找到 {len(videos)} 个视频文件")
            all_videos.extend(videos)
        else:
            print(f"  -> 未找到视频文件")
    
    return all_videos


def copy_videos_to_test_samples(video_paths: List[str], output_dir: str, num_samples: int = 5) -> List[Tuple[str, str, str, float]]:
    """
    随机选择视频并复制到测试目录
    
    Args:
        video_paths: 所有找到的视频路径列表
        output_dir: 输出目录
        num_samples: 要选择的视频数量
        
    Returns:
        元数据列表，每个元素为 (原始路径, 原始文件名, 新文件名, 文件大小MB)
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 随机选择视频
    if len(video_paths) < num_samples:
        print(f"\n[警告] 只找到 {len(video_paths)} 个视频，将选择所有视频")
        selected_videos = video_paths
    else:
        selected_videos = random.sample(video_paths, num_samples)
    
    metadata = []
    
    print(f"\n[复制] 开始复制 {len(selected_videos)} 个视频到 {output_dir}...")
    
    for i, video_path in enumerate(selected_videos, 1):
        # 获取原始文件名
        original_filename = os.path.basename(video_path)
        
        # 生成新文件名
        new_filename = f"test_video_{i}.mp4"
        new_path = os.path.join(output_dir, new_filename)
        
        # 获取文件大小
        file_size_mb = get_file_size_mb(video_path)
        
        try:
            # 复制文件
            shutil.copy2(video_path, new_path)
            print(f"  [{i}/{len(selected_videos)}] {original_filename} -> {new_filename} ({file_size_mb:.2f} MB)")
            
            # 记录元数据
            metadata.append((video_path, original_filename, new_filename, file_size_mb))
        except Exception as e:
            print(f"  [错误] 复制失败: {video_path} - {str(e)}")
    
    return metadata


def save_metadata(metadata: List[Tuple[str, str, str, float]], output_file: str):
    """
    保存视频元数据到文件
    
    Args:
        metadata: 元数据列表
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("测试视频元数据\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"总共 {len(metadata)} 个测试视频\n\n")
        
        for i, (original_path, original_name, new_name, file_size) in enumerate(metadata, 1):
            f.write(f"测试视频 {i}:\n")
            f.write(f"  新文件名: {new_name}\n")
            f.write(f"  原始文件名: {original_name}\n")
            f.write(f"  原始路径: {original_path}\n")
            f.write(f"  文件大小: {file_size:.2f} MB\n")
            f.write("-" * 80 + "\n")


def print_summary(video_paths: List[str], metadata: List[Tuple[str, str, str, float]]):
    """
    打印摘要信息
    
    Args:
        video_paths: 所有找到的视频路径
        metadata: 已复制的视频元数据
    """
    print("\n" + "=" * 80)
    print("摘要信息")
    print("=" * 80)
    print(f"总共找到视频文件: {len(video_paths)} 个")
    print(f"已选择测试视频: {len(metadata)} 个")
    print("\n已复制的测试视频:")
    for i, (original_path, original_name, new_name, file_size) in enumerate(metadata, 1):
        print(f"  {i}. {new_name}")
        print(f"     原始文件: {original_name}")
        print(f"     原始路径: {original_path}")
        print(f"     文件大小: {file_size:.2f} MB")
        print()


def main():
    """主函数"""
    print("=" * 80)
    print("测试视频查找脚本")
    print("=" * 80)
    print()
    
    # 定义搜索路径（包括 dataset 目录）
    search_paths = [
        "./dataset",  # 当前目录下的 dataset
        "../dataset",  # 上一级目录的 dataset
        "./data",
        "../data",
        "/content/data",  # Colab 路径
        "/content/LLaMA-Factory/data",  # Colab 路径
        "/content/drive/MyDrive",  # Google Drive
    ]
    
    # 添加当前工作目录（如果 dataset 就在当前目录）
    current_dir = os.getcwd()
    if "dataset" in current_dir.lower():
        # 如果当前目录包含 dataset，也搜索当前目录
        search_paths.insert(0, current_dir)
    
    print("[步骤 1] 搜索视频文件...")
    print()
    
    # 查找所有视频文件
    all_videos = find_all_mp4_files(search_paths)
    
    # 去重（避免重复）
    all_videos = list(set(all_videos))
    
    print()
    print(f"[结果] 总共找到 {len(all_videos)} 个唯一的视频文件")
    
    # 检查是否找到视频
    if not all_videos:
        print("\n" + "=" * 80)
        print("\033[91m[错误] 未找到任何视频文件！\033[0m")  # 红色文字（如果终端支持）
        print("=" * 80)
        print("\n建议:")
        print("1. 检查数据集是否已下载并解压")
        print("2. 确认视频文件路径是否正确")
        print("3. 如果使用 Colab，可以使用以下命令下载 ActivityNet 样本:")
        print("   !huggingface-cli download activitynet/ActivityNet-QA --repo-type dataset")
        print("   或")
        print("   !wget <视频下载链接>")
        return
    
    # 设置随机种子（可选，用于可重复性）
    # random.seed(42)
    
    print("\n[步骤 2] 随机选择并复制视频...")
    
    # 输出目录
    output_dir = "./test_samples"
    
    # 复制视频
    metadata = copy_videos_to_test_samples(all_videos, output_dir, num_samples=5)
    
    # 保存元数据
    metadata_file = os.path.join(output_dir, "video_info.txt")
    save_metadata(metadata, metadata_file)
    print(f"\n[完成] 元数据已保存到: {metadata_file}")
    
    # 打印摘要
    print_summary(all_videos, metadata)
    
    print("=" * 80)
    print("脚本执行完成！")
    print("=" * 80)
    print(f"\n测试视频已保存到: {os.path.abspath(output_dir)}")
    print(f"元数据文件: {os.path.abspath(metadata_file)}")


if __name__ == "__main__":
    main()


