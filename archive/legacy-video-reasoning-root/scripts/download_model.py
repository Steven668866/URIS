#!/usr/bin/env python3
"""
下载Qwen2.5-VL模型到本地缓存
"""

import os
from huggingface_hub import snapshot_download
import sys

def download_model():
    """下载Qwen2.5-VL模型"""
    print("=" * 60)
    print("📥 下载Qwen2.5-VL模型到本地缓存")
    print("=" * 60)

    # 模型信息
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    adapter_path = "./Qwen2.5-VL-URIS-Final-LoRA"

    print(f"🎯 目标模型: {model_name}")
    print("📦 预计大小: ~8GB")
    print()

    try:
        # 检查LoRA适配器是否存在
        if not os.path.exists(adapter_path):
            print(f"❌ LoRA适配器目录不存在: {adapter_path}")
            return False

        files = os.listdir(adapter_path)
        print(f"✅ LoRA适配器检查通过: {len(files)} 个文件")
        print()

        # 设置缓存目录
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        os.makedirs(cache_dir, exist_ok=True)

        print(f"📁 缓存目录: {cache_dir}")
        print("⏳ 开始下载模型... (这可能需要几分钟到几十分钟)")
        print()

        # 下载模型
        model_path = snapshot_download(
            repo_id=model_name,
            local_dir=None,  # 使用默认缓存
            local_dir_use_symlinks=False,
            cache_dir=cache_dir,
            resume_download=True,  # 支持断点续传
            max_workers=4,  # 并行下载
        )

        print()
        print("✅ 模型下载完成！")
        print(f"📁 模型路径: {model_path}")

        # 检查下载的文件
        if os.path.exists(model_path):
            files = os.listdir(model_path)
            print(f"📊 下载文件数量: {len(files)}")
            for file in sorted(files)[:10]:  # 显示前10个文件
                file_path = os.path.join(model_path, file)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  - {file}: {size_mb:.1f} MB")
        print()
        print("🎉 模型已准备就绪！现在可以运行应用了。")
        print("启动命令: streamlit run app.py")

        return True

    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = download_model()
    print("\n" + "=" * 60)
    if success:
        print("✅ 模型下载成功！可以启动应用了。")
    else:
        print("❌ 模型下载失败，请检查网络连接。")
    print("=" * 60)
