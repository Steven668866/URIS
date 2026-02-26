#!/usr/bin/env python3
"""
使用ModelScope（魔搭社区）下载Qwen2.5-VL模型
"""

import os
from modelscope import snapshot_download
import torch

def download_with_modelscope():
    """使用ModelScope下载模型"""
    print("=" * 60)
    print("📥 使用ModelScope（魔搭社区）下载Qwen2.5-VL模型")
    print("=" * 60)

    # 模型信息
    model_name = "qwen/Qwen2.5-VL-7B-Instruct"

    print(f"🎯 目标模型: {model_name}")
    print("📦 预计大小: ~8GB")
    print("🔗 下载源: 魔搭社区 (ModelScope)")

    try:
        # 设置缓存目录
        cache_dir = os.path.expanduser("~/.cache/modelscope/hub")

        print(f"📁 缓存目录: {cache_dir}")
        print("⏳ 开始下载模型... (这可能需要几分钟到几十分钟)")
        print()

        # 下载模型
        model_path = snapshot_download(
            model_name,
            cache_dir=cache_dir,
            revision="master"
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

        return True, model_path

    except Exception as e:
        print(f"❌ ModelScope下载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None

def create_huggingface_symlink(modelscope_path):
    """创建符号链接到HuggingFace缓存目录"""
    try:
        # ModelScope缓存路径
        ms_cache = os.path.expanduser("~/.cache/modelscope/hub")

        # HuggingFace缓存路径
        hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
        hf_model_dir = os.path.join(hf_cache, "models--Qwen--Qwen2.5-VL-7B-Instruct")

        # 创建HuggingFace目录结构
        os.makedirs(hf_model_dir, exist_ok=True)

        # 创建符号链接
        if os.path.exists(modelscope_path):
            # 找到ModelScope下载的实际模型目录
            model_dirs = [d for d in os.listdir(ms_cache) if d.startswith("qwen")]
            if model_dirs:
                ms_model_path = os.path.join(ms_cache, model_dirs[0])
                print(f"🔗 创建符号链接: {ms_model_path} -> {hf_model_dir}")

                # 如果目标已存在，先删除
                if os.path.exists(hf_model_dir):
                    import shutil
                    shutil.rmtree(hf_model_dir)

                # 创建符号链接
                os.symlink(ms_model_path, hf_model_dir)
                print("✅ 符号链接创建成功")
                return True
            else:
                print("❌ 未找到ModelScope下载的模型目录")
                return False
        else:
            print(f"❌ ModelScope路径不存在: {modelscope_path}")
            return False

    except Exception as e:
        print(f"❌ 创建符号链接失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("选择下载方式:")
    print("1. 使用ModelScope（魔搭社区）下载")
    print("2. 尝试其他镜像平台")
    print()

    success, model_path = download_with_modelscope()

    if success and model_path:
        print("\n🔗 创建与HuggingFace兼容的符号链接...")
        if create_huggingface_symlink(model_path):
            print("\n🎯 现在可以启动应用了！")
            print("命令: streamlit run app.py")
        else:
            print("\n⚠️ 符号链接创建失败，但模型已下载")
            print("你可能需要手动配置模型路径")

    print("\n" + "=" * 60)
    if success:
        print("✅ 模型下载成功！可以启动应用了。")
    else:
        print("❌ 模型下载失败，尝试其他方法。")
        print("\n其他解决方案:")
        print("1. 使用VPN连接网络")
        print("2. 在其他网络环境下下载")
        print("3. 联系网络管理员")
    print("=" * 60)
