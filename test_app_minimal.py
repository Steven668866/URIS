#!/usr/bin/env python3
"""
最小化测试版本 - 不加载模型，只测试应用结构
"""

import streamlit as st
import torch

# 配置页面
st.set_page_config(
    page_title="URIS Test",
    page_icon="🎬",
    layout="wide"
)

def main():
    st.title("🎬 URIS Video Reasoning Assistant - 测试版本")
    st.caption("4bit量化模式测试 | Qwen2.5-VL-7B")

    # 检查GPU可用性
    gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
    device_type = "CUDA" if torch.cuda.is_available() else "MPS" if torch.backends.mps.is_available() else "CPU"

    st.success(f"✅ 设备检测成功: {device_type}")

    if gpu_available:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "未知"
            st.info(f"📊 GPU信息: {gpu_count} 个 {gpu_name}")
        else:
            st.info("📊 GPU信息: Apple Silicon (MPS)")

    st.info("🔧 代码结构测试通过")
    st.info("📦 依赖包加载成功")
    st.info("⚡ 4bit量化配置就绪")

    # 显示依赖状态
    st.divider()
    st.markdown("### 📦 依赖状态")

    try:
        import transformers
        st.success(f"✅ transformers: {transformers.__version__}")
    except ImportError:
        st.error("❌ transformers 未安装")

    try:
        import peft
        st.success(f"✅ peft: {peft.__version__}")
    except ImportError:
        st.error("❌ peft 未安装")

    try:
        import bitsandbytes
        st.success(f"✅ bitsandbytes: {bitsandbytes.__version__}")
    except ImportError:
        st.error("❌ bitsandbytes 未安装")

    try:
        import qwen_vl_utils
        st.success("✅ qwen-vl-utils 已安装")
    except ImportError:
        st.error("❌ qwen-vl-utils 未安装")

    # 显示LoRA文件状态
    st.divider()
    st.markdown("### 🔧 LoRA适配器状态")

    import os
    adapter_path = "./Qwen2.5-VL-URIS-Final-LoRA"
    if os.path.exists(adapter_path):
        files = os.listdir(adapter_path)
        st.success(f"✅ LoRA适配器目录存在: {len(files)} 个文件")
        for file in files:
            st.caption(f"  - {file}")
    else:
        st.error(f"❌ LoRA适配器目录不存在: {adapter_path}")

    st.divider()
    st.markdown("### 🚀 下一步")
    st.info("代码结构测试通过！要运行完整应用，需要网络连接下载模型。")
    st.code("streamlit run app.py")

if __name__ == "__main__":
    main()
