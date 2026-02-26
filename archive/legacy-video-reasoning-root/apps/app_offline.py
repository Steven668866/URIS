#!/usr/bin/env python3
"""
URIS Video Reasoning Assistant - 离线演示版本
不加载模型，只展示界面功能
"""

import streamlit as st
import torch
import os
from pathlib import Path

# 配置页面
st.set_page_config(
    page_title="URIS Video Reasoning Assistant (离线演示)",
    page_icon="🎬",
    layout="wide"
)

# LoRA 适配器路径
ADAPTER_PATH = "./Qwen2.5-VL-URIS-Final-LoRA"

def main():
    # 标题
    st.title("🎬 URIS Video Reasoning Assistant (离线演示)")
    st.caption("4bit量化模式 | Qwen2.5-VL-7B | 离线界面演示")

    # 显示离线模式警告
    st.warning("🔌 当前处于离线演示模式 - 无法加载模型进行推理")
    st.info("💡 要启用完整功能，请确保网络连接正常，然后运行: `streamlit run app.py`")

    # 检查LoRA适配器
    if os.path.exists(ADAPTER_PATH):
        files = os.listdir(ADAPTER_PATH)
        st.success(f"✅ LoRA适配器已就绪: {len(files)} 个文件")
    else:
        st.error(f"❌ LoRA适配器未找到: {ADAPTER_PATH}")

    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 配置面板")

        # 模型参数（仅显示，不生效）
        st.subheader("模型参数")
        max_tokens = st.slider("Max New Tokens", 128, 4096, 2048, disabled=True)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.7, disabled=True)

        # 显存状态
        st.subheader("💾 内存状态")
        gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
        device_type = "CUDA" if torch.cuda.is_available() else "MPS" if torch.backends.mps.is_available() else "CPU"

        if gpu_available:
            if torch.backends.mps.is_available():
                st.info("📊 Apple Silicon GPU (MPS) - 24GB Unified Memory")
            else:
                st.info("📊 NVIDIA GPU (CUDA)")
        else:
            st.warning("⚠️ 未检测到GPU，使用CPU模式")

        # 用户偏好（模拟）
        st.subheader("✨ 用户偏好")
        preferences = ["回答要详细具体", "使用中文回复", "注重观察细节"]
        for pref in preferences:
            st.checkbox(pref, value=True, disabled=True)

        # 视频上传区域
        st.subheader("📤 视频上传")
        uploaded_file = st.file_uploader(
            "上传视频文件",
            type=['mp4'],
            help="选择MP4视频文件",
            disabled=True
        )

        if uploaded_file:
            st.video(uploaded_file)

        # 清除历史
        if st.button("🗑️ 清除对话历史", disabled=True):
            pass

    # 主界面
    st.header("💬 对话界面")

    # 显示模拟对话
    demo_messages = [
        {"role": "user", "content": "这个视频里的人在做什么？"},
        {"role": "assistant", "content": "抱歉，当前处于离线演示模式，无法进行实际的视频分析推理。要启用完整功能，请：\n\n1. 确保网络连接正常\n2. 重新运行: `streamlit run app.py`\n3. 等待模型下载完成（首次运行约需10-15分钟）\n\n下载完成后，你就可以上传视频并进行智能分析了！"},
    ]

    for msg in demo_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 模拟输入框
    user_input = st.chat_input("输入你的问题...", disabled=True)

    if user_input:
        st.info("💭 离线模式：输入已接收，但无法处理推理")

    # 使用说明
    st.divider()
    st.header("📚 使用说明")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 功能特性")
        st.markdown("""
        - 🎬 **视频理解**: 上传MP4视频，AI自动分析内容
        - 🧠 **深度思考**: 显示AI的推理过程
        - 💬 **多轮对话**: 连续提问，上下文保持
        - 🔧 **参数调节**: 调整推理参数
        - 💾 **内存优化**: 4bit量化，适配24G内存Mac
        """)

    with col2:
        st.subheader("🚀 部署信息")
        st.markdown(f"""
        - **量化模式**: 4bit NF4
        - **显存占用**: ~8GB (vs 原始28GB)
        - **硬件支持**: Apple Silicon MPS
        - **模型大小**: ~8GB
        - **适配器**: LoRA微调
        """)

    # 技术栈
    st.divider()
    st.header("🔧 技术栈")

    tech_cols = st.columns(4)
    tech_stack = [
        ("🤖 Qwen2.5-VL", "多模态大语言模型"),
        ("⚡ 4bit量化", "bitsandbytes优化"),
        ("🎯 LoRA适配器", "ActivityNet微调"),
        ("🍎 Apple Silicon", "MPS GPU加速")
    ]

    for i, (name, desc) in enumerate(tech_stack):
        with tech_cols[i]:
            st.markdown(f"**{name}**")
            st.caption(desc)

    # 故障排除
    st.divider()
    st.header("🔍 故障排除")

    with st.expander("❌ 网络连接问题"):
        st.markdown("""
        **问题**: 无法下载模型，显示网络连接错误

        **解决方案**:
        1. 检查网络连接: `ping huggingface.co`
        2. 使用代理（如果需要）
        3. 等待网络恢复后重试
        4. 联系网络管理员
        """)

    with st.expander("💾 显存不足"):
        st.markdown("""
        **问题**: GPU显存不足

        **解决方案**:
        1. 确保使用24G内存Mac
        2. 关闭其他GPU程序
        3. 调整Max Tokens参数
        4. 清除对话历史释放内存
        """)

if __name__ == "__main__":
    main()
