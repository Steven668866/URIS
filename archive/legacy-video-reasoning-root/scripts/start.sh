#!/bin/bash

# URIS 快速启动脚本

echo "🎬 URIS Video Reasoning Assistant"
echo "=================================="
echo ""

# 检查是否在正确的目录
if [ ! -f "app.py" ]; then
    echo "❌ 错误: 请在 URIS 项目根目录运行此脚本"
    exit 1
fi

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 Python 3"
    echo "   请安装 Python 3.8 或更高版本"
    exit 1
fi

echo "✅ Python 版本: $(python3 --version)"

# 检查依赖
echo ""
echo "📦 检查依赖..."

if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "⚠️  未找到 streamlit，正在安装依赖..."
    pip3 install -r requirements.txt
else
    echo "✅ 依赖已安装"
fi

# 检查摄像头权限提示
echo ""
echo "📹 摄像头功能提示:"
echo "   - Mac 用户: 需要在系统设置中授权摄像头访问"
echo "   - 首次使用时系统会弹出权限请求"
echo ""

# 运行测试（可选）
read -p "🧪 是否运行功能测试？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "运行测试..."
    python3 test_features.py
    echo ""
    read -p "按 Enter 继续启动应用..." 
fi

# 启动应用
echo ""
echo "🚀 正在启动 URIS..."
echo ""
echo "提示:"
echo "  - 应用会在浏览器中自动打开"
echo "  - 默认地址: http://localhost:8501"
echo "  - 按 Ctrl+C 停止应用"
echo ""

streamlit run app.py
