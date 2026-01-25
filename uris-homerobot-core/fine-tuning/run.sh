#!/bin/bash
# Quick start script for Linux/Mac
# Generates synthetic dataset for Qwen2-VL fine-tuning

echo "========================================"
echo "🤖 Qwen2-VL Dataset Generator"
echo "========================================"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found!"
    echo ""
    echo "Please create .env file from env.example:"
    echo "  1. Copy env.example to .env"
    echo "  2. Fill in your OPENAI_API_KEY"
    echo "  3. Configure BASE_URL and MODEL"
    echo ""
    echo "Quick command:"
    echo "  cp env.example .env && nano .env"
    echo ""
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Check if dependencies are installed
echo "📦 Checking dependencies..."
if ! python3 -c "import openai" 2>/dev/null; then
    echo "⚠️  Dependencies not installed. Installing now..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
else
    echo "✅ Dependencies installed"
fi

echo ""
echo "🚀 Starting dataset generation..."
echo ""

# Run the generator
python3 generate_data.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Generation failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "🎉 Generation Complete!"
echo "========================================"
echo ""

# Validate the dataset
if [ -f dataset_personalization.json ]; then
    echo "🔍 Validating dataset..."
    echo ""
    python3 validate_dataset.py
    echo ""
fi

echo ""
echo "📁 Files generated:"
ls -lh dataset_*.json 2>/dev/null || echo "No dataset files found"
echo ""

echo "✅ All done! You can now use dataset_personalization.json for fine-tuning."






