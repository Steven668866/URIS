#!/usr/bin/env python3
"""
测试模型加载和基本推理功能
"""

import torch
import sys
import os

# 添加当前目录到Python路径
sys.path.append('.')

def test_model_loading():
    """测试模型加载"""
    try:
        print("=" * 60)
        print("🧪 测试模型加载和推理")
        print("=" * 60)

        # 检查GPU可用性
        gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
        device_type = "CUDA" if torch.cuda.is_available() else "MPS" if torch.backends.mps.is_available() else "CPU"

        print(f"📊 设备信息:")
        print(f"   - PyTorch版本: {torch.__version__}")
        print(f"   - CUDA可用: {torch.cuda.is_available()}")
        print(f"   - MPS可用: {torch.backends.mps.is_available()}")
        print(f"   - 使用设备: {device_type}")

        if gpu_available:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "未知"
                print(f"   - GPU数量: {gpu_count}")
                print(f"   - GPU型号: {gpu_name}")
            else:
                print("   - Apple Silicon GPU (MPS)")
        else:
            print("   - 无GPU，使用CPU")
            return False

        print("\n🔄 正在加载模型...")

        # 导入必要的模块
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        from peft import PeftModel

        # 配置4bit量化
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        # 模型路径
        model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        adapter_path = "./Qwen2.5-VL-URIS-Final-LoRA"

        print(f"📥 加载基础模型: {model_name}")

        # 加载模型
        model_kwargs = {
            "quantization_config": quantization_config,
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs
        )

        # 检查并加载LoRA适配器
        if os.path.exists(adapter_path):
            print(f"🔧 加载LoRA适配器: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            print("✅ LoRA适配器加载成功")
        else:
            print(f"⚠️ LoRA适配器未找到: {adapter_path}")

        # 加载处理器
        print("🔄 加载处理器...")
        processor = AutoProcessor.from_pretrained(model_name)

        print("✅ 模型和处理器加载成功！")

        # 显示模型信息
        print("\n📊 模型信息:")
        print(f"   - 模型类型: {type(model).__name__}")
        print(f"   - 设备: {model.device}")

        # 计算模型参数量（估算）
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   - 总参数量: {total_params:,}")

        # 测试基本推理
        print("\n🧪 测试基本推理...")
        test_prompt = "你好，请介绍一下你自己。"

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": test_prompt}
        ]

        # 准备输入
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = processor(
            text=[text],
            return_tensors="pt"
        ).to(model.device)

        # 生成回复
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True
            )

        # 解码输出
        response = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        print("✅ 推理测试成功！")
        print(f"📝 输入: {test_prompt}")
        print(f"🤖 输出: {response}")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    print("\n" + "=" * 60)
    if success:
        print("🎉 模型测试通过！可以运行Streamlit应用了。")
        print("运行命令: streamlit run app.py")
    else:
        print("💥 模型测试失败，请检查配置和依赖。")
    print("=" * 60)
