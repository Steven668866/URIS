#!/usr/bin/env python3
"""
URIS 功能测试脚本
测试摄像头录制和性能优化功能
"""

import cv2
import time
import os
import sys
from datetime import datetime

def test_camera_availability():
    """测试摄像头是否可用"""
    print("\n" + "="*60)
    print("📹 测试 1: 摄像头可用性检测")
    print("="*60)
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 失败: 无法打开摄像头")
            print("   请检查:")
            print("   1. 摄像头是否连接")
            print("   2. 摄像头是否被其他应用占用")
            print("   3. 系统隐私设置是否允许访问")
            return False
        
        # 尝试读取一帧
        ret, frame = cap.read()
        if not ret:
            print("❌ 失败: 无法读取摄像头画面")
            cap.release()
            return False
        
        # 获取摄像头信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"✅ 成功: 摄像头工作正常")
        print(f"   分辨率: {width}x{height}")
        print(f"   帧率: {fps} FPS")
        print(f"   画面尺寸: {frame.shape}")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        return False

def test_opencv_video_recording():
    """测试 OpenCV 视频录制功能"""
    print("\n" + "="*60)
    print("🎬 测试 2: 视频录制功能")
    print("="*60)
    
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 失败: 无法打开摄像头")
            return False
        
        # 设置录制参数
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30
        duration = 3  # 录制3秒测试
        
        # 创建临时文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"test_recording_{timestamp}.mp4"
        
        # 创建 VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"⏺️  开始录制 {duration} 秒测试视频...")
        
        frame_count = 0
        total_frames = duration * fps
        start_time = time.time()
        
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                print("❌ 警告: 无法读取画面")
                break
            
            out.write(frame)
            frame_count += 1
            
            # 显示进度
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                print(f"   进度: {frame_count}/{total_frames} 帧 ({elapsed:.1f}s)")
        
        # 释放资源
        cap.release()
        out.release()
        
        # 检查文件
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024  # KB
            print(f"✅ 成功: 视频已保存")
            print(f"   文件: {output_path}")
            print(f"   大小: {file_size:.2f} KB")
            print(f"   帧数: {frame_count}")
            
            # 验证视频可读
            test_cap = cv2.VideoCapture(output_path)
            if test_cap.isOpened():
                test_frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"   验证: 视频可读，共 {test_frame_count} 帧")
                test_cap.release()
                
                # 清理测试文件
                try:
                    os.remove(output_path)
                    print(f"   清理: 测试文件已删除")
                except:
                    print(f"   提示: 请手动删除 {output_path}")
                
                return True
            else:
                print("❌ 失败: 视频文件损坏")
                return False
        else:
            print("❌ 失败: 视频文件未创建")
            return False
            
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        return False

def test_dependencies():
    """测试依赖项是否正确安装"""
    print("\n" + "="*60)
    print("📦 测试 3: 依赖项检查")
    print("="*60)
    
    dependencies = {
        'streamlit': 'streamlit',
        'torch': 'torch',
        'transformers': 'transformers',
        'opencv-python': 'cv2',
        'numpy': 'numpy',
        'peft': 'peft',
    }
    
    all_ok = True
    for package, import_name in dependencies.items():
        try:
            __import__(import_name)
            
            # 获取版本信息
            if import_name == 'cv2':
                version = cv2.__version__
            else:
                module = __import__(import_name)
                version = getattr(module, '__version__', 'unknown')
            
            print(f"✅ {package:20s} v{version}")
        except ImportError:
            print(f"❌ {package:20s} 未安装")
            all_ok = False
    
    return all_ok

def test_performance_features():
    """测试性能优化功能"""
    print("\n" + "="*60)
    print("⚡ 测试 4: 性能优化功能")
    print("="*60)
    
    try:
        import torch
        
        # 测试 GPU 可用性
        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available()
        
        print(f"CUDA 可用: {'✅ 是' if cuda_available else '❌ 否'}")
        print(f"MPS 可用: {'✅ 是' if mps_available else '❌ 否'}")
        
        if cuda_available:
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"GPU 数量: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
        
        # 测试推理模式
        with torch.inference_mode():
            test_tensor = torch.randn(10, 10)
            result = test_tensor * 2
        print("✅ torch.inference_mode() 工作正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        return False

def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("🧪 URIS 功能测试套件")
    print("="*60)
    print("\n开始运行测试...")
    
    results = {
        "依赖项检查": test_dependencies(),
        "摄像头可用性": test_camera_availability(),
        "视频录制功能": test_opencv_video_recording(),
        "性能优化功能": test_performance_features(),
    }
    
    # 输出总结
    print("\n" + "="*60)
    print("📊 测试结果总结")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20s}: {status}")
    
    # 计算通过率
    passed = sum(results.values())
    total = len(results)
    pass_rate = (passed / total) * 100
    
    print("\n" + "="*60)
    print(f"总通过率: {passed}/{total} ({pass_rate:.1f}%)")
    print("="*60)
    
    if passed == total:
        print("\n🎉 所有测试通过！系统已准备就绪。")
        print("\n下一步:")
        print("1. 运行: streamlit run app.py")
        print("2. 在浏览器中打开应用")
        print("3. 尝试使用摄像头录制功能")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查上述错误信息。")
        print("\n常见解决方案:")
        print("1. 安装缺失的依赖: pip install -r requirements.txt")
        print("2. 检查摄像头权限设置")
        print("3. 确保 GPU 驱动正确安装")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 未预期的错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
