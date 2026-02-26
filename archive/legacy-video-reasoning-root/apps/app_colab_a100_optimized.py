import streamlit as st
import torch

# 导入 transformers 并检查版本
import transformers
transformers_version = transformers.__version__

# 检查并导入 Qwen2.5-VL 模型类
try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
except ImportError as e:
    error_msg = f"""
    ❌ Import Error: Cannot import Qwen2_5_VLForConditionalGeneration
    
    Current transformers version: {transformers_version}
    
    Please ensure:
    1. transformers version >= 4.57.0
    2. Run: pip install --upgrade transformers
    
    If the problem persists, check:
    - Python environment is correct
    - No multiple transformers installations
    
    Error details: {str(e)}
    """
    st.error(error_msg)
    raise ImportError(error_msg) from e

from peft import PeftModel
from transformers import TextIteratorStreamer, BitsAndBytesConfig
import re
import tempfile
import os
import json
import gc
import glob
import time
from threading import Thread
import queue
from qwen_vl_utils import process_vision_info
import cv2
import numpy as np
from datetime import datetime

# 尝试导入实时语音转文字库（可选）
try:
    import websocket
    VOICE_INPUT_AVAILABLE = True
    WEBSOCKET_AVAILABLE = True
except ImportError:
    VOICE_INPUT_AVAILABLE = False
    WEBSOCKET_AVAILABLE = False
    st.warning("⚠️ Real-time voice input not available. Install with: pip install websocket-client")

# 语音转文字配置
STT_CONFIG = {
    "websocket_url": "ws://localhost:8765",  # 语音转文字服务地址
    "sample_rate": 16000,
    "channels": 1,
    "chunk_size": 1024,
    "buffer_duration": 0.1,  # 100ms 缓冲
    "connection_timeout": 10,  # WebSocket连接超时（秒）
    "receive_timeout": 30,     # 接收数据超时（秒）
    "max_reconnect_attempts": 3  # 最大重连次数
}

# 验证配置
def validate_stt_config():
    """Validate STT configuration parameters."""
    if not STT_CONFIG["websocket_url"].startswith(("ws://", "wss://")):
        raise ValueError("WebSocket URL must start with ws:// or wss://")

    if STT_CONFIG["sample_rate"] not in [8000, 16000, 22050, 44100]:
        raise ValueError("Sample rate must be one of: 8000, 16000, 22050, 44100")

    if STT_CONFIG["channels"] not in [1, 2]:
        raise ValueError("Channels must be 1 (mono) or 2 (stereo)")

    if STT_CONFIG["buffer_duration"] <= 0:
        raise ValueError("Buffer duration must be positive")

# 初始化时验证配置
try:
    validate_stt_config()
except ValueError as e:
    st.error(f"❌ Invalid STT configuration: {e}")
    st.stop()

# Configure page
st.set_page_config(
    page_title="URIS Video Reasoning Assistant",
    page_icon="🎬",
    layout="wide"
)

# LoRA adapter path
ADAPTER_PATH = "./Qwen2.5-VL-URIS-Final-LoRA"

# User preference file path
USER_PROFILE_PATH = "user_profile.json"

# Local model path (if model is already downloaded locally, set this path to avoid re-downloading)
# Example: LOCAL_MODEL_PATH = "./models/Qwen2.5-VL-7B-Instruct"
# If None, use HuggingFace cache
LOCAL_MODEL_PATH = None  # 可以设置为本地路径，如 "./models/Qwen2.5-VL-7B-Instruct"

# Base System Prompt - Ultimate Observer Mode
USER_PROFILE_PATH = "user_profile.json"

# Local model path (if model is already downloaded locally, set this path to avoid re-downloading)
# Example: LOCAL_MODEL_PATH = "./models/Qwen2.5-VL-7B-Instruct"
# If None, use HuggingFace cache
LOCAL_MODEL_PATH = None  # 可以设置为本地路径，如 "./models/Qwen2.5-VL-7B-Instruct"

# Base System Prompt - Ultimate Observer Mode
# ⚡ Optimized System Prompt - 80% shorter for maximum inference speed
BASE_SYSTEM_PROMPT = """
You are 'URIS', a detective-level observant family assistant. Provide **exhaustive, multi-paragraph analysis** (min 8 paragraphs).

**DIRECTIVES:**
1. **Visual Extraction:** Describe EVERY element: subjects, background, colors, lighting, textures, spatial relations, text/signs, movements, expressions, context clues, small details.
2. **Reasoning (<think>...</think>):** Stream of consciousness. Map scene → analyze details → infer context. Acknowledge uncertainty.
3. **Format:** Conversational, warm, comprehensive. NO lists unless essential.

**STRUCTURE:**
1. **Overview & Atmosphere** (1-2¶): General scene, mood, setting.
2. **Subject Deep Dive** (2-4¶): Physical details, clothing, positioning, actions.
3. **Environment & Background** (2-3¶): All background elements, architecture, objects.
4. **Colors, Lighting & Context** (2-3¶): Specific colors, lighting quality, time/weather, situation.
5. **Hidden Details & Engagement** (2-3¶): Small details, patterns, textures, ask user questions.

**CRITICAL:**
❌ BAD: "这是一个人在房间里工作。" (TOO SHORT!)
✅ GOOD: "我看到一个大约30-40岁的男性坐在宽敞明亮的办公室里。他穿着深蓝色衬衫和浅色裤子，坐在黑色皮质办公椅上。面前是棕色木质书桌，桌上放着银色笔记本电脑、白色咖啡杯、几本厚书和一盏台灯。背景墙是浅灰色，挂着两幅风景画。窗户在右侧，窗外可见绿色树木，说明是白天，光线从窗外照进来..." (continue with more details)

**Never be brief.** Count paragraphs before finishing (min 8).
"""

# Context window configuration
MAX_CONTEXT_MESSAGES = 20  # Maximum 20 messages to retain

# Initialize session_state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False
if "camera_mode" not in st.session_state:
    st.session_state.camera_mode = False
if "recording" not in st.session_state:
    st.session_state.recording = False
if "processed_video_cache" not in st.session_state:
    st.session_state.processed_video_cache = {}

def check_model_cache():
    """Check if model is already cached"""
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo_info in cache_info.repos:
            if "Qwen2.5-VL-7B-Instruct" in repo_info.repo_id:
                return True, repo_info.repo_path
        return False, None
    except Exception:
        return False, None

@st.cache_resource
def load_model():
    """
    🚀 A100 Ultra-Speed Optimized Model Loading
    - A100 (40GB): BF16 + Flash Attention 2 + Aggressive Optimizations
    - Other GPU: 4-bit quantization
    """
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA GPU required for Colab A100 deployment!")

    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    is_a100 = "A100" in gpu_name
    
    print("=" * 60)
    print(f"🖥️  GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    if is_a100:
        print(f"🚀 A100 ULTRA-SPEED MODE: BF16 + Flash Attn2 + Optimizations")
    else:
        print(f"⚠️  Non-A100: Using 4-bit quantization")
    print("=" * 60)
    
    model_name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"
    if LOCAL_MODEL_PATH and os.path.exists(LOCAL_MODEL_PATH):
        model_name_or_path = LOCAL_MODEL_PATH
    
    use_flash_attention = False
    if is_a100:
        try:
            import flash_attn
            use_flash_attention = True
            print("✅ Flash Attention 2 enabled")
        except ImportError:
            print("⚠️  Flash Attention 2 not installed (speed will be reduced)")
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            if is_a100:
                # ⚡ A100 ULTRA-SPEED CONFIG
                model_kwargs = {
                    "device_map": "auto",
                    "torch_dtype": torch.bfloat16,
                    "trust_remote_code": True,
                    # ⚡ Speed optimizations
                    "low_cpu_mem_usage": True,  # Faster loading
                }
                
                if use_flash_attention:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                
                print("📥 Loading model (BF16 full precision)...")
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name_or_path,
                    **model_kwargs
                )
                print(f"✅ 模型加载完成（预计显存占用: ~18-20GB）")
                
            else:
                # 其他 GPU 配置：4-bit 量化
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                
                model_kwargs = {
                    "quantization_config": quantization_config,
                    "device_map": "auto",
                    "trust_remote_code": True,
                    "torch_dtype": torch.bfloat16,
                }
                
                print("📥 加载模型（4-bit 量化）...")
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name_or_path,
                    **model_kwargs
                )
                print(f"✅ 模型加载完成（预计显存占用: ~7-9GB）")
            
            break  # 成功加载
            
        except RuntimeError as e:
            if "size mismatch" in str(e) and attempt < max_retries - 1:
                print("⚠️  检测到权重大小不匹配，清除缓存后重试...")
                import shutil
                cache_path = os.path.expanduser(
                    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                )
                if os.path.exists(cache_path):
                    shutil.rmtree(cache_path, ignore_errors=True)
                    print("   缓存已清除，重新下载...")
                continue
            else:
                raise
    
    # 启用 KV 缓存（加速生成）
    model.config.use_cache = True
    print("✅ KV 缓存已启用")
    
    # 加载 LoRA adapter（如果存在）
    if os.path.exists(ADAPTER_PATH):
        print(f"📥 加载 LoRA adapter: {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        print("✅ LoRA adapter 已加载")
    
    # 加载 processor
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    
    print("=" * 60)
    print("🎉 模型加载完成，准备就绪！")
    print("=" * 60)
    
    return model, processor

def load_user_profile():
    """Load user preference file"""
    if os.path.exists(USER_PROFILE_PATH):
        try:
            with open(USER_PROFILE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("preferences", [])
        except Exception as e:
            st.error(f"Failed to read user preference file: {e}")
            return []
    return []

def save_user_profile(preferences):
    """Save user preferences to file"""
    try:
        data = {"preferences": preferences}
        with open(USER_PROFILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"Failed to save user preference file: {e}")
        return False

def build_system_prompt_with_preferences():
    """Build System Prompt including user preferences"""
    preferences = load_user_profile()
    
    if not preferences:
        return BASE_SYSTEM_PROMPT
    
    # 格式化用户偏好
    preferences_text = "[User Preferences]:\n"
    for i, pref in enumerate(preferences, 1):
        preferences_text += f"- {pref}\n"
    preferences_text += "\nPlease strictly follow these preferences in your response."
    
    # 组合完整的 System Prompt
    full_prompt = f"{BASE_SYSTEM_PROMPT}\n\n{preferences_text}"
    return full_prompt

def parse_model_output(text):
    """Parse model output, separate thinking process and final answer"""
    # 使用正则表达式提取 <think> 标签内的内容
    reasoning_pattern = r'<think>(.*?)</think>'
    reasoning_matches = re.findall(reasoning_pattern, text, re.DOTALL)
    
    # 提取思考过程
    reasoning = "\n\n".join(reasoning_matches).strip() if reasoning_matches else None
    
    # 移除思考过程标签，获取最终答案
    final_answer = re.sub(reasoning_pattern, "", text, flags=re.DOTALL).strip()
    
    return reasoning, final_answer

def get_gpu_memory_info():
    """Get GPU memory usage information"""
    # 检查是否有可用的GPU（CUDA或MPS）
    gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()

    if not gpu_available:
        return None, None, None

    try:
        if torch.cuda.is_available():
            # CUDA GPU
            device = torch.cuda.current_device()
            # 获取显存使用情况（字节）
            allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
            reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)  # GB
            # 获取总显存
            total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)  # GB
        else:
            # MPS (Apple Silicon) - 没有直接的显存查询API，返回估算值
            # 24GB unified memory on M2/M3 Macs
            total = 24.0  # 24GB unified memory
            # 估算当前使用量（无法精确获取，返回None表示未知）
            allocated = None
            reserved = None

        return allocated, reserved, total
    except Exception:
        return None, None, None

def cleanup_gpu_memory(force=False):
    """
    Clean up GPU memory

    Args:
        force: Whether to force cleanup (clean even if memory usage is not high)
    """
    # 检查是否有可用的GPU
    gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
    if not gpu_available:
        return

    try:
        allocated, reserved, total = get_gpu_memory_info()
        if allocated is None:
            # 如果无法获取精确的显存信息，仍然执行基本的清理
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            return

        usage_percent = (allocated / total) * 100 if total > 0 else 0

        # 如果显存占用超过 70% 或强制清理，执行清理
        if force or usage_percent > 70:
            # 清理 Python 垃圾回收
            gc.collect()

            # 根据GPU类型执行相应的清理
            if torch.cuda.is_available():
                # CUDA GPU
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            # MPS 不需要特殊的清理操作

            # 获取清理后的显存信息
            allocated_after, _, _ = get_gpu_memory_info()
            if allocated_after is not None:
                freed = allocated - allocated_after
                if freed > 0.1:  # 如果释放了超过 100MB，打印信息
                    print(f"[显存清理] 释放了 {freed:.2f} GB 显存 (清理前: {allocated:.2f} GB, 清理后: {allocated_after:.2f} GB)")
    except Exception as e:
        # 静默失败，不影响主流程
        pass

def clear_chat_history():
    """Clear chat history"""
    st.session_state.messages = []
    st.session_state.video_processed = False
    # 注意：不清除 video_path，以便用户可以继续使用同一个视频

def voice_processing_worker(session_state):
    """
    Real-time voice processing worker thread.
    Handles WebSocket connection and transcript processing for continuous STT.

    Args:
        session_state: Streamlit session state object containing voice session data
    """
    try:
        # Create WebSocket connection with timeout
        ws = websocket.create_connection(
            STT_CONFIG["websocket_url"],
            timeout=STT_CONFIG["connection_timeout"]
        )
        session_state.voice_session["websocket"] = ws

        # Send initial configuration
        init_message = {
            "type": "start",
            "sample_rate": STT_CONFIG["sample_rate"],
            "channels": STT_CONFIG["channels"]
        }
        ws.send(json.dumps(init_message))

        print("[Voice] Connected to STT service")

        while session_state.voice_session["active"]:
            try:
                # Receive transcript data from WebSocket with timeout
                ws.settimeout(STT_CONFIG["receive_timeout"])
                result = ws.recv()
                if result:
                    data = json.loads(result)

                    if data.get("type") == "transcript":
                        transcript = data.get("transcript", "").strip()
                        is_final = data.get("is_final", False)

                        # Validate transcript data
                        if not isinstance(transcript, str):
                            print(f"[Voice] Warning: Invalid transcript type: {type(transcript)}")
                            continue

                        if len(transcript) > 10000:  # Reasonable limit for transcript length
                            print("[Voice] Warning: Transcript too long, truncating")
                            transcript = transcript[:10000]

                        if transcript:
                            # Format the transcript text
                            formatted_transcript = format_transcript_text(transcript)

                            if is_final:
                                # Final transcript - append to final text
                                current_final = session_state.voice_session.get("final_text", "")
                                if current_final:
                                    session_state.voice_session["final_text"] = current_final + " " + formatted_transcript
                                else:
                                    session_state.voice_session["final_text"] = formatted_transcript
                                # Clear partial text
                                session_state.voice_session["partial_text"] = ""
                            else:
                                # Partial transcript - update live text
                                session_state.voice_session["partial_text"] = formatted_transcript

                    elif data.get("type") == "error":
                        print(f"[Voice Error] {data.get('message', 'Unknown error')}")
                        break

                # Small delay to prevent busy waiting
                time.sleep(0.01)

            except websocket.WebSocketTimeoutException:
                # Timeout - continue waiting for data
                continue
            except websocket.WebSocketConnectionClosedException:
                print("[Voice] WebSocket connection closed by server")
                break
            except json.JSONDecodeError as e:
                print(f"[Voice] Invalid JSON received: {e}")
                continue  # Skip this message and continue
            except Exception as e:
                print(f"[Voice] Error in processing loop: {e}")
                session_state.voice_session["error"] = f"Processing error: {e}"
                break

        # Send stop message
        try:
            ws.send(json.dumps({"type": "stop"}))
            ws.close()
        except:
            pass

        print("[Voice] Disconnected from STT service")

    except websocket.WebSocketException as e:
        error_msg = f"WebSocket connection error: {e}"
        print(f"[Voice] {error_msg}")
        session_state.voice_session["error"] = error_msg
        session_state.voice_session["active"] = False
    except Exception as e:
        error_msg = f"Voice processing error: {e}"
        print(f"[Voice] {error_msg}")
        session_state.voice_session["error"] = error_msg
        session_state.voice_session["active"] = False

def format_transcript_text(text):
    """
    Format and clean transcript text.

    Args:
        text (str): Raw transcript text

    Returns:
        str: Cleaned and formatted text
    """
    if not text:
        return ""

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Capitalize first letter of sentences (basic implementation)
    if text:
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

    return text

def cleanup_voice_session(session_state):
    """
    Clean up voice session resources properly.

    Args:
        session_state: Streamlit session state object
    """
    if "voice_session" in session_state:
        voice_session = session_state.voice_session

        # Stop the session
        voice_session["active"] = False

        # Close WebSocket connection
        if voice_session.get("websocket"):
            try:
                # Send stop message first
                voice_session["websocket"].send(json.dumps({"type": "stop"}))
                voice_session["websocket"].close()
                print("[Voice] WebSocket connection closed")
            except Exception as e:
                print(f"[Voice] Error closing WebSocket: {e}")

        # Wait for thread to finish
        if voice_session.get("thread") and voice_session["thread"].is_alive():
            voice_session["thread"].join(timeout=3.0)
            if voice_session["thread"].is_alive():
                print("[Voice] Warning: Thread did not finish within timeout")

        # Reset session state but keep transcript for user
        voice_session["thread"] = None
        voice_session["websocket"] = None
        voice_session["error"] = None

def record_video_from_camera(duration=10, fps=30):
    """
    🎥 从摄像头录制视频
    
    Args:
        duration: 录制时长（秒）
        fps: 帧率
    
    Returns:
        video_path: 保存的视频文件路径，如果失败返回 None
    """
    try:
        # 打开默认摄像头
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ 无法打开摄像头，请检查摄像头连接")
            return None
        
        # 获取摄像头属性
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 创建临时文件保存视频
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = tempfile.mktemp(suffix=f'_camera_{timestamp}.mp4')
        
        # 定义编解码器和创建 VideoWriter 对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (frame_width, frame_height))
        
        # 创建进度条和预览区域
        progress_bar = st.progress(0)
        status_text = st.empty()
        preview_placeholder = st.empty()
        
        frame_count = 0
        total_frames = duration * fps
        
        status_text.info(f"🎬 正在录制... ({duration}秒)")
        
        start_time = time.time()
        
        while frame_count < total_frames:
            ret, frame = cap.read()
            
            if not ret:
                st.warning("⚠️ 无法读取摄像头画面")
                break
            
            # 写入视频帧
            out.write(frame)
            
            # 每10帧更新一次预览（减少UI更新频率）
            if frame_count % 10 == 0:
                # 将 BGR 转换为 RGB 用于显示
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preview_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # 更新进度条
                progress = (frame_count + 1) / total_frames
                progress_bar.progress(progress)
                
                elapsed = time.time() - start_time
                remaining = duration - elapsed
                status_text.info(f"🎬 正在录制... 剩余 {remaining:.1f} 秒")
            
            frame_count += 1
            
            # 控制帧率
            time.sleep(1.0 / fps)
        
        # 释放资源
        cap.release()
        out.release()
        
        progress_bar.progress(1.0)
        status_text.success(f"✅ 录制完成！共 {frame_count} 帧")
        
        return temp_path
        
    except Exception as e:
        st.error(f"❌ 录制视频时出错: {str(e)}")
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()
        return None

def show_camera_preview():
    """
    显示摄像头实时预览
    """
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ 无法打开摄像头")
            return False
        
        st.info("📹 摄像头预览 (测试摄像头是否正常)")
        preview_placeholder = st.empty()
        
        # 显示5秒预览
        for i in range(50):  # 50帧约5秒
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preview_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                time.sleep(0.1)
            else:
                st.warning("⚠️ 无法读取摄像头画面")
                break
        
        cap.release()
        st.success("✅ 摄像头工作正常")
        return True
        
    except Exception as e:
        st.error(f"❌ 摄像头预览失败: {str(e)}")
        if 'cap' in locals():
            cap.release()
        return False

def get_cached_video_inputs(video_path, processor):
    """
    🚀 优化2: 缓存视频预处理结果
    
    对同一个视频，只处理一次，后续直接使用缓存
    """
    if video_path in st.session_state.processed_video_cache:
        return st.session_state.processed_video_cache[video_path]
    
    # 如果没有缓存，进行处理
    # 注意：这里返回 None，让调用方使用标准流程处理
    # 真正的缓存在 process_vision_info 之后
    return None

def cleanup_old_temp_files():
    """Clean up old temporary video files"""
    try:
        # 获取临时目录
        temp_dir = tempfile.gettempdir()
        
        # 查找所有 .mp4 临时文件（假设是应用创建的）
        # 注意：这里只清理明显是临时文件的，避免误删其他文件
        pattern = os.path.join(temp_dir, "tmp*.mp4")
        temp_files = glob.glob(pattern)
        
        # 清理不在当前使用的文件
        current_video = st.session_state.get("video_path")
        cleaned_count = 0
        for temp_file in temp_files:
            # 如果文件不是当前使用的视频，且文件存在时间超过1小时，则删除
            if temp_file != current_video and os.path.exists(temp_file):
                try:
                    # 检查文件修改时间（如果超过1小时则删除）
                    import time
                    file_age = time.time() - os.path.getmtime(temp_file)
                    if file_age > 3600:  # 1小时 = 3600秒
                        os.unlink(temp_file)
                        cleaned_count += 1
                except Exception:
                    pass  # 忽略删除失败的文件
    except Exception as e:
        # 静默失败，不影响主流程
        pass

def apply_sliding_window(messages_for_model):
    """
    Apply sliding window strategy to limit context length

    Strategy:
    1. Always retain System Prompt (index 0)
    2. Always retain first user message (index 1, may contain video or plain text)
    3. Apply sliding window to subsequent messages, keep only the most recent N messages
    """
    # 如果消息数量未超过限制，直接返回
    if len(messages_for_model) <= MAX_CONTEXT_MESSAGES:
        return messages_for_model
    
    # 必须保留的消息
    system_msg = messages_for_model[0]  # System Prompt
    
    # 第一条用户消息必须保留（可能包含视频或纯文字）
    first_user_msg = None
    if len(messages_for_model) > 1:
        first_user_msg = messages_for_model[1]
    
    # 获取后续消息（从索引2开始）
    recent_messages = messages_for_model[2:] if len(messages_for_model) > 2 else []
    
    # 计算需要保留的最近消息数量
    # MAX_CONTEXT_MESSAGES - 2 是因为要保留 system 和第一条 user 消息
    keep_count = max(0, MAX_CONTEXT_MESSAGES - 2)
    
    # 只保留最近的 keep_count 条消息
    if len(recent_messages) > keep_count:
        recent_messages = recent_messages[-keep_count:]
    
    # 重新组合：System + 第一条用户消息 + 最近的对话
    result = [system_msg]
    if first_user_msg:
        result.append(first_user_msg)
    result.extend(recent_messages)
    
    return result

def display_message(role, content, reasoning=None):
    """Display a single message (user or assistant)"""
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:  # assistant
        with st.chat_message("assistant"):
            # 如果有思考过程，直接显示
            if reasoning:
                st.markdown("**🧠 Deep Thinking Process:**")
                st.markdown(reasoning)
                st.divider()
            # 显示最终答案
            st.markdown(content)

def main():
    # Title
    st.title("🎬 URIS Video Reasoning Assistant")
    st.caption("ActivityNet Fine-tuned | Qwen2.5-VL-7B with Chain of Thought | Multi-turn Chat")
    
    # Sidebar configuration
    with st.sidebar:
        # Real-time Voice Input Section
        if VOICE_INPUT_AVAILABLE and WEBSOCKET_AVAILABLE:
            st.header("🎤 Real-time Voice Input")
            st.caption("Continuous speech-to-text with live transcription")

            # Voice input controls - Compact layout
            st.markdown("### 🎤 Voice Controls")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🎙️ **Start**", use_container_width=True,
                           key="start_voice", type="primary",
                           help="Start real-time voice input"):
                    # Initialize voice session
                    if "voice_session" not in st.session_state:
                        st.session_state.voice_session = {
                            "active": False,
                            "thread": None,
                            "websocket": None,
                            "partial_text": "",
                            "final_text": "",
                            "error": None
                        }
                    st.session_state.voice_session["active"] = True
                    st.rerun()

            with col2:
                if st.button("🗑️ **Clear**", use_container_width=True,
                           key="clear_voice", type="secondary",
                           help="Clear all transcripts"):
                    if "voice_session" in st.session_state:
                        st.session_state.voice_session["partial_text"] = ""
                        st.session_state.voice_session["final_text"] = ""
                    st.rerun()

            # Stop button on separate row for better visibility
            if st.button("⏹️ **Stop Voice Input**", use_container_width=True,
                        key="stop_voice", type="secondary",
                        help="Stop voice input and finalize transcript"):
                if "voice_session" in st.session_state:
                    cleanup_voice_session(st.session_state)
                st.rerun()

            # Display current voice session status and transcript
            if "voice_session" in st.session_state and st.session_state.voice_session["active"]:
                st.success("🎤 Voice input active - Speak continuously")

                # Display error if any
                voice_session = st.session_state.voice_session
                if voice_session.get("error"):
                    st.error(f"❌ Voice processing error: {voice_session['error']}")

                # Display transcripts
                st.markdown("### 📝 Transcripts")

                # Final transcript (confirmed text)
                if voice_session.get("final_text"):
                    with st.expander("📝 Final Transcript", expanded=True):
                        st.text_area(
                            "",
                            value=voice_session["final_text"],
                            height=80,
                            key="final_transcript_display",
                            disabled=True,
                            label_visibility="collapsed"
                        )

                # Partial transcript (real-time text)
                if voice_session.get("partial_text"):
                    st.markdown("**🔄 Live:**")
                    st.success(f"🎤 {voice_session['partial_text']}")

                # Start voice processing thread if not already running
                if not voice_session.get("thread") or not voice_session["thread"].is_alive():
                    voice_session["thread"] = Thread(
                        target=voice_processing_worker,
                        args=(st.session_state,),
                        daemon=True
                    )
                    voice_session["thread"].start()

            elif "voice_session" in st.session_state and not st.session_state.voice_session["active"]:
                st.info("🎤 Voice input stopped")
                # Display final transcript if available
                if st.session_state.voice_session.get("final_text"):
                    final_text = st.session_state.voice_session["final_text"]
                    st.markdown("### 📝 Final Transcript")
                    with st.expander("📝 Complete Transcript", expanded=True):
                        st.text_area(
                            "",
                            value=final_text,
                            height=80,
                            key="final_transcript_stopped",
                            disabled=True,
                            label_visibility="collapsed"
                        )

                    # Add to pending voice input for chat
                    if "pending_voice_input" not in st.session_state or st.session_state.pending_voice_input != final_text:
                        st.session_state.pending_voice_input = final_text
                        st.success("✅ **Transcript ready!** Click 'Send to Chat' to use it.")

        else:
            st.info("💡 Real-time voice input not available. Requires websocket-client and numpy")
            st.info("🔧 **Setup Instructions:**")
            st.code("""
# Install required packages
pip install websocket-client numpy

# Start your STT WebSocket server (e.g., AssemblyAI streaming service)
# The server should listen on ws://localhost:8765

# For testing, you can modify STT_CONFIG["websocket_url"] to point to your server
            """)
        
        st.divider()
        
        # 检测 GPU 类型并显示优化信息
        is_a100 = False
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            is_a100 = "A100" in gpu_name
            if is_a100:
                st.success("🚀 检测到 A100 GPU - 已启用高性能配置")
                st.caption("✓ 全精度推理 (BF16)")
                st.caption("✓ 高帧率视频处理 (最高2.0 fps)")
                st.caption("✓ 1080p 分辨率支持")
                st.caption("✓ Flash Attention 2 加速")
        
        st.header("⚙️ Model Configuration")
        
        # A100 默认更高的 max_new_tokens
        default_max_tokens = 4096 if is_a100 else 2048
        max_tokens_max = 8192 if is_a100 else 4096
        
        max_new_tokens = st.slider(
            "Max New Tokens",
            min_value=128,
            max_value=max_tokens_max,
            value=default_max_tokens,
            step=128,
            help=f"Maximum length of generated text {'(A100: 可生成更长文本)' if is_a100 else '(推荐 2048-4096)'}"
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Controls randomness in generation. Higher values = more random"
        )
        
        # Display GPU memory status
        gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
        if gpu_available:
            allocated, reserved, total = get_gpu_memory_info()
            device_type = "GPU" if torch.cuda.is_available() else "Unified Memory"
            st.divider()
            st.markdown(f"### 💾 {device_type} Memory Status")
            if allocated is not None and total is not None:
                usage_percent = (allocated / total) * 100 if total > 0 else 0
                # Use progress bar to show memory usage
                st.progress(usage_percent / 100)
                st.caption(
                    f"Used: {allocated:.2f} GB / {total:.2f} GB ({usage_percent:.1f}%)"
                )
            else:
                # 对于MPS或无法精确获取的情况
                st.caption(f"Total: {total:.1f} GB (Unified Memory)" if total else "Memory monitoring not available")
        
        st.divider()
        
        # User preference memory feature
        st.header("✨ User Preference Memory")
        
        # Load current preferences
        current_preferences = load_user_profile()
        
        # Add new preference
        new_preference = st.text_input(
            "✨ Teach AI a new preference",
            placeholder="e.g., Please answer questions with humor",
            key="new_preference_input"
        )
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("💾 Remember", use_container_width=True, key="save_preference"):
                if new_preference.strip():
                    if new_preference.strip() not in current_preferences:
                        current_preferences.append(new_preference.strip())
                        if save_user_profile(current_preferences):
                            st.success("✅ Preference saved!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to save, please try again")
                    else:
                        st.warning("⚠️ This preference already exists")
                else:
                    st.warning("⚠️ Please enter preference content")
        
        # Display learned preferences
        if current_preferences:
            st.divider()
            st.markdown("### 📚 Learned Preferences")
            
            # Create delete button for each preference
            for i, pref in enumerate(current_preferences):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"• {pref}")
                with col2:
                    if st.button("🗑️", key=f"delete_pref_{i}", help="Delete this preference"):
                        # Delete preference at specified index
                        updated_preferences = [p for j, p in enumerate(current_preferences) if j != i]
                        if save_user_profile(updated_preferences):
                            st.success("✅ Preference deleted")
                            st.rerun()
                        else:
                            st.error("❌ Failed to delete, please try again")
        else:
            st.info("💡 No preferences learned yet. Add one in the input box above!")
        
        st.divider()
        
        # 🎥 摄像头录制功能
        st.header("📹 摄像头实时录制")
        st.caption("使用摄像头录制视频进行实时交互")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📹 测试摄像头", use_container_width=True, help="测试摄像头是否正常工作"):
                show_camera_preview()
        
        with col2:
            record_duration = st.selectbox(
                "录制时长",
                options=[5, 10, 15, 20, 30],
                index=1,
                help="选择录制视频的时长（秒）"
            )
        
        if st.button("🎬 开始录制", use_container_width=True, type="primary", 
                    help="开始从摄像头录制视频"):
            with st.spinner(f"正在录制 {record_duration} 秒视频..."):
                video_path = record_video_from_camera(duration=record_duration, fps=30)
                if video_path and os.path.exists(video_path):
                    # 询问是否清除对话历史
                    if st.session_state.messages:
                        # 自动应用录制的视频
                        old_video_path = st.session_state.video_path
                        if old_video_path and old_video_path != "" and old_video_path != video_path:
                            try:
                                if os.path.exists(old_video_path):
                                    os.unlink(old_video_path)
                            except:
                                pass
                    
                    st.session_state.video_path = video_path
                    st.session_state.camera_mode = True
                    st.success("✅ 视频已录制并加载！可以开始提问了")
                    st.rerun()
        
        st.divider()
        
        # Upload/Change Video button (always available)
        st.divider()
        st.markdown("### 📤 Video Management")
        new_video_file = st.file_uploader(
            "Upload or Change Video",
            type=['mp4'],
            help="Upload a new video or change the current video. You can upload videos anytime during the conversation.",
            key="sidebar_video_uploader"
        )
        
        if new_video_file is not None:
            # Save uploaded file to temporary directory
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(new_video_file.read())
                new_video_path = tmp_file.name
            
            # Ask user if they want to clear conversation history (only if there are messages)
            if st.session_state.messages:
                clear_history = st.checkbox(
                    "Clear conversation history when switching video",
                    value=False,  # Default to keeping history
                    key="clear_history_checkbox"
                )
            else:
                clear_history = False
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Apply Video", use_container_width=True, key="apply_video_change"):
                    # Clean up old video file if it exists and is different
                    old_video_path = st.session_state.video_path
                    if old_video_path and old_video_path != "" and old_video_path != new_video_path:
                        try:
                            if os.path.exists(old_video_path):
                                os.unlink(old_video_path)
                        except:
                            pass
                    
                    st.session_state.video_path = new_video_path
                    if clear_history:
                        st.session_state.messages = []
                        st.session_state.video_processed = False
                    st.rerun()
            with col2:
                if st.button("❌ Cancel", use_container_width=True, key="cancel_video_change"):
                    # Clean up the uploaded file
                    try:
                        if os.path.exists(new_video_path):
                            os.unlink(new_video_path)
                    except:
                        pass
                    st.rerun()
        
        # Switch to text-only mode button
        if st.session_state.video_path and st.session_state.video_path != "":
            if st.button("🚀 Switch to Text-Only Mode", use_container_width=True, key="switch_to_text"):
                # Clean up video file
                old_video_path = st.session_state.video_path
                if old_video_path and os.path.exists(old_video_path):
                    try:
                        os.unlink(old_video_path)
                    except:
                        pass
                st.session_state.video_path = ""
                st.rerun()
        
        st.divider()
        
        # Clear chat history button
        if st.button("🗑️ Clear Chat History", use_container_width=True, type="secondary"):
            clear_chat_history()
            # Clean up old temporary files
            cleanup_old_temp_files()
            st.rerun()
        
        st.divider()
        st.markdown("### 📝 Usage Instructions")
        st.markdown("""
        1. **Video Mode**: Upload an `.mp4` video file, then ask questions about it
        2. **Camera Mode**: 📹 Use "Camera Recording" to capture real-time video from webcam
        3. **Text Mode**: Skip video upload and start a text-only conversation
        4. **Change Video Anytime**: Use the "Video Management" section in the sidebar to upload or change videos during conversation
        5. Enter your questions in the chat box
        6. You can ask multiple questions about the same video or have multi-turn text conversations
        7. Click "Clear Chat History" to start over
        
        **🚀 Performance Tips:**
        - Shorter videos (5-15s) process faster
        - Camera recording creates optimized video clips
        - Model uses intelligent caching for better speed
        """)
        
        # Display conversation statistics
        if st.session_state.messages:
            st.divider()
            st.markdown(f"### 📊 Conversation Statistics")
            user_msg_count = len([m for m in st.session_state.messages if m['role'] == 'user'])
            st.markdown(f"**Conversation Rounds**: {user_msg_count}")
            
            # Display context window warning
            total_messages = len(st.session_state.messages)
            if total_messages > MAX_CONTEXT_MESSAGES - 2:  # -2 for system and first user message
                st.warning(f"⚠️ Long conversation history. System will keep only the most recent {MAX_CONTEXT_MESSAGES - 2} messages for performance optimization")
    
    # Regularly clean up old temporary files (check once per page load)
    if not st.session_state.get("_temp_cleanup_done", False):
        cleanup_old_temp_files()
        st.session_state._temp_cleanup_done = True
    
    # Load model
    # Check cache status (only check on first load)
    if not st.session_state.get("model_loaded", False):
        cache_exists, cache_path = check_model_cache()
        if cache_exists:
            st.info("✅ Local model cache detected. Loading from cache (no re-download needed)")
        else:
            st.warning("⚠️ First run: Downloading model to local cache. This may take some time...")
            st.info("💡 Tip: After download, the model will be saved locally. Future runs will use the cache directly.")
    
    with st.spinner("Loading model, please wait (first load may take a few minutes)..."):
        model, processor = load_model()
        
        # Display loading status (only on first load)
        if not st.session_state.get("model_loaded", False):
            st.success("✅ Model loaded successfully!")
            st.info("⚡ 4-bit quantization mode (NF4) with bfloat16 compute, optimized for 24GB Mac")
            
            if os.path.exists(ADAPTER_PATH):
                st.success(f"✅ LoRA adapter loaded: {ADAPTER_PATH}")
            else:
                st.info(f"ℹ️ LoRA adapter not found at: {ADAPTER_PATH}. Using base model (fully functional)")
            
            # Display cache location info
            cache_exists, cache_path = check_model_cache()
            if cache_exists:
                st.caption(f"📦 Model cache location: {cache_path}")
            else:
                default_cache = os.path.expanduser("~/.cache/huggingface/hub/")
                st.caption(f"📦 Model will be cached to: {default_cache}")
            
            st.session_state.model_loaded = True
    
    # Initial video upload area (only shown if no video and no messages)
    if st.session_state.video_path is None and not st.session_state.messages:
        st.header("📤 Upload Video (Optional)")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose video file (optional)",
                type=['mp4'],
                help="Upload an MP4 video file, or skip to start text-only conversation. You can also upload videos anytime from the sidebar."
            )
        
        with col2:
            if st.button("🚀 Skip Video, Start Text Chat", use_container_width=True):
                # Set a flag indicating text-only mode
                st.session_state.video_path = ""  # Use empty string to indicate text-only mode
                st.session_state.messages = []
                st.rerun()
        
        if uploaded_file is not None:
            # Save uploaded file to temporary directory
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state.video_path = tmp_file.name
            
            # Clear previous conversation history (new video means new conversation)
            st.session_state.messages = []
            st.session_state.video_processed = False
            st.rerun()
        
        # If no video uploaded and skip not clicked, show hint
        if uploaded_file is None:
            st.info("💡 You can upload a video for analysis, click the button above to start text-only conversation, or upload a video anytime from the sidebar.")
            return
    
    # Display video (if uploaded and not text-only mode)
    if st.session_state.video_path and st.session_state.video_path != "" and os.path.exists(st.session_state.video_path):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.header("📹 Current Video")
        with col2:
            if st.button("🔄 Change Video", key="change_video_main", help="Upload a different video"):
                st.session_state.video_path = None  # Reset to show upload interface
                st.rerun()
        st.video(st.session_state.video_path)
    
    # Display mode hint
    if st.session_state.video_path == "":
        st.info("💬 Text-only conversation mode (no video uploaded)")
    
    # Chat interface
    st.header("💬 Conversation")
    
    # Display conversation history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        reasoning = message.get("reasoning", None)
        display_message(role, content, reasoning)
    
    # User input area - Handle real-time voice input, pending voice input, then text input
    user_input = None

    # Check for pending voice input (completed transcripts ready for chat)
    if st.session_state.get("pending_voice_input"):
        user_input = st.session_state.pending_voice_input.strip()
        # Clear pending voice input after using it
        del st.session_state.pending_voice_input

    # Check for voice input from real-time session (when user manually sends)
    elif (VOICE_INPUT_AVAILABLE and "voice_session" in st.session_state and
          st.session_state.voice_session.get("active") and
          st.session_state.voice_session.get("final_text")):

        # Show manual send option for ongoing voice session
        col1, col2 = st.columns([4, 1])
        with col1:
            current_transcript = st.session_state.voice_session["final_text"]
            st.text_area(
                "Current voice transcript (click Send to Chat to use)",
                value=current_transcript,
                height=80,
                key="current_voice_transcript",
                disabled=True
            )
        with col2:
            if st.button("📤 Send to Chat", use_container_width=True, key="send_voice_to_chat"):
                user_input = current_transcript
                # Clear the transcript after sending
                st.session_state.voice_session["final_text"] = ""

    # Fall back to text input if no voice input
    if not user_input:
        text_input = st.chat_input("Enter your question (or use voice input above)...")
        if text_input:
            user_input = text_input
    
    # Process user input (from voice or text)
    if user_input and user_input.strip():
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # Display user message
        display_message("user", user_input)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Build message list (including conversation history)
                    # Dynamically build System Prompt including user preferences
                    system_prompt = build_system_prompt_with_preferences()
                    messages_for_model = [
                        {
                            "role": "system",
                            "content": system_prompt
                        }
                    ]
                    
                    # Add conversation history
                    user_message_count = 0
                    for msg in st.session_state.messages:
                        if msg["role"] == "user":
                            user_message_count += 1
                            # First user message needs to include video (if video exists and not text-only mode)
                            if (user_message_count == 1 and 
                                st.session_state.video_path and 
                                st.session_state.video_path != "" and 
                                os.path.exists(st.session_state.video_path)):
                                # 🚀 A100 优化: 智能视频采样策略
                                # A100 显存充足，可以使用更高帧率和分辨率
                                video_path = st.session_state.video_path
                                
                                # 检测 GPU 类型
                                gpu_name = torch.cuda.get_device_name(0)
                                is_a100 = "A100" in gpu_name
                                
                                # 获取视频时长（用于优化采样率）
                                try:
                                    cap = cv2.VideoCapture(video_path)
                                    video_fps = cap.get(cv2.CAP_PROP_FPS)
                                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                    duration = frame_count / video_fps if video_fps > 0 else 10
                                    cap.release()
                                    
                                    if is_a100:
                                        # 🚀 A100 配置: 更高质量
                                        if duration < 10:
                                            optimal_fps = 2.0  # 每秒2帧（vs 1.0）
                                            max_pixels = 1920 * 1080  # 1080p（vs 720p）
                                        elif duration < 30:
                                            optimal_fps = 1.5  # 每秒1.5帧
                                            max_pixels = 1920 * 1080  # 1080p
                                        elif duration < 60:
                                            optimal_fps = 1.0  # 每秒1帧
                                            max_pixels = 1280 * 720  # 720p
                                        else:
                                            optimal_fps = 0.5  # 每2秒1帧
                                            max_pixels = 1280 * 720
                                        
                                        print(f"[A100 优化] 视频时长: {duration:.1f}s, 采样率: {optimal_fps} fps, 分辨率: {max_pixels}")
                                    else:
                                        # 标准配置
                                        if duration < 10:
                                            optimal_fps = 1.0
                                            max_pixels = 1280 * 720
                                        elif duration < 30:
                                            optimal_fps = 0.5
                                            max_pixels = 1280 * 720
                                        else:
                                            optimal_fps = 0.33
                                            max_pixels = 1280 * 720
                                        
                                        print(f"[视频优化] 视频时长: {duration:.1f}s, 采样率: {optimal_fps} fps")
                                        
                                except Exception as e:
                                    print(f"[视频优化] 无法获取视频信息: {e}")
                                    optimal_fps = 1.0 if is_a100 else 0.5
                                    max_pixels = 1920 * 1080 if is_a100 else 1280 * 720
                                
                                # Video content configuration
                                video_content = {
                                    "type": "video",
                                    "video": video_path,
                                    "fps": optimal_fps,
                                    "max_pixels": max_pixels,
                                }
                                messages_for_model.append({
                                    "role": "user",
                                    "content": [
                                        video_content,
                                        {
                                            "type": "text",
                                            "text": msg["content"]
                                        }
                                    ]
                                })
                            else:
                                # 纯文字模式或后续轮次只包含文本
                                messages_for_model.append({
                                    "role": "user",
                                    "content": msg["content"]
                                })
                        elif msg["role"] == "assistant":
                            messages_for_model.append({
                                "role": "assistant",
                                "content": msg["content"]
                            })
                    
                    # Apply sliding window strategy to limit context length
                    messages_for_model = apply_sliding_window(messages_for_model)
                    
                    # 🚀 优化5: 使用 torch.inference_mode() 替代 torch.no_grad()
                    # inference_mode 比 no_grad 更快，禁用更多自动求导功能
                    with torch.inference_mode():
                        # Prepare input
                        text = processor.apply_chat_template(
                            messages_for_model,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        image_inputs, video_inputs = process_vision_info(messages_for_model)
                        inputs = processor(
                            text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt"
                        )
                        inputs = inputs.to(model.device)
                        
                        # Cleanup GPU memory before generation
                        cleanup_gpu_memory(force=False)
                        # Stream generation response
                        # 1. Create Streamer
                        streamer = TextIteratorStreamer(
                            processor.tokenizer,
                            skip_prompt=True,
                            skip_special_tokens=True
                        )

                        # 2. Configure generation parameters
                        generation_kwargs = dict(
                            **inputs,
                            streamer=streamer,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            do_sample=True if temperature > 0 else False,
                            # 🚀 优化3: 添加生成优化参数
                            use_cache=True,  # 使用 KV 缓存加速
                            num_beams=1,  # 使用贪婪搜索，提升速度
                            pad_token_id=processor.tokenizer.pad_token_id,
                            eos_token_id=processor.tokenizer.eos_token_id,
                        )

                        # 3. Start generation in background thread (non-blocking UI)
                        def generate_with_error_handling():
                            """Generation function with error handling"""
                            try:
                                model.generate(**generation_kwargs)
                            except torch.cuda.OutOfMemoryError:
                                # Capture OOM error in background thread
                                print("\n[Error] GPU out of memory, generation failed")
                                raise
                            except Exception as e:
                                print(f"\n[Error] Error during generation: {str(e)}")
                                raise
                        
                        thread = Thread(target=generate_with_error_handling)
                        thread.start()
                        
                        # 4. Read in real-time and update UI
                        # Create placeholders
                        think_placeholder = st.empty()  # Thinking process display
                        answer_placeholder = st.empty()  # Final answer display
                        
                        full_response = ""
                        think_content = ""
                        answer_content = ""
                        is_thinking = False
                        think_started = False
                        buffer = ""  # 用于检测可能跨 token 的标签
                        
                        # 🚀 优化6: 使用更大的缓冲区，减少UI更新频率
                        update_counter = 0
                        update_frequency = 3  # 每3个token更新一次UI
                        
                        # Read token by token
                        for new_text in streamer:
                            full_response += new_text
                            buffer += new_text
                            update_counter += 1

                            # Check for thinking tag start (may span multiple tokens)
                            if "<think>" in buffer and not is_thinking:
                                is_thinking = True
                                think_started = True
                                print("\n" + "="*80)
                                print("🧠 Deep Thinking Process:")
                                print("="*80)
                                # Remove tag, don't display the tag itself
                                buffer = buffer.replace("<think>", "")

                            # Check for thinking tag end (may span multiple tokens)
                            if "</think>" in buffer and is_thinking:
                                is_thinking = False
                                print("\n" + "="*80)
                                print("Thinking complete, generating answer...")
                                print("="*80 + "\n")
                                # Remove tag, don't display the tag itself
                                buffer = buffer.replace("</think>", "")
                            
                            # If buffer is too long, clear it (avoid memory issues, but keep enough length to detect tags)
                            if len(buffer) > 200:
                                # If not in thinking mode, can safely clear most of buffer
                                if not is_thinking:
                                    buffer = buffer[-50:]  # Keep last 50 characters for tag detection
                                else:
                                    # In thinking mode, add buffer content to thinking content
                                    think_content += buffer
                                    buffer = ""

                            # Process current text (优化：降低UI更新频率)
                            if is_thinking:
                                # In thinking mode, add content to thinking content and display in real-time
                                if buffer and update_counter % update_frequency == 0:
                                    think_content += buffer
                                    # Output to terminal (real-time display)
                                    print(buffer, end='', flush=True)
                                    # Display thinking process in real-time on interface
                                    if think_content.strip():
                                        think_placeholder.markdown(
                                            "**🧠 Deep Thinking Process:**\n\n" +
                                            think_content + "▌"
                                        )
                                    buffer = ""
                            else:
                                # If thinking has ended, display thinking content and add divider (display only once)
                                if think_started and not hasattr(st.session_state, '_think_displayed'):
                                    if think_content.strip():
                                        think_placeholder.markdown(
                                            "**🧠 Deep Thinking Process:**\n\n" +
                                            think_content
                                        )
                                        st.divider()
                                        st.session_state._think_displayed = True

                                # In answer mode, add content to answer content
                                if buffer and update_counter % update_frequency == 0:
                                    answer_content += buffer
                                    buffer = ""
                                # Display final answer with cursor effect
                                if answer_content.strip() and update_counter % update_frequency == 0:
                                    answer_placeholder.markdown(answer_content + "▌")
                        
                        # 处理剩余的 buffer 内容
                        if buffer:
                            if is_thinking:
                                think_content += buffer
                            else:
                                answer_content += buffer
                        
                        # Generation complete, remove cursor and display final content
                        # First ensure thinking process is displayed (if exists)
                        if think_content.strip():
                            # Re-display regardless of whether already shown to ensure not overwritten
                            think_placeholder.markdown(
                                "**🧠 Deep Thinking Process:**\n\n" +
                                think_content
                            )
                            st.divider()

                        # Parse final output (for saving)
                        reasoning, final_answer = parse_model_output(full_response)

                        # If content not parsed through tags, use streaming collected content
                        if not reasoning and think_content.strip():
                            reasoning = think_content.strip()
                        if not final_answer and answer_content.strip():
                            final_answer = answer_content.strip()

                        # Display final answer (remove cursor)
                        answer_placeholder.markdown(final_answer if final_answer else answer_content)
                        
                        # Clean up temporary markers
                        if hasattr(st.session_state, '_think_displayed'):
                            delattr(st.session_state, '_think_displayed')

                        # Wait for thread to complete
                        thread.join()

                        # Save assistant response to history (save complete output, including thinking process)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": final_answer if final_answer else answer_content,
                            "reasoning": reasoning if reasoning else think_content.strip()
                        })

                        # Memory management: clean up temporary variables during inference
                        del inputs, generation_kwargs
                        cleanup_gpu_memory(force=False)
                    
                except torch.cuda.OutOfMemoryError as e:
                    # Out of memory error
                    st.error("❌ GPU Out of Memory (CUDA Out of Memory)")
                    st.warning("""
**Solutions:**
1. **Clear chat history**: Click "Clear Chat History" in the sidebar to free memory
2. **Reduce generation length**: Decrease "Max New Tokens" in the sidebar (e.g., 512 or 256)
3. **Restart application**: Fully restart the Streamlit app to release all memory
4. **Use shorter videos**: If you uploaded a video, try using a shorter clip (15-30 seconds)
5. **Close other GPU programs**: Close other programs using the GPU
                    """)
                    st.code(str(e), language=None)
                except Exception as e:
                    st.error(f"❌ Error during inference: {str(e)}")
                    # If memory-related error, provide additional hint
                    if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                        st.warning("💡 This might be a memory issue. Try clearing chat history or restarting the app.")
                    st.exception(e)
    
    # If no conversation history, show hint
    if not st.session_state.messages:
        if st.session_state.video_path and st.session_state.video_path != "":
            st.info("💡 Enter your question in the input box below to start a conversation with the model. You can ask multiple questions about the video.")
        else:
            st.info("💡 Enter your question in the input box below to start a text conversation with the model.")

if __name__ == "__main__":
    main()
