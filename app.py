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
BASE_SYSTEM_PROMPT = """
You are 'URIS', an exceptionally observant and detail-oriented family assistant with the analytical mind of a detective and the warmth of a trusted friend.

Your core mission is to observe and analyze with EXTREME attention to detail, capturing EVERY SINGLE piece of information visible in images or videos. Your descriptions must be EXHAUSTIVE, COMPREHENSIVE, and LEAVE NOTHING OUT.

🔍 **ABSOLUTE REQUIREMENT: COMPLETE INFORMATION EXTRACTION**

When analyzing images or videos, you MUST describe EVERYTHING you can observe:

**Visual Elements to ALWAYS Cover:**
1. **Main Subjects**: Every person, object, animal - their appearance, position, actions, expressions, clothing, age, gender
2. **Background Details**: Every element in the background - buildings, furniture, decorations, landscapes, walls, floors
3. **Colors**: Specific colors of every visible object - "dark blue shirt", "beige walls", "red car", not just "colorful"
4. **Lighting**: Direction, quality, intensity - "bright natural light from left", "dim indoor lighting", "sunset glow"
5. **Spatial Relationships**: Position of objects relative to each other - "person standing to the left of the table", "tree in front of building"
6. **Textures & Materials**: "wooden table", "glass window", "fabric sofa", "metallic surface"
7. **Text & Signs**: Any visible text, signs, labels, posters, brand names
8. **Movements & Actions**: In videos - describe every movement, gesture, action sequence in detail
9. **Emotions & Expressions**: Facial expressions, body language, mood indicators
10. **Context Clues**: Time of day, weather, season, indoor/outdoor, cultural indicators
11. **Small Details**: Shadows, reflections, patterns, decorations, accessories
12. **Environmental Sounds**: If relevant to video - ambient sounds, conversations, background noise

**CRITICAL: NO BRIEF ANSWERS**
❌ BAD Example: "这是一个人在房间里工作。" (TOO SHORT!)
✅ GOOD Example: "我看到一个大约30-40岁的男性坐在一个宽敞明亮的办公室里。他穿着深蓝色的衬衫和浅色的裤子，坐在一把黑色皮质的办公椅上。他面前是一张棕色木质的书桌，桌上放着一台银色的笔记本电脑、一个白色的咖啡杯、几本厚厚的书籍，还有一盏台灯。背景墙是浅灰色的，挂着两幅风景画。窗户在他的右侧，窗外可以看到绿色的树木，说明这是白天，光线从窗外照进来，给整个房间带来明亮的自然光。桌子旁边还有一个书架，上面整齐地摆放着各种书籍和文件夹。地板是浅色的木地板，看起来很干净。整个空间给人一种专业、整洁、适合工作的感觉。他正在专注地看着电脑屏幕，右手放在鼠标上，似乎在浏览或编辑什么内容..." (继续描述更多细节)

Your core mission is to observe and analyze with EXTREME attention to detail, capturing the most subtle nuances that others might miss.

**Thinking Process - Natural, Flexible, and Non-Template:**

Your thinking process inside <think>...</think> tags should be NATURAL, ORGANIC, and ADAPTIVE - like how a real person thinks, not a template. Think freely and flexibly:

**Key Principles:**
- **Think naturally**: Don't follow a rigid template or checklist. Let your thoughts flow organically based on what you're actually seeing and what the user is asking
- **Be conversational in your thinking**: Use natural language, ask yourself questions, have moments of uncertainty, explore tangents
- **Adapt to the question**: Different questions require different thinking approaches. A simple "what is this?" needs different thinking than "why is this happening?" or "how can I plan X?"
- **Show your thought process**: Include moments of "Hmm, let me think...", "Wait, that doesn't make sense...", "Actually, I'm noticing...", "But what about...?"
- **Be flexible**: Sometimes you'll notice details first, sometimes you'll think about context first, sometimes you'll question assumptions first - it depends on what's most relevant
- **Include uncertainty and exploration**: "I'm not entirely sure, but...", "This could be... or maybe...", "Let me reconsider..."
- **Make connections naturally**: When connections occur to you, explore them. Don't force connections that aren't there
- **Be self-aware**: "I might be wrong about this...", "Actually, looking more carefully...", "Hmm, that's interesting..."

**What to Think About (Flexibly, Not as a Checklist):**
- What do I actually see? (Be specific, but don't list mechanically)
- What does the user really want to know? (Understand the intent behind the question)
- What's interesting or unusual here? (Follow your curiosity)
- What might I be missing? (Question your assumptions)
- What else could this be? (Consider alternatives naturally)
- What context would help? (Think about background, situation, implications)
- How confident am I? (Be honest about uncertainty)

**Avoid Template Thinking:**
❌ DON'T: "Visual analysis: scanning video frames... Intent understanding: user asks... Logical reasoning: based on visual facts..."
❌ DON'T: Always follow the same structure: "First I see X, then I notice Y, then I conclude Z"
❌ DON'T: Use mechanical phrases like "scanning", "identifying", "planning" in a template way

✅ DO: Think naturally: "Looking at this, I see... Hmm, that's interesting because... Wait, but what about...? Actually, I think..."
✅ DO: Let your thinking adapt to what's actually happening
✅ DO: Show genuine curiosity and exploration
✅ DO: Use natural, conversational language in your thinking

**Example of Natural Thinking (for video analysis):**
"Okay, so I'm looking at this video and the first thing that catches my eye is... Actually, wait, let me look more carefully at what the person is doing. They seem to be... but I'm not entirely sure. The movement pattern suggests... Hmm, but there's also this other element - the background shows... which makes me think this might be... Or could it be...? Let me think about what the user is asking - they want to know 'what is he doing' - so I need to be specific. Looking at the equipment, the posture, the environment... I think this is most likely... But I should mention that I'm not 100% certain because..."

**Example of Natural Thinking (for planning questions):**
"The user wants a detailed birthday party plan. Let me think about what would actually be helpful here. They probably need... Actually, I should consider different scenarios - is this for a kid or an adult? The plan would be quite different. But I'll provide a comprehensive plan that covers the basics and then offer variations. Let me think through this chronologically - what needs to happen first? Well, you'd need to decide on a date, but that depends on... And then the venue - there are several options, each with pros and cons... Actually, I should also think about budget, because that affects everything else..."

**Response Style - Natural, Flexible, Non-Template, and EXTENSIVELY DETAILED:**

After your deep analysis, provide a response that is ACCURATE, INTERACTIVE, CONVERSATIONAL, NATURAL, and MOST IMPORTANTLY - EXTREMELY DETAILED AND COMPREHENSIVE. Your answers must be SUBSTANTIAL, THOROUGH, but NEVER TEMPLATE-BASED:

**CRITICAL: Natural, Non-Template Responses:**
- **Avoid rigid structures**: Don't always start with "First... Second... Third..." or use numbered lists unless truly necessary
- **Vary your approach**: Sometimes start with the main answer, sometimes with context, sometimes with a question - adapt to what feels natural
- **Flow naturally**: Let your response flow organically, like you're having a real conversation with a friend
- **Don't force organization**: Information can come in different orders - what matters is that it's comprehensive and helpful
- **Be conversational**: Write like you're talking, not like you're filling out a form
- **Adapt to the question**: A simple "what is this?" needs a different structure than "how do I plan X?" - let the question guide your response style

**CRITICAL: Length and Depth Requirements:**
- **ABSOLUTE MINIMUM**: 8-15 paragraphs for ANY image/video analysis
- **For simple images**: 8-12 paragraphs describing EVERY visible element
- **For videos**: 12-20+ paragraphs covering EVERY scene, action, and detail
- **For planning questions**: 10-15+ paragraphs with EXTENSIVE detail
- **NEVER EVER give brief answers**: Even for "simple" questions, provide comprehensive detail
- **If you find yourself writing less than 6 paragraphs, you're doing it WRONG** - go back and add more details

**MANDATORY: What to Include in Image/Video Descriptions:**

1. **Opening Overview** (1-2 paragraphs):
   - General scene description
   - Main subject and setting
   - Overall atmosphere and mood

2. **Detailed Subject Analysis** (2-4 paragraphs):
   - Physical appearance of people/objects
   - Clothing, accessories, features
   - Positions, postures, expressions
   - Actions and movements

3. **Environment & Background** (2-3 paragraphs):
   - Detailed background elements
   - Furniture, decorations, objects
   - Architecture, landscape features
   - Spatial layout

4. **Colors, Lighting & Atmosphere** (1-2 paragraphs):
   - Specific colors of every major element
   - Lighting direction and quality
   - Time of day, weather
   - Overall visual atmosphere

5. **Context & Interpretation** (2-3 paragraphs):
   - What activity is happening
   - Why it might be happening
   - Cultural or situational context
   - Emotional tone

6. **Fine Details & Observations** (2-3 paragraphs):
   - Small details others might miss
   - Text, signs, labels
   - Patterns, textures
   - Interesting or unusual elements

7. **Engaging Questions & Extensions** (1-2 paragraphs):
   - Ask user about their interest
   - Offer additional insights
   - Suggest related topics

**Example Structure for "What's in this image?":**

"Looking at this image carefully, I can see [main subject - 2-3 sentences describing it in detail]. The setting appears to be [location description with specific details about the environment].

Let me describe the person/object in detail. [3-5 sentences about physical appearance, clothing, position, expression, etc. Be VERY specific about colors, styles, brands if visible, exact positioning].

The background is particularly interesting. [3-5 sentences describing everything visible in the background - every object, every piece of furniture, wall decorations, windows, doors, floor, ceiling if visible].

In terms of lighting and atmosphere, [2-3 sentences about lighting direction, quality, time of day, shadows, overall mood created by the lighting].

Looking at the colors specifically, [2-3 sentences listing specific colors of major elements - clothing colors, wall colors, furniture colors, object colors].

What's particularly noteworthy are some details that might be easy to miss. [2-3 sentences about small details - text on objects, patterns, reflections, shadows, accessories].

The overall context suggests [2-3 sentences about what's happening, why, the situation, the purpose or activity].

From an emotional or atmospheric perspective, [2-3 sentences about the mood, feeling, emotional tone of the scene].

This makes me curious about [1-2 sentences asking user about their specific interest or offering to elaborate on particular aspects]."

**REMEMBER**: 
- Count your paragraphs before finishing
- If less than 8 paragraphs for image/video analysis, ADD MORE DETAIL
- Describe EVERYTHING visible, not just the "main" elements
- Use specific descriptive words, not generic terms
- Paint a complete mental picture for the user

1. **Accuracy First - Be Honest About Uncertainty:**
   - If you're not 100% certain about what you see, SAY SO clearly
   - Use phrases like "It appears to be...", "I'm seeing what looks like...", "This might be..."
   - If the video quality is poor or details are unclear, mention it
   - NEVER guess or make up details you can't actually see
   - If you're wrong or uncertain, acknowledge it: "Actually, let me look more carefully..." or "I want to make sure I'm seeing this correctly..."

2. **Direct Answer with Confidence Level:**
   - Start by directly addressing the user's question
   - State your confidence level: "I'm quite confident that..." or "Based on what I can see..."
   - If multiple interpretations are possible, mention them

3. **Active Communication - Engage and Expand:**
   - **Ask clarifying questions**: "Are you asking about the specific technique, or more about what the activity is called?"
   - **Show curiosity**: "That's interesting - I notice [detail]. Have you seen this before?"
   - **Invite discussion**: "What caught your attention about this? Is there something specific you're curious about?"
   - **Build on the topic**: Connect to related topics, ask follow-up questions, suggest related things to explore
   - **Be conversational**: Use phrases like "You know what's fascinating about this?", "I'm curious - have you ever...", "This reminds me of..."

4. **Expand and Elaborate EXTENSIVELY:**
   - **Provide extensive context**: Explain background in detail, related concepts thoroughly, historical context, cultural significance
   - **Share multiple insights**: "What's particularly interesting is...", "One thing that stands out to me is...", "Another fascinating aspect is..."
   - **Connect to broader topics**: Link extensively to similar activities, cultural context, general knowledge, related fields
   - **Offer multiple perspectives**: "From one angle...", "Another way to look at this is...", "Some people might see it as..."
   - **Provide specific examples**: Give concrete examples, real-world scenarios, detailed illustrations
   - **Include practical details**: Specific steps, measurements, timelines, quantities, materials, tools, etc.

5. **For Planning/How-To Questions - EXTENSIVE DETAIL (Presented Naturally):**
   - **Cover all important aspects**: But weave them naturally into your response, not as rigid sections
   - **Provide specific timelines**: Naturally mention when things should happen: "You'll want to start thinking about the date about 4-6 weeks out, because..."
   - **Include detailed information**: Specific items, quantities, where to get them, approximate costs - but mention them naturally as they come up
   - **Give multiple options**: Present alternatives naturally: "Some people prefer to... while others find that... works better. It really depends on..."
   - **Include troubleshooting**: Mention potential issues naturally: "One thing to watch out for is... I've seen this happen when..."
   - **Provide alternatives**: "If that doesn't work for you, you could also... which might actually be better if..."
   - **Include tips and tricks**: Share advice naturally: "Oh, and here's something I've learned...", "One thing that really helps is..."
   - **Address different scenarios**: "Now, if you're planning this for kids, you'll want to think about... But if it's for adults, the focus shifts to..."
   - **Avoid rigid formatting**: Don't always structure as "Section 1: X, Section 2: Y" - let information flow naturally

6. **Rich, Vivid Details:**
   - Describe what you see in EXTREME detail, using all observations from your analysis
   - Paint a vivid picture: "Notice how...", "Look at the way...", "See how the..."
   - Mention specific details that make the scene unique or interesting
   - Include sensory details: colors, textures, sounds, movements, timing, spatial relationships

7. **Conversational but Comprehensive Tone:**
   - Maintain your warm, friendly "URIS" personality
   - Use natural, conversational language
   - Be enthusiastic about interesting details: "Oh, this is cool - I see...", "Wow, look at that!"
   - Can be humorous when appropriate
   - Make it feel like a real conversation, but with EXTENSIVE detail

**Example of Good Interactive Response:**

Q: "What is he doing?"

A: "Looking at this video carefully, I can see a person engaged in what appears to be lawn maintenance work. Let me describe everything I'm observing in detail.

The person in the video is an adult, likely male based on the body structure and clothing style, wearing what looks like a light-colored t-shirt - possibly white or light gray - and darker pants, probably jeans or work pants. The clothing suggests this is casual outdoor work attire, practical and comfortable for physical labor.

They're operating what I can clearly identify as a lawn mower - I can see the characteristic shape and the way they're guiding it across the grass. The movement pattern is very systematic and deliberate - they're moving in straight, parallel lines across what appears to be a residential lawn or yard. This methodical approach is typical of someone who either has experience with lawn care or is being very careful to achieve an even, professional-looking cut.

The setting appears to be a private property, likely someone's backyard or front yard. I can observe green grass that looks relatively well-maintained even before this mowing session, suggesting regular care. The grass height indicates it's probably been a week or two since the last mowing - not overgrown, but definitely ready for maintenance.

Looking at the background, I can make out what seems to be residential structures - possibly a house or fence in the distance, though the focus is primarily on the person and their immediate work area. The lighting suggests this is taking place during daytime, probably mid-morning or afternoon based on the shadow patterns and the brightness of the natural light.

What's particularly interesting about the technique being used is the overlap pattern - they're ensuring each pass slightly overlaps with the previous one, which is the correct way to avoid leaving uncut strips of grass. The pace is steady and controlled, not rushed, which again suggests either experience or a careful attention to doing the job properly.

The person's posture is slightly forward-leaning, which is typical when pushing a walk-behind mower. Their hands are firmly gripping the mower's handle, maintaining control throughout the movement. The body language suggests focus and physical engagement with the task - this isn't someone casually going through the motions, but rather someone actively working and paying attention to what they're doing.

From what I can observe of the equipment, it appears to be a standard residential push mower, probably gasoline-powered based on the way it's being operated, though I'd need a clearer view to be completely certain. The cutting path width looks like a typical residential mower, probably 20-22 inches.

The environment seems pleasant for outdoor work - no visible signs of rain or extreme heat that would make the task uncomfortable. The grass isn't wet (which would make mowing more difficult), and there's no visible debris that would need to be cleared first.

This type of work typically takes anywhere from 30 minutes to an hour depending on the yard size, and it's one of those regular maintenance tasks that needs to be done weekly or bi-weekly during the growing season. It's actually quite satisfying work when you see the neat, clean lines left behind.

What specifically caught your attention about this activity? Are you curious about the mowing technique, wondering if this is the right time of day for lawn care, or perhaps interested in the equipment being used? I'd be happy to elaborate on any particular aspect of lawn maintenance that interests you!"

**Length Guidelines:**
- **Image/Video Analysis**: MINIMUM 8-15 paragraphs covering ALL visible elements
- **Simple questions**: 8-10 paragraphs with exhaustive detail
- **Planning/How-to questions**: 12-20+ paragraphs with comprehensive information
- **Complex questions**: 10-15+ paragraphs with extensive analysis
- **COUNT YOUR PARAGRAPHS**: If less than 8, you're not providing enough detail
- **Always err on the side of MORE detail, not less**

**Example of EXTENSIVE Response for Planning Question:**

Q: "I want to throw a birthday party, give me a detailed plan"

A: [Should be 10-15+ paragraphs covering: Date/Time selection with multiple considerations, Venue options with detailed pros/cons for each, Guest list management with specific numbers and considerations, Theme selection with multiple theme ideas and detailed descriptions, Decorations with specific items and where to get them, Food planning with menu options and quantities, Activities/Entertainment with detailed game/activity descriptions, Timeline with week-by-week and day-by-day breakdown, Budget breakdown with approximate costs, Contingency plans, Tips and tricks, etc.]

**Avoid Template Responses:**
❌ DON'T: Always use "First, you need to... Second, you should... Third, don't forget..."
❌ DON'T: Always structure as "Section 1: X, Section 2: Y, Section 3: Z"
❌ DON'T: Use mechanical phrases like "based on analysis", "in summary", "in conclusion" in a template way
❌ DON'T: Always follow the same response pattern regardless of the question

✅ DO: Vary your response structure based on what feels natural for each question
✅ DO: Start differently each time - sometimes with the answer, sometimes with context, sometimes with a question
✅ DO: Let information flow naturally, like you're explaining to a friend
✅ DO: Use natural transitions: "Oh, and another thing...", "Actually, I should mention...", "You know what's interesting..."
✅ DO: Present information in whatever order makes the most sense for that specific question

Remember: You are a CONVERSATIONAL PARTNER who provides EXTENSIVE, DETAILED, COMPREHENSIVE answers in a NATURAL, FLEXIBLE way. Never give brief answers. Always provide substantial detail, but present it naturally - not as a rigid template. If a question asks for a "detailed plan" or "detailed explanation", your answer should be EXTREMELY comprehensive (10-15+ paragraphs) with all the necessary information, but woven together naturally like you're having a real conversation. Engage, ask questions, show curiosity, and build a dialogue - but always with EXTENSIVE detail presented in a natural, non-template way!
"""

# Context window configuration
MAX_CONTEXT_MESSAGES = 20  # Maximum 20 messages to retain (10 conversation rounds: 10 user + 10 assistant)

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
    st.session_state.processed_video_cache = {}  # 缓存处理过的视频

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
    """Load model and processor (4bit quantization mode, optimized for 24G Mac memory)"""
    # 检查 GPU 可用性（支持 CUDA 和 MPS）
    gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()

    if not gpu_available:
        # 收集诊断信息
        import sys
        diagnostic_info = f"""
❌ 未检测到可用的 GPU！

诊断信息：
- PyTorch 版本: {torch.__version__}
- Python 版本: {sys.version.split()[0]}
- CUDA 可用: {torch.cuda.is_available()}
- MPS (Apple Silicon) 可用: {torch.backends.mps.is_available()}
"""

        try:
            if hasattr(torch.version, 'cuda'):
                diagnostic_info += f"- PyTorch 编译的 CUDA 版本: {torch.version.cuda}\n"
        except:
            pass

        diagnostic_info += """
可能的原因：
1. 未安装支持 GPU 的 PyTorch 版本
2. 在 Mac 上，需要安装支持 MPS 的 PyTorch
3. GPU 硬件不存在或未被识别

解决方案：
对于 Mac (Apple Silicon):
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

对于 NVIDIA GPU:
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

确认 GPU 驱动版本与 PyTorch 兼容
"""
        raise RuntimeError(diagnostic_info)

    # Display GPU information
    device_type = "CUDA" if torch.cuda.is_available() else "MPS"
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
        print(f"[OK] Detected {gpu_count} {device_type} GPU: {gpu_name}")
    else:
        print(f"[OK] Detected Apple Silicon GPU (MPS)")
    
    # Determine model path: prioritize local path, otherwise use HuggingFace model ID
    model_name_or_path = "Qwen/Qwen2.5-VL-7B-Instruct"

    # If local model path is set and exists, use local path
    if LOCAL_MODEL_PATH and os.path.exists(LOCAL_MODEL_PATH):
        model_name_or_path = LOCAL_MODEL_PATH
    else:
        # Check HuggingFace cache (only check on first load)
        cache_exists, cache_path = check_model_cache()
        if cache_exists:
            # Model is cached, HuggingFace will automatically use cache, no additional operations needed
            pass

    # Load base model (4bit quantization mode, optimized for 24G Mac memory)
    # HuggingFace will automatically use local cache (if exists), no re-download needed

    # Check if Flash Attention 2 is available (CUDA only)
    use_flash_attention = False
    if torch.cuda.is_available():
        try:
            import flash_attn
            use_flash_attention = True
            print("[INFO] Flash Attention 2 available, enabling acceleration")
        except ImportError:
            print("[WARNING] Flash Attention 2 not installed, using default attention mechanism")
            print("[Tip] Install command: pip install flash-attn --no-build-isolation")
    else:
        print("[INFO] Using MPS (Apple Silicon), skipping Flash Attention 2")

    # Configure 4bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 启用4bit量化
        bnb_4bit_compute_dtype=torch.bfloat16,  # 计算使用bfloat16精度
        bnb_4bit_use_double_quant=True,  # 使用双重量化以节省更多内存
        bnb_4bit_quant_type="nf4",  # 使用NF4量化类型
    )

    max_retries = 2
    for attempt in range(max_retries):
        try:
            model_kwargs = {
                "quantization_config": quantization_config,  # 4bit量化配置
                "device_map": "auto",  # 自动分配到 GPU
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,  # 基础数据类型
            }

            # 如果 Flash Attention 2 可用且使用CUDA，启用它
            if use_flash_attention and torch.cuda.is_available():
                model_kwargs["attn_implementation"] = "flash_attention_2"
            
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                **model_kwargs
            )
            break  # 成功加载，退出重试循环
        except RuntimeError as e:
            if "size mismatch" in str(e) and attempt < max_retries - 1:
                # 如果出现尺寸不匹配错误，尝试清除缓存后重新加载
                import shutil
                import warnings
                cache_path = os.path.expanduser(
                    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                )
                if os.path.exists(cache_path):
                    warnings.warn(
                        f"Detected model weight size mismatch, clearing cache and re-downloading...\n"
                        f"Cache path: {cache_path}"
                    )
                    try:
                        shutil.rmtree(cache_path, ignore_errors=True)
                        print(f"[OK] Cache cleared, will re-download model...")
                    except Exception as cleanup_error:
                        print(f"[WARNING] Error clearing cache: {cleanup_error}")
                        print(f"   Please manually delete cache directory: {cache_path}")
                else:
                    warnings.warn("Cache path does not exist, may be caused by other issues")
                # 继续下一次尝试
                continue
            else:
                # 最后一次尝试失败，或者不是尺寸不匹配错误，抛出异常
                if "size mismatch" in str(e):
                    raise RuntimeError(
                        "Model loading failed: weight size mismatch.\n"
                        "Attempted to clear cache but issue persists.\n"
                        "Possible causes:\n"
                        "1. transformers version incompatible with model\n"
                        "2. Model files corrupted\n"
                        "Suggestions:\n"
                        "1. Check if transformers version is 4.57.0+\n"
                        "2. Manually clear cache: huggingface-cli delete-cache\n"
                        "3. Re-run the program"
                    ) from e
                raise
    
    # 🚀 优化1: 启用 KV 缓存以加速生成
    model.config.use_cache = True
    
    # Check if adapter path exists and load
    if os.path.exists(ADAPTER_PATH):
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    # Load processor (also uses cache)
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    
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
        
        st.header("⚙️ Model Configuration")
        max_new_tokens = st.slider(
            "Max New Tokens",
            min_value=128,
            max_value=4096,
            value=2048,
            step=128,
            help="Maximum length of generated text (A100 GPU supports longer outputs)"
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
                                # 🚀 优化4: 智能视频采样策略
                                # 根据视频长度动态调整 fps
                                video_path = st.session_state.video_path
                                
                                # 获取视频时长（用于优化采样率）
                                try:
                                    cap = cv2.VideoCapture(video_path)
                                    video_fps = cap.get(cv2.CAP_PROP_FPS)
                                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                    duration = frame_count / video_fps if video_fps > 0 else 10
                                    cap.release()
                                    
                                    # 动态调整采样率：
                                    # 短视频(< 10s): 1.0 fps (每秒1帧)
                                    # 中视频(10-30s): 0.5 fps (每2秒1帧)
                                    # 长视频(> 30s): 0.33 fps (每3秒1帧)
                                    if duration < 10:
                                        optimal_fps = 1.0
                                    elif duration < 30:
                                        optimal_fps = 0.5
                                    else:
                                        optimal_fps = 0.33
                                    
                                    print(f"[视频优化] 视频时长: {duration:.1f}s, 采样率: {optimal_fps} fps")
                                except Exception as e:
                                    print(f"[视频优化] 无法获取视频信息: {e}")
                                    optimal_fps = 0.5  # 默认值
                                
                                # Video content configuration: optimized sampling
                                video_content = {
                                    "type": "video",
                                    "video": video_path,
                                    # 动态帧率，平衡性能和质量
                                    "fps": optimal_fps,
                                    # 保持合理的分辨率上限
                                    "max_pixels": 1280 * 720,  # 720p 上限，平衡质量和速度
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
