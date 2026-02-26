#!/usr/bin/env python3
from __future__ import annotations
"""
URIS Qwen2.5-VL-7B LoRA 微调训练脚本（A100 最佳效果版）
===========================================================
目标：训练一个能输出丰富、有逻辑、结合上下文的回答，
      并具备用户行为预测能力的家居交互 VLM。

模型：Qwen2.5-VL-7B-Instruct (bf16 全精度 LoRA，A100 优化)
数据集：
  1. 自动合成的家居交互预测数据（核心，2000 条）
  2. 歧义消歧 + 澄清策略数据
  3. 多轮上下文连贯对话数据

运行环境：A100 (40GB/80GB)
用法：直接在 Colab 中逐 Cell 运行，或 `python train_qwen25vl_uris_predictive.py`
"""

# ============================================================
# Cell 1: 环境安装（Colab 中运行）
# ============================================================

INSTALL_COMMANDS = """
# 在 Colab 中运行以下命令安装依赖
!pip install -q transformers>=4.45.0 peft>=0.13.0 bitsandbytes>=0.43.0
!pip install -q accelerate>=0.34.0 datasets>=2.20.0 trl>=0.12.0
!pip install -q qwen-vl-utils Pillow requests tqdm
!pip install -q wandb  # 可选：训练可视化
"""

import copy
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ============================================================
# Cell 2: 核心配置
# ============================================================

@dataclass
class URISTrainingConfig:
    """训练超参数与路径配置 — A100 最佳效果版"""
    # --- 模型 ---
    base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    use_4bit: bool = False   # A100 不量化，bf16 全精度 LoRA 效果最好
    use_flash_attn: bool = True  # A100 原生支持 flash_attention_2

    # --- LoRA（拉满配置）---
    lora_r: int = 128           # 更高 rank = 更强表达能力
    lora_alpha: int = 256       # alpha = 2 * r，标准高质量配比
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj",       # MLP
    ])

    # --- 训练（A100 优化）---
    num_epochs: int = 5             # 更多 epoch 让模型充分学习行为预测模式
    batch_size: int = 4             # A100 80GB 可开 4; 40GB 开 2
    gradient_accumulation_steps: int = 4  # effective batch = 16
    learning_rate: float = 5e-5     # 全精度 LoRA 用稍低 lr 更稳定
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.06
    warmup_steps: int = 0           # 0 表示用 warmup_ratio 计算
    weight_decay: float = 0.01      # 正则化防过拟合
    max_grad_norm: float = 1.0      # 梯度裁剪
    max_seq_length: int = 8192      # A100 显存充足，拉长上下文窗口
    bf16: bool = True               # A100 原生 bf16
    gradient_checkpointing: bool = True  # 用计算换显存，支持更大 batch
    optim: str = "adamw_torch_fused"     # A100 上 fused Adam 更快

    # --- 数据 ---
    num_synthetic_samples: int = 2000  # 拉满样本量
    val_ratio: float = 0.1
    seed: int = 42

    # --- 路径 (Colab) ---
    output_dir: str = "/content/drive/MyDrive/URIS/checkpoints/Qwen2.5-VL-URIS-Predictive-A100"
    data_dir: str = "/content/drive/MyDrive/URIS/train_data"
    cache_dir: str = "/content/cache"

    # --- 评估 ---
    eval_steps: int = 50
    save_steps: int = 100
    save_total_limit: int = 5
    logging_steps: int = 5     # 更频繁的日志
    load_best_model_at_end: bool = True  # 训练结束自动加载最优 checkpoint
    metric_for_best_model: str = "eval_loss"


# ============================================================
# Cell 3: URIS 专用 System Prompt（含行为预测指令）
# ============================================================

URIS_SYSTEM_PROMPT = """你是 URIS，一个专业的家居环境多模态交互助手。你的核心能力包括：

【回答原则】
1. 先给出直接结论，再展开详细分析和依据
2. 回答必须丰富、有层次、有逻辑，包含：观察事实→分析推理→建议行动→注意事项
3. 严格区分"我观察到的"和"我推断的"，标注置信度
4. 结合对话上下文和用户历史行为模式，给出个性化回答

【行为预测机制 - 核心能力】
你必须根据用户的当前行为和上下文，主动预测用户的下一步需求：
- 用户询问水杯 → 预测："需要我帮你确认杯子里是否有水吗？" 或 "需要提醒你喝水吗？"
- 用户找钥匙 → 预测："你是准备出门吗？需要我帮你检查还有没有其他随身物品？"
- 用户问桌面物品 → 预测："需要我帮你整理桌面吗？我可以建议物品的分类归位。"
- 用户关心时间 → 预测："你是否有即将到来的安排？需要提醒吗？"

【输出格式】
你的回答分为两部分：
1. 面向用户的自然语言回复（丰富、友好、有预测性建议）
2. 结构化 JSON 分析块

JSON 必须包含以下字段：
- intent: 用户意图分类
- user_goal: 用户目标描述
- observed_objects: 观察到的物体列表
- spatial_relations: 空间关系
- scene_summary: 场景摘要
- recommendation_steps: 建议步骤列表
- predicted_next_action: 预测用户下一步行为
- proactive_suggestion: 主动预测性建议
- clarification_needed: 是否需要澄清
- clarifying_question: 澄清问题
- confidence: 置信度 0-1
- evidence_basis: 证据来源列表
- limitations: 局限性说明
"""

# ============================================================
# Cell 4: 家居交互场景数据合成器
# ============================================================

# 家居场景模板库
HOME_SCENES = {
    "kitchen": {
        "name": "厨房",
        "objects": ["杯子", "碗", "盘子", "筷子", "锅", "水壶", "微波炉", "冰箱", "调料瓶", "砧板", "刀", "抹布"],
        "relations": ["在台面上", "在水槽旁", "在冰箱里", "在架子上", "在灶台旁"],
    },
    "living_room": {
        "name": "客厅",
        "objects": ["遥控器", "杯子", "书", "手机", "充电器", "抱枕", "茶几", "花瓶", "纸巾盒", "眼镜"],
        "relations": ["在沙发上", "在茶几上", "在电视柜旁", "在窗台上", "在地上"],
    },
    "bedroom": {
        "name": "卧室",
        "objects": ["手机", "闹钟", "眼镜", "书", "水杯", "台灯", "充电线", "钥匙", "钱包", "耳机"],
        "relations": ["在床头柜上", "在床上", "在书桌上", "在衣柜旁", "在抽屉里"],
    },
    "study": {
        "name": "书房",
        "objects": ["电脑", "键盘", "鼠标", "笔", "笔记本", "文件夹", "台灯", "水杯", "手机", "耳机"],
        "relations": ["在书桌上", "在显示器旁", "在键盘左边", "在书架上", "在抽屉里"],
    },
    "entrance": {
        "name": "玄关",
        "objects": ["钥匙", "鞋", "伞", "包", "帽子", "手套", "口罩", "外套", "快递", "鞋柜"],
        "relations": ["在鞋柜上", "在挂钩上", "在门旁", "在地上", "在篮子里"],
    },
}

# 用户行为模式 → 预测性建议映射
BEHAVIOR_PREDICTION_MAP = [
    {
        "trigger_keywords": ["水杯", "杯子", "喝水", "水"],
        "user_queries": [
            "帮我看看桌上的杯子",
            "我的水杯在哪里？",
            "杯子里还有水吗？",
            "那个杯子是我的吗？",
        ],
        "predicted_actions": [
            "用户可能需要喝水或补充水分",
            "用户可能在寻找自己的专属水杯",
        ],
        "proactive_suggestions": [
            "需要我帮你确认杯子里是否还有水吗？如果需要补水，厨房水壶里应该还有热水。",
            "看起来你在找水杯，需要我提醒你定时喝水吗？保持充足的水分很重要。",
        ],
        "intent": "find_drink",
    },
    {
        "trigger_keywords": ["钥匙", "出门", "门"],
        "user_queries": [
            "我的钥匙放在哪了？",
            "帮我看看钥匙在不在桌上",
            "钥匙在哪个位置？",
        ],
        "predicted_actions": [
            "用户可能准备出门",
            "用户可能在做出门前的检查",
        ],
        "proactive_suggestions": [
            "你是准备出门吗？除了钥匙，需要我帮你检查手机、钱包等随身物品是否都在吗？",
            "钥匙找到了。建议出门前确认：手机、钱包、钥匙三件套是否齐全？今天天气预报显示可能下雨，要不要带伞？",
        ],
        "intent": "prepare_to_leave",
    },
    {
        "trigger_keywords": ["整理", "乱", "桌面", "收拾"],
        "user_queries": [
            "桌面上都有什么东西？",
            "帮我看看桌面需不需要整理",
            "这些东西怎么归类比较好？",
        ],
        "predicted_actions": [
            "用户想整理桌面或房间",
            "用户可能在准备工作或学习环境",
        ],
        "proactive_suggestions": [
            "我可以帮你分析桌面物品的分类归位建议：常用物品放近处、不常用的收纳起来。需要我给出具体的整理方案吗？",
            "桌面看起来有些凌乱。建议按使用频率分三区：高频区（正前方）、中频区（侧面）、低频区（抽屉）。要我详细说明吗？",
        ],
        "intent": "organize_space",
    },
    {
        "trigger_keywords": ["手机", "充电", "电量"],
        "user_queries": [
            "我的手机在哪？",
            "手机放在哪里了？",
            "充电器在旁边吗？",
        ],
        "predicted_actions": [
            "用户可能需要使用手机",
            "用户可能需要给手机充电",
        ],
        "proactive_suggestions": [
            "手机找到了。需要我确认附近是否有充电器？如果电量低的话可以顺便充电。",
            "你的手机在那边。对了，需要我帮你检查一下充电线是否也在附近吗？",
        ],
        "intent": "find_device",
    },
    {
        "trigger_keywords": ["吃", "食物", "饿", "零食", "做饭"],
        "user_queries": [
            "厨房台面上有什么？",
            "冰箱旁边有什么食物？",
            "看看有没有可以吃的东西",
        ],
        "predicted_actions": [
            "用户可能饿了想找食物",
            "用户可能在计划准备餐食",
        ],
        "proactive_suggestions": [
            "你是不是饿了？除了台面上这些，需要我帮你看看冰箱附近还有什么可用的食材吗？",
            "看起来你在找吃的。如果需要做饭，我可以根据台面上现有的食材帮你建议简单菜谱。",
        ],
        "intent": "find_food",
    },
    {
        "trigger_keywords": ["书", "学习", "笔", "文件", "作业"],
        "user_queries": [
            "书桌上的书是哪一本？",
            "我的笔记本在哪？",
            "帮我看看学习资料在不在",
        ],
        "predicted_actions": [
            "用户准备开始学习或工作",
            "用户在准备学习环境",
        ],
        "proactive_suggestions": [
            "你是准备开始学习吗？需要我帮你确认学习区域是否整洁、光线是否充足吗？良好的学习环境能提升效率。",
            "学习资料找到了。需要我帮你规划一下学习区域的物品摆放吗？建议把常用参考书放在触手可及的位置。",
        ],
        "intent": "prepare_study",
    },
    {
        "trigger_keywords": ["眼镜", "看不清"],
        "user_queries": [
            "我的眼镜在哪？",
            "帮我找一下眼镜",
            "眼镜放桌上了吗？",
        ],
        "predicted_actions": [
            "用户需要眼镜来阅读或使用电子设备",
        ],
        "proactive_suggestions": [
            "眼镜找到了。你是准备看书或用电脑吗？记得注意用眼时间，建议每40分钟休息一次眼睛。",
        ],
        "intent": "find_accessory",
    },
]

# 多同类物体的歧义场景
AMBIGUITY_SCENARIOS = [
    {
        "scene": "kitchen",
        "situation": "桌上有两个杯子，一个白色在左边，一个蓝色在右边",
        "query": "帮我看看那个杯子",
        "expected_action": "clarify_reference",
        "response": "我在桌上看到了两个杯子：左边有一个白色的杯子，右边有一个蓝色的杯子。你指的是哪一个呢？可以告诉我颜色或位置。",
        "analysis": {
            "intent": "clarify_reference",
            "clarification_needed": True,
            "clarifying_question": "你指的是左边的白色杯子，还是右边的蓝色杯子？",
            "confidence": 0.35,
            "predicted_next_action": "用户会指定具体的杯子特征",
            "proactive_suggestion": "确认后我可以帮你检查杯子的状态，需要的话也可以提醒你补水。",
        },
    },
    {
        "scene": "living_room",
        "situation": "茶几上有两个遥控器，一个电视遥控器一个空调遥控器",
        "query": "把那个遥控器给我",
        "expected_action": "clarify_reference",
        "response": "茶几上有两个遥控器：一个看起来是电视遥控器，另一个是空调遥控器。你需要哪一个？",
        "analysis": {
            "intent": "clarify_reference",
            "clarification_needed": True,
            "clarifying_question": "你需要的是电视遥控器还是空调遥控器？",
            "confidence": 0.4,
            "predicted_next_action": "用户会指定需要哪个遥控器",
            "proactive_suggestion": "如果你要看电视，需要我帮你确认电视是否已经开机吗？",
        },
    },
]


def _generate_detection_list(objects: list[str], relations: list[str], n: int = 3) -> list[dict]:
    """生成模拟检测结果"""
    selected = random.sample(objects, min(n, len(objects)))
    detections = []
    for i, obj in enumerate(selected):
        detections.append({
            "label": obj,
            "confidence": round(random.uniform(0.65, 0.98), 3),
            "bbox": [
                random.randint(50, 300),
                random.randint(50, 300),
                random.randint(301, 600),
                random.randint(301, 600),
            ],
            "center_norm": [round(random.uniform(0.1, 0.9), 3), round(random.uniform(0.1, 0.9), 3)],
            "obj_id": f"obj-{i+1:04d}",
        })
    return detections


def _generate_object_registry(detections: list[dict]) -> list[dict]:
    """从检测结果生成 Object Registry"""
    registry = []
    for det in detections:
        registry.append({
            "obj_id": det["obj_id"],
            "label": det["label"],
            "center_norm": det["center_norm"],
            "confidence": det["confidence"],
            "status": "visible",
            "seen_count": random.randint(1, 20),
            "mention_count": random.randint(0, 5),
        })
    return registry


def _build_rich_response(
    query: str,
    scene_name: str,
    objects: list[str],
    behavior: dict,
    detections: list[dict],
) -> dict[str, Any]:
    """构建丰富的、有预测能力的训练样本"""
    observed = [d["label"] for d in detections]
    pred_idx = random.randint(0, len(behavior["predicted_actions"]) - 1)
    sug_idx = random.randint(0, len(behavior["proactive_suggestions"]) - 1)

    # 构建丰富的自然语言回复
    response_parts = [
        f"好的，让我帮你看看{scene_name}的情况。",
        f"\n\n**【观察结果】**\n我在当前画面中检测到了以下物品：{'、'.join(observed)}。",
    ]

    # 添加空间关系描述
    if len(observed) > 1:
        response_parts.append(
            f"\n\n**【空间分析】**\n"
            f"从画面位置来看，{observed[0]}在画面的"
            f"{'左侧' if detections[0]['center_norm'][0] < 0.5 else '右侧'}，"
            f"而{observed[1]}在{'左侧' if detections[1]['center_norm'][0] < 0.5 else '右侧'}。"
        )

    # 添加针对性分析
    response_parts.append(
        f"\n\n**【分析与建议】**\n"
        f"针对你的问题「{query}」，"
        f"根据当前场景和你的使用习惯，{behavior['predicted_actions'][pred_idx]}。"
    )

    # 添加预测性建议（核心差异化能力）
    response_parts.append(
        f"\n\n**【贴心预测】** 💡\n{behavior['proactive_suggestions'][sug_idx]}"
    )

    user_response = "".join(response_parts)

    analysis_json = {
        "intent": behavior["intent"],
        "user_goal": query,
        "observed_objects": observed,
        "spatial_relations": [
            f"{d['label']}位于画面{'左侧' if d['center_norm'][0] < 0.5 else '右侧'}"
            for d in detections[:3]
        ],
        "scene_summary": f"{scene_name}场景，检测到{len(observed)}个物体：{'、'.join(observed)}",
        "recommendation_steps": [
            f"确认{observed[0]}的位置和状态",
            "根据用户需求提供具体操作建议",
            "预测用户可能的后续需求并主动提示",
        ],
        "predicted_next_action": behavior["predicted_actions"][pred_idx],
        "proactive_suggestion": behavior["proactive_suggestions"][sug_idx],
        "clarification_needed": False,
        "clarifying_question": "",
        "confidence": round(random.uniform(0.7, 0.92), 2),
        "evidence_basis": ["camera_frame", "yolo_detection", "object_registry", "behavior_pattern"],
        "limitations": ["基于2D图像分析，深度信息有限", "物体状态（如杯中水量）需要更近距离确认"],
    }

    return {
        "user_response": user_response,
        "analysis_json": analysis_json,
    }


def synthesize_uris_dataset(config: URISTrainingConfig) -> list[dict[str, Any]]:
    """合成 URIS 家居交互预测训练数据集"""
    random.seed(config.seed)
    samples = []

    # === 类型1: 行为预测交互样本（主力，60%）===
    n_predictive = int(config.num_synthetic_samples * 0.6)
    for _ in range(n_predictive):
        scene_key = random.choice(list(HOME_SCENES.keys()))
        scene = HOME_SCENES[scene_key]
        behavior = random.choice(BEHAVIOR_PREDICTION_MAP)

        query = random.choice(behavior["user_queries"])
        n_det = random.randint(2, 5)
        detections = _generate_detection_list(scene["objects"], scene["relations"], n=n_det)

        # 确保至少有一个触发物体在检测结果中
        trigger_obj = random.choice(behavior["trigger_keywords"])
        for cand_obj in scene["objects"]:
            if any(kw in cand_obj for kw in behavior["trigger_keywords"]):
                trigger_obj = cand_obj
                break
        if not any(d["label"] == trigger_obj for d in detections):
            detections[0]["label"] = trigger_obj

        registry = _generate_object_registry(detections)
        target = _build_rich_response(query, scene["name"], scene["objects"], behavior, detections)

        scene_summary = f"{scene['name']}，检测到 {len(detections)} 个物体"

        samples.append({
            "user_query": query,
            "scene_summary": scene_summary,
            "detections": detections,
            "object_registry": registry,
            "preferences": ["中文回答", "详细分析", "行为预测"],
            "recent_turns": [],
            "recent_scene_events": [
                {"ts": time.time(), "type": "scene_stable", "message": "场景稳定"}
            ],
            "reference_resolution": {"resolved": True, "method": "label_match"},
            "target": target,
            "task_type": "predictive_interaction",
            "scene_id": scene_key,
        })

    # === 类型2: 歧义消歧 + 澄清样本（20%）===
    n_ambiguity = int(config.num_synthetic_samples * 0.2)
    for _ in range(n_ambiguity):
        scenario = random.choice(AMBIGUITY_SCENARIOS)
        scene = HOME_SCENES[scenario["scene"]]
        detections = _generate_detection_list(scene["objects"], scene["relations"], n=4)
        registry = _generate_object_registry(detections)

        full_analysis = {
            **scenario["analysis"],
            "user_goal": scenario["query"],
            "observed_objects": [d["label"] for d in detections],
            "spatial_relations": [f"{d['label']}在画面中" for d in detections[:2]],
            "scene_summary": scenario["situation"],
            "recommendation_steps": ["先澄清目标对象", "确认后给出针对性建议"],
            "evidence_basis": ["yolo_detection", "object_registry"],
            "limitations": ["多个同类物体需要用户进一步指定"],
        }

        samples.append({
            "user_query": scenario["query"],
            "scene_summary": scenario["situation"],
            "detections": detections,
            "object_registry": registry,
            "preferences": ["中文回答"],
            "recent_turns": [],
            "recent_scene_events": [],
            "reference_resolution": {
                "resolved": False,
                "clarification_needed": True,
                "method": "ambiguous_deictic",
            },
            "target": {
                "user_response": scenario["response"],
                "analysis_json": full_analysis,
            },
            "task_type": "disambiguation",
            "scene_id": scenario["scene"],
        })

    # === 类型3: 多轮上下文连贯样本（20%）===
    n_context = config.num_synthetic_samples - n_predictive - n_ambiguity
    for _ in range(n_context):
        scene_key = random.choice(list(HOME_SCENES.keys()))
        scene = HOME_SCENES[scene_key]
        detections = _generate_detection_list(scene["objects"], scene["relations"], n=3)
        registry = _generate_object_registry(detections)
        obj_label = detections[0]["label"]

        # 模拟之前的对话轮次
        recent_turns = [
            {"role": "user", "content": f"帮我看看{scene['name']}有什么"},
            {"role": "assistant", "content": f"我在{scene['name']}检测到了{'、'.join(d['label'] for d in detections)}。"},
        ]

        followup_query = random.choice([
            f"那个{obj_label}具体在什么位置？",
            f"{obj_label}旁边还有什么？",
            f"帮我详细描述一下{obj_label}的情况",
        ])

        nearby = [d["label"] for d in detections[1:3]] if len(detections) > 1 else ["暂无其他物品"]
        target_response = (
            f"根据刚才的检测结果，让我为你详细说明{obj_label}的情况。\n\n"
            f"**【位置信息】**\n"
            f"{obj_label}位于画面的{'左侧' if detections[0]['center_norm'][0] < 0.5 else '右侧'}区域，"
            f"检测置信度为 {detections[0]['confidence']}。\n\n"
            f"**【周围环境】**\n"
            f"在{obj_label}附近，我还观察到了：{'、'.join(nearby)}。\n\n"
            f"**【贴心建议】** 💡\n"
            f"需要我对{obj_label}做更详细的分析吗？或者你可以调整摄像头角度让我看得更清楚。"
        )

        samples.append({
            "user_query": followup_query,
            "scene_summary": f"{scene['name']}，持续检测中",
            "detections": detections,
            "object_registry": registry,
            "preferences": ["中文回答", "结合上下文"],
            "recent_turns": recent_turns,
            "recent_scene_events": [
                {"ts": time.time() - 10, "type": "scene_stable", "message": "场景稳定"},
            ],
            "reference_resolution": {
                "resolved": True,
                "selected_obj_id": detections[0]["obj_id"],
                "method": "context_continuation",
            },
            "target": {
                "user_response": target_response,
                "analysis_json": {
                    "intent": "context_followup",
                    "user_goal": followup_query,
                    "observed_objects": [d["label"] for d in detections],
                    "spatial_relations": [
                        f"{obj_label}在画面{'左' if detections[0]['center_norm'][0] < 0.5 else '右'}侧"
                    ],
                    "scene_summary": f"{scene['name']}场景持续观察",
                    "recommendation_steps": [
                        f"提供{obj_label}的详细位置和状态",
                        "描述周围物品关系",
                        "预测用户可能的后续操作",
                    ],
                    "predicted_next_action": f"用户可能需要对{obj_label}进行进一步操作",
                    "proactive_suggestion": f"如果需要更清晰的画面，可以调整摄像头角度。",
                    "clarification_needed": False,
                    "clarifying_question": "",
                    "confidence": 0.82,
                    "evidence_basis": ["camera_frame", "yolo_detection", "dialogue_context"],
                    "limitations": ["基于2D图像", "物体细节可能需更近距离确认"],
                },
            },
            "task_type": "context_followup",
            "scene_id": scene_key,
        })

    random.shuffle(samples)
    return samples


# ============================================================
# Cell 5: 数据格式化（LLaMA-Factory ShareGPT 兼容）
# ============================================================

def format_for_training(
    samples: list[dict[str, Any]],
    system_prompt: str = URIS_SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    """将合成数据转为 SFT 训练格式"""
    formatted = []
    for sample in samples:
        context_payload = {
            "scene_summary": sample.get("scene_summary", ""),
            "detections": sample.get("detections", []),
            "object_registry": sample.get("object_registry", []),
            "recent_scene_events": sample.get("recent_scene_events", []),
            "reference_resolution": sample.get("reference_resolution", {}),
            "preferences": sample.get("preferences", []),
            "recent_turns": sample.get("recent_turns", []),
        }

        user_content = (
            f"请基于以下实时场景信息进行交互分析与建议（仅仿真，不控制机器人运动）。\n"
            f"user_query: {sample['user_query']}\n"
            f"context_json:\n{json.dumps(context_payload, ensure_ascii=False, indent=2)}\n\n"
            f"请输出：\n"
            f"1) 面向用户的自然语言回答（先给结论，内容丰富，包含预测性建议）\n"
            f"2) 一个JSON代码块，包含 user_response 和 analysis_json 字段\n"
            f"3) analysis_json 中必须包含 predicted_next_action 和 proactive_suggestion 字段"
        )

        target = sample["target"]
        assistant_content = (
            f"{target['user_response']}\n\n"
            f"```json\n"
            f"{json.dumps(target, ensure_ascii=False, indent=2)}\n"
            f"```"
        )

        formatted.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "task_type": sample.get("task_type", "unknown"),
            "scene_id": sample.get("scene_id", ""),
        })

    return formatted


# ============================================================
# Cell 6: 训练主流程
# ============================================================

def train(config: URISTrainingConfig | None = None):
    """主训练函数"""
    if config is None:
        config = URISTrainingConfig()

    print("=" * 60)
    print("URIS Qwen2.5-VL Predictive Interaction Training")
    print("=" * 60)

    # --- Step 1: 合成数据 ---
    print("\n[1/5] 合成训练数据...")
    raw_samples = synthesize_uris_dataset(config)
    formatted = format_for_training(raw_samples)

    # 切分 train / val
    random.seed(config.seed)
    random.shuffle(formatted)
    val_size = max(1, int(len(formatted) * config.val_ratio))
    train_data = formatted[val_size:]
    val_data = formatted[:val_size]

    print(f"  总样本: {len(formatted)}")
    print(f"  训练集: {len(train_data)}")
    print(f"  验证集: {len(val_data)}")

    # 保存数据
    data_dir = Path(config.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(data_dir / "val.json", "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    print(f"  数据已保存到: {data_dir}")

    # --- Step 2: 加载模型 ---
    load_mode = "QLoRA 4-bit" if config.use_4bit else "bf16 全精度 LoRA (A100 最佳)"
    print(f"\n[2/5] 加载 Qwen2.5-VL-7B ({load_mode})...")

    import torch
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from transformers import Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset

    # A100 显存检测与自动适配
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  GPU: {gpu_name} ({vram_gb:.1f} GB)")
        if vram_gb < 50:  # A100 40GB 或更小
            print("  ⚠️  VRAM ≤ 40GB，自动切换到 QLoRA 4-bit + batch=2 + seq=4096")
            config.use_4bit = True
            config.batch_size = 2
            config.max_seq_length = 4096

    bnb_config = None
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if config.use_flash_attn else "eager",
    )

    processor = AutoProcessor.from_pretrained(
        config.base_model,
        trust_remote_code=True,
    )

    if config.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # A100: 开启 gradient checkpointing 节省显存
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # LoRA + gradient checkpointing 兼容
        print("  ✅ Gradient checkpointing enabled")

    # --- Step 3: 配置 LoRA ---
    print("\n[3/5] 配置 LoRA adapter...")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Step 4: 准备数据集 ---
    print("\n[4/5] 准备 HuggingFace Dataset...")

    def tokenize_fn(example):
        messages = example["messages"]
        text = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        encodings = processor.tokenizer(
            text,
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length",
            return_tensors="pt",
        )
        encodings["labels"] = encodings["input_ids"].clone()
        return {k: v.squeeze(0) for k, v in encodings.items()}

    train_dataset = Dataset.from_list(train_data).map(
        tokenize_fn, remove_columns=["messages", "task_type", "scene_id"],
    )
    val_dataset = Dataset.from_list(val_data).map(
        tokenize_fn, remove_columns=["messages", "task_type", "scene_id"],
    )

    # --- Step 5: 训练 ---
    print("\n[5/5] 开始训练...")
    from transformers import Trainer, TrainingArguments

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # A100 优化训练参数
    total_train_steps = (len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps)) * config.num_epochs
    warmup_steps = int(total_train_steps * config.warmup_ratio)
    print(f"  预计总训练步数: {total_train_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler,
        warmup_steps=warmup_steps,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        optim=config.optim,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor.tokenizer,
    )

    print(f"\n{'='*60}")
    print(f"  🚀 开始训练 (A100 全力模式)")
    print(f"  Model: {config.base_model}")
    print(f"  LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"  4-bit: {config.use_4bit} | bf16: {config.bf16}")
    print(f"  Batch: {config.batch_size} x {config.gradient_accumulation_steps} = {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Epochs: {config.num_epochs} | LR: {config.learning_rate}")
    print(f"  Max seq len: {config.max_seq_length}")
    print(f"  Samples: train={len(train_dataset)}, val={len(val_dataset)}")
    print(f"{'='*60}\n")

    trainer.train()

    # 保存 LoRA adapter
    adapter_save_path = output_dir / "final-adapter"
    model.save_pretrained(str(adapter_save_path))
    processor.save_pretrained(str(adapter_save_path))
    print(f"\n✅ LoRA adapter 已保存到: {adapter_save_path}")
    print("训练完成！")

    return model, processor


# ============================================================
# Cell 7: 推理测试
# ============================================================

def test_inference(model=None, processor=None, config=None):
    """训练后快速推理测试"""
    if config is None:
        config = URISTrainingConfig()

    if model is None or processor is None:
        import torch
        from transformers import AutoProcessor
        from transformers import Qwen2_5_VLForConditionalGeneration
        from peft import PeftModel

        print("加载基础模型...")
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.base_model,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        adapter_path = str(Path(config.output_dir) / "final-adapter")
        print(f"加载 LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        processor = AutoProcessor.from_pretrained(adapter_path, trust_remote_code=True)

    model.eval()

    # 测试用例
    test_cases = [
        {
            "query": "帮我看看桌上的水杯",
            "scene": "书房，检测到 水杯、键盘、鼠标、书本",
            "expect": "应该包含行为预测：是否需要喝水",
        },
        {
            "query": "我的钥匙在哪？",
            "scene": "玄关，检测到 钥匙、鞋、包",
            "expect": "应该预测用户准备出门",
        },
        {
            "query": "那个杯子是什么颜色的",
            "scene": "厨房，检测到 白色杯子(左)、蓝色杯子(右)",
            "expect": "应该触发澄清询问",
        },
    ]

    print("\n" + "=" * 60)
    print("推理测试")
    print("=" * 60)

    for i, tc in enumerate(test_cases):
        messages = [
            {"role": "system", "content": URIS_SYSTEM_PROMPT},
            {"role": "user", "content": f"user_query: {tc['query']}\nscene_summary: {tc['scene']}"},
        ]
        text = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor.tokenizer(text, return_tensors="pt").to(model.device)

        import torch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        response = processor.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        print(f"\n--- 测试 {i+1}: {tc['query']} ---")
        print(f"期望: {tc['expect']}")
        print(f"模型输出:\n{response[:500]}...")
        print()


# ============================================================
# Cell 8: 主入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="URIS Qwen2.5-VL Predictive Training")
    parser.add_argument("--mode", choices=["train", "test", "data-only"], default="train")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = URISTrainingConfig(
        num_synthetic_samples=args.samples,
        num_epochs=args.epochs,
    )
    if args.output_dir:
        config.output_dir = args.output_dir

    if args.mode == "data-only":
        print("仅生成数据...")
        samples = synthesize_uris_dataset(config)
        formatted = format_for_training(samples)
        out_path = Path(config.data_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "all_samples.json", "w", encoding="utf-8") as f:
            json.dump(formatted, f, ensure_ascii=False, indent=2)
        print(f"已生成 {len(formatted)} 条样本 -> {out_path / 'all_samples.json'}")

    elif args.mode == "train":
        model, processor = train(config)
        test_inference(model, processor, config)

    elif args.mode == "test":
        test_inference(config=config)
