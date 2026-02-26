from __future__ import annotations

import json
from typing import Sequence


PROMPT_VERSION = "uris-qwen-live-v1"

URIS_SYSTEM_PROMPT = """你是 URIS，一个专业的家居环境多模态交互助手。你的核心能力包括：

【回答原则】
1. 先给出直接结论，再展开详细分析和依据
2. 回答必须丰富、有层次、有逻辑，包含：观察事实→分析推理→建议行动→注意事项
3. 严格区分"我观察到的"和"我推断的"，标注置信度
4. 结合对话上下文和用户历史行为模式，给出个性化回答

【行为预测机制 - 核心能力】
根据用户当前行为和上下文，主动预测用户的下一步需求：
- 用户询问水杯 → 预测："需要我帮你确认杯子里是否有水吗？"
- 用户找钥匙 → 预测："你是准备出门吗？需要检查随身物品吗？"
- 用户问桌面物品 → 预测："需要我帮你整理桌面吗？"
- 用户找眼镜 → 预测："你是准备看书吗？记得注意用眼时间。"

【歧义处理】
当场景中有多个同类物体时，优先询问澄清问题，而非猜测用户意图。

【输出格式】
1. 面向用户的自然语言回复（丰富、友好、有预测性建议）
2. 结构化 JSON 分析块，必须包含以下字段：
   - intent, user_goal, observed_objects, spatial_relations
   - scene_summary, recommendation_steps
   - predicted_next_action, proactive_suggestion
   - clarification_needed, clarifying_question
   - confidence, evidence_basis, limitations
"""

REQUIRED_JSON_FIELDS = [
    "intent",
    "user_goal",
    "observed_objects",
    "spatial_relations",
    "scene_summary",
    "recommendation_steps",
    "clarification_needed",
    "clarifying_question",
    "confidence",
    "evidence_basis",
    "limitations",
]


def _compact_detections(detections: Sequence[dict], *, max_items: int) -> list[dict]:
    compacted: list[dict] = []
    for det in list(detections)[:max_items]:
        compacted.append(
            {
                "label": det.get("label"),
                "confidence": round(float(det.get("confidence", 0.0)), 3)
                if det.get("confidence") is not None
                else None,
                "bbox": det.get("bbox"),
                "center_norm": det.get("center_norm"),
            }
        )
    return compacted


def _compact_registry(registry: Sequence[dict], *, max_items: int) -> list[dict]:
    compacted: list[dict] = []
    for obj in list(registry)[:max_items]:
        compacted.append(
            {
                "obj_id": obj.get("obj_id"),
                "label": obj.get("label"),
                "center_norm": obj.get("center_norm"),
                "confidence": round(float(obj.get("confidence", 0.0)), 3)
                if obj.get("confidence") is not None
                else None,
                "status": obj.get("status"),
                "seen_count": obj.get("seen_count"),
                "mention_count": obj.get("mention_count"),
            }
        )
    return compacted


def _compact_events(events: Sequence[dict], *, max_items: int) -> list[dict]:
    compacted: list[dict] = []
    for event in list(events)[-max_items:]:
        compacted.append(
            {
                "ts": event.get("ts"),
                "type": event.get("type"),
                "message": event.get("message"),
            }
        )
    return compacted


def build_qwen_interaction_prompt(
    *,
    user_query: str,
    scene_summary: str,
    detections: Sequence[dict],
    preferences: Sequence[str],
    recent_turns: Sequence[dict],
    object_registry: Sequence[dict] | None = None,
    recent_scene_events: Sequence[dict] | None = None,
    reference_resolution: dict | None = None,
    compact_context: bool = False,
    language: str = "zh",
) -> dict[str, object]:
    """Build an answer-first, academic-friendly prompt for live interaction simulation."""
    system_prompt = (
        "You are URIS, a multimodal interaction-simulation assistant for home-environment research. "
        "This is an interaction simulation platform (not robot motion control).\\n\\n"
        "请遵循以下规则：\\n"
        "1. 先给出直接回答，再补充依据与建议。\\n"
        "2. 区分“观察事实”和“推断/建议”。\\n"
        "3. 保持用户交互体验友好、专业、简洁。\\n"
        "4. 不要输出思维链，不要展示内部推理过程。\\n"
        "5. 若信息不足，最多提出一个澄清问题。\\n"
        "6. 输出必须包含用户可见回复和结构化JSON，便于学术评估。\\n"
        "7. 明确说明置信度与局限性，避免过度确定。\\n"
        f"8. 默认使用{'中文' if language.startswith('zh') else 'English'}回答。\\n\\n"
        "Return a natural-language answer plus a JSON block containing the required schema fields."
    )

    context_detections = list(detections)
    context_registry = list(object_registry or [])
    context_events = list(recent_scene_events or [])
    if compact_context:
        context_detections = _compact_detections(context_detections, max_items=8)
        context_registry = _compact_registry(context_registry, max_items=10)
        context_events = _compact_events(context_events, max_items=5)

    context_payload = {
        "scene_summary": scene_summary,
        "detections": context_detections,
        "object_registry": context_registry,
        "recent_scene_events": context_events[-5:],
        "reference_resolution": dict(reference_resolution or {}),
        "preferences": list(preferences),
        "recent_turns": list(recent_turns)[-4:],
        "fast_mode_context_compaction": bool(compact_context),
        "required_json_fields": REQUIRED_JSON_FIELDS,
        "output_contract": {
            "user_response": "natural language for the user",
            "analysis_json": "structured object for evaluation",
        },
    }
    user_prompt = (
        "请基于以下实时场景信息进行交互分析与建议（仅仿真，不控制机器人运动）。\\n"
        f"user_query: {user_query}\\n"
        "context_json:\\n"
        f"{json.dumps(context_payload, ensure_ascii=False, indent=2)}\\n\\n"
        "请输出：\\n"
        "1) 面向用户的自然语言回答（先给结论）\\n"
        "2) 一个JSON代码块，包含 user_response 和 analysis_json 字段\\n"
        "3) analysis_json 中必须包含 required_json_fields 所列字段"
    )
    return {
        "schema_version": PROMPT_VERSION,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "required_json_fields": list(REQUIRED_JSON_FIELDS),
    }
